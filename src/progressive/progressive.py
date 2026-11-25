"""ReconDreamer / Difix3D+-style progressive update of one lane's splat with a fixer.

Round r (offsets d_r = 3, then 6 m): take every 3rd training pose of lane A, shift it d_r to the left or right
(alternating; the procedure does not know where the other lanes are), render with the current splat, restore the
render with the fixer, and add it as a pseudo-view. As in DriveRestorer, the pseudo-view loss is masked to the
reliable region: no sky (sky-colored or empty pixels in the upper half) and no far region (rendered depth
> 40 m); empty pixels in the lower half (road holes at large offsets) are kept, since the fixer fills them. The splat is then refined on real views + all pseudo-views so far (src/splats/fit_splat.py --init_splat
--pseudo, 2k iters), and the next round starts from the refined splat. The final splat is rendered at the other
lanes' REAL poses (no post-fix) and written like a fixer output, method name prog_<fixer>, so evaluate.py scores it.

usage: scripts/lib/py.sh src/progressive/progressive.py <scene> <clip> <fixer: ours|difix_ft|difix_reimpl|nv_fixer|none> <ckpt|-> [--offsets 3,6] [--iters 2000]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

from splats.paralane import load_clip, lateral_offset
from splats.gs import load_splat, render, pixel_dirs
from fixers.registry import load_fixer
from common.config import DATA

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("clip"); ap.add_argument("fixer"); ap.add_argument("ckpt")
    ap.add_argument("--offsets", default="3,6"); ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--far", type=float, default=40.0)
    ap.add_argument("--tag", default=None, help="output name (default prog_<fixer>)")
    a = ap.parse_args()
    tag = a.tag or f"prog_{a.fixer}"
    sdir = f"{DATA}/splats/{a.scene}/{a.clip}"; work = f"{sdir}/{tag}"; os.makedirs(work, exist_ok=True)
    clip = load_clip(a.scene, a.clip); K = clip["K"]; dirs = pixel_dirs(K)
    tr = sorted(torch.load(f"{sdir}/splat.pt", map_location="cpu")["train_idx"])[::3]
    fix = load_fixer(a.fixer, None if a.ckpt == "-" else a.ckpt)
    cur = f"{sdir}/splat.pt"; P = dict(c2w=[], imgs=[], keep=[]); log = []
    for r, d in enumerate(float(x) for x in a.offsets.split(",")):
        t0 = time.time()
        params, sky = load_splat(cur)
        with torch.no_grad():
            for k, i in enumerate(tr):
                c = clip["c2w"][i].clone(); c[:3, 3] += (d if k % 2 == 0 else -d) * c[:3, 0]
                img, alpha, depth, _ = render(params, sky, c, K, cam_dirs=dirs, with_depth=True)
                out = fix(img, clip["imgs"][i].float() / 255)  # ref = the lane's real frame (used by ref-based fixers)
                x = out.float()
                bl, rd = x[..., 2], x[..., 0]
                rows = torch.arange(x.shape[0], device="cuda")[:, None].expand_as(alpha)
                upper = rows < x.shape[0] * 0.5
                # sky: sky-colored or empty (alpha < 0.5) in the upper half; far: rendered depth > 40 m.
                # Empty regions in the LOWER half (holes in the road/sidewalk at large offsets) are kept: that is
                # exactly where the fixer's restoration should be distilled back into the splat.
                skym = upper & (((bl > 0.45) & (bl > rd + 0.05)) | (alpha < 0.5))
                far = (depth > a.far) & (alpha >= 0.5)
                keep = ~skym & ~far
                P["c2w"].append(c); P["imgs"].append((out.clamp(0, 1) * 255).round().to(torch.uint8)); P["keep"].append(keep)
                if k in (0, 1):
                    Image.fromarray(np.concatenate([(img.cpu().numpy() * 255).astype(np.uint8), P["imgs"][-1].cpu().numpy(),
                                                    (keep.cpu().numpy()[..., None].repeat(3, -1) * 255).astype(np.uint8)], 1)
                                    ).resize((1440, 270)).save(f"{work}/round{r}_example{k}.jpg")
        del params, sky; torch.cuda.empty_cache()
        pp = f"{work}/pseudo.pt"
        torch.save({k: torch.stack(v) for k, v in P.items()}, pp)
        name = f"{tag}_r{r}"
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "splats", "fit_splat.py"), a.scene, a.clip, sdir, "--iters", str(a.iters), "--name", name,
               "--init_splat", cur, "--pseudo", pp, "--pseudo_prob", "0.4"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        on = [l for l in res.stdout.splitlines() if l.startswith("ON-PATH")]
        if res.returncode != 0 or not on:
            print(res.stdout[-2000:], res.stderr[-3000:], flush=True); raise SystemExit("refine failed")
        cur = f"{sdir}/{name}.pt"
        m = json.loads(on[0].split(":", 1)[1]); m.update(round=r, offset=d, n_pseudo=len(P["imgs"]), s=time.time() - t0)
        log.append(m); print(f"round {r} (+-{d} m): {len(P['imgs'])} pseudo views, on-path held-out PSNR {m['psnr']:.2f} "
                             f"LPIPS {m['lpips']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    os.remove(f"{work}/pseudo.pt")
    json.dump(log, open(f"{work}/log.json", "w"), indent=1)
    # final refined splat -> renders at the other lanes' real poses + own held-out poses, written as a "method"
    meta = [m for m in json.load(open(f"{DATA}/pairs/{a.scene}/meta.json")) if m["variant"] == "full" and m["src"] == a.clip]
    params, sky = load_splat(cur)
    others = {}
    with torch.no_grad():
        for m in meta:
            if m["dst"] not in others:
                others[m["dst"]] = load_clip(a.scene, m["dst"]) if m["dst"] != a.clip else clip
            cb = others[m["dst"]]; i = cb["ts"].index(m["ts"])
            img = render(params, sky, cb["c2w"][i], cb["K"], cam_dirs=dirs)[0]
            p = f"{DATA}/fixed/{tag}/{m['path']}"; os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)
    # keep only the final refined splat
    for r in range(len(log) - 1):
        f = f"{sdir}/{tag}_r{r}.pt"
        if os.path.exists(f):
            os.remove(f)
    print("done", a.scene, a.clip, a.fixer, flush=True)


if __name__ == "__main__":
    main()
