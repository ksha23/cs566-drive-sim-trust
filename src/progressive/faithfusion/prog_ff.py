"""FaithFusion arm of the progressive update: identical to our progressive loop except for how pseudo-views are
weighted in the splat refinement.

  restorer : released Difix-ref (nvidia/difix_ref) with the lane's own real image at the source pose as reference,
             as in FaithFusion's repo (tools/eval.py:do_fix, fix_model=difix); or our Difix-FT (--restorer difix_ft)
  arm flat : FaithFusion's Difix path: every pseudo-view pixel weight 0.7 (their `extra_diff_weight`), SSIM x 0.7
  arm eig  : FaithFusion's EIG weighting (models/trainers/base.py, EIGent branch): per-pixel L1 weight from the
             log-EIG map of the CURRENT splat at that pose: <0.05 -> 0.1, [0.05,0.2) -> 0.3, [0.2,1) -> 0.7, >=1 -> 1.0
             (uncovered pixels have EIG = 100 -> 1.0); SSIM x 0.7. In their repo this weighting is only switched on
             for their EIGent restorer (weights unreleased); here it is applied to Difix outputs.
Both arms use our DriveRestorer-style validity mask (no sky / far region) and the same schedule as our loop:
rounds at +-2, +-4, +-6 m (alternating sides, every 3rd training pose), 2k refinement iterations per round,
pseudo-view sampling probability 0.4 (FaithFusion samples shifted views with p = 0.4 too).

usage (faithfusion env, see scripts/14_faithfusion.sh): python -m progressive.faithfusion.prog_ff <scene> <clip> <flat|eig> [--restorer difix_ref|difix_ft] [--offsets 2,4,6]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

from common.config import DATA as D, REPO
from splats.paralane import load_clip
from splats.gs import load_splat, render, pixel_dirs
from fixers.registry import load_fixer
from progressive.faithfusion.eig import EIG, eig_weight

SRC = os.path.join(REPO, "src")
FIT = os.path.join(SRC, "splats", "fit_splat.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("clip"); ap.add_argument("arm", choices=["flat", "eig"])
    ap.add_argument("--restorer", default="difix_ref"); ap.add_argument("--offsets", default="2,4,6")
    ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--far", type=float, default=40.0)
    a = ap.parse_args()
    name = f"ff_{a.arm}" + ("" if a.restorer == "difix_ref" else f"_{a.restorer}")
    sdir = f"{D}/splats/{a.scene}/{a.clip}"; work = f"{sdir}/prog_{name}"; os.makedirs(work, exist_ok=True)
    clip = load_clip(a.scene, a.clip); K = clip["K"]; dirs = pixel_dirs(K)
    ck0 = torch.load(f"{sdir}/splat.pt", map_location="cpu"); train_all = sorted(ck0["train_idx"]); tr = train_all[::3]; del ck0
    fix = load_fixer(a.restorer, f"{D}/difix_ft/best_unet.pt" if a.restorer == "difix_ft" else None)
    cur = f"{sdir}/splat.pt"; P = dict(c2w=[], imgs=[], keep=[], weight=[]); log = []
    for r, d in enumerate(float(x) for x in a.offsets.split(",")):
        t0 = time.time()
        params, sky = load_splat(cur)
        if a.arm == "eig":
            e = EIG(torch.load(cur, map_location="cuda")["params"], K, 960, 540)
            for i in train_all:
                with torch.no_grad():
                    _, al, _, _ = render(params, sky, clip["c2w"][i], K, cam_dirs=dirs)
                e.add_train_view(clip["c2w"][i], al > 0.5)
            e.finalize()
        gstats = []
        for k, i in enumerate(tr):
            c = clip["c2w"][i].clone(); c[:3, 3] += (d if k % 2 == 0 else -d) * c[:3, 0]
            with torch.no_grad():
                img, alpha, depth, _ = render(params, sky, c, K, cam_dirs=dirs, with_depth=True)
                ref = clip["imgs"][i].float() / 255 if a.restorer == "difix_ref" else None
                out = fix(img, Image.fromarray(clip["imgs"][i].cpu().numpy()) if ref is not None else None).float()
                bl, rd = out[..., 2], out[..., 0]
                rows = torch.arange(out.shape[0], device="cuda")[:, None].expand_as(alpha)
                skym = (rows < out.shape[0] * 0.5) & (((bl > 0.45) & (bl > rd + 0.05)) | (alpha < 0.5))
                keep = ~skym & ~((depth > a.far) & (alpha >= 0.5))
            if a.arm == "eig":
                g = e.gain(c, alpha); w = eig_weight(g); gstats.append(g[keep & (g < 99)].float())
            else:
                w = torch.full_like(alpha, 0.7)
            P["c2w"].append(c); P["imgs"].append((out.clamp(0, 1) * 255).round().to(torch.uint8)); P["keep"].append(keep)
            P["weight"].append(w.half())
            if k == 0:
                Image.fromarray(np.concatenate([(img.cpu().numpy() * 255).astype(np.uint8), P["imgs"][-1].cpu().numpy(),
                                                (np.repeat((w * keep).cpu().numpy()[..., None], 3, -1) * 255).astype(np.uint8)], 1)
                                ).resize((1440, 270)).save(f"{work}/round{r}_example.jpg")
        if a.arm == "eig":
            del e
        del params, sky; torch.cuda.empty_cache()
        pp = f"{work}/pseudo.pt"
        torch.save({k: torch.stack(v) for k, v in P.items()} | {"ssim_w": 0.7}, pp)
        nm = f"prog_{name}_r{r}"
        cmd = [sys.executable, FIT, a.scene, a.clip, sdir, "--iters", str(a.iters), "--name", nm,
               "--init_splat", cur, "--pseudo", pp, "--pseudo_prob", "0.4"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ | {"PYTHONPATH": SRC})
        on = [l for l in res.stdout.splitlines() if l.startswith("ON-PATH")]
        if res.returncode != 0 or not on:
            print(res.stdout[-2000:], res.stderr[-3000:], flush=True); raise SystemExit("refine failed")
        if r > 0 and os.path.exists(cur) and "prog_" in cur:
            os.remove(cur)
        cur = f"{sdir}/{nm}.pt"
        m = json.loads(on[0].split(":", 1)[1]); m.update(round=r, offset=d, n_pseudo=len(P["imgs"]), s=time.time() - t0)
        if gstats:
            gs = torch.cat(gstats); q = torch.quantile(gs[torch.randperm(len(gs))[:200000]], torch.tensor([.1, .5, .9], device=gs.device))
            m["eig_q10_50_90"] = [float(x) for x in q]
            ww = torch.stack(P["weight"][-len(tr):]).float(); kk = torch.stack(P["keep"][-len(tr):])
            m["mean_weight_in_keep"] = float(ww[kk].mean())
        log.append(m)
        print(f"round {r} (+-{d} m): on-path PSNR {m['psnr']:.2f} LPIPS {m['lpips']:.3f} "
              f"{'EIG q10/50/90 %s, mean w %.2f' % (m['eig_q10_50_90'], m['mean_weight_in_keep']) if gstats else ''} ({time.time()-t0:.0f}s)", flush=True)
    os.remove(f"{work}/pseudo.pt")
    meta =[m for m in json.load(open(f"{D}/pairs/{a.scene}/meta.json")) if m["variant"] == "full" and m["src"] == a.clip]
    params, sky = load_splat(cur); others = {a.clip: clip}
    with torch.no_grad():
        for m in meta:
            if m["dst"] not in others:
                others[m["dst"]] = load_clip(a.scene, m["dst"])
            cb = others[m["dst"]]; i = cb["ts"].index(m["ts"])
            img = render(params, sky, cb["c2w"][i], cb["K"], cam_dirs=dirs)[0]
            p = f"{D}/fixed/prog_{name}/{m['path']}"; os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)
    json.dump(log, open(f"{work}/log.json", "w"), indent=1)  # last: marks the clip finished (resume, requeue)
    print("done", a.scene, a.clip, name, flush=True)


if __name__ == "__main__":
    main()
