"""Apply fixers to the evaluation renders of a scene (full splats; cross-lane + on-path held-out frames).
Methods (see fixers/registry.py):
  difix          released Difix, zero-shot, 1 step
  difix_ref      released Difix-ref; reference = the REAL frame of the SOURCE lane nearest to the target pose
                 (what a simulator built from lane A has). For on-path test frames the held-out frame itself is excluded,
                 so the reference is the adjacent recorded frame (~0.1 s / ~0.7 m away).
  difix_ref_far  same, but the reference is >= 2 s (20 frames) away from that nearest frame along the source pass
  difix_ft       released Difix with our fine-tuned U-Net (--ckpt best_unet.pt)
  ours           our U-Net (--ckpt best.pt); --tag renames the output (e.g. ours_hf_s1)
Writes <out>/<method or tag>/<scene>/render/.../<ts>.png and merges timing into <out>/timing.json.

  difix_reimpl   our Difix reimplementation [1] with reference view (same reference rule as difix_ref; --ckpt best.pt)
  nv_fixer       NVIDIA Fixer [4] through its env (fixers/registry.py)
--meta J uses the items of another meta list (e.g. the heading/height renders of src/splats/render_deviation.py,
whose y+0_z+0.0 renders are symlinks to already-fixed outputs and are skipped); --skip_existing resumes a run.

usage: scripts/lib/py.sh src/fixers/apply.py <pairs_root> <scene> <out> --methods difix,difix_ref --ckpt X [--tag name] [--stride 2]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

from fixers.registry import load_fixer


def eval_items(pairs, scene, stride=2, variant="full"):
    meta = [m for m in json.load(open(f"{pairs}/{scene}/meta.json")) if m["variant"] == variant]
    return [m for k, m in enumerate(meta) if m["onpath_test"] or k % stride == 0]


_cache = {}


def ref_path(pairs, m, far_frames=0):
    from splats.paralane import load_cameras, clip_dir
    key = (m["scene"], m["src"], m["dst"])
    if key not in _cache:
        _, tsA, cA = load_cameras(clip_dir(m["scene"], m["src"]))
        _, tsB, cB = load_cameras(clip_dir(m["scene"], m["dst"]))
        _cache[key] = (tsA, cA[:, :3, 3], dict(zip(tsB, cB[:, :3, 3])))
    tsA, pA, posB = _cache[key]
    d = np.linalg.norm(pA - posB[m["ts"]], axis=1)
    if m["src"] == m["dst"]:
        d[tsA.index(m["ts"])] = 1e9  # never use the held-out image itself
    j = int(d.argmin())
    if far_frames:
        j = j - far_frames if j - far_frames >= 0 else j + far_frames
        j = min(j, len(tsA) - 1)
    return f"{pairs}/{m['scene']}/real/{m['src']}/{tsA[j]}.png", abs(int(tsA[j]) - int(tsA[int(d.argmin())])) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("scene"); ap.add_argument("out")
    ap.add_argument("--methods", default="difix,difix_ref"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--tag", default=None)
    ap.add_argument("--meta", default=None); ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    if a.meta:
        seen = set(); items = []
        for m in json.load(open(a.meta)):
            if m["path"] not in seen and m.get("cond") != "y+0_z+0.0":
                seen.add(m["path"]); items.append(m)
    else:
        items = eval_items(a.pairs, a.scene, a.stride)
    print(len(items), "eval renders", flush=True)
    tpath = f"{a.out}/timing.json"
    for meth in a.methods.split(","):
        name = a.tag or meth
        base = "difix_ref" if meth.startswith("difix_ref") else meth
        todo = [m for m in items if not (a.skip_existing and os.path.exists(f"{a.out}/{name}/{m['path']}"))]
        print(name, len(todo), "to do", flush=True)
        if not todo:
            continue
        fix = load_fixer(base, a.ckpt)
        times, ref_dt = [], []
        for m in todo:
            img = torch.from_numpy(np.array(Image.open(f"{a.pairs}/{m['path']}").convert("RGB"))).float().cuda() / 255
            ref = None
            if base in ("difix_ref", "difix_reimpl"):
                if "ref" in m:  # the item names its reference (e.g. EUVS: nearest training view, src/euvs/fit.py)
                    rp, dt = f"{a.pairs}/{m['ref']}", 0.0
                else:
                    rp, dt = ref_path(a.pairs, m, far_frames=20 if meth == "difix_ref_far" else 0)
                ref = Image.open(rp).convert("RGB"); ref_dt.append(dt)
            torch.cuda.synchronize(); t = time.time()
            o = fix(img, ref)
            torch.cuda.synchronize(); times.append(time.time() - t)
            p = f"{a.out}/{name}/{m['path']}"; os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray((o.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)
        timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
        timing[name] = dict(ms_per_img_median=float(np.median(times[3:]) * 1000), n=len(times),
                            peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
        if ref_dt:
            timing[name]["ref_time_gap_s_median"] = float(np.median(ref_dt))
        print(name, timing[name], flush=True)
        json.dump(timing, open(tpath, "w"), indent=1)
        del fix; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
