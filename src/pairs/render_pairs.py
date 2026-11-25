"""Build training/eval pairs (render, REAL image) for one scene, following Difix3D+'s data curation adapted to
Para-Lane's parallel passes. Variants (meta "variant"):
  full      lane A's splat at lane B's real poses (cross-lane, both directions) + A's held-out on-path frames
            (onpath_test=True, the 0 m reference)
  identity  A's splat at A's own TRAINING poses (near-perfect renders -> teaches the fixer to leave good renders alone)
  underfit  A's deliberately under-fit splat (2k iters) at B's poses and at all of A's poses
  cycle     cycle reconstruction: splat fitted only on A-renders at +/-3 m shifted poses, rendered back at A's poses
Outputs (under <pairs_root>/<scene>/):
  real/<clip>/<ts>.png, fg/<clip>/<ts>.png       working-res real image and movable-object mask
  render/<A>_to_<B>/<variant>/<ts>.png
  meta.json  list of {src, dst, ts, variant, offset_m (signed, +left), onpath_test, path, real, fg}

usage: scripts/lib/py.sh src/pairs/render_pairs.py <scene> <splat_root> <pairs_root> [--variants full,identity,underfit,cycle]
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

from splats.paralane import load_clip, lateral_offset
from splats.gs import load_splat, render, pixel_dirs

CLIPS = ["clip_0", "clip_1", "clip_2"]


def along(c2w_q, c2w_p):
    """Is each query camera within the longitudinal extent of path p (projected on p's overall direction)?"""
    P = c2w_p[:, :2, 3].cpu().numpy(); Q = c2w_q[:, :2, 3].cpu().numpy()
    d = P[-1] - P[0]; d /= np.linalg.norm(d)
    s = Q @ d; sp = P @ d
    return (s >= sp.min() + 2) & (s <= sp.max() - 2)


def save_png(x, p):
    Image.fromarray((x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("sroot"); ap.add_argument("proot")
    ap.add_argument("--variants", default="full,identity,underfit,cycle")
    a = ap.parse_args()
    scene, sroot, proot = a.scene, a.sroot, a.proot
    variants = a.variants.split(",")
    out = f"{proot}/{scene}"; os.makedirs(out, exist_ok=True)
    clips = {c: load_clip(scene, c) for c in CLIPS if os.path.exists(f"{sroot}/{scene}/{c}/splat.pt")}
    for c, cl in clips.items():
        if os.path.isdir(f"{out}/fg/{c}") and len(os.listdir(f"{out}/fg/{c}")) == len(cl["ts"]):
            continue
        os.makedirs(f"{out}/real/{c}", exist_ok=True); os.makedirs(f"{out}/fg/{c}", exist_ok=True)
        for i, t in enumerate(cl["ts"]):
            Image.fromarray(cl["imgs"][i].cpu().numpy()).save(f"{out}/real/{c}/{t}.png")
            Image.fromarray((cl["fg"][i].cpu().numpy() * 255).astype(np.uint8)).save(f"{out}/fg/{c}/{t}.png")
    mp = f"{out}/meta.json"
    meta = [m for m in json.load(open(mp)) if m["variant"] not in variants] if os.path.exists(mp) else []
    ckname = dict(full="splat.pt", identity="splat.pt", underfit="splat_underfit.pt", cycle="splat_cycle.pt")
    with torch.no_grad():
        for A in clips:
            for variant in variants:
                ck = f"{sroot}/{scene}/{A}/{ckname[variant]}"
                if not os.path.exists(ck):
                    print("missing", ck, flush=True); continue
                params, sky = load_splat(ck)
                ckd = torch.load(ck, map_location="cpu")
                test_idx, train_idx = sorted(ckd["test_idx"]), sorted(ckd["train_idx"]); del ckd
                for B, cb in clips.items():
                    if variant == "full":
                        idx = test_idx if B == A else np.where(along(cb["c2w"], clips[A]["c2w"]))[0].tolist()
                    elif variant == "identity":
                        idx = train_idx if B == A else []
                    elif variant == "underfit":
                        idx = list(range(len(cb["ts"]))) if B == A else np.where(along(cb["c2w"], clips[A]["c2w"]))[0].tolist()
                    else:  # cycle
                        idx = list(range(len(cb["ts"]))) if B == A else []
                    if not idx:
                        continue
                    offs = lateral_offset(cb["c2w"][idx], clips[A]["c2w"])
                    d = f"{out}/render/{A}_to_{B}/{variant}"; os.makedirs(d, exist_ok=True)
                    dirs = pixel_dirs(cb["K"])
                    for j, i in enumerate(idx):
                        img = render(params, sky, cb["c2w"][i], cb["K"], cam_dirs=dirs)[0]
                        t = cb["ts"][i]; save_png(img, f"{d}/{t}.png")
                        meta.append(dict(scene=scene, src=A, dst=B, ts=t, variant=variant, offset_m=float(offs[j]),
                                         onpath_test=(variant == "full" and A == B), path=f"{scene}/render/{A}_to_{B}/{variant}/{t}.png",
                                         real=f"{scene}/real/{B}/{t}.png", fg=f"{scene}/fg/{B}/{t}.png"))
                    print(A, "->", B, variant, len(idx), "frames, |offset| mean %.2f m" % np.abs(offs).mean(), flush=True)
                del params, sky; torch.cuda.empty_cache()
    json.dump(meta, open(mp, "w"), indent=0)
    from collections import Counter
    print("wrote", len(meta), "pairs:", dict(Counter(m["variant"] for m in meta)), flush=True)


if __name__ == "__main__":
    main()
