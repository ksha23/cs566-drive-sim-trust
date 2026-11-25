"""Per-pixel "where can we trust the render" maps for the cross-lane evaluation renders of a test scene:
  gain    FaithFusion EIG (log scale, 100 = not covered), from the SOURCE lane's splat and training views (eig.py)
  unobs   reprojection consistency (HAD-style confidence, our simple version): the fraction of the K=3 nearest source-lane
          training cameras in which the pixel's 3D point (from the rendered depth) is outside the frustum or occluded
          (|z_proj - d_src| > 10% of z_proj, d_src = splat depth rendered at that training camera)
  alpha   splat coverage (1 - alpha is a trivial baseline score)
Saved at half resolution (480x270) to <out>/<scene>/<A>_to_<B>/<ts>.npz. Eval subset = same as run_fixers.eval_items
(variant 'full', every 2nd item + all on-path held-out frames).

usage (faithfusion env, see scripts/14_faithfusion.sh): python -m progressive.faithfusion.maps <scene>
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from common.config import DATA as D
from splats.paralane import load_clip
from splats.gs import load_splat, render, pixel_dirs
from progressive.faithfusion.eig import EIG


def eval_items(scene, stride=2):
    meta = [m for m in json.load(open(f"{D}/pairs/{scene}/meta.json")) if m["variant"] == "full"]
    return [m for k, m in enumerate(meta) if m["onpath_test"] or k % stride == 0]


def half(x):
    return F.interpolate(x[None, None].float(), size=(270, 480), mode="area")[0, 0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--out", default=f"{D}/faithfusion/maps")
    a = ap.parse_args()
    items = eval_items(a.scene)
    clips = {}
    for A in sorted({m["src"] for m in items}):
        t0 = time.time()
        sp = f"{D}/splats/{a.scene}/{A}/splat.pt"
        ck = torch.load(sp, map_location="cuda"); params, sky = load_splat(sp)
        if A not in clips:
            clips[A] = load_clip(a.scene, A)
        cA = clips[A]; K = cA["K"]; dirs = pixel_dirs(K)
        e = EIG(ck["params"], K, 960, 540)
        src_depth, src_c2w = [], []
        for i in ck["train_idx"]:
            with torch.no_grad():
                _, al, dep, _ = render(params, sky, cA["c2w"][i], K, cam_dirs=dirs, with_depth=True)
            e.add_train_view(cA["c2w"][i], al > 0.5)
            src_depth.append(torch.where(al > 0.5, dep, torch.full_like(dep, 1e4))); src_c2w.append(cA["c2w"][i])
        e.finalize()
        src_depth = torch.stack(src_depth); src_c2w = torch.stack(src_c2w); src_pos = src_c2w[:, :3, 3]
        n = 0
        for m in [m for m in items if m["src"] == A]:
            p = f"{a.out}/{a.scene}/{m['src']}_to_{m['dst']}/{m['ts']}.npz"
            if os.path.exists(p):
                continue
            if m["dst"] not in clips:
                clips[m["dst"]] = load_clip(a.scene, m["dst"])
            cB = clips[m["dst"]]; i = cB["ts"].index(m["ts"]); c = cB["c2w"][i]
            with torch.no_grad():
                img, al, dep, _ = render(params, sky, c, K, cam_dirs=dirs, with_depth=True)
            g = e.gain(c, al)
            # reprojection consistency against the K nearest source training cameras
            with torch.no_grad():
                pts = (dirs * dep[..., None]) @ c[:3, :3].T + c[:3, 3]            # (H,W,3) world points
                near = torch.argsort((src_pos - c[:3, 3]).norm(dim=1))[:3]
                bad = torch.zeros_like(dep)
                for j in near:
                    w2c = torch.linalg.inv(src_c2w[j]); pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
                    z = pc[..., 2]; u = pc[..., 0] / z.clamp(min=1e-3) * K[0, 0] + K[0, 2]; v = pc[..., 1] / z.clamp(min=1e-3) * K[1, 1] + K[1, 2]
                    inside = (z > 0.1) & (u >= 0) & (u <= 959) & (v >= 0) & (v <= 539)
                    grid = torch.stack([u / 959 * 2 - 1, v / 539 * 2 - 1], -1)[None]
                    ds = F.grid_sample(src_depth[j][None, None], grid, mode="nearest", align_corners=True)[0, 0]
                    ok = inside & ((ds - z).abs() < 0.1 * z)
                    bad += (~ok).float()
                unobs = bad / len(near)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez_compressed(p, gain=half(g).cpu().numpy().astype(np.float16), unobs=half(unobs).cpu().numpy().astype(np.float16),
                                alpha=half(al).cpu().numpy().astype(np.float16), offset_m=m["offset_m"])
            n += 1
        print(f"{a.scene} {A}: {n} maps, {time.time()-t0:.0f}s", flush=True)
        del e, params, sky, ck, src_depth; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
