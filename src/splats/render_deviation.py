"""Render each test-scene lane splat at heading (yaw) and camera-height deviations (proposal step 1; conventions in
src/eval/heading_geom.py).

Frames and conditions (per scene, per source lane A):
  on-path   A's HELD-OUT frames (every 8th, never trained on; the 0 m rows of the lateral table), rendered at
            yaw in --yaws (dz = 0) and dz in --dzs (yaw = 0)
  cross     lateral + yaw: lane A's splat at lane B's REAL poses (the cross-lane eval pairs, every --cross_stride-th
            of the stride-2 eval subset) with yaw in --cross_yaws
The recorded-pose condition y+0_z+0.0 is not re-rendered: its directory is a symlink to the existing 'full' pair
renders, so the 0-deviation outputs of every fixer are exactly the ones behind the lateral results.

Output root <out> (default $CS566D_DATA/heading):
  <scene>/render/<A>_to_<B>/<cond>/<ts>.png, <scene>/real -> pairs real, <scene>/fg -> pairs fg
  <scene>/meta.json: [{scene, src, dst, ts, cond, yaw, dz, axis, offset_m, onpath, path, real, fg, zero, c2w0, c2w1, K, cam_h}]
    axis = yaw | height | lateral_yaw;  zero = path of the same frame at y+0_z+0.0;  c2w0 = recorded pose.

usage: scripts/lib/py.sh src/splats/render_deviation.py <scene> [--yaws -15,-10,-5,5,10,15] [--dzs -1,-0.5,0.5,1]
                                                          [--cross_yaws -10,10] [--cross_stride 3]
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image

from common.config import DATA
from splats.paralane import load_cameras, clip_dir, load_lidar
from splats.gs import load_splat, render, pixel_dirs
from fixers.apply import eval_items
from eval.heading_geom import cond_name, deviate, camera_heights

ZERO = cond_name(0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--out", default=f"{DATA}/heading")
    ap.add_argument("--yaws", default="-15,-10,-5,5,10,15"); ap.add_argument("--dzs", default="-1,-0.5,0.5,1")
    ap.add_argument("--cross_yaws", default="-10,10"); ap.add_argument("--cross_stride", type=int, default=3)
    a = ap.parse_args()
    S, P = a.scene, f"{DATA}/pairs"
    root = f"{a.out}/{S}"; os.makedirs(f"{root}/render", exist_ok=True)
    for d in ("real", "fg"):
        if not os.path.exists(f"{root}/{d}"):
            os.symlink(f"{P}/{S}/{d}", f"{root}/{d}")
    yaws = [float(x) for x in a.yaws.split(",") if x]; dzs = [float(x) for x in a.dzs.split(",") if x]
    cyaws = [float(x) for x in a.cross_yaws.split(",") if x]
    items = eval_items(P, S, 2)
    onpath = [m for m in items if m["onpath_test"]]
    cross = [m for m in items if not m["onpath_test"]][::a.cross_stride]
    cams, hts = {}, {}
    for c in sorted({m["dst"] for m in items}):
        K, ts, c2w = load_cameras(clip_dir(S, c)); cams[c] = (K, dict(zip(ts, c2w)))
        hts[c] = dict(zip(ts, camera_heights(c2w, load_lidar(S, c))))
    jobs = []  # (item, yaw, dz, axis)
    for m in onpath:
        jobs += [(m, y, 0.0, "yaw") for y in yaws] + [(m, 0.0, z, "height") for z in dzs]
    for m in cross:
        jobs += [(m, y, 0.0, "lateral_yaw") for y in cyaws]
    meta = []
    for m in onpath + cross:  # the recorded-pose condition, once per frame (symlinked render)
        jobs.append((m, 0.0, 0.0, "zero"))
    by_src = {}
    for j in jobs:
        by_src.setdefault(j[0]["src"], []).append(j)
    n_new = 0
    with torch.no_grad():
        for A, js in sorted(by_src.items()):
            params, sky = load_splat(f"{DATA}/splats/{S}/{A}/splat.pt")
            for m, yaw, dz, axis in js:
                K, poses = cams[m["dst"]]; c0 = poses[m["ts"]]; cond = cond_name(yaw, dz)
                pdir = f"{S}/render/{m['src']}_to_{m['dst']}"
                rel = f"{pdir}/{cond}/{m['ts']}.png"
                if axis == "zero":
                    link = f"{a.out}/{pdir}/{ZERO}"
                    os.makedirs(os.path.dirname(link), exist_ok=True)
                    if not os.path.lexists(link):
                        os.symlink(f"{P}/{pdir}/full", link)
                    axes = ["yaw", "height"] if m["onpath_test"] else ["lateral_yaw"]
                else:
                    axes = [axis]
                    c1 = deviate(c0, yaw, dz)
                    out = f"{a.out}/{rel}"
                    if not os.path.exists(out):
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        Kt = torch.tensor(K, device="cuda")
                        img = render(params, sky, torch.tensor(c1, dtype=torch.float32, device="cuda"), Kt,
                                     cam_dirs=pixel_dirs(Kt))[0]
                        Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(out); n_new += 1
                c1 = deviate(c0, yaw, dz)
                for ax in axes:
                    meta.append(dict(scene=S, src=m["src"], dst=m["dst"], ts=m["ts"], cond=cond, yaw=yaw, dz=dz, axis=ax,
                                     offset_m=m["offset_m"], onpath=m["onpath_test"], path=rel, real=m["real"], fg=m["fg"],
                                     zero=f"{pdir}/{ZERO}/{m['ts']}.png", c2w0=np.asarray(c0).tolist(),
                                     c2w1=c1.tolist(), K=np.asarray(K).tolist(), cam_h=float(hts[m["dst"]][m["ts"]])))
            del params, sky; torch.cuda.empty_cache()
    json.dump(meta, open(f"{root}/meta.json", "w"))
    from collections import Counter
    print(S, "rendered", n_new, "new images;", len(meta), "meta rows", dict(Counter(r["axis"] for r in meta)),
          "cam height %.2f-%.2f m" % (min(r["cam_h"] for r in meta), max(r["cam_h"] for r in meta)), flush=True)


if __name__ == "__main__":
    main()
