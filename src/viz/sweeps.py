"""Continuous deviation sweeps along one recorded Para-Lane lane, for the demo videos (not used for any metric).

kinds (frame i of the lane's N recorded poses; conventions of src/eval/heading_geom.py):
  lane    lateral offset toward the other two lanes: 0 m -> the middle lane's mean offset (~3.3 m) -> the far lane's
          (~6.2 m), with holds. Where the offset is within 0.4 m of a real lane, the NEAREST REAL FRAME of that lane is
          stored as a reference ("approximately the same place"); elsewhere there is no real frame.
  yaw     heading 0 -> +15 -> 0 -> -15 -> 0 deg about the camera centre. Exact ground truth: the recorded real frame
          warped by K R1^T R0 K^-1 (Lanczos); pixels the real camera never saw are black.
  height  camera height 0 -> -1 -> +1 -> 0 m. No real frame exists; the recorded frame is kept only as a reference.
Also renders alternative splats of the same lane at the same poses (--alt name=path.pt, e.g. road prior, progressive).
Difix-ref style reference = the lane's REAL recorded frame at the same index (what a simulator has).

Output $CS566D_DATA/viz/<name>/: raw/, alt_<name>/, ref/, gt/ (when defined), meta.json [{i, path, ref, gt, off, yaw, dz,
gt_kind}] -- paths relative to $CS566D_DATA/viz so src/fixers/apply.py --meta can run fixers on raw/.
usage: scripts/lib/py.sh src/viz/sweeps.py <scene> <clip> <lane|yaw|height> <name> [--alt road=... --alt prog=...]
"""
import argparse, json, os
import cv2
import numpy as np
import torch
from PIL import Image

from common.config import DATA
from splats.paralane import load_cameras, clip_dir, load_image
from splats.gs import load_splat, render, pixel_dirs
from eval.heading_geom import deviate, homography


def schedule(kind, N, lanes):
    t = np.arange(N) / max(N - 1, 1)
    if kind == "lane":
        a, b = lanes
        return np.interp(t, [0, 0.12, 0.32, 0.5, 0.7, 1.0], [0, 0, a, a, b, b])
    if kind == "yaw":
        return np.interp(t, [0, 0.08, 0.3, 0.5, 0.72, 0.92, 1.0], [0, 0, 15, 0, -15, 0, 0])
    return np.interp(t, [0, 0.08, 0.35, 0.75, 1.0], [0, 0, -1, 1, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("clip"); ap.add_argument("kind", choices=["lane", "yaw", "height"])
    ap.add_argument("name"); ap.add_argument("--alt", action="append", default=[])
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    root = f"{DATA}/viz"; out = f"{root}/{a.name}"
    for d in ["raw", "ref", "gt"] + [f"alt_{x.split('=')[0]}" for x in a.alt]:
        os.makedirs(f"{out}/{d}", exist_ok=True)
    d = clip_dir(a.scene, a.clip); K, ts, c2w = load_cameras(d)
    idx = list(range(0, len(ts), a.every)); N = len(idx)
    # the other lanes: mean signed lateral offset (+ = camera right) of their camera centres
    others = {}
    for c in ("clip_0", "clip_1", "clip_2"):
        if c == a.clip:
            continue
        _, tsB, cB = load_cameras(clip_dir(a.scene, c))
        CA = c2w[:, :3, 3]; right = c2w[:, :3, 0]
        offs = []
        for C in cB[:, :3, 3]:
            j = np.linalg.norm(CA - C, axis=1).argmin(); offs.append(float((C - CA[j]) @ right[j]))
        others[c] = (float(np.median(offs)), tsB, cB)
    lanes = sorted(others.items(), key=lambda kv: abs(kv[1][0]))
    sgn = np.sign(lanes[-1][1][0])
    sched = schedule(a.kind, N, (lanes[0][1][0] * sgn, lanes[-1][1][0] * sgn))
    Kt = torch.tensor(K, device="cuda"); dirs = pixel_dirs(Kt)
    splats = {"raw": load_splat(f"{DATA}/splats/{a.scene}/{a.clip}/splat.pt")}
    for x in a.alt:
        k, p = x.split("="); splats[f"alt_{k}"] = load_splat(p)
    meta = []
    with torch.no_grad():
        for k, i in enumerate(idx):
            c0 = c2w[i].astype(np.float64); v = float(sched[k]); off = yaw = dz = 0.0
            if a.kind == "lane":
                off = v * sgn; c1 = c0.copy(); r = c0[:3, 0].copy(); r[2] = 0; r /= np.linalg.norm(r)
                c1[:3, 3] += off * r
            elif a.kind == "yaw":
                yaw = v; c1 = deviate(c0, yaw=yaw)
            else:
                dz = v; c1 = deviate(c0, dz=dz)
            stem = f"{k:04d}.png"
            for key, (params, sky) in splats.items():
                img = render(params, sky, torch.tensor(c1, dtype=torch.float32, device="cuda"), Kt, cam_dirs=dirs)[0]
                Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(f"{out}/{key}/{stem}")
            real = load_image(d, ts[i]); Image.fromarray(real).save(f"{out}/ref/{stem}")
            gt, gk = None, "none"
            if a.kind == "lane":
                for c, (mo, tsB, cB) in others.items():
                    if abs(abs(off) - abs(mo)) < 0.4:
                        j = int(np.linalg.norm(cB[:, :3, 3] - c1[:3, 3], axis=1).argmin())
                        Image.fromarray(load_image(clip_dir(a.scene, c), tsB[j])).save(f"{out}/gt/{stem}")
                        gt, gk = f"{a.name}/gt/{stem}", f"nearest real frame of {c} ({abs(mo):.1f} m lane)"
            elif a.kind == "yaw":
                Hm = homography(K, c0, c1)
                w = cv2.warpPerspective(real, Hm, (real.shape[1], real.shape[0]), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                Image.fromarray(w).save(f"{out}/gt/{stem}"); gt, gk = f"{a.name}/gt/{stem}", "real frame rotated (exact)"
            meta.append(dict(i=k, ts=ts[i], path=f"{a.name}/raw/{stem}", ref=f"{a.name}/ref/{stem}", gt=gt, gt_kind=gk,
                             off=off, yaw=yaw, dz=dz, scene=a.scene, clip=a.clip))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=0)
    print(a.name, N, "frames; lanes", {c: round(v[0], 2) for c, v in others.items()}, flush=True)


if __name__ == "__main__":
    main()
