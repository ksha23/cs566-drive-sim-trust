"""Render a fitted street splat along its recorded lane, displaced sideways / turned / raised (command-line tool).

Every recorded pose of <clip> (or every --every-th) is shifted by --lateral metres along the camera's horizontal
right axis (+ = LEFT, the convention of the results), rotated by --yaw degrees about the world up axis through the
camera centre (+ = left) and raised by --dz metres; the splat is rendered there at 960x540. No real image exists at
these poses unless another Para-Lane lane drove there (see src/pairs/render_pairs.py for the scored renders).

usage: scripts/lib/py.sh src/tools/render_offpath.py <scene> <clip> <splat.pt> <out_dir> [--lateral 3.0] [--yaw 0]
                                                     [--dz 0] [--every 4]
example: scripts/lib/py.sh src/tools/render_offpath.py scene_16 clip_0 $CS566D_DATA/splats/scene_16/clip_0/splat.pt \
             /tmp/offpath_3m --lateral 3.3
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image

from splats.paralane import load_cameras, clip_dir
from splats.gs import load_splat, render, pixel_dirs
from eval.heading_geom import deviate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("clip"); ap.add_argument("splat"); ap.add_argument("out")
    ap.add_argument("--lateral", type=float, default=0.0); ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0); ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    K, ts, c2w = load_cameras(clip_dir(a.scene, a.clip))
    Kt = torch.as_tensor(K, dtype=torch.float32, device="cuda")
    params, sky = load_splat(a.splat); dirs = pixel_dirs(Kt); meta = []
    with torch.no_grad():
        for i in range(0, len(ts), a.every):
            P = np.array(c2w[i], np.float64)
            r = P[:3, 0].copy(); r[2] = 0; r /= np.linalg.norm(r) + 1e-9
            P[:3, 3] -= a.lateral * r  # + = left
            P = deviate(P, a.yaw, a.dz)
            img = render(params, sky, torch.as_tensor(P, dtype=torch.float32, device="cuda"), Kt, cam_dirs=dirs)[0]
            f = f"{ts[i]}.png"
            Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(f"{a.out}/{f}")
            meta.append(dict(file=f, ts=ts[i], lateral_m=a.lateral, yaw_deg=a.yaw, dz_m=a.dz, c2w=P.tolist()))
    json.dump(meta, open(f"{a.out}/meta.json", "w"))
    print(f"wrote {len(meta)} renders to {a.out}")


if __name__ == "__main__":
    main()
