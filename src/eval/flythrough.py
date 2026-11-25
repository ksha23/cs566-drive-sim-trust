"""On-path -> off-path fly-through: drive along lane A's recorded poses while sliding sideways to ~6 m.
Left: raw splat render. Right: either our U-Net applied to it (right = fixer ckpt) or another splat of the same lane,
e.g. the progressively refined one (right = splat:<path>).
usage: scripts/lib/py.sh src/eval/flythrough.py <splat.pt> <scene> <clip> <right> <out.mp4> [max_shift_m] [right_label]
"""
import os, sys
import numpy as np
import torch
import imageio
import cv2

from splats.paralane import load_cameras, clip_dir
from splats.gs import load_splat, render, pixel_dirs


def main():
    sp, scene, clip, right, out = sys.argv[1:6]
    shift = float(sys.argv[6]) if len(sys.argv) > 6 else 6.3
    rlabel = sys.argv[7] if len(sys.argv) > 7 else "right"
    params, sky = load_splat(sp)
    K, ts, c2w = load_cameras(clip_dir(scene, clip))
    K = torch.tensor(K).cuda(); c2w = torch.tensor(c2w).cuda(); dirs = pixel_dirs(K)
    if right.startswith("splat:"):
        p2, s2 = load_splat(right[6:]); fix = None
    else:
        from fixers.registry import load_fixer
        fix = load_fixer("ours", right)
    n = len(ts); w = imageio.get_writer(out, fps=10, quality=7)
    with torch.no_grad():
        for i in range(n):
            off = shift * min(1.0, max(0.0, (i - 10) / (n * 0.6)))  # hold on-path, then slide right
            c = c2w[i].clone(); c[:3, 3] += off * c[:3, 0]  # camera +x = right
            img = render(params, sky, c, K, cam_dirs=dirs)[0]
            r = fix(img) if fix is not None else render(p2, s2, c, K, cam_dirs=dirs)[0]
            fr = np.ascontiguousarray((torch.cat([img, r], 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[::2, ::2])
            cv2.putText(fr, f"raw splat  +{off:.1f} m right", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(fr, rlabel, (fr.shape[1] // 2 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            w.append_data(fr)
    w.close(); print("wrote", out)


if __name__ == "__main__":
    main()
