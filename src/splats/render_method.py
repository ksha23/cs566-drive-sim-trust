"""Render an alternative splat of lane A (e.g. the road-surface-prior fit) at the eval poses of the scene's 'full'
pairs (other lanes' real poses + A's held-out poses) and write it as a pseudo-method <fixed>/<method>/... so
evaluate.py can score it like a fixer output.
usage: scripts/lib/py.sh src/splats/render_method.py <scene> <clip> <splat.pt> <method_name>
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image

from splats.paralane import load_clip
from splats.gs import load_splat, render, pixel_dirs
from common.config import DATA



def main():
    scene, clip, sp, name = sys.argv[1:5]
    meta = [m for m in json.load(open(f"{DATA}/pairs/{scene}/meta.json")) if m["variant"] == "full" and m["src"] == clip]
    params, sky = load_splat(sp); clips = {}
    with torch.no_grad():
        for m in meta:
            if m["dst"] not in clips:  # (setdefault would reload the whole clip for every frame)
                clips[m["dst"]] = load_clip(scene, m["dst"])
            cb = clips[m["dst"]]; i = cb["ts"].index(m["ts"])
            img = render(params, sky, cb["c2w"][i], cb["K"], cam_dirs=pixel_dirs(cb["K"]))[0]
            p = f"{DATA}/fixed/{name}/{m['path']}"; os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)
    print("wrote", len(meta), "renders for", name, scene, clip)


if __name__ == "__main__":
    main()
