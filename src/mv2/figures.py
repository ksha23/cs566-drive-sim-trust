"""MV2 panels: real | raw splat | each fixer, for a few cross-vehicle frames (labels burned in).
usage: scripts/lib/py.sh src/mv2/figures.py <dir> <seq> <frame> [<seq> <frame> ...]  -> results/mv2/panel_<dir>_<seq>_<frame>.jpg
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.config import REPO
from mv2.evaluate import ROOT, FIXED

COLS = [("real", "real (other vehicle)"), ("raw", "raw splat"), ("difix", "Difix [1]"), ("difix_ref", "Difix-ref [1]"),
        ("nv_fixer", "NVIDIA Fixer [4]"), ("difix_ft", "Difix-FT"), ("difix_reimpl_ref", "our Difix-ref reimpl."),
        ("ours_hfgan_s0", "our U-Net")]


def main():
    d = sys.argv[1]; args = sys.argv[2:]
    meta = {(m["scene"], m["frame"], m["kind"]): m for m in json.load(open(f"{ROOT}/{d}/meta_all.json"))}
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for s, k in zip(args[::2], args[1::2]):
        m = meta[(s, int(k), "cross")]
        tiles = []
        for key, lab in COLS:
            p = f"{ROOT}/{m['real']}" if key == "real" else f"{ROOT}/{m['path']}" if key == "raw" else f"{FIXED}/{key}/{m['path']}"
            im = Image.open(p).convert("RGB").resize((480, 270), Image.BICUBIC); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 480, 26], fill=(0, 0, 0)); dr.text((6, 3), lab, fill=(255, 255, 255), font=font)
            tiles.append(np.array(im))
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
        im = Image.fromarray(grid); dr = ImageDraw.Draw(im)
        dr.text((6, 540 - 24), f"MV2 {s} frame {k}: {d}, lateral {m['lat_m']:+.1f} m, camera height {m['dh_m']:+.2f} m",
                fill=(255, 255, 0), font=font)
        im.save(f"{REPO}/results/mv2/panel_{d}_{s}_{int(k):03d}.jpg", quality=80)


if __name__ == "__main__":
    main()
