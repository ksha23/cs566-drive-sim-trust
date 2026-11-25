"""Figures for the HUGSIM track: NTA-IoU vs lateral offset (mean + 95% bootstrap CI) and an example panel
(raw / fixers at +2 m with projected GT boxes in red and YOLO vehicle detections in green).
usage: python plot_nta.py [--res results/hugsim] [--renders $CS566D_DATA/hugsim/renders] [--fixed $CS566D_DATA/hugsim/fixed]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from common.config import HUGSIM_DATA, YOLO_WEIGHTS

COL = {"raw": "#2a78d6", "difix": "#eb6834", "difix_ref": "#1baf7a", "difix_ft": "#e87ba4", "ours": "#eda100", "ours_best": "#4a3aa7"}
LAB = {"raw": "raw render", "difix": "Difix", "difix_ref": "Difix-ref (ref = 0 m render)", "difix_ft": "Difix-FT (Para-Lane)",
       "ours": "our U-Net v0 (night 1)", "ours_best": "our U-Net (selected, Para-Lane)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/hugsim")
    ap.add_argument("--renders", default=f"{HUGSIM_DATA}/renders")
    ap.add_argument("--fixed", default=f"{HUGSIM_DATA}/fixed")
    ap.add_argument("--scene", default="scene-0064"); ap.add_argument("--frame", type=int, default=10)
    a = ap.parse_args()
    rows = [l.split(",") for l in open(f"{a.res}/nta_iou_summary.csv").read().split("\n\n")[0].splitlines()[1:]]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in [m for m in COL if any(r[0] == m for r in rows)]:
        R = sorted([r for r in rows if r[0] == m], key=lambda r: int(r[1]))
        x = np.array([int(r[1]) for r in R]); y = np.array([float(r[2]) for r in R])
        lo = np.array([float(r[3]) for r in R]); hi = np.array([float(r[4]) for r in R])
        ax.plot(x, y, "-o", color=COL[m], lw=2, ms=7, label=LAB.get(m, m))
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.15, lw=0)
    ax.set_xlabel("|lateral offset| from recorded path (m)"); ax.set_ylabel("NTA-IoU  ↑")
    ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("HUGSIM nuScenes scenes 0064/0167/0661/0071: NTA-IoU (95% bootstrap CI)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{a.res}/nta_iou_vs_offset.png", dpi=130)

    # example panel
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    rec = json.load(open(f"{a.renders}/{a.scene}/index.json"))[a.frame]
    v = rec["views"]["2"]; ok = {g["id"] for g in rec["views"]["0"]["gt"] if g.get("vis") is None or g["vis"] >= 0.5}
    tiles = []
    for m in ["raw", "difix", "difix_ref", "difix_ft", "ours", "ours_best"]:
        p = f"{a.renders}/{v['path']}" if m == "raw" else f"{a.fixed}/{m}/{v['path']}"
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
        for g in v["gt"]:
            if g["id"] in ok:
                d.rectangle(g["box"], outline=(255, 0, 0), width=2)
        r = yolo(p, verbose=False, conf=0.25)[0].boxes
        for b, c in zip(r.xyxy.cpu().numpy(), r.cls.tolist()):
            if int(c) in {2, 3, 5, 7}:
                d.rectangle(b.tolist(), outline=(0, 255, 0), width=2)
        d.rectangle([0, 0, 330, 22], fill=(0, 0, 0)); d.text((6, 5), f"{LAB.get(m, m)}  (+2 m)", fill=(255, 255, 255))
        tiles.append(np.array(im))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    Image.fromarray(np.concatenate(rows, 0)).save(f"{a.res}/nta_example_{a.scene}_{a.frame}.jpg", quality=88)
    print("figures written")


if __name__ == "__main__":
    main()
