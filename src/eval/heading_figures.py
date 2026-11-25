"""Figures for the heading / camera-height axis: trust-vs-deviation curves next to the existing lateral curves, and
side-by-side panels at +-10 deg yaw and +1 m camera height.

  results/figures/trust_vs_heading_height.png   columns: lateral (real GT; results/eval/summary_test.csv) | yaw on-path
      (pseudo-GT = rotation-warped real frame) | lateral + yaw (real adjacent-lane pose, rotated) | camera height
      (road-plane-warped real frame, road metrics only); rows: PSNR, LPIPS, paint F1, CLRNet lane F1, fake vehicles,
      lane stability (consistency with the same method's 0-deviation output). Error bars: 95% bootstrap over lane-pair
      sequences.
  results/figures/panel_heading_<cond>_<scene>_<src>_<ts>.jpg   rows: 0-deviation, deviated; first column = real frame
      (0) / warped real frame (deviated, black = unknown).

usage: scripts/lib/py.sh src/eval/heading_figures.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from common.config import DATA, REPO
from eval.heading_geom import homography, warp_img

OUT = f"{REPO}/results/heading"; FIG = f"{REPO}/results/figures"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
METHODS = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]
LABEL = {"raw": "raw splat", "difix": "Difix [1] zero-shot", "difix_ref": "Difix-ref [1]", "nv_fixer": "NVIDIA Fixer [4]",
         "difix_ft": "Difix-FT", "difix_reimpl_ref": "our Difix-ref reimpl.", "ours_hfgan_s0": "our U-Net (hfgan_s0)"}
# (row label, lateral key, yaw key, lateral_yaw key, height key)
ROWS = [("PSNR ↑ (dB)", "psnr", "psnr", "psnr", "road_psnr"),
        ("LPIPS ↓", "lpips", "lpips", "lpips", None),  # height: road LPIPS is not comparable across dz (region shrinks)
        ("lane-paint F1 ↑", "mark_f1", "mark_f1", "mark_f1", "mark_f1"),
        ("CLRNet lane F1 ↑", "lane_f1", "lane_f1", "lane_f1", "lane_f1"),
        ("fake vehicles / image ↓", "yolo_halluc_veh", "yolo_halluc_veh", "yolo_halluc_veh", "yolo_halluc_veh"),
        ("lane stability F1 ↑ (vs own 0-dev.)", None, "lane_stab_lane_f1", "lane_stab_lane_f1", "lane_stab_lane_f1")]
COLS = ["lateral offset (m), real GT", "yaw on-path (deg), rotation-warped real",
        "lateral + yaw (deg) at adjacent-lane poses", "camera height change (m), road-plane-warped real (road only)"]


def curves():
    S = pd.read_csv(f"{OUT}/summary.csv")
    L = pd.read_csv(f"{REPO}/results/eval/summary_test.csv")
    rows = pd.concat([pd.DataFrame(json.load(open(f"{REPO}/results/eval/{f}"))) for f in
                      ("eval_scene_16_a.json", "eval_scene_17_a.json")])
    xs = rows.assign(a=rows.offset_m.abs(), b=pd.cut(rows.offset_m.abs(), [-0.1, 1.0, 4.5, 20], labels=["0m", "~3m", "~6m"])
                     ).groupby("b", observed=True).a.mean()
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(20, 3.1 * len(ROWS)), sharey="row")
    for i, (lab, kl, ky, kly, kh) in enumerate(ROWS):
        for j, (ax, key, axis) in enumerate(zip(axs[i], [kl, ky, kly, kh], [None, "yaw", "lateral_yaw", "height"])):
            for c, m in enumerate(METHODS):
                if key is None:
                    continue
                if axis is None:
                    g = L[L.method == m].set_index("bin").reindex(xs.index)
                    if key not in g:
                        continue
                    x = xs.values; y, lo, hi = g[key].values, g[key + "_clo"].values, g[key + "_chi"].values
                else:
                    g = S[(S.axis == axis) & (S.method == m)].sort_values("level")
                    if key not in g or g[key].isna().all():
                        continue
                    x = g.level.values; y, lo, hi = g[key].values, g[key + "_clo"].values, g[key + "_chi"].values
                dx = (c - len(METHODS) / 2) * (0.04 if axis in (None, "height") else 0.25)
                ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=PAL[c], lw=1.8, marker="o", ms=4, capsize=2,
                            label=LABEL[m] if (i == 0 and j == 0) else None)
            if key is None:
                ax.text(0.5, 0.5, "n/a (no 0-deviation\ncounterpart for lateral)" if j == 0 else
                        "n/a: the road region with pseudo-GT\nshrinks with dz (46 % of the frame at -1 m);\nsee matched d_road_psnr in summary.md",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
            ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(COLS[j], fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(lab, fontsize=10)
            if i == len(ROWS) - 1:
                ax.set_xlabel(["|offset| (m)", "yaw (deg, + = left)", "yaw (deg)", "dz (m, + = up)"][j], fontsize=9)
    fig.legend(loc="upper center", ncol=7, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Para-Lane test scenes (scene_16 + scene_17): trust vs. lateral offset, heading and camera height "
                 "(mean, 95% bootstrap CI over lane-pair sequences)", fontsize=12, x=0.01, ha="left", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{FIG}/trust_vs_heading_height.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def panels(conds=("y+10_z+0.0", "y-10_z+0.0", "y+0_z+1.0"), per_scene=1):
    rng = np.random.default_rng(3)
    for scene in ("scene_16", "scene_17"):
        meta = json.load(open(f"{DATA}/heading/{scene}/meta.json"))
        for cond in conds:
            cand = [m for m in meta if m["cond"] == cond and m["onpath"]]
            for m in [cand[k] for k in rng.choice(len(cand), per_scene, replace=False)]:
                real = np.array(Image.open(f"{DATA}/pairs/{m['real']}").convert("RGB"))
                gw = warp_img(real, homography(m["K"], m["c2w0"], m["c2w1"], m["cam_h"]))[0]
                w, h = 384, 216; cols = 1 + len(METHODS)
                P = Image.new("RGB", (w * cols, 2 * h + 44), "white"); D = ImageDraw.Draw(P)
                for r, (rel, first, tag) in enumerate([(m["zero"], real, "0-deviation (recorded pose)"),
                                                       (m["path"], gw, f"{cond} ({'road-plane' if m['dz'] else 'rotation'}-warped real)")]):
                    y = 22 + r * (h + 22)
                    P.paste(Image.fromarray(first).resize((w, h)), (0, y)); D.text((4, y - 16), "REAL " + tag, fill="black")
                    for c, meth in enumerate(METHODS):
                        p = f"{DATA}/heading/{rel}" if meth == "raw" else f"{DATA}/heading/fixed/{meth}/{rel}"
                        if os.path.exists(p):
                            P.paste(Image.open(p).convert("RGB").resize((w, h)), ((c + 1) * w, y))
                        D.text(((c + 1) * w + 4, y - 16), LABEL[meth], fill="black")
                P.save(f"{FIG}/panel_heading_{cond}_{scene}_{m['src']}_{m['ts']}.jpg", quality=88)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    curves(); panels()
    print("figures in", FIG)
