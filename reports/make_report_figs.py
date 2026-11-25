"""Figures of the mid-term report (reports/figs/), sized for a 7.2 in text column so that labels print at 7-8 pt.
Reads only files already in results/final/ (written by src/eval/final_report.py and src/viz/qualitative.py):
  fig1_trust_vs_offset.png      <- pooled_all_metrics.csv
  fig2_lane_f1_per_road_3m.png  <- fig2_lane_f1_per_road_3m.csv
  fig3_hallucination.png        <- qual_hallucination.jpg (tiles cropped below their labels) + qual_examples.json;
                                   tile metrics from the per-image rows (results/{eval,lanes,difix_reimpl,eval_more})
Method names follow Table 1 of the report.
usage (from the repo root): python reports/make_report_figs.py
"""
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RES, OUT = "results/final", "reports/figs"
ARMS = ["raw", "difix", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0", "prog_difix_reimpl", "road_prior"]
NAME = {"raw": "Raw render", "difix": "Released Difix", "nv_fixer": "NVIDIA Fixer", "difix_ft": "Difix-FT",
        "difix_reimpl_ref": "Our re-implementation", "ours_hfgan_s0": "Small U-Net",
        "prog_difix_reimpl": "Progressive 3D update", "road_prior": "Road-surface prior"}
COL = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#8c564b", "#e87ba4", "#008300", "#4a3aa7", "#7a7a7a"]))
MK = dict(zip(ARMS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
LS = dict(zip(ARMS, ["-", "--", "-.", "-", "--", ":", "-.", ":"]))
BINS = ["0m", "~3m", "~6m"]
ROAD = {"scene_16": "16\nstraight,\ncrosswalk", "scene_17": "17\nstraight,\nwide", "scene_14": "14\ncurve",
        "scene_15": "15\nunder\nhighway", "scene_10": "10\nlong\nstraight", "scene_3": "3\nboulevard",
        "scene_11": "11\nboulevard", "scene_4": "4\nboulevard"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02, "figure.facecolor": "white"})


def rows(path):
    return list(csv.DictReader(open(path)))


def fig1():
    S = rows(f"{RES}/pooled_all_metrics.csv")
    get = lambda m, b: next(r for r in S if r["method"] == m and r["bin"] == b)
    xs = [float(np.median([float(r["med_offset_m"]) for r in S if r["bin"] == b])) for b in BINS]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.85))
    panels = [("lane_f1", "Lane F1 (CLRNet) ↑", (0, 1.02)), ("lpips", "LPIPS to the real image ↓", None),
              ("yolo_halluc_veh", "Spurious vehicles per image ↓", None)]
    for ax, (k, t, yl) in zip(axs, panels):
        for i, m in enumerate(ARMS):
            y = np.array([float(get(m, b)[k]) for b in BINS])
            lo = np.array([float(get(m, b)[k + "_lo"]) for b in BINS]); hi = np.array([float(get(m, b)[k + "_hi"]) for b in BINS])
            ax.errorbar(np.array(xs) + (i - 3.5) * 0.08, y, yerr=[y - lo, hi - y], color=COL[m], marker=MK[m], ls=LS[m],
                        lw=1.1, ms=3.5, capsize=1.5, elinewidth=0.7, label=NAME[m], mec="white", mew=0.4)
        ax.set_title(t, loc="left", fontweight="bold")
        ax.set_xticks(xs); ax.set_xticklabels([f"{x:.1f} m" for x in xs])
        if yl:
            ax.set_ylim(*yl)
    axs[1].set_xlabel("lateral offset from the recorded path (median of each bin)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17), handlelength=2.6,
               columnspacing=1.2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig1_trust_vs_offset.png"); plt.close(fig)


def fig2():
    R = rows(f"{RES}/fig2_lane_f1_per_road_3m.csv")
    roads = list(dict.fromkeys(r["road"] for r in R))
    fig, ax = plt.subplots(figsize=(7.2, 1.42))
    w = 0.8 / len(ARMS)
    for j, s in enumerate(roads):
        for i, m in enumerate(ARMS):
            r = next((r for r in R if r["road"] == s and r["method"] == m), None)
            if r is None:
                continue
            v, lo, hi = float(r["lane_f1"]), float(r["lo"]), float(r["hi"]); x = j + (i - (len(ARMS) - 1) / 2) * w
            ax.bar(x, v, w * 0.9, color=COL[m], label=NAME[m] if j == 0 else None, zorder=2)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="#1f1f1e", lw=0.6, capsize=0.8, zorder=3)
    ax.set_xticks(range(len(roads))); ax.set_xticklabels([ROAD.get(s, s) for s in roads], fontsize=6.5)
    ax.set_ylabel("Lane F1 at ~3 m ↑"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.55, len(roads) - 0.45)
    ax.legend(ncol=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0.2)
    fig.savefig(f"{OUT}/fig2_lane_f1_per_road_3m.png"); plt.close(fig)


def fig3():
    # qual_hallucination.jpg: 3 x 2 tiles of 640 x 360 px below a 78 px header, 44 px caption under each row
    # (src/viz/qualitative.py, grid()); tile order real, raw, Difix, NVIDIA Fixer, Difix-FT, our re-impl.
    im = Image.open(f"{RES}/qual_hallucination.jpg").convert("RGB")
    pos = {"real": (0, 0), "raw": (1, 0), "difix_ft": (1, 1), "difix_reimpl_ref": (2, 1)}
    ex = next(e for e in json.load(open(f"{RES}/qual_examples.json")) if e["file"] == "qual_hallucination.jpg")
    cap = {"real": "Real image at this pose", "raw": "Raw render", "difix_ft": "Difix-FT",
           "difix_reimpl_ref": "Our re-implementation"}
    key = (ex["scene"], ex["src"], ex["dst"], ex["ts"]); val = {}
    for f in sorted(glob.glob(f"results/*/*{ex['scene']}*.json")):
        rows = json.load(open(f))
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and (r.get("scene"), r.get("src"), r.get("dst"), str(r.get("ts"))) == key:
                val.setdefault(r.get("method"), {}).update({k: r[k] for k in ("lane_f1", "mark_prec", "mark_rec") if k in r})
    sub = {"real": f"{ex['offset_m']:.1f} m from the training lane"}
    sub.update({k: f"lane F1 {val[k]['lane_f1']:.2f}\npaint P / R {val[k]['mark_prec']:.2f} / {val[k]['mark_rec']:.2f}"
                for k in pos if k != "real"})
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.45))
    for ax, k in zip(axs, ["real", "raw", "difix_ft", "difix_reimpl_ref"]):
        c, r = pos[k]; x0, y0 = c * 640, 78 + r * (360 + 44)
        tile = np.asarray(im.crop((x0, y0 + 30, x0 + 640, y0 + 360)))   # drop the burned-in label strip
        ax.imshow(tile)
        # retrace the thin erased-paint box (blue, drawn by qualitative.py) so it stays visible at print size
        blue = (np.abs(tile.astype(int) - (42, 120, 214)).max(-1) < 45); blue[: tile.shape[0] // 2] = False
        if blue.sum() > 30:
            yy, xx = np.nonzero(blue)
            ax.add_patch(plt.Rectangle((xx.min(), yy.min()), xx.max() - xx.min(), yy.max() - yy.min(), fill=False,
                                       ec="#2a78d6", lw=1.4))
        ax.set_title(cap[k], fontsize=8, fontweight="bold", pad=2)
        ax.set_xlabel(sub[k], fontsize=7.5, labelpad=2, linespacing=1.15)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(wspace=0.04, left=0, right=1)
    fig.savefig(f"{OUT}/fig3_hallucination.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); style()
    fig1(); fig2(); fig3()
