"""Figures for the FaithFusion arm: (1) one example row per offset bin: raw render | Difix output | real target lane |
PAINT / CHANGE hallucination labels | EIG | reprojection inconsistency; (2) AUROC per score, method and offset bin.
usage: python -m progressive.faithfusion.figures [--out results/faithfusion]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common.config import DATA as D
from progressive.faithfusion.halluc_auroc import labels_for, HW

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def examples(out, scene="scene_16", method="difix"):
    meta = [m for m in json.load(open(f"{D}/pairs/{scene}/meta.json")) if m["variant"] == "full"]
    items = [m for k, m in enumerate(meta) if k % 2 == 0 and not m["onpath_test"]]
    pick = []
    for lo, hi in [(1, 4.5), (4.5, 20)]:
        c = [m for m in items if lo < abs(m["offset_m"]) <= hi]
        pick.append(c[len(c) // 2])
    cols = ["raw splat render", f"{method} output", "REAL target lane", "PAINT label", "CHANGE label", "FaithFusion log-EIG", "reprojection inconsistency"]
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(3.2 * len(cols), 2.1 * len(pick)))
    for r, m in enumerate(pick):
        Z = np.load(f"{D}/faithfusion/maps/{scene}/{m['src']}_to_{m['dst']}/{m['ts']}.npz")
        L = labels_for(method, m)
        imgs = [Image.open(f"{D}/pairs/{m['path']}").resize(HW), Image.open(f"{D}/fixed/{method}/{m['path']}").resize(HW),
                Image.open(f"{D}/pairs/{m['real']}").resize(HW)]
        for c, im in enumerate(imgs):
            ax[r, c].imshow(im)
        ax[r, 3].imshow(np.where(L["paint"][1], L["paint"][0], np.nan), cmap="Reds", vmin=0, vmax=1)
        ax[r, 4].imshow(np.where(L["change"][1], L["change"][0], np.nan), cmap="Reds", vmin=0, vmax=1)
        g = Z["gain"].astype(np.float32); ax[r, 5].imshow(np.where(g > 99, np.nan, g), cmap="magma", vmin=0, vmax=3)
        ax[r, 6].imshow(Z["unobs"].astype(np.float32), cmap="magma", vmin=0, vmax=1)
        ax[r, 0].set_ylabel(f"{m['src']}->{m['dst']}\n{m['offset_m']:+.1f} m", fontsize=8)
        for c in range(len(cols)):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{out}/ff_examples_{scene}_{method}.jpg", dpi=110); plt.close(fig)


def auroc_chart(out):
    rows = json.load(open(f"{out}/halluc_auroc.json"))
    labs = ["paint", "change"]; scores = [("gain", "FaithFusion EIG"), ("unobs", "reprojection inconsistency"), ("1-alpha", "1 - splat alpha")]
    meths = sorted({r["method"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, lab in zip(axs, labs):
        xs = []; labels = []
        for mi, meth in enumerate(meths):
            for bi, b in enumerate(["~3m", "~6m"]):
                xs.append(mi * 3 + bi); labels.append(f"{meth.replace('ours_base_s0', 'our U-Net')}\n{b}")
        for j, (k, nm) in enumerate(scores):
            vals, lo, hi = [], [], []
            for meth in meths:
                for b in ["~3m", "~6m"]:
                    r = [x for x in rows if x["method"] == meth and x["bin"] == b and x["label"] == lab]
                    v = r[0][f"auroc_{k}"] if r else np.nan
                    vals.append(v)
                    lo.append(v - r[0].get(f"auroc_{k}_lo", v) if r else 0); hi.append(r[0].get(f"auroc_{k}_hi", v) - v if r else 0)
            ax.bar(np.array(xs) + (j - 1) * 0.28, vals, width=0.26, color=PAL[j], label=nm,
                   yerr=[lo, hi] if j < 2 else None, capsize=2, error_kw=dict(lw=0.8))
        ax.axhline(0.5, color="#888", lw=0.8, ls="--")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7); ax.set_ylim(0.1, 1.0)
        ax.set_title(f"{lab.upper()} hallucination: AUROC of per-pixel scores", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("AUROC (0.5 = chance)"); axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Do trust maps predict where fixers hallucinate? (Para-Lane test scenes 16+17, cross-lane renders; 95% CI over images)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{out}/ff_auroc.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/faithfusion"); a = ap.parse_args()
    examples(a.out); auroc_chart(a.out); print("figures in", a.out)


if __name__ == "__main__":
    main()
