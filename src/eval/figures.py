"""Figures: trust-vs-offset curves per method with 95% bootstrap CIs (from summarize.py), and side-by-side panels
(real target lane | raw | Difix | Difix-FT | ours | progressive-refined) per offset bin.

usage: scripts/lib/py.sh src/eval/figures.py <summary.csv> <rows.json[,rows2.json]> <pairs_root> <fixed_root> <fig_dir> --ours ours_base_s0
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
METRICS = [("psnr", "PSNR vs real (dB) ↑"), ("lpips", "LPIPS vs real ↓"), ("road_lpips", "road-region LPIPS ↓"),
           ("mark_prec", "lane-paint precision ↑ (low = hallucinated paint)"), ("mark_rec", "lane-paint recall ↑ (low = erased paint)"),
           ("mark_f1", "lane-paint F1 ↑"), ("ntl_iou", "CLRNet lane NTL-IoU ↑"), ("lane_f1", "CLRNet lane F1 (CULane) ↑"),
           ("yolo_halluc_veh", "fake vehicles / image ↓")]


def chart(S, rows, methods, labels, path, ci="c"):
    xs = rows.assign(a=rows.offset_m.abs()).groupby("bin", observed=True).a.mean()
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (k, lab) in zip(axs.flat, METRICS):
        for j, m in enumerate(methods):
            g = S[S.method == m].set_index("bin").reindex(xs.index)
            if k not in g or g[k].isna().all():
                continue
            lo, hi = g[f"{k}_{ci}lo"], g[f"{k}_{ci}hi"]
            ax.errorbar(xs.values + (j - len(methods) / 2) * 0.05, g[k].values, yerr=[g[k] - lo, hi - g[k]],
                        color=PAL[j % len(PAL)], lw=2, marker="o", ms=6, capsize=3, label=labels[m])
        ax.set_title(lab, fontsize=10, loc="left"); ax.set_xlabel("lateral offset from recorded path (m)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(xs.values); ax.set_xticklabels([f"{v:.1f}" for v in xs.values])
    axs.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Held-out test scenes (Para-Lane scene_16 + scene_17): trust vs. lateral offset, against REAL target-lane "
                 "images (mean, 95% bootstrap CI over lane-pair sequences)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def panels(rows, pairs, fixed, methods, labels, figdir, n_per_bin=2):
    rng = np.random.default_rng(1)
    raw = rows[rows.method == "raw"].copy(); raw["a"] = raw.offset_m.abs()
    for lo, hi, name in [(-0.1, 1, "0m"), (1, 4.5, "3m"), (4.5, 20, "6m")]:
        sub = raw[(raw.a > lo) & (raw.a <= hi)]
        for sc in sorted(sub.scene.unique()):
            ss = sub[sub.scene == sc]
            for _, r in ss.iloc[rng.choice(len(ss), min(n_per_bin, len(ss)), replace=False)].iterrows():
                rel = f"{r.scene}/render/{r.src}_to_{r.dst}/full/{r.ts}.png"
                srcs = [(f"REAL {r.dst}", f"{pairs}/{r.scene}/real/{r.dst}/{r.ts}.png")]
                srcs += [(labels[m], f"{pairs}/{rel}" if m == "raw" else f"{fixed}/{m}/{rel}") for m in methods]
                w, h = 480, 270
                P = Image.new("RGB", (w * len(srcs), h + 22), "white"); D = ImageDraw.Draw(P)
                for c, (lab, p) in enumerate(srcs):
                    if os.path.exists(p):
                        P.paste(Image.open(p).convert("RGB").resize((w, h)), (c * w, 22))
                    D.text((c * w + 4, 5), lab, fill="black")
                D.text((4, 26), f"{r.scene} {r.src} splat -> {r.dst} pose, offset {r.offset_m:+.1f} m", fill=(255, 255, 0))
                P.save(f"{figdir}/panel_{name}_{r.scene}_{r.src}_to_{r.dst}_{r.ts}.jpg", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary"); ap.add_argument("rows"); ap.add_argument("pairs"); ap.add_argument("fixed"); ap.add_argument("figdir")
    ap.add_argument("--ours", default="ours_hf_s0")
    a = ap.parse_args(); os.makedirs(a.figdir, exist_ok=True)
    S = pd.read_csv(a.summary)
    rows = pd.DataFrame(sum([json.load(open(p)) for p in a.rows.split(",")], []))
    rows["bin"] = pd.cut(rows.offset_m.abs(), [-0.1, 1.0, 4.5, 20], labels=["0m", "~3m", "~6m"])
    labels = {"raw": "raw splat", "difix": "Difix (zero-shot)", "difix_ref": "Difix-ref (near ref)",
              "difix_ref_far": "Difix-ref (ref 2 s away)", "difix_ft": "Difix fine-tuned", a.ours: f"our U-Net ({a.ours[5:]})",
              "prog_ours": "progressive (ours), no post-fix", "prog_difix_ft": "progressive (Difix-FT), no post-fix",
              "prog_none": "refine w/o fixer (control)"}
    curve = [m for m in ["raw", "difix", "difix_ref", "difix_ft", a.ours, "prog_ours", "prog_difix_ft"] if m in set(S.method)]
    chart(S, rows, curve, labels, f"{a.figdir}/trust_vs_offset.png")
    panel_m = [m for m in ["raw", "difix", "difix_ft", a.ours, "prog_ours", "prog_difix_ft"] if m in set(S.method)]
    panels(rows, a.pairs, a.fixed, panel_m, labels, a.figdir)
    print("figures in", a.figdir)


if __name__ == "__main__":
    main()
