"""Compact markdown tables for RESULTS.md from results/eval/summary_*.csv (written by summarize.py).
usage: scripts/lib/py.sh src/eval/results_tables.py > results/eval/tables.md"""
import sys
import pandas as pd

NAMES = {"raw": "raw splat", "difix": "Difix [1] zero-shot", "difix_ref": "Difix-ref [1], ref = nearest real source-lane frame",
         "difix_ref_far": "Difix-ref [1], ref >= 2 s away", "difix_ft": "Difix-FT (released Difix fine-tuned on our pairs)",
         "ours_base_s0": "our fixer (small, from scratch) base seed 0", "ours_base_s1": "our fixer base seed 1",
         "ours_hf_s0": "our fixer +HF loss seed 0", "ours_hf_s1": "our fixer +HF loss seed 1",
         "ours_hfgan_s0": "our fixer +HF+GAN seed 0", "ours_hfgan_s1": "our fixer +HF+GAN seed 1",
         "prog_difix_ft": "progressive update w/ Difix-FT (splat only)", "prog_ours": "progressive update w/ our fixer (splat only)",
         "prog_difix_ft_paper": "progressive w/ Difix-FT, 1.5 m steps x 1.5k iters (splat only)",
         "prog_nv_fixer": "progressive update w/ NVIDIA Fixer (splat only)",
         "prog_ff_flat": "progressive w/ Difix-ref, flat weight (FaithFusion arm)",
         "prog_ff_eig": "progressive w/ Difix-ref, EIG weight [2]",
         "prog_none": "refine w/o fixer (control)", "road_prior": "road-surface prior splat (no fixer)",
         "nv_fixer": "NVIDIA Fixer [4] zero-shot",
         "difix_reimpl_ref": "our Difix-ref reimplementation [1] (SD-Turbo init, driving pairs, 576x1024)",
         "difix_reimpl448": "our Difix trainer, matched run (no ref, 448x768, 1801 steps)",
         "difix_official448": "official Difix trainer, matched run (no ref, 448x768, 1801 steps)",
         "prog_difix_reimpl": "progressive update w/ our Difix-ref reimplementation (splat only)"}
COLS = [("psnr", "PSNR↑", 2), ("lpips", "LPIPS↓", 3), ("road_lpips", "road LPIPS↓", 3), ("mark_prec", "paint P↑", 2),
        ("mark_rec", "paint R↑", 2), ("mark_f1", "paint F1↑", 2), ("ntl_iou", "NTL-IoU↑", 2), ("lane_f1", "lane F1↑", 2),
        ("yolo_halluc_veh", "fake veh.↓", 3)]


def table(csv, title, order):
    S = pd.read_csv(csv)
    out = [f"### {title}\n", f"Source: `{csv}` (mean; [95% bootstrap CI over lane-pair sequences] for LPIPS and lane F1).\n",
           "| offset | method | n | " + " | ".join(c[1] for c in COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
    for b in ["0m", "~3m", "~6m"]:
        for m in order:
            r = S[(S.bin == b) & (S.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]; cells = []
            for k, _, d in COLS:
                v = r.get(k)
                if pd.isna(v):
                    cells.append("–"); continue
                s = f"{v:.{d}f}"
                if k in ("lpips", "lane_f1") and not pd.isna(r.get(k + "_clo")):
                    s += f" [{r[k + '_clo']:.{d}f}, {r[k + '_chi']:.{d}f}]"
                cells.append(s)
            out.append(f"| {b} | {NAMES.get(m, m)} | {int(r.n)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    order = list(NAMES)
    s16_only = ("prog_none", "road_prior", "prog_difix_ft_paper", "prog_ff_flat", "prog_ff_eig")  # run on scene_16 only
    both = [m for m in order if m not in s16_only]
    print(table("results/eval/summary_test.csv", "Both test scenes (scene_16 + scene_17)", both))
    print(table("results/eval/summary_scene_16.csv", "scene_16 only (includes the no-fixer control and road prior)", order))
    print(table("results/eval/summary_scene_17.csv", "scene_17 only", order))


if __name__ == "__main__":
    main()
