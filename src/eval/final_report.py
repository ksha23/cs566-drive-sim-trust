"""Final consolidated figures and tables (results/final/). CPU only; reads result files already in results/.

  fig1_trust_vs_offset.png    lane F1, LPIPS and fake vehicles per image vs lateral offset, every arm, pooled over all
                              Para-Lane test roads (95% bootstrap CI over lane-pair sequences)
  fig2_lane_f1_per_road_3m.png  lane F1 at ~3 m per test road (bars, CI over lane-pair sequences of that road)
  fig3_heading_height.png     heading (yaw) and camera-height curves (results/heading, scene_16 + scene_17)
  table1..4 *.csv + tables.md hallucination; Difix reimplementation validation; cross-dataset; closed loop
  pooled_all_metrics.csv      every metric, every arm, every offset bin, pooled over all test roads
usage: scripts/lib/py.sh src/eval/final_report.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.summarize import METRICS, LABELS, boot
from eval.more_scenes_summary import load_rows

OUT = "results/final"
# arms pooled over ALL test roads (released Difix-ref covers all eight since stage 22 'ref')
ARMS = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0", "prog_difix_reimpl", "road_prior"]
NAME = {"raw": "raw splat", "difix": "released Difix", "difix_ref": "released Difix-ref", "nv_fixer": "NVIDIA Fixer",
        "difix_ft": "Difix-FT (released weights fine-tuned)", "difix_reimpl_ref": "our Difix-ref reimpl.",
        "ours_hfgan_s0": "our small U-Net", "prog_difix_reimpl": "progressive 3D update (our reimpl.)",
        "road_prior": "road-surface prior (no fixer)"}
SHORT = {"raw": "raw", "difix": "Difix", "difix_ref": "Difix-ref", "nv_fixer": "NV Fixer", "difix_ft": "Difix-FT", "difix_reimpl_ref": "our reimpl.",
         "ours_hfgan_s0": "U-Net", "prog_difix_reimpl": "progressive", "road_prior": "road prior"}
# validated categorical palette (dataviz reference instance, fixed order) + marker/linestyle as secondary encoding
PAL = ["#2a78d6", "#eb6834", "#9c6b30", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COL = dict(zip(ARMS, PAL))
MK = dict(zip(ARMS, ["o", "s", "h", "^", "D", "v", "P", "X", "*"]))
LS = dict(zip(ARMS, ["-", "--", ":", "-.", "-", "--", ":", "-.", ":"]))
INK, INK2, GRID = "#1f1f1e", "#5c5b55", "#e4e3dc"
SCENE_DESC = {"scene_16": "scene_16\nstraight, crosswalk", "scene_17": "scene_17\nstraight, wide",
              "scene_14": "scene_14\ncurve", "scene_15": "scene_15\nunder highway",
              "scene_10": "scene_10\nlong straight", "scene_3": "scene_3\nboulevard",
              "scene_11": "scene_11\nboulevard", "scene_4": "scene_4\nboulevard"}


def style():
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def pooled(df):
    recs = []
    for (b, m), g in df.groupby(["bin", "method"], observed=True):
        r = dict(bin=b, method=m, n_img=len(g), n_seq=g.seq.nunique(), n_roads=g.scene.nunique(),
                 roads=",".join(sorted(g.scene.unique(), key=lambda s: int(s.split("_")[1]))),
                 med_offset_m=float(g.offset_m.abs().median()))
        for k in METRICS:
            if k in g and not g[k].isna().all():
                r[k] = float(np.nanmean(g[k])); r[k + "_lo"], r[k + "_hi"] = boot(g[k].values, groups=g.seq.values)
        recs.append(r)
    return pd.DataFrame(recs)


def fig1(S, roads):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
    panels = [("lane_f1", "Lane agreement (CLRNet lane F1) ↑", (0, 1.02)),
              ("lpips", "Image distance to real (LPIPS) ↓", None),
              ("yolo_halluc_veh", "Fake vehicles per image ↓", None)]
    xs = {b: S[S.bin == b].med_offset_m.median() for b in LABELS}
    for ax, (k, t, yl) in zip(axs, panels):
        for i, m in enumerate(ARMS):
            g = S[S.method == m].set_index("bin").reindex(LABELS)
            x = np.array([xs[b] for b in LABELS]) + (i - 3.5) * 0.07
            y, lo, hi = g[k].values, g[k + "_lo"].values, g[k + "_hi"].values
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COL[m], marker=MK[m], ls=LS[m], lw=2, ms=8, capsize=3,
                        elinewidth=1.3, label=NAME[m], mec="white", mew=0.8)
        ax.set_title(t, loc="left", fontweight="bold")
        ax.set_xticks([xs[b] for b in LABELS])
        ax.set_xticklabels([f"on path\n0 m", f"one lane over\n{xs['~3m']:.1f} m", f"two lanes over\n{xs['~6m']:.1f} m"])
        if k == "lpips":
            ax.set_xlabel("lateral offset from the recorded path (median of each bin)")
        if yl:
            ax.set_ylim(*yl)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))   # below the axis label
    fig.suptitle(f"Trust vs. lateral offset, pooled over {len(roads)} held-out Para-Lane test roads "
                 f"(95% bootstrap CI over lane-pair sequences)", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/fig1_trust_vs_offset.png"); plt.close(fig)


def fig2(df, roads):
    d = df[(df.bin == "~3m") & df.method.isin(ARMS)]
    fig, ax = plt.subplots(figsize=(17, 5.8))
    w = 0.8 / len(ARMS); rows = []
    for j, s in enumerate(roads):
        for i, m in enumerate(ARMS):
            g = d[(d.scene == s) & (d.method == m)]
            if not len(g):
                continue
            v = float(np.nanmean(g.lane_f1)); lo, hi = boot(g.lane_f1.values, groups=g.seq.values)
            rows.append(dict(road=s, method=m, n_img=len(g), n_seq=g.seq.nunique(), lane_f1=v, lo=lo, hi=hi))
            ax.bar(j + (i - (len(ARMS) - 1) / 2) * w, v, w * 0.9, color=COL[m], edgecolor="white", lw=0.8,
                   label=NAME[m] if j == 0 else None, zorder=2)
            ax.errorbar(j + (i - (len(ARMS) - 1) / 2) * w, v, yerr=[[v - lo], [hi - v]] if not np.isnan(lo) else None,
                        color=INK, lw=1.1, capsize=2, zorder=3)
    ax.set_xticks(range(len(roads))); ax.set_xticklabels([SCENE_DESC.get(s, s) for s in roads])
    ax.set_ylabel("CLRNet lane F1 at ~3 m ↑"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.36))
    ax.set_title("Lane agreement one lane over (~3 m), per held-out test road (95% CI over lane-pair sequences)",
                 loc="left", fontweight="bold")
    fig.savefig(f"{OUT}/fig2_lane_f1_per_road_3m.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{OUT}/fig2_lane_f1_per_road_3m.csv", index=False)


def fig3():
    H = pd.read_csv("results/heading/summary.csv")
    arms = ["raw", "difix", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    spec = [(axs[0, 0], "yaw", "d_psnr", "Heading: ΔPSNR vs. the same method at 0° (dB) ↑", "yaw about the camera center (°, + = left)"),
            (axs[0, 1], "yaw", "lane_f1", "Heading: lane F1 vs. rotated real frame ↑", "yaw about the camera center (°, + = left)"),
            (axs[1, 0], "height", "d_road_psnr", "Camera height: Δ road PSNR vs. same method at 0 m (dB) ↑", "camera height change (m)"),
            (axs[1, 1], "height", "lane_f1", "Camera height: lane F1 vs. road-plane warp ↑", "camera height change (m)")]
    rows = []
    for ax, axis, k, t, xl in spec:
        for i, m in enumerate(arms):
            g = H[(H.axis == axis) & (H.method == m)].sort_values("level")
            if not len(g) or k not in g:
                continue
            x = g.level.values + (i - 2.5) * (0.35 if axis == "yaw" else 0.025)
            y = g[k].values; lo = g.get(k + "_clo", pd.Series(y)).values; hi = g.get(k + "_chi", pd.Series(y)).values
            lo = np.where(np.isnan(lo), y, lo); hi = np.where(np.isnan(hi), y, hi)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COL[m], marker=MK[m], ls=LS[m], lw=2, ms=7, capsize=2.5,
                        label=NAME[m], mec="white", mew=0.7)
            for lv, v, a, b in zip(g.level, y, lo, hi):
                rows.append(dict(axis=axis, level=lv, method=m, metric=k, value=v, lo=a, hi=b))
        ax.set_title(t, loc="left", fontweight="bold", fontsize=13); ax.set_xlabel(xl)
        if k == "lane_f1":
            ax.set_ylim(-0.03, 1.03)
        ax.axhline(0, color=INK2, lw=0.8) if k.startswith("d_") else None
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    nr = len(glob.glob("results/heading/rows_scene_*_raw.json"))
    fig.suptitle(f"Heading and camera height on Para-Lane ({nr} held-out test roads, on-path frames; 95% CI over lane-pair "
                 "sequences)\n"
                 "Yaw has an exact ground truth (rotation homography); height is scored on the road plane only",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(f"{OUT}/fig3_heading_height.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{OUT}/fig3_heading_height.csv", index=False)


def ci(r, k, d=2, lo="_lo", hi="_hi"):
    v = r.get(k)
    if v is None or pd.isna(v):
        return "–"
    a, b = r.get(k + lo), r.get(k + hi)
    return f"{v:.{d}f}" + (f" [{a:.{d}f}, {b:.{d}f}]" if a is not None and not pd.isna(a) else "")


def table1(S):
    rows, md = [], ["| offset | method | images | paint precision ↑ | paint recall ↑ | paint F1 ↑ | fake vehicles / img ↓ | lane F1 ↑ |",
                    "|---|---|---|---|---|---|---|---|"]
    for b in ("~3m", "~6m"):
        for m in ARMS:
            r = S[(S.bin == b) & (S.method == m)]
            if not len(r):
                continue
            r = r.iloc[0].to_dict()
            rows.append({k: r.get(k) for k in ["bin", "method", "n_img", "n_seq", "n_roads", "mark_prec", "mark_prec_lo", "mark_prec_hi",
                                                "mark_rec", "mark_rec_lo", "mark_rec_hi", "mark_f1", "mark_f1_lo", "mark_f1_hi",
                                                "yolo_halluc_veh", "yolo_halluc_veh_lo", "yolo_halluc_veh_hi", "lane_f1", "lane_f1_lo", "lane_f1_hi"]})
            md.append(f"| {b} | {NAME[m]} | {r['n_img']} | {ci(r, 'mark_prec')} | {ci(r, 'mark_rec')} | {ci(r, 'mark_f1')} | "
                      f"{ci(r, 'yolo_halluc_veh', 3)} | {ci(r, 'lane_f1')} |")
    pd.DataFrame(rows).to_csv(f"{OUT}/table1_hallucination.csv", index=False)
    return "\n".join(md)


def table2():
    D = pd.read_csv("results/dl3dv/summary.csv")
    V = json.load(open("results/difix_reimpl/val_scene19.json"))
    T = pd.read_csv("results/difix_reimpl/summary_test.csv")
    L = [("raw", "raw", "raw", "raw render (no fixer)"), ("released", "released", "difix", "released Difix (nvidia/difix)"),
         ("released_ref", "released_ref", "difix_ref", "released Difix-ref (nvidia/difix_ref)"),
         ("official448", "official448", "difix_official448", "official trainer, matched (SD-Turbo, no ref, 448x768)"),
         ("ours448", "ours448", "difix_reimpl448", "our trainer, matched (SD-Turbo, no ref, 448x768)"),
         ("oursref", "ours_ref", "difix_reimpl_ref", "our trainer, ref view (576x1024)")]
    md = ["| model | DL3DV S=30: PSNR / LPIPS [CI] / FID | DL3DV S=60: PSNR / LPIPS [CI] / FID | Para-Lane val (scene_19): PSNR / LPIPS / FID | Para-Lane test LPIPS ~3 m / ~6 m | Para-Lane test lane F1 ~3 m [CI] / ~6 m |",
          "|---|---|---|---|---|---|"]
    rows = []
    for dk, vk, tk, name in L:
        cells = []; rec = dict(model=name)
        for lv in (30, 60):
            r = D[(D.level == lv) & (D.method == dk)]
            if len(r):
                r = r.iloc[0]
                cells.append(f"{r.psnr:.2f} / {r.lpips:.3f} [{r.lpips_lo:.3f}, {r.lpips_hi:.3f}] / {r.fid:.1f}")
                rec.update({f"dl3dv{lv}_psnr": r.psnr, f"dl3dv{lv}_lpips": r.lpips, f"dl3dv{lv}_lpips_lo": r.lpips_lo,
                            f"dl3dv{lv}_lpips_hi": r.lpips_hi, f"dl3dv{lv}_fid": r.fid})
            else:
                cells.append("–")
        v = V.get(vk)
        cells.append(f"{v['psnr']:.2f} / {v['lpips']:.3f} / {v['fid']:.1f}" if v else "–")
        if v:
            rec.update(val_psnr=v["psnr"], val_lpips=v["lpips"], val_fid=v["fid"])
        t3, t6 = T[(T.bin == "~3m") & (T.method == tk)], T[(T.bin == "~6m") & (T.method == tk)]
        if len(t3):
            a, b = t3.iloc[0], t6.iloc[0]
            cells.append(f"{a.lpips:.3f} / {b.lpips:.3f}")
            cells.append(f"{a.lane_f1:.2f} [{a.lane_f1_clo:.2f}, {a.lane_f1_chi:.2f}] / {b.lane_f1:.2f}")
            rec.update(test_lpips_3m=a.lpips, test_lpips_6m=b.lpips, test_lane_f1_3m=a.lane_f1, test_lane_f1_3m_lo=a.lane_f1_clo,
                       test_lane_f1_3m_hi=a.lane_f1_chi, test_lane_f1_6m=b.lane_f1)
        else:
            cells += ["–", "–"]
        md.append(f"| {name} | " + " | ".join(cells) + " |"); rows.append(rec)
    P = pd.read_csv("results/dl3dv/paired.csv")
    md.append("\nPaired differences on DL3DV test images (a − b, 95% bootstrap CI over images):\n")
    md.append("| S | a vs b | Δ PSNR (dB) | Δ LPIPS |"); md.append("|---|---|---|---|")
    for _, r in P[P.b != "raw"].iterrows():
        md.append(f"| {r.level} | {r.a} vs {r.b} | {r.dpsnr:+.2f} [{r.dpsnr_lo:+.2f}, {r.dpsnr_hi:+.2f}] | "
                  f"{r.dlpips:+.3f} [{r.dlpips_lo:+.3f}, {r.dlpips_hi:+.3f}] |")
    pd.DataFrame(rows).to_csv(f"{OUT}/table2_difix_reimpl_validation.csv", index=False)
    P.to_csv(f"{OUT}/table2_dl3dv_paired.csv", index=False)
    return "\n".join(md)


def table3():
    E = pd.read_csv("results/euvs/summary.csv"); M = pd.read_csv("results/mv2/summary.csv")
    arms = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]
    md = ["| dataset / condition | method | PSNR ↑ [CI] | LPIPS ↓ [CI] | paint F1 ↑ [CI] | fake vehicles / img ↓ [CI] |",
          "|---|---|---|---|---|---|"]
    rows = []
    for label, T in [("EUVS L3 (5.5 m, 89°)", E[E.proto == "L3"]), ("EUVS rot60 (±60°)", E[E.proto == "rot60"]),
                     ("MV2 car→scooter (7.6 m, lower cam)", M[(M.dir == "car2scooty") & (M.kind == "cross")]),
                     ("MV2 scooter→car (higher cam)", M[(M.dir == "scooty2car") & (M.kind == "cross")])]:
        for m in arms:
            r = T[T.method == m]
            if not len(r):
                continue
            r = r.iloc[0].to_dict()
            md.append(f"| {label} | {NAME.get(m, m)} | {ci(r, 'psnr', 2, '_clo', '_chi')} | {ci(r, 'lpips', 3, '_clo', '_chi')} | "
                      f"{ci(r, 'mark_f1', 2, '_clo', '_chi')} | {ci(r, 'yolo_halluc_veh', 3, '_clo', '_chi')} |")
            rows.append(dict(condition=label, method=m, **{k: r.get(k) for k in
                        ["n", "psnr", "psnr_clo", "psnr_chi", "lpips", "lpips_clo", "lpips_chi", "mark_f1", "mark_f1_clo",
                         "mark_f1_chi", "yolo_halluc_veh", "yolo_halluc_veh_clo", "yolo_halluc_veh_chi"]}))
    pd.DataFrame(rows).to_csv(f"{OUT}/table3_cross_dataset.csv", index=False)
    return "\n".join(md)


def table4():
    runs = json.load(open("results/hugsim/closed_loop_runs.json"))
    R = pd.DataFrame(runs)
    names = {"none": "no fixer", "difix": "released Difix", "difix_ft": "Difix-FT", "nv_fixer": "NVIDIA Fixer",
             "ours": "our U-Net v0 (first-night)", "ours_best": "our U-Net (base_s0)",
             "difix_reimpl": "our Difix-ref reimpl. (on-path reference)"}

    def bci(x):
        rng = np.random.default_rng(0)  # fresh per interval (order-independent)
        x = np.asarray(x, float)
        if len(x) < 2:
            return np.nan, np.nan
        b = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    md = ["| fixer between simulator and policy | LTF seeds | runs | HD-Score ↑ [CI] | route compl. ↑ | no-collision ↑ | complete / collision / off-route | paired Δ HD-Score vs no fixer (same scenario+seed) [CI] | fixer ms / step |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    base = {(r["scen"], r["seed"]): r["hdscore"] for r in runs if r["fixer"] == "none"}
    for f in ["none", "difix", "difix_ft", "nv_fixer", "difix_reimpl", "ours_best", "ours"]:
        g = R[R.fixer == f]
        if not len(g):
            continue
        for seeds in ([sorted(g.seed.unique())] + ([[0]] if f != "none" and g.seed.nunique() > 1 else [])):
            h = g[g.seed.isin(seeds)]
            lo, hi = bci(h.hdscore)
            d = [r.hdscore - base[(r.scen, r.seed)] for r in h.itertuples() if (r.scen, r.seed) in base] if f != "none" else []
            dlo, dhi = bci(d) if d else (np.nan, np.nan)
            ends = f"{(h.end == 'Complete').sum()} / {h.end.str.contains('Collision').sum()} / {h.end.str.contains('Far').sum()}"
            ms = h.fix_ms_per_step.median() if "fix_ms_per_step" in h and f != "none" else np.nan
            md.append(f"| {names[f]} | {','.join(map(str, seeds))} | {len(h)} | {h.hdscore.mean():.3f} [{lo:.3f}, {hi:.3f}] | "
                      f"{h.rc.mean():.3f} | {h.nc.mean():.3f} | {ends} | "
                      + (f"{np.mean(d):+.3f} [{dlo:+.3f}, {dhi:+.3f}] (n={len(d)})" if d else "–") + " | "
                      + (f"{ms:.0f}" if not pd.isna(ms) else "–") + " |")
            rows.append(dict(fixer=f, seeds=",".join(map(str, seeds)), runs=len(h), hdscore=h.hdscore.mean(), hd_lo=lo, hd_hi=hi,
                             rc=h.rc.mean(), nc=h.nc.mean(), complete=int((h.end == "Complete").sum()),
                             collision=int(h.end.str.contains("Collision").sum()), off_route=int(h.end.str.contains("Far").sum()),
                             d_hd=np.mean(d) if d else np.nan, d_lo=dlo, d_hi=dhi, n_pairs=len(d), fix_ms_per_step=ms))
    pd.DataFrame(rows).to_csv(f"{OUT}/table4_closed_loop.csv", index=False)
    return "\n".join(md)


def main():
    os.makedirs(OUT, exist_ok=True); style()
    df = load_rows()
    df = df[df.method.isin(ARMS + ["difix_ref"])]
    roads = sorted(df[df.method == "raw"].scene.unique(), key=lambda s: ["scene_16", "scene_17", "scene_14", "scene_15", "scene_10",
                                                                       "scene_3", "scene_11", "scene_4"].index(s))
    full = df[df.method.isin(ARMS)]
    cover = full.groupby("method").scene.nunique()
    missing = [m for m in ARMS if cover.get(m, 0) < len(roads)]
    S = pooled(full); S.to_csv(f"{OUT}/pooled_all_metrics.csv", index=False)
    fig1(S, roads); fig2(df, roads); fig3()
    o = ["# Final consolidated results (generated by `src/eval/final_report.py`)\n",
         f"Para-Lane test roads pooled: {', '.join(roads)} ({len(roads)} held-out roads; never used for training or selection). "
         f"Images per offset bin (raw splat): " + ", ".join(f"{b}: {int(S[(S.bin == b) & (S.method == 'raw')].n_img.iloc[0])}" for b in LABELS)
         + f"; {int(S[S.method == 'raw'].n_seq.max())} lane-pair sequences. All CIs: 95% bootstrap over lane-pair sequences "
         "(clusters of correlated frames), 2,000 resamples.\n",
         (f"Arms missing on some roads: {missing}." if missing else "Every arm, released Difix-ref included, is pooled over all roads.")
         + "\n",
         "Figures: `fig1_trust_vs_offset.png`, `fig2_lane_f1_per_road_3m.png`, `fig3_heading_height.png`; qualitative grids "
         "`qual_*.jpg` (`src/viz/qualitative.py`).\n",
         "## Table 1. Hallucination per method at ~3 m and ~6 m (pooled over all test roads)\n",
         "Paint precision low = painted markings in the render that the real image does not have (invented); paint recall "
         "low = real markings missing (erased). Fake vehicles = YOLO11 vehicle/person detections in the output where the real "
         "frame has no movable object.\n", table1(S), "",
         "## Table 2. Our Difix training reimplementation vs. the official trainer vs. the released weights\n",
         "DL3DV-10K sample (the paper's data type; 3 held-out scenes, sparse-view renders every 30th / 60th frame) and "
         "driving (Para-Lane; val scene_19, test scene_16 + scene_17). Matched runs: same pairs, SD-Turbo init, 1,801 steps. "
         "DL3DV LPIPS CI: bootstrap over images; test lane-F1 CI: over lane-pair sequences.\n", table2(), "",
         "## Table 3. Cross-dataset transfer (EUVS / MARS, MV2); all fixers unchanged (no retraining)\n",
         "CIs: 95% bootstrap over sequences (EUVS: 9 test sequences per level; MV2: 5 sequences, so wide).\n", table3(), "",
         "## Table 4. HUGSIM closed loop (released LTF policy, 25 released nuScenes scenarios per seed)\n",
         "The fixer repairs the three front cameras the policy sees at every step; collisions and route checks use the "
         "simulator state. HD-Score CI: bootstrap over runs; paired Δ vs. no fixer over the same (scenario, seed).\n", table4(), ""]
    open(f"{OUT}/tables.md", "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
