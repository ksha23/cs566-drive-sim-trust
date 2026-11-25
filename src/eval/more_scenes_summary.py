"""Stage 18 summary: all Para-Lane TEST scenes (scene_16, scene_17 + the new ones of docs/more_scenes.md).
Writes to results/eval_more/:
  summary_pooled.{csv,md}      mean + 95% bootstrap CIs (frames; and clusters = lane-pair sequences) per (bin, method), all scenes
  summary_<scene>.{csv,md}     the same per scene
  deltas.csv                   per scene, bin and arm (prog_difix_reimpl, road_prior, every fixer): paired per-image
                               difference arm - raw (same target image), mean and cluster-bootstrap CI over sequences
  deltas_across_scenes.csv     mean over scenes of the per-scene deltas with a bootstrap over SCENES (+ #scenes > 0)
  tables.md                    compact markdown for RESULTS.md
Row sources: scene_16/17 from results/eval, results/lanes, results/difix_reimpl (the runs of stages 6/7d/8 and the
reimplementation eval), scene_17 road prior and all new scenes from results/eval_more.
usage: scripts/lib/py.sh src/eval/more_scenes_summary.py
"""
import glob, json, os
import numpy as np
import pandas as pd
from eval.summarize import METRICS, BINS, LABELS, boot

R = "results/eval_more"
ARMS = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0", "prog_difix_reimpl", "road_prior"]
NAMES = {"raw": "raw splat", "difix": "Difix [1] zero-shot", "difix_ref": "Difix-ref [1]", "nv_fixer": "NVIDIA Fixer [4]",
         "difix_ft": "Difix-FT", "difix_reimpl_ref": "our Difix-ref reimpl.", "ours_hfgan_s0": "our U-Net (hfgan_s0)",
         "prog_difix_reimpl": "progressive w/ our reimpl. (splat only)", "road_prior": "road-prior splat (no fixer)"}
KEY = ["method", "scene", "src", "dst", "ts"]
DM = ["lpips", "mark_f1", "ntl_iou", "lane_f1", "yolo_halluc_veh", "psnr"]


def load_rows(paint_nan=True):
    """paint_nan: paint precision is undefined (NaN) for outputs without lane paint (results/final/paint_counts.csv,
    written by paint_counts.py; skipped if that file does not exist yet)."""
    files = []
    for s in ("scene_16", "scene_17"):
        files += [f"results/eval/eval_{s}_{k}.json" for k in ("a", "gan", "nvfixer", "progdx", "road")]
        files += [f"results/lanes/ntl_{s}_{k}.json" for k in ("a", "gan", "nvfixer", "progdx", "road")]
        files += [f"results/difix_reimpl/eval_{s}_difix_reimpl_ref.json", f"results/difix_reimpl/ntl_{s}_difix_reimpl_ref.json"]
    files += sorted(glob.glob(f"{R}/eval_scene_*.json")) + sorted(glob.glob(f"{R}/ntl_scene_*.json"))
    rows = []
    for f in files:
        if os.path.exists(f):
            rows += [r for r in json.load(open(f)) if r["method"] in ARMS]
    df = pd.DataFrame(rows).groupby(KEY, as_index=False).first()
    pc = "results/final/paint_counts.csv"
    if paint_nan and os.path.exists(pc):
        c = pd.read_csv(pc, dtype={"ts": str})
        df = df.merge(c, on=KEY, how="left")
        df.loc[df.n_paint_out < 50, "mark_prec"] = np.nan  # no paint in the output: precision undefined (paint_counts.py)
    df["bin"] = pd.cut(df.offset_m.abs(), BINS, labels=LABELS)
    df["seq"] = df.scene + ":" + df.src + ">" + df.dst
    return df


def summary(df, out):
    recs = []
    for (b, m), g in df.groupby(["bin", "method"], observed=True):
        r = dict(bin=b, method=m, n=len(g), n_seq=g.seq.nunique(), scenes=",".join(sorted(g.scene.unique())))
        for k in METRICS:
            if k in g and not g[k].isna().all():
                r[k] = float(np.nanmean(g[k])); r[k + "_lo"], r[k + "_hi"] = boot(g[k].values)
                r[k + "_clo"], r[k + "_chi"] = boot(g[k].values, groups=g.seq.values)
        recs.append(r)
    S = pd.DataFrame(recs); S.to_csv(out + ".csv", index=False)
    return S


def md_table(S, title):
    cols = [("psnr", "PSNR↑", 2), ("lpips", "LPIPS↓", 3), ("mark_f1", "paint F1↑", 2), ("ntl_iou", "NTL-IoU↑", 2),
            ("lane_f1", "lane F1↑", 2), ("yolo_halluc_veh", "fake veh.↓", 3)]
    o = [f"#### {title}\n", "| offset | method | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for b in LABELS:
        for m in ARMS:
            r = S[(S.bin == b) & (S.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]; cells = []
            for k, _, d in cols:
                v = r.get(k)
                if v is None or pd.isna(v):
                    cells.append("–"); continue
                s = f"{v:.{d}f}"
                if k in ("lpips", "lane_f1", "ntl_iou") and not pd.isna(r.get(k + "_clo")):
                    s += f" [{r[k + '_clo']:.{d}f}, {r[k + '_chi']:.{d}f}]"
                cells.append(s)
            o.append(f"| {b} | {NAMES[m]} | {int(r.n)} | " + " | ".join(cells) + " |")
    return "\n".join(o) + "\n"


def deltas(df):
    base = df[df.method == "raw"].set_index(["scene", "src", "dst", "ts"])
    recs = []
    for m in ARMS[1:]:
        g = df[df.method == m].set_index(["scene", "src", "dst", "ts"])
        j = g.join(base, rsuffix="_raw", how="inner")
        for (s, b), h in j.groupby(["scene", "bin"], observed=True):
            r = dict(method=m, scene=s, bin=b, n=len(h), n_seq=h.seq.nunique())
            for k in DM:
                if k in h and (k + "_raw") in h:
                    d = (h[k] - h[k + "_raw"]).values
                    if np.isnan(d).all():
                        continue
                    r[k] = float(np.nanmean(d)); r[k + "_clo"], r[k + "_chi"] = boot(d, groups=h.seq.values)
            recs.append(r)
    D = pd.DataFrame(recs); D.to_csv(f"{R}/deltas.csv", index=False)
    rng = np.random.default_rng(0); A = []
    for (m, b), h in D.groupby(["method", "bin"], observed=True):
        r = dict(method=m, bin=b, n_scenes=len(h), scenes=",".join(h.scene))
        for k in DM:
            if k not in h:
                continue
            v = h[k].dropna().values
            if len(v) < 2:
                continue
            idx = rng.integers(0, len(v), (4000, len(v))); bm = v[idx].mean(1)
            r[k] = float(v.mean()); r[k + "_lo"], r[k + "_hi"] = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
            r[k + "_min"], r[k + "_max"] = float(v.min()), float(v.max())
            r[k + "_npos"] = int((v > 0).sum())
        A.append(r)
    A = pd.DataFrame(A); A.to_csv(f"{R}/deltas_across_scenes.csv", index=False)
    return D, A


def main():
    df = load_rows()
    scenes = sorted(df.scene.unique(), key=lambda s: int(s.split("_")[1]))
    S = summary(df, f"{R}/summary_pooled")
    per = {s: summary(df[df.scene == s], f"{R}/summary_{s}") for s in scenes}
    D, A = deltas(df)
    o = ["# More Para-Lane test scenes (stage 18) — tables\n",
         f"Generated by `src/eval/more_scenes_summary.py` from the row files listed in its docstring. Scenes: {', '.join(scenes)}. "
         "CIs: 95% bootstrap over lane-pair sequences (clusters).\n"]
    cov = df.groupby(["method"]).scene.apply(lambda x: ",".join(sorted(x.unique(), key=lambda s: int(s.split('_')[1]))))
    o.append("Scenes covered per arm: " + "; ".join(f"{NAMES[m]}: {cov[m]}" for m in ARMS if m in cov) + "\n")
    o.append(md_table(S, f"Pooled over all test scenes"))
    # per-scene compact: lane F1 and LPIPS at ~3 m and ~6 m for every arm
    for b in ("~3m", "~6m"):
        for k, d in (("lane_f1", 2), ("ntl_iou", 2), ("lpips", 3), ("mark_f1", 2), ("yolo_halluc_veh", 3)):
            o.append(f"#### Per scene, {k} at {b} (mean over frames)\n")
            o.append("| method | " + " | ".join(scenes) + " |"); o.append("|---|" + "---|" * len(scenes))
            for m in ARMS:
                cells = []
                for s in scenes:
                    r = per[s][(per[s].bin == b) & (per[s].method == m)]
                    cells.append(f"{r.iloc[0][k]:.{d}f}" if len(r) and k in r and not pd.isna(r.iloc[0][k]) else "–")
                o.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
            o.append("")
    o.append("#### Paired change vs. the raw splat, per scene (arm − raw on the same target images; [95% CI over sequences])\n")
    o.append("| arm | offset | scene | n | Δ lane F1 | Δ NTL-IoU | Δ LPIPS | Δ paint F1 |"); o.append("|---|---|---|---|---|---|---|---|")
    for m in ("prog_difix_reimpl", "road_prior", "difix_reimpl_ref", "difix_ft", "nv_fixer"):
        for b in LABELS:
            for s in scenes:
                r = D[(D.method == m) & (D.bin == b) & (D.scene == s)]
                if not len(r):
                    continue
                r = r.iloc[0]
                f = lambda k, d: (f"{r[k]:+.{d}f} [{r[k+'_clo']:+.{d}f}, {r[k+'_chi']:+.{d}f}]" if k in r and not pd.isna(r[k]) and not pd.isna(r.get(k + '_clo')) else (f"{r[k]:+.{d}f}" if k in r and not pd.isna(r[k]) else "–"))
                o.append(f"| {NAMES[m]} | {b} | {s} | {int(r.n)} | {f('lane_f1', 2)} | {f('ntl_iou', 2)} | {f('lpips', 3)} | {f('mark_f1', 2)} |")
    o.append("\n#### Across scenes: mean of the per-scene deltas, [95% bootstrap over scenes], (min … max), #scenes with Δ > 0\n")
    o.append("| arm | offset | scenes | Δ lane F1 | Δ NTL-IoU | Δ LPIPS | Δ paint F1 | Δ fake veh. |"); o.append("|---|---|---|---|---|---|---|---|")
    for m in ARMS[1:]:
        for b in LABELS:
            r = A[(A.method == m) & (A.bin == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            def f(k, d):
                if k not in r or pd.isna(r[k]):
                    return "–"
                return f"{r[k]:+.{d}f} [{r[k+'_lo']:+.{d}f}, {r[k+'_hi']:+.{d}f}] ({r[k+'_min']:+.{d}f} … {r[k+'_max']:+.{d}f}) {int(r[k+'_npos'])}/{int(r.n_scenes)}"
            o.append(f"| {NAMES[m]} | {b} | {int(r.n_scenes)} | {f('lane_f1', 2)} | {f('ntl_iou', 2)} | {f('lpips', 3)} | {f('mark_f1', 2)} | {f('yolo_halluc_veh', 3)} |")
    open(f"{R}/tables.md", "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
