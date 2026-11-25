"""Summarise per-image eval rows (evaluate.py output, optionally merged with NTL-IoU rows) into a table of
mean and 95% bootstrap CI per (offset bin, method). Two bootstraps: over frames (default), and over
(scene, src->dst) sequences ("cluster" CI, more honest because neighboring frames are correlated).

usage: scripts/lib/py.sh src/eval/summarize.py <out_prefix> <eval_rows.json> [more.json ...]
writes <out_prefix>.csv (long format) and <out_prefix>.md (compact table)
"""
import json, sys
import numpy as np
import pandas as pd

METRICS = ["psnr", "ssim", "lpips", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "tex_ratio",
           "ntl_iou", "lane_prec", "lane_rec", "lane_f1", "yolo_static_f1", "yolo_halluc_veh"]
BINS = [-0.1, 1.0, 4.5, 20]; LABELS = ["0m", "~3m", "~6m"]


def boot(v, n=2000, groups=None, seed=0):
    # a fresh generator per call, so an interval does not depend on how many intervals were computed before it
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.nan, np.nan
    if groups is None:
        x = v[ok]; idx = rng.integers(0, len(x), (n, len(x)))
        m = np.nanmean(x[idx], 1)
    else:
        g = np.asarray(groups)[ok]; x = v[ok]; ug = np.unique(g)
        if len(ug) < 2:
            return np.nan, np.nan
        sums = np.array([x[g == u].sum() for u in ug]); cnt = np.array([(g == u).sum() for u in ug])
        idx = rng.integers(0, len(ug), (n, len(ug)))
        m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out = sys.argv[1]
    rows = []
    for p in sys.argv[2:]:
        rows += json.load(open(p))
    df = pd.DataFrame(rows)
    key = ["method", "scene", "src", "dst", "ts"]
    df["scene"] = df["scene"].fillna("scene_16") if "scene" in df else "scene_16"
    # merge rows of the same image from several files (e.g. evaluate.py + ntl_iou.py)
    df = df.groupby(key, as_index=False).first()
    df["abs_off"] = df.offset_m.abs()
    df["bin"] = pd.cut(df.abs_off, BINS, labels=LABELS)
    df["seq"] = df.scene + ":" + df.src + ">" + df.dst
    recs = []
    for (b, m), g in df.groupby(["bin", "method"], observed=True):
        r = dict(bin=b, method=m, n=len(g), scenes=",".join(sorted(g.scene.unique())))
        for k in METRICS:
            if k not in g or g[k].isna().all():
                continue
            r[k] = float(np.nanmean(g[k]))
            r[k + "_lo"], r[k + "_hi"] = boot(g[k].values)
            r[k + "_clo"], r[k + "_chi"] = boot(g[k].values, groups=g.seq.values)
        recs.append(r)
    S = pd.DataFrame(recs); S.to_csv(out + ".csv", index=False)
    with open(out + ".md", "w") as f:
        cols = [k for k in METRICS if k in S]
        f.write("| offset | method | n | " + " | ".join(cols) + " |\n|" + "---|" * (3 + len(cols)) + "\n")
        for _, r in S.iterrows():
            cells = [f"{r[k]:.3f} [{r[k+'_lo']:.3f},{r[k+'_hi']:.3f}]" if not pd.isna(r.get(k)) else "–" for k in cols]
            f.write(f"| {r['bin']} | {r['method']} | {r['n']} | " + " | ".join(cells) + " |\n")
    print(open(out + ".md").read())


if __name__ == "__main__":
    main()
