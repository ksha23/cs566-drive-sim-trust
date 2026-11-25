"""Robustness checks quoted in the mid-term report, from the per-image rows that final_report.py pools.
-> results/final/midterm_checks.csv (check, bin, method, value, lo, hi, n, note)
  paint_drop       share of frames whose paint precision is at least 0.5 below the raw render's on the same frame
  vs_nv_fixer      paired lane F1 difference to NVIDIA Fixer on the same frames: pooled with a bootstrap over lane-pair
                   sequences (lo/hi), and the number of roads with a positive per-road mean (note)
  lane_f1_no_traffic  lane F1 on frames whose real image has no moving-object pixel in its lower half
                   (results/final/traffic_coverage.csv, written by traffic_coverage.py), next to lane F1 on all frames
CPU only. usage: scripts/lib/py.sh src/eval/midterm_checks.py
"""
import numpy as np
import pandas as pd

from eval.more_scenes_summary import load_rows
from eval.final_report import ARMS, OUT

K = ["scene", "src", "dst", "ts"]


def seq_boot(d, seq, n=10000, seed=0):
    # ratio bootstrap over lane-pair sequences, own generator so the interval does not depend on call order
    s = pd.DataFrame(dict(d=d, seq=seq)).groupby("seq").d.agg(["sum", "count"]).values
    rng = np.random.default_rng(seed); i = rng.integers(0, len(s), (n, len(s)))
    b = s[i, 0].sum(1) / s[i, 1].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    df = load_rows(); df = df[df.method.isin(ARMS)]
    recs = []
    for b in ["~3m", "~6m"]:
        d = df[df.bin == b]
        prec = d.pivot_table(index=K, columns="method", values="mark_prec")
        for m in ARMS[1:]:
            x = prec[[m, "raw"]].dropna(); v = ((x.raw - x[m]) >= 0.5)
            recs.append(dict(check="paint_drop", bin=b, method=m, value=v.mean(), n=len(x)))
        f1 = d.pivot_table(index=K, columns="method", values="lane_f1")
        for m in ["difix_ft", "difix_reimpl_ref"]:
            x = f1[[m, "nv_fixer"]].dropna().reset_index(); x["d"] = x[m] - x.nv_fixer
            lo, hi = seq_boot(x.d.values, (x.scene + ":" + x.src + ">" + x.dst).values)
            per = x.groupby("scene").d.mean()
            recs.append(dict(check="vs_nv_fixer", bin=b, method=m, value=x.d.mean(), lo=lo, hi=hi, n=len(x),
                             note=f"{int((per > 0).sum())}/{len(per)} roads > 0"))
    cov = pd.read_csv(f"{OUT}/traffic_coverage.csv", dtype={"ts": str}).rename(columns={"clip": "dst"})
    t = df.merge(cov, on=["scene", "dst", "ts"], how="left")
    assert t.fg_frac_lower.notna().all()
    for (b, m), g in t.groupby(["bin", "method"], observed=True):
        free = g[g.fg_frac_lower == 0]
        recs.append(dict(check="lane_f1_no_traffic", bin=b, method=m, value=free.lane_f1.mean(), n=len(free),
                         note=f"all frames {g.lane_f1.mean():.4f} (n={len(g)})"))
    out = pd.DataFrame(recs); out.to_csv(f"{OUT}/midterm_checks.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
