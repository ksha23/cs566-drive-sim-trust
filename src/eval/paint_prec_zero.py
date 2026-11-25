"""How often is paint precision exactly 0, and does excluding those frames change the comparison with the raw render?
-> results/final/paint_prec_zero.csv
mark_agree() (evaluate.py) gives precision 0 when no rendered paint pixel lies within 3 px of real paint, which includes
renders with no detected paint at all. CPU only; reads the per-image rows that final_report.py pools.
usage: scripts/lib/py.sh src/eval/paint_prec_zero.py
"""
import numpy as np
import pandas as pd

from eval.more_scenes_summary import load_rows
from eval.final_report import ARMS, OUT


def main():
    df = load_rows(); df = df[df.method.isin(ARMS)]
    recs = []
    for (b, m), g in df.groupby(["bin", "method"], observed=True):
        p = g.mark_prec
        recs.append(dict(bin=b, method=m, n_img=len(g), n_prec_defined=int(p.notna().sum()),
                         frac_prec0_of_all=float((p == 0).mean()), frac_prec0_of_defined=float((p == 0).sum() / p.notna().sum()),
                         mark_prec=float(np.nanmean(p)), mark_prec_excl0=float(np.nanmean(p[p > 0]))))
    out = pd.DataFrame(recs); out.to_csv(f"{OUT}/paint_prec_zero.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
