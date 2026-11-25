"""Lane-paint pixel counts of every output in the lateral evaluation, so that paint precision can be left undefined
when an output has no paint at all. mark_agree() (evaluate.py) scores such an output precision 0, which mixes "invented
paint" with "no paint"; load_rows() (more_scenes_summary.py) sets mark_prec to NaN where n_paint_out < MIN_PAINT
(recall and F1 are unchanged: an output without paint has erased all real paint). CPU only; reads the same rows as
load_rows() and the images behind them.
-> results/final/paint_counts.csv (method, scene, src, dst, ts, n_paint_out)
usage: scripts/lib/py.sh src/eval/paint_counts.py [--jobs 8]
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image

from common.config import DATA
from eval.evaluate import marking_mask
from eval.more_scenes_summary import load_rows, KEY
from fixers.apply import eval_items

OUT = "results/final/paint_counts.csv"
MIN_PAINT = 50  # the threshold mark_agree() already uses for "no paint in either image"


def count(job):
    out_path, fg_path = job
    if not os.path.exists(out_path):
        return np.nan
    x = np.array(Image.open(out_path).convert("RGB")); fg = np.array(Image.open(fg_path)) > 127
    if x.shape[:2] != fg.shape:
        x = np.array(Image.fromarray(x).resize(fg.shape[::-1], Image.BICUBIC))
    return int(marking_mask(x, fg).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    df = load_rows(paint_nan=False)[KEY]
    P = f"{DATA}/pairs"; meta = {}
    for s in df.scene.unique():
        for m in eval_items(P, s, 2):
            meta[(s, m["src"], m["dst"], str(m["ts"]))] = m
    jobs, keep = [], []
    for r in df.itertuples(index=False):
        m = meta.get((r.scene, r.src, r.dst, str(r.ts)))
        if m is None:
            continue
        p = f"{P}/{m['path']}" if r.method == "raw" else f"{DATA}/fixed/{r.method}/{m['path']}"
        jobs.append((p, f"{P}/{m['fg']}")); keep.append(r)
    with Pool(a.jobs) as pool:
        n = pool.map(count, jobs, chunksize=32)
    out = pd.DataFrame(keep, columns=KEY); out["n_paint_out"] = n
    out.to_csv(OUT, index=False)
    print(f"{len(out)} of {len(df)} rows; missing images {int(out.n_paint_out.isna().sum())}; "
          f"outputs with < {MIN_PAINT} paint px: {(out.n_paint_out < MIN_PAINT).mean():.3f}")


if __name__ == "__main__":
    main()
