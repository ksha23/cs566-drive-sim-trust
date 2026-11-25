"""Share of each real target frame covered by moving objects (Para-Lane foreground masks), for the test roads.
-> <out_csv> with scene, clip, ts, fg_frac (whole frame), fg_frac_lower (lower half, where the lanes are).
Used by src/eval/midterm_checks.py to recompute lane F1 on frames with little traffic. CPU only, no project imports.
usage: python src/eval/traffic_coverage.py $CS566D_DATA/pairs results/final/traffic_coverage.csv
"""
import csv, glob, os, sys
import numpy as np
from PIL import Image

SCENES = ["scene_3", "scene_4", "scene_10", "scene_11", "scene_14", "scene_15", "scene_16", "scene_17"]


def main(pairs, out):
    rows = []
    for s in SCENES:
        for f in sorted(glob.glob(f"{pairs}/{s}/fg/*/*.png")):
            m = np.asarray(Image.open(f)) > 127
            m = m if m.ndim == 2 else m[..., 0]
            rows.append(dict(scene=s, clip=os.path.basename(os.path.dirname(f)), ts=os.path.splitext(os.path.basename(f))[0],
                             fg_frac=f"{m.mean():.5f}", fg_frac_lower=f"{m[m.shape[0] // 2:].mean():.5f}"))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "frames ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
