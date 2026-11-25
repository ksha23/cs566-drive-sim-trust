"""FaithFusion's EIG-weighted progressive update against the same update with a flat weight, over every road where both
arms ran (results/faithfusion/{eval,ntl}_<scene>_ff.json, written by scripts/14_faithfusion.sh and stage 22 'ff').
Per road: mean paired difference EIG - flat on the same images, with a bootstrap over that road's lane-pair sequences;
over roads: mean of the per-road differences, bootstrap over roads. CPU only.
-> results/faithfusion/eig_vs_flat_roads.{csv,md}
usage: scripts/lib/py.sh src/progressive/faithfusion/summary.py
"""
import glob, json
import numpy as np
import pandas as pd

from eval.summarize import BINS, LABELS, boot

R = "results/faithfusion"
KEY = ["scene", "src", "dst", "ts"]
METRICS = [("lane_f1", "lane F1"), ("lpips", "LPIPS"), ("mark_f1", "paint F1"), ("psnr", "PSNR"), ("yolo_halluc_veh", "spurious vehicles")]


def rows():
    out = []
    for f in sorted(glob.glob(f"{R}/eval_scene_*_ff.json")) + sorted(glob.glob(f"{R}/ntl_scene_*_ff.json")):
        out += json.load(open(f))
    df = pd.DataFrame(out); df["ts"] = df.ts.astype(str)
    df = df.groupby(["method"] + KEY, as_index=False).first()
    df["bin"] = pd.cut(df.offset_m.abs(), BINS, labels=LABELS)
    return df


def main():
    df = rows()
    recs = []
    for k, _ in METRICS:
        if k not in df:
            continue
        p = df.pivot_table(index=KEY + ["bin"], columns="method", values=k, observed=True).dropna(subset=["prog_ff_flat", "prog_ff_eig"])
        p = p.reset_index(); p["d"] = p.prog_ff_eig - p.prog_ff_flat; p["seq"] = p.scene + ":" + p.src + ">" + p.dst
        for b, g in p.groupby("bin", observed=True):
            per = []
            for s, h in g.groupby("scene"):
                lo, hi = boot(h.d.values, groups=h.seq.values)
                per.append(h.d.mean())
                recs.append(dict(metric=k, bin=b, scene=s, n=len(h), flat=h.prog_ff_flat.mean(), eig=h.prog_ff_eig.mean(),
                                 delta=h.d.mean(), lo=lo, hi=hi))
            lo, hi = boot(np.array(per)) if len(per) > 1 else (np.nan, np.nan)
            recs.append(dict(metric=k, bin=b, scene="over roads", n=len(per), flat=g.prog_ff_flat.mean(), eig=g.prog_ff_eig.mean(),
                             delta=float(np.mean(per)), lo=lo, hi=hi, roads_pos=int(np.sum(np.array(per) > 0))))
    out = pd.DataFrame(recs); out.to_csv(f"{R}/eig_vs_flat_roads.csv", index=False, float_format="%.4f")
    md = ["# FaithFusion EIG weighting vs. flat weighting of the progressive update (paired, same images)", "",
          "| metric | offset | road | images / roads | flat | EIG | EIG - flat [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        md.append(f"| {r.metric} | {r.bin} | {r.scene} | {r.n} | {r.flat:.3f} | {r.eig:.3f} | {r.delta:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] |")
    open(f"{R}/eig_vs_flat_roads.md", "w").write("\n".join(md) + "\n")
    print("\n".join(l for l in md if "over roads" in l or l.startswith("| metric")))


if __name__ == "__main__":
    main()
