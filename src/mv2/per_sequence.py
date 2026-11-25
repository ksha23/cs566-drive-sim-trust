"""MV2 per-sequence breakdown of the cross-vehicle rows (results/mv2/rows_<dir>.json): the car/scooter lateral
separation differs a lot between sequences (3.9-8.8 m) while the camera-height difference is ~0.4 m in all of them,
so per-sequence means show which of the two drives the loss. -> results/mv2/per_sequence.{csv,md}
usage: scripts/lib/py.sh src/mv2/per_sequence.py"""
import json
import numpy as np
import pandas as pd

R = "results/mv2"
rows = []
for d in ("car2scooty", "scooty2car"):
    rows += [dict(r, dir=d) for r in json.load(open(f"{R}/rows_{d}.json"))]
df = pd.DataFrame(rows); df = df[df.kind == "cross"]
g = df.groupby(["dir", "scene", "method"]).agg(n=("psnr", "size"), lat=("lat_m", lambda x: np.median(np.abs(x))),
                                               dh=("dh_m", "median"), psnr=("psnr", "mean"), lpips=("lpips", "mean"),
                                               mark_f1=("mark_f1", "mean"), lane_f1=("lane_f1", "mean"),
                                               fake=("yolo_halluc_veh", "mean")).reset_index()
g.to_csv(f"{R}/per_sequence.csv", index=False)
o = ["# MV2 per sequence (cross-vehicle frames) — `src/mv2/per_sequence.py`\n",
     "| dir | seq | median lateral (m) | camera-height diff (m) | raw LPIPS | best fixer LPIPS (method) | raw PSNR | raw paint F1 |",
     "|---|---|---|---|---|---|---|---|"]
for (d, s), h in g.groupby(["dir", "scene"]):
    raw = h[h.method == "raw"].iloc[0]; fx = h[h.method != "raw"].sort_values("lpips").iloc[0]
    o.append(f"| {d} | {s} | {raw.lat:.1f} | {raw.dh:+.2f} | {raw.lpips:.3f} | {fx.lpips:.3f} ({fx.method}) | {raw.psnr:.2f} | {raw.mark_f1:.2f} |")
cor = {}
for d in ("car2scooty", "scooty2car"):
    h = g[(g.dir == d) & (g.method == "raw")]
    cor[d] = (float(np.corrcoef(h.lat, h.lpips)[0, 1]), float(np.corrcoef(h.lat, h.psnr)[0, 1]))
o.append("")
o.append("Correlation over the 5 sequences of the raw splat's mean LPIPS / PSNR with the lateral separation: " +
         "; ".join(f"{d}: r = {c[0]:+.2f} / {c[1]:+.2f}" for d, c in cor.items()) + " (5 points each; indicative only).")
open(f"{R}/per_sequence.md", "w").write("\n".join(o) + "\n"); print("\n".join(o))
