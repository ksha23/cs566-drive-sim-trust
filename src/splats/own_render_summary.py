"""Markdown summary of the own-renderer refit comparison (scripts/21_own_renderer.sh eval): per offset bin, the
existing evaluation metrics for the original gsplat-fitted splat (raw), our-renderer refit, a gsplat refit with
another seed (run-to-run noise) and the original splat rendered by our renderer; paired differences vs raw with a
95% bootstrap CI over images; fit time / size / on-path PSNR of each fit.
usage: scripts/lib/py.sh src/splats/own_render_summary.py <scene>  (reads results/own_renderer/, $DATA/own_renderer/)
       scripts/lib/py.sh src/splats/own_render_summary.py --fitlogs <out.json> <fit log>...  (loss / #Gaussians per
       1k iterations, train time, peak memory, on-path metrics of fit_splat.py logs -> json)"""
import json, os, re, sys
import numpy as np
import pandas as pd

from common.config import DATA

if sys.argv[1] == "--fitlogs":
    out = {}
    for f in sys.argv[3:]:
        L = open(f).read()
        its = [dict(it=int(m[0]), loss=float(m[1]), l1=float(m[2]), n=int(m[3]), t=int(m[4]))
               for m in re.findall(r"^it (\d+) loss ([\d.]+) l1 ([\d.]+) depth [\d.]+ N (\d+) t (\d+)s", L, re.M)]
        tr = re.search(r"TRAIN done (\d+)s peak ([\d.]+) GiB N=(\d+)", L)
        on = re.search(r"ON-PATH held-out: (\{.*\})", L)
        out[os.path.basename(f)] = dict(iters=its, train_s=int(tr[1]), peak_gib=float(tr[2]), n_gauss=int(tr[3]),
                                        onpath=json.loads(on[1]))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    sys.exit()
S = sys.argv[1]
R = "results/own_renderer"
ev = pd.DataFrame(json.load(open(f"{R}/eval_{S}.json")))
ln = pd.DataFrame(json.load(open(f"{R}/ntl_{S}.json")))
df = ev.merge(ln[["method", "src", "dst", "ts", "ntl_iou", "lane_f1"]], on=["method", "src", "dst", "ts"], how="left")
df["bin"] = pd.cut(df.offset_m.abs(), [-0.1, 1.0, 4.5, 20], labels=["0 m", "~3 m", "~6 m"])
cols = ["psnr", "ssim", "lpips", "road_lpips", "mark_f1", "ntl_iou", "lane_f1", "yolo_halluc_veh"]
names = dict(raw="gsplat fit (existing, seed 0), gsplat render", own_refit="**our renderer: fit + render** (seed 0)",
             gsplat_refit_s1="gsplat refit, seed 1 (noise ref.)", own_render_orig="existing gsplat fit, rendered by ours")
print(f"### {S} / clip_0 splat: existing evaluation on the renders (eval subset, src = clip_0)\n")
print("| offset | splat / renderer | n | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 3) + "|")
for b in ["0 m", "~3 m", "~6 m"]:
    for m in names:
        d = df[(df.bin == b) & (df.method == m)]
        print(f"| {b} | {names[m]} | {len(d)} | " + " | ".join(f"{d[c].mean():.3f}" for c in cols) + " |")

rng = np.random.default_rng(0)
print("\nPaired difference vs the existing splat (raw), mean [95% bootstrap CI over images]:\n")
print("| offset | method | ΔPSNR | ΔLPIPS | Δpaint F1 | Δlane F1 |")
print("|---|---|---|---|---|---|")
for b in ["0 m", "~3 m", "~6 m"]:
    base = df[(df.bin == b) & (df.method == "raw")].set_index(["dst", "ts"])
    for m in ["own_refit", "gsplat_refit_s1", "own_render_orig"]:
        d = df[(df.bin == b) & (df.method == m)].set_index(["dst", "ts"])
        cells = []
        for c in ["psnr", "lpips", "mark_f1", "lane_f1"]:
            x = (d[c] - base[c]).dropna().values
            bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            cells.append(f"{x.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
        print(f"| {b} | {m} | " + " | ".join(cells) + " |")

print("\nFits (12k iterations, same settings; RTX 3090):\n")
print("| fit | renderer | seed | train time (s) | Gaussians | on-path held-out PSNR / SSIM / LPIPS |")
print("|---|---|---|---|---|---|")
for lab, p, rn, sd in [("existing", f"{DATA}/splats/{S}/clip_0/onpath_metrics.json", "gsplat", 0),
                       ("refit", f"{DATA}/own_renderer/splats/{S}/clip_0/onpath_metrics.json", "ours", 0),
                       ("refit", f"{DATA}/own_renderer/splats_gsplat_s1/{S}/clip_0/onpath_metrics.json", "gsplat", 1)]:
    if os.path.exists(p):
        j = json.load(open(p))
        print(f"| {lab} | {rn} | {sd} | {j['train_s']:.0f} | {j['n_gauss']:,} | {j['psnr']:.2f} / {j['ssim']:.3f} / {j['lpips']:.3f} |")
