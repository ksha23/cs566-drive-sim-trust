"""Collect milestone-1/2 bookkeeping: on-path held-out quality per lane (+ cycle splats) and pair counts per scene."""
import json, glob, collections, os
from common.config import DATA
SPLIT = dict(scene_20="train", scene_21="train", scene_22="train", scene_23="train", scene_24="train",
             scene_19="val", scene_16="test", scene_17="test")
rows = []
for s in SPLIT:
    for c in ["clip_0", "clip_1", "clip_2"]:
        for kind, fn in [("full", "onpath_metrics.json"), ("cycle", "splat_cycle_onpath_metrics.json")]:
            p = f"{DATA}/splats/{s}/{c}/{fn}"
            if os.path.exists(p):
                d = json.load(open(p))
                rows.append(dict(scene=s, split=SPLIT[s], clip=c, kind=kind, **{k: d[k] for k in ["psnr", "ssim", "lpips", "n_gauss", "n_test", "n_train"]}))
os.makedirs("results/onpath", exist_ok=True); os.makedirs("results/pairs", exist_ok=True)
json.dump(rows, open("results/onpath/onpath_all.json", "w"), indent=1)
with open("results/onpath/ONPATH.md", "w") as f:
    f.write("| scene | split | lane | PSNR | SSIM | LPIPS | #Gaussians | cycle splat PSNR / LPIPS |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        if r["kind"] != "full":
            continue
        cy = [c for c in rows if c["kind"] == "cycle" and c["scene"] == r["scene"] and c["clip"] == r["clip"]]
        cys = "%.2f / %.3f" % (cy[0]["psnr"], cy[0]["lpips"]) if cy else "–"
        f.write("| %s | %s | %s | %.2f | %.3f | %.3f | %.2fM | %s |\n" % (r["scene"], r["split"], r["clip"], r["psnr"], r["ssim"], r["lpips"], r["n_gauss"] / 1e6, cys))
cnt = {}
for s in SPLIT:
    m = json.load(open(f"{DATA}/pairs/{s}/meta.json"))
    c = collections.Counter(x["variant"] + ("_onpath_test" if x["onpath_test"] else "") for x in m)
    cnt[s] = dict(split=SPLIT[s], **c)
json.dump(cnt, open("results/pairs/pair_counts.json", "w"), indent=1)
print(open("results/onpath/ONPATH.md").read()); print(json.dumps(cnt, indent=0))
