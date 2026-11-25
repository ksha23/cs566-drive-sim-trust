"""MV2 [9] evaluation of every fixer against the REAL images of the other vehicle (car <-> scooter; lateral offset and
camera height at once) and of the training vehicle's held-out frames (0 m). Same metrics and conventions as the
Para-Lane / EUVS evaluations (src/eval/evaluate.py, src/euvs/evaluate.py): PSNR / SSIM / LPIPS with movable-object and
ego-vehicle pixels pasted from the real image, road LPIPS, lane-paint P/R/F1, road-texture ratio, YOLO fake vehicles
and static-object F1, CLRNet NTL-IoU / lane F1 vs. CLRNet on the real view (crop band at the per-view horizon).
All fixers are trained on Para-Lane or generic: this is a TRANSFER test (India, GoPro).

usage:
  scripts/lib/py.sh src/mv2/evaluate.py eval <dir> --methods raw,difix,...   -> results/mv2/rows_<dir>.json
  scripts/lib/py.sh src/mv2/evaluate.py summary                              -> results/mv2/summary.csv / .md
"""
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image

from common.config import DATA, REPO, YOLO_WEIGHTS
from eval.metrics import Metrics
from eval.evaluate import marking_mask, mark_agree, tex_energy, det_f1, STATIC, MOVABLE, ROAD_Y0
from eval.lanes.ntl_iou import compare

ROOT = f"{DATA}/mv2/out"; FIXED = f"{DATA}/mv2/fixed"; OUT = f"{REPO}/results/mv2"
METHODS = "raw,difix,difix_ref,nv_fixer,difix_ft,difix_reimpl_ref,ours_hfgan_s0"


def evaluate(d, methods):
    from ultralytics import YOLO
    meta = json.load(open(f"{ROOT}/{d}/meta_all.json"))
    yolo = YOLO(YOLO_WEIGHTS); M = Metrics("cuda")
    det = lambda im: (lambda r: (r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int)))(
        yolo(np.ascontiguousarray(im[..., ::-1]), verbose=False, conf=0.3, device=0)[0].boxes)
    lanes = {m: json.load(open(f"{ROOT}/{d}/lanes_{m}.json")) for m in methods + ["real"]
             if os.path.exists(f"{ROOT}/{d}/lanes_{m}.json")}
    rows = []
    for n, it in enumerate(meta):
        real = np.array(Image.open(f"{ROOT}/{it['real']}").convert("RGB")); fg = np.array(Image.open(f"{ROOT}/{it['fg']}")) > 127
        H, W = real.shape[:2]; y0 = int(ROAD_Y0 * H)
        G = torch.from_numpy(real).float().cuda() / 255; FG = torch.from_numpy(fg).cuda()
        gm = marking_mask(real, fg); gtex = tex_energy(real, fg); gb, gc = det(real); gs = np.isin(gc, list(STATIC))
        for meth in methods:
            p = f"{ROOT}/{it['path']}" if meth == "raw" else f"{FIXED}/{meth}/{it['path']}"
            if not os.path.exists(p):
                continue
            x = np.array(Image.open(p).convert("RGB"))
            if x.shape != real.shape:
                x = np.array(Image.fromarray(x).resize((W, H), Image.BICUBIC))
            X = torch.from_numpy(x).float().cuda() / 255
            r = M(X, G, FG); r["road_lpips"] = M(X[y0:], G[y0:], FG[y0:])["lpips"]
            f1, pr, rc = mark_agree(marking_mask(x, fg), gm)
            b, c = det(x); s = np.isin(c, list(STATIC)); mv = np.isin(c, list(MOVABLE)); hall = 0
            for x1, y1, x2, y2 in b[mv].astype(int):
                patch = fg[max(y1, 0):y2, max(x1, 0):x2]
                if patch.size == 0 or patch.mean() < 0.3:
                    hall += 1
            r.update(mark_f1=f1, mark_prec=pr, mark_rec=rc, tex_ratio=float(tex_energy(x, fg) / (gtex + 1e-9)),
                     yolo_static_f1=det_f1(gb[gs], gc[gs], b[s], c[s]), yolo_halluc_veh=hall, real_lanes=np.nan,
                     method=meth, dir=d, kind=it["kind"], scene=it["scene"], name=it["name"], frame=it["frame"],
                     dist_m=it["dist_m"], rot_deg=it["rot_deg"], lat_m=it["lat_m"], dz_m=it["dz_m"], dh_m=it["dh_m"])
            if "real" in lanes and it["real"] in lanes["real"]:
                g = lanes["real"][it["real"]]; r["real_lanes"] = len(g["lanes"])
                if meth in lanes and it["path"] in lanes[meth] and len(g["lanes"]):
                    r.update(compare(lanes[meth][it["path"]]["lanes"], g["lanes"], (int(min(g["band"][1], H)), W), 30 * W / 1640))
            rows.append(r)
        if n % 50 == 0:
            print(d, n, "/", len(meta), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/rows_{d}.json", "w"), default=float)


KS = ["psnr", "ssim", "lpips", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "ntl_iou", "lane_f1", "yolo_halluc_veh",
      "tex_ratio", "yolo_static_f1"]
ORDER = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]


def summary():
    import pandas as pd
    from eval.summarize import boot
    df = pd.DataFrame(sum([json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/rows_*.json"))], []))
    recs = []
    for (d, kind, meth), g in df.groupby(["dir", "kind", "method"]):
        r = dict(dir=d, kind=kind, method=meth, n=len(g), n_seq=g.scene.nunique(), lat_abs_m=g.lat_m.abs().median(),
                 dz_m=g.dz_m.median(), dh_m=g.dh_m.median(), rot_deg=g.rot_deg.median(),
                 frac_real_with_lanes=float((g.real_lanes > 0).mean()) if "real_lanes" in g else np.nan)
        for k in KS:
            if k in g and not g[k].isna().all():
                r[k] = float(np.nanmean(g[k])); r[k + "_n"] = int(g[k].notna().sum())
                r[k + "_clo"], r[k + "_chi"] = boot(g[k].values, groups=g.scene.values)   # cluster: sequences
                r[k + "_lo"], r[k + "_hi"] = boot(g[k].values)                              # frames
        recs.append(r)
    S = pd.DataFrame(recs); S.to_csv(f"{OUT}/summary.csv", index=False)
    # paired difference vs raw (same frame), cluster CI over sequences
    P = []
    for (d, kind), g in df.groupby(["dir", "kind"]):
        base = g[g.method == "raw"].set_index(["scene", "name"])
        for meth in ORDER[1:]:
            q = g[g.method == meth].set_index(["scene", "name"])
            j = q.join(base, rsuffix="_raw", how="inner")
            if not len(j):
                continue
            rec = dict(dir=d, kind=kind, method=meth, n=len(j))
            for k in ("psnr", "lpips", "mark_f1", "yolo_halluc_veh"):
                dd = (j[k] - j[k + "_raw"]).values; rec[f"d_{k}"] = float(np.nanmean(dd))
                rec[f"d_{k}_clo"], rec[f"d_{k}_chi"] = boot(dd, groups=j.index.get_level_values(0).values)
            P.append(rec)
    Pd = pd.DataFrame(P); Pd.to_csv(f"{OUT}/paired_vs_raw.csv", index=False)
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("# MV2 car <-> scooter -- generated by src/mv2/evaluate.py summary\n\nMean [95% bootstrap CI over "
                "sequences]; movable objects and the car bonnet pasted from the real image. `onpath` = the training "
                "vehicle's own held-out frames (every 8th); `cross` = the other vehicle's simultaneous frames.\n")
        for d in sorted(S.dir.unique()):
            for kind in ("onpath", "cross"):
                A = S[(S.dir == d) & (S.kind == kind)]
                if not len(A):
                    continue
                r0 = A.iloc[0]
                f.write(f"\n## {d} / {kind} (n_seq {r0.n_seq}; median |lateral| {r0.lat_abs_m:.1f} m, dz {r0.dz_m:+.2f} m, "
                        f"camera-height diff {r0.dh_m:+.2f} m, rotation {r0.rot_deg:.0f} deg; real frames with CLRNet "
                        f"lanes {r0.frac_real_with_lanes:.0%})\n\n| method | n | " + " | ".join(KS) + " |\n|" + "---|" * (2 + len(KS)) + "\n")
                for m in [m for m in ORDER if m in set(A.method)]:
                    r = A[A.method == m].iloc[0]
                    cells = [f"{r[k]:.3f} [{r[k+'_clo']:.2f},{r[k+'_chi']:.2f}]" if k in r and not pd.isna(r[k]) else "–" for k in KS]
                    f.write(f"| {m} | {r['n']} | " + " | ".join(cells) + " |\n")
        f.write("\n## Paired change vs. raw splat (same frames), mean [95% CI over sequences]\n\n"
                "| dir | kind | method | n | dPSNR | dLPIPS | d paint F1 | d fake veh./img |\n|---|---|---|---|---|---|---|---|\n")
        for _, r in Pd.iterrows():
            c = lambda k, p: f"{r['d_'+k]:+.{p}f} [{r['d_'+k+'_clo']:+.{p}f},{r['d_'+k+'_chi']:+.{p}f}]"
            f.write(f"| {r.dir} | {r.kind} | {r.method} | {r.n} | {c('psnr', 2)} | {c('lpips', 3)} | {c('mark_f1', 3)} | {c('yolo_halluc_veh', 3)} |\n")
    print(open(f"{OUT}/summary.md").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "summary"]); ap.add_argument("dir", nargs="?")
    ap.add_argument("--methods", default=METHODS)
    a = ap.parse_args()
    evaluate(a.dir, a.methods.split(",")) if a.cmd == "eval" else summary()


if __name__ == "__main__":
    main()
