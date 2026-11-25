"""EUVS [6] evaluation of every fixer against the REAL test views (proposal step 6, heading axis with real ground truth).
Same metrics and conventions as the Para-Lane lateral evaluation (src/eval/evaluate.py, src/eval/lanes/ntl_iou.py):
PSNR / SSIM / LPIPS with movable-object pixels (YOLO11m-seg masks of the real view) pasted from the real image, road
LPIPS, lane-paint P/R/F1, road-texture ratio, YOLO fake vehicles and static-object F1; CLRNet NTL-IoU and lane F1 vs.
CLRNet on the real view -- ONLY for the front camera (MARS channel 2, used in L3), since a CULane lane detector is not
meaningful on +-60 deg side cameras (all rot60 test views). CLRNet's crop band is placed at the per-view horizon row
from the metric, z-up pose (median over the evaluated views; src/euvs/fit.py 'horizon').

All fixers were trained / fine-tuned on Para-Lane (or are generic), so this is also a TRANSFER test (other country,
other camera, 1440x928 -> 960x608).

usage:
  scripts/lib/py.sh src/euvs/evaluate.py eval <proto> --methods raw,difix,...   -> results/euvs/rows_<proto>.json
  scripts/lib/py.sh src/euvs/evaluate.py summary                                -> results/euvs/summary.csv / .md
"""
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image

from common.config import DATA, REPO, YOLO_WEIGHTS
from eval.metrics import Metrics
from eval.evaluate import marking_mask, mark_agree, tex_energy, det_f1, STATIC, MOVABLE, ROAD_Y0
from eval.lanes.ntl_iou import compare

EROOT = f"{DATA}/euvs/out"; FIXED = f"{DATA}/euvs/fixed"; OUT = f"{REPO}/results/euvs"
LANE_CH = 2  # front camera


def evaluate(proto, methods):
    from ultralytics import YOLO
    meta = json.load(open(f"{EROOT}/{proto}/meta_all.json"))
    yolo = YOLO(YOLO_WEIGHTS); M = Metrics("cuda")
    det = lambda im: (lambda r: (r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int)))(
        yolo(np.ascontiguousarray(im[..., ::-1]), verbose=False, conf=0.3, device=0)[0].boxes)
    lanes = {m: json.load(open(f"{EROOT}/{proto}/lanes_{m}.json")) for m in methods + ["real"]
             if os.path.exists(f"{EROOT}/{proto}/lanes_{m}.json")}
    rows = []
    for n, it in enumerate(meta):
        real = np.array(Image.open(f"{EROOT}/{it['real']}").convert("RGB")); fg = np.array(Image.open(f"{EROOT}/{it['fg']}")) > 127
        H, W = real.shape[:2]; y0 = int(ROAD_Y0 * H)
        G = torch.from_numpy(real).float().cuda() / 255; FG = torch.from_numpy(fg).cuda()
        gm = marking_mask(real, fg); gtex = tex_energy(real, fg); gb, gc = det(real); gs = np.isin(gc, list(STATIC))
        for meth in methods:
            p = f"{EROOT}/{it['path']}" if meth == "raw" else f"{FIXED}/{meth}/{it['path']}"
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
                     yolo_static_f1=det_f1(gb[gs], gc[gs], b[s], c[s]), yolo_halluc_veh=hall,
                     method=meth, proto=proto, scene=it["scene"], name=it["name"], channel=it["channel"],
                     dist_m=it["dist_m"], rot_deg=it["rot_deg"])
            if it["channel"] == LANE_CH and meth in lanes and it["path"] in lanes[meth] and it["real"] in lanes["real"]:
                g, q = lanes["real"][it["real"]], lanes[meth][it["path"]]
                r.update(compare(q["lanes"], g["lanes"], (int(min(g["band"][1], H)), W), 30 * W / 1640))
            rows.append(r)
        if n % 50 == 0:
            print(proto, n, "/", len(meta), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/rows_{proto}.json", "w"), default=float)


METRICS = ["psnr", "ssim", "lpips", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "tex_ratio", "ntl_iou", "lane_f1",
           "yolo_static_f1", "yolo_halluc_veh"]


def summary():
    import pandas as pd
    from eval.summarize import boot
    df = pd.DataFrame(sum([json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/rows_*.json"))], []))
    df["seq"] = df.scene + ":ch" + df.channel.astype(str)
    recs = []
    for (proto, meth), g in df.groupby(["proto", "method"]):
        r = dict(proto=proto, method=meth, n=len(g), n_seq=g.seq.nunique(), dist_m=g.dist_m.median(), rot_deg=g.rot_deg.median())
        for k in METRICS:
            if k in g and not g[k].isna().all():
                r[k] = float(np.nanmean(g[k])); r[k + "_clo"], r[k + "_chi"] = boot(g[k].values, groups=g.seq.values)
        recs.append(r)
    S = pd.DataFrame(recs); S.to_csv(f"{OUT}/summary.csv", index=False)
    order = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]
    ks = ["psnr", "ssim", "lpips", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "ntl_iou", "lane_f1", "yolo_halluc_veh", "tex_ratio"]
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("# EUVS (MARS scenes) -- generated by src/euvs/evaluate.py summary\n\nMean [95% bootstrap CI over "
                "(scene, camera) sequences]; lanes (ntl_iou, lane_f1) on the front camera only.\n")
        for proto in sorted(S.proto.unique()):
            A = S[S.proto == proto]
            f.write(f"\n## {proto} (median nearest-train distance {A.dist_m.iloc[0]:.1f} m, rotation {A.rot_deg.iloc[0]:.0f} deg)\n\n"
                    "| method | n | " + " | ".join(ks) + " |\n|" + "---|" * (2 + len(ks)) + "\n")
            for m in [m for m in order if m in set(A.method)] + sorted(set(A.method) - set(order)):
                r = A[A.method == m].iloc[0]
                cells = [f"{r[k]:.3f} [{r[k+'_clo']:.2f},{r[k+'_chi']:.2f}]" if k in r and not pd.isna(r[k]) else "–" for k in ks]
                f.write(f"| {m} | {r['n']} | " + " | ".join(cells) + " |\n")
    print(open(f"{OUT}/summary.md").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "summary"]); ap.add_argument("proto", nargs="?")
    ap.add_argument("--methods", default="raw,difix,difix_ref,nv_fixer,difix_ft,difix_reimpl_ref,ours_hfgan_s0")
    a = ap.parse_args()
    evaluate(a.proto, a.methods.split(",")) if a.cmd == "eval" else summary()


if __name__ == "__main__":
    main()
