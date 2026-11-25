"""Lane agreement between a render (raw / fixed) and the REAL image at the same pose (NTL-IoU style).

ReconDreamer's NTL-IoU compares lanes detected on the novel-trajectory render with reference lanes. Para-Lane has no
HD-map lanes, but it has the REAL image of the adjacent lane at the same pose, so the reference is CLRNet's detection on
that real image (same detector, same settings, same crop).

Definitions (per image pair; lanes from src/eval/lanes/clrnet_infer.py, in original-image pixels):
  * rasterise every lane polyline with thickness t = 30 px at CULane scale (1640 px wide), i.e. 30*W/1640 px
    (~17.6 px at W=960); only rows inside CLRNet's field of view (the band reported by clrnet_infer) exist for both.
  * NTL-IoU = |M_render AND M_real| / |M_render OR M_real| of the union-of-lanes masks.
    Both empty -> undefined (NaN, excluded from means); exactly one empty -> 0.
  * Per-lane matching exactly like the CULane metric: IoU between individual rasterised lanes, Hungarian matching,
    a match with IoU >= 0.5 is a TP; unmatched render lanes are FP (hallucinated lanes), unmatched real lanes FN.
    lane_prec = TP/(TP+FP), lane_rec = TP/(TP+FN), lane_f1 per image (NaN if both have no lanes). The dataset-level
    CULane F1 (sum TP/FP/FN over images) is printed in the summary.
  * n_lanes_render - n_lanes_real (count bias) is also stored.

usage (main env):
  python ntl_iou.py --pairs PAIRS_ROOT --scene scene_16 --real real.json --method raw=raw.json \
      --method difix=difix.json ... --out rows.json [--stride 2]
Row keys: method, scene, src, dst, ts, offset_m, onpath + metrics (merge with src/eval/evaluate.py rows on
(method, scene, src, dst, ts)).
"""
import argparse, json, os, sys
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment



def raster(lane, shape, t):
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(lane, np.float64) * 8).astype(np.int32)  # sub-pixel precision (shift=3)
    cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 1, thickness=max(int(round(t)), 1), lineType=cv2.LINE_8, shift=3)
    return m.astype(bool)


def compare(lr, lg, shape, t, thr=0.5):
    """lr, lg: lists of polylines (render, real). Returns dict of metrics."""
    Mr = [raster(l, shape, t) for l in lr]; Mg = [raster(l, shape, t) for l in lg]
    ur = np.any(Mr, 0) if Mr else np.zeros(shape, bool); ug = np.any(Mg, 0) if Mg else np.zeros(shape, bool)
    union = (ur | ug).sum()
    ntl = np.nan if (not Mr and not Mg) else float((ur & ug).sum() / max(union, 1))
    tp = 0
    if Mr and Mg:
        iou = np.zeros((len(Mr), len(Mg)))
        for i, a in enumerate(Mr):
            for j, b in enumerate(Mg):
                u = (a | b).sum(); iou[i, j] = (a & b).sum() / u if u else 0
        r, c = linear_sum_assignment(-iou)
        tp = int((iou[r, c] >= thr).sum())
    fp, fn = len(Mr) - tp, len(Mg) - tp
    p = tp / (tp + fp) if tp + fp else np.nan
    rc = tp / (tp + fn) if tp + fn else np.nan
    f1 = np.nan if (not Mr and not Mg) else (2 * tp / (2 * tp + fp + fn))
    return dict(ntl_iou=ntl, lane_prec=p, lane_rec=rc, lane_f1=f1, lane_tp=tp, lane_fp=fp, lane_fn=fn,
                n_lanes_render=len(Mr), n_lanes_real=len(Mg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--scene", required=True)
    ap.add_argument("--real", required=True, help="CLRNet json for real images (keys = pairs-relative paths)")
    ap.add_argument("--method", action="append", default=[], help="name=json (keys = meta 'path' entries)")
    ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--W", type=int, default=960); ap.add_argument("--H", type=int, default=540)
    a = ap.parse_args()
    from fixers.apply import eval_items
    items = eval_items(a.pairs, a.scene, a.stride)
    real = json.load(open(a.real)); t = 30 * a.W / 1640
    rows = []
    for spec in a.method:
        name, path = spec.split("=", 1); det = json.load(open(path))
        for m in items:
            if m["path"] not in det or m["real"] not in real:
                continue
            g, r = real[m["real"]], det[m["path"]]
            y1 = int(min(g["band"][1], a.H))  # both detections live in the same band (same camera), clip to it
            res = compare(r["lanes"], g["lanes"], (y1, a.W), t)
            res.update(method=name, scene=m["scene"], src=m["src"], dst=m["dst"], ts=m["ts"],
                       offset_m=m["offset_m"], onpath=m["onpath_test"])
            rows.append(res)
    json.dump(rows, open(a.out, "w"), default=float)
    # summary by |offset| bin
    bins = [(-0.1, 1.0, "0m"), (1.0, 4.5, "~3m"), (4.5, 20, "~6m")]
    print(f"{'method':10s} {'bin':5s} {'n':>4s} {'NTL-IoU':>8s} {'laneF1(img)':>11s} {'CULaneF1':>8s} {'prec':>6s} {'rec':>6s} {'dLanes':>6s}")
    for name in dict.fromkeys(r["method"] for r in rows):
        for lo, hi, lab in bins:
            R = [r for r in rows if r["method"] == name and lo < abs(r["offset_m"]) <= hi]
            if not R:
                continue
            f = lambda k: np.nanmean([r[k] for r in R])
            TP, FP, FN = (sum(r[k] for r in R) for k in ("lane_tp", "lane_fp", "lane_fn"))
            cf1 = 2 * TP / max(2 * TP + FP + FN, 1)
            print(f"{name:10s} {lab:5s} {len(R):4d} {f('ntl_iou'):8.3f} {f('lane_f1'):11.3f} {cf1:8.3f} "
                  f"{TP/max(TP+FP,1):6.3f} {TP/max(TP+FN,1):6.3f} {np.mean([r['n_lanes_render']-r['n_lanes_real'] for r in R]):+6.2f}")


if __name__ == "__main__":
    main()
