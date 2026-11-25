"""Trust metrics vs. lateral offset for each method's output against the REAL image of the target lane.

Per image:
  psnr / ssim / lpips          full-frame vs real (movable-object pixels pasted from real)
  road_lpips                   LPIPS on the road crop (lower 40% of the frame)
  mark_f1 / mark_prec / mark_rec   lane-marking agreement: white/yellow paint masks from a color + local-contrast
                               threshold in the road region, 3-px tolerant precision/recall vs the real image.
                               mark_prec low = paint that is not in the real image (hallucinated / misplaced markings)
  tex_ratio                    road texture energy (mean |Laplacian|) of output / real (1 = same "sharpness")
  yolo_static_f1               YOLO11 agreement (IoU>=0.5, same class) on static classes (traffic light, stop sign, ...)
  yolo_halluc_veh              # vehicle/person detections in the output that don't overlap any real movable object
usage: scripts/lib/py.sh src/eval/evaluate.py <pairs_root> <scene> <fixed_root> <out_json> --methods raw,difix,difix_ref,ours
"""
import argparse, json, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

from eval.metrics import Metrics
from fixers.apply import eval_items
from common.config import YOLO_WEIGHTS

DEV = os.environ.get("EVAL_DEVICE", "cuda")  # evaluation can run on CPU while the GPU is busy

ROAD_Y0 = 0.60  # road region = rows below 60% of image height (camera pitched ~6 deg down; horizon ~45%)
STATIC = {9, 11, 12, 13}  # traffic light, stop sign, parking meter, bench
MOVABLE = {0, 1, 2, 3, 5, 7}


def marking_mask(img, fg):
    """White/yellow paint in the road region: brighter than local road median by a margin, low saturation (white)
    or yellow hue. img uint8 HxWx3."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[..., 2] / 255; s = hsv[..., 1] / 255; h = hsv[..., 0]
    # asphalt brightness per image row = 25th percentile of the row (paint covers < 75% of any row)
    vv = np.where(fg, np.nan, v)
    with np.errstate(all="ignore"):
        rowbg = np.nanpercentile(vv, 25, axis=1)
    rowbg = np.nan_to_num(rowbg, nan=np.nanmedian(rowbg))
    rowbg = np.convolve(np.pad(rowbg, 7, mode="edge"), np.ones(15) / 15, mode="valid")
    med = np.repeat(rowbg[:, None], W, 1)
    white = (v > med + 0.12) & (s < 0.25) & (v > 0.55)
    yellow = (h > 15) & (h < 35) & (s > 0.35) & (v > 0.45) & (v > med + 0.05)
    m = (white | yellow)
    m[: int(ROAD_Y0 * H)] = False
    m &= ~fg
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return m


def mark_agree(p, g, tol=3):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gd, pd = cv2.dilate(g.astype(np.uint8), k) > 0, cv2.dilate(p.astype(np.uint8), k) > 0
    if g.sum() < 50 and p.sum() < 50:
        return np.nan, np.nan, np.nan
    prec = (p & gd).sum() / max(p.sum(), 1); rec = (g & pd).sum() / max(g.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def tex_energy(img, fg):
    H = img.shape[0]; g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    r = np.zeros_like(fg); r[int(ROAD_Y0 * H):] = True; r &= ~fg
    return lap[r].mean()


def box_iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def det_f1(A, ca, B, cb):
    if len(A) == 0 and len(B) == 0:
        return np.nan
    if len(A) == 0 or len(B) == 0:
        return 0.0
    M = box_iou(A, B) * (ca[:, None] == cb[None]); tp = 0
    while M.size and M.max() >= 0.5:
        i, j = np.unravel_index(M.argmax(), M.shape); tp += 1; M[i, :] = 0; M[:, j] = 0
    return 2 * tp / (len(A) + len(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("scene"); ap.add_argument("fixed"); ap.add_argument("out")
    ap.add_argument("--methods", default="raw,difix,difix_ref,ours"); ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    def det(img):
        r = yolo(np.ascontiguousarray(img[..., ::-1]), verbose=False, conf=0.3, device=0 if DEV == "cuda" else "cpu")[0].boxes
        return r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int)

    M = Metrics(DEV); rows = []
    items = eval_items(a.pairs, a.scene, a.stride)
    for n, m in enumerate(items):
        real = np.array(Image.open(f"{a.pairs}/{m['real']}"))
        fg = np.array(Image.open(f"{a.pairs}/{m['fg']}")) > 127
        gmask = marking_mask(real, fg); gtex = tex_energy(real, fg)
        gb, gc = det(real)
        gs = np.isin(gc, list(STATIC))
        G = torch.from_numpy(real).float().to(DEV) / 255; FG = torch.from_numpy(fg).to(DEV)
        H = real.shape[0]; y0 = int(ROAD_Y0 * H)
        for meth in a.methods.split(","):
            p = f"{a.pairs}/{m['path']}" if meth == "raw" else f"{a.fixed}/{meth}/{m['path']}"
            if not os.path.exists(p):
                continue
            x = np.array(Image.open(p).convert("RGB"))
            X = torch.from_numpy(x).float().to(DEV) / 255
            r = M(X, G, FG)
            r["road_lpips"] = M(X[y0:], G[y0:], FG[y0:])["lpips"]
            f1, pr, rc = mark_agree(marking_mask(x, fg), gmask)
            b, c = det(x); s = np.isin(c, list(STATIC))
            mv = np.isin(c, list(MOVABLE))
            hall = 0
            for x1, y1, x2, y2 in b[mv].astype(int):
                patch = fg[max(y1, 0):y2, max(x1, 0):x2]
                if patch.size == 0 or patch.mean() < 0.3:  # detected object where the real frame has none
                    hall += 1
            r.update(mark_f1=f1, mark_prec=pr, mark_rec=rc, tex_ratio=float(tex_energy(x, fg) / (gtex + 1e-9)),
                     yolo_static_f1=det_f1(gb[gs], gc[gs], b[s], c[s]), yolo_halluc_veh=hall,
                     method=meth, scene=m["scene"], offset_m=m["offset_m"], src=m["src"], dst=m["dst"], ts=m["ts"], onpath=m["onpath_test"])
            rows.append(r)
        if n % 20 == 0:
            print(n, "/", len(items), flush=True)
    json.dump(rows, open(a.out, "w"), default=float)
    # summary by offset bin
    import pandas as pd
    df = pd.DataFrame(rows); df["abs_off"] = df.offset_m.abs()
    df["bin"] = pd.cut(df.abs_off, [-0.1, 1.0, 4.5, 20], labels=["0m (on-path)", "~3.3m", "~6.3m"])
    cols = ["psnr", "ssim", "lpips", "road_lpips", "mark_f1", "mark_prec", "mark_rec", "tex_ratio", "yolo_static_f1", "yolo_halluc_veh"]
    s = df.groupby(["bin", "method"], observed=True)[cols].mean().round(4)
    s["n"] = df.groupby(["bin", "method"], observed=True).size()
    print(s.to_string())
    s.reset_index().to_csv(a.out.replace(".json", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
