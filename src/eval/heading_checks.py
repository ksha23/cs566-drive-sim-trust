"""Sanity checks of the heading axis, written to results/heading/checks.json (CPU):
  zero_repro   the y+0_z+0.0 rows of every method vs. the lateral evaluation's rows of the same images
               (results/eval/eval_scene_1[67]_a.json + difix_reimpl/nv_fixer files): max |difference| per metric
  resampling   raw splat, scene_16, +-5/10 deg: PSNR of the render vs. the warped real frame and of the 0-deg render vs.
               the real frame on the same content, and a real frame warped there and back, for bilinear vs. Lanczos-4
usage: scripts/lib/py.sh src/eval/heading_checks.py
"""
import glob, json
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from common.config import DATA, REPO
from eval.heading_geom import homography, warp_img
from eval.heading import psnr_mask

OUT = f"{REPO}/results/heading"


def zero_repro():
    H = pd.DataFrame(sum([json.load(open(p)) for p in glob.glob(f"{OUT}/rows_scene_*.json")], []))
    H = H[(H.cond == "y+0_z+0.0") & (H.axis != "height")]
    E = pd.DataFrame(sum([json.load(open(p)) for p in glob.glob(f"{REPO}/results/eval/eval_scene_1[67]_*.json")
                          + glob.glob(f"{REPO}/results/difix_reimpl/eval_scene_1[67]_*.json") if "summary" not in p], []))
    L = pd.DataFrame(sum([json.load(open(p)) for p in glob.glob(f"{REPO}/results/lanes/ntl_scene_1[67]_*.json")
                          + glob.glob(f"{REPO}/results/difix_reimpl/ntl_scene_1[67]_*.json")], []))
    k = ["method", "scene", "src", "dst", "ts"]
    E = E.drop_duplicates(k); L = L.drop_duplicates(k)
    m = H.merge(E, on=k, suffixes=("", "_lat")).merge(L[k + ["ntl_iou", "lane_f1"]], on=k, suffixes=("", "_lat"))
    out = {"n_rows": int(len(m)), "methods": sorted(m.method.unique())}
    for c in ["psnr", "ssim", "lpips", "mark_f1", "mark_prec", "mark_rec", "yolo_halluc_veh", "yolo_static_f1", "tex_ratio", "ntl_iou", "lane_f1"]:
        d = (m[c] - m[c + "_lat"]).abs()
        out[c + "_max_abs_diff"] = float(np.nanmax(d.values))
    return out


def resampling():
    M = [m for m in json.load(open(f"{DATA}/heading/scene_16/meta.json")) if m["axis"] == "yaw" and abs(m["yaw"]) in (5.0, 10.0)]
    L = lambda p: np.array(Image.open(p).convert("RGB"))
    res = {}
    for name, ip in [("bilinear", cv2.INTER_LINEAR), ("lanczos4", cv2.INTER_LANCZOS4)]:
        acc = []
        for m in M:
            real = L(f"{DATA}/pairs/{m['real']}"); fg = np.array(Image.open(f"{DATA}/pairs/{m['fg']}")) > 127
            x = L(f"{DATA}/heading/{m['path']}"); x0 = L(f"{DATA}/heading/{m['zero']}")
            H = homography(m["K"], m["c2w0"], m["c2w1"], m["cam_h"]); Hi = np.linalg.inv(H)
            gw, v = warp_img(real, H, ip); fgw = warp_img(fg.astype(np.uint8), H, cv2.INTER_NEAREST)[0] > 0
            sv = warp_img(v.astype(np.uint8), Hi, cv2.INTER_NEAREST)[0] > 0
            rt = warp_img(gw, Hi, ip)[0]; inner = cv2.erode(sv.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            acc.append([psnr_mask(x, gw, v & ~fgw), psnr_mask(x0, real, sv & ~fg), psnr_mask(rt, real, inner & ~fg)])
        a = np.nanmean(acc, 0)
        res[name] = dict(n=len(acc), psnr_render_vs_warped_real=float(a[0]), psnr_0deg_render_vs_real_same_content=float(a[1]),
                         psnr_real_round_trip=float(a[2]))
    return res


if __name__ == "__main__":
    out = dict(zero_repro=zero_repro(), resampling=resampling())
    json.dump(out, open(f"{OUT}/checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
