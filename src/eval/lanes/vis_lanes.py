"""Draw CLRNet lanes on real vs. render/fixed images side by side.
usage: python vis_lanes.py <out.jpg> <real_json> <real_img_root> <name>=<json>=<img_root> ... --keys k1,k2  (keys = meta 'path')
       real key is derived from the meta item; or pass explicit image pairs with --pairs real_key:render_key,...
"""
import argparse, json, os
import numpy as np
import cv2

COL = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0)]


def draw(img, det, t=3):
    img = img.copy()
    if det is not None:
        y0, y1 = det["band"]
        cv2.line(img, (0, int(y1)), (img.shape[1], int(y1)), (80, 80, 80), 1)
        for i, l in enumerate(det["lanes"]):
            cv2.polylines(img, [np.round(np.array(l)).astype(np.int32).reshape(-1, 1, 2)], False, COL[i % 6], t)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--real", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--meta", required=True, help="pairs meta.json"); ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--keys", required=True, help="comma-separated meta 'path' values")
    a = ap.parse_args()
    meta = {m["path"]: m for m in json.load(open(a.meta))}
    real = json.load(open(a.real)); meths = []
    for s in a.method:
        n, j, r = s.split("=", 2); meths.append((n, json.load(open(j)), r))
    rows = []
    for k in a.keys.split(","):
        m = meta[k]
        tiles = [("REAL " + m["dst"], draw(cv2.imread(f"{a.root}/{m['real']}"), real.get(m["real"])))]
        for n, d, r in meths:
            tiles.append((n, draw(cv2.imread(f"{r}/{k}"), d.get(k))))
        tiles = [cv2.putText(cv2.resize(im, (480, 270)), f"{n}  off {m['offset_m']:+.1f}m", (6, 18),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) for n, im in tiles]
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
