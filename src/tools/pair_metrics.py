"""Trust metrics of one render (or fixer output) against the real image at the same pose (command-line tool).
Same definitions as the results (src/eval/evaluate.py): PSNR / SSIM / LPIPS with movable-object pixels taken from the
real image (--fg mask, optional), road LPIPS (rows below 60%), lane-paint precision / recall / F1 (3-px tolerance),
road texture ratio, YOLO11 fake vehicles (vehicle/person detections where the real frame has no movable object; needs
--fg). Paint precision is undefined (null) when the render has no lane paint (as in the results, paint_counts.py).
CLRNet lane F1 needs the separate lanes env (scripts/lib/lanes_scene.sh) and is not computed here. Without the YOLO
weights at $CS566D_SMOKE/weights/yolo11m.pt, ultralytics downloads yolo11m.pt into the current folder.
example (CPU, no dataset needed): python src/tools/pair_metrics.py examples/scene_17_frame/our_reimplementation.png \
             examples/scene_17_frame/real.png --fg examples/scene_17_frame/moving_mask.png --cpu
usage: scripts/lib/py.sh src/tools/pair_metrics.py <render.png> <real.png> [--fg mask.png] [--cpu]
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/, also when run without py.sh
from eval.metrics import Metrics
from eval.evaluate import marking_mask, mark_agree, tex_energy, MOVABLE, ROAD_Y0
from common.config import YOLO_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render"); ap.add_argument("real"); ap.add_argument("--fg", default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(); dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    real = np.array(Image.open(a.real).convert("RGB"))
    x = np.array(Image.open(a.render).convert("RGB").resize(real.shape[1::-1], Image.BICUBIC))
    fg = np.array(Image.open(a.fg).convert("L")) > 127 if a.fg else np.zeros(real.shape[:2], bool)
    M = Metrics(dev); X = torch.from_numpy(x).float().to(dev) / 255; G = torch.from_numpy(real).float().to(dev) / 255
    FG = torch.from_numpy(fg).to(dev); y0 = int(ROAD_Y0 * real.shape[0])
    r = M(X, G, FG if a.fg else None); r["road_lpips"] = M(X[y0:], G[y0:], FG[y0:] if a.fg else None)["lpips"]
    px = marking_mask(x, fg); f1, p, rc = mark_agree(px, marking_mask(real, fg))
    if px.sum() < 50:  # no paint in the render: precision undefined
        p = float("nan")
    r.update(paint_precision=p, paint_recall=rc, paint_f1=f1, tex_ratio=float(tex_energy(x, fg) / (tex_energy(real, fg) + 1e-9)))
    if a.fg:
        from ultralytics import YOLO
        b = YOLO(YOLO_WEIGHTS if os.path.exists(YOLO_WEIGHTS) else "yolo11m.pt")(np.ascontiguousarray(x[..., ::-1]), verbose=False, conf=0.3, device=0 if dev == "cuda" else "cpu")[0].boxes
        bb, c = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int); n = 0
        for x1, y1, x2, y2 in bb[np.isin(c, list(MOVABLE))].astype(int):
            patch = fg[max(y1, 0):y2, max(x1, 0):x2]
            n += int(patch.size == 0 or patch.mean() < 0.3)
        r["fake_vehicles"] = n
    print(json.dumps({k: (round(float(v), 4) if v == v else None) for k, v in r.items()}, indent=1))


if __name__ == "__main__":
    main()
