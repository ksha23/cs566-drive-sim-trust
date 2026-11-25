"""Mark invented and erased content on an output frame, with the same rules as the metrics (src/eval/evaluate.py):
  invented paint  output lane paint farther than 3 px from any real lane paint      -> orange
  erased paint    real lane paint farther than 3 px from any output lane paint      -> blue
  spurious vehicle YOLO11 vehicle/person box less than 30% covered by the real frame's moving-object mask -> red box
Marks are thickened by 2 px so that they stay visible when the tile is shrunk for the video. Used by make_videos.py only.
"""
import cv2
import numpy as np

from common.config import YOLO_WEIGHTS
from eval.evaluate import marking_mask, MOVABLE

ORANGE, BLUE, RED = (255, 140, 0), (40, 120, 255), (230, 30, 30)
_yolo = None


def spurious_boxes(img, fg):
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_WEIGHTS)
    r = _yolo(np.ascontiguousarray(img[..., ::-1]), verbose=False, conf=0.3, device="cpu")[0].boxes
    out = []
    for (x1, y1, x2, y2), c in zip(r.xyxy.cpu().numpy().astype(int), r.cls.cpu().numpy().astype(int)):
        if c in MOVABLE:
            patch = fg[max(y1, 0):y2, max(x1, 0):x2]
            if patch.size == 0 or patch.mean() < 0.3:
                out.append((x1, y1, x2, y2))
    return out


def annotate(img, real, fg, vehicles=True, tol=3):
    """img, real: HxWx3 uint8 (same size); fg: HxW bool moving-object mask of the real frame. Returns the marked copy
    and counts (invented px, erased px, spurious boxes)."""
    p, g = marking_mask(img, fg), marking_mask(real, fg)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    inv = p & ~(cv2.dilate(g.astype(np.uint8), k) > 0)
    era = g & ~(cv2.dilate(p.astype(np.uint8), k) > 0)
    thick = np.ones((5, 5), np.uint8)
    out = img.copy()
    for m, col in ((era, BLUE), (inv, ORANGE)):
        m = cv2.dilate(m.astype(np.uint8), thick) > 0
        out[m] = (0.25 * out[m] + 0.75 * np.array(col)).astype(np.uint8)
    boxes = spurious_boxes(img, fg) if vehicles else []
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 4)
    return out, (int(inv.sum()), int(era.sum()), len(boxes))
