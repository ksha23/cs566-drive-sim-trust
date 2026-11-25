"""Para-Lane clip loader: front-camera images, COLMAP poses, foreground (movable-object) masks,
and the per-clip merged LiDAR cloud. All three clips of a scene share one world frame
(x forward, y left, z up), so a splat from one lane can be rendered at another lane's poses."""
import os
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from common.config import PARALANE_ROOTS

W, H = 960, 540  # working resolution (native 1910x1075, i.e. ~0.5x)


def clip_dir(scene, clip):
    for root in PARALANE_ROOTS:
        d = f"{root}/{scene}/{clip}"
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"{scene}/{clip} not found under {PARALANE_ROOTS}")


def load_cameras(d):
    """Returns K (3x3, at working res), sorted list of timestamps, c2w (N,4,4)."""
    cam = [l.split() for l in open(f"{d}/sparse/0/cameras.txt") if not l.startswith("#")][0]
    w0, h0 = int(cam[2]), int(cam[3]); fx, fy, cx, cy = map(float, cam[4:8])
    sx, sy = W / w0, H / h0
    K = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
    lines = [l for l in open(f"{d}/sparse/0/images.txt") if not l.startswith("#")][::2]
    recs = []
    for l in lines:
        p = l.split(); qw, qx, qy, qz = map(float, p[1:5]); t = np.array(list(map(float, p[5:8])))
        Rw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c = np.eye(4); w2c[:3, :3] = Rw; w2c[:3, 3] = t
        recs.append((p[9].split("/")[0], np.linalg.inv(w2c)))
    recs.sort(key=lambda r: int(r[0]))
    return K, [r[0] for r in recs], np.stack([r[1] for r in recs]).astype(np.float32)


def load_image(d, ts):
    return np.array(Image.open(f"{d}/images/{ts}/CAMERA_FRONT.png").convert("RGB").resize((W, H), Image.BICUBIC))


def load_fg(d, ts):
    p = f"{d}/foreground_labels/{ts}/CAMERA_FRONT.png"
    if not os.path.exists(p):
        return np.zeros((H, W), bool)
    m = np.array(Image.open(p).convert("L").resize((W, H), Image.NEAREST)) > 127
    # dilate a little so object borders/shadows are also excluded
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def load_clip(scene, clip, device="cuda"):
    d = clip_dir(scene, clip)
    K, ts, c2w = load_cameras(d)
    imgs = np.stack([load_image(d, t) for t in ts])
    fgs = np.stack([load_fg(d, t) for t in ts])
    return dict(dir=d, K=torch.tensor(K, device=device), ts=ts, c2w=torch.tensor(c2w, device=device),
                imgs=torch.from_numpy(imgs).to(device), fg=torch.from_numpy(fgs).to(device))


def load_lidar(scene, clip):
    v = PlyData.read(f"{clip_dir(scene, clip)}/sparse/0/merged_lidar_pcd.ply")["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)


def lateral_offset(c2w_query, c2w_path):
    """Signed lateral distance (m, +left) of each query camera from the polyline of path camera centers,
    measured in the ground plane."""
    P = c2w_path[:, :2, 3].cpu().numpy(); Q = c2w_query[:, :2, 3].cpu().numpy()
    out = []
    for q in Q:
        best = (1e9, 0.0)
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a; t = np.clip(np.dot(q - a, ab) / (ab @ ab + 1e-9), 0, 1); pr = a + t * ab
            dist = np.linalg.norm(q - pr)
            if dist < best[0]:
                cross = ab[0] * (q - a)[1] - ab[1] * (q - a)[0]
                best = (dist, np.sign(cross) * dist)
        out.append(best[1])
    return np.array(out)
