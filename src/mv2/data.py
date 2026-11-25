"""MV2 [9] loader (HF sanjay810/mv2-dataset): car front camera ("car") and scooter front camera ("scooty") of one
sequence, time-synchronized, in a METRIC, z-up world frame.

What MV2 ships per view and frame k (checked on the data, see docs/mv2.md):
  images/<cam>_<k:06d>.jpg   1920x1080 (GoPro, 60 FPS video sampled at 2 FPS, 100 frames per sequence)
  poses/<cam>_<k:06d>.npy    4x4 WORLD-TO-CAMERA, OpenCV axes (x right, y down, z forward); one world frame for
                             all vehicles of a sequence (COLMAP registration, nerfstudio-style normalized, z up)
  intrinsics/<cam>_<k>.npy   [fx, fy, cx, cy, W, H] for 960x540  -> we load images at 960x540 and use K as is
  scale_factor.npy / transform.npy  the normalization from the COLMAP frame (NOT metric: COLMAP has no scale)
Time sync: MV2 syncs all vehicles by wall clock and samples every sensor at 2 FPS, so frame k of the car and frame k
of the scooter are simultaneous (the car->scooter offset is nearly constant over a sequence, std ~ 1-5 % of it).

Metric scale (MV2 gives none): we triangulate points from the car's own views with the given poses, fit the road
plane under the car, and set the scale so the car camera is CAR_CAM_H meters above the road (our assumption, see
docs/mv2.md). Camera heights above the road (and so the car/scooter height ratio) are measured, not assumed.

Masks: movable objects = YOLO11m-seg (as EUVS, src/euvs/data.py); ego = a fixed bottom band of the car
view (its bonnet), EGO_ROWS.
"""
import glob, json, os
import cv2
import numpy as np
from PIL import Image

from common.config import DATA

RAW = f"{DATA}/mv2/raw"
W, H = 960, 540
CAR_CAM_H = 1.40   # assumed car-camera height above the road [m] (sets the metric scale)
PREFIX = {"car": "car_front", "scooty": "scooty_front"}


def frame_ids(seq, v):
    P = sorted(glob.glob(f"{RAW}/{seq}/{v}/poses/{PREFIX[v]}_0*.npy"))
    ids = [int(os.path.basename(p)[-10:-4]) for p in P]
    return [k for k in ids if os.path.exists(f"{RAW}/{seq}/{v}/images/{PREFIX[v]}_{k:06d}.jpg")]


def img_path(seq, v, k):
    return f"{RAW}/{seq}/{v}/images/{PREFIX[v]}_{k:06d}.jpg"


def load_raw(seq):
    """dict v -> dict(ids, c2w (N,4,4) in MV2 units, K (N,3,3))."""
    out = {}
    for v in PREFIX:
        ids = frame_ids(seq, v); c2w, K = [], []
        for k in ids:
            w2c = np.load(f"{RAW}/{seq}/{v}/poses/{PREFIX[v]}_{k:06d}.npy").astype(np.float64)
            c2w.append(np.linalg.inv(w2c))
            fx, fy, cx, cy, w, h = np.load(f"{RAW}/{seq}/{v}/intrinsics/{PREFIX[v]}_{k:06d}.npy").ravel()[:6]
            assert (int(w), int(h)) == (W, H), (w, h)
            K.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
        out[v] = dict(ids=ids, c2w=np.stack(c2w), K=np.stack(K))
    return out


def load_image(seq, v, k):
    return np.array(Image.open(img_path(seq, v, k)).convert("RGB").resize((W, H), Image.BICUBIC))


# ---------------------------------------------------------------- masks
def cache_dir(seq):
    d = f"{DATA}/mv2/cache/{seq}"; os.makedirs(d, exist_ok=True)
    return d


EGO_ROWS = {"car": 18, "scooty": 0}   # bottom rows showing the ego vehicle (car bonnet), by inspection (docs/mv2.md)


def ego_mask(seq, v, ids=None):
    """Ego-vehicle pixels: a fixed bottom band (the car's bonnet reaches ~14 px up at 960x540; scooter: none). A
    temporal-std detector does not work here: the bonnet flickers with auto-exposure as much as the road does."""
    m = np.zeros((H, W), bool)
    if EGO_ROWS[v]:
        m[-EGO_ROWS[v]:] = True
    return m


def dyn_mask_path(seq, v, k):
    return f"{cache_dir(seq)}/dyn_{v}/{k:06d}.png"


def make_dyn_masks(seq, v, ids, weights=f"{DATA}/euvs/weights/yolo11m-seg.pt", batch=16):
    """YOLO11m-seg movable objects (person, bicycle, car, motorcycle, bus, train, truck), conf 0.25, dilated 9 px
    (as src/euvs/data.py)."""
    todo = [k for k in ids if not os.path.exists(dyn_mask_path(seq, v, k))]
    if not todo:
        return
    from ultralytics import YOLO
    os.makedirs(os.path.dirname(dyn_mask_path(seq, v, todo[0])), exist_ok=True)
    y = YOLO(weights)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = y([load_image(seq, v, k)[..., ::-1].copy() for k in chunk], verbose=False, conf=0.25,
                classes=[0, 1, 2, 3, 5, 6, 7], retina_masks=True)
        for k, r in zip(chunk, res):
            m = np.zeros((H, W), np.uint8)
            if r.masks is not None:
                m = (r.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(cv2.dilate(m, np.ones((9, 9), np.uint8)) * 255).save(dyn_mask_path(seq, v, k))


def load_mask(seq, v, k, ego):
    """True = do not use (movable object or ego vehicle)."""
    return (np.array(Image.open(dyn_mask_path(seq, v, k))) > 127) | ego


# ---------------------------------------------------------------- triangulation with known poses
def triangulate(seq, v, ids, c2w, K, masks, gaps=(1, 2, 3), ratio=0.75, max_err=1.5, min_par=1.0):
    """SIFT matches between frames i and i+g of one camera, triangulated with the GIVEN poses (no pose estimation);
    kept if in front of both cameras, reprojection error < max_err px in both and parallax > min_par deg.
    Returns (N,3) points in the frame of c2w and (N,3) uint8 colors."""
    sift = cv2.SIFT_create(nfeatures=6000)
    feats = []
    for j, k in enumerate(ids):
        im = load_image(seq, v, k)
        kp, de = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (~masks[j]).astype(np.uint8) * 255)
        feats.append((np.array([p.pt for p in kp], np.float32), de, im))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    P = [K[j] @ np.linalg.inv(c2w[j])[:3] for j in range(len(ids))]
    pts, cols = [], []
    for a in range(len(ids)):
        for g in gaps:
            b = a + g
            if b >= len(ids) or feats[a][1] is None or feats[b][1] is None:
                continue
            mm = [m for m, n in bf.knnMatch(feats[a][1], feats[b][1], k=2) if m.distance < ratio * n.distance]
            if len(mm) < 8:
                continue
            xa = feats[a][0][[m.queryIdx for m in mm]]; xb = feats[b][0][[m.trainIdx for m in mm]]
            X = cv2.triangulatePoints(P[a], P[b], xa.T.astype(np.float64), xb.T.astype(np.float64))
            X = (X[:3] / X[3]).T
            ok = np.ones(len(X), bool)
            for j, x in ((a, xa), (b, xb)):
                w2c = np.linalg.inv(c2w[j]); Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
                pr = Xc @ K[j].T; uv = pr[:, :2] / pr[:, 2:3]
                ok &= (Xc[:, 2] > 0) & (np.linalg.norm(uv - x, axis=1) < max_err)
            ra = X - c2w[a][:3, 3]; rb = X - c2w[b][:3, 3]
            cos = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
            ok &= np.degrees(np.arccos(np.clip(cos, -1, 1))) > min_par
            xi = np.round(xa[ok]).astype(int)
            pts.append(X[ok]); cols.append(feats[a][2][xi[:, 1].clip(0, H - 1), xi[:, 0].clip(0, W - 1)])
    return np.concatenate(pts), np.concatenate(cols)


def cam_heights(pts, c2w, up=np.array([0, 0, 1.0]), r_min=0.0, r_max=None, n_iter=300, seed=0):
    """Height of each camera above a local road plane: RANSAC plane (normal within 15 deg of up) on the triangulated
    points below the camera and 0.5-3.5 camera heights ahead / around it (horizontal radius), per camera.
    Units: those of pts. Returns array (N,) (nan where no plane found)."""
    rng = np.random.default_rng(seed)
    out = []
    for T in c2w:
        C = T[:3, 3]; d = pts - C; hz = d @ up
        horiz = np.linalg.norm(d - hz[:, None] * up, axis=1)
        # coarse height guess: median of the lowest points nearby
        near = horiz < (r_max if r_max else 1e9)
        cand = pts[near & (hz < 0)]
        if len(cand) < 30:
            out.append(np.nan); continue
        h0 = -np.percentile((cand - C) @ up, 10)
        sel = near & (hz < -0.3 * h0) & (horiz > 0.5 * h0) & (horiz < 6 * h0)
        Q = pts[sel]
        if len(Q) < 30:
            out.append(np.nan); continue
        best, bn = None, 0
        tol = 0.03 * h0
        for _ in range(n_iter):
            s = Q[rng.choice(len(Q), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            if abs(n @ up) < np.cos(np.radians(15)):
                continue
            inl = np.abs((Q - s[0]) @ n) < tol
            if inl.sum() > bn:
                bn, best = inl.sum(), inl
        if best is None or bn < 20:
            out.append(np.nan); continue
        A = Q[best]; c = A.mean(0); n = np.linalg.svd(A - c)[2][-1]; n = n if n @ up > 0 else -n
        out.append(float((C - c) @ n))
    return np.array(out)


def prepare(seq, force=False):
    """Loads a sequence, builds masks, triangulates each vehicle's points, measures camera heights, and writes
    cache/<seq>/geom.json + pts_<v>.npz (points in METRIC z-up world). Returns the geometry dict."""
    gp = f"{cache_dir(seq)}/geom.json"
    if os.path.exists(gp) and not force:
        return json.load(open(gp))
    R = load_raw(seq)
    info = dict(seq=seq)
    pts_u = {}
    for v in PREFIX:
        ids = R[v]["ids"]; ego = ego_mask(seq, v, ids); make_dyn_masks(seq, v, ids)
        M = [load_mask(seq, v, k, ego) for k in ids]
        p, c = triangulate(seq, v, ids, R[v]["c2w"], R[v]["K"], M)
        pts_u[v] = (p, c)
        info[f"ego_frac_{v}"] = float(ego.mean())
    # up direction: mean of the cameras' -y axes (OpenCV y down), both vehicles
    up = -np.concatenate([R[v]["c2w"][:, :3, 1] for v in PREFIX]).mean(0); up /= np.linalg.norm(up)
    info["up_units"] = up.tolist()
    for v in PREFIX:
        h = cam_heights(pts_u[v][0], R[v]["c2w"], up)
        info[f"h_{v}_units"] = float(np.nanmedian(h)); info[f"h_{v}_valid"] = int(np.isfinite(h).sum())
        info[f"h_{v}_iqr_units"] = [float(np.nanpercentile(h, 25)), float(np.nanpercentile(h, 75))]
    s = CAR_CAM_H / info["h_car_units"]  # meters per MV2 unit
    info["m_per_unit"] = float(s)
    # rotation that maps 'up' to +z (MV2 is already ~z up; this removes the residual tilt)
    z = np.array([0, 0, 1.0]); ax = np.cross(up, z); sa = np.linalg.norm(ax); ca = up @ z
    Rz = np.eye(3) if sa < 1e-9 else cv2.Rodrigues(ax / sa * np.arctan2(sa, ca))[0]
    info["R_level"] = Rz.tolist()
    for v in PREFIX:
        p, c = pts_u[v]
        np.savez_compressed(f"{cache_dir(seq)}/pts_{v}.npz", xyz=(s * p @ Rz.T).astype(np.float32), rgb=c)
        info[f"n_pts_{v}"] = int(len(p))
    # time-synchronized offsets scooter - car, in the car's heading frame (meters)
    Cc = to_metric(R["car"]["c2w"], info)[:, :3, 3]; Cs = to_metric(R["scooty"]["c2w"], info)[:, :3, 3]
    ids_c, ids_s = R["car"]["ids"], R["scooty"]["ids"]; common = sorted(set(ids_c) & set(ids_s))
    Tc = to_metric(R["car"]["c2w"], info)
    lat, fwd, dz = [], [], []
    for k in common:
        i, j = ids_c.index(k), ids_s.index(k)
        f = Tc[i][:3, 2].copy(); f[2] = 0; f /= np.linalg.norm(f); left = np.cross([0, 0, 1.0], f)
        d = Cs[j] - Cc[i]; lat.append(d @ left); fwd.append(d @ f); dz.append(d[2])
    info.update(sync_lat_m=float(np.median(lat)), sync_fwd_m=float(np.median(fwd)), sync_dz_m=float(np.median(dz)),
                sync_lat_std_m=float(np.std(lat)), sync_fwd_std_m=float(np.std(fwd)),
                h_car_m=CAR_CAM_H, h_scooty_m=float(info["h_scooty_units"] * s),
                car_path_m=float(np.linalg.norm(np.diff(Cc, axis=0), axis=1).sum()),
                scooty_path_m=float(np.linalg.norm(np.diff(Cs, axis=0), axis=1).sum()))
    json.dump(info, open(gp, "w"), indent=1)
    return info


def to_metric(c2w, info):
    s = info["m_per_unit"]; Rz = np.array(info["R_level"])
    out = c2w.copy()
    out[:, :3, :3] = Rz @ c2w[:, :3, :3]; out[:, :3, 3] = s * c2w[:, :3, 3] @ Rz.T
    return out


def load_seq(seq):
    """Metric, levelled sequence: dict v -> dict(ids, c2w, K), plus geometry info."""
    info = prepare(seq); R = load_raw(seq)
    for v in PREFIX:
        R[v]["c2w"] = to_metric(R[v]["c2w"], info)
    return R, info


if __name__ == "__main__":
    # masks <seq...>  YOLO masks (GPU job);  geom <seq...>  triangulation + heights + scale (CPU; needs the masks)
    import sys
    mode, seqs = sys.argv[1], sys.argv[2:]
    for s in seqs:
        if mode == "masks":
            for v in PREFIX:
                make_dyn_masks(s, v, frame_ids(s, v))
            print(s, "masks done", flush=True)
        else:
            g = prepare(s, force=True)
            print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if not k.startswith("R_")}, flush=True)
