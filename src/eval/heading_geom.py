"""Pose conventions and the homographies used by the heading / camera-height axis (proposal step 1 and 6).

Conventions (Para-Lane COLMAP world frame; verified on scene_16/17: world z up, camera pitch 5.0-5.3 deg down,
roll < 0.7 deg, camera 1.86-1.91 m above the LiDAR ground):
  * camera frame = OpenCV (x right, y down, z forward); c2w = [R | C] maps camera to world.
  * yaw psi (deg): the camera is rotated about the WORLD z axis through its OWN centre, R' = Rz(psi) R, C' = C.
    psi > 0 = turn LEFT (counter-clockwise seen from above; world y points left).
  * dz (m): the camera centre is moved along world +z, R' = R, C' = C + dz e_z. dz > 0 = higher (truck-mounted),
    dz < 0 = lower. Nominal height h = C_z - ground_z (ground = 10th percentile of LiDAR z within 4 m, per frame).
  * condition name: 'y{psi:+d}_z{dz:+.1f}', e.g. y+10_z+0.0 or y+0_z-1.0; y+0_z+0.0 is the recorded pose.
  * "lateral + yaw": the real ADJACENT-lane pose (Para-Lane cross-lane pairs) with the same yaw rotation applied.

Warps from the recorded pose 0 (real frame, 0-deviation render) to the deviated pose 1:
  H = K [ R1^T R0 - R1^T (C0 - C1) n0^T / h ] K^-1,   n0 = R0^T e_z (world up in camera-0 coordinates)
  * pure yaw (C1 = C0): H = K R1^T R0 K^-1. This is EXACT for every scene point, whatever its depth, because the
    camera rotates about its optical centre (no parallax). Only pixels whose source falls outside the recorded frame are
    unknown (invalid). A real vehicle turns about its rear axle, which adds a lateral shift of L sin(psi) (not modelled
    here; that is what the lateral + yaw condition probes).
  * height change: the plane-induced homography of the road plane z = ground (normal world z). Exact for points ON the
    road (lane paint, road texture), wrong for anything above it (cars, poles, buildings) -> only road-region
    metrics are computed against the warped real frame for this axis.
"""
import numpy as np
import cv2


def cond_name(yaw, dz):
    return f"y{int(round(yaw)):+d}_z{dz:+.1f}"


def rotz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def deviate(c2w, yaw=0.0, dz=0.0):
    out = np.array(c2w, np.float64).copy()
    out[:3, :3] = rotz(yaw) @ out[:3, :3]
    out[2, 3] += dz
    return out


def homography(K, c2w0, c2w1, h=None):
    """Pixel map from the image at pose 0 to the image at pose 1 (see module doc). h = camera height above the road
    plane at pose 0 (needed only if the centres differ)."""
    K = np.asarray(K, np.float64); c0, c1 = np.asarray(c2w0, np.float64), np.asarray(c2w1, np.float64)
    R0, R1, C0, C1 = c0[:3, :3], c1[:3, :3], c0[:3, 3], c1[:3, 3]
    M = R1.T @ R0
    if np.linalg.norm(C1 - C0) > 1e-6:
        assert h is not None and h > 0.3, "plane homography needs the camera height"
        n0 = R0.T @ np.array([0.0, 0.0, 1.0])
        M = M - np.outer(R1.T @ (C0 - C1), n0) / h
    Hm = K @ M @ np.linalg.inv(K)
    return Hm / Hm[2, 2]


def warp_img(img, Hm, interp=cv2.INTER_LANCZOS4):
    """Warp an image of pose 0 into pose 1. Returns (warped, valid) with valid = the source pixel lies inside the frame.
    Lanczos-4, not bilinear: bilinear resampling blurs the pseudo-ground truth (a real frame warped there and back loses
    to 31.9 dB, Lanczos-4 to 41.9 dB on scene_16), which made renders look ~0.8 dB BETTER at +-5-10 deg yaw than at 0 deg
    on the same content; with Lanczos-4 the raw render scores the same at 0 and 5-10 deg (25.75 vs 25.78 dB;
    results/heading/checks.json)."""
    h, w = img.shape[:2]
    out = cv2.warpPerspective(img, Hm, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # drop interpolated border pixels
    return out, valid


def warp_pts(pts, Hm):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ Hm.T
    ok = q[:, 2] > 1e-6
    return q[:, :2] / np.where(ok, q[:, 2], 1)[:, None], ok


def warp_lanes(lanes, Hm, W, H):
    out = []
    for l in lanes:
        q, ok = warp_pts(l, Hm)
        ok &= (q[:, 0] > -W) & (q[:, 0] < 2 * W) & (q[:, 1] > -H) & (q[:, 1] < 2 * H)
        if ok.sum() >= 2:
            out.append(q[ok].tolist())
    return out


def warp_boxes(b, Hm, W, H):
    """Axis-aligned box around the warped corners (exact corners for a pure rotation, box re-fitted)."""
    if len(b) == 0:
        return b.reshape(0, 4)
    out = []
    for x1, y1, x2, y2 in b:
        q, _ = warp_pts([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], Hm)
        out.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
    out = np.array(out)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, W); out[:, [1, 3]] = out[:, [1, 3]].clip(0, H)
    return out


def camera_heights(c2w, lidar, radius=4.0, pct=10):
    """Per-frame camera height above the LiDAR ground (m); falls back to the clip median where the cloud is sparse."""
    from scipy.spatial import cKDTree
    C = np.asarray(c2w)[:, :3, 3]; tr = cKDTree(lidar[:, :2]); hs = np.full(len(C), np.nan)
    for i, c in enumerate(C):
        idx = tr.query_ball_point(c[:2], radius)
        if len(idx) > 20:
            hs[i] = c[2] - np.percentile(lidar[idx, 2], pct)
    med = np.nanmedian(hs)
    hs[~np.isfinite(hs) | (np.abs(hs - med) > 0.5)] = med
    return hs
