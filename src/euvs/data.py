"""EUVS [6] scene loader: COLMAP binary model (cameras.bin / images.bin / points3D.bin), train/test lists, images,
movable-object masks. Images are resized to a working size of width 960 (height rounded to a multiple of 32; the
pinhole K is scaled per axis, so the resize is exact).

EUVS scene layout (as downloaded by src/euvs/download.py):
  <scene>/<inner>/{images/*.jpg, sparse/0/*.bin, train_set.txt, test_set.txt, geo_registration/, poses/}
  MARS (Setting3) names: trav_<T>_channel_<C>_img_<N>.jpg ; the test set is a held-out TRAVERSAL (another route
  through the same place), all channels.
Dynamic masks: EUVS masks movable objects with Grounded-SAM-2 (not shipped); we use YOLO11m-seg instance masks
(person, bicycle, car, motorcycle, bus, truck, train), dilated 9 px like the Para-Lane masks, cached as PNGs.
"""
import glob, os, struct
import numpy as np
from PIL import Image

MODEL_PARAMS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
                5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}
WORK_W = 960


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(p):
    cams = {}
    with open(p, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, model, w, h = _read(f, "iiQQ")
            name, n = MODEL_PARAMS[model]
            cams[cid] = dict(model=name, w=w, h=h, params=np.array(_read(f, "d" * n)))
    return cams


def read_images_bin(p):
    ims = {}
    with open(p, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid = _read(f, "i")[0]; q = np.array(_read(f, "dddd")); t = np.array(_read(f, "ddd")); cid = _read(f, "i")[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = _read(f, "Q")[0]; f.seek(24 * n2d, 1)  # skip 2D points (x, y, point3D id)
            ims[name.decode()] = dict(id=iid, qvec=q, tvec=t, cam=cid)
    return ims


def read_points_bin(p):
    xyz, rgb = [], []
    with open(p, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            _read(f, "Q"); xyz.append(_read(f, "ddd")); rgb.append(_read(f, "BBB")); _read(f, "d")
            n = _read(f, "Q")[0]; f.seek(8 * n, 1)
    return np.array(xyz, np.float32), np.array(rgb, np.uint8)


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def scene_root(d):
    """Directory that holds sparse/0 (the archives add one or two wrapper folders)."""
    hits = glob.glob(f"{d}/**/sparse/0/images.bin", recursive=True)
    assert hits, f"no sparse/0/images.bin under {d}"
    return os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))


def working_size(w, h):
    s = WORK_W / w
    return WORK_W, int(round(h * s / 32) * 32)


def load_scene(d):
    """Returns dict(root, names, split{name: train|test}, K{name}, c2w{name}, size (W,H), points, colors, orig)."""
    root = scene_root(d)
    cams = read_cameras_bin(f"{root}/sparse/0/cameras.bin"); ims = read_images_bin(f"{root}/sparse/0/images.bin")
    split = {}
    for s in ("train", "test"):
        for l in open(f"{root}/{s}_set.txt"):
            if l.strip():
                split[l.strip()] = s
    names = sorted(n for n in ims if n in split and os.path.exists(f"{root}/images/{n}"))
    K, c2w, sizes = {}, {}, set()
    for n in names:
        im = ims[n]; c = cams[im["cam"]]; W, H = working_size(c["w"], c["h"]); sizes.add((W, H))
        p = c["params"]
        if c["model"] == "SIMPLE_PINHOLE" or c["model"].startswith("SIMPLE_RADIAL") or c["model"] == "RADIAL":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        assert c["model"] in ("PINHOLE", "SIMPLE_PINHOLE"), f"distorted camera model {c['model']} not supported"
        sx, sy = W / c["w"], H / c["h"]
        K[n] = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
        R = qvec2rot(im["qvec"]); w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = im["tvec"]
        c2w[n] = np.linalg.inv(w2c).astype(np.float32)
    assert len(sizes) == 1, f"mixed image sizes {sizes}"
    pts, rgb = read_points_bin(f"{root}/sparse/0/points3D.bin")
    return dict(root=root, names=names, split=split, K=K, c2w=c2w, size=sizes.pop(), points=pts, colors=rgb)


def load_image(sc, n):
    W, H = sc["size"]
    return np.array(Image.open(f"{sc['root']}/images/{n}").convert("RGB").resize((W, H), Image.BICUBIC))


def dyn_mask_path(sc, n):
    return f"{sc['root']}/dyn_masks_yolo/{os.path.splitext(n)[0]}.png"


def make_dyn_masks(sc, weights, batch=16):
    """YOLO11m-seg movable-object masks for every image (cached). Classes: person, bicycle, car, motorcycle, bus,
    train, truck (COCO 0,1,2,3,5,6,7), conf 0.25, dilated 9 px."""
    import cv2
    from ultralytics import YOLO
    todo = [n for n in sc["names"] if not os.path.exists(dyn_mask_path(sc, n))]
    if not todo:
        return
    os.makedirs(os.path.dirname(dyn_mask_path(sc, todo[0])), exist_ok=True)
    y = YOLO(weights); W, H = sc["size"]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = y([load_image(sc, n)[..., ::-1].copy() for n in chunk], verbose=False, conf=0.25, classes=[0, 1, 2, 3, 5, 6, 7],
                retina_masks=True)
        for n, r in zip(chunk, res):
            m = np.zeros((H, W), np.uint8)
            if r.masks is not None:
                m = (r.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = cv2.dilate(m, np.ones((9, 9), np.uint8))
            Image.fromarray(m * 255).save(dyn_mask_path(sc, n))


def load_dyn(sc, n):
    return np.array(Image.open(dyn_mask_path(sc, n))) > 127


def rel_pose_stats(sc):
    """For every test view: distance (m, in the scene's metric frame) and rotation angle (deg) to the nearest TRAIN
    camera center, and the smallest rotation angle to any train camera within 5 m (how much 'heading' change)."""
    tr = [n for n in sc["names"] if sc["split"][n] == "train"]; te = [n for n in sc["names"] if sc["split"][n] == "test"]
    Ct = np.stack([sc["c2w"][n][:3, 3] for n in tr]); Rt = np.stack([sc["c2w"][n][:3, :3] for n in tr])
    out = {}
    for n in te:
        C = sc["c2w"][n][:3, 3]; R = sc["c2w"][n][:3, :3]
        d = np.linalg.norm(Ct - C, axis=1)
        ang = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rt, R) - 1) / 2, -1, 1)))
        # forward-axis angle ~ heading change for level cameras
        fang = np.degrees(np.arccos(np.clip(Rt[:, :, 2] @ R[:, 2], -1, 1)))
        j = int(d.argmin()); near = d < 5
        out[n] = dict(dist=float(d[j]), rot_nearest=float(ang[j]), fwd_nearest=float(fang[j]),
                      min_fwd_within5m=float(fang[near].min()) if near.any() else np.nan)
    return out


def align_metric(sc):
    """Similarity (Umeyama) from the COLMAP frame to the scene's geo_registration positions (metric, local frame with
    constant height for MARS), applied in place to poses and points -> meters, and the ground plane horizontal.
    The up direction is then fixed so that cameras' -y axes point to +z. Returns the scale (m per SfM unit)."""
    g = {}
    for l in open(f"{sc['root']}/geo_registration/geo_registration.txt"):
        p = l.split()
        if len(p) >= 4:
            g[p[0]] = np.array(list(map(float, p[1:4])))
    n = [k for k in sc["names"] if k in g]
    A = np.stack([sc["c2w"][k][:3, 3] for k in n]).astype(np.float64); B = np.stack([g[k] for k in n])
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(n))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()
    t = mb - s * R @ ma
    resid = np.linalg.norm(s * A @ R.T + t - B, axis=1)
    up = np.mean([-(R @ sc["c2w"][k][:3, 1]) for k in n], 0)
    if up[2] < 0:  # handedness of the geo frame differs: rotate 180 deg about x (still metric, now z-up)
        F = np.diag([1.0, -1.0, -1.0]); R = F @ R; t = F @ t
    for k in sc["names"]:
        c = sc["c2w"][k].astype(np.float64); c2 = np.eye(4)
        c2[:3, :3] = R @ c[:3, :3]; c2[:3, 3] = s * R @ c[:3, 3] + t
        sc["c2w"][k] = c2.astype(np.float32)
    sc["points"] = (s * sc["points"].astype(np.float64) @ R.T + t).astype(np.float32)
    sc["align"] = dict(scale=float(s), resid_med_m=float(np.median(resid)), flipped=bool(up[2] < 0))
    return float(s)
