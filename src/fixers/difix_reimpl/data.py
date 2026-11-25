"""Data pipeline for our Difix re-implementation.

`build_pairs_json` turns our Para-Lane pair metadata (pairs/<scene>/meta.json) into a JSON in the layout the
official Difix trainer also reads ({"train": {id: {image, target_image, ref_image, prompt, [fg]}}, "test": {...}}),
so both trainers see exactly the same (render, target, reference) triplets.
Reference view: the REAL frame of the SOURCE lane (the pass the splat was built from, i.e. what a simulator has)
nearest to the target pose; for on-path pairs (src == dst) the target frame itself is excluded, so the reference is
the neighbouring recorded frame. `PairDataset` loads a triplet, resizes to a fixed size, returns tensors in [-1, 1].
"""
import json, os, random, sys
import numpy as np
import torch
from PIL import Image

PROMPT = "remove degradation"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # src/
from common import config  # noqa: E402
PL_ROOTS = config.PARALANE_ROOTS


def _cam_centers(scene, clip):
    """{timestamp: camera centre} from the clip's COLMAP images.txt (world-to-camera quaternion + translation)."""
    from scipy.spatial.transform import Rotation
    for r in PL_ROOTS:
        p = f"{r}/{scene}/{clip}/sparse/0/images.txt"
        if os.path.exists(p):
            break
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")][::2]
    for l in lines:
        q = l.split(); qw, qx, qy, qz = map(float, q[1:5]); t = np.array(list(map(float, q[5:8])))
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        out[q[9].split("/")[0]] = -R.T @ t
    return out


def ref_path(pairs_root, m, cache):
    """REAL frame of the source lane nearest to the target pose (never the target frame itself)."""
    for c in (m["src"], m["dst"]):
        if (m["scene"], c) not in cache:
            cache[(m["scene"], c)] = _cam_centers(m["scene"], c)
    src, dst = cache[(m["scene"], m["src"])], cache[(m["scene"], m["dst"])]
    ts = sorted(src); P = np.stack([src[t] for t in ts]); d = np.linalg.norm(P - dst[m["ts"]], axis=1)
    if m["src"] == m["dst"]:
        d[ts.index(m["ts"])] = 1e9
    return f"{pairs_root}/{m['scene']}/real/{m['src']}/{ts[int(d.argmin())]}.png"


def build_pairs_json(pairs_root, train_scenes, test_scenes, out, test_stride=6,
                     variants=("full", "underfit", "cycle", "identity")):
    cache = {}
    ref_for = lambda m: ref_path(pairs_root, m, cache)

    J = {"train": {}, "test": {}}
    for split, scenes in (("train", train_scenes), ("test", test_scenes)):
        for s in scenes:
            meta = json.load(open(f"{pairs_root}/{s}/meta.json"))
            if split == "train":
                meta = [m for m in meta if m["variant"] in variants and not m["onpath_test"]]
            else:
                meta = [m for k, m in enumerate([m for m in meta if m["variant"] == "full"]) if m["onpath_test"] or k % test_stride == 0]
            for m in meta:
                key = m["path"].replace("/", "__")[:-4]
                J[split][key] = dict(image=f"{pairs_root}/{m['path']}", target_image=f"{pairs_root}/{m['real']}",
                                     ref_image=ref_for(m), prompt=PROMPT, fg=f"{pairs_root}/{m['fg']}",
                                     variant=m["variant"], offset_m=m["offset_m"])
    json.dump(J, open(out, "w"))
    return {k: len(v) for k, v in J.items()}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, split, size=(576, 1024), use_ref=True, use_fg=False, hflip=True):
        self.items = list(json.load(open(json_path))[split].values())
        self.size, self.use_ref, self.use_fg, self.hflip = size, use_ref, use_fg, hflip and split == "train"

    def __len__(self):
        return len(self.items)

    def _load(self, p, mode="RGB"):
        im = Image.open(p).convert(mode).resize(self.size[::-1], Image.BICUBIC if mode == "RGB" else Image.NEAREST)
        a = torch.from_numpy(np.array(im)).float()
        return (a.permute(2, 0, 1) / 127.5 - 1) if mode == "RGB" else (a[None] > 127).float()

    def __getitem__(self, k):
        it = self.items[k]
        views = [self._load(it["image"])] + ([self._load(it["ref_image"])] if self.use_ref else [])
        x, y = torch.stack(views), self._load(it["target_image"])
        valid = (1 - self._load(it["fg"], "L")) if (self.use_fg and "fg" in it) else torch.ones(1, *self.size)
        if self.hflip and random.random() < 0.5:
            x, y, valid = x.flip(-1), y.flip(-1), valid.flip(-1)
        return x, y, valid
