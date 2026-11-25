"""Screen Para-Lane scenes for road type (stage 18): per clip the path length, total and max heading change
(curvature), and per scene the lateral gaps between the three passes, the fraction of frames with movable objects
and their mean mask area (traffic proxy). Also writes a contact sheet (6 frames of clip_1) per scene for a visual check
(intersection / crosswalk / curve) to $CS566D_DATA/screen/<scene>.jpg.
usage: scripts/lib/py.sh src/eval/screen_scenes.py <out.json> <scene> [...]
"""
import json, os, sys
import numpy as np
from PIL import Image
from splats.paralane import clip_dir, load_cameras, load_fg
from common.config import DATA


def main():
    out = sys.argv[1]; res = json.load(open(out)) if os.path.exists(out) else {}
    os.makedirs(f"{DATA}/screen", exist_ok=True)
    for s in sys.argv[2:]:
        r = dict(clips={})
        P = {}
        for c in ("clip_0", "clip_1", "clip_2"):
            d = clip_dir(s, c); K, ts, c2w = load_cameras(d)
            C = c2w[:, :3, 3]; f = c2w[:, :3, 2]; yaw = np.unwrap(np.arctan2(f[:, 1], f[:, 0]))
            seg = np.linalg.norm(np.diff(C, axis=0), axis=1); L = float(seg.sum())
            # heading change per 10 m of travel (max over the pass) = local curvature
            cum = np.concatenate([[0], np.cumsum(seg)]); k = []
            for i in range(len(C)):
                j = np.searchsorted(cum, cum[i] + 10)
                if j < len(C):
                    k.append(abs(np.degrees(yaw[j] - yaw[i])))
            fgs = [load_fg(d, t).mean() for t in ts[::5]]
            r["clips"][c] = dict(n=len(ts), length_m=L, heading_change_deg=float(np.degrees(yaw[-1] - yaw[0])),
                                 max_turn_deg_per_10m=float(max(k) if k else 0), dz_m=float(C[-1, 2] - C[0, 2]),
                                 frac_frames_movable=float(np.mean(np.array(fgs) > 0.002)), mean_movable_area=float(np.mean(fgs)))
            P[c] = (C, f)
        gaps = {}
        for a, b in (("clip_0", "clip_1"), ("clip_1", "clip_2"), ("clip_0", "clip_2")):
            A, fa = P[a]; B, _ = P[b]
            d = np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=2); j = d.argmin(1)
            # signed lateral distance: B's nearest point relative to A, projected on A's left direction
            left = np.stack([-fa[:, 1], fa[:, 0]], 1); left /= np.linalg.norm(left, axis=1, keepdims=True)
            lat = ((B[j, :2] - A[:, :2]) * left).sum(1)
            gaps[f"{a}-{b}"] = dict(median_m=float(np.median(lat)), p10=float(np.percentile(lat, 10)), p90=float(np.percentile(lat, 90)))
        r["gaps"] = gaps
        d = clip_dir(s, "clip_1"); _, ts, _ = load_cameras(d)
        ims = [Image.open(f"{d}/images/{ts[i]}/CAMERA_FRONT.png").convert("RGB").resize((480, 270)) for i in np.linspace(0, len(ts) - 1, 6).astype(int)]
        sheet = Image.new("RGB", (480 * 3, 270 * 2))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 3) * 480, (i // 3) * 270))
        sheet.save(f"{DATA}/screen/{s}.jpg", quality=80)
        res[s] = r
        print(s, json.dumps({c: {k: round(v, 2) for k, v in x.items()} for c, x in r["clips"].items()}),
              {k: round(v["median_m"], 2) for k, v in gaps.items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
