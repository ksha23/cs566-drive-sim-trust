"""Figure for docs/own_renderer.md: real | gsplat | ours | |ours - gsplat| x1000 (float renders, max over RGB) for one
~3 m frame of scene_16 (clip_0 splat at clip_1's pose), plus the renders of the own-renderer refit at the same pose.
usage: scripts/lib/py.sh src/splats/own_render_figure.py <out.jpg>"""
import json, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

from common.config import DATA
from splats.paralane import clip_dir, load_cameras
from splats.gs import load_splat, pixel_dirs, render as gs_render
from splats.own_render import render as own_render

S = "scene_16"
M = [m for m in json.load(open(f"{DATA}/pairs/{S}/meta.json")) if m["variant"] == "full" and m["src"] == "clip_0"
     and 1 < abs(m["offset_m"]) < 4.5]
m = M[len(M) // 2]
K, ts, c2w = load_cameras(clip_dir(S, m["dst"])); K = torch.tensor(K).cuda(); c2w = torch.tensor(c2w[ts.index(m["ts"])]).cuda()
real = torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['real']}"))).cuda().float() / 255
tiles = [("real image (lane clip_1)", real)]
with torch.no_grad():
    p, sky = load_splat(f"{DATA}/splats/{S}/clip_0/splat.pt")
    g = gs_render(p, sky, c2w, K, cam_dirs=pixel_dirs(K))[0]; o = own_render(p, sky, c2w, K, cam_dirs=pixel_dirs(K))[0]
    d = (o - g).abs().max(-1).values
    tiles += [("gsplat fit, gsplat render", g), ("gsplat fit, OUR render", o),
              (f"|ours - gsplat| x1000 (max {float(d.max()):.3f}, {float((d > 1 / 255).float().mean()) * 100:.3f}% > 1/255)",
               (d * 1000).clamp(0, 1)[..., None].expand(-1, -1, 3))]
    p2, sky2 = load_splat(f"{DATA}/own_renderer/splats/{S}/clip_0/splat.pt")
    tiles += [("refit with OUR renderer, our render", own_render(p2, sky2, c2w, K, cam_dirs=pixel_dirs(K))[0])]
    p3, sky3 = load_splat(f"{DATA}/own_renderer/splats_gsplat_s1/{S}/clip_0/splat.pt")
    tiles += [("gsplat refit seed 1 (noise ref.)", gs_render(p3, sky3, c2w, K, cam_dirs=pixel_dirs(K))[0])]
ims = []
for lab, t in tiles:
    im = Image.fromarray((t.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).resize((480, 270))
    dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 480, 16], fill=(0, 0, 0)); dr.text((4, 2), lab, fill=(255, 255, 255))
    ims.append(np.array(im))
grid = np.concatenate([np.concatenate(ims[i:i + 3], 1) for i in (0, 3)], 0)
Image.fromarray(grid).save(sys.argv[1], quality=88)
print("saved", sys.argv[1], m["dst"], m["ts"], m["offset_m"])
