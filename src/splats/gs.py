"""Gaussian-splat street model: gsplat Gaussians + a learnable sky environment map
(direction -> color), composited behind the splat with (1 - alpha)."""
import math
import torch
import torch.nn.functional as F
from gsplat import rasterization

from splats.paralane import W, H


class SkyEnv(torch.nn.Module):
    """Equirectangular sky texture indexed by world ray direction (world z up)."""

    def __init__(self, h=32, w=128, init=0.6):
        super().__init__()
        self.tex = torch.nn.Parameter(torch.full((1, 3, h, w), math.log(init / (1 - init))))

    def forward(self, dirs):  # dirs (H,W,3) unit world directions
        az = torch.atan2(dirs[..., 1], dirs[..., 0]) / math.pi
        el = torch.asin(dirs[..., 2].clamp(-1, 1)) / (math.pi / 2)
        grid = torch.stack([az, -el], -1)[None]
        return torch.sigmoid(F.grid_sample(self.tex, grid, align_corners=False, padding_mode="border"))[0].permute(1, 2, 0)


def pixel_dirs(K, device="cuda"):
    v, u = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5, indexing="ij")
    d = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
    return d  # camera frame, not normalized


def render(params, sky, c2w, K, sh_degree=3, cam_dirs=None, with_depth=False):
    """Returns rgb (H,W,3) in [0,1], alpha (H,W), depth (H,W) or None, gsplat info."""
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, torch.linalg.inv(c2w)[None], K[None], W, H,
        sh_degree=sh_degree, packed=False, render_mode="RGB+ED" if with_depth else "RGB")
    out, alpha = out[0], alpha[0, ..., 0]
    rgb = out[..., :3]
    if sky is not None:
        if cam_dirs is None:
            cam_dirs = pixel_dirs(K, c2w.device)
        dw = F.normalize(cam_dirs @ c2w[:3, :3].T, dim=-1)
        rgb = rgb + (1 - alpha[..., None]) * sky(dw)
    depth = out[..., 3] if with_depth else None
    return rgb.clamp(0, 1), alpha, depth, info


def load_splat(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    params = {k: v.to(device) for k, v in ck["params"].items()}
    sky = SkyEnv().to(device); sky.load_state_dict(ck["sky"])
    return params, sky
