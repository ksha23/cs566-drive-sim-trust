"""Pure-PyTorch reference rasterizer: no gsplat at all. Slow (Python loop over depth chunks), so it is only for
correctness checks on small crops (e.g. 64x64); see src/splats/own_render_check.py and docs/own_renderer.md.

Pipeline: our projection / SH colour from `splats.own_render` -> tile binning written here (same tile bounds as
gsplat's isect_tiles) -> one global sort by (tile, depth) -> per-tile padded index table -> front-to-back alpha
compositing in chunks of K Gaussians, all in autograd. Compositing follows gsplat's rasterize_to_pixels exactly:
  sigma = 1/2 d^T Conic d (d = centre - pixel centre), alpha = min(0.999, o exp(-sigma)),
  skip if sigma < 0 or alpha < 1/255, stop the pixel (exclusive) once T (1 - alpha) <= 1e-4.
Ported from our proposal-C prototype (src/gslam/torchgs_ref.py), adapted to D's parameterisation and gsplat 1.4.
"""
import math
import torch
import torch.nn.functional as F

from splats.own_render import project, eval_sh, camera_dirs, TILE

ALPHA_MIN = 1.0 / 255.0
T_MIN = 1e-4


@torch.no_grad()
def bin_tiles(mean2d, radius, depth, width, height, ts=TILE):
    """For M projected Gaussians: table (n_tiles, Kmax) of Gaussian indices per tile sorted front to back
    (-1 = padding) and the count per tile."""
    tw, th = math.ceil(width / ts), math.ceil(height / ts)
    r = radius.float() / ts
    tx, ty = mean2d[:, 0] / ts, mean2d[:, 1] / ts
    x0 = (tx - r).floor().clamp(0, tw).long(); x1 = (tx + r).ceil().clamp(0, tw).long()
    y0 = (ty - r).floor().clamp(0, th).long(); y1 = (ty + r).ceil().clamp(0, th).long()
    nx, cnt = x1 - x0, (x1 - x0) * (y1 - y0)
    g = torch.repeat_interleave(torch.arange(len(cnt), device=mean2d.device), cnt)   # Gaussian of each (g, tile)
    local = torch.arange(len(g), device=g.device) - (torch.cumsum(cnt, 0) - cnt)[g]
    tile = (y0[g] + local // nx[g]) * tw + x0[g] + local % nx[g]
    # sort by depth, then (stable) by tile -> within a tile, front to back; ties keep the Gaussian order
    o = torch.argsort(depth[g], stable=True); tile, g = tile[o], g[o]
    o = torch.argsort(tile, stable=True); tile, g = tile[o], g[o]
    count = torch.bincount(tile, minlength=tw * th)
    pos = torch.arange(len(tile), device=g.device) - (torch.cumsum(count, 0) - count)[tile]
    table = torch.full((tw * th, max(int(count.max()), 1) if len(tile) else 1), -1, dtype=torch.long, device=g.device)
    table[tile, pos] = g
    return table, count, tw, th


def composite(mean2d, conic, feats, opac, table, count, tw, th, width, height, ts=TILE, chunk=64):
    """Front-to-back alpha compositing. feats (M,D). Returns image (H,W,D) and alpha (H,W)."""
    dev, (M, D) = mean2d.device, feats.shape
    # an extra "empty" Gaussian at index M (zero opacity) fills the padding of the table
    mean2d = torch.cat([mean2d, mean2d.new_zeros(1, 2)]); conic = torch.cat([conic, conic.new_ones(1, 3)])
    feats = torch.cat([feats, feats.new_zeros(1, D)]); opac = torch.cat([opac, opac.new_zeros(1)])
    table = torch.where(table < 0, torch.full_like(table, M), table)
    n_tiles, P = tw * th, ts * ts
    py, px = torch.meshgrid(torch.arange(ts, device=dev), torch.arange(ts, device=dev), indexing="ij")
    tyi, txi = torch.arange(n_tiles, device=dev) // tw, torch.arange(n_tiles, device=dev) % tw
    pix = torch.stack([(txi[:, None] * ts + px.reshape(1, -1)).float() + 0.5,
                       (tyi[:, None] * ts + py.reshape(1, -1)).float() + 0.5], -1)       # (tiles, P, 2) centres
    out = feats.new_zeros(n_tiles, P, D)
    T = feats.new_ones(n_tiles, P)                                   # transmittance so far
    done = torch.zeros(n_tiles, P, dtype=torch.bool, device=dev)     # pixel reached T <= 1e-4 (gsplat stops it)
    for k0 in range(0, table.shape[1], chunk):
        act = torch.nonzero((count > k0) & ~done.all(1)).squeeze(1)  # tiles with Gaussians left and live pixels
        if len(act) == 0:
            break
        g = table[act, k0:k0 + chunk]                                # (A, C) Gaussian ids, front to back
        d = mean2d[g][:, None] - pix[act][:, :, None]                # (A, P, C, 2)
        cn = conic[g][:, None]                                       # (A, 1, C, 3)
        sigma = 0.5 * (cn[..., 0] * d[..., 0] ** 2 + cn[..., 2] * d[..., 1] ** 2) + cn[..., 1] * d[..., 0] * d[..., 1]
        al = torch.clamp(opac[g][:, None] * torch.exp(-sigma), max=0.999)
        al = torch.where((sigma < 0) | (al < ALPHA_MIN) | done[act][..., None], torch.zeros_like(al), al)
        T0 = T[act]
        # transmittance in front of each Gaussian of the chunk: T0 * prod_{j<i} (1 - alpha_j)
        logT = torch.cumsum(torch.log1p(-al), -1)
        T_before = T0[..., None] * torch.exp(torch.cat([torch.zeros_like(logT[..., :1]), logT[..., :-1]], -1))
        with torch.no_grad():  # gsplat: a Gaussian that would bring T to <= 1e-4 is not added, and ends the pixel
            stop = (T_before * (1 - al) <= T_MIN) & (al > 0)
            use = torch.cumsum(stop.int(), -1) == 0
        w = al * T_before * use                                      # blending weights (A, P, C)
        out = out.index_add(0, act, torch.einsum("apc,acd->apd", w, feats[g]))
        T = T.index_copy(0, act, T0 * torch.exp(torch.sum(torch.log1p(-al * use), -1)))
        done = done.index_copy(0, act, done[act] | stop.any(-1))
    img = out.reshape(th, tw, ts, ts, D).permute(0, 2, 1, 3, 4).reshape(th * ts, tw * ts, D)[:height, :width]
    alpha = (1 - T).reshape(th, tw, ts, ts).permute(0, 2, 1, 3).reshape(th * ts, tw * ts)[:height, :width]
    return img, alpha


def render(params, sky, c2w, K, width, height, sh_degree=3, cam_dirs=None, with_depth=True, chunk=64):
    """Same outputs as splats.own_render.render (rgb, alpha, depth, info) for a width x height image (small!)."""
    viewmat = torch.linalg.inv(c2w)
    pr = project(params["means"], params["quats"], params["scales"], viewmat, K, width, height)
    idx = pr["idx"]
    coeffs = torch.cat([params["sh0"][idx], params["shN"][idx]], 1)
    rgb_g = torch.clamp_min(eval_sh(sh_degree, params["means"][idx] - c2w[:3, 3], coeffs) + 0.5, 0.0)
    feats = torch.cat([rgb_g, pr["depth"][:, None]], -1)
    opac = torch.sigmoid(params["opacities"][idx])
    table, count, tw, th = bin_tiles(pr["mean2d"].detach(), pr["radius"], pr["depth"].detach(), width, height)
    img, alpha = composite(pr["mean2d"], pr["conic"], feats, opac, table, count, tw, th, width, height, chunk=chunk)
    rgb = img[..., :3]
    depth = img[..., 3] / alpha.clamp(min=1e-10) if with_depth else None
    if sky is not None:
        if cam_dirs is None:
            cam_dirs = camera_dirs(K, width, height)
        rgb = rgb + (1 - alpha[..., None]) * sky(F.normalize(cam_dirs @ c2w[:3, :3].T, dim=-1))
    return rgb.clamp(0, 1), alpha, depth, dict(n_visible=int(idx.numel()), kmax=table.shape[1])
