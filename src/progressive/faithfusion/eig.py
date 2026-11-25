"""FaithFusion's pixel-wise Expected Information Gain (EIG) for one of our gsplat street splats.

We call FaithFusion's own modified rasterizer (github.com/wangyuanbiubiubiu/FaithFusion, `diff/`, built into
$CS566D_DATA/faithfusion/env by scripts/14_faithfusion.sh), whose backward pass accumulates *squared* per-pixel gradients (atomicAddSquare),
i.e. a diagonal Fisher approximation as in FisherRF. The procedure mirrors their code
(models/trainers/base.py:render_uncertainly_gaussians and models/video_utils.py):

  1. H_train: for every training view of the source lane, render with SH colours and back-propagate
     grad = non_sky / hit_count (per pixel, all 3 channels); sum the squared gradients over the parameter groups
     (means, rotations, scales, opacities, SH) into one scalar per Gaussian.  I_train = 1 / (H_train + 1e-6).
  2. For a novel view: back-propagate ones -> H_view (per Gaussian); I_acq = H_view * I_train.
  3. Render I_acq as a per-Gaussian colour (alpha-blended), mask pixels with depth <= 0.1, divide by the per-pixel
     hit count and by 1000, then gain = log(1 + gain) (their render_novel_views_uncertainty); pixels the splat does
     not cover (alpha < 0.1) get gain = 100 ("fully generate").

Differences from their pipeline (documented in docs/faithfusion.md): our Gaussians come from gsplat (same
parameterisation: wxyz quaternions, log-scales, logit-opacities, degree-3 SH; their EIG rasteriser evaluates SH at
degree 1 as in their code), the sky mask for step 1 is our splat's alpha < 0.5 (they use a Depth-Anything sky mask),
and the coverage alpha in step 3 comes from our gsplat render of the same Gaussians.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from modified_diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

PARAMS = ["means", "rotations", "scales", "opacities", "shs"]


def _proj(znear, zfar, fovx, fovy, cx, cy):
    # FaithFusion origin_gs_renderer.getProjectionMatrix (principal-point offsets cx, cy in NDC)
    th, tw = math.tan(fovy / 2), math.tan(fovx / 2)
    top, right = th * znear, tw * znear
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (2 * right); P[1, 1] = 2.0 * znear / (2 * top)
    P[0, 2] = cx; P[1, 2] = cy; P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class EIG:
    def __init__(self, splat_params, K, W, H):
        p = splat_params
        self.leaf = {
            "means": p["means"].detach().clone().requires_grad_(True),
            "rotations": F.normalize(p["quats"].detach(), dim=-1).clone().requires_grad_(True),
            "scales": torch.exp(p["scales"].detach()).clone().requires_grad_(True),
            "opacities": torch.sigmoid(p["opacities"].detach())[:, None].clone().requires_grad_(True),
            "shs": torch.cat([p["sh0"], p["shN"]], 1).detach().clone().requires_grad_(True),
        }
        self.K, self.W, self.H = K, W, H
        self.H_train = torch.zeros(self.leaf["means"].shape[0], device="cuda")
        self.I_train = None

    def _raster(self, c2w, colors=None):
        K, W, H = self.K, self.W, self.H
        fovx, fovy = 2 * math.atan(W / (2 * float(K[0, 0]))), 2 * math.atan(H / (2 * float(K[1, 1])))
        cx = (float(K[0, 2]) - W / 2) / W * 2; cy = (float(K[1, 2]) - H / 2) / H * 2
        wv = torch.linalg.inv(c2w).transpose(0, 1)
        P = _proj(0.01, 1000.0, fovx, fovy, cx, cy).transpose(0, 1).cuda()
        full = (wv[None] @ P[None])[0]
        rs = GaussianRasterizationSettings(image_height=H, image_width=W, tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
                                           bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=wv, projmatrix=full,
                                           sh_degree=1, campos=wv.inverse()[3, :3], prefiltered=False, debug=False)
        L = self.leaf
        m2d = torch.zeros_like(L["means"], requires_grad=True)
        return GaussianRasterizer(raster_settings=rs)(
            means3D=L["means"], means2D=m2d, shs=None if colors is not None else L["shs"], colors_precomp=colors,
            opacities=L["opacities"], scales=L["scales"], rotations=L["rotations"], cov3D_precomp=None)

    def _grad_sum(self):
        h = None
        for k in PARAMS:
            g = self.leaf[k].grad
            if g is None:
                continue
            s = reduce(g.detach(), "n ... -> n", "sum")
            h = s if h is None else h + s
            self.leaf[k].grad = None
        return h

    def add_train_view(self, c2w, nonsky):
        """nonsky: (H,W) bool/float, 1 where the pixel is not sky."""
        rgb, depth, radii, hit = self._raster(c2w)
        g = torch.ones_like(rgb) * nonsky.float()[None] / torch.clamp(hit.float(), min=1e-6)[None]
        rgb.backward(gradient=g)
        self.H_train += self._grad_sum()

    def finalize(self):
        self.I_train = torch.reciprocal(self.H_train + 1e-6)

    def gain(self, c2w, alpha):
        """alpha: (H,W) coverage of the splat at this pose (from our gsplat render). Returns log-EIG (H,W);
        uncovered pixels (alpha < 0.1) = 100."""
        rgb, depth, radii, hit = self._raster(c2w)
        rgb.backward(gradient=torch.ones_like(rgb))
        H_view = self._grad_sum()
        I_acq = H_view * self.I_train
        col = torch.stack([self.H_train, torch.sqrt(self.I_train), I_acq], 1).detach()
        with torch.no_grad():
            unc, depth2, _, hit2 = self._raster(c2w, colors=col)
        mask = (depth2.reshape(self.H, self.W) > 0.1).float() if depth2.numel() == self.H * self.W else 1.0
        g = unc[2] * mask / torch.clamp(hit2.float().reshape(self.H, self.W), min=1e-6) / 1000.0
        g = torch.log(g + 1.0) + 1e-9
        g[alpha < 0.1] = 100.0
        return g.detach()


def eig_weight(gain, low=0.05, high=0.2, vh=1.0, w_high=0.7):
    """FaithFusion's gain -> per-pixel L1 weight (models/trainers/base.py, EIGent branch)."""
    w = torch.full_like(gain, 0.1)
    w[(gain >= low) & (gain < high)] = 0.3
    w[(gain >= high) & (gain < vh)] = w_high
    w[gain >= vh] = 1.0
    return w
