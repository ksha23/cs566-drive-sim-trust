"""Our own Gaussian-splat renderer (hybrid): a drop-in replacement for `splats.gs.render`.

What is OURS (plain PyTorch, so autograd differentiates it and every line can be read and explained):
  * camera chain: world -> camera transform (viewmat = inverse of the camera-to-world pose),
  * 3D covariance Sigma = R S S^T R^T from the wxyz quaternion and the log-scales (rotated to the camera),
  * EWA splatting: 2D covariance J W Sigma W^T J^T with the perspective Jacobian J (with gsplat's clamp of
    off-screen centers), the low-pass dilation (+eps2d on the diagonal, gsplat's "classic" mode as D uses;
    the "antialiased" opacity compensation is also written out), the conic (inverse 2D covariance),
  * culling: near/far plane, degenerate (non-positive-definite) 2D covariance, 3-sigma pixel radius, off-screen,
  * view-dependent color: degree-0..3 real spherical harmonics -> RGB (+0.5, clamped at 0, as gsplat),
  * expected depth (alpha-normalized), and the learnable sky environment map composited behind the splat
    with (1 - alpha), exactly as `splats.gs.render` does.
What is gsplat's (CUDA kernels, used as black boxes):
  * `isect_tiles` + `isect_offset_encode`: which 16x16 pixel tiles each projected Gaussian touches, and the
    sort of all (tile, depth) pairs,
  * `rasterize_to_pixels`: per-pixel front-to-back alpha compositing and its backward pass (returns gradients
    w.r.t. our 2D means, conics, colors and opacities; autograd carries them back through our code).

The output (rgb, alpha, depth, info) has the same meaning as `splats.gs.render`, and `info` carries the fields
gsplat's DefaultStrategy reads for densification ("means2d" [1,N,2] with a retained gradient, "radii" [1,N],
"width", "height", "n_cameras", "gaussian_ids"), in gsplat's non-packed layout, so `fit_splat.py` can train with it.
Validated against gsplat and against the pure-PyTorch reference `splats.torch_ref_render`: docs/own_renderer.md.
"""
import math
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_pixels

from splats.paralane import W, H

TILE = 16         # tile size of gsplat's rasterizer (pixels)
EPS2D = 0.3       # low-pass filter: added to the 2D covariance diagonal (pixels^2), gsplat's default
NEAR, FAR = 0.01, 1e10  # gsplat's default near / far planes (m)

# real spherical-harmonics constants (same basis and signs as 3DGS / gsplat)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def quat_to_rotmat(q):
    """(N,4) quaternions in (w, x, y, z) order -> (N,3,3) rotation matrices (normalized first)."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def small_matmul(A, B):
    """A (...,m,k) @ B (...,k,n) for tiny matrices (2x3, 3x3) as a broadcast multiply + sum over k.
    Same maths as torch.matmul, but torch.matmul on millions of 3x3 matrices launches batched GEMMs that
    (with their backward) took >80% of the render time in our profile; this is plain elementwise work."""
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def eval_sh(degree, dirs, coeffs):
    """View-dependent color from real SH. dirs (N,3) viewing directions (camera -> Gaussian, any length),
    coeffs (N,16,3). Only the first (degree+1)^2 coefficients are used. Returns (N,3)."""
    x, y, z = F.normalize(dirs, dim=-1).unbind(-1)
    basis = [torch.full_like(x, SH_C0)]
    if degree >= 1:
        basis += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        basis += [SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * x * z,
                  SH_C2[4] * (xx - yy)]
    if degree >= 3:
        basis += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
                  SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
                  SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy)]
    basis = torch.stack(basis, -1)                                   # (N, (degree+1)^2)
    return (basis[..., None] * coeffs[:, :basis.shape[-1]]).sum(1)   # (N, 3)


def project(means, quats, log_scales, viewmat, K, width, height, eps2d=EPS2D, near=NEAR, far=FAR,
            antialiased=False):
    """EWA projection of the Gaussians in front of the camera, with gsplat's culling rules.

    Returns a dict over the M Gaussians that survive culling:
      idx (M,) indices into the N input Gaussians, mean2d (M,2) pixel centers, conic (M,3) = upper triangle
      (a, b, c) of the inverse 2D covariance, depth (M,) camera z, radius (M,) int32 pixel radius,
      comp (M,) opacity compensation (1 unless antialiased).
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    # 1) world -> camera for the centers. Culling by depth first (no gradient needed for the decision), so the
    #    heavier covariance math below only runs on Gaussians between the near and far planes.
    z_all = means.detach() @ R.detach()[2] + t.detach()[2]
    idx = torch.nonzero((z_all > near) & (z_all < far)).squeeze(1)
    p_cam = (means[idx][:, None, :] * R).sum(-1) + t                 # (M,3) = R p + t, differentiable in means and pose
    x, y, z = p_cam.unbind(-1)

    # 2) 3D covariance Sigma = R_g S S^T R_g^T = A A^T with A = R_g S (the Gaussian's scaled principal axes as
    #    columns; R_g from the quaternion, S = diag(exp(log-scale))). Rotated into the camera frame:
    #    Sigma_c = R Sigma R^T = (R A)(R A)^T, so we only rotate the three axes.
    A = quat_to_rotmat(quats[idx]) * torch.exp(log_scales[idx])[:, None, :]  # (M,3,3): R_g @ diag(s)
    A_cam = small_matmul(R, A)                                       # (M,3,3)

    # 3) EWA: linearize the perspective projection u = fx x/z + cx, v = fy y/z + cy at the center.
    #    Jacobian J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]. As in 3DGS/gsplat, x/z and y/z are clamped
    #    to 1.3x the field of view so centers far off-screen do not get a huge, unstable Jacobian.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    lim_x_pos = (width - cx) / fx + 0.3 * (0.5 * width / fx)
    lim_x_neg = cx / fx + 0.3 * (0.5 * width / fx)
    lim_y_pos = (height - cy) / fy + 0.3 * (0.5 * height / fy)
    lim_y_neg = cy / fy + 0.3 * (0.5 * height / fy)
    tx = z * torch.minimum(torch.maximum(x / z, -lim_x_neg), lim_x_pos)
    ty = z * torch.minimum(torch.maximum(y / z, -lim_y_neg), lim_y_pos)
    zero = torch.zeros_like(z)
    J = torch.stack([fx / z, zero, -fx * tx / z ** 2,
                     zero, fy / z, -fy * ty / z ** 2], -1).view(-1, 2, 3)
    # 2D covariance J Sigma_c J^T = (J R A)(J R A)^T = T T^T, T = J R A (M,2,3): the axes projected to pixels
    T = small_matmul(J, A_cam)
    a0 = (T[:, 0] * T[:, 0]).sum(-1)                                 # cov2d = [[a0, b], [b, c0]]  (pixel^2)
    b = (T[:, 0] * T[:, 1]).sum(-1)
    c0 = (T[:, 1] * T[:, 1]).sum(-1)
    mean2d = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)     # (M,2) pixel coordinates

    # 4) low-pass dilation: every splat covers at least ~a pixel (a pixel-sized box filter, approximated by a
    #    Gaussian of variance eps2d). "antialiased" mode (Mip-Splatting) also scales opacity by
    #    sqrt(det before / det after) so dilation does not brighten small splats; D uses "classic" (no scaling).
    a, c = a0 + eps2d, c0 + eps2d
    det = a * c - b * b
    comp = torch.sqrt(((a0 * c0 - b * b) / det).clamp(min=0)) if antialiased else torch.ones_like(det)

    # 5) culling, exactly gsplat's rules: degenerate covariance, radius = ceil(3 sigma_max) > 0, overlaps the image
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam_max = mid + torch.sqrt((mid * mid - det).clamp(min=0.01))  # largest eigenvalue of the 2D covariance
        radius = torch.ceil(3 * torch.sqrt(lam_max))
        keep = (det > 0) & (radius > 0)
        keep &= (mean2d[:, 0] + radius > 0) & (mean2d[:, 0] - radius < width)
        keep &= (mean2d[:, 1] + radius > 0) & (mean2d[:, 1] - radius < height)
        keep = torch.nonzero(keep).squeeze(1)
    a, b, c, det = a[keep], b[keep], c[keep], det[keep]
    conic = torch.stack([c / det, -b / det, a / det], -1)            # inverse of [[a, b], [b, c]]
    return dict(idx=idx[keep], mean2d=mean2d[keep], conic=conic, depth=z[keep],
                radius=radius[keep].int(), comp=comp[keep])


def camera_dirs(K, width=W, height=H):
    """Camera-frame ray direction (not normalized) through each pixel center (as splats.gs.pixel_dirs)."""
    v, u = torch.meshgrid(torch.arange(height, device=K.device) + 0.5, torch.arange(width, device=K.device) + 0.5,
                          indexing="ij")
    return torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)


def render(params, sky, c2w, K, sh_degree=3, cam_dirs=None, with_depth=False, width=W, height=H,
           antialiased=False):
    """Same contract as `splats.gs.render`: rgb (H,W,3) in [0,1], alpha (H,W), depth (H,W) or None, info."""
    N = params["means"].shape[0]
    viewmat = torch.linalg.inv(c2w)
    pr = project(params["means"], params["quats"], params["scales"], viewmat, K, width, height,
                 antialiased=antialiased)
    idx = pr["idx"]

    # color of each visible Gaussian from its SH coefficients, seen from the camera center
    nb = (sh_degree + 1) ** 2                                        # SH coefficients in use (1, 4, 9 or 16)
    coeffs = torch.cat([params["sh0"][idx], params["shN"][idx, :nb - 1]], 1)  # (M,nb,3)
    dirs = params["means"][idx] - c2w[:3, 3]
    rgb_g = torch.clamp_min(eval_sh(sh_degree, dirs, coeffs) + 0.5, 0.0)
    feats = torch.cat([rgb_g, pr["depth"][:, None]], -1) if with_depth else rgb_g
    opac = torch.sigmoid(params["opacities"][idx]) * pr["comp"]

    # scatter the M visible Gaussians into dense (1,N,...) tensors: gsplat's non-packed layout. Culled Gaussians
    # get radius 0 (skipped by the kernels) and zero gradients, exactly as with gsplat.rasterization(packed=False).
    def dense(v, fill=0.0):
        out = v.new_full((N,) + v.shape[1:], fill)
        return out.index_put((idx,), v)[None]
    means2d, conics = dense(pr["mean2d"]), dense(pr["conic"])
    feats_d, opac_d = dense(feats), dense(opac)
    radii = torch.zeros(1, N, dtype=torch.int32, device=idx.device)
    radii[0, idx] = pr["radius"]
    depths_d = dense(pr["depth"].detach())

    # gsplat: tile intersection + global sort by (tile, depth) ...
    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    _, isect_ids, flatten_ids = isect_tiles(means2d.detach(), radii, depths_d, TILE, tw, th, packed=False, n_cameras=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th)
    # ... and front-to-back alpha compositing (forward + backward in CUDA)
    img, alpha = rasterize_to_pixels(means2d, conics, feats_d, opac_d, width, height, TILE, offsets, flatten_ids)
    img, alpha = img[0], alpha[0, ..., 0]

    rgb = img[..., :3]
    depth = img[..., 3] / alpha.clamp(min=1e-10) if with_depth else None  # expected depth, as gsplat's "ED"
    if sky is not None:  # sky environment map behind the splat, composited with the remaining transmittance
        if cam_dirs is None:
            cam_dirs = camera_dirs(K, width, height)
        dw = F.normalize(cam_dirs @ c2w[:3, :3].T, dim=-1)
        rgb = rgb + (1 - alpha[..., None]) * sky(dw)
    info = dict(means2d=means2d, radii=radii, depths=depths_d, conics=conics, width=width, height=height,
                n_cameras=1, gaussian_ids=None, camera_ids=None, n_visible=int(idx.numel()),
                isect_ids=isect_ids, flatten_ids=flatten_ids)
    return rgb.clamp(0, 1), alpha, depth, info
