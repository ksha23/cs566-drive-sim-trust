"""Image metrics vs. a real image. Pixels on movable objects (Para-Lane foreground mask of the REAL image)
are pasted from the real image before computing, so cars that differ between passes don't count."""
import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure


class Metrics:
    def __init__(self, device="cuda"):
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def __call__(self, pred, gt, fg=None):
        """pred, gt: (H,W,3) float [0,1]; fg: (H,W) bool movable-object mask or None."""
        if fg is not None:
            k = (~fg).float()[..., None]
            pred = pred * k + gt * (1 - k)
            mse = (((pred - gt) ** 2) * k).sum() / (k.sum() * 3 + 1)
        else:
            mse = ((pred - gt) ** 2).mean()
        a, b = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        return dict(psnr=float(-10 * torch.log10(mse + 1e-10)), ssim=float(self.ssim(a, b)),
                    lpips=float(self.lp(a * 2 - 1, b * 2 - 1)))
