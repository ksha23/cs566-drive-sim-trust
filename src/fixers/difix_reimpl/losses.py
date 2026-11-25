"""Difix training loss, from the paper: L = L2 + LPIPS(VGG-16) + 0.5 * L_Gram, where L_Gram compares
auto-correlation (Gram) matrices of VGG-16 features of output and target.
Our choices where the paper is silent: Gram on ImageNet-normalized images at relu1_2, relu2_2, relu3_3, relu4_3,
relu5_3, each Gram normalized by C*H*W, squared Frobenius distance per layer (the Johnson et al. style loss),
layers summed with equal weight; no warm-up, full image (no crop).
Optional per-pixel validity mask (1 = supervise), used to ignore movable objects in driving pairs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lpips

VGG_LAYERS = (3, 8, 15, 22, 29)   # indices of relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 in vgg16.features


class DifixLoss(nn.Module):
    def __init__(self, w_l2=1.0, w_lpips=1.0, w_gram=0.5):
        super().__init__()
        self.w = (w_l2, w_lpips, w_gram)
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:VGG_LAYERS[-1] + 1]
        self.vgg = vgg.eval().requires_grad_(False)
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def grams(self, x):   # x in [-1, 1]
        h = ((x + 1) / 2 - self.mu) / self.sd; out = []
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in VGG_LAYERS:
                b, c, hh, ww = h.shape
                f = h.reshape(b, c, hh * ww).float()
                out.append(f @ f.transpose(1, 2) / (c * hh * ww))
        return out

    def forward(self, pred, tgt, mask=None):
        """pred, tgt: (B,3,H,W) in [-1,1]; mask: (B,1,H,W) or None."""
        if mask is not None:   # outside the mask, the prediction is replaced by the target (no gradient there)
            pred = pred * mask + tgt * (1 - mask)
            l2 = (((pred - tgt) ** 2) * mask).sum() / (mask.sum() * 3 + 1)
        else:
            l2 = F.mse_loss(pred, tgt)
        lp = self.lpips(pred, tgt).mean()
        lg = sum(((a - b) ** 2).sum(dim=(1, 2)).mean() for a, b in zip(self.grams(pred), self.grams(tgt))) if self.w[2] > 0 else pred.new_zeros(())
        total = self.w[0] * l2 + self.w[1] * lp + self.w[2] * lg
        return total, dict(l2=float(l2), lpips=float(lp), gram=float(lg))
