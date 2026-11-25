"""Compact residual U-Net "fixer" (about 17M parameters at base=48): predicts a correction that is added
to the input render, so the identity map is easy to represent (conservative by construction)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNetFixer(nn.Module):
    def __init__(self, base=48, mults=(1, 2, 4, 8)):
        super().__init__()
        ch = [base * m for m in mults]
        self.inc = nn.Conv2d(3, ch[0], 3, padding=1)
        self.down = nn.ModuleList(); self.pool = nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.down.append(nn.Sequential(ResBlock(c, co), ResBlock(co, co))); c = co
            if i < len(ch) - 1:
                self.pool.append(nn.Conv2d(co, co, 3, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(c, c), ResBlock(c, c))
        self.up = nn.ModuleList(); self.upc = nn.ModuleList()
        for co in reversed(ch[:-1]):
            self.upc.append(nn.Conv2d(c, co, 3, padding=1))
            self.up.append(nn.Sequential(ResBlock(2 * co, co), ResBlock(co, co))); c = co
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):  # x in [0,1], (B,3,H,W), any H,W (padded to /8 internally)
        H, W = x.shape[-2:]; m = 2 ** (len(self.down) - 1)
        ph, pw = (-H) % m, (-W) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
        h = self.inc(xp * 2 - 1); skips = []
        for i, blk in enumerate(self.down):
            h = blk(h)
            if i < len(self.pool):
                skips.append(h); h = self.pool[i](h)
        h = self.mid(h)
        for upc, blk in zip(self.upc, self.up):
            h = upc(F.interpolate(h, scale_factor=2, mode="nearest"))
            h = blk(torch.cat([h, skips.pop()], 1))
        return (xp + self.out(h))[..., :H, :W].clamp(0, 1)


if __name__ == "__main__":
    m = UNetFixer()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
