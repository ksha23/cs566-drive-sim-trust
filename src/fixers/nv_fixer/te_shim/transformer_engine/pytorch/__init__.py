import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, zero_centered_gamma=False, **kw):
        super().__init__()
        self.eps, self.zero_centered_gamma = eps, zero_centered_gamma
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def reset_parameters(self):
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(self.weight)
        else:
            torch.nn.init.ones_(self.weight)

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float() + (1.0 if self.zero_centered_gamma else 0.0)
        return (y * w).to(x.dtype)


from . import attention, distributed, module  # noqa: E402,F401
