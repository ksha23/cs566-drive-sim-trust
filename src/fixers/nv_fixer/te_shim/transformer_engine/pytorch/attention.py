import math
import torch
import torch.nn.functional as F


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=False, cu_seqlens=None, **kw):
    assert tensor_format in ("sbhd", "bshd")
    seq = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    f = freqs[:seq]
    if tensor_format == "bshd":
        f = f.transpose(0, 1)
    rot = f.shape[-1]
    tr, tp = t[..., :rot], t[..., rot:]
    fc, fs = torch.cos(f.float()), torch.sin(f.float())
    out = (tr.float() * fc + _rotate_half(tr.float()) * fs).to(t.dtype)
    return torch.cat([out, tp], dim=-1) if tp.shape[-1] else out


class DotProductAttention(torch.nn.Module):
    def __init__(self, num_attention_heads, kv_channels, num_gqa_groups=None, attention_dropout=0.0,
                 qkv_format="sbhd", attn_mask_type="no_mask", softmax_scale=None, **kw):
        super().__init__()
        assert attn_mask_type == "no_mask"
        self.qkv_format = qkv_format
        self.scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(kv_channels)

    def forward(self, q, k, v, *a, **kw):
        if self.qkv_format == "sbhd":
            q, k, v = (x.permute(1, 2, 0, 3) for x in (q, k, v))  # -> b h s d
        else:
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.permute(2, 0, 1, 3) if self.qkv_format == "sbhd" else o.transpose(1, 2)
        return o.reshape(o.shape[0], o.shape[1], -1)
