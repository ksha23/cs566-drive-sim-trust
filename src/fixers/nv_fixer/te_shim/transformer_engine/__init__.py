"""Minimal pure-PyTorch stand-in for NVIDIA Transformer Engine, covering ONLY the symbols Cosmos-Predict2's
MiniTrainDIT (used by NVIDIA Fixer) imports: pytorch.RMSNorm, pytorch.attention.{DotProductAttention,
apply_rotary_pos_emb}, and two stubs used by imaginaire.utils.graph (CUDA graphs, unused here).
Why: Transformer Engine's PyTorch extension has no prebuilt wheel (sdist only) and the Fixer repo expects
NVIDIA's Docker container, which we cannot use (no docker/sudo on sbel). Semantics follow TE 1.13:
  RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight (fp32 math, cast back), parameter name "weight"
  apply_rotary_pos_emb(t, freqs, tensor_format): non-interleaved rotate-half RoPE on the first freqs.shape[-1]
    channels; freqs is [s, 1, 1, d]; for "bshd" it is transposed to [1, s, 1, d]
  DotProductAttention(no mask): softmax(q k^T / sqrt(d)) v via torch SDPA, output [b, s, h*d] ("bshd")
Numerics differ from TE's fused kernels only at bf16 rounding level."""
from . import pytorch  # noqa: F401
