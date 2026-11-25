"""Our re-implementation of the Difix single-step fixer (Difix3D+, arXiv 2503.01774, Sec. 4.1), written from the paper.

Architecture (paper -> this file):
  * base: pretrained SD-Turbo (tokenizer, CLIP text encoder, VAE, U-Net, noise schedule); prompt "remove degradation".
  * single-step denoising: the VAE latent of the degraded render is treated as a noisy latent at tau = 200
    (timestep index 199 of 1000); the U-Net predicts epsilon and we take the one-step x0 estimate
    z0 = (z_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t).  (No extra noise is added to z_t.)
  * reference-view mixing: in every U-Net transformer block the self-attention runs jointly over the V views of a
    sample: tokens are rearranged (b v) n c -> b (v n) c before attention and back afterwards, so the layer inherits
    the pretrained 2D self-attention weights (implemented as an attention processor, `MultiViewAttnProcessor`).
  * VAE: encoder frozen; decoder LoRA-tuned (rank 4) and given encoder->decoder skip connections (1x1 convs from the
    encoder's down-block inputs to the decoder's up-block inputs, as in Pix2pix-Turbo, which the paper follows).
  * U-Net: fully fine-tuned.
The released Difix code builds the same architecture (Difix3D/src/model.py + mv_unet.py); this file does not import
it. Differences in detail are listed in docs/IMPLEMENTATION_NOTES_difix.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel

BASE = "stabilityai/sd-turbo"
PROMPT = "remove degradation"


class MultiViewAttnProcessor:
    """Self-attention over all tokens of all views of one sample (reference-view mixing)."""

    def __init__(self):
        self.n_views = 1

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        assert encoder_hidden_states is None  # only used for attn1 (self-attention)
        bv, n, c = hidden_states.shape
        v = self.n_views
        x = hidden_states.reshape(bv // v, v * n, c)
        q, k, val = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        h = attn.heads
        q, k, val = (t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2) for t in (q, k, val))
        o = F.scaled_dot_product_attention(q, k, val)
        o = o.transpose(1, 2).reshape(bv // v, v * n, -1).to(q.dtype)
        o = attn.to_out[1](attn.to_out[0](o))
        return o.reshape(bv, n, -1)


class DifixRe(nn.Module):
    ENC_CH = [128, 128, 256, 512]   # channels at the input of each VAE encoder down block
    DEC_CH = [512, 512, 512, 256]   # channels at the input of each VAE decoder up block

    def __init__(self, lora_rank=4, timestep=199, base=BASE):
        super().__init__()
        tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
        ids = tok(PROMPT, max_length=tok.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.register_buffer("prompt_emb", te(ids)[0], persistent=False)   # (1, 77, 1024), fixed prompt
        del te
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
        ac = DDPMScheduler.from_pretrained(base, subfolder="scheduler").alphas_cumprod
        self.t = timestep
        self.register_buffer("abar", ac[timestep].clone().float(), persistent=False)
        # encoder -> decoder skip connections (decoder up-block i takes encoder down-block input 3-i)
        self.skips = nn.ModuleList([nn.Conv2d(self.ENC_CH[3 - i], self.DEC_CH[i], 1, bias=False) for i in range(4)])
        for s in self.skips:
            nn.init.constant_(s.weight, 1e-5)   # start as (almost) the plain SD-Turbo decoder
        # frozen encoder; LoRA on the decoder's convs / attention projections
        self.vae.requires_grad_(False)
        from peft import LoraConfig
        names = [n for n, m in self.vae.decoder.named_modules()
                 if isinstance(m, (nn.Conv2d, nn.Linear)) and n.split(".")[-1] in
                 ("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_q", "to_k", "to_v", "0")
                 and ("to_out" in n or not n.endswith(".0"))]
        self.vae.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian",
                                        target_modules=["decoder." + n for n in names]))
        self.lora_targets = names
        self.decoder_ckpt = False
        # multi-view self-attention in every transformer block
        self.mv_procs = []
        procs = {}
        for name, p in self.unet.attn_processors.items():
            if name.endswith("attn1.processor"):
                mp = MultiViewAttnProcessor(); self.mv_procs.append(mp); procs[name] = mp
            else:
                procs[name] = p
        self.unet.set_attn_processor(procs)

    # ---- parameters that are trained ----
    def trainable_parameters(self):
        ps = list(self.unet.parameters()) + list(self.skips.parameters())
        ps += [p for n, p in self.vae.named_parameters() if "lora" in n]
        return ps

    def set_trainable(self):
        self.unet.requires_grad_(True)
        self.skips.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            p.requires_grad_("lora" in n)

    # ---- VAE with skips ----
    @torch.no_grad()   # the encoder is frozen and its input needs no gradient
    def encode(self, x):
        enc = self.vae.encoder
        h = enc.conv_in(x); acts = []
        for blk in enc.down_blocks:
            acts.append(h); h = blk(h)
        h = enc.conv_act(enc.conv_norm_out(enc.mid_block(h)))
        h = self.vae.quant_conv(enc.conv_out(h))
        mean, logvar = h.chunk(2, dim=1)
        z = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * torch.randn_like(mean)   # posterior sample
        return z * self.vae.config.scaling_factor, acts

    def decode(self, z, acts, ckpt=False):
        dec = self.vae.decoder
        h = dec.conv_in(self.vae.post_quant_conv(z / self.vae.config.scaling_factor))
        h = dec.mid_block(h, None)
        for i, blk in enumerate(dec.up_blocks):
            h = h + self.skips[i](acts[3 - i])
            if ckpt and self.training:   # activation checkpointing of the full-resolution decoder blocks
                h = torch.utils.checkpoint.checkpoint(blk, h, None, use_reentrant=False)
            else:
                h = blk(h, None)
        return dec.conv_out(dec.conv_act(dec.conv_norm_out(h)))

    def forward(self, x):
        """x: (B, V, 3, H, W) in [-1, 1]; view 0 = degraded render, views 1.. = reference(s). Returns view 0, (B,3,H,W)."""
        B, V = x.shape[:2]
        for p in self.mv_procs:
            p.n_views = V
        zt, acts = self.encode(x.flatten(0, 1))
        eps = self.unet(zt, self.t, encoder_hidden_states=self.prompt_emb.expand(zt.shape[0], -1, -1).to(zt.dtype)).sample
        z0 = (zt - (1 - self.abar).sqrt() * eps) / self.abar.sqrt()
        # only the first view (the render being fixed) is decoded; reference views only condition the U-Net
        z0 = z0.unflatten(0, (B, V))[:, 0]
        acts = [a.unflatten(0, (B, V))[:, 0] for a in acts]
        return self.decode(z0, acts, ckpt=self.decoder_ckpt).clamp(-1, 1)

    # ---- checkpoint of the trained parts only ----
    def trained_state_dict(self):
        sd = {"unet." + k: v for k, v in self.unet.state_dict().items()}
        sd.update({"skips." + k: v for k, v in self.skips.state_dict().items()})
        sd.update({"vae." + k: v for k, v in self.vae.state_dict().items() if "lora" in k})
        return sd

    def load_trained_state_dict(self, sd):
        missing = self.load_state_dict(sd, strict=False)
        assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
