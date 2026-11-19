"""The FLUX.2 latent as the 16 px leaf, decoder side (review R5).

Opt-in: a token stream with the per-image `fluxleaf` mode bit carries, for every coded 16 px leaf, the image's own
FLUX.2 latent (128 floats = one token) instead of a fixed-code payload.  Decoding needs the FLUX.2 VAE: the other
leaves are pasted, the flux holes are filled from the decoded neighbours, the image is encoded, the transmitted
latents are overwritten, the image is decoded and only the flux leaves are pasted; 2-4 passes until the pasted
pixels stop moving.  On an A4500 that is about .55 s per image (the VAE encoder+decoder, not the tokenizer).

`Flux2VAE` mirrors qtok/vae.py's Flux2 (diffusers `AutoencoderKLFlux2`, f8, 32 latent channels, 2 x 2 cells per
token); the package does not depend on qtok.  Needs the optional `encoder` extras (diffusers) and the FLUX.2 weights.
"""
import math

import torch
import torch.nn.functional as F

from .grid import G, K, P, S, masks, size

LEVEL16 = 5  # S / 2**5 = 16 px
S16 = 2  # cells per side in a 16 px leaf
N16 = G // S16
K16 = K  # 32 channels * 2 * 2 cells = 128 floats, exactly one token
PLACEHOLDERS = ("fill",)


class Flux2VAE:
    """FLUX.2-dev VAE: f8, 32 channels, 2 x 2 cells per 128-float token (the qtok.vae.Flux2 wrapper, self-contained)."""

    cell, pack = 8, 2

    def __init__(self, dev="cpu"):
        from diffusers import AutoencoderKLFlux2

        self.dev = dev
        self.vae = (AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae",
                                                      torch_dtype=torch.float32).to(dev).eval().requires_grad_(False))

    @torch.no_grad()
    def enc(self, x):
        return self.vae.encode(x).latent_dist.mode()

    @torch.no_grad()
    def dec(self, z):
        return self.vae.decode(z).sample.clamp(-1, 1)


def up_cell(m):
    """[N16, N16] flux leaves -> latent-cell mask [1, 1, G, G]."""
    return m[None, None].float().repeat_interleave(S16, 2).repeat_interleave(S16, 3)


def up_px(m):
    """[N16, N16] flux leaves -> pixel mask [1, 1, S, S]."""
    return m[None, None].float().repeat_interleave(S16 * P, 2).repeat_interleave(S16 * P, 3)


@torch.no_grad()
def fill_holes(xh, pm, rounds=4):
    """Fill the flux holes from the decoded neighbours (only transmitted pixels are read): cell-average the known
    neighbours and propagate the averages into the hole one 8 px cell per round."""
    pmb = pm > .5
    known = (~pmb).float()  # [1, 1, S, S]
    k8 = F.avg_pool2d(known, P)
    y8 = F.avg_pool2d(xh * known, P) / k8.clamp_min(1e-6)
    k = (k8 > .5).float()
    y8 = torch.where(k > .5, y8, torch.zeros_like(y8))
    for _ in range(rounds):
        ws = F.avg_pool2d(k, 3, 1, 1)
        ys = F.avg_pool2d(y8 * k, 3, 1, 1)
        fill = ys / ws.clamp_min(1e-6)
        add = (ws > 0) & (k < .5)
        y8 = torch.where(add, fill, y8)
        k = ((k > .5) | add).float()
    return torch.where(pmb, F.interpolate(y8, scale_factor=P, mode="nearest"), xh)


@torch.no_grad()
def mixed_decode(xh, lvl, zt, vae, iters=2, max_iters=4, tol=2e-3, placeholder="fill"):
    """Decode a layout whose 16 px leaves are FLUX latents: xh = paste decode (flux leaves zero), zt = the transmitted
    latents [1, 32, G, G].  Returns (y, pixel_mask, latent_mask, passes)."""
    return mixed_decode_mask(xh, masks(lvl)[LEVEL16][0], zt, vae, iters, max_iters, tol, placeholder)


@torch.no_grad()
def mixed_decode_mask(xh, m5, zt, vae, iters=2, max_iters=4, tol=2e-3, placeholder="fill"):
    """mixed_decode from the flux-leaf cell mask [N16, N16] instead of a layout (used by the token decoder)."""
    if not bool(m5.any()):
        return xh, None, None, 0
    if placeholder not in PLACEHOLDERS:
        raise ValueError(placeholder)
    pm, cm = up_px(m5), up_cell(m5)
    y = fill_holes(torch.where(pm > 0, torch.zeros_like(xh), xh), pm)
    n = 0
    with torch.autocast("cuda", torch.bfloat16, enabled=str(xh.device).startswith("cuda")):
        while n < max_iters:
            z = torch.where(cm > 0, zt, vae.enc(y))
            yn = torch.where(pm > 0, vae.dec(z).float().clamp(-1, 1), xh)
            n += 1
            delta = float((yn - y).abs().max()) if n >= iters and tol is not None else 1e9
            y = yn
            if tol is not None and delta < tol:
                break
    return y, pm, cm, n
