"""TileUp: the learned latent upsampler of the rectangular tokenizer, and the FLUX.2 VAE decoder.

The checkpoint (`tileup_rect.pt`) holds the TileUp weights together with the RectTileDown / RectTileLift weights, the
rectangular basis and the latent mean / std; `tessera.rect.search.load_tokenizer_ckpt` rebuilds all of it."""

import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE = 64                    # FLUX.2 latent cells per side of a 512 px image (one cell = 8 px)
CH = 32                      # FLUX.2 latent channels
SIZES = (2, 4, 8, 16, 32)    # tile side in cells (16, 32, 64, 128, 256 px)
N_SIZE = len(SIZES)
IN_CH = CH + N_SIZE + 2 + 1  # canvas, tile size, position, noise
WIDTHS = (96, 144, 192)      # channels at 64 x 64, 32 x 32, 16 x 16


def gn(c):
    """GroupNorm with as many groups as possible up to 32 (the review's "GroupNorm 32").  144 % 32 != 0 at the pilot's
    32 x 32 width, so that level uses 16 groups; 96 and 192 use 32.  Deterministic and documented."""
    g = 32
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class Res(nn.Module):
    """Standard residual block, with the TileUp group count: GroupNorm, SiLU, conv 3x3, twice."""

    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                               gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class Attn(nn.Module):
    """One pre-norm self-attention block (multi-head, residual).

    Keeps `nn.MultiheadAttention` parameter layout for checkpoint compatibility while projecting QKV directly into
    `F.scaled_dot_product_attention`."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = gn(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.heads = int(heads)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        d = c // self.heads
        y = self.norm(x).flatten(2).transpose(1, 2)  # [B, N, C]
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.view(b, n, 3, self.heads, d).permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, N, D]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class TileUp(nn.Module):
    """Whole-canvas latent upsampler: layout input [B,40,64,64] -> residual [B,32,64,64] on the bilinear token canvas.

    forward(canvas, size_map, pos, noise) returns canvas + residual (so a zero-initialised model returns its canvas
    unchanged); the learned part is `residual`.  Use predict_canvas/predict_patches for decoding with the bilinear base
    and the copy-through mask."""

    def __init__(self, c=WIDTHS, in_ch=IN_CH, heads=4, attn_blocks=1, attn_32=False):
        super().__init__()
        c0, c1, c2 = c
        self.c = (int(c0), int(c1), int(c2))
        self.attn_blocks = int(attn_blocks)
        self.attn_32 = bool(attn_32)
        if self.attn_blocks < 1:
            raise ValueError(f"attn_blocks must be >= 1, got {attn_blocks}")
        self.stem = nn.Conv2d(in_ch, c0, 3, padding=1)
        self.e0 = nn.Sequential(Res(c0), Res(c0))          # 64 x 64
        self.d1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.e1 = nn.Sequential(Res(c1), Res(c1))          # 32 x 32
        self.attn_e1 = Attn(c1, heads) if self.attn_32 else None
        self.d2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(Res(c2), Res(c2))          # 16 x 16
        self.attn = Attn(c2, heads)
        self.attn_extra = nn.ModuleList([Attn(c2, heads) for _ in range(self.attn_blocks - 1)])
        self.mid = nn.Sequential(Res(c2), Res(c2))         # the 16 x 16 decoder level
        self.u1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.k1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.p1 = nn.Sequential(Res(c1), Res(c1))          # 32 x 32 decoder level
        self.attn_p1 = Attn(c1, heads) if self.attn_32 else None
        self.u0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.k0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.p0 = nn.Sequential(Res(c0), Res(c0))          # 64 x 64 decoder level
        self.out = nn.Sequential(gn(c0), nn.SiLU(), nn.Conv2d(c0, CH, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)  # zero-initialised output: the model starts as the bilinear/lift baseline
        nn.init.zeros_(self.out[-1].bias)
        self._basis_keys = ()

    def set_basis(self, basis):
        """Attach optional DC+96AC orthonormal lift basis {s: U_s [32*s*s, 96]} as module buffers."""
        for k in list(self._basis_keys):
            if hasattr(self, f"_basis_{k}"):
                delattr(self, f"_basis_{k}")
        if not basis:
            self._basis_keys = ()
            return self
        keys = []
        dev = self.stem.weight.device
        for s, U in sorted(basis.items()):
            si = int(s)
            self.register_buffer(f"_basis_{si}", U.detach().float().to(dev))
            keys.append(si)
        self._basis_keys = tuple(keys)
        return self

    def get_basis(self):
        if not self._basis_keys:
            return None
        return {s: getattr(self, f"_basis_{s}") for s in self._basis_keys}

    def residual(self, canvas, size_map, pos, noise):
        x = torch.cat((canvas, size_map, pos, noise), 1).to(self.stem.weight.dtype)
        h0 = self.e0(self.stem(x))                 # 64 x 64
        h1 = self.e1(self.d1(h0))                  # 32 x 32
        if self.attn_e1 is not None:
            h1 = self.attn_e1(h1)
        h2 = self.attn(self.e2(self.d2(h1)))       # 16 x 16
        for blk in self.attn_extra:
            h2 = blk(h2)
        h2 = self.mid(h2)
        u1 = self.p1(self.k1(self.u1(h2)) + h1)    # nearest + conv, skip from the 32 x 32 encoder level
        if self.attn_p1 is not None:
            u1 = self.attn_p1(u1)
        u0 = self.p0(self.k0(self.u0(u1)) + h0)    # nearest + conv, skip from the 64 x 64 encoder level
        return self.out(u0)

    def forward(self, canvas, size_map, pos, noise):
        return canvas + self.residual(canvas, size_map, pos, noise)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def decoder_only_flux(dev):
    """FLUX.2 VAE decoder (`black-forest-labs/FLUX.2-dev`, gated) without the encoder weights."""
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.float32
    )
    vae.encoder = None
    if hasattr(vae, "quant_conv"):
        vae.quant_conv = None
    vae = vae.to(dev).eval().requires_grad_(False)

    def dec(z):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            return vae.decode(z).sample.clamp(-1, 1)

    dec.vae = vae
    return dec
