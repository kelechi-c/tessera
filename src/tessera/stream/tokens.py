"""The token stream: header bits (tree, rates, leaf-code flags) and the leaves' floats, and decoding from them alone.

Header, depth first from the whole-image node: per node above the native level, 1 bit "split"; per coded leaf, 4 bits
(2 tokens, leaf code as 2 bits: plain / edge / poly / learned, grain).  A stream may also be a FLUX-leaf stream
(format tessera/1): one mode bit first, then at every coded 16 px leaf one more bit "flux": 1 = the leaf carries the
image's own FLUX.2 latent instead of a code payload, 0 = an ordinary fixed-code leaf (the per-leaf pick of the vae16 arm); decoding it needs the FLUX.2 VAE (see fluxleaf.py).. Payload: every leaf's floats in the same order (t x 128 per coded leaf, 192 per native
cell). Token count = header tokens (bits / 4096, i.e. 128 fp32 floats' worth, at least 1) + payload floats / 128.
"""
import dataclasses
import math

import torch

from .codec import LeafCodec
from .fluxleaf import LEVEL16, S16
from .grid import C, G, K, L, P, S, leaves, size


@dataclasses.dataclass
class Tokens:
    bits: torch.Tensor  # uint8 [n_bits] of 0/1
    floats: torch.Tensor  # [n_floats] float32
    side: int = S
    fluxleaf: bool = False  # per-image mode bit: coded 16 px leaves carry FLUX.2 latents

    @property
    def header_tokens(self):
        return max(1, math.ceil(len(self.bits) / (K * 32)))

    @property
    def n_tokens(self):
        return self.header_tokens + len(self.floats) / K

    def as_tokens(self):
        """The payload as [ceil(n/128), 128] tokens (the last one zero-padded); the header is separate."""
        f = self.floats
        return torch.nn.functional.pad(f, (0, (-len(f)) % K)).reshape(-1, K)

    def save(self, path):
        torch.save(dict(bits=torch.from_numpy(__import__("numpy").packbits(self.bits.cpu().numpy())), n_bits=len(self.bits),
                        floats=self.floats.half().cpu(), side=self.side,
                        format="tessera/1" if self.fluxleaf else "tessera/0"), path)

    @classmethod
    def load(cls, path):
        import numpy as np

        d = torch.load(path, weights_only=False)
        bits = torch.from_numpy(np.unpackbits(d["bits"].numpy())[: d["n_bits"]].copy())
        return cls(bits, d["floats"].float(), d["side"], fluxleaf=d.get("format", "tessera/0") == "tessera/1")


def pack(tab, lvl, two, fluxlat=None, sel=None):
    """Layout + prepared table -> Tokens.  fluxlat = the image's FLUX.2 latents [1, 32, 64, 64]: every coded 16 px leaf
    then carries its 128-float latent (one token) and the stream gets the per-image flux-leaf mode bit.
    sel [N16, N16] bool (optional): which 16 px leaves are FLUX leaves (default all); the others keep their fixed code."""
    bits, fl = [], []
    if fluxlat is not None:
        bits.append(1)  # mode bit: flux leaves
    lv = leaves(lvl, two)
    ms_leaf = {(l, i, j): t for l, i, j, t in lv}

    def walk(l, i, j):
        if (l, i, j) in ms_leaf:
            if l < L - 1:
                bits.append(0)
                if fluxlat is not None and l == LEVEL16:
                    isf = sel is None or bool(sel[i, j])
                    bits.append(int(isf))
                    if isf:
                        bits.extend([0, 0, 0, 0])  # 1 token, code 0, grain 0: the payload is a latent block
                        zi, zj = i * S16, j * S16
                        fl.append(fluxlat[0, :, zi:zi + S16, zj:zj + S16].reshape(-1))
                        return
                t = 2 if ms_leaf[(l, i, j)] else 1
                e = tab[l]
                k = i * e["n"] + j
                m = int(e["mode"][t][k])
                bits.extend([int(t == 2), m >> 1, m & 1, int(e["grain"][k])])
                fl.append(e["pay"][t][k])
            else:
                x = tab[L - 1]["rec"][1]
                fl.append(x[0, :, i * P:(i + 1) * P, j * P:(j + 1) * P].reshape(-1))
            return
        bits.append(1)
        for di in (0, 1):
            for dj in (0, 1):
                walk(l + 1, 2 * i + di, 2 * j + dj)

    walk(0, 0, 0)
    return Tokens(torch.tensor(bits, dtype=torch.uint8), torch.cat(fl).float().cpu(), fluxleaf=fluxlat is not None)


@torch.no_grad()
def unpack(tok: Tokens, codec: LeafCodec, vae=None, iters=2, max_iters=4, tol=2e-3):
    """Tokens -> image [1, 3, S, S] in [-1, 1].  A flux-leaf stream (tok.fluxleaf) needs the FLUX.2 VAE in `vae` (any
    object with enc/dec, e.g. retina_codec.fluxleaf.Flux2VAE); the fixed codes decode from the bases alone."""
    dev = codec.dev
    bits = tok.bits.tolist()
    fl = tok.floats.to(dev)
    pos = [0, 0]  # bit, float
    groups = {}  # (l, t, ) -> list of (i, j, edge, grain, floats)
    natives = []
    flux = {}  # (i, j) -> latent block [32, 2, 2]
    if tok.fluxleaf:
        if vae is None:
            raise ValueError("token stream uses FLUX leaves: pass a FLUX.2 VAE to decode it (Decoder(..., vae=...))")
        pos[0] = 1  # per-image mode bit
        pos[1] = 0

    def walk(l, i, j):
        if l < L - 1:
            b = bits[pos[0]]
            pos[0] += 1
            if b:
                for di in (0, 1):
                    for dj in (0, 1):
                        walk(l + 1, 2 * i + di, 2 * j + dj)
                return
            if tok.fluxleaf and l == LEVEL16:
                isf = bits[pos[0]]
                pos[0] += 1
                if isf:
                    pos[0] += 4  # 1 token, code 0, grain 0
                    flux[(i, j)] = fl[pos[1]:pos[1] + K].reshape(32, S16, S16)
                    pos[1] += K
                    return
            t, m1, m0, g = bits[pos[0]:pos[0] + 4]
            pos[0] += 4
            t = 2 if t else 1
            groups.setdefault((l, t), []).append((i, j, 2 * m1 + m0, bool(g), fl[pos[1]:pos[1] + t * K]))
            pos[1] += t * K
        else:
            natives.append((i, j, fl[pos[1]:pos[1] + C]))
            pos[1] += C

    walk(0, 0, 0)
    y = torch.zeros(1, 3, S, S, device=dev)
    for (l, t), v in groups.items():
        n = G // size(l)
        idx = torch.tensor([i * n + j for i, j, *_ in v], device=dev)
        out = codec.decode_leaves(l, t, torch.stack([q[4] for q in v]), torch.tensor([q[2] for q in v], device=dev, dtype=torch.long),
                                  torch.tensor([q[3] for q in v], device=dev), idx)
        px = size(l) * P
        for (i, j, *_), o in zip(v, out):
            y[0, :, i * px:(i + 1) * px, j * px:(j + 1) * px] = o
    for i, j, f in natives:
        y[0, :, i * P:(i + 1) * P, j * P:(j + 1) * P] = f.reshape(3, P, P)
    if flux:
        from .fluxleaf import mixed_decode_mask

        m5 = torch.zeros((G // S16, G // S16), dtype=torch.bool, device=dev)
        zt = torch.zeros((1, 32, G, G), device=dev)
        for (i, j), blk in flux.items():
            m5[i, j] = True
            zt[0, :, i * S16:(i + 1) * S16, j * S16:(j + 1) * S16] = blk
        y = mixed_decode_mask(y, m5, zt, vae, iters, max_iters, tol)[0]
    return y


def pack_flux(tab, lvl, two, zt, sel=None):
    """Convenience: pack a layout whose 16 px leaves carry the image's FLUX.2 latents zt [1, 32, 64, 64]."""
    return pack(tab, lvl, two, fluxlat=zt, sel=sel)
