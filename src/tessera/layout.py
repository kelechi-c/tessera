"""Layouts and tiles: what a token stream says about the image, readable without any weights.

A layout is `lvl` [1, 64, 64] (leaf level per 8 px cell; level 0 = whole 512 px image, 5 = 16 px, 6 = native 8 px cell),
`two` [1, 64, 64] (coded leaf carries 2 tokens) and, for the FLUX arm, `sel` [32, 32] (which 16 px leaves are FLUX leaves).
A tile is (y, x, size_px, kind) with kind in KINDS.  `tiles_from_tokens` walks a stream's header bits, so the overlay and
`tessera info` need no codec, no VAE and no GPU."""
import numpy as np
import torch

from .stream.fluxleaf import LEVEL16
from .stream.grid import G, K, L, P, S, leaves, masks, size

KINDS = ("one", "two", "fixed16", "flux16", "native")
# overlay colours, as in the demo pack key: 1-token pink, 2-token cyan, fixed 16 px orange, FLUX 16 px green, native yellow
COLOURS = {"one": (255, 40, 160), "two": (40, 220, 255), "fixed16": (255, 140, 40), "flux16": (60, 220, 90),
           "native": (255, 230, 40)}
LABELS = {"one": "1-token tile", "two": "2-token tile", "fixed16": "fixed-code 16 px", "flux16": "FLUX latent 16 px",
          "native": "native 8 px cell"}


def validate(lvl, two=None):
    """True if every level-l block of the map is one leaf or fully split (the quadtree rule)."""
    lvl = lvl if lvl.dim() == 3 else lvl[None]
    total = sum(int(m.sum()) * size(l) ** 2 for l, m in enumerate(masks(lvl)))
    return total == G * G


def tile_list(lvl, two, sel=None):
    """Layout -> [(y, x, size_px, kind)] in depth-first order (the order of the stream)."""
    out = []
    for l, i, j, t in leaves(lvl, two):
        if l == L - 1:
            out.append((i * P, j * P, P, "native"))
        elif l == LEVEL16 and sel is not None:
            out.append((i * size(l) * P, j * size(l) * P, size(l) * P, "flux16" if bool(sel[i, j]) else "fixed16"))
        else:
            out.append((i * size(l) * P, j * size(l) * P, size(l) * P, "two" if t else "one"))
    return out


def count_tokens(lvl, two, sel=None):
    """Payload tokens of a layout (the header's 1 token is separate): 1 or 2 per coded leaf, 1.5 per native cell; every
    FLUX leaf is one token."""
    n = 0.0
    for l, i, j, t in leaves(lvl, two):
        n += 1.5 if l == L - 1 else (1 if (l == LEVEL16 and sel is not None and bool(sel[i, j])) else (2 if t else 1))
    return n


def tiles_from_tokens(tok):
    """Walk a Tokens object's bits -> [(y, x, size_px, kind)]; raises ValueError if the bits do not parse."""
    bits = tok.bits.tolist() if torch.is_tensor(tok.bits) else list(tok.bits)
    pos = [1 if tok.fluxleaf else 0]
    out = []

    def take(n):
        if pos[0] + n > len(bits):
            raise ValueError("token header is truncated")
        b = bits[pos[0]:pos[0] + n]
        pos[0] += n
        return b

    def walk(l, i, j):
        if l == L - 1:
            out.append((i * P, j * P, P, "native"))
            return
        if take(1)[0]:
            for di in (0, 1):
                for dj in (0, 1):
                    walk(l + 1, 2 * i + di, 2 * j + dj)
            return
        s = size(l) * P
        if tok.fluxleaf and l == LEVEL16:
            if take(1)[0]:
                take(4)
                out.append((i * s, j * s, s, "flux16"))
                return
            kind = "fixed16"
        t, _, _, _ = take(4)
        out.append((i * s, j * s, s, kind if (tok.fluxleaf and l == LEVEL16) else ("two" if t else "one")))

    walk(0, 0, 0)
    if pos[0] != len(bits):
        raise ValueError(f"token header has {len(bits) - pos[0]} unread bits")
    return out


def tile_map(tiles):
    """[(y, x, size, kind)] -> (kind index map [64, 64] int8 per 8 px cell, covered bool): -1 where uncovered."""
    m = np.full((G, G), -1, np.int8)
    for y, x, s, k in tiles:
        m[y // P:(y + s) // P, x // P:(x + s) // P] = KINDS.index(k)
    return m, bool((m >= 0).all())
