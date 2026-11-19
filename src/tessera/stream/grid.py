"""The cell grid and the quadtree over it.

A 512 px image is a 64 x 64 grid of 8 px cells. A layout is a level map `lvl` [B, 64, 64] of ints: the level of the leaf
covering each cell. Level l has leaves of 64 >> l cells per side, so level 0 is the whole image (512 px), level 5 is
16 px, and level 6 is the native 8 px cell. A map is valid when every level-l block is one leaf (all cells == l) or
split (all cells > l). `two` [B, 64, 64] marks coded leaves (levels 0-5) that carry 2 tokens instead of 1.
"""
import torch
import torch.nn.functional as F

S = 512  # image side
P = 8  # cell side, px
C = 3 * P * P  # floats per cell
G = S // P  # cells per side
ROOT = G  # level-0 leaf side, cells
L = ROOT.bit_length()  # 7 levels: 64, 32, 16, 8, 4, 2, 1 cells
K = 128  # floats per token
NATIVE_TOKENS = C / K  # 1.5


def size(l):
    return ROOT >> l


def cells(x):
    return F.pixel_unshuffle(x, P)


def image(c):
    return F.pixel_shuffle(c, P)


def masks(lvl):
    """Per level l: bool [B, n, n], True where that block is a leaf at level l."""
    out = []
    x = lvl[:, None].float()
    for l in range(L):
        s = size(l)
        hi = F.max_pool2d(x, s)[:, 0]
        lo = -F.max_pool2d(-x, s)[:, 0]
        out.append((hi == l) & (lo == l))
    return out


def up(m, s):
    return m.repeat_interleave(s, -2).repeat_interleave(s, -1)


def block_any(mask, s):
    return F.max_pool2d(mask[None, None].float(), s)[0, 0] > 0


def refine(lvl, two, bad):
    """One step finer wherever a leaf touches a bad cell: 1 token -> 2 tokens, 2 tokens -> split into four."""
    nl, nt = lvl.clone(), two.clone()
    for l, m in enumerate(masks(lvl)[:-1]):
        s = size(l)
        hit = up(m[0] & block_any(bad, s), s)[None]
        go2 = hit & ~two
        sp = hit & ~go2
        nt = (nt | go2) & ~sp
        nl = torch.where(sp, lvl + 1, nl)
    return nl, nt


def leaves(lvl, two):
    """The leaves of a layout [1, G, G] in depth-first (Z) order: [(level, i, j, two)]."""
    ms = masks(lvl)
    out = []

    def walk(l, i, j):
        if bool(ms[l][0, i, j]):
            out.append((l, i, j, bool(two[0, i * size(l), j * size(l)]) if l < L - 1 else False))
            return
        for di in (0, 1):
            for dj in (0, 1):
                walk(l + 1, 2 * i + di, 2 * j + dj)

    walk(0, 0, 0)
    return out
