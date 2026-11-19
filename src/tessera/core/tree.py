"""Quadtree layouts over the packed FLUX.2 token grid.

A layout is a level map `lvl` [B, G, G] of ints: the level of the leaf that covers each cell. Level l has leaves of
ROOT >> l cells per side; at 512 px G = 32, ROOT = 16, so levels 0..4 are 16, 8, 4, 2, 1 cells (level 4 = native).
A map is valid when every level-l block is either one leaf (all its cells == l) or split (all cells > l).

In-between rate: `two` [B, G, G] bool marks coded leaves that carry 2 tokens instead of 1 (constant over a leaf, ignored
on native cells). Per region the rate ladder is then 1, 2, 4 (= split into four 1-token children), 8, ... tokens:
octave steps instead of 4x jumps, so a 2 x 2 block can sit at 2:1 between 4:1 and native.
"""
import math

import torch
import torch.nn.functional as F

ROOT = 16


def levels(G, root=ROOT):
    return int(math.log2(root)) + 1  # 16 -> 5 levels


def size(l, root=ROOT):
    return root >> l


def leaf_masks(lvl, root=ROOT):
    """Per level l: bool [B, G/s, G/s], True where that block is a leaf at level l."""
    out = []
    x = lvl[:, None].float()
    for l in range(levels(lvl.shape[-1], root)):
        s = size(l, root)
        hi = F.max_pool2d(x, s)[:, 0]
        lo = -F.max_pool2d(-x, s)[:, 0]
        out.append((hi == l) & (lo == l))
    return out


def count(lvl, two=None, root=ROOT):
    """Tokens per image: leaves, plus one per 2-token leaf, plus one header token for the split and rate flags."""
    ms = leaf_masks(lvl, root)
    n = sum(m.flatten(1).sum(1) for m in ms) + 1
    if two is not None:
        n = n + sum((m & two[:, ::size(l, root), ::size(l, root)]).flatten(1).sum(1) for l, m in enumerate(ms[:-1]))
    return n


def leaf_two(lvl, two, root=ROOT):
    """Per coded level l: bool [B, n, n], True where that block is a leaf carrying 2 tokens."""
    return [m & two[:, ::size(l, root), ::size(l, root)] for l, m in enumerate(leaf_masks(lvl, root)[:-1])]


def sample_two(lvl, p=.3, gen=None, root=ROOT):
    """Random 2-token flags: each coded leaf independently with probability p (p may be a [B] tensor)."""
    B, G, _ = lvl.shape
    two = torch.zeros_like(lvl, dtype=torch.bool)
    p = torch.as_tensor(p, device=lvl.device, dtype=torch.float).expand(B)
    for l, m in enumerate(leaf_masks(lvl, root)[:-1]):
        s = size(l, root)
        pick = m & (torch.rand(m.shape, device=lvl.device, generator=gen) < p[:, None, None])
        two |= pick.repeat_interleave(s, 1).repeat_interleave(s, 2)
    return two


def valid(lvl, root=ROOT):
    G = lvl.shape[-1]
    cover = torch.zeros_like(lvl, dtype=torch.long)
    for l, m in enumerate(leaf_masks(lvl, root)):
        cover += m.long().repeat_interleave(size(l, root), 1).repeat_interleave(size(l, root), 2)
    return bool((cover == 1).all())


def _split(lvl, l, mask, root=ROOT):
    """Split the level-l nodes where mask [B, n, n] is True (only nodes that currently exist at level l)."""
    s = size(l, root)
    node = leaf_masks(lvl, root)[l] & mask
    cells = node.repeat_interleave(s, 1).repeat_interleave(s, 2)
    return torch.where(cells, lvl + 1, lvl)


def from_splits(splits, G, root=ROOT):
    """splits[l]: bool [B, n_l, n_l] (n_l = G / size(l)); a node splits only if it exists. Returns lvl."""
    B = splits[0].shape[0]
    lvl = torch.zeros(B, G, G, dtype=torch.long, device=splits[0].device)
    for l, m in enumerate(splits):
        lvl = _split(lvl, l, m, root)
    return lvl


def energy(z, root=ROOT):
    """Per level, the std of the normalized latent inside each block: [B, n_l, n_l]."""
    out = []
    for l in range(levels(z.shape[-1], root) - 1):
        s = size(l, root)
        m = F.avg_pool2d(z, s)
        v = (F.avg_pool2d(z * z, s) - m * m).clamp_min(0).mean(1)
        out.append(v.sqrt())
    return out


def sample(B, G, dev, z=None, root=ROOT, mix=(.2, .3, .2, .3), gen=None):
    """Random training layouts, a mix of: uniform depth, random per-level split rates, islands, latent-energy thresholds."""
    L = levels(G, root)
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    mode = torch.multinomial(torch.tensor(mix, device=dev), B, replacement=True, generator=gen)
    E = energy(z, root) if z is not None else None
    splits = []
    depth = torch.randint(0, L, (B,), device=dev, generator=gen)
    rate = r(B, L) ** 1.5  # fewer all-native layouts, which carry no loss
    thr = torch.exp(torch.empty(B, device=dev).uniform_(math.log(.05), math.log(1.5), generator=gen))
    # islands: an island block at level ifull around a random cell; split along the path down to it, then split
    # everything inside it down to leaf level ideep. The rest of the image stays at the roots.
    cy, cx = torch.randint(0, G, (2, B), device=dev, generator=gen)
    ifull = torch.randint(0, L - 1, (B,), device=dev, generator=gen)
    ideep = ifull + 1 + (r(B) * (L - 1 - ifull)).long()
    for l in range(L - 1):
        n = G // size(l, root)
        yy = torch.arange(n, device=dev)[None, :, None]
        xx = torch.arange(n, device=dev)[None, None, :]
        m0 = (l < depth)[:, None, None].expand(B, n, n)
        m1 = r(B, n, n) < rate[:, l, None, None]
        on_path = (yy == (cy // size(l, root))[:, None, None]) & (xx == (cx // size(l, root))[:, None, None])
        inside = _inside(cy, cx, ifull, n, root, G, dev)
        m2 = (on_path & (l < ifull)[:, None, None]) | (inside & ((l >= ifull) & (l < ideep))[:, None, None])
        m3 = E[l] > thr[:, None, None] if E is not None else m1
        m = torch.stack((m0, m1, m2, m3), 0)[mode, torch.arange(B, device=dev)]
        splits.append(m)
    return from_splits(splits, G, root)


def _inside(cy, cx, lf, n, root, G, dev):
    """True for level blocks (n x n grid) that lie inside the level-lf block containing (cy, cx)."""
    B = cy.shape[0]
    bs = torch.tensor([size(int(v), root) for v in lf.tolist()], device=dev)  # island block size in cells
    y0 = (cy // bs) * bs
    x0 = (cx // bs) * bs
    s = G // n
    yy = torch.arange(n, device=dev)[None, :, None] * s
    xx = torch.arange(n, device=dev)[None, None, :] * s
    return (yy >= y0[:, None, None]) & (yy < (y0 + bs)[:, None, None]) & (xx >= x0[:, None, None]) & (xx < (x0 + bs)[:, None, None])


def uniform(B, G, depth, dev, root=ROOT):
    return torch.full((B, G, G), depth, dtype=torch.long, device=dev)
