"""Sequences for the tessera generator: one image = a tree (split bits) + one token per leaf.

  ex = make_example(lat_pyr, size_map, label, stats)   # lat_pyr {s: [32, 128/s, 128/s]}, size_map [64, 64]
  batch = collate([ex, ...])                            # padded tensors + masks

A position is (row, col, size index) with row = r0 // 2 and col = c0 // 2 (the 16 px grid, 0..31) and size index into
SIZES; it names a tree node or a leaf tile.  Tokens are normalised per pyramid level and channel (`LevelStats`):
the 512 px latent and the downsampled levels need not share statistics, so each level gets its own.

The grid baseline is the same code with the all-grid size_map: 1,024 leaves of 16 px, a fixed tree.
"""
import torch

from ..latent import layout as LY


class LevelStats:
    """Per-level, per-channel mean and std of the FLUX.2 pyramid latents.  Identity until `fit` is called."""

    def __init__(self, mean=None, std=None):
        self.mean = mean or {s: torch.zeros(LY.CH) for s in LY.SIZES}
        self.std = std or {s: torch.ones(LY.CH) for s in LY.SIZES}

    @torch.no_grad()
    def fit(self, pyramids):
        """`pyramids`: iterable of {s: [32, 128/s, 128/s]} (or batched [B, 32, ...]) dicts."""
        acc = {s: [] for s in LY.SIZES}
        for p in pyramids:
            for s in LY.SIZES:
                if s in p:
                    z = p[s].float()
                    acc[s].append(z.reshape(-1, LY.CH, z.shape[-2] * z.shape[-1]).permute(1, 0, 2).reshape(LY.CH, -1))
        for s, v in acc.items():
            if v:
                v = torch.cat(v, 1)
                self.mean[s], self.std[s] = v.mean(1), v.std(1).clamp_min(1e-4)
        return self

    def _ms(self, sizes, dev):
        m = torch.stack([self.mean[int(s)] for s in sizes]).to(dev)[:, :, None]   # [N, 32, 1]
        d = torch.stack([self.std[int(s)] for s in sizes]).to(dev)[:, :, None]
        return m, d

    def norm(self, tokens, sizes):
        """tokens [N, 128] (channel-major) of tiles with the given sizes -> normalised."""
        m, d = self._ms(sizes, tokens.device)
        return ((tokens.reshape(-1, LY.CH, 4) - m) / d).reshape(-1, LY.TOK)

    def denorm(self, tokens, sizes):
        m, d = self._ms(sizes, tokens.device)
        return (tokens.reshape(-1, LY.CH, 4) * d + m).reshape(-1, LY.TOK)

    def state_dict(self):
        return dict(mean={int(k): v for k, v in self.mean.items()}, std={int(k): v for k, v in self.std.items()})

    @classmethod
    def from_state_dict(cls, d):
        return cls({int(k): v for k, v in d["mean"].items()}, {int(k): v for k, v in d["std"].items()})


def pos_of(nodes):
    """[(r0, c0, s)] -> LongTensor [N, 3] (row, col, size index)."""
    if not nodes:
        return torch.zeros(0, 3, dtype=torch.long)
    k = {s: i for i, s in enumerate(LY.SIZES)}
    return torch.tensor([(r0 // 2, c0 // 2, k[s]) for r0, c0, s in nodes], dtype=torch.long)


def make_example(lat_pyr, size_map, label=0, stats=None):
    """One training sequence.  Returns dict(bits [T], node_pos [T, 3], leaf_pos [N, 3], tokens [N, 128] normalised,
    label, size_map)."""
    sm = size_map.reshape(LY.SIDE, LY.SIDE).long()
    bits, nodes = LY.tree_bits(sm)
    order = LY.leaves(sm)
    tok = LY.pack_tokens(lat_pyr, sm, order).float()
    if stats is not None:
        tok = stats.norm(tok, [s for _, _, s in order])
    return dict(bits=torch.tensor(bits, dtype=torch.long), node_pos=pos_of(nodes), leaf_pos=pos_of(order), tokens=tok,
                label=int(label), size_map=sm)


def _pad(xs, fill):
    n = max(1, max(len(x) for x in xs))
    out = torch.full((len(xs), n) + tuple(xs[0].shape[1:]), fill, dtype=xs[0].dtype)
    mask = torch.zeros(len(xs), n, dtype=torch.bool)
    for i, x in enumerate(xs):
        out[i, :len(x)] = x
        mask[i, :len(x)] = True
    return out, mask


def collate(examples):
    """Pad a list of examples.  Masks are True on real entries."""
    bits, node_mask = _pad([e["bits"] for e in examples], -1)
    node_pos, _ = _pad([e["node_pos"] for e in examples], 0)
    leaf_pos, leaf_mask = _pad([e["leaf_pos"] for e in examples], 0)
    tokens, _ = _pad([e["tokens"] for e in examples], 0.)
    return dict(bits=bits, node_pos=node_pos, node_mask=node_mask, leaf_pos=leaf_pos, leaf_mask=leaf_mask, tokens=tokens,
                label=torch.tensor([e["label"] for e in examples], dtype=torch.long))


def to_pyramid(tokens, size_map, stats=None):
    """Generated (normalised) tokens in canonical leaf order + size_map -> {s: [32, 128/s, 128/s]} raw latents, ready
    for TileUp's canvas (`qtok.tileup.build_pyramid_canvas` / `predict_canvas(size_map=..., lat_pyr=...)`)."""
    order = LY.leaves(size_map)
    tok = tokens[:len(order)].float()
    if stats is not None:
        tok = stats.denorm(tok, [s for _, _, s in order])
    return LY.unpack_tokens(tok, order)


# ---------------------------------------------------------------- breadth-first (interleaved) ----------------------------
BREADTH_LEVELS = (32, 16, 8, 4, 2)  # coarse-to-fine tile sides in cells (256, 128, 64, 32, 16 px)


def breadth_nodes(size_map):
    """Visited quadtree nodes (internal + leaves) in breadth-first order (256 -> 128 -> 64 -> 32 -> 16 px).

    Returns (nodes, levels, splits):
      nodes  : list of (r0, c0, s)
      levels : list of level index 0..4 (0 = 256 px root, 4 = 16 px leaf)
      splits : list of split bit (1 = split into 4 children, 0 = leaf, -1 = 16 px forced leaf)."""
    sm = size_map.reshape(LY.SIDE, LY.SIDE)
    nodes, levels, splits = [], [], []
    cur = [(r0, c0, LY.ROOT) for r0 in (0, LY.ROOT) for c0 in (0, LY.ROOT)]
    for lv, s in enumerate(BREADTH_LEVELS):
        nxt = []
        for r0, c0, _ in cur:
            if s == LY.LEAF:
                nodes.append((r0, c0, s))
                levels.append(lv)
                splits.append(-1)
            else:
                sp = int(int(sm[r0, c0]) < s)
                nodes.append((r0, c0, s))
                levels.append(lv)
                splits.append(sp)
                if sp:
                    h = s // 2
                    for dr in (0, h):
                        for dc in (0, h):
                            nxt.append((r0 + dr, c0 + dc, h))
        cur = nxt
    return nodes, levels, splits


def make_breadth_example(lat_pyr, size_map, label=0, stats=None):
    """One training sequence for BreadthGen: every visited node (internal + leaf) carries its pyramid token,
    its position, its coarse-to-fine level index 0..4, and its split bit (-1 on 16 px leaves)."""
    sm = size_map.reshape(LY.SIDE, LY.SIDE).long()
    nodes, levels, splits = breadth_nodes(sm)
    tok = LY.pack_tokens(lat_pyr, sm, nodes).float()
    if stats is not None:
        tok = stats.norm(tok, [s for _, _, s in nodes])
    return dict(node_pos=pos_of(nodes), node_tok=tok,
                node_level=torch.tensor(levels, dtype=torch.long),
                node_split=torch.tensor(splits, dtype=torch.long),
                label=int(label), size_map=sm)


def collate_breadth(examples):
    """Pad a list of breadth-first examples.  `node_mask` is True on real nodes."""
    node_pos, node_mask = _pad([e["node_pos"] for e in examples], 0)
    node_tok, _ = _pad([e["node_tok"] for e in examples], 0.)
    node_level, _ = _pad([e["node_level"] for e in examples], 0)
    node_split, _ = _pad([e["node_split"] for e in examples], -1)
    return dict(node_pos=node_pos, node_tok=node_tok, node_level=node_level, node_split=node_split,
                node_mask=node_mask, label=torch.tensor([e["label"] for e in examples], dtype=torch.long))
