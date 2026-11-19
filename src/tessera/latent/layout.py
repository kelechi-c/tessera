"""Pure-latent layouts: a 512 px image as a quadtree of 16-256 px tiles, one FLUX.2 pyramid token per tile.

A layout is a `size_map` [B, 64, 64] (one entry per 8 px cell of the 512 px FLUX.2 latent grid) holding the side, in
cells, of the tile that covers the cell: 2, 4, 8, 16 or 32 (16, 32, 64, 128, 256 px).  This is the representation
`qtok.tileup.merge_quadtree_pure` returns and `qtok.tileup.build_pyramid_canvas` reads.

A tile of side s takes its token from the pyramid level `lat_pyr[s]`: the image's FLUX.2 latent at 1024 / s px, a
[B, 32, 128 / s, 128 / s] map in which the tile at (r0, c0) owns the 2 x 2 cells starting at (2 r0 / s, 2 c0 / s).
`lat_pyr[2]` is the 512 px latent itself, so a 16 px leaf is the true FLUX.2 latent.  A token is those 2 x 2 x 32
numbers flattened channel-major (c, dy, dx): 128 floats, the same order as `qtok.tileup._pyramid_blocks`.

Tree code: the image is 2 x 2 roots of 256 px (32 cells).  Nodes are visited depth first, children in raster order
(top-left, top-right, bottom-left, bottom-right); every node larger than 16 px emits one bit (1 = split, 0 = leaf) and
16 px nodes are always leaves and emit nothing.  Leaves in that same visiting order are the canonical token order, so
the order is derivable from the bits alone.
"""
import torch
import torch.nn.functional as F

SIDE = 64                    # cells per side at 512 px
CH = 32                      # FLUX.2 latent channels
TOK = CH * 4                 # numbers per token (2 x 2 cells x 32 channels)
SIZES = (2, 4, 8, 16, 32)    # tile side in cells: 16, 32, 64, 128, 256 px
LEAF = 2                     # 16 px: the true FLUX.2 latent, never split
ROOT = 32                    # 256 px roots, 2 x 2 of them
NR = len(SIZES)              # rungs 0..4 = index into SIZES


def px(s):
    return 8 * int(s)


def grid_tokens():
    """Token count of the all-grid layout (every tile 16 px): 1,024."""
    return (SIDE // LEAF) ** 2


def all_grid(b=1, device=None):
    """The all-grid layout: [B, 64, 64] filled with 2.  It reproduces the grid VAE's own latent exactly."""
    return torch.full((b, SIDE, SIDE), LEAF, dtype=torch.long, device=device)


def _check_shape(size_map):
    if size_map.ndim != 3 or size_map.shape[1:] != (SIDE, SIDE):
        raise ValueError(f"size_map must be [B, {SIDE}, {SIDE}], got {tuple(size_map.shape)}")


def validate(size_map):
    """True if every image in the batch is a valid quadtree over SIZES: every cell holds a size in SIZES and every
    tile of side s is an aligned, uniformly labelled s x s block."""
    _check_shape(size_map)
    sm = size_map.long()
    ok = torch.zeros_like(sm, dtype=torch.bool)
    for s in SIZES:
        ok |= sm == s
    if not bool(ok.all()):
        return False
    for s in SIZES[1:]:
        m = (sm == s).float()[:, None]
        # an aligned s x s block is either all s or holds no s at all
        frac = F.avg_pool2d(m, s)[:, 0]
        if bool(((frac > 0) & (frac < 1)).any()):
            return False
    return True


def tokens_of(size_map):
    """[B] token count: one per tile (counted exactly as `merge_quadtree_pure` does)."""
    _check_shape(size_map)
    return sum((size_map == s).flatten(1).sum(1) // (s * s) for s in SIZES)


def rungs_of(size_map):
    """[B, 64, 64] sizes -> rung index 0..4 (index into SIZES)."""
    _check_shape(size_map)
    r = torch.zeros_like(size_map, dtype=torch.long)
    for k, s in enumerate(SIZES):
        r = torch.where(size_map == s, torch.full_like(r, k), r)
    return r


def sizes_of(rungs):
    """[B, 64, 64] rung index 0..4 -> tile side in cells."""
    lut = torch.tensor(SIZES, device=rungs.device)
    return lut[rungs.long().clamp(0, NR - 1)]


def project(rungs, blocked=None):
    """Per-cell rungs [B, 64, 64] (the largest tile each cell allows) -> the coarsest valid size_map that respects every
    cell: a block of side s becomes one tile only if every cell in it allows at least s.  `blocked` [B, 64, 64] bool
    (e.g. text) forces 16 px leaves there."""
    _check_shape(rungs)
    allow = sizes_of(rungs)
    if blocked is not None:
        allow = torch.where(blocked.to(allow.device).bool(), torch.full_like(allow, LEAF), allow)
    a = allow.float()[:, None]
    out = torch.zeros_like(allow)
    for s in reversed(SIZES[1:]):
        mn = -F.max_pool2d(-a, s)                                   # min allowance over each aligned s-block
        fits = (mn >= s).repeat_interleave(s, 2).repeat_interleave(s, 3)[:, 0]
        out = torch.where((out == 0) & fits, torch.full_like(out, s), out)
    return torch.where(out == 0, torch.full_like(out, LEAF), out)


def split_bad(size_map, bad, to_leaf=False):
    """Split every tile larger than 16 px that touches a bad cell: one level down, or straight to 16 px with
    `to_leaf`.  `bad` [B, 64, 64] bool.  Returns a new valid size_map; tiles away from bad cells are kept."""
    _check_shape(size_map)
    sm = size_map.clone()
    b = bad.to(sm.device).bool()[:, None].float()
    for s in SIZES[1:]:
        hit = (F.max_pool2d(b, s) > 0).repeat_interleave(s, 2).repeat_interleave(s, 3)[:, 0] & (size_map == s)
        new = LEAF if to_leaf else s // 2
        sm = torch.where(hit, torch.full_like(sm, new), sm)
    return sm


# ---------------------------------------------------------------- tree code ---------------------------------------------
def leaves(size_map):
    """One image's tiles [(r0, c0, s), ...] in canonical (depth-first, raster children) order.  size_map [64, 64]."""
    sm = size_map.reshape(SIDE, SIDE)
    out = []

    def walk(r0, c0, s):
        if s == LEAF or int(sm[r0, c0]) == s:
            out.append((r0, c0, s))
            return
        h = s // 2
        for dr in (0, h):
            for dc in (0, h):
                walk(r0 + dr, c0 + dc, h)

    for r0 in (0, ROOT):
        for c0 in (0, ROOT):
            walk(r0, c0, ROOT)
    return out


def tree_bits(size_map):
    """One image's split bits in visiting order, and the node of each bit [(r0, c0, s), ...] (16 px nodes emit no bit)."""
    sm = size_map.reshape(SIDE, SIDE)
    bits, nodes = [], []

    def walk(r0, c0, s):
        if s == LEAF:
            return
        split = int(sm[r0, c0]) < s
        bits.append(int(split))
        nodes.append((r0, c0, s))
        if split:
            h = s // 2
            for dr in (0, h):
                for dc in (0, h):
                    walk(r0 + dr, c0 + dc, h)

    for r0 in (0, ROOT):
        for c0 in (0, ROOT):
            walk(r0, c0, ROOT)
    return bits, nodes


class TreeWalker:
    """Incremental decoder of the tree code: `node` is the next node that needs a bit (or None when the tree is
    complete); `push(bit)` consumes one.  Used by the generator's sampler so every sample is a valid tree."""

    def __init__(self):
        self.stack = [(r0, c0, ROOT) for r0 in (ROOT, 0) for c0 in (ROOT, 0)]  # popped in raster order
        self.size_map = torch.zeros(SIDE, SIDE, dtype=torch.long)
        self.leaves = []
        self.bits = []
        self.nodes = []
        self._advance()

    def _advance(self):
        while self.stack and self.stack[-1][2] == LEAF:
            r0, c0, s = self.stack.pop()
            self.size_map[r0:r0 + s, c0:c0 + s] = s
            self.leaves.append((r0, c0, s))
        self.node = self.stack[-1] if self.stack else None

    @property
    def done(self):
        return self.node is None

    def push(self, bit):
        if self.node is None:
            raise RuntimeError("tree is already complete")
        r0, c0, s = self.stack.pop()
        self.bits.append(int(bit))
        self.nodes.append((r0, c0, s))
        if bit:
            h = s // 2
            for dr, dc in ((h, h), (h, 0), (0, h), (0, 0)):   # reversed so the stack pops raster order
                self.stack.append((r0 + dr, c0 + dc, h))
        else:
            self.size_map[r0:r0 + s, c0:c0 + s] = s
            self.leaves.append((r0, c0, s))
        self._advance()


def from_tree_bits(bits):
    """Split bits -> size_map [64, 64].  Raises if the bits are too few or too many for one tree."""
    w = TreeWalker()
    for b in bits:
        w.push(b)
    if not w.done:
        raise ValueError(f"{len(bits)} bits do not complete a tree")
    return w.size_map


# ---------------------------------------------------------------- tokens ------------------------------------------------
def level_shape(s):
    n = 2 * SIDE // s
    return (CH, n, n)


def pack_tokens(lat_pyr, size_map, order=None):
    """One image's tokens [N, 128] in canonical leaf order (or the given [(r0, c0, s)] order).
    `lat_pyr` = {s: [32, 128/s, 128/s]} (no batch dim) for every s used by size_map."""
    order = leaves(size_map) if order is None else order
    out = []
    for r0, c0, s in order:
        z = lat_pyr[s]
        if tuple(z.shape) != level_shape(s):
            raise ValueError(f"pyramid level s={s} must be {level_shape(s)}, got {tuple(z.shape)}")
        i, j = 2 * r0 // s, 2 * c0 // s
        out.append(z[:, i:i + 2, j:j + 2].reshape(TOK))
    return torch.stack(out) if out else torch.zeros(0, TOK)


def unpack_tokens(tokens, order, dtype=None):
    """Tokens [N, 128] at tiles [(r0, c0, s)] -> {s: [32, 128/s, 128/s]} with every other cell zero; the inverse of
    `pack_tokens` on the cells a layout uses (what `build_pyramid_canvas` reads)."""
    dtype = tokens.dtype if dtype is None else dtype
    pyr = {s: torch.zeros(level_shape(s), dtype=dtype, device=tokens.device) for s in SIZES}
    for t, (r0, c0, s) in zip(tokens, order):
        i, j = 2 * r0 // s, 2 * c0 // s
        pyr[s][:, i:i + 2, j:j + 2] = t.reshape(CH, 2, 2).to(dtype)
    return pyr


def level_canvas(lat_s, s, mode="nearest"):
    """[B, 32, 128/s, 128/s] -> [B, 32, 64, 64]: every s-tile's token spread over its s x s cells ('nearest', what
    TileUp's canvas holds) or bilinearly inside the tile ('bilinear', TileUp's residual base).  Same maths as
    `qtok.tileup._pyramid_blocks`."""
    nt = SIDE // s
    b = lat_s.shape[0]
    if s == LEAF:
        return lat_s
    tok = lat_s.reshape(b, CH, nt, 2, nt, 2).permute(0, 1, 2, 4, 3, 5).reshape(b, CH, nt * nt, 2, 2)
    if mode == "nearest":
        t = tok.repeat_interleave(s // 2, dim=-2).repeat_interleave(s // 2, dim=-1)
    elif mode == "bilinear":
        t = F.interpolate(tok.reshape(b * CH * nt * nt, 1, 2, 2), size=(s, s), mode="bilinear", align_corners=False)
        t = t.reshape(b, CH, nt * nt, s, s)
    else:
        raise ValueError(mode)
    return t.reshape(b, CH, nt, nt, s, s).permute(0, 1, 2, 4, 3, 5).reshape(b, CH, SIDE, SIDE)
