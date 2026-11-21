"""MAR + Rectified Flow generators with 2D Axial RoPE and packed varlen FlexAttention (plan S9).

Two architectures sharing the same PackedRoPETransformer, 2D tile-center RoPE, and Rectified Flow head:

1. TesseraGen (`TreeAR` + `TokenMAR`):
   - `TreeAR` predicts quadtree split bits causally (depth-first via `TreeWalker` -> 100% valid trees).
   - `TokenMAR` generates leaf tokens via masked autoregression (Li et al. 2024) + Rectified Flow velocity head
     (Heun 2nd-order ODE sampler, zero residual noise at t=0), with aligned coarse-to-fine + random training masks.
   - Grid baseline: `TesseraGen(..., fixed_layout=LY.all_grid(1)[0])` runs the exact same token model on 1,024 leaves.

2. BreadthGen (interleaved scale-by-scale coarse-to-fine generation):
   - Generates level-by-level from 256 px -> 128 px -> 64 px -> 32 px -> 16 px:
     at each scale, generates the active visual tokens conditioned on all coarser levels, then predicts each node's
     split bit conditioned on its OWN visual token (visually grounded splitting, no blind layout guessing!).
   - Supports progressive per-level snapshots (`return_pyramid=True`) and a `split_bias` complexity/detail knob.

Varlen efficiency:
   - Variable-length image sequences (e.g. 17 tokens alongside 1,024 tokens) are PACKED into one flat unpadded
     sequence `[1, sum(1 + L_b), d]` with per-token `doc_id` block-diagonal masks — zero padding tokens wasted.
   - On CUDA, uses PyTorch's native `flex_attention` (`create_block_mask` + compiled Triton kernel) when available,
     with an exact packed `F.scaled_dot_product_attention` fallback on CPU / eager mode.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..latent import layout as LY
from . import data as GD

NPOS = LY.SIDE // 2                      # top-left row / col on the 16 px grid: 0..31
NCENTER = LY.SIDE                        # exact tile center coordinate on the 8 px cell half-grid: 0..63
CONFIGS = {   # token model; "tree" is the TreeAR's own (small) size, so tessera and grid token models match
    "tiny": dict(d=64, depth=2, heads=4, head_w=64, head_depth=2, tree=dict(d=64, depth=2, heads=4)),
    "base": dict(d=768, depth=20, heads=12, head_w=1024, head_depth=6, tree=dict(d=256, depth=4, heads=4)),
}
P_UNCOND = .1                            # label dropout: trains the unconditional class for classifier-free guidance


# ---------------------------------------------------------------- 2D RoPE + Packed FlexAttention -------------------------
_FLEX_FN = None
_FLEX_TRIED = False


def _get_flex_attention():
    """Return compiled PyTorch `flex_attention` + `create_block_mask` if available on CUDA, else None."""
    global _FLEX_FN, _FLEX_TRIED
    if _FLEX_TRIED:
        return _FLEX_FN
    _FLEX_TRIED = True
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        _FLEX_FN = (torch.compile(flex_attention, dynamic=True), create_block_mask)
    except Exception:
        _FLEX_FN = None
    return _FLEX_FN


def _rope2d_cos_sin(pos, is_cls, head_dim, base=1000.0):
    """Exact 2D Axial RoPE cos/sin on continuous tile CENTER coordinates `(cy, cx)` in [0, 63].

    pos: [L, 3] (row, col, size_idx); is_cls: [L] bool (cls tokens receive identity rotation 0).
    Returns (cos, sin) each [1, 1, L, head_dim // 2]."""
    if head_dim % 4 != 0:
        raise ValueError(f"2D RoPE requires head_dim divisible by 4, got {head_dim}")
    dev = pos.device
    w = torch.bitwise_left_shift(torch.ones_like(pos[:, 2]), pos[:, 2]).float()
    cy = 2.0 * pos[:, 0].float() + w - 1.0
    cx = 2.0 * pos[:, 1].float() + w - 1.0
    q_freq = head_dim // 4
    inv = 1.0 / (base ** (torch.arange(q_freq, device=dev, dtype=torch.float32) / q_freq))
    ang = torch.cat((cy[:, None] * inv[None, :], cx[:, None] * inv[None, :]), -1)        # [L, head_dim // 2]
    ang = ang.masked_fill(is_cls[:, None], 0.0)
    return ang.cos()[None, None], ang.sin()[None, None]


def _apply_rope2d(x, cos, sin):
    """Apply 2D rotary embedding to x [1, H, L, head_dim] using precomputed cos/sin [1, 1, L, head_dim // 2]."""
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


class PosEmb(nn.Module):
    """Scale embedding + continuous 2D tile-center coordinate projection -> d.

    Relative 2D geometry in attention is handled by 2D Axial RoPE on Q/K; adding a continuous `(cy, cx)` MLP on
    tile centers gives masked tokens distinct spatial identity from Layer 0 while remaining resolution-free."""

    def __init__(self, d):
        super().__init__()
        self.size = nn.Embedding(LY.NR, d)
        self.coord = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, pos):
        w = torch.bitwise_left_shift(torch.ones_like(pos[..., 2]), pos[..., 2]).float()
        cy = (2.0 * pos[..., 0].float() + w - 1.0) / (NCENTER - 1) * 2.0 - 1.0
        cx = (2.0 * pos[..., 1].float() + w - 1.0) / (NCENTER - 1) * 2.0 - 1.0
        return self.size(pos[..., 2]) + self.coord(torch.stack((cy, cx), -1))


class _RoPEBlock(nn.Module):
    """Pre-norm Transformer block with 2D Axial RoPE on Q/K and packed varlen FlexAttention / SDPA."""

    def __init__(self, d, heads):
        super().__init__()
        if d % heads != 0 or (d // heads) % 4 != 0:
            raise ValueError(f"d={d}, heads={heads}: head_dim must be divisible by 4 for 2D RoPE")
        self.heads = int(heads)
        self.head_dim = d // heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin, block_mask=None, sdpa_mask=None, flex_fn=None):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = _apply_rope2d(q, cos, sin)
        k = _apply_rope2d(k, cos, sin)
        if flex_fn is not None and block_mask is not None:
            out = flex_fn(q, k, v, block_mask=block_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        x = x + self.proj(out.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class PackedRoPETransformer(nn.Module):
    """Varlen-packed Transformer with 2D Axial RoPE and PyTorch FlexAttention (exact SDPA fallback on CPU/eager).

    Concatenates `[cls_b, real_tokens_b...]` across all batch items into ONE unpadded tensor `[1, L_tot, d]` so
    mixing 17-token and 1,024-token images wastes ZERO padding tokens."""

    def __init__(self, d, depth, heads, use_flex="auto", loops=1):
        super().__init__()
        self.d, self.depth, self.heads = int(d), int(depth), int(heads)
        self.loops = int(loops)
        if self.loops < 1:
            raise ValueError(f"loops must be >= 1, got {loops}")
        self.head_dim = self.d // self.heads
        self.use_flex = use_flex
        self.blocks = nn.ModuleList(_RoPEBlock(self.d, self.heads) for _ in range(self.depth))
        if self.loops > 1:
            self.loop_emb = nn.Embedding(self.loops, self.d)
            nn.init.zeros_(self.loop_emb.weight)
            self.loop_norm = nn.LayerNorm(self.d)
            self.loop_x0 = nn.Parameter(torch.full((self.loops,), 0.25))

    def forward_packed(self, cls_tok, tok_emb, pos, valid_mask, causal=False, level=None):
        """Pack `[B, T, d]` (with `valid_mask` [B, T] bool) + `cls_tok` [B, d] -> unpacked `[B, T, d]`.

        - `causal=True`: per-document causal mask (for `TreeAR`).
        - `level=[B, T]`: per-document level-causal mask (attend to cls and nodes with level_k <= level_q, for `BreadthGen`).
        - both False/None: per-document bidirectional mask (for `TokenMAR`)."""
        b, t, d = tok_emb.shape
        dev = tok_emb.device

        # Pack only real tokens (+ 1 cls token per item) into a single flat sequence [1, L_tot, d]
        xs, ps, clss, docs, lvs = [], [], [], [], []
        b_idx, t_idx, pack_pos = [], [], []
        offset = 0
        for bi in range(b):
            real = valid_mask[bi].nonzero(as_tuple=False).flatten()
            k = len(real)
            xs.append(cls_tok[bi:bi + 1])
            ps.append(torch.zeros(1, 3, dtype=torch.long, device=dev))
            clss.append(torch.ones(1, dtype=torch.bool, device=dev))
            docs.append(torch.full((1,), bi, dtype=torch.long, device=dev))
            if level is not None:
                lvs.append(torch.full((1,), -1, dtype=torch.long, device=dev))
            if k > 0:
                xs.append(tok_emb[bi, real])
                ps.append(pos[bi, real])
                clss.append(torch.zeros(k, dtype=torch.bool, device=dev))
                docs.append(torch.full((k,), bi, dtype=torch.long, device=dev))
                if level is not None:
                    lvs.append(level[bi, real])
                b_idx.append(torch.full((k,), bi, dtype=torch.long, device=dev))
                t_idx.append(real)
                pack_pos.append(torch.arange(offset + 1, offset + 1 + k, dtype=torch.long, device=dev))
            offset += 1 + k

        x = torch.cat(xs, 0)[None]                                                       # [1, L_tot, d]
        pos_p = torch.cat(ps, 0)                                                         # [L_tot, 3]
        is_cls = torch.cat(clss, 0)                                                      # [L_tot]
        doc_id = torch.cat(docs, 0)                                                      # [L_tot]
        lv_p = torch.cat(lvs, 0) if level is not None else None
        L = x.shape[1]

        cos, sin = _rope2d_cos_sin(pos_p, is_cls, self.head_dim)

        flex_fn = None
        block_mask = None
        sdpa_mask = None
        want_flex = (self.use_flex is True) or (self.use_flex == "auto" and x.is_cuda and L >= 128)
        if want_flex:
            pair = _get_flex_attention()
            if pair is not None:
                flex_compiled, create_bm = pair
                try:
                    if causal:
                        def mask_mod(_b, _h, q_idx, kv_idx):
                            return (doc_id[q_idx] == doc_id[kv_idx]) & (kv_idx <= q_idx)
                    elif lv_p is not None:
                        def mask_mod(_b, _h, q_idx, kv_idx):
                            return (doc_id[q_idx] == doc_id[kv_idx]) & ((lv_p[kv_idx] <= lv_p[q_idx]) | (lv_p[kv_idx] < 0))
                    else:
                        def mask_mod(_b, _h, q_idx, kv_idx):
                            return doc_id[q_idx] == doc_id[kv_idx]
                    block_mask = create_bm(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=dev)
                    flex_fn = flex_compiled
                except Exception:
                    flex_fn = None
                    block_mask = None

        if flex_fn is None:
            same_doc = doc_id[:, None] == doc_id[None, :]                                # [L, L]
            if causal:
                idx = torch.arange(L, device=dev)
                allow = same_doc & (idx[None, :] <= idx[:, None])
            elif lv_p is not None:
                allow = same_doc & ((lv_p[None, :] <= lv_p[:, None]) | (lv_p[None, :] < 0))
            else:
                allow = same_doc
            sdpa_mask = allow[None, None]                                                # [1, 1, L, L] bool

        if self.loops == 1:
            for blk in self.blocks:
                x = blk(x, cos, sin, block_mask=block_mask, sdpa_mask=sdpa_mask, flex_fn=flex_fn)
        else:
            x0 = x
            for lp in range(self.loops):
                if lp == 0:
                    x = x + self.loop_emb.weight[lp]
                else:
                    x = self.loop_norm(x) + self.loop_x0[lp] * x0 + self.loop_emb.weight[lp]
                for blk in self.blocks:
                    x = blk(x, cos, sin, block_mask=block_mask, sdpa_mask=sdpa_mask, flex_fn=flex_fn)

        out = torch.zeros(b, t, d, device=dev, dtype=x.dtype)
        if b_idx:
            out[torch.cat(b_idx), torch.cat(t_idx)] = x[0, torch.cat(pack_pos)]
        return out


# ---------------------------------------------------------------- tree ---------------------------------------------------
class TreeAR(nn.Module):
    START = 2

    def __init__(self, n_classes, d=64, depth=2, heads=4, loops=1, **_):
        super().__init__()
        self.n_classes = n_classes
        self.cls = nn.Embedding(n_classes + 1, d)          # last index = unconditional
        self.pos = PosEmb(d)
        self.bit = nn.Embedding(3, d)                      # 0, 1, START
        self.tf = PackedRoPETransformer(d, depth, heads, loops=loops)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def logits(self, bits, node_pos, node_mask, label):
        """Teacher-forced logits [B, T] for bit t given bits < t (varlen-packed + 2D RoPE)."""
        b = node_pos.shape[0]
        prev = torch.cat((torch.full((b, 1), self.START, dtype=torch.long, device=bits.device), bits[:, :-1].clamp_min(0)), 1)
        x = self.pos(node_pos) + self.bit(prev)
        h = self.tf.forward_packed(self.cls(label), x, node_pos, node_mask, causal=True)
        return self.out(self.norm(h))[..., 0]

    def loss(self, batch):
        lg = self.logits(batch["bits"], batch["node_pos"], batch["node_mask"], batch["label"])
        m = batch["node_mask"]
        return F.binary_cross_entropy_with_logits(lg[m], batch["bits"][m].float())

    @torch.no_grad()
    def sample(self, label, temperature=1.0, cfg_scale=1.0, split_bias=0.0, generator=None):
        """One tree for one label -> size_map [64, 64].  Always valid: the walker supplies the next node."""
        w = LY.TreeWalker()
        dev = self.cls.weight.device
        lab = torch.tensor([int(label)], device=dev)
        uncond = torch.tensor([self.n_classes], device=dev)
        while not w.done:
            nodes = w.nodes + [w.node]
            pos = GD.pos_of(nodes).to(dev)[None]
            bits = torch.tensor(w.bits + [0], dtype=torch.long, device=dev)[None]
            mask = torch.ones(1, len(nodes), dtype=torch.bool, device=dev)
            lg = self.logits(bits, pos, mask, lab)[0, -1]
            if cfg_scale != 1.0:
                lg_u = self.logits(bits, pos, mask, uncond)[0, -1]
                lg = lg_u + float(cfg_scale) * (lg - lg_u)
            p = torch.sigmoid((lg + float(split_bias)) / max(temperature, 1e-4))
            w.push(int(torch.rand((), generator=generator, device="cpu") < p.cpu()))
        return w.size_map


# ---------------------------------------------------------------- Rectified Flow head ------------------------------------
X0_CLIP = 6.                             # tokens are normalised per level and channel: |x0| beyond 6 sd is an error
T_EPS = 1e-4


def _t_emb(t, d):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * f[None]
    return torch.cat((a.sin(), a.cos()), 1)


class _AdaBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.norm = nn.LayerNorm(w, elementwise_affine=False)
        self.mod = nn.Linear(w, 3 * w)
        self.mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        sh, sc, g = self.mod(c).chunk(3, -1)
        return x + g * self.mlp(self.norm(x) * (1 + sc) + sh)


class DiffHead(nn.Module):
    """Rectified Flow velocity head for 128-number continuous tokens conditioned on transformer vector `z`.

    Linear path:  x_t = (1 - t) * x_0 + t * eps,  t in [0, 1],  eps ~ N(0, I).
    Velocity:     v   = eps - x_0   (so dx/dt = v; integrating t: 1 -> 0 lands exactly on x_0 with zero noise).
    Uses logit-normal + uniform timestep sampling in training and Heun 2nd-order (or Euler) ODE sampling."""

    def __init__(self, d, w=64, depth=2):
        super().__init__()
        self.inp = nn.Linear(LY.TOK, w)
        self.cond = nn.Linear(d, w)
        self.temb = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(_AdaBlock(w) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, LY.TOK))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.w = w

    def forward(self, xt, t, z):
        c = self.cond(z) + self.temb(_t_emb(t, self.w))
        h = self.inp(xt)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)

    def loss(self, x0, z, repeats=4):
        """Rectified Flow MSE on tokens x0 [M, 128] given z [M, d]; each token is noised `repeats` times."""
        x0, z = x0.repeat(repeats, 1), z.repeat(repeats, 1)
        m = len(x0)
        t_ln = torch.sigmoid(torch.randn(m, device=x0.device))
        t_u = torch.rand(m, device=x0.device)
        t = torch.where(torch.rand(m, device=x0.device) < 0.9, t_ln, t_u).clamp(T_EPS, 1.0 - T_EPS)
        eps = torch.randn_like(x0)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * eps
        v_target = eps - x0
        return F.mse_loss(self(xt, t, z), v_target)

    @torch.no_grad()
    def sample(self, z, steps=25, temperature=1.0, z_uncond=None, cfg_scale=1.0, solver="heun", generator=None):
        """Rectified Flow ODE integration from t=1 -> 0 (Heun 2nd-order or Euler) -> tokens [M, 128]."""
        x = torch.randn(len(z), LY.TOK, generator=generator).to(z.device) * temperature
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
        use_cfg = z_uncond is not None and float(cfg_scale) != 1.0

        def vel(x_cur, t_val):
            t_vec = torch.full((len(z),), float(t_val), device=z.device)
            v = self(x_cur, t_vec, z)
            if use_cfg:
                vu = self(x_cur, t_vec, z_uncond)
                v = vu + float(cfg_scale) * (v - vu)
            return v

        for a, b in zip(ts[:-1], ts[1:]):
            dt = float(b - a)                              # negative step from 1 -> 0
            v1 = vel(x, a)
            x_pred = x + dt * v1
            if solver == "heun" and float(b) > 0.0:
                v2 = vel(x_pred, b)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                x = x_pred
        return x.clamp(-X0_CLIP, X0_CLIP)


FlowHead = DiffHead


# ---------------------------------------------------------------- tokens -------------------------------------------------
def reveal_order(leaf_pos, order="size", generator=None):
    """Indices of leaves in the order the sampler reveals them.  'size': largest tiles first, random within a size
    (coarse to fine); 'random': plain MAR."""
    r = torch.rand(len(leaf_pos), generator=generator)
    if order == "size":
        return (-leaf_pos[:, 2].float().cpu() + r * .5).argsort()
    if order == "random":
        return r.argsort()
    raise ValueError(order)


class TokenMAR(nn.Module):
    def __init__(self, n_classes, d=64, depth=2, heads=4, head_w=64, head_depth=2, loops=1):
        super().__init__()
        self.n_classes = n_classes
        self.cls = nn.Embedding(n_classes + 1, d)
        self.pos = PosEmb(d)
        self.inp = nn.Linear(LY.TOK, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.tf = PackedRoPETransformer(d, depth, heads, loops=loops)
        self.norm = nn.LayerNorm(d)
        self.head = DiffHead(d, head_w, head_depth)

    def encode(self, tokens, leaf_pos, leaf_mask, known, label):
        """z [B, N, d] for every leaf (varlen-packed + 2D RoPE); `known` marks revealed tokens."""
        b = tokens.shape[0]
        x = torch.where(known[..., None], self.inp(tokens), self.mask_emb.expand(b, tokens.shape[1], -1))
        x = x + self.pos(leaf_pos)
        return self.norm(self.tf.forward_packed(self.cls(label), x, leaf_pos, leaf_mask, causal=False))

    def loss(self, batch, repeats=4, c2f_prob=0.5, generator=None):
        """Mask a cosine-scheduled share of each sequence's real leaves; flow-matching loss on masked leaves only.

        Mixes coarse-to-fine masking (`c2f_prob`, masking smallest tiles first so training matches `order='size'`
        sampling) with uniform random masking (for general MAR inpainting/editing)."""
        tok, m = batch["tokens"], batch["leaf_mask"]
        b, n = m.shape
        u = torch.rand(b, generator=generator).to(tok.device)
        n_real = m.sum(1)
        n_mask = (torch.cos(u * math.pi / 2) * n_real).ceil().clamp(min=1).long()
        rand_score = torch.rand(b, n, generator=generator).to(tok.device)
        use_c2f = (torch.rand(b, 1, generator=generator).to(tok.device) < c2f_prob)
        sz = batch["leaf_pos"][..., 2].float()
        c2f_score = sz + rand_score * 0.5
        score = torch.where(use_c2f, c2f_score, rand_score).masked_fill(~m, 1e6)
        rank = score.argsort(1).argsort(1)
        masked = (rank < n_mask[:, None]) & m
        z = self.encode(tok, batch["leaf_pos"], m, m & ~masked, batch["label"])
        return self.head.loss(tok[masked], z[masked], repeats)

    @torch.no_grad()
    def sample(self, leaf_pos, label, steps=8, diff_steps=25, temperature=1.0, cfg_scale=1.0, order="size",
               solver="heun", generator=None):
        """Tokens [N, 128] (normalised) for one layout's leaves leaf_pos [N, 3], revealed over `steps` steps on a cosine
        schedule.  order='size' reveals the largest tiles first (ties random), 'random' is plain MAR."""
        dev = self.mask_emb.device
        n = len(leaf_pos)
        pos = leaf_pos.to(dev)[None]
        tok = torch.zeros(1, n, LY.TOK, device=dev)
        known = torch.zeros(1, n, dtype=torch.bool, device=dev)
        real = torch.ones(1, n, dtype=torch.bool, device=dev)
        lab = torch.tensor([int(label)], device=dev)
        uncond = torch.tensor([self.n_classes], device=dev)
        seq = reveal_order(leaf_pos, order, generator).to(dev)
        done = 0
        for k in range(steps):
            remain = int(math.floor(n * math.cos(math.pi / 2 * (k + 1) / steps)))
            take = seq[done:n - remain] if k < steps - 1 else seq[done:]
            if len(take) == 0:
                continue
            z = self.encode(tok, pos, real, known, lab)[0, take]
            zu = self.encode(tok, pos, real, known, uncond)[0, take] if float(cfg_scale) != 1.0 else None
            tok[0, take] = self.head.sample(z, diff_steps, temperature, z_uncond=zu, cfg_scale=cfg_scale,
                                            solver=solver, generator=generator)
            known[0, take] = True
            done += len(take)
        return tok[0]


# ---------------------------------------------------------------- the pair ----------------------------------------------
def drop_labels(label, n_classes, p=P_UNCOND, generator=None):
    """Replace a share p of labels by the unconditional index n_classes."""
    if p <= 0:
        return label
    drop = torch.rand(label.shape, generator=generator).to(label.device) < p
    return torch.where(drop, torch.full_like(label, n_classes), label)


class TesseraGen(nn.Module):
    """Tree + tokens.  `fixed_layout` (e.g. LY.all_grid(1)[0]) skips the tree: the grid baseline.  `stats` (the
    per-level token normalisation the model was trained with) is saved with it."""

    def __init__(self, n_classes, config="tiny", fixed_layout=None, stats=None):
        super().__init__()
        cfg = dict(CONFIGS[config] if isinstance(config, str) else config)
        tree_cfg = cfg.pop("tree", dict(d=cfg["d"], depth=cfg["depth"], heads=cfg["heads"]))
        self.n_classes, self.config = n_classes, config
        self.tree = None if fixed_layout is not None else TreeAR(n_classes, **tree_cfg)
        self.tokens = TokenMAR(n_classes, **cfg)
        self.register_buffer("fixed_layout", None if fixed_layout is None else fixed_layout.reshape(LY.SIDE, LY.SIDE).long())
        self.stats = stats

    def loss(self, batch, p_uncond=P_UNCOND, generator=None):
        batch = dict(batch, label=drop_labels(batch["label"], self.n_classes, p_uncond, generator))
        out = dict(tokens=self.tokens.loss(batch, generator=generator))
        if self.tree is not None:
            out["tree"] = self.tree.loss(batch)
        out["total"] = sum(out.values())
        return out

    def save(self, path):
        torch.save(dict(format="tessera/gen/2", n_classes=self.n_classes, config=self.config, model=self.state_dict(),
                        fixed_layout=self.fixed_layout, stats=None if self.stats is None else self.stats.state_dict()), path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or ck.get("format") not in ("tessera/gen/1", "tessera/gen/2"):
            raise ValueError(f"{path} is not a tessera generator checkpoint")
        st = None if ck["stats"] is None else GD.LevelStats.from_state_dict(ck["stats"])
        m = cls(ck["n_classes"], ck["config"], ck["fixed_layout"], st)
        m.load_state_dict(ck["model"])
        return m.to(device).eval()

    @torch.no_grad()
    def sample(self, label, steps=8, diff_steps=25, temperature=1.0, cfg_scale=1.0, split_bias=0.0,
               order="size", solver="heun", generator=None):
        """-> (size_map [64, 64], tokens [N, 128] normalised, in canonical leaf order)."""
        sm = self.fixed_layout.cpu() if self.tree is None else self.tree.sample(
            label, temperature, cfg_scale=cfg_scale, split_bias=split_bias, generator=generator)
        lp = GD.pos_of(LY.leaves(sm))
        return sm, self.tokens.sample(lp, label, steps, diff_steps, temperature, cfg_scale=cfg_scale,
                                      order=order, solver=solver, generator=generator).cpu()


# ---------------------------------------------------------------- interleaved breadth-first ------------------------------
class BreadthGen(nn.Module):
    """Interleaved scale-by-scale (breadth-first) generator with 2D Axial RoPE + packed varlen FlexAttention.

    At every scale s in (256, 128, 64, 32, 16 px):
      1. Generates active visual tokens at level `l` via MAR + Rectified Flow, conditioned on [cls] + all coarser
         visual tokens `< l` (packed level-causal attention).
      2. For s > 16 px, predicts each node's split bit conditioned on its OWN generated visual token and all coarser
         context -> visually grounded quadtree decisions (no blind layout guessing!).
    """

    def __init__(self, n_classes, config="tiny", stats=None):
        super().__init__()
        cfg = dict(CONFIGS[config] if isinstance(config, str) else config)
        cfg.pop("tree", None)
        d, depth, heads = cfg["d"], cfg["depth"], cfg["heads"]
        loops = int(cfg.get("loops", 1))
        head_w, head_depth = cfg["head_w"], cfg["head_depth"]
        self.n_classes, self.config = n_classes, config
        self.heads = heads
        self.cls = nn.Embedding(n_classes + 1, d)
        self.pos = PosEmb(d)
        self.inp = nn.Linear(LY.TOK, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.tf = PackedRoPETransformer(d, depth, heads, loops=loops)
        self.norm = nn.LayerNorm(d)
        self.head = DiffHead(d, head_w, head_depth)
        self.split_head = nn.Linear(d, 1)
        self.stats = stats

    def encode(self, node_tok, node_pos, node_level, node_mask, known, label):
        """Packed level-causal + 2D RoPE contextual features [B, T, d]."""
        b = node_tok.shape[0]
        x = torch.where(known[..., None], self.inp(node_tok), self.mask_emb.expand(b, node_tok.shape[1], -1))
        x = x + self.pos(node_pos)
        return self.norm(self.tf.forward_packed(self.cls(label), x, node_pos, node_mask, level=node_level))

    def split_loss(self, batch):
        """Visually grounded split BCE across all splittable levels (256, 128, 64, 32 px)."""
        m = batch["node_mask"]
        sp = batch["node_split"]
        keep = m & (batch["node_level"] < 4)
        if not bool(keep.any()):
            return batch["node_tok"].new_zeros(())
        h = self.encode(batch["node_tok"], batch["node_pos"], batch["node_level"], keep, keep, batch["label"])
        lg = self.split_head(h)[..., 0]
        valid = keep & (sp >= 0)
        return F.binary_cross_entropy_with_logits(lg[valid], sp[valid].float())

    def token_loss(self, batch, repeats=4, target_level=None, generator=None):
        """Per-level MAR + Rectified Flow loss: all coarser levels < l are 100% clean/known, level l has a
        cosine-scheduled random subset masked, finer levels > l are unpacked/omitted."""
        tok, pos, lv, m = batch["node_tok"], batch["node_pos"], batch["node_level"], batch["node_mask"]
        b, t = m.shape
        dev = tok.device

        tgt_lv = torch.zeros(b, dtype=torch.long, device=dev)
        for bi in range(b):
            if target_level is not None and bool((m[bi] & (lv[bi] == target_level)).any()):
                tgt_lv[bi] = int(target_level)
            else:
                avail = torch.unique(lv[bi][m[bi]])
                pick = int(torch.randint(len(avail), (1,), generator=generator))
                tgt_lv[bi] = avail[pick]

        in_tgt = m & (lv == tgt_lv[:, None])
        coarser = m & (lv < tgt_lv[:, None])
        active_m = coarser | in_tgt

        u = torch.rand(b, generator=generator).to(dev)
        n_tgt = in_tgt.sum(1).clamp_min(1)
        n_mask = (torch.cos(u * math.pi / 2) * n_tgt).ceil().clamp(min=1).long()
        score = torch.rand(b, t, generator=generator).to(dev).masked_fill(~in_tgt, 2.0)
        rank = score.argsort(1).argsort(1)
        masked = (rank < n_mask[:, None]) & in_tgt
        known = coarser | (in_tgt & ~masked)

        z = self.encode(tok, pos, lv, active_m, known, batch["label"])
        return self.head.loss(tok[masked], z[masked], repeats=repeats)

    def loss(self, batch, p_uncond=P_UNCOND, split_w=1.0, all_levels=False, generator=None):
        batch = dict(batch, label=drop_labels(batch["label"], self.n_classes, p_uncond, generator))
        l_sp = self.split_loss(batch)
        if all_levels:
            present = sorted(set(int(v) for v in batch["node_level"][batch["node_mask"]].tolist()))
            l_tok = sum(self.token_loss(batch, target_level=lv, generator=generator) for lv in present) / max(len(present), 1)
        else:
            l_tok = self.token_loss(batch, generator=generator)
        return dict(tokens=l_tok, split=l_sp, total=l_tok + float(split_w) * l_sp)

    def save(self, path):
        torch.save(dict(format="tessera/breadth-gen/2", n_classes=self.n_classes, config=self.config,
                        model=self.state_dict(), stats=None if self.stats is None else self.stats.state_dict()), path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or ck.get("format") not in ("tessera/breadth-gen/1", "tessera/breadth-gen/2"):
            raise ValueError(f"{path} is not a tessera breadth generator checkpoint")
        st = None if ck["stats"] is None else GD.LevelStats.from_state_dict(ck["stats"])
        m = cls(ck["n_classes"], ck["config"], st)
        m.load_state_dict(ck["model"])
        return m.to(device).eval()

    @torch.no_grad()
    def sample(self, label, steps_per_level=(2, 4, 6, 8, 12), diff_steps=25, temperature=1.0, tree_temp=1.0,
               cfg_scale=1.0, split_bias=0.0, solver="heun", return_pyramid=False, generator=None):
        """Interleaved coarse-to-fine sampling: 256 -> 128 -> 64 -> 32 -> 16 px."""
        dev = self.mask_emb.device
        lab = torch.tensor([int(label)], device=dev)
        uncond = torch.tensor([self.n_classes], device=dev)
        if isinstance(steps_per_level, int):
            steps_per_level = (steps_per_level,) * 5

        ctx_nodes, ctx_toks, ctx_lvls = [], [], []
        sm = torch.zeros(LY.SIDE, LY.SIDE, dtype=torch.long)
        leaf_map = {}
        snapshots = []

        cur = [(r0, c0, LY.ROOT) for r0 in (0, LY.ROOT) for c0 in (0, LY.ROOT)]
        for lv, s in enumerate(GD.BREADTH_LEVELS):
            if not cur:
                break
            n_cur = len(cur)
            n_ctx = len(ctx_nodes)
            all_nodes = ctx_nodes + cur
            pos = GD.pos_of(all_nodes).to(dev)[None]
            lv_t = torch.tensor(ctx_lvls + [lv] * n_cur, dtype=torch.long, device=dev)[None]
            mask = torch.ones(1, n_ctx + n_cur, dtype=torch.bool, device=dev)

            tok = torch.zeros(1, n_ctx + n_cur, LY.TOK, device=dev)
            if n_ctx > 0:
                tok[0, :n_ctx] = torch.stack(ctx_toks).to(dev)
            known = torch.zeros(1, n_ctx + n_cur, dtype=torch.bool, device=dev)
            known[0, :n_ctx] = True

            # 1) Generate visual tokens for this scale via MAR + Rectified Flow
            seq = torch.rand(n_cur, generator=generator).argsort().to(dev)
            n_steps = min(int(steps_per_level[lv]), n_cur)
            done = 0
            for k in range(n_steps):
                remain = int(math.floor(n_cur * math.cos(math.pi / 2 * (k + 1) / n_steps)))
                take_local = seq[done:n_cur - remain] if k < n_steps - 1 else seq[done:]
                if len(take_local) == 0:
                    continue
                take = n_ctx + take_local
                z = self.encode(tok, pos, lv_t, mask, known, lab)[0, take]
                zu = self.encode(tok, pos, lv_t, mask, known, uncond)[0, take] if float(cfg_scale) != 1.0 else None
                tok[0, take] = self.head.sample(z, diff_steps, temperature, z_uncond=zu, cfg_scale=cfg_scale,
                                                solver=solver, generator=generator)
                known[0, take] = True
                done += len(take_local)

            cur_toks = [tok[0, n_ctx + j].detach().cpu() for j in range(n_cur)]
            ctx_nodes = all_nodes
            ctx_toks = ctx_toks + cur_toks
            ctx_lvls = ctx_lvls + [lv] * n_cur

            if return_pyramid:
                sm_snap = sm.clone()
                snap_map = dict(leaf_map)
                for node, tk in zip(cur, cur_toks):
                    r0, c0, ss = node
                    sm_snap[r0:r0 + ss, c0:c0 + ss] = ss
                    snap_map[node] = tk
                snap_order = LY.leaves(sm_snap)
                snapshots.append(dict(level=lv, size_px=LY.px(s), size_map=sm_snap,
                                      tokens=torch.stack([snap_map[nd] for nd in snap_order])))

            # 2) Visually grounded split decision (using the just-generated visual tokens!)
            if s == LY.LEAF:
                for node, tk in zip(cur, cur_toks):
                    r0, c0, ss = node
                    sm[r0:r0 + ss, c0:c0 + ss] = ss
                    leaf_map[node] = tk
                cur = []
            else:
                h = self.encode(tok, pos, lv_t, mask, known, lab)[0, n_ctx:]
                lg = self.split_head(h)[..., 0]
                if float(cfg_scale) != 1.0:
                    hu = self.encode(tok, pos, lv_t, mask, known, uncond)[0, n_ctx:]
                    lgu = self.split_head(hu)[..., 0]
                    lg = lgu + float(cfg_scale) * (lg - lgu)
                probs = torch.sigmoid((lg + float(split_bias)) / max(tree_temp, 1e-4)).cpu()
                u_sp = torch.rand(n_cur, generator=generator)
                nxt = []
                for j, (node, tk) in enumerate(zip(cur, cur_toks)):
                    r0, c0, ss = node
                    if bool(u_sp[j] < probs[j]):
                        h2 = ss // 2
                        for dr in (0, h2):
                            for dc in (0, h2):
                                nxt.append((r0 + dr, c0 + dc, h2))
                    else:
                        sm[r0:r0 + ss, c0:c0 + ss] = ss
                        leaf_map[node] = tk
                cur = nxt

        canon = LY.leaves(sm)
        out_tok = torch.stack([leaf_map[nd] for nd in canon])
        if return_pyramid:
            return sm, out_tok, snapshots
        return sm, out_tok


def nparams(m):
    return sum(p.numel() for p in m.parameters())
