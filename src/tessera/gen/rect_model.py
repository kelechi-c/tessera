"""2:1 k-d tree Rectangular BreadthGen with finest-grid center RoPE, 4D log-shape MLP, and looped blocks.

Generates an image scale-by-scale across the 9 binary k-d tree levels (level 0 = 256x256 roots -> level 8 = 16x16):
  1. Generates active visual tokens at level `l` via MAR + Rectified Flow conditioned on [cls] + all coarser levels.
  2. For non-(2,2) nodes, predicts a 3-way split action {0: leaf, 1: split-H, 2: split-V} masked to valid 2:1 actions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.latent import rect as RL
from tessera.gen.model import _RoPEBlock, _apply_rope2d, DiffHead, drop_labels, P_UNCOND, nparams


def _rope2d_rect_cos_sin(pos, is_cls, head_dim, base=1000.0):
    """Finest-grid-aligned 2D Axial RoPE on continuous rectangular tile CENTERS.

    pos: [L, 4] (r0, c0, sh, sw) in 8 px cells on [0, 64); center is (r0 + (sh-1)/2, c0 + (sw-1)/2).
    """
    dev = pos.device
    cy = pos[:, 0].float() + 0.5 * (pos[:, 2].float() - 1.0)
    cx = pos[:, 1].float() + 0.5 * (pos[:, 3].float() - 1.0)
    q_freq = head_dim // 4
    inv = 1.0 / (base ** (torch.arange(q_freq, device=dev, dtype=torch.float32) / q_freq))
    ang = torch.cat((cy[:, None] * inv[None, :], cx[:, None] * inv[None, :]), -1)
    ang = ang.masked_fill(is_cls[:, None], 0.0)
    return ang.cos()[None, None], ang.sin()[None, None]


class PosEmbRect(nn.Module):
    """LoT-style 4D log-shape MLP + continuous 2D center coordinate projection -> d."""

    def __init__(self, d):
        super().__init__()
        self.shape_mlp = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.shape_mlp[-1].weight)
        nn.init.zeros_(self.shape_mlp[-1].bias)
        self.coord_mlp = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, pos):
        """pos: [..., 4] (r0, c0, sh, sw)."""
        sh = pos[..., 2].float().clamp_min(2.0)
        sw = pos[..., 3].float().clamp_min(2.0)
        lsh = torch.log2(sh / 2.0)
        lsw = torch.log2(sw / 2.0)
        s4 = torch.stack((lsh, lsw, lsh + lsw, lsh - lsw), dim=-1)            # [..., 4]
        cy = (pos[..., 0].float() + 0.5 * (sh - 1.0)) / (RL.SIDE - 1.0) * 2.0 - 1.0
        cx = (pos[..., 1].float() + 0.5 * (sw - 1.0)) / (RL.SIDE - 1.0) * 2.0 - 1.0
        return self.shape_mlp(s4) + self.coord_mlp(torch.stack((cy, cx), dim=-1))


class PackedRoPERectTransformer(nn.Module):
    """Varlen-packed Transformer with rectangular center 2D RoPE + recurrent `loops`."""

    def __init__(self, d, depth, heads, loops=2):
        super().__init__()
        self.d, self.depth, self.heads, self.loops = int(d), int(depth), int(heads), int(loops)
        self.head_dim = self.d // self.heads
        self.blocks = nn.ModuleList(_RoPEBlock(self.d, self.heads) for _ in range(self.depth))
        if self.loops > 1:
            self.loop_emb = nn.Embedding(self.loops, self.d)
            nn.init.zeros_(self.loop_emb.weight)
            self.loop_norm = nn.LayerNorm(self.d)
            self.loop_x0 = nn.Parameter(torch.full((self.loops,), 0.25))

    def forward_packed(self, cls_tok, tok_emb, pos, valid_mask, level):
        """Pack `[B, T, d]` + `cls_tok` [B, d] with per-doc level-causal mask -> `[B, T, d]`."""
        b, t, d = tok_emb.shape
        dev = tok_emb.device
        xs, ps, clss, docs, lvs = [], [], [], [], []
        b_idx, t_idx, pack_pos = [], [], []
        offset = 0
        for bi in range(b):
            real = valid_mask[bi].nonzero(as_tuple=False).flatten()
            k = len(real)
            xs.append(cls_tok[bi:bi + 1])
            ps.append(torch.tensor([[0, 0, 2, 2]], dtype=torch.long, device=dev))
            clss.append(torch.ones(1, dtype=torch.bool, device=dev))
            docs.append(torch.full((1,), bi, dtype=torch.long, device=dev))
            lvs.append(torch.full((1,), -1, dtype=torch.long, device=dev))
            if k > 0:
                xs.append(tok_emb[bi, real])
                ps.append(pos[bi, real])
                clss.append(torch.zeros(k, dtype=torch.bool, device=dev))
                docs.append(torch.full((k,), bi, dtype=torch.long, device=dev))
                lvs.append(level[bi, real])
                b_idx.append(torch.full((k,), bi, dtype=torch.long, device=dev))
                t_idx.append(real)
                pack_pos.append(torch.arange(offset + 1, offset + 1 + k, dtype=torch.long, device=dev))
            offset += 1 + k

        x = torch.cat(xs, 0)[None]                                            # [1, L, d]
        pos_p = torch.cat(ps, 0)                                              # [L, 4]
        is_cls = torch.cat(clss, 0)
        doc_id = torch.cat(docs, 0)
        lv_p = torch.cat(lvs, 0)

        cos, sin = _rope2d_rect_cos_sin(pos_p, is_cls, self.head_dim)
        same_doc = doc_id[:, None] == doc_id[None, :]
        allow = same_doc & ((lv_p[None, :] <= lv_p[:, None]) | (lv_p[None, :] < 0))
        sdpa_mask = allow[None, None]

        if self.loops == 1:
            for blk in self.blocks:
                x = blk(x, cos, sin, sdpa_mask=sdpa_mask)
        else:
            x0 = x
            for lp in range(self.loops):
                if lp == 0:
                    x = x + self.loop_emb.weight[lp]
                else:
                    x = self.loop_norm(x) + self.loop_x0[lp] * x0 + self.loop_emb.weight[lp]
                for blk in self.blocks:
                    x = blk(x, cos, sin, sdpa_mask=sdpa_mask)

        out = torch.zeros(b, t, d, device=dev, dtype=x.dtype)
        if b_idx:
            out[torch.cat(b_idx), torch.cat(t_idx)] = x[0, torch.cat(pack_pos)]
        return out


class ShapeStats:
    """Per-shape [128] mean/std normaliser across all 13 rectangular shapes."""

    def __init__(self, stats=None):
        self.stats = stats or {}

    def fit(self, rect_pyrs):
        for sh_sw in RL.SHAPES:
            xs = [p[sh_sw].reshape(-1, RL.TOK).float() for p in rect_pyrs if sh_sw in p]
            if not xs:
                continue
            x = torch.cat(xs, 0)
            self.stats[sh_sw] = (x.mean(0), x.std(0).clamp_min(1e-3))
        return self

    def norm(self, tok, sh_sw):
        m, s = self.stats[sh_sw]
        return (tok.float() - m.to(tok.device)) / s.to(tok.device)

    def denorm(self, tok, sh_sw):
        m, s = self.stats[sh_sw]
        return tok.float() * s.to(tok.device) + m.to(tok.device)

    def state_dict(self):
        return {f"{k[0]}x{k[1]}": (m.cpu(), s.cpu()) for k, (m, s) in self.stats.items()}

    @classmethod
    def from_state_dict(cls, sd):
        out = {}
        for k, (m, s) in sd.items():
            a, b = k.split("x")
            out[(int(a), int(b))] = (m.float(), s.float())
        return cls(out)


def make_rect_breadth_example(rect_pyr, shape_map_1, label, stats=None):
    """Build one breadth-first 2:1 k-d tree training example from a single-image `rect_pyr` and `shape_map` [2,64,64]."""
    nodes = RL.kdtree_trace(shape_map_1)
    toks, pos, lvl, act = [], [], [], []
    for r0, c0, sh, sw, lv, a in nodes:
        t = rect_pyr[(sh, sw)][0, r0 // sh, c0 // sw].float().cpu()
        if stats is not None:
            t = stats.norm(t, (sh, sw))
        toks.append(t)
        pos.append((r0, c0, sh, sw))
        lvl.append(lv)
        act.append(a)
    return dict(
        node_tok=torch.stack(toks),                                           # [T, 128]
        node_pos=torch.tensor(pos, dtype=torch.long),                         # [T, 4]
        node_level=torch.tensor(lvl, dtype=torch.long),                       # [T]
        node_action=torch.tensor(act, dtype=torch.long),                      # [T] in {0, 1, 2}
        label=torch.tensor(int(label), dtype=torch.long),
    )


def collate_rect_breadth(examples):
    b = len(examples)
    tmax = max(len(e["node_tok"]) for e in examples)
    tok = torch.zeros(b, tmax, RL.TOK)
    pos = torch.zeros(b, tmax, 4, dtype=torch.long)
    lvl = torch.zeros(b, tmax, dtype=torch.long)
    act = torch.zeros(b, tmax, dtype=torch.long)
    msk = torch.zeros(b, tmax, dtype=torch.bool)
    lbl = torch.stack([e["label"] for e in examples])
    for i, e in enumerate(examples):
        t = len(e["node_tok"])
        tok[i, :t] = e["node_tok"]
        pos[i, :t] = e["node_pos"]
        lvl[i, :t] = e["node_level"]
        act[i, :t] = e["node_action"]
        msk[i, :t] = True
    return dict(node_tok=tok, node_pos=pos, node_level=lvl, node_action=act, node_mask=msk, label=lbl)


class BreadthGenRect(nn.Module):
    """Interleaved 2:1 k-d tree breadth-first generator (4D log-shape + center RoPE + loops=2)."""

    def __init__(self, n_classes, d=256, depth=4, heads=8, loops=2, head_w=512, head_depth=4, stats=None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.cfg = dict(d=d, depth=depth, heads=heads, loops=loops, head_w=head_w, head_depth=head_depth)
        self.cls = nn.Embedding(n_classes + 1, d)
        self.pos = PosEmbRect(d)
        self.inp = nn.Linear(RL.TOK, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.tf = PackedRoPERectTransformer(d, depth, heads, loops=loops)
        self.norm = nn.LayerNorm(d)
        self.head = DiffHead(d, head_w, head_depth)
        self.split_head = nn.Linear(d, 3)                                     # {0: leaf, 1: split-H, 2: split-V}
        self.stats = stats

    def encode(self, node_tok, node_pos, node_level, node_mask, known, label):
        b = node_tok.shape[0]
        x = torch.where(known[..., None], self.inp(node_tok), self.mask_emb.expand(b, node_tok.shape[1], -1))
        x = x + self.pos(node_pos)
        return self.norm(self.tf.forward_packed(self.cls(label), x, node_pos, node_mask, level=node_level))

    def loss(self, batch, p_uncond=P_UNCOND, generator=None):
        tok = batch["node_tok"]
        pos = batch["node_pos"]
        lvl = batch["node_level"]
        act = batch["node_action"]
        msk = batch["node_mask"]
        lbl = drop_labels(batch["label"], self.n_classes, p_uncond, generator)
        b, t = msk.shape
        dev = tok.device

        # 1) Token loss across all levels simultaneously (random MAR mask rate per item)
        rates = torch.rand(b, 1, device=dev, generator=generator) * 0.85 + 0.15
        u = torch.rand(b, t, device=dev, generator=generator)
        predict = msk & (u < rates)
        known = msk & (~predict)
        z_tok = self.encode(tok, pos, lvl, msk, known, lbl)
        l_tok = self.head.loss(tok[predict], z_tok[predict]) if bool(predict.any()) else tok.new_zeros(())

        # 2) 3-way split loss on non-(2,2) nodes with valid-action masking (all tokens known at that level)
        z_sp = self.encode(tok, pos, lvl, msk, msk, lbl)
        logits = self.split_head(z_sp)                                        # [B, T, 3]
        vmask = RL.valid_action_mask(pos[..., 2], pos[..., 3])                # [B, T, 3]
        logits = logits.masked_fill(~vmask, -1e9)
        can_split = msk & ((pos[..., 2] > 2) | (pos[..., 3] > 2))
        l_sp = F.cross_entropy(logits[can_split], act[can_split]) if bool(can_split.any()) else tok.new_zeros(())
        return dict(tokens=l_tok, split=l_sp, total=l_tok + l_sp)

    @torch.no_grad()
    def sample(self, label, mar_steps=4, diff_steps=25, temperature=1.0, solver="heun", leaf_bias=0.0, generator=None):
        """Generate a 2:1 k-d tree `(shape_map [2, 64, 64], rect_pyr_raw)` breadth-first from level 0..8.

        `leaf_bias > 0` favours stopping earlier (bigger tiles, fewer tokens); `< 0` favours finer splits."""
        dev = self.cls.weight.device
        lbl = torch.tensor([int(label)], dtype=torch.long, device=dev)
        frontier = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
        all_tok, all_pos, all_lvl = [], [], []
        shape_map = torch.zeros((2, RL.SIDE, RL.SIDE), dtype=torch.long)
        pyr_out = {sh_sw: torch.zeros((1, RL.SIDE // sh_sw[0], RL.SIDE // sh_sw[1], RL.TOK), device=dev)
                   for sh_sw in RL.SHAPES}

        for lv in range(RL.N_LEVELS):
            if not frontier:
                break
            k = len(frontier)
            cur_pos = torch.tensor(frontier, dtype=torch.long, device=dev)
            cur_lvl = torch.full((k,), lv, dtype=torch.long, device=dev)
            cur_tok = torch.zeros((k, RL.TOK), device=dev)
            known_cur = torch.zeros(k, dtype=torch.bool, device=dev)

            prev_n = len(all_tok)
            prev_tok = torch.stack(all_tok) if prev_n else torch.zeros((0, RL.TOK), device=dev)
            prev_pos = torch.stack(all_pos) if prev_n else torch.zeros((0, 4), dtype=torch.long, device=dev)
            prev_lvl = torch.stack(all_lvl) if prev_n else torch.zeros((0,), dtype=torch.long, device=dev)

            # MAR reveal over current level
            perm = torch.randperm(k, generator=generator).to(dev)
            n_steps = min(mar_steps, k)
            for st in range(n_steps):
                a = (st * k) // n_steps
                b_end = ((st + 1) * k) // n_steps
                pick = perm[a:b_end]
                if len(pick) == 0:
                    continue
                t_cat = torch.cat((prev_tok, cur_tok), 0)[None]
                p_cat = torch.cat((prev_pos, cur_pos), 0)[None]
                l_cat = torch.cat((prev_lvl, cur_lvl), 0)[None]
                m_cat = torch.ones((1, prev_n + k), dtype=torch.bool, device=dev)
                kn_cat = torch.cat((torch.ones(prev_n, dtype=torch.bool, device=dev), known_cur))[None]
                z = self.encode(t_cat, p_cat, l_cat, m_cat, kn_cat, lbl)[0, prev_n:]
                gen_t = self.head.sample(z[pick], steps=diff_steps, temperature=temperature, solver=solver, generator=generator)
                cur_tok[pick] = gen_t
                known_cur[pick] = True

            # Predict 3-way split actions for this level
            t_cat = torch.cat((prev_tok, cur_tok), 0)[None]
            p_cat = torch.cat((prev_pos, cur_pos), 0)[None]
            l_cat = torch.cat((prev_lvl, cur_lvl), 0)[None]
            m_cat = torch.ones((1, prev_n + k), dtype=torch.bool, device=dev)
            z_sp = self.encode(t_cat, p_cat, l_cat, m_cat, m_cat, lbl)[0, prev_n:]
            logits = self.split_head(z_sp)
            if leaf_bias != 0.0:
                logits = logits.clone()
                logits[:, RL.ACT_LEAF] += float(leaf_bias)
            vmask = RL.valid_action_mask(cur_pos[:, 2], cur_pos[:, 3])
            logits = logits.masked_fill(~vmask, -1e9)
            acts = logits.argmax(dim=-1).tolist()

            nxt = []
            for j, (r0, c0, sh, sw) in enumerate(frontier):
                t_raw = cur_tok[j] if self.stats is None else self.stats.denorm(cur_tok[j], (sh, sw))
                pyr_out[(sh, sw)][0, r0 // sh, c0 // sw] = t_raw
                all_tok.append(cur_tok[j])
                all_pos.append(cur_pos[j])
                all_lvl.append(cur_lvl[j])
                a = acts[j] if (sh > 2 or sw > 2) else RL.ACT_LEAF
                if a == RL.ACT_SPLIT_H:
                    nxt.append((r0, c0, sh // 2, sw))
                    nxt.append((r0 + sh // 2, c0, sh // 2, sw))
                elif a == RL.ACT_SPLIT_V:
                    nxt.append((r0, c0, sh, sw // 2))
                    nxt.append((r0, c0 + sw // 2, sh, sw // 2))
                else:
                    shape_map[0, r0:r0 + sh, c0:c0 + sw] = sh
                    shape_map[1, r0:r0 + sh, c0:c0 + sw] = sw
            frontier = nxt

        return shape_map, pyr_out
