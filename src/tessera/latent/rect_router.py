"""One-pass 2:1 k-d tree rectangular router (`LatentRouterRect`) + fast verify loop.

Given a single `z512_n` [B, 32, 64, 64] and the orthonormal `rect_basis`, computes cheap per-shape linear-lift
residual features (zero VAE decode!) and predicts a valid 2:1 k-d tree `shape_map` [B, 2, 64, 64] in ONE pass:
  - Dilated FCN backbone produces a [B, C, 64, 64] spatial feature map.
  - A lightweight pooled split head predicts 3-way k-d actions `{0: leaf, 1: split-H, 2: split-V}` for any node
    `(r0, c0, sh, sw)` by average-pooling the feature map over `[r0:r0+sh, c0:c0+sw]` + 4D log-shape embedding.
  - At inference, decoding top-down from the 4 roots (32x32) guarantees a 100% valid 2:1 k-d tree in <5 ms!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.latent import rect as RL


def rect_router_features(z512_n, basis, tiledown=None):
    """Cheap per-cell router features [B, 32 + 12, 64, 64]: z512_n + mean-abs lift error for all 12 shapes."""
    z = z512_n.float()
    errs = []
    for sh, sw in RL.COARSE_SHAPES:
        entry = basis[(sh, sw)]
        tok = RL.project_rect_tokens(z, sh, sw, entry) if tiledown is None else tiledown.encode_shape(z, (sh, sw), basis)
        zhat = RL.lift_rect_tokens(tok, sh, sw, entry)
        e = (zhat - z).abs().mean(dim=1, keepdim=True)                        # [B, 1, 64, 64]
        errs.append(e)
    return torch.cat([z] + errs, dim=1).to(z512_n.dtype)                      # [B, 44, 64, 64]


def _integral_image(h_1):
    """2D summed-area table in float32: `[W, H, W_sp] -> [W, H+1, W_sp+1]`."""
    return F.pad(h_1.float(), (1, 0, 1, 0)).cumsum(dim=1).cumsum(dim=2)


class _DilBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentRouterRect(nn.Module):
    """One-pass 2:1 k-d tree router: features [B, 44, 64, 64] -> valid `shape_map` [B, 2, 64, 64]."""

    IN_CH = RL.CH + len(RL.COARSE_SHAPES)                                     # 32 + 12 = 44

    def __init__(self, in_ch=IN_CH, width=96, dilations=(1, 2, 4, 8, 2, 1), maxpool_feats=False):
        super().__init__()
        self.in_ch, self.width = int(in_ch), int(width)
        self.maxpool_feats = bool(maxpool_feats)                              # EXPERIMENT: append box max-pool to node feats
        self.stem = nn.Conv2d(self.in_ch, self.width, 3, padding=1)
        self.blocks = nn.Sequential(*[_DilBlock(self.width, d) for d in dilations])
        self.norm = nn.GroupNorm(8, self.width)
        self.shape_mlp = nn.Sequential(nn.Linear(4, self.width), nn.SiLU(), nn.Linear(self.width, self.width))
        self.head = nn.Linear(self.width * (2 if self.maxpool_feats else 1), 3)   # {0: leaf, 1: split-H, 2: split-V}

    def encode_grid(self, feat):
        return F.silu(self.norm(self.blocks(self.stem(feat))))                # [B, W, 64, 64]

    def node_logits(self, h_1, nodes, sat=None):
        """Single-image feature map `h_1` [W, 64, 64] + list of `(r0, c0, sh, sw)` -> logits [N, 3]."""
        if not nodes:
            return h_1.new_zeros((0, 3))
        dev = h_1.device
        if sat is None:
            sat = _integral_image(h_1)
        t = torch.tensor(nodes, dtype=torch.long, device=dev)                 # [N, 4]
        r0, c0, sh, sw = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        r1, c1 = r0 + sh, c0 + sw
        box_sum = sat[:, r1, c1] - sat[:, r0, c1] - sat[:, r1, c0] + sat[:, r0, c0]
        p = (box_sum / (sh * sw).to(box_sum.dtype).unsqueeze(0)).transpose(0, 1).to(h_1.dtype)
        lsh = torch.log2(sh.to(p.dtype) / 2.0)
        lsw = torch.log2(sw.to(p.dtype) / 2.0)
        s4 = torch.stack((lsh, lsw, lsh + lsw, lsh - lsw), dim=1)             # [N, 4]
        if self.maxpool_feats:
            mp = self._box_max(h_1, nodes)                                    # [N, W]
            return self.head(torch.cat([p + self.shape_mlp(s4), mp], dim=1))
        return self.head(p + self.shape_mlp(s4))                              # [N, 3]

    @staticmethod
    def _box_max(h_1, nodes):
        """Per-node max over its box of `h_1` [W, 64, 64] -> [N, W] (nodes are tile-aligned, so a strided max-pool per shape)."""
        out = h_1.new_zeros((len(nodes), h_1.shape[0]))
        by_shape = {}
        for i, (r0, c0, sh, sw) in enumerate(nodes):
            by_shape.setdefault((sh, sw), []).append((i, r0 // sh, c0 // sw))
        for (sh, sw), items in by_shape.items():
            pooled = F.max_pool2d(h_1.unsqueeze(0), (sh, sw), stride=(sh, sw))[0]   # [W, 64/sh, 64/sw]
            idx = torch.tensor([i for i, _, _ in items], device=h_1.device)
            rr = torch.tensor([a for _, a, _ in items], device=h_1.device)
            cc = torch.tensor([b for _, _, b in items], device=h_1.device)
            out[idx] = pooled[:, rr, cc].transpose(0, 1).to(out.dtype)
        return out

    def loss(self, feat, shape_maps, node_weight=None):
        """Supervised 3-way cross-entropy over all non-(2,2) visited nodes of ground-truth `shape_maps` [B,2,64,64].

        `node_weight=None` (default): every node counts equally.  `"area_pow"`: weight (sh*sw)**0.5 per node,
        normalised by the summed weight (big nodes count more, so a shattered flat quadrant no longer dominates)."""
        h = self.encode_grid(feat)
        dev = h.device
        tot, count = h.new_zeros(()), 0
        for bi in range(h.shape[0]):
            trace = RL.kdtree_trace(shape_maps[bi])
            splittable = [(r0, c0, sh, sw, a) for r0, c0, sh, sw, _, a in trace if sh > 2 or sw > 2]
            if not splittable:
                continue
            nodes = [(r0, c0, sh, sw) for r0, c0, sh, sw, _ in splittable]
            targets = torch.tensor([a for *_, a in splittable], dtype=torch.long, device=dev)
            sat = _integral_image(h[bi])
            logits = self.node_logits(h[bi], nodes, sat=sat)
            sh_t = torch.tensor([sh for _, _, sh, _ in nodes], dtype=torch.long, device=dev)
            sw_t = torch.tensor([sw for _, _, _, sw in nodes], dtype=torch.long, device=dev)
            logits = logits.masked_fill(~RL.valid_action_mask(sh_t, sw_t), -1e9)
            if node_weight is None:
                tot = tot + F.cross_entropy(logits, targets, reduction="sum")
                count += len(targets)
            else:
                if node_weight != "area_pow":
                    raise ValueError(f"unknown node_weight {node_weight!r}")
                w = (sh_t * sw_t).to(logits.dtype).sqrt()
                tot = tot + (F.cross_entropy(logits, targets, reduction="none") * w).sum()
                count += float(w.sum())
        return tot / max(count, 1)

    @torch.no_grad()
    def route(self, feat, leaf_bias=0.0, leaf_bias_slope=0.0):
        """One-pass top-down decode -> guaranteed-valid 2:1 k-d tree `shape_map` [B, 2, 64, 64].

        `leaf_bias_slope` (default 0 = off): leaf logit += leaf_bias + slope * log2(sh*sw) per node (affine in log area)."""
        h = self.encode_grid(feat)
        b = h.shape[0]
        out = torch.zeros((b, 2, RL.SIDE, RL.SIDE), dtype=torch.long, device=h.device)
        for bi in range(b):
            sat = _integral_image(h[bi])
            frontier = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
            while frontier:
                logits = self.node_logits(h[bi], frontier, sat=sat)
                sh_t = torch.tensor([sh for _, _, sh, _ in frontier], dtype=torch.long, device=h.device)
                sw_t = torch.tensor([sw for _, _, _, sw in frontier], dtype=torch.long, device=h.device)
                if leaf_bias != 0.0 and leaf_bias_slope == 0.0:
                    logits = logits.clone()
                    logits[:, RL.ACT_LEAF] += float(leaf_bias)
                elif leaf_bias_slope != 0.0:
                    logits = logits.clone()
                    logits[:, RL.ACT_LEAF] += float(leaf_bias) + float(leaf_bias_slope) * torch.log2((sh_t * sw_t).to(logits.dtype))
                logits = logits.masked_fill(~RL.valid_action_mask(sh_t, sw_t), -1e9)
                acts = logits.argmax(dim=-1).tolist()
                nxt = []
                for (r0, c0, sh, sw), a in zip(frontier, acts):
                    if a == RL.ACT_SPLIT_H and (sh > 2):
                        nxt += [(r0, c0, sh // 2, sw), (r0 + sh // 2, c0, sh // 2, sw)]
                    elif a == RL.ACT_SPLIT_V and (sw > 2):
                        nxt += [(r0, c0, sh, sw // 2), (r0, c0 + sw // 2, sh, sw // 2)]
                    else:
                        out[bi, 0, r0:r0 + sh, c0:c0 + sw] = sh
                        out[bi, 1, r0:r0 + sh, c0:c0 + sw] = sw
                frontier = nxt
        return out


def verify_rect(shape_map, check_fn, max_rounds=8, split_choice_fn=None):
    """Fast verify loop for a routed `shape_map` [1, 2, 64, 64]: split failing `bad8` tiles until `check_fn` passes."""
    dev = shape_map.device
    sm = shape_map.clone()
    for rnd in range(max_rounds):
        ev = check_fn(sm)
        if ev["passed"]:
            return dict(shape_map=sm, tokens=int(RL.tokens_of_rect(sm)[0]), passed=True,
                        rounds=rnd + 1, fallback=False, score=float(ev.get("score", 0.0)))
        bad = ev.get("bad8")
        leaves = RL.leaves_rect(sm[0])
        splittable = [t for t in leaves if t[2] > 2 or t[3] > 2]
        if not splittable:
            break
        if bad is not None:
            bad_b = torch.as_tensor(bad, device="cpu", dtype=torch.bool)
            to_split = [t for t in splittable if bool(bad_b[t[0]:t[0] + t[2], t[1]:t[1] + t[3]].any())]
        else:
            to_split = splittable
        if not to_split:
            to_split = splittable
        nxt = torch.zeros_like(sm)
        split_set = set(to_split)
        for r0, c0, sh, sw in leaves:
            if (r0, c0, sh, sw) not in split_set:
                nxt[0, 0, r0:r0 + sh, c0:c0 + sw] = sh
                nxt[0, 1, r0:r0 + sh, c0:c0 + sw] = sw
            else:
                acts = [a for a in RL.valid_actions(sh, sw) if a != RL.ACT_LEAF]
                a = acts[0]
                if len(acts) > 1 and split_choice_fn is not None:
                    cand = int(split_choice_fn(r0, c0, sh, sw))
                    if cand in acts:
                        a = cand
                if a == RL.ACT_SPLIT_H:
                    nxt[0, 0, r0:r0 + sh, c0:c0 + sw] = sh // 2
                    nxt[0, 1, r0:r0 + sh, c0:c0 + sw] = sw
                else:
                    nxt[0, 0, r0:r0 + sh, c0:c0 + sw] = sh
                    nxt[0, 1, r0:r0 + sh, c0:c0 + sw] = sw // 2
        sm = nxt
    sm_fb = RL.all_grid_rect(1, device=dev)
    ev = check_fn(sm_fb)
    return dict(shape_map=sm_fb, tokens=int(RL.tokens_of_rect(sm_fb)[0]), passed=bool(ev["passed"]),
                rounds=max_rounds + 1, fallback=True, score=float(ev.get("score", 0.0)))
