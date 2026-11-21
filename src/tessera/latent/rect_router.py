"""One-pass 2:1 k-d tree rectangular router (`LatentRouterRect`) + fast verify loop.

Given a single `z512_n` [B, 32, 64, 64] and the orthonormal `rect_basis`, computes cheap per-shape linear-lift
residual features (zero VAE decode!) and predicts a valid 2:1 k-d tree `shape_map` [B, 2, 64, 64] in ONE pass:
  - Dilated FCN backbone produces a [B, C, 64, 64] spatial feature map.
  - A lightweight pooled split head predicts 3-way k-d actions `{0: leaf, 1: split-H, 2: split-V}` for any node
    `(r0, c0, sh, sw)` by average-pooling the feature map over `[r0:r0+sh, c0:c0+sw]` + 4D log-shape embedding.
  - At inference, decoding top-down from the 4 roots (32x32) guarantees a 100% valid 2:1 k-d tree in <5 ms!
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.latent import rect as RL


def rect_router_features(z512_n, basis):
    """Cheap per-cell router features [B, 32 + 12, 64, 64]: z512_n + mean-abs linear-lift error for all 12 shapes."""
    z = z512_n.float()
    errs = []
    for sh, sw in RL.COARSE_SHAPES:
        entry = basis[(sh, sw)]
        tok = RL.project_rect_tokens(z, sh, sw, entry)
        zhat = RL.lift_rect_tokens(tok, sh, sw, entry)
        e = (zhat - z).abs().mean(dim=1, keepdim=True)                        # [B, 1, 64, 64]
        errs.append(e)
    return torch.cat([z] + errs, dim=1).to(z512_n.dtype)                      # [B, 44, 64, 64]


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

    def __init__(self, in_ch=IN_CH, width=96, dilations=(1, 2, 4, 8, 2, 1)):
        super().__init__()
        self.in_ch, self.width = int(in_ch), int(width)
        self.stem = nn.Conv2d(self.in_ch, self.width, 3, padding=1)
        self.blocks = nn.Sequential(*[_DilBlock(self.width, d) for d in dilations])
        self.norm = nn.GroupNorm(8, self.width)
        self.shape_mlp = nn.Sequential(nn.Linear(4, self.width), nn.SiLU(), nn.Linear(self.width, self.width))
        self.head = nn.Linear(self.width, 3)                                  # {0: leaf, 1: split-H, 2: split-V}

    def encode_grid(self, feat):
        return F.silu(self.norm(self.blocks(self.stem(feat))))                # [B, W, 64, 64]

    def node_logits(self, h_1, nodes):
        """Single-image feature map `h_1` [W, 64, 64] + list of `(r0, c0, sh, sw)` -> logits [N, 3]."""
        dev = h_1.device
        pooled, s4s = [], []
        for r0, c0, sh, sw in nodes:
            pooled.append(h_1[:, r0:r0 + sh, c0:c0 + sw].mean(dim=(1, 2)))
            lsh, lsw = math.log2(sh / 2.0), math.log2(sw / 2.0)
            s4s.append((lsh, lsw, lsh + lsw, lsh - lsw))
        p = torch.stack(pooled, 0)                                            # [N, W]
        s4 = torch.tensor(s4s, dtype=p.dtype, device=dev)                     # [N, 4]
        return self.head(p + self.shape_mlp(s4))                              # [N, 3]

    def loss(self, feat, shape_maps):
        """Supervised 3-way cross-entropy over all non-(2,2) visited nodes of ground-truth `shape_maps` [B,2,64,64]."""
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
            logits = self.node_logits(h[bi], nodes)
            sh_t = torch.tensor([sh for _, _, sh, _ in nodes], dtype=torch.long, device=dev)
            sw_t = torch.tensor([sw for _, _, _, sw in nodes], dtype=torch.long, device=dev)
            logits = logits.masked_fill(~RL.valid_action_mask(sh_t, sw_t), -1e9)
            tot = tot + F.cross_entropy(logits, targets, reduction="sum")
            count += len(targets)
        return tot / max(count, 1)

    @torch.no_grad()
    def route(self, feat, leaf_bias=0.0):
        """One-pass top-down decode -> guaranteed-valid 2:1 k-d tree `shape_map` [B, 2, 64, 64]."""
        h = self.encode_grid(feat)
        b = h.shape[0]
        out = torch.zeros((b, 2, RL.SIDE, RL.SIDE), dtype=torch.long, device=h.device)
        for bi in range(b):
            frontier = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
            while frontier:
                logits = self.node_logits(h[bi], frontier)
                if leaf_bias != 0.0:
                    logits = logits.clone()
                    logits[:, RL.ACT_LEAF] += float(leaf_bias)
                sh_t = torch.tensor([sh for _, _, sh, _ in frontier], dtype=torch.long, device=h.device)
                sw_t = torch.tensor([sw for _, _, _, sw in frontier], dtype=torch.long, device=h.device)
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


def verify_rect(shape_map, check_fn, max_rounds=8):
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
            bad_b = bad.to("cpu").bool()
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
