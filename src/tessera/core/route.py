"""Router layer of the current path: per-cell rungs <-> layouts, the per-image context (Ctx: table, features, mixed-decode
scoring), the heuristic, the verify loop and the Router network definition.  Trimmed copy of qtok/route512.py: the label /
train / eval commands stay in qtok.  No trained router weights ship with this package."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import hires as H

G, L = H.G, H.L
NR = 2 * (L - 1) + 1  # rungs: 2 per coded level, plus native
NF = 2 * (L - 1) + 6


# ---------------------------------------------------------------- layouts <-> per-cell rungs ----------------------------
def rungs_of(lvl, two, vae16=False):
    """[B, G, G] layout -> per-cell rung 2 l + two (coded), NR - 1 (native).  With vae16 the 16 px rung is terminal:
    a level-5 or finer cell encodes rung 2 * LEVEL16 (= 10), so neither the 2-token-16px nor the native rung exists."""
    if vae16:
        from . import vae16 as V

        return torch.where(lvl >= V.LEVEL16, torch.full_like(lvl, 2 * V.LEVEL16), 2 * lvl + two.long())
    return torch.where(lvl >= L - 1, torch.full_like(lvl, NR - 1), 2 * lvl + two.long())


def project(r, vae16=False):
    """Per-cell rungs [1, G, G] -> a valid (lvl, two): top-down, a block stays a leaf unless one of its cells wants a
    finer level; a leaf carries 2 tokens when its finest cell wants 2 tokens at that level.  With vae16 the ladder is
    pruned: rungs at or above 2 * LEVEL16 (16 px / 2 tokens, native) project to one 16 px 1-token leaf (rung 10), and
    a 16 px leaf never carries 2 tokens."""
    if vae16:
        from . import vae16 as V

        r = r.clamp(max=2 * V.LEVEL16)
        pl = (r // 2).clamp(max=V.LEVEL16).float()[:, None]
    else:
        pl = (r // 2).clamp(max=L - 1).float()[:, None]
    rf = r.float()[:, None]
    lvl = torch.zeros_like(r)
    for l in range(L - 1):
        s = H.size(l)
        need = F.max_pool2d(pl, s).repeat_interleave(s, 2).repeat_interleave(s, 3)[:, 0] > l
        lvl = torch.where((lvl == l) & need, lvl + 1, lvl)
    two = torch.zeros_like(r, dtype=torch.bool)
    for l in range(L - 1):
        s = H.size(l)
        rmax = F.max_pool2d(rf, s).repeat_interleave(s, 2).repeat_interleave(s, 3)[:, 0]
        two |= (lvl == l) & (rmax == 2 * l + 1)
    if vae16:
        from . import vae16 as V

        two &= (lvl != V.LEVEL16)
    return lvl, two


# ---------------------------------------------------------------- features ----------------------------------------------
class Ctx:
    """Everything the router, the heuristic, the verify loop and the oracle share for one image.

    vae16=None is the shipped codec, bit-identical to before.  vae16=dict(...) turns on the P3.1 codec: the level-5
    (16 px) leaf carries the image's own FLUX.2 latent (the per-node pick baked into the codec table), the 2-token-16px
    and native rungs are pruned, and every pass/fail or cell-excess judgement is made on the MIXED decode, not the crop.
    Keys: flux (FLUX.2 wrapper), zt (its latents of x), placeholder="fill", iters=2, max_iters=4, tol=2e-3,
    repair="cheap", and optionally state={tab0, fixed1, fixed2, sel, fx, dg} from qtok.vae16.setup to avoid re-preparing.
    """

    def __init__(self, codec, judge, x, g, grain, eps_w=.3, eps_img=.1, tab=None, vae16=None, eps_loc=None):
        self.codec, self.judge, self.x, self.g, self.eps_w, self.eps_img = codec, judge, x, g, eps_w, eps_img
        self.gm = H.grain_mask(x, v2=getattr(judge, "v2", False)) if grain else None
        self.eps_loc = eps_loc  # judge v2 R1.1 local cap (None = v1)
        judge.gm = None
        if grain:
            judge.set_grain(x, self.gm)
        self.use_vae16 = vae16 is not None
        st = vae16.get("state") if self.use_vae16 else None
        if st is not None:  # a run/verdict already made fx, dg and the table: reuse them
            self.fx, self.dg = st["fx"], st["dg"]
            self.tab = tab
        else:
            self.fx = judge.j.feats(x)
            self.dg = judge.cellmap(g, x, self.fx)
            self.tab = codec.prepare(x, self.gm, judge, self.fx, self.dg) if tab is None else tab
        if self.use_vae16:
            from . import vae16 as V

            self.V, self.flux, self.zt = V, vae16["flux"], vae16["zt"]
            self.mix = dict(placeholder=vae16.get("placeholder", "fill"), iters=vae16.get("iters", 2),
                            max_iters=vae16.get("max_iters", 4), tol=vae16.get("tol", 2e-3))
            self.repair_policy = vae16.get("repair", "cheap")
            if st is not None:
                self.tab0, self.fixed1, self.fixed2 = st["tab0"], st["fixed1"], st["fixed2"]
                self.sel = st["sel"].clone()
                if self.tab is None:
                    self.tab = V.baked_tab(self.tab0, g, self.fixed1, self.fixed2, self.sel)
            else:
                self.tab0 = self.tab
                self.fixed1, self.fixed2 = self.tab0[V.LEVEL16][0][1], self.tab0[V.LEVEL16][0][2]
                self.tab, self.sel = V.pick(judge, x, g, self.tab0, self.fx, self.dg)
            self._cache = {}

    def _bake(self):
        """Rebuild the codec table after sel changed and drop the cached mixed decodes."""
        self.tab = self.V.baked_tab(self.tab0, self.g, self.fixed1, self.fixed2, self.sel)
        self._cache.clear()

    def repair(self, lvl, two, bad):
        """Cheap local repair after the pruned ladder stalls (drop/two/leaf/refine); updates sel+table in place."""
        if not self.use_vae16:
            return None
        plan = self.V.repair_plan(lvl, two, self.sel, bad, self.repair_policy)
        if plan is None:
            return None
        self.sel = plan[2]
        self._bake()
        return plan[0], plan[1]

    @torch.no_grad()
    def score(self, lvl, two):
        if self.use_vae16:  # judge the real mixed decode, cached per layout
            key = (lvl.cpu().numpy().tobytes(), two.cpu().numpy().tobytes())
            hit = self._cache.get(key)
            if hit is None:
                xh = H.Codec.decode(self.tab, lvl, two)
                y, _, _, _ = self.V.mixed_decode(xh, lvl, two, self.sel, self.zt, self.flux, **self.mix)
                d = self.judge.cellmap(y, self.x, self.fx)
                r = H.fail(d, self.dg, self.eps_w, self.eps_img, eps_loc=self.eps_loc)
                lc = H.learned_cells(self.tab, lvl, two) if getattr(self.judge, "v2", False) else None
                if lc is None or bool(lc.any()):  # R1.5: guard only on learned leaves
                    structure_bad, _ = H.structure_fail(y, self.x, self.g)
                    if lc is not None:
                        structure_bad = structure_bad & lc
                    hit = (y, r[0] | structure_bad, r[1], r[2])
                else:
                    hit = (y, r[0], r[1], r[2])
                if len(self._cache) >= 12:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = hit
            return hit
        xh = H.Codec.decode(self.tab, lvl, two)
        d = self.judge.cellmap(xh, self.x, self.fx)
        r = H.fail(d, self.dg, self.eps_w, self.eps_img, eps_loc=self.eps_loc)
        bad = r[0]
        lc = H.learned_cells(self.tab, lvl, two) if getattr(self.judge, "v2", False) else None
        if lc is None or bool(lc.any()):  # R1.5: guard only on learned leaves
            structure_bad, _ = H.structure_fail(xh, self.x, self.g)
            if lc is not None:
                structure_bad = structure_bad & lc
            bad = bad | structure_bad
        return xh, bad, r[1], r[2]

    @torch.no_grad()
    def features(self, chunk=4):
        """[1, NF, G, G]: per-rung cell excess (12), reference score, log fine-scale std, grain mask, mean colour.
        With vae16 the level-5 rungs (10, 11) are judged on the mixed decode, since that is what the codec ships."""
        lv, tw = [], []
        for l in range(L - 1):
            for t in (0, 1):
                lv.append(torch.full((1, G, G), l, dtype=torch.long, device=self.x.device))
                tw.append(torch.full((1, G, G), bool(t), device=self.x.device))
        ex = []
        for i in range(0, len(lv), chunk):
            a_, b_ = lv[i:i + chunk], tw[i:i + chunk]
            xh = torch.cat([H.Codec.decode(self.tab, a, b) for a, b in zip(a_, b_)])
            if self.use_vae16:
                parts = []
                for a, b, y in zip(a_, b_, xh):
                    if int(a.max()) >= self.V.LEVEL16:
                        y = self.V.mixed_decode(y[None], a, b, self.sel, self.zt, self.flux, **self.mix)[0][0]
                    parts.append(y)
                xh = torch.stack(parts)
            ex.append(self.judge.cellmap(xh, self.x.expand(len(xh), -1, -1, -1), self.fx) - self.dg)
        ex = torch.cat(ex)[None]  # [1, 12, G, G]
        hs = torch.log(H.hf_std(self.x) + 1e-3)[:, None]
        gm = (self.gm if self.gm is not None else H.grain_mask(self.x, v2=getattr(self.judge, "v2", False))).float()[None, None]
        col = F.avg_pool2d(self.x, H.P)
        return torch.cat((ex, self.dg[:, None], hs, gm, col), 1).float()


def heuristic(f, eps_w):
    """Per cell: the coarsest rung whose 8 x 8-cell window-averaged excess is below eps_w (else native)."""
    ex = F.avg_pool2d(f[:, :NR - 1], 7, 1, 3, count_include_pad=False)
    ok = ex <= eps_w
    r = torch.full(ok.shape[:1] + ok.shape[2:], NR - 1, dtype=torch.long, device=f.device)
    for k in reversed(range(NR - 1)):
        r = torch.where(ok[:, k], torch.full_like(r, k), r)
    return r


def verify(ctx, lvl, two, rounds=30):
    """Refine failing leaves until the layout passes (on the mixed decode when vae16 is on).  With vae16 the ladder is
    pruned (16 px is terminal), so once it stalls the cheap local repair (drop / two / leaf / refine) takes over.
    Returns (lvl, two, xh, passed, evals)."""
    for n in range(1, rounds + 1):
        xh, bad, _, _ = ctx.score(lvl, two)
        if not bad.any():
            return lvl, two, xh, True, n
        if ctx.use_vae16:
            nl, nt = ctx.V.refine_vae16(lvl, two, bad)
            if torch.equal(nl, lvl) and torch.equal(nt, two):
                rep = ctx.repair(lvl, two, bad)
                if rep is None:
                    return lvl, two, xh, False, n
                lvl, two = rep
            else:
                lvl, two = nl, nt
        else:
            nl, nt = H.refine(lvl, two, bad)
            if torch.equal(nl, lvl) and torch.equal(nt, two):
                return lvl, two, xh, False, n
            lvl, two = nl, nt
    xh, bad, _, _ = ctx.score(lvl, two)
    return lvl, two, xh, not bool(bad.any()), rounds + 1


# ---------------------------------------------------------------- network -----------------------------------------------
class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d),
                               nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d))

    def forward(self, x):
        return x + self.b(x)


class Router(nn.Module):
    def __init__(self, c=64, dil=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, NF, 1, 1))
        self.register_buffer("sd", torch.ones(1, NF, 1, 1))
        self.inp = nn.Conv2d(NF, c, 3, padding=1)
        self.body = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, NR - 1, 1))

    def forward(self, f):
        return self.out(self.body(self.inp((f - self.mu) / self.sd)))  # [B, NR-1, G, G]: logit P(rung > k)

    @torch.no_grad()
    def rungs(self, f, tau=.5):
        return (torch.sigmoid(self(f)) > tau).sum(1)
