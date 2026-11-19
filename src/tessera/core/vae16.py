"""P3.1: the image's own FLUX.2 latent as a 16 px leaf ("vae16"), and the mixed decode that judges it.

A level-5 leaf in hires' 64-cell grid is 2 x 2 cells = 16 px = 2 x 2 FLUX.2 latent cells = 32 x 4 floats = one token
of 128 floats, the same width as a fixed code.  The encoder already runs the FLUX.2 encoder to make the judge's
reference g = vae.dec(vae.enc(x)), so the latents are free: one encode per image supplies every vae16 leaf.

Pick (P0.3, per 16 px node): the candidate decode is the crop of the VAE decode, whose excess over the reference is
zero, so it wins wherever the fixed 1-token code is not already better.  The pick is baked into the level-5 slot of
Codec.prepare's table, so the ordinary layout search (refine / coarsen / polish) runs on the crop decode unchanged.

Mixed decode (what the oracle and verify must judge, not the crop):
  1. pixel-decode every non-vae16 leaf with its own code; the vae16 leaves' pixels must be estimated from transmitted
     data only.  Placeholders:
       latent  decode the transmitted latents with the other latent cells zeroed, keep the vae16 pixels; legal and the
               best first guess (default)
       fill    cell-average the decoded neighbours and propagate those averages into the hole; legal
       zero    a black hole (the P1 reference)
       paste   the crop of the VAE decode (DIAGNOSTIC ONLY: needs the whole latent grid, not just the token)
  2. encode that image with the FLUX.2 encoder;
  3. overwrite the latents of the vae16 leaves with the transmitted ones;
  4. decode, and paste only the vae16 leaves' pixels;
  5. repeat 3-4 until the pasted pixels stop moving (2 to 4 passes; the P1 reference used 2).
With every 16 px leaf vae16 the mixed decode is exactly g on every pass.

Ladder: with vae16 on, the 2-token 16 px rung and the native 8 px rung are removed from the candidate ladder (a level-5
leaf is terminal: it carries the latent at 1 token).  The layout search judges the crop decode; the reported pass/fail
is always the mixed decode.  Residual failures are repaired at leaf granularity, cheapest first: select a fixed 16 px
leaf (free), split a failing leaf into 16 px vae16 leaves (+1..3 tokens for a 32 px leaf), grow vae16 into the fixed
leaves touching a failing vae16 leaf (moves the seam outward), then last refine the failing leaf to exact native 8 px
cells (+5 tokens per 16 px).  Only the worst leaves are repaired per round, then the mixed decode is re-judged.

Opt-in: nothing here runs unless called; qtok.hires and the shipped codec are untouched.
"""
import time

import torch
import torch.nn.functional as F

from . import hires as H

LEVEL16 = H.L - 2  # 5: 2 cells = 16 px
S16 = H.size(LEVEL16)  # 2 cells per level-16 leaf side
N16 = H.G // S16  # 32 level-16 leaves per side
TOK16 = S16 * S16 * 32  # 128 floats = one token
assert S16 * H.P == 16, (S16, H.P)
assert TOK16 == H.K
assert LEVEL16 == 5 and H.L == 7 and H.P == 8 and H.G == 64
PLACEHOLDERS = ("latent", "fill", "zero", "paste")


# ---------------------------------------------------------------- masks and the pick --------------------------------------
def up_px(m):
    """[N16, N16] level-16 leaf mask -> [1, 1, S, S] pixel mask."""
    return m.repeat_interleave(S16 * H.P, 0).repeat_interleave(S16 * H.P, 1)[None, None]


def up_cell(m):
    """[N16, N16] -> [1, 1, 64, 64]: the FLUX.2 latent grid (8 px per latent cell) is the hires 8 px cell grid."""
    return m.repeat_interleave(S16, 0).repeat_interleave(S16, 1)[None, None]


def vae16_cells(lvl, sel):
    """[G, G] bool: the cells covered by level-5 leaves that carry the latent in this layout."""
    return (H.masks(lvl)[LEVEL16][0] & sel).repeat_interleave(S16, 0).repeat_interleave(S16, 1)


@torch.no_grad()
def pick(judge, x, g, tab, fx, dg):
    """Bake the per-16-px-node pick into the table: crop of the VAE decode where its mean judge cell excess is lower
    than the fixed 1-token code's.  Its excess is zero by definition, so it wins wherever the fixed code is over zero.
    Returns (tab, sel [N16, N16] bool) with sel True where the leaf's token is the FLUX.2 latent."""
    fixed1, fixed2 = tab[LEVEL16][0][1], tab[LEVEL16][0][2]
    cex1 = judge.cellmap(fixed1, x, fx) - dg  # [1, G, G]
    e1 = F.avg_pool2d(cex1[:, None], S16)[0, 0]  # [N16, N16]
    sel = e1 > 0
    return baked_tab(tab, g, fixed1, fixed2, sel), sel


def baked_tab(tab, g, fixed1, fixed2, sel):
    """Table with level-5 code 1 replaced by the crop wherever the node is selected, and code 2 (the 2-token fixed
    code, normally unused because the search ladder prunes that rung) left as the fixed 2-token decode so a repair can
    fall back to it.  The layout search pastes this table, so reaching level 5 is what makes a leaf vae16."""
    tb = dict(tab)
    used, selected = tab[LEVEL16][1], tab[LEVEL16][2]
    tb[LEVEL16] = ({1: torch.where(up_px(sel), g, fixed1), 2: torch.where(up_px(sel), g, fixed2)}, used, selected)
    return tb


# ---------------------------------------------------------------- mixed decode --------------------------------------------
@torch.no_grad()
def _fill_vae16(xh, pm, rounds=4):
    """Estimate the vae16 leaves' pixels from the decoded neighbours: cell-average them and propagate those averages
    into the hole, one 8 px cell per round.  Only transmitted pixels are used, so this is a legal decoder guess."""
    known = (~pm).float()  # [1, 1, S, S]
    k8 = F.avg_pool2d(known, H.P)  # [1, 1, G, G] fraction decoded per 8 px cell
    y8 = F.avg_pool2d(xh * known, H.P) / k8.clamp_min(1e-6)
    k = (k8 > .5).float()
    y8 = torch.where(k > .5, y8, torch.zeros_like(y8))
    for _ in range(rounds):
        ws = F.avg_pool2d(k, 3, 1, 1)
        ys = F.avg_pool2d(y8 * k, 3, 1, 1)
        fill = ys / ws.clamp_min(1e-6)
        add = (ws > 0) & (k < .5)
        y8 = torch.where(add, fill, y8)
        k = ((k > .5) | add).float()
    return torch.where(pm, F.interpolate(y8, scale_factor=H.P, mode="nearest"), xh)


@torch.no_grad()
def _placeholder(xh, pm, cm, zt, flux, kind):
    if kind == "zero":
        return torch.where(pm, torch.zeros_like(xh), xh)
    if kind == "paste":  # diagnostic only: uses the crop of the whole VAE decode at the vae16 leaves
        return xh
    if kind == "fill":
        return _fill_vae16(xh, pm)
    if kind == "latent":  # decode the transmitted latents with the other latent cells zeroed; legal
        z0 = torch.where(cm, zt, torch.zeros_like(zt))
        with torch.autocast("cuda", torch.bfloat16, enabled=xh.is_cuda):
            y0 = flux.dec(z0).float().clamp(-1, 1)
        return torch.where(pm, y0, xh)
    raise ValueError(kind)


@torch.no_grad()
def mixed_decode(xh, lvl, two, sel, zt, flux, iters=2, max_iters=4, tol=2e-3, placeholder="latent"):
    """The real decoder with vae16 on.  xh = paste decode of the layout (non-vae16 leaves only), zt = the image's own
    FLUX.2 latents, sel = per-node vae16 pick.  Iterates at least `iters` and at most `max_iters` encode/decode passes,
    stopping when the pasted vae16 pixels move less than `tol` (in [-1, 1] units).
    Returns (y, pixel_mask, latent_mask, passes)."""
    m5 = H.masks(lvl)[LEVEL16][0] & sel  # [N16, N16]: vae16 leaves that exist in this layout
    if not bool(m5.any()):
        return xh, None, None, 0
    pm = up_px(m5)  # [1, 1, S, S]
    cm = up_cell(m5)  # [1, 1, 64, 64]
    y = _placeholder(xh, pm, cm, zt, flux, placeholder)
    n = 0
    with torch.autocast("cuda", torch.bfloat16, enabled=xh.is_cuda):
        while n < max_iters:
            z = torch.where(cm, zt, flux.enc(y))
            yn = torch.where(pm, flux.dec(z).float().clamp(-1, 1), xh)
            n += 1
            delta = float((yn - y).abs().max()) if n >= iters and tol is not None else 1e9
            y = yn
            if delta < tol:
                break
    return y, pm, cm, n


# ---------------------------------------------------------------- the ladder with vae16 on ---------------------------------
def refine_vae16(lvl, two, bad):
    """One rung up for every leaf under a bad cell, without the dominated rungs: a level-5 (16 px) leaf is terminal
    (it already carries the latent at 1 token; there is no 2-token and no native 8 px rung)."""
    nl, nt = lvl.clone(), two.clone()
    for l, m in enumerate(H.masks(lvl)[:-1]):
        if l == LEVEL16:
            continue
        s = H.size(l)
        hit = (m[0] & H.block_any(bad, s)).repeat_interleave(s, 0).repeat_interleave(s, 1)[None]
        up = hit & ~two
        sp = hit & ~up
        nt = (nt | up) & ~sp
        nl = torch.where(sp, lvl + 1, nl)
    return nl, nt


# ---------------------------------------------------------------- layout helpers -------------------------------------------
def _rebuild(lvl, target):
    """Top-down quadtree consistent with per-cell target depths: a level-l block splits iff one of its cells needs a
    depth >= l + 1.  Same tree as T.from_splits, from a depth map."""
    cur = torch.zeros_like(lvl)
    for l in range(H.L - 1):
        s = H.size(l)
        node = H.masks(cur)[l][0]
        need = F.max_pool2d(target.float()[:, None], s)[0, 0] > l
        split = node & need
        if bool(split.any()):
            cells = split.repeat_interleave(s, 0).repeat_interleave(s, 1)[None]
            cur = torch.where(cells, torch.full_like(cur, l + 1), cur)
    return cur


def _node_box(l, i, j):
    """Leaf (l, i, j) -> its box on the 32 x 32 level-16 node grid (y0, x0, y1, x1), half-open."""
    if l <= LEVEL16:
        per = H.size(l) // S16  # nodes per leaf side (>= 1)
        return i * per, j * per, (i + 1) * per, (j + 1) * per
    return i // S16, j // S16, i // S16 + 1, j // S16 + 1  # a native 8 px cell: half a node


def _leaf_cells(l, i, j, dev="cpu"):
    """[G, G] bool of the cells covered by leaf (l, i, j); a native cell is widened to its 16 px node (its merge unit)."""
    if l < H.L - 1:
        s = H.size(l)
        y0, x0, y1, x1 = i * s, j * s, (i + 1) * s, (j + 1) * s
    else:
        y0, x0 = (i // S16) * S16, (j // S16) * S16
        y1, x1 = y0 + S16, x0 + S16
    cells = torch.zeros((H.G, H.G), dtype=torch.bool, device=dev)
    cells[y0:y1, x0:x1] = True
    return cells


def _apply(lvl, two, sel, vae_cells=None, refine_cells=None):
    """Apply a vae16 conversion (cells that become level-16 vae16 leaves) and a refine (cells that become native)."""
    target = lvl.clone()
    if vae_cells is not None:
        target = torch.where(vae_cells, torch.full_like(target, LEVEL16), target)
    if refine_cells is not None:
        target = torch.where(refine_cells, torch.full_like(target, H.L - 1), target)
    new_lvl = _rebuild(lvl, target)
    new_two = torch.where(new_lvl == lvl, two, torch.zeros_like(two))
    if vae_cells is not None:  # a leaf that becomes vae16 cannot also carry a second fixed-code token
        new_two = new_two & ~vae_cells
    new_sel = sel.clone()
    if vae_cells is not None:
        new_sel |= F.max_pool2d(vae_cells[None, None].float(), S16)[0, 0] > 0
    return new_lvl, new_two, new_sel


def _covering_leaves(lvl, cells):
    """[(l, i, j)] leaves of layout that contain any True cell of the [G, G] mask."""
    out = []
    for l, m in enumerate(H.masks(lvl)):
        s = H.size(l)
        hit = m[0] & H.block_any(cells, s)
        for i, j in hit.nonzero().tolist():
            out.append((l, i, j))
    return out


def boundary_cells(lvl, sel):
    """[G, G] bool: cells whose 3 x 3 neighbourhood contains both vae16 and non-vae16 cells (the mixed-decode seam)."""
    v = vae16_cells(lvl, sel).float()
    mx = F.max_pool2d(v[None, None], 3, 1, 1)[0, 0]
    mn = -F.max_pool2d(-v[None, None], 3, 1, 1)[0, 0]
    return mx > mn


# ---------------------------------------------------------------- local repair ---------------------------------------------
POLICIES = {
    "cheap": {"drop", "two", "leaf", "refine"},
    "cheapgrow": {"drop", "two", "leaf", "grow", "refine"},
    "drop": {"drop"},
    "two": {"two"},
    "leaf": {"leaf"},
    "grow": {"grow"},
    "refine": {"refine"},
    "boundary": {"drop", "two", "leaf", "refine"},
    "none": set(),
}


def _vae_neighbour_frac(sel, i, j):
    """Fraction of the 3 x 3 node neighbourhood (self included) that is already vae16."""
    box = sel[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
    return float(box.float().mean())


def _set_two(two, l, i, j, val):
    nt = two.clone()
    nt[0, i * H.size(l), j * H.size(l)] = val
    return nt


def repair_plan(lvl, two, sel, bad, policy="cheap", max_leaves=12, frozen=None):
    """Cheapest repair for the worst failing leaves, at leaf granularity.  The key fact: a 16 px leaf costs the same
    1 token whether it carries the FLUX latent (vae16) or the fixed 1-token code, and +1 more token for the fixed
    2-token code.  So a failing vae16 leaf can be repaired DOWN the rate ladder for free before buying tokens.

    Options, cheapest first:
      drop   put the fixed 1-token code back (0 tokens): the leaf is no longer part of the VAE grid, so it stops
             contributing a mixed-decode seam; if its fixed code is good enough the cell passes at no cost;
      two    put the fixed 2-token code back (+1 token): its mean excess is -0.02, and being a plain fixed decode it
             cannot create a VAE context seam at all;
      leaf   (coarse fixed leaf only) split it into 16 px vae16 leaves (+1..3 tokens for a 32 px leaf) - cheaper than
             native refinement and improves the local context;
      grow   (vae16 leaf only, cheapgrow policy) convert the fixed leaves touching it to vae16;
      refine split the failing leaf into exact native 8 px cells (+5 tokens per 16 px leaf).
    policy: cheap = drop/two/leaf/refine (no grow: growing a seam indefinitely was what inflated the first smoke);
    cheapgrow adds grow; drop/two/leaf/grow/refine force one kind.  Only `max_leaves` worst leaves are touched per
    round.
    frozen (R4 bug 20): bool [G, G] cells of leaves already repaired in an earlier round; such leaves are not touched
    again, so a round cannot undo the previous round's repair (the leaf->drop->two oscillation seen on 6/124 images).
    Returns (new_lvl, new_two, new_sel, info) or None when nothing changes; info["leaves"] lists the touched leaves."""
    allowed = POLICIES.get(policy)
    if not allowed:
        return None
    failing = []
    for l, m in enumerate(H.masks(lvl)):
        s = H.size(l)
        hit = m[0] & H.block_any(bad, s)
        for i, j in hit.nonzero().tolist():
            if l == H.L - 1:  # a native leaf is exact; a bad cell there is a window culprit, repairing it is useless
                continue
            n_bad = int((bad & _leaf_cells(l, i, j, dev=bad.device)).sum())
            if frozen is not None and bool((_leaf_cells(l, i, j, dev=bad.device) & frozen).any()):
                continue
            failing.append((n_bad, l, i, j))
    if not failing:
        return None
    failing.sort(reverse=True)
    failing = failing[:max_leaves]
    base = int(H.count(lvl, two))
    vcells = vae16_cells(lvl, sel)
    convs, refin, drops, twos, picked = [], [], [], [], []
    for n_bad, l, i, j in failing:
        cells = _leaf_cells(l, i, j, dev=lvl.device)
        is_vae16 = l == LEVEL16 and bool(sel[i, j])
        opts = []
        if is_vae16:
            if "drop" in allowed:
                opts.append((0, 0, "drop", (l, i, j)))
            if "two" in allowed:
                opts.append((int(H.count(lvl, _set_two(two, l, i, j, True))) - base, 1, "two", (l, i, j)))
            if "grow" in allowed:
                y0, x0, y1, x1 = _node_box(l, i, j)
                box = torch.zeros((H.G, H.G), dtype=torch.bool, device=lvl.device)
                box[max(0, (y0 - 1) * S16):min(H.G, (y1 + 1) * S16), max(0, (x0 - 1) * S16):min(H.G, (x1 + 1) * S16)] = True
                grow = box & ~vcells
                if bool(grow.any()):
                    gcells = torch.zeros_like(cells)
                    for ll, ii, jj in _covering_leaves(lvl, grow):
                        gcells |= _leaf_cells(ll, ii, jj, dev=lvl.device)
                    if bool(gcells.any()) and not torch.equal(gcells, cells):
                        opts.append((int(H.count(*_apply(lvl, two, sel, gcells, None)[:2])) - base, 3, "grow", gcells))
            if "refine" in allowed:
                opts.append((int(H.count(*_apply(lvl, two, sel, None, cells)[:2])) - base, 2, "refine", cells))
        else:
            if "two" in allowed and l == LEVEL16 and not bool(two[0, i * S16, j * S16]):
                opts.append((int(H.count(lvl, _set_two(two, l, i, j, True))) - base, 0, "two", (l, i, j)))
            if "leaf" in allowed and l < LEVEL16:  # a coarse fixed leaf: split it into 16 px vae16 leaves
                opts.append((int(H.count(*_apply(lvl, two, sel, cells, None)[:2])) - base, 1, "leaf", cells))
            if "refine" in allowed:
                opts.append((int(H.count(*_apply(lvl, two, sel, None, cells)[:2])) - base, 2, "refine", cells))
        if not opts:
            continue
        cost, rank, kind, payload = min(opts, key=lambda o: (o[0], o[1]))
        picked.append(kind)
        if kind == "drop":
            drops.append(payload)
        elif kind == "two":
            twos.append(payload)
        elif kind in ("leaf", "grow"):
            convs.append(payload)
        else:
            refin.append(payload)
    vae_union = torch.zeros((H.G, H.G), dtype=torch.bool, device=lvl.device)
    for c in convs:
        vae_union |= c
    refine_union = torch.zeros((H.G, H.G), dtype=torch.bool, device=lvl.device)
    for c in refin:
        refine_union |= c
    nl, nt, ns = _apply(lvl, two, sel, vae_union if bool(vae_union.any()) else None,
                        refine_union if bool(refine_union.any()) else None)
    for l, i, j in drops:
        ns[i, j] = False
        nt[0, i * S16, j * S16] = False
    for l, i, j in twos:
        ns[i, j] = False
        nt[0, i * S16, j * S16] = True
    if torch.equal(nl, lvl) and torch.equal(nt, two) and torch.equal(ns, sel):
        return None
    kinds = {k: picked.count(k) for k in ("drop", "two", "leaf", "grow", "refine")}
    info = dict(round=None, failing=len(failing), kinds=kinds, cost=int(H.count(nl, nt)) - base,
                tokens=int(H.count(nl, nt)), frozen=0 if frozen is None else int(frozen.sum()))
    return nl, nt, ns, info


def boundary_polish(lvl, two, sel, max_delta=2, max_leaves=24):
    """Greedy seam cleanup: convert the non-vae16 leaves that touch a vae16 leaf to vae16 when the token delta is small
    (<= max_delta), cheapest first.  This is the plan's "turn the vae16 neighbours' ring into vae16 leaves" done at leaf
    granularity and priced per leaf.  Returns (lvl, two, sel, info) or None."""
    vcells = vae16_cells(lvl, sel)
    bnd = boundary_cells(lvl, sel) & ~vcells
    if not bool(bnd.any()):
        return None
    base = int(H.count(lvl, two))
    opts, seen = [], set()
    for l, i, j in _covering_leaves(lvl, bnd):
        if (l, i, j) in seen:
            continue
        seen.add((l, i, j))
        if l == LEVEL16 and bool(sel[i, j]):
            continue
        cells = _leaf_cells(l, i, j, dev=lvl.device)
        nl, nt, _ = _apply(lvl, two, sel, cells, None)
        d = int(H.count(nl, nt)) - base
        if d <= max_delta:
            opts.append((d, l, i, j))
    if not opts:
        return None
    opts.sort(key=lambda o: (o[0], -o[1]))
    opts = opts[:max_leaves]
    union = torch.zeros((H.G, H.G), dtype=torch.bool, device=lvl.device)
    for d, l, i, j in opts:
        union |= _leaf_cells(l, i, j, dev=lvl.device)
    nl, nt, ns = _apply(lvl, two, sel, union, None)
    return nl, nt, ns, dict(kind="boundary", converted=len(opts), cost=int(H.count(nl, nt)) - base)


# ---------------------------------------------------------------- search and verdict ----------------------------------------
def layout_stats(lvl, two, sel):
    m5 = H.masks(lvl)[LEVEL16][0]
    return dict(n_vae16=int((m5 & sel).sum()), n_level5=int(m5.sum()), n_native=int(H.masks(lvl)[-1][0].sum()),
                leaves={H.size(l) * H.P: int(m.sum()) for l, m in enumerate(H.masks(lvl))})


def _structure_bad(judge, tab, y, x, g, lv, tw):
    """R1.5: judge v2 applies the structure guard only to learned/pretrained cells; FLUX.2 and fixed leaves are off."""
    if not getattr(judge, "v2", False) or not any(tab[l][2] is not None for l in range(H.L - 1)):
        return torch.zeros_like(H.masks(lv)[0][0]) if getattr(judge, "v2", False) else H.structure_fail(y, x, g)[0]
    lc = H.learned_cells(tab, lv, tw)
    return torch.zeros_like(lc) if not bool(lc.any()) else (H.structure_fail(y, x, g)[0] & lc)


def setup(codec, judge, x, g, zt, flux, grain=True):
    """One image's shared state: judge maps, the prepared table, and the per-node vae16 pick."""
    gm = H.grain_mask(x, v2=getattr(judge, "v2", False)) if grain else None
    judge.gm = None
    if grain:
        judge.set_grain(x, gm)
    if getattr(judge, "v3", False):
        judge.set_floor(x, g) if getattr(judge, "reference_relative", False) else judge.set_floor(x)
    fx = judge.j.feats(x)
    dg = judge.cellmap(g, x, fx)
    tab0 = codec.prepare(x, gm, judge, fx, dg)
    fixed1, fixed2 = tab0[LEVEL16][0][1], tab0[LEVEL16][0][2]
    tab, sel = pick(judge, x, g, tab0, fx, dg)
    return dict(codec=codec, judge=judge, x=x, g=g, zt=zt, flux=flux, fx=fx, dg=dg, tab0=tab0, tab=tab,
                fixed1=fixed1, fixed2=fixed2, sel=sel, gm=gm)


def search_layout(st, eps_w=.3, eps_img=.1, polish=150, rounds=30, down_rounds=20, search="crop", mix=None,
                  margin_w=0., margin_img=0., eps_loc=None):
    """The hires layout search with vae16 on (crop-decode judged unless search='mixed').  A positive margin makes the
    search stricter than the final judge, leaving the crop-optimistic layout headroom for the mixed decode.  Adds lvl,
    two, tab, evals, pre_polish, post_polish, crop (the crop-judged layout verdict) to st."""
    eps_w, eps_img = eps_w - margin_w, eps_img - margin_img
    judge, x, g = st["judge"], st["x"], st["g"]
    dev = x.device
    mix = mix or {}
    mixed_search = search == "mixed"
    cache, ver = {}, [0]
    lvl = torch.zeros((1, H.G, H.G), dtype=torch.long, device=dev)
    two = torch.zeros_like(lvl, dtype=torch.bool)
    tab, sel = st["tab"], st["sel"]

    def terminal(lv):
        return (H.masks(lv)[LEVEL16][0] & sel).repeat_interleave(S16, 0).repeat_interleave(S16, 1)

    def score(lv, tw, mixed):
        key = (mixed, ver[0], tuple(sorted(mix.items())), lv.cpu().numpy().tobytes(), tw.cpu().numpy().tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
        xh = H.Codec.decode(tab, lv, tw)
        passes = 0
        if mixed:
            y, _, _, passes = mixed_decode(xh, lv, tw, sel, st["zt"], st["flux"], **mix)
        else:
            y = xh
        d = judge.cellmap(y, x, st["fx"])
        r = H.fail(d, st["dg"], eps_w, eps_img, eps_loc=eps_loc)
        jbad, worst, img = r[0], r[1], r[2]
        sbad = _structure_bad(judge, tab, y, x, g, lv, tw)
        fbad = judge.floor_bad(y) if getattr(judge, "v3", False) else torch.zeros_like(jbad)
        wbad = judge.window_bad(y) if getattr(judge, "win", False) else None
        wb = wbad if wbad is not None else torch.zeros_like(jbad)
        res = dict(bad=(jbad | sbad | fbad | wb) & ~terminal(lv), worst=worst, img=img, jbad=jbad, sbad=sbad,
                   fbad=fbad, wbad=wb, passes=passes)
        cache[key] = res
        return res

    nev = 0
    for _ in range(rounds):  # top-down: split/climb until every window passes (crop decode unless search=mixed)
        bad = score(lvl, two, mixed_search)["bad"]
        nev += 1
        if not bad.any():
            break
        nl, nt = refine_vae16(lvl, two, bad)
        if torch.equal(nl, lvl) and torch.equal(nt, two):
            # level-5 leaves are terminal; the free upgrade is to put the latent on a bad fixed-code level-5 leaf
            upg = H.masks(lvl)[LEVEL16][0] & ~sel & (F.max_pool2d(bad[None, None].float(), S16)[0, 0] > 0)
            if bool(upg.any()):
                sel = sel | upg
                tab = baked_tab(st["tab0"], g, st["fixed1"], st["fixed2"], sel)
                ver[0] += 1
                cache.clear()
                continue
            break
        lvl, two = nl, nt
    ok = lambda c, b: not bool(H.block_any(b, H.size(c[1]))[c[2], c[3]])
    for _ in range(down_rounds):  # bottom-up merges that still leave no fixable bad cell
        cand = H.candidates(lvl, two)
        if not cand:
            break
        bad = score(*H.coarsen(lvl, two, cand), mixed=mixed_search)["bad"]
        keep = [c for c in cand if ok(c, bad)]
        nev += 1
        for _ in range(10):
            if not keep:
                break
            nl, nt = H.coarsen(lvl, two, keep)
            bad = score(nl, nt, mixed_search)["bad"]
            nev += 1
            if not bad.any():
                break
            near = F.max_pool2d(bad[None, None].float(), 17, 1, 8)[0, 0] > 0
            k2 = [c for c in keep if ok(c, near)]
            keep = k2 if len(k2) < len(keep) else keep[: len(keep) // 2]
        if not keep or bad.any():
            break
        lvl, two = nl, nt
    pre = int(H.count(lvl, two))
    tried, npol = set(), 0
    fast_pol = (hasattr(judge, "evaluate_batch") and not mixed_search and getattr(judge, "v2", False)
                and not any(tab[l][2] is not None for l in range(H.L - 1))
                and not getattr(judge, "v3", False) and not getattr(judge, "win", False))
    key_fn = lambda c: (c[1], {"merge1": 0, "merge": 1, "drop": 2}[c[0]])
    while npol < polish:  # needless splits: each remaining step down alone, biggest leaves first
        cand = [c for c in sorted(H.candidates(lvl, two), key=key_fn) if c not in tried]
        if not cand:
            break
        if fast_pol:
            batch = cand[:min(8, polish - npol)]
            layouts = [H.coarsen(lvl, two, [c]) for c in batch]
            xhs = [H.Codec.decode(tab, nl, nt) for nl, nt in layouts]
            res_b = judge.evaluate_batch(xhs)
            npol += len(batch)
            for c, (nl, nt), (_, _, bad8) in zip(batch, layouts, res_b):
                tried.add(c)
                d_b = torch.where(bad8.to(dev), 1e4, 0.0)[None]
                jbad = H.fail(d_b, st["dg"], eps_w, eps_img, eps_loc=eps_loc)[0]
                if not (jbad & ~terminal(nl)).any():
                    lvl, two = nl, nt
                    break
        else:
            c = cand[0]
            tried.add(c)
            nl, nt = H.coarsen(lvl, two, [c])
            bad = score(nl, nt, mixed_search)["bad"]
            npol += 1
            if not bad.any():
                lvl, two = nl, nt
    post = int(H.count(lvl, two))
    crop = score(lvl, two, False)
    crop_raw = crop["jbad"] | crop["sbad"] | crop["fbad"] | crop.get("wbad", torch.zeros_like(crop["jbad"]))
    st.update(lvl=lvl, two=two, tab=tab, sel=sel, evals=nev + npol, pre_polish=pre, post_polish=post,
              crop=dict(tokens=post, passed=not bool(crop_raw.any()), worst=crop["worst"], img=crop["img"],
                        bad_cells=int(crop_raw.sum()),
                        boundary=int((crop_raw & boundary_cells(lvl, sel)).sum().item())))
    return st


def verdict(st, eps_w=.3, eps_img=.1, polish=150, iters=2, max_iters=4, tol=2e-3, placeholder="latent",
            repair="cheap", repair_rounds=8, repair_max_leaves=12, boundary_max_delta=2, budget=3, fallback=True,
            eps_loc=None, frozen_repairs=False, force_native_after=2):
    """Mixed-decode verdict and local repair on the searched layout.  Returns the result dict (passed/tokens/...)."""
    judge, x, g, flux, zt = st["judge"], st["x"], st["g"], st["flux"], st["zt"]
    dev = x.device
    lvl, two, sel, tab = st["lvl"].clone(), st["two"].clone(), st["sel"].clone(), st["tab"]
    cache, ver = {}, [0]

    def fin():
        xh = H.Codec.decode(tab, lvl, two)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y, _, _, passes = mixed_decode(xh, lvl, two, sel, zt, flux, iters, max_iters, tol, placeholder)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_dec = time.time() - t0
        d = judge.cellmap(y, x, st["fx"])
        r = H.fail(d, st["dg"], eps_w, eps_img, eps_loc=eps_loc)
        jbad, worst, img = r[0], r[1], r[2]
        sbad = _structure_bad(judge, tab, y, x, g, lvl, two)
        fbad = judge.floor_bad(y) if getattr(judge, "v3", False) else torch.zeros_like(jbad)
        wbad = judge.window_bad(y) if getattr(judge, "win", False) else None
        wb = wbad if wbad is not None else torch.zeros_like(jbad)
        bad = jbad | sbad | fbad | wb
        bnd = boundary_cells(lvl, sel)
        return dict(xh=xh, y=y, bad=bad, jbad=jbad, sbad=sbad, fbad=fbad, wbad=wb, worst=worst, img=img, t_dec=t_dec, passes=passes,
                    bad_boundary=int((bad & bnd).sum()), bad_interior=int((bad & ~bnd).sum()),
                    bad_in_vae16=int((bad & vae16_cells(lvl, sel)).sum()),
                    bad_in_fixed=int((bad & ~vae16_cells(lvl, sel)).sum()))

    cur = fin()
    pre_repair = dict(bad_share=float(cur["bad"].float().mean()), bad_cells=int(cur["bad"].sum()),
                      judge_bad=int(cur["jbad"].sum()), structure_bad=int(cur["sbad"].sum()),
                      worst=float(cur["worst"]), img=float(cur["img"]), bad_boundary=cur["bad_boundary"],
                      bad_interior=cur["bad_interior"], bad_in_vae16=cur["bad_in_vae16"],
                      bad_in_fixed=cur["bad_in_fixed"], passes=cur["passes"],
                      floor_bad=int(cur["fbad"].sum()), window_bad=int(cur.get("wbad", torch.zeros_like(cur["jbad"])).sum()),
                      floor=judge.floor_stats(cur["y"]) if getattr(judge, "v3", False) else None)
    repairs, fallback = [], False
    # bug 20 (R4): freeze leaves already repaired so a later round cannot undo an earlier repair (the
    # leaf -> drop -> two oscillation), and force native refinement when the failing set stops shrinking.
    frozen = torch.zeros((H.G, H.G), dtype=torch.bool, device=dev) if frozen_repairs else None
    stall, prev_bad = 0, int(cur["bad"].sum())
    for r in range(repair_rounds):
        if not cur["bad"].any():
            break
        forced = bool(frozen_repairs and force_native_after and stall >= force_native_after)
        plan = None
        if repair == "boundary" and r < budget:
            plan = boundary_polish(lvl, two, sel, boundary_max_delta)
        if plan is None:
            plan = repair_plan(lvl, two, sel, cur["bad"], "refine" if forced else repair, repair_max_leaves,
                               frozen=None if forced else frozen)
        if plan is None:
            break
        old_lvl, old_two, old_sel = lvl, two, sel
        lvl, two, sel = plan[0], plan[1], plan[2]
        info = dict(plan[3])
        info["round"], info["stalled"], info["forced_native"] = r, stall, forced
        if frozen is not None and not forced:  # cells whose layout or vae16 selection changed = the leaves touched
            changed = ((old_lvl != lvl) | (old_two != two))[0]
            new_sel = sel
            if new_sel.shape == old_sel.shape:
                sel_changed = (new_sel != old_sel).repeat_interleave(S16, 0).repeat_interleave(S16, 1)
                changed = changed | sel_changed
            frozen |= changed
        repairs.append(info)
        tab = baked_tab(st["tab0"], g, st["fixed1"], st["fixed2"], sel)
        ver[0] += 1
        cache.clear()
        cur = fin()
        nb = int(cur["bad"].sum())
        stall = 0 if nb < prev_bad else stall + 1
        prev_bad = nb
    if cur["bad"].any() and fallback:
        lvl = _rebuild(lvl, torch.full_like(lvl, LEVEL16))
        two = torch.zeros_like(two)
        sel = torch.ones((N16, N16), dtype=torch.bool, device=dev)
        tab = baked_tab(st["tab0"], g, st["fixed1"], st["fixed2"], sel)
        ver[0] += 1
        cache.clear()
        fallback = True
        cur = fin()
    out = dict(y=cur["y"], xh=cur["xh"], tokens=int(H.count(lvl, two)), lvl=lvl, two=two, sel=sel, tab=tab,
               pre_repair=pre_repair, repairs=repairs, fallback=fallback, t_mixed=cur["t_dec"],
               crop=st.get("crop"), post_polish=st.get("post_polish"), pre_polish=st.get("pre_polish"),
               mix_passes=cur["passes"], passed=not bool(cur["bad"].any()), worst=cur["worst"], img=cur["img"],
               final_bad_share=float(cur["bad"].float().mean()), final_bad_cells=int(cur["bad"].sum()),
               floor_bad=int(cur["fbad"].sum()), window_bad=int(cur.get("wbad", torch.zeros_like(cur["jbad"])).sum()),
               floor=judge.floor_stats(cur["y"]) if getattr(judge, "v3", False) else None)
    out.update(layout_stats(lvl, two, sel))
    return out


def run_oracle_vae16(codec, judge, x, g, zt, flux, eps_w=.3, eps_img=.1, polish=150, rounds=30, down_rounds=20,
                     search="crop", iters=2, max_iters=4, tol=2e-3, placeholder="latent", repair="cheap",
                     repair_rounds=8, repair_max_leaves=12, boundary_max_delta=2, grain=True,
                     margin_w=0., margin_img=0., eps_loc=None, frozen_repairs=False, force_native_after=2):
    """The hires oracle with vae16 on.  search='crop' runs the layout search on the crop decode (cheap, as the pilot
    did); 'mixed' judges every search step on the mixed decode instead.  The returned pass flag, worst and img are
    always measured on the mixed decode, and failing leaves are locally repaired before the verdict."""
    mix = dict(iters=iters, max_iters=max_iters, tol=tol, placeholder=placeholder)
    st = setup(codec, judge, x, g, zt, flux, grain)
    search_layout(st, eps_w, eps_img, polish, rounds, down_rounds, search, mix, margin_w, margin_img, eps_loc=eps_loc)
    out = verdict(st, eps_w, eps_img, polish, **mix, repair=repair, repair_rounds=repair_rounds,
                  repair_max_leaves=repair_max_leaves, boundary_max_delta=boundary_max_delta, eps_loc=eps_loc,
                  frozen_repairs=frozen_repairs, force_native_after=force_native_after)
    out["crop"] = st["crop"]
    out["pre_polish"] = st["pre_polish"]
    out["post_polish"] = st["post_polish"]
    out["evals"] = st["evals"]
    out["fx"] = st["fx"]
    out["dg"] = st["dg"]
    out["pick_share"] = float(st["sel"].float().mean())
    out["placeholder"] = placeholder
    judge.gm = None
    return out


def sweep_variants(codec, judge, x, g, zt, flux, variants, eps_w=.3, eps_img=.1, polish=150, grain=True):
    """Search the crop layout once, then run the mixed-decode verdict for every variant on the same layout.
    variants: [dict(name, iters, max_iters, placeholder, repair, repair_rounds, boundary_max_delta, tol)].
    Returns (crop layout info, [(variant name, verdict dict)])."""
    st = setup(codec, judge, x, g, zt, flux, grain)
    search_layout(st, eps_w, eps_img, polish, 30, 20, "crop", None)
    out = []
    for v in variants:
        vv = dict(v)
        name = vv.pop("name")
        out.append((name, verdict(st, eps_w, eps_img, polish, **vv)))
    judge.gm = None
    return st, out
