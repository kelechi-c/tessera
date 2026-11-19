"""The leaf codec: what a leaf's 1 or 2 tokens hold, and how they decode.

The learned code (2-bit header value 3) is opt-in: pass `learned=[leafae.pt, ...]` to LeafCodec and a leaf whose fixed
code loses to the learned autoencoder's own decode by squared error uses the learned code instead (grain leaves keep the
fixed codes). A learned leaf's token is exactly the LeafAE token (K or 2K floats) and decodes through LeafAE's decoder;
see learned.py for the checkpoint format.

Leaf codes (the encoder picks per leaf the one with the lower squared error; flags go in the header):
- plain: PCA of the leaf's normalised cells (one basis per leaf size), components 1-128 in token 1, 129-256 in token 2.
  The whole-image leaf (level 0) is PCA of the image downsampled 4x (128 px), upsampled bicubically at decode.
- edge (levels 1-5): a three-region band model (16 angles, band start and width 0..24 px, one RGB colour per region;
  12 floats) plus PCA of the residual after it in the remaining floats.
- poly: a cubic surface per channel (30 floats), exact for flat colours and gradients and free of PCA ripple, which
  perceptual metrics (and eyes) see on flat areas; chosen unless another code's squared error is 2x lower.
- grain flag: a leaf that is mostly fine grain (8 px cells whose fine-scale std is below TAU) gives up its last 9 PCA
  floats for 9 grain statistics (robust std in 3 bands x 3 channels); the decoder adds band-limited noise with those
  amplitudes on the cells its own decode finds smooth. Grain is resynthesised, not transmitted.
- native cell (level 6): the 192 pixel values (1.5 tokens).

Every leaf decodes from its own floats only, so a whole layout decodes by pasting leaves.
"""
import math
import pathlib

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .grid import C, G, K, L, P, S, cells, image, size

NE = 12  # floats of the edge model
A = 16  # edge angles
W = 24  # max band width, px
NG = 9  # grain floats
TAU = .045  # grain cell: fine-scale std below this
WQ = 4  # whole-image leaf: PCA at S / WQ px


# ---------------------------------------------------------------- grain -------------------------------------------------
def bands(v):
    a, b, c = TF.gaussian_blur(v, 5, 1.), TF.gaussian_blur(v, 9, 2.), TF.gaussian_blur(v, 17, 4.)
    return v - a, a - b, b - c


def hf_std(x):
    """Fine-scale std per 8 px cell: [B, h/8, w/8]."""
    return F.avg_pool2d((x - TF.gaussian_blur(x, 9, 1.5)).square(), P).mean(1).sqrt()


def grain_mask(x):
    return hf_std(x)[0] < TAU


def smooth_px(y):
    return (hf_std(y) < TAU).float()[:, None].repeat_interleave(P, 2).repeat_interleave(P, 3)


def grain_measure(r, m):
    out = []
    for b in bands(r):
        v = torch.where(m > 0, b.abs(), torch.full_like(b, float("nan"))).flatten(2)
        out.append(torch.nan_to_num(v.nanmedian(-1).values / .6745, nan=0.))
    return torch.stack(out, 1)  # [m, 3 bands, 3 channels]


_noise = {}


def noise(l, t, idx, px, dev):
    """The decoder's noise for leaves idx of level l at t tokens: crops of a fixed field (generated on the CPU, so every
    device decodes the same grain)."""
    if (l, t) not in _noise:
        g = torch.Generator().manual_seed(1000 * l + t)
        _noise[(l, t)] = torch.randn(1, 1, S, S, generator=g)
    n = S // px
    f = _noise[(l, t)].to(dev)
    b = f.unfold(2, px, px).unfold(3, px, px).reshape(n * n, 1, px, px)
    return b[idx]


def grain_synth(sig, nz):
    add = 0
    for k, b in enumerate(bands(nz)):
        add = add + b / b.flatten(1).std(1).clamp_min(1e-6)[:, None, None, None] * sig[:, k, :, None, None]
    return add


# ---------------------------------------------------------------- edge model --------------------------------------------
_geo = {}


def geometry(px, dev):
    key = (px, str(dev))
    if key not in _geo:
        yy, xx = torch.meshgrid(torch.arange(px, device=dev) + .5 - px / 2, torch.arange(px, device=dev) + .5 - px / 2, indexing="ij")
        nb = int(math.ceil(px * math.sqrt(2))) + 2
        bins = []
        for a in range(A):
            th = math.pi * a / A
            t = xx * math.cos(th) + yy * math.sin(th)
            bins.append((t + nb / 2).floor().long().clamp(0, nb - 1).flatten())
        _geo[key] = (torch.stack(bins), nb)
    return _geo[key]


@torch.no_grad()
def edge_fit(x):
    """x [n, 3, px, px] -> params [n, 3] (angle, start bin, width), exhaustive; colours are region means."""
    n, _, px, _ = x.shape
    bins, nb = geometry(px, x.device)
    xf = x.reshape(n, 3, -1).float()
    wmax = min(W, nb - 1)
    best = torch.full((n,), -1.0, device=x.device)
    bp = torch.zeros(n, 3, dtype=torch.long, device=x.device)
    b0 = torch.arange(nb + 1, device=x.device)[:, None]
    w0 = torch.arange(wmax + 1, device=x.device)[None]
    b1, b2 = b0.expand(-1, wmax + 1), (b0 + w0).clamp(max=nb)
    for a in range(A):
        oh = torch.zeros(px * px, nb, device=x.device)
        oh[torch.arange(px * px, device=x.device), bins[a]] = 1
        Sb = xf @ oh
        Nb = oh.sum(0)
        cs = F.pad(Sb.cumsum(-1), (1, 0))
        cn = F.pad(Nb.cumsum(-1), (1, 0))
        tot, ntot = cs[..., -1:], cn[-1]
        s0, s1 = cs[..., b1], cs[..., b2] - cs[..., b1]
        n0, n1 = cn[b1], cn[b2] - cn[b1]
        s2, n2 = tot[..., None] - cs[..., b2], ntot - cn[b2]
        ex = sum((s.square() / m.clamp_min(1)).sum(1) for s, m in ((s0, n0), (s1, n1), (s2, n2)))
        v, i = ex.flatten(1).max(1)
        upd = v > best
        best = torch.where(upd, v, best)
        bp[upd] = torch.stack((torch.full_like(i, a), i // (wmax + 1), i % (wmax + 1)), 1)[upd]
    return bp


def regions(bp, px, dev):
    bins, nb = geometry(px, dev)
    t = bins[bp[:, 0]]
    lo, hi = bp[:, 1:2], (bp[:, 1] + bp[:, 2]).clamp(max=nb)[:, None]
    return (t >= lo).long() + (t >= hi).long()


def region_means(x, bp):
    n, _, px, _ = x.shape
    oh = F.one_hot(regions(bp, px, x.device), 3).float()
    return ((x.reshape(n, 3, -1).float() @ oh) / oh.sum(1, keepdim=True).clamp_min(1)).transpose(1, 2)  # [n, region, rgb]


def paint(bp, col, px):
    n = len(bp)
    oh = F.one_hot(regions(bp, px, col.device), 3).float()
    return (col.transpose(1, 2) @ oh.transpose(1, 2)).reshape(n, 3, px, px)


# ---------------------------------------------------------------- poly (smooth) code ------------------------------------
NPOLY = 10  # cubic surface per channel (30 floats): exact for flat colours and gradients, adds no ripple
POLY_BIAS = 2.0  # poly is chosen unless another code's squared error is this much lower
_poly = {}


def poly_basis(px, dev):
    key = (px, str(dev))
    if key not in _poly:
        v, u = torch.meshgrid(torch.linspace(-1, 1, px, device=dev, dtype=torch.float64), torch.linspace(-1, 1, px, device=dev, dtype=torch.float64), indexing="ij")
        u, v = u.flatten(), v.flatten()
        X = torch.stack((torch.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3), 1)
        _poly[key] = torch.linalg.qr(X)[0].float()
    return _poly[key]


def poly_fit(x):
    n, _, px, _ = x.shape
    c = x.reshape(n, 3, -1).float() @ poly_basis(px, x.device)
    return c.flatten(1), poly_render(c.flatten(1), px)


def poly_render(c, px):
    return (c.reshape(-1, 3, NPOLY) @ poly_basis(px, c.device).T).reshape(-1, 3, px, px)


PLAIN, EDGE, POLY, LEARNED = 0, 1, 2, 3


# ---------------------------------------------------------------- leaf codec --------------------------------------------
def nodes(x, l):
    px = size(l) * P
    b = x.unfold(2, px, px).unfold(3, px, px)
    n = b.shape[2]
    return b.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, px, px), n


def unnodes(b, n):
    px = b.shape[-1]
    return b.reshape(1, n, n, 3, px, px).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, n * px, n * px)


class LeafCodec:
    """Bases: `mean`, `std` (per cell feature), `plain` {level: (mu, V, sd)}, `edge` {level: (mu, V, sd)} (residual after
    the edge model), `whole` (mu, V, sd) for level 0."""

    def __init__(self, ck, dev="cpu", grain=True, edge=True, learned=None):
        self.dev, self.grain = dev, grain
        self.learned = {}  # {level: {tokens: LeafAE model}}
        if learned:
            if isinstance(learned, dict):
                self.learned = {int(k): {int(t): m for t, m in v.items()} for k, v in learned.items()}
            else:
                from .learned import load_all

                self.learned = load_all(list(learned) if not isinstance(learned, (str, pathlib.Path)) else [learned], dev)
        self.mean, self.std = ck["mean"].to(dev).float(), ck["std"].to(dev).float()
        sh = 1 if ck.get("root", 32) == 32 else 0
        self.plain = {l + sh: tuple(t.to(dev).float() for t in b) for l, b in ck["plain"].items()}
        self.plain[0] = tuple(t.to(dev).float() for t in ck["whole"])
        self.edge = {l + sh: tuple(t.to(dev).float() for t in b) for l, b in ck.get("edge", {}).items()} if edge else {}

    def flat(self, b):
        return ((cells(b) - self.mean) / self.std).flatten(1)

    def unflat(self, v, px):
        return image(v.reshape(-1, C, px // P, px // P) * self.std + self.mean)

    def _plain(self, l, c, px):
        mu, V, _ = self.plain[l]
        j = c.shape[1]
        if l == 0:
            y = self.unflat(c @ V[:, :j].T + mu, px // WQ)
            return F.interpolate(y, scale_factor=WQ, mode="bicubic", align_corners=False)
        return self.unflat(c @ V[:, :j].T + mu, px)

    def _resid(self, l, c, px):
        mu, V, _ = self.edge[l]
        j = c.shape[1]
        return image((c @ V[:, :j].T + mu).reshape(-1, C, px // P, px // P) * self.std)

    @torch.no_grad()
    def encode_level(self, x, l, gfrac=None, chunk=64):
        """All nodes of level l at both token counts. Returns {t: (payload [n*n, t*K], mode [n*n] long: PLAIN, EDGE or
        POLY)}, grain [n*n] bool, n. t is 1 or 2. Payloads are exactly what the tokens carry (poly leaves use 30 floats,
        plus 9 grain floats at the end of the token when grain is on; the rest is zero)."""
        b, n = nodes(x, l)
        px = b.shape[-1]
        pay = {1: [], 2: []}
        edg = {1: [], 2: []}
        gr = (gfrac > .5) if (self.grain and gfrac is not None) else torch.zeros(len(b), dtype=torch.bool, device=x.device)
        for i in range(0, len(b), chunk):
            bb = b[i:i + chunk].float()
            g = gr[i:i + chunk]
            mu, V, _ = self.plain[l]
            c = ((self.flat(F.avg_pool2d(bb, WQ)) if l == 0 else self.flat(bb)) - mu) @ V
            has_e = l in self.edge
            if has_e:
                bp = edge_fit(bb)
                col = region_means(bb, bp)
                e = paint(bp, col, px)
                mu2, V2, _ = self.edge[l]
                c2 = ((cells(bb - e) / self.std).flatten(1) - mu2) @ V2
                e12 = torch.cat((bp.float(), col.flatten(1)), 1)
            pc, pr = poly_fit(bb)
            ep = (pr - bb).square().mean((1, 2, 3))
            for t in (1, 2):
                k = t * K
                rp = self._plain(l, c[:, :k], px)
                pick = torch.zeros(len(bb), dtype=torch.bool, device=x.device)
                if has_e:
                    re = e + self._resid(l, c2[:, :k - NE], px)
                    pick = (re - bb).square().mean((1, 2, 3)) < (rp - bb).square().mean((1, 2, 3))
                p = c[:, :k].clone()
                err = torch.where(pick, (re - bb).square().mean((1, 2, 3)), (rp - bb).square().mean((1, 2, 3))) if has_e else (rp - bb).square().mean((1, 2, 3))
                if has_e:
                    p = torch.where(pick[:, None], torch.cat((e12, c2[:, :k - NE]), 1), p)
                mode = pick.long()
                pp = ep < POLY_BIAS * err
                mode = torch.where(pp, torch.full_like(mode, POLY), mode)
                p = torch.where(pp[:, None], F.pad(pc, (0, k - pc.shape[1])), p)
                if g.any():  # grain leaves: last NG floats = grain statistics of what the reduced decode misses
                    red = self._plain(l, c[:, :k - NG], px)
                    if has_e:
                        red = torch.where(pick[:, None, None, None], e + self._resid(l, c2[:, :k - NE - NG], px), red)
                    red = torch.where(pp[:, None, None, None], pr, red)
                    sig = grain_measure(bb - red, smooth_px(red)).flatten(1)
                    pg = torch.cat((p[:, :k - NG], sig), 1)
                    p = torch.where(g[:, None], pg, p)
                lrn = self.learned.get(l, {}).get(t)
                if lrn is not None and bool((~g).any()):
                    token = lrn.encode(bb, self.mean, self.std)  # float32: the token must match the decoder exactly
                    yl = lrn.decode(token, self.mean, self.std)[0].float().clamp(-1, 1)
                    el = (yl - bb).square().mean((1, 2, 3))
                    win = (el < torch.where(pp, ep, err)) & ~g
                    if bool(win.any()):
                        p = torch.where(win[:, None], token, p)
                        mode = torch.where(win, torch.full_like(mode, LEARNED), mode)
                pay[t].append(p)
                edg[t].append(mode)
        return {t: (torch.cat(pay[t]), torch.cat(edg[t])) for t in (1, 2)}, gr, n

    @torch.no_grad()
    def decode_leaves(self, l, t, p, mode, grain, idx):
        """Leaves of level l with t tokens from their payloads p [m, t*K], modes [m], grain flags [m]; idx = their node
        index (for the noise)."""
        px = size(l) * P
        k = t * K
        out = torch.zeros(len(p), 3, px, px, device=p.device)
        for e_ in (PLAIN, EDGE, POLY):
            for g_ in (False, True):
                sel = (mode == e_) & (grain == g_)
                if not sel.any():
                    continue
                q = p[sel]
                end = k - (NG if g_ else 0)
                if e_ == POLY:
                    y = poly_render(q[:, :3 * NPOLY], px)
                elif e_ == EDGE:
                    bp = q[:, :3].round().long()
                    y = paint(bp, q[:, 3:12].reshape(-1, 3, 3), px) + self._resid(l, q[:, NE:end], px)
                else:
                    y = self._plain(l, q[:, :end], px)
                if g_:
                    y = y + smooth_px(y) * grain_synth(q[:, end:k].reshape(-1, 3, 3), noise(l, t, idx[sel], px, p.device))
                out[sel] = y
        lrn = self.learned.get(l, {}).get(t)  # learned leaf: the token is the LeafAE token, decode it exactly
        sel = mode == LEARNED
        if bool(sel.any()):
            if lrn is None:
                raise ValueError(f"token has {int(sel.sum())} learned leaves at level {l} / {t} tokens but this decoder "
                                 f"was built without a LeafAE checkpoint; pass learned=[leafae.pt, ...] to decode them")
            out[sel] = lrn.decode(p[sel], self.mean, self.std)[0].float()
        return out

    @torch.no_grad()
    def prepare(self, x, gm=None):
        """x [1, 3, S, S] in [-1, 1] -> table: per coded level, both token counts, every node's payload and its decode
        (from the payload, so the table is exactly what a decoder will produce). gm: grain cells [G, G]."""
        if self.grain and gm is None:
            gm = grain_mask(x)
        tab = {}
        for l in range(L - 1):
            gf = F.avg_pool2d(gm[None, None].float(), size(l)).flatten() if self.grain else None
            enc, gr, n = self.encode_level(x, l, gf)
            idx = torch.arange(n * n, device=x.device)
            rec = {t: unnodes(self.decode_leaves(l, t, enc[t][0], enc[t][1], gr, idx), n) for t in (1, 2)}
            tab[l] = dict(rec=rec, pay={t: enc[t][0] for t in (1, 2)}, mode={t: enc[t][1] for t in (1, 2)}, grain=gr, n=n)
        tab[L - 1] = dict(rec={1: x, 2: x}, n=G)
        return tab


def paste(tab, lvl, two):
    """Decode a layout from a prepared table (no token round trip)."""
    from .grid import masks, up

    y = None
    for l, m in enumerate(masks(lvl)):
        if not m.any():
            continue
        px = size(l) * P
        r1, r2 = tab[l]["rec"][1], tab[l]["rec"][2]
        r = torch.where(up(two[0, ::size(l), ::size(l)] & m[0], px), r2, r1) if l < L - 1 else r1
        y = r.clone() if y is None else torch.where(up(m[0], px), r, y)
    return y
