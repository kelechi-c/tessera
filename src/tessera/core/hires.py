"""Retina-Q at 512 px: per-leaf-size PCA with an optional edge-aware leaf code, and the oracle on a 64 x 64 cell grid.

Grid: 8 x 8 RGB cells (192 floats) on a 64 x 64 grid; quadtree leaves of 64 / 32 / 16 / 8 / 4 / 2 cells (512 = the
whole image / 256 / 128 / 64 / 32 / 16 px) plus native 8 px cells (192 floats = 1.5 tokens). A token is 128 floats; a
coded leaf carries 1 or 2. The whole-image leaf is PCA of the 4x downsampled image (128 px), upsampled bicubically at
decode (plus grain): smooth content only, e.g. plain colours and skies at 2 tokens.

Leaf codes (no learned network; the encoder picks per leaf the one with lower squared error, a 1-bit header flag):
- plain: whitened PCA of the flattened leaf, components 1-128 in token 1 and 129-256 in token 2.
- edge: a three-region model along a straight band (angle from 16, two 1-px offsets: band start and width 0..24 px;
  width 0 = a single edge) with one RGB colour per region, fitted exactly by search over projection histograms; 12 floats
  (angle, start, width, 9 colour values). The rest of token 1 (116 floats) and token 2 hold PCA of the residual after it.
  This is a general piecewise-flat basis (edges, bars, strokes, borders), not a Mondrian detector.

Because no part of the codec looks across leaves, every node's reconstruction at every level and token count is fixed
by the image, so any layout decodes by pasting (the oracle never re-runs a network).

usage:
  python -m qtok.hires fit DATA OUT [--synth_n 1200]          (PCA bases at 512 px, plain and edge residual)
  python -m qtok.hires oracle DATA BASES OUT [--mode edge|plain] [--polish 200]
"""
import io
import json
import math
import pathlib
import random
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from . import tree as T
from .structure_guard import failing_cells as structure_fail

import torchvision.transforms.functional as TF

S = 512
P = 8
C = 192
G = 64
ROOT = 64
L = T.levels(G, ROOT)  # 7: leaf sides 64, 32, 16, 8, 4, 2, 1 cells (level 0 = whole image)
WQ = 4  # whole-image leaf: PCA at 512 / WQ px
K = 128
NE = 12  # floats of the edge model inside token 1
A = 16  # edge angles
W = 24  # max band width in px
NATIVE = G * G * 1.5 + 1
NG = 9  # grain floats per leaf: robust std of the missing signal in 3 bands x 3 channels (taken out of the PCA part)
TAU = .045  # a cell is "grain" when its fine-scale std (x - blur) is below this (about 6 grey levels)
E_CLEAN = .006  # judge v2 (R1.2): cells with less fine energy than this are clean: no noise, judged exactly
CREST = 4.5  # judge v2 (R1.2): grain cells only if peak/std of the fine band is below this (noise has a low crest)
E_F = .004  # judge v2 (R1.3): relative fine-energy floor
LAM_R = .3  # judge v2 (R1.3): weight of the relative fine-energy mismatch (replaces LAM for grain cells)
LAM = 2.0  # grain cells: weight of the fine-scale energy mismatch in the judge


def bands(v):
    """Three high-frequency bands (about 1-2, 2-4, 4-8 px), as in Retina v1's grainfill."""
    a, b, c = TF.gaussian_blur(v, 5, 1.), TF.gaussian_blur(v, 9, 2.), TF.gaussian_blur(v, 17, 4.)
    return v - a, a - b, b - c


def hf_std(x):
    """Fine-scale std per 8 px cell: [B, G, G]."""
    return F.avg_pool2d((x - TF.gaussian_blur(x, 9, 1.5)).square(), P).mean(1).sqrt()


def grain_mask(x, v2=False, tau=TAU, e_clean=E_CLEAN, crest=CREST):
    """Grain cells [G, G].  v1: fine-scale std below TAU.  v2 (R1.2): also above the clean floor e_clean, and with a
    peak-over-std of the fine band below crest, so strokes, specks and small text are never grain (no noise on them)."""
    h = hf_std(x)[0]
    if not v2:
        return h < tau
    hf = (x - TF.gaussian_blur(x, 9, 1.5))[0]
    peak = F.max_pool2d(hf.abs()[None], P)[0].amax(0)  # max |fine band| over channels, per 8 px cell
    return (h < tau) & (h > e_clean) & (peak / h.clamp_min(1e-6) < crest)


def smooth_px(y):
    """Decoder-side grain mask (free: computed from the grain-less decode itself): pixels of 8 px cells whose own
    fine-scale std is below TAU, so edges, lines and their ringing never get noise. y [m, 3, px, px] -> [m, 1, px, px]."""
    return (hf_std(y) < TAU).float()[:, None].repeat_interleave(P, 2).repeat_interleave(P, 3)


def grain_measure(r, m):
    """r [m, 3, px, px] = what the decode misses, m = smooth-pixel mask -> [m, 3 bands, 3 channels] robust std
    (median |.| / .6745) over the masked pixels only (0 where a leaf has none)."""
    out = []
    for b in bands(r):
        v = torch.where(m > 0, b.abs(), torch.full_like(b, float("nan"))).flatten(2)
        out.append(torch.nan_to_num(v.nanmedian(-1).values / .6745, nan=0.))
    return torch.stack(out, 1)


def grain_synth(sig, px, seed):
    """Band-limited noise with the measured amplitudes; one luminance field drives all channels."""
    g = torch.Generator(device=sig.device).manual_seed(seed)
    n = torch.randn(sig.shape[0], 1, px, px, device=sig.device, generator=g)
    add = 0
    for k, b in enumerate(bands(n)):
        add = add + b / b.flatten(1).std(1).clamp_min(1e-6)[:, None, None, None] * sig[:, k, :, None, None]
    return add


def grain_field(dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = torch.randn(1, 1, S, S, device=dev, generator=g)
    return torch.cat([b / b.std().clamp_min(1e-6) for b in bands(n)], 1)


def cells(x):
    return F.pixel_unshuffle(x, P)


def image(c):
    return F.pixel_shuffle(c, P)


def size(l):
    return T.size(l, ROOT)


def masks(lvl):
    return T.leaf_masks(lvl, ROOT)


def lowres_tokens(enc):
    """T3 lowres bill at codec level: the stored raster's floats -> tokens, plus the package's tree header token."""
    from . import lowres as LR

    return LR.tokens(enc) + 1


def count(lvl, two):
    """Tokens of 128 floats, including the package's tree and leaf header."""
    ms = masks(lvl)
    coded = sum(int(m.sum()) for m in ms[:-1])
    native = int(ms[-1].sum())
    n = coded + sum(int((m & two[:, ::size(l), ::size(l)]).sum()) for l, m in enumerate(ms[:-1]))
    splits = (coded + native - 1) // 3
    header = max(1, math.ceil((splits + 5 * coded) / (K * 32)))
    return n + 1.5 * native + header


# ---------------------------------------------------------------- edge model --------------------------------------------
_geo = {}


def geometry(px, dev):
    """Per angle: bin index of every pixel of a px x px leaf along the projection, and the bin count."""
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
    """x [n, 3, px, px] -> (params [n, 3] long: angle, start bin, width; colours [n, 3 regions, 3]; render [n, 3, px, px]).
    Exhaustive over angle x start x width; colours are the region means (closed form)."""
    n, _, px, _ = x.shape
    bins, nb = geometry(px, x.device)
    xf = x.reshape(n, 3, -1).float()
    wmax = min(W, nb - 1)
    best = torch.full((n,), -1.0, device=x.device)
    bp = torch.zeros(n, 3, dtype=torch.long, device=x.device)
    b0 = torch.arange(nb + 1, device=x.device)[:, None]
    w0 = torch.arange(wmax + 1, device=x.device)[None]
    b1, b2 = b0.expand(-1, wmax + 1), (b0 + w0).clamp(max=nb)  # region 0: bins < b1, 1: b1..b2-1, 2: >= b2
    for a in range(A):
        oh = torch.zeros(px * px, nb, device=x.device)
        oh[torch.arange(px * px, device=x.device), bins[a]] = 1
        Sb = xf @ oh  # [n, 3, nb]
        Nb = oh.sum(0)
        cs = F.pad(Sb.cumsum(-1), (1, 0))  # [n, 3, nb+1]
        cn = F.pad(Nb.cumsum(-1), (1, 0))
        tot, ntot = cs[..., -1:], cn[-1]
        s0, s1 = cs[..., b1], cs[..., b2] - cs[..., b1]  # [n, 3, nb+1, w+1]
        n0, n1 = cn[b1], cn[b2] - cn[b1]
        s2, n2 = tot[..., None] - cs[..., b2], ntot - cn[b2]
        ex = sum((s.square() / m.clamp_min(1)).sum(1) for s, m in ((s0, n0), (s1, n1), (s2, n2)))  # explained [n, nb+1, w+1]
        v, i = ex.flatten(1).max(1)
        up = v > best
        best = torch.where(up, v, best)
        bp[up] = torch.stack((torch.full_like(i, a), i // (wmax + 1), i % (wmax + 1)), 1)[up]
    return bp, *render(x, bp)


def regions(bp, px, dev):
    bins, nb = geometry(px, dev)
    t = bins[bp[:, 0]]  # [n, px*px]
    lo, hi = bp[:, 1:2], (bp[:, 1] + bp[:, 2]).clamp(max=nb)[:, None]
    return (t >= lo).long() + (t >= hi).long()  # 0, 1, 2


def render(x, bp):
    """Region means of x under params bp, and the piecewise-flat render."""
    n, _, px, _ = x.shape
    r = regions(bp, px, x.device)
    oh = F.one_hot(r, 3).float()  # [n, px*px, 3]
    xf = x.reshape(n, 3, -1).float()
    col = (xf @ oh) / oh.sum(1, keepdim=True).clamp_min(1)  # [n, 3ch, 3reg]
    out = (col @ oh.transpose(1, 2)).reshape(n, 3, px, px)
    return col.transpose(1, 2), out


# ---------------------------------------------------------------- poly (smooth) leaf code -------------------------------
POLY_BIAS = float(__import__("os").environ.get("POLY_BIAS", 2.0))  # poly wins unless the others' squared error is this much lower (it adds no ripple)
NPOLY = 10  # cubic surface per channel: 1, u, v, u2, uv, v2, u3, u2v, uv2, v3 (30 floats)
_poly = {}


def poly_basis(px, dev):
    """Orthonormal basis [px*px, 10] of cubic surfaces over a px x px leaf: least squares = projection."""
    key = (px, str(dev))
    if key not in _poly:
        v, u = torch.meshgrid(torch.linspace(-1, 1, px, device=dev, dtype=torch.float64), torch.linspace(-1, 1, px, device=dev, dtype=torch.float64), indexing="ij")
        u, v = u.flatten(), v.flatten()
        X = torch.stack((torch.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3), 1)
        _poly[key] = torch.linalg.qr(X)[0].float()
    return _poly[key]


def poly_fit(x):
    """x [n, 3, px, px] -> (coefs [n, 30], render [n, 3, px, px]). Flat colours and gradients exactly, no ripple."""
    n, _, px, _ = x.shape
    Q = poly_basis(px, x.device)
    c = x.reshape(n, 3, -1).float() @ Q  # [n, 3, 10]
    return c.flatten(1), (c @ Q.T).reshape(n, 3, px, px)


def poly_render(c, px):
    Q = poly_basis(px, c.device)
    return (c.reshape(-1, 3, NPOLY) @ Q.T).reshape(-1, 3, px, px)


# ---------------------------------------------------------------- PCA ---------------------------------------------------
def nodes(x, l):
    """x [B, 3, S, S] -> [B * n * n, 3, px, px] leaf blocks at level l (row-major)."""
    px = size(l) * P
    B = x.shape[0]
    b = x.unfold(2, px, px).unfold(3, px, px)  # [B, 3, n, n, px, px]
    n = b.shape[2]
    return b.permute(0, 2, 3, 1, 4, 5).reshape(B * n * n, 3, px, px), n


def unnodes(b, B, n):
    px = b.shape[-1]
    return b.reshape(B, n, n, 3, px, px).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, n * px, n * px)


def flat(b, mean, std):
    """Leaf blocks [m, 3, px, px] -> normalized cell vectors [m, D] (same cell order inside every leaf)."""
    return ((cells(b) - mean) / std).flatten(1)


def unflat(v, px, mean, std):
    return image(v.reshape(-1, C, px // P, px // P) * std + mean)


@torch.no_grad()
def pca(v, k, name):
    mu = v.mean(0)
    v = v - mu
    _, Sv, V = torch.svd_lowrank(v, q=min(k + 64, min(v.shape)), niter=4)
    V, Sv = V[:, :k], Sv[:k]
    if V.shape[1] < k:
        V, Sv = F.pad(V, (0, k - V.shape[1])), F.pad(Sv, (0, k - len(Sv)))
    sd = Sv / math.sqrt(len(v) - 1)
    tot = v.square().sum(1).mean()
    print(f"  {name}: D={v.shape[1]} n={len(v)} var kept {float(sd[:k // 2].square().sum() / tot):.3f} / {float(sd.square().sum() / tot):.3f}", flush=True)
    return mu, V.contiguous(), sd.clamp_min(1e-4)


class Codec:
    def __init__(self, ck, dev, mode="edge", leafae=None, lp=None, grain=False, poly=True, v2=False):
        self.dev, self.mode, self.grain, self.gm, self.poly = dev, mode, grain, None, poly
        self.lowres = None  # T3: whole-image lowres leaf (exact pixel art), set by prepare()
        self.v2 = v2  # R1.4: grain amplitude measured only over pixels in v2-grain cells
        self.learned, self.lp = {}, lp
        if leafae is not None:
            model, level, *tokens = leafae
            self.learned.setdefault(level, {})[tokens[0] if tokens else 1] = model
        self.mean, self.std = ck["mean"].to(dev), ck["std"].to(dev)
        sh = 1 if ck.get("root", 32) == 32 else 0  # bases fitted before the whole-image leaf: their level 0 = 256 px
        self.plain = {l + sh: tuple(t.to(dev) for t in b) for l, b in ck["plain"].items()}
        self.edge = {l + sh: tuple(t.to(dev) for t in b) for l, b in ck.get("edge", {}).items()}
        self.whole = tuple(t.to(dev) for t in ck["whole"]) if "whole" in ck else None

    @torch.no_grad()
    def recon_level(self, x, l, chunk=64):
        """All nodes of level l, both token counts: ({1: img, 2: img}, edge-used maps {1: [n, n], 2: [n, n]})."""
        b, n = nodes(x, l)
        px = b.shape[-1]
        out = {1: [], 2: []}
        used = {1: [], 2: []}
        lused = []
        for i in range(0, len(b), chunk):
            bb = b[i:i + chunk].float()
            if l == 0:  # whole-image leaf: PCA of the downsampled image, upsampled at decode
                mu, V, sd = self.whole
                c = (flat(F.avg_pool2d(bb, WQ), self.mean, self.std) - mu) @ V
                prec = lambda k: F.interpolate(unflat(c[:, :k] @ V[:, :k].T + mu, px // WQ, self.mean, self.std), scale_factor=WQ, mode="bicubic", align_corners=False)
            else:
                mu, V, sd = self.plain[l]
                c = (flat(bb, self.mean, self.std) - mu) @ V
                prec = lambda k: unflat(c[:, :k] @ V[:, :k].T + mu, px, self.mean, self.std)
            recs = {t: prec(t * K) for t in (1, 2)}
            red = {t: prec(t * K - NG) for t in (1, 2)} if self.grain else None  # grain leaves give up NG floats
            errs = {t: (recs[t] - bb).square().mean((1, 2, 3)) for t in (1, 2)}
            if self.mode == "edge" and l in self.edge:
                bp, _, e = edge_fit(bb)
                mu2, V2, _ = self.edge[l]
                c2 = ((cells(bb - e) / self.std).flatten(1) - mu2) @ V2
                erec = lambda k: e + image((c2[:, :k] @ V2[:, :k].T + mu2).reshape(-1, C, px // P, px // P) * self.std)
                for t in (1, 2):
                    k = t * K - NE
                    r2 = erec(k)
                    e2 = (r2 - bb).square().mean((1, 2, 3))
                    pick = e2 < errs[t]
                    recs[t] = torch.where(pick[:, None, None, None], r2, recs[t])
                    if self.grain:
                        red[t] = torch.where(pick[:, None, None, None], erec(k - NG), red[t])
                    used[t].append(pick)
            if self.poly:  # smooth leaves: a cubic surface per channel, exact for flat colours and gradients
                _, pr = poly_fit(bb)
                ep = (pr - bb).square().mean((1, 2, 3))
                for t in (1, 2):
                    pk = ep < POLY_BIAS * (recs[t] - bb).square().mean((1, 2, 3))
                    recs[t] = torch.where(pk[:, None, None, None], pr, recs[t])
                    if self.grain:
                        red[t] = torch.where(pk[:, None, None, None], pr, red[t])
            if self.grain:  # leaves that are mostly grain cells: decode = reduced code + synthesised grain
                gfrac = self._gfrac[l][i:i + chunk]
                on = gfrac > .5
                for t in (1, 2):
                    m = smooth_px(red[t])
                    if getattr(self, "_gmn", None) is not None:
                        m = m * self._gmn[l][i:i + chunk]  # R1.4: amplitude from v2-grain cells only
                    sig = grain_measure(bb - red[t], m)
                    field = self._grain_nodes[l][i:i + chunk]
                    noise = (field[:, :, None] * sig[:, :, :, None, None]).sum(1)
                    g = red[t] + m * noise
                    recs[t] = torch.where(on[:, None, None, None], g, recs[t])
            for t in (1, 2):
                out[t].append(recs[t])
                if not (self.mode == "edge" and l in self.edge):
                    used[t].append(torch.zeros(len(bb), dtype=torch.bool, device=self.dev))
        lu = torch.cat(lused).reshape(n, n) if lused else None
        return {t: unnodes(torch.cat(out[t]), 1, n) for t in (1, 2)}, {t: torch.cat(used[t]).reshape(n, n) for t in (1, 2)}, lu

    def _grain_node_masks(self, gm, l):
        """gm [G, G] -> per-node pixel masks [n*n, 1, px, px] (row-major node order, as nodes())."""
        px = size(l) * P
        m = gm.float()[None, None].repeat_interleave(P, 2).repeat_interleave(P, 3)  # [1, 1, S, S]
        b = m.unfold(2, px, px).unfold(3, px, px)  # [1, 1, n, n, px, px]
        return b.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, px, px)

    @torch.no_grad()
    def prepare_lowres(self, x):
        """T3 lowres candidate for the whole image: None unless the image is an exact s-lattice (pixel art).

        The raster is stored at native resolution; `run_oracle` replaces the fixed-code layout with it when the
        lowres bill (`lowres_tokens`, raster floats -> tokens + the 1-token tree header) is cheaper.  A VAE decode
        of a 32 x 32 sprite shown at 512 px costs about 1,000 tokens, the lowres raster 25 + 1.
        x [1, 3, S, S] in [-1, 1]."""
        from . import lowres as LR

        self.lowres = LR.encode(LR.to_u8(x))
        return self.lowres

    @torch.no_grad()
    def prepare(self, x, gm=None, judge=None, fx=None, dg=None):
        """x [1, 3, S, S] -> table of every level's reconstructions, for decode by pasting. gm [G, G] grain cells."""
        self.lowres = self.prepare_lowres(x)
        if self.grain:
            self.gm = grain_mask(x, v2=self.v2) if gm is None else gm
            self._gfrac = {l: F.avg_pool2d(self.gm[None, None].float(), size(l)).flatten() for l in range(L - 1)}
            self._gmn = {l: self._grain_node_masks(self.gm, l) for l in range(L - 1)} if self.v2 else None
            field = grain_field(x.device)
            self._grain_nodes = {l: nodes(field, l)[0] for l in range(L - 1)}
        tab = {l: self.recon_level(x, l) for l in range(L - 1)}
        if self.learned:
            if judge is None or fx is None or dg is None:
                raise ValueError("learned codes require the judge, original features, and reference scores")
            for l, models in self.learned.items():
                rec, used, _ = tab[l]
                s = size(l)
                px = s * P
                leaf, n = nodes(x, l)
                err = lambda y: F.avg_pool2d((judge.cellmap(y, x, fx) - dg)[:, None], s)[0, 0]
                up = lambda a: a.repeat_interleave(px, 0).repeat_interleave(px, 1)[None, None]
                scores = {t: err(rec[t]) for t in (1, 2)}
                selected = {t: torch.zeros((n, n), dtype=torch.int8, device=x.device) for t in (1, 2)}
                for t, model in models.items():
                    outputs, grains = [], []
                    for offset in range(0, len(leaf), 64):
                        chunk = leaf[offset:offset + 64]
                        with torch.autocast("cuda", torch.bfloat16, enabled=x.is_cuda):
                            token = model.encode(chunk, self.mean, self.std)
                            y = model.decode(token, self.mean, self.std)[0].float().clamp(-1, 1)
                            reduced = token.clone()
                            reduced[:, -NG:] = 0
                            yg = model.decode(reduced, self.mean, self.std)[0].float().clamp(-1, 1)
                        mask = smooth_px(yg)
                        if getattr(self, "_gmn", None) is not None:
                            mask = mask * self._gmn[l][offset:offset + len(chunk)]  # R1.4
                        sig = grain_measure(chunk - yg, mask)
                        if self.grain:
                            field = self._grain_nodes[l][offset:offset + len(chunk)]
                            yg = yg + mask * (field[:, :, None] * sig[:, :, :, None, None]).sum(1)
                        outputs.append(y)
                        grains.append(yg)
                    for code, candidate in ((1, unnodes(torch.cat(outputs), 1, n)), (2, unnodes(torch.cat(grains), 1, n))):
                        e = err(candidate)
                        pick = e < scores[t]
                        if code == 2:
                            pick &= self._gfrac[l].reshape(n, n) > .5 if self.grain else torch.zeros_like(pick)
                        rec[t] = torch.where(up(pick), candidate, rec[t])
                        scores[t] = torch.where(pick, e, scores[t])
                        selected[t] = torch.where(pick, code, selected[t])
                better = scores[1] < scores[2]
                rec[2] = torch.where(up(better), rec[1], rec[2])
                selected[2] = torch.where(better, selected[1], selected[2])
                used[2] = torch.where(better, used[1], used[2])
                tab[l] = rec, used, selected
        tab[L - 1] = ({1: x, 2: x}, None, None)
        return tab

    @staticmethod
    def decode(tab, lvl, two):
        y = None
        for l, m in enumerate(masks(lvl)):
            px = size(l) * P
            if not m.any():
                continue
            cellmask = m[0].repeat_interleave(px, 0).repeat_interleave(px, 1)
            r1, r2 = tab[l][0][1], tab[l][0][2]
            if l < L - 1:
                tw = (two[0, ::size(l), ::size(l)] & m[0]).repeat_interleave(px, 0).repeat_interleave(px, 1)
                r = torch.where(tw, r2, r1)
            else:
                r = r1
            y = r.clone() if y is None else y
            y = torch.where(cellmask, r, y)
        return y

    @staticmethod
    def decode_lowres(enc, side=S, dev="cpu"):
        """Render a whole-image lowres leaf at any target side (512, 1024, 2048): nearest, hard block edges."""
        from . import lowres as LR

        return LR.to_tensor(LR.decode(enc, side)).to(dev)

    @staticmethod
    def learned_share(tab, lvl, two):
        """Fraction of 1-token leaves at the learned codec's level that use it (and how many such leaves there are)."""
        for l, m in enumerate(masks(lvl)[:-1]):
            if tab[l][2] is not None:
                one = m[0] & ~two[0, ::size(l), ::size(l)]
                return int((one & (tab[l][2][1] > 0)).sum()), int(one.sum())
        return 0, 0

    @staticmethod
    def edge_share(tab, lvl, two):
        n = e = 0
        for l, m in enumerate(masks(lvl)[:-1]):
            if tab[l][1] is None:
                continue
            tw = two[0, ::size(l), ::size(l)]
            u = torch.where(tw, tab[l][1][2], tab[l][1][1])
            n += int(m[0].sum())
            e += int((m[0] & u).sum())
        return e / max(n, 1)


# ---------------------------------------------------------------- data --------------------------------------------------
def decode512(jpgs, dev):
    return torch.stack([torch.from_numpy(np.asarray(Image.open(io.BytesIO(j)).convert("RGB")).copy()).permute(2, 0, 1) for j in jpgs]).to(dev)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("data")
@click.argument("out")
@click.option("--n_img", type=int, default=1600)
@click.option("--synth_n", type=int, default=480)
@click.option("--max_floats", type=float, default=5e8)
def fit(data, out, n_img, synth_n, max_floats):
    """PCA bases per level at 512 px: plain, and of the residual after the edge model."""
    from concurrent.futures import ProcessPoolExecutor

    from .synth import train_pool
    from .train import load

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, _ = load(data)
    idx = sorted(random.Random(0).sample(range(len(tr["jpg"])), min(n_img, len(tr["jpg"]))))
    X = decode512([tr["jpg"][i] for i in idx], dev)
    del tr
    if synth_n:
        with ProcessPoolExecutor(8) as ex:
            ims = sum(ex.map(train_pool, [synth_n // 8] * 8, range(8), [S] * 8), [])
        X = torch.cat([X, torch.from_numpy(np.stack([np.asarray(im) for im in ims])).permute(0, 3, 1, 2).to(dev)])
    X = X[torch.randperm(len(X), device=dev)]
    print("images", tuple(X.shape), f"{time.time() - t0:.0f}s", flush=True)
    f = lambda u8: u8.float() / 127.5 - 1
    cs = torch.cat([cells(f(X[i:i + 64])) for i in range(0, min(512, len(X)), 64)])
    mean, std = cs.mean((0, 2, 3), keepdim=True), cs.std((0, 2, 3), keepdim=True).clamp_min(1e-3)
    del cs
    plain, edge = {}, {}
    for l in range(1, L - 1):  # level 0 (whole image) is fitted by fit-whole
        px = size(l) * P
        D = C * size(l) ** 2
        per = (S // px) ** 2
        nimg = min(len(X), max(1, int(max_floats // D) // per))
        vs, rs = [], []
        for i in range(0, nimg, 16):
            b, _ = nodes(f(X[i:min(i + 16, nimg)]), l)
            vs.append(flat(b, mean, std).half())
            _, _, e = edge_fit(b)
            rs.append((cells(b - e) / std).flatten(1).half())
        v, r = torch.cat(vs).float(), torch.cat(rs).float()
        print(f"level {l} ({px} px) {time.time() - t0:.0f}s", flush=True)
        plain[l] = tuple(t.cpu() for t in pca(v, 2 * K, "plain"))
        edge[l] = tuple(t.cpu() for t in pca(r, 2 * K - NE, "edge residual"))
        del v, r, vs, rs
        torch.cuda.empty_cache() if dev == "cuda" else None
    torch.save(dict(plain=plain, edge=edge, mean=mean.cpu(), std=std.cpu(), side=S, n_img=len(X), root=ROOT), out / "bases.pt")
    print("saved", out / "bases.pt", f"{time.time() - t0:.0f}s", flush=True)


@cli.command("fit-whole")
@click.argument("data")
@click.argument("bases")
@click.argument("out")
@click.option("--n_img", type=int, default=2400)
@click.option("--synth_n", type=int, default=600)
def fit_whole(data, bases, out, n_img, synth_n):
    """Add the whole-image leaf basis (PCA of the 4x downsampled image) to existing bases; writes root-64 bases."""
    from concurrent.futures import ProcessPoolExecutor

    from .synth import train_pool
    from .train import load

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(bases, map_location="cpu", weights_only=False)
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    tr, _ = load(data)
    idx = sorted(random.Random(2).sample(range(len(tr["jpg"])), min(n_img, len(tr["jpg"]))))
    jp = [tr["jpg"][i] for i in idx]
    del tr
    with ProcessPoolExecutor(8) as ex:
        ims = sum(ex.map(train_pool, [synth_n // 8] * 8, range(40, 48), [S] * 8), [])
    vs = []
    for i in range(0, len(jp), 64):
        x = decode512(jp[i:i + 64], dev).float() / 127.5 - 1
        vs.append(flat(F.avg_pool2d(x, WQ), mean, std))
    xs = torch.from_numpy(np.stack([np.asarray(im) for im in ims])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
    vs.append(flat(F.avg_pool2d(xs, WQ), mean, std))
    v = torch.cat(vs)
    ck["whole"] = tuple(t.cpu() for t in pca(v, 2 * K, "whole image (4x down)"))
    if ck.get("root", 32) == 32:  # older bases: their level 0 was the 256 px leaf
        ck["plain"] = {l + 1: b for l, b in ck["plain"].items()}
        ck["edge"] = {l + 1: b for l, b in ck.get("edge", {}).items()}
    ck["root"] = ROOT
    torch.save(ck, out)
    print("saved", out, flush=True)


# ---------------------------------------------------------------- oracle ------------------------------------------------
WIN = (8, 16, 32)  # cells: 64, 128, 256 px windows, stride half (the same pixel windows as the 256 px oracle at 512)


class Judge:
    reference_relative = False  # judge v4 subclass sets True and requires the actual grid-VAE reference via set_floor(x, g)

    def __init__(self, dev, v2=False, v3=False, floor=None):
        from .metrics import Judge as J

        self.j = J(dev)
        self.v2 = v2  # R1: relative fine-energy term, so the judge does not compare absolute high-frequency energy
        self.v3 = v3  # R5/T1: sharpness floor; a decode must keep >= FLOOR of the original's fine-detail energy
        self.floor = .6 if floor is None else float(floor)  # T1: .6 adopted; tighten to .7 if the eye test fails

        self.gm = None  # grain cells [G, G]: judged on blurred images + fine-scale energy, not exact pixels
        self.s_orig = None  # [G // 2, G // 2]: fine std per 16 px block of the original (floor reference)
        # U1 window rule: every 64 px window (stride 32) with >= 4 detail blocks must average >= .9 fine-energy ratio
        self.win = False
        self.win_floor = .9
        self.win_min = 4
        self.x01 = None

    def set_grain(self, x, gm):
        self.gm, self.xb, self.e0 = gm, TF.gaussian_blur(x, 9, 1.5), hf_std(x)

    def set_window(self, on=True, floor=.9, min_blocks=4):
        """U1: switch on the 64 px / stride 32 window rule on top of the floor."""
        self.win = bool(on)
        self.win_floor = float(floor)
        self.win_min = int(min_blocks)
        return self

    def set_floor(self, x):
        """Precompute the original's 16 px block fine energies (judge v3).  x [1, 3, S, S] in [-1, 1]."""
        from . import sharp

        self.s_orig = sharp.block_std_m11(x, uint8=True)[0]
        self.x01 = ((x.clamp(-1, 1) + 1) / 2).detach()
        return self

    @torch.no_grad()
    def floor_bad(self, xh):
        """[G, G] bool: cells of 16 px blocks whose decode keeps less than FLOOR of the original's fine energy."""
        from . import sharp

        if not self.v3:
            return None
        sd = sharp.block_std_m11(xh, uint8=True)[0]
        # same formula (with the +1e-4 epsilon) as the reference script docs/reviews/fable-results/t0/blur_blocks.py,
        # so the floor and the reported blurred-block count agree exactly
        bad = (self.s_orig > sharp.ACT) & (sd / (self.s_orig + 1e-4) < self.floor)
        return bad.repeat_interleave(2, 0).repeat_interleave(2, 1)  # 16 px blocks -> 8 px judge cells

    @torch.no_grad()
    def window_bad(self, xh):
        """[G, G] bool: 8 px cells below the window floor inside 64 px windows (stride 32) that break the U1 rule."""
        from . import sharp

        if not getattr(self, "win", False) or self.x01 is None:
            return None
        x01 = ((xh.clamp(-1, 1) + 1) / 2).detach()
        st = sharp.window_stats(x01, self.x01, act=sharp.ACT, floor=self.win_floor, min_blocks=self.win_min)[0]
        return st["culprit_blocks"].repeat_interleave(2, 0).repeat_interleave(2, 1)

    @torch.no_grad()
    def floor_stats(self, xh):
        """Blur audit of one decode: blurred/detail block counts, median fine-energy ratio, 64 px window-rule fails."""
        from . import sharp

        so = self.s_orig
        sd = sharp.block_std_m11(xh, uint8=True)[0]
        act = so > sharp.ACT
        ratio = sd / (so + 1e-4)
        st = sharp.window_stats(((xh.clamp(-1, 1) + 1) / 2).detach(), self.x01, floor=self.win_floor,
                                min_blocks=self.win_min)[0] if self.x01 is not None else None
        return dict(blurred=int((act & (ratio < self.floor)).sum()), detail=int(act.sum()),
                    median=round(float(ratio[act].median()), 4) if bool(act.any()) else 1.,
                    window_fail=int(st["fail_windows"]) if st else 0,
                    window_total=int(st["window_total"]) if st else 0,
                    blocks_below=int(st["blocks_below"]) if st else 0)

    @torch.no_grad()
    def grain_ratio(self, xh, gm):
        """U1: fine-energy ratio of the decode in the original's grain cells (16 px blocks), median and mean."""
        from . import sharp

        if gm is None or self.s_orig is None:
            return dict(grain_cells=0, grain_med=None, grain_mean=None)
        sd = sharp.block_std_m11(xh, uint8=True)[0]
        ratio = sd / (self.s_orig + 1e-4)
        g = gm[0] if gm.ndim == 3 else gm
        gb = F.avg_pool2d(g.float()[None, None], 2)[0, 0] > .5
        n = int(gb.sum())
        if n == 0:
            return dict(grain_cells=0, grain_med=None, grain_mean=None)
        v = ratio[gb]
        return dict(grain_cells=n, grain_med=round(float(v.median()), 4), grain_mean=round(float(v.mean()), 4))

    @torch.no_grad()
    def cellmap(self, xh, x, fx):
        ac = lambda: torch.autocast("cuda", torch.bfloat16, enabled=xh.is_cuda)  # judge networks always in bf16
        with ac():
            lp = F.avg_pool2d(self.j.lp(xh, x).float(), P)[:, 0]  # [1, 64, 64]
        if self.gm is not None:
            with ac():
                lpb = F.avg_pool2d(self.j.lp(TF.gaussian_blur(xh, 9, 1.5), self.xb).float(), P)[:, 0]
            if self.v2:  # R1.3 (bug 19): relative fine energy; noise of .04 on a cell at .005 costs .48 and fails
                rel = LAM_R * ((hf_std(xh) + E_F) / (self.e0 + E_F)).log().abs()
                lp = torch.where(self.gm, lpb + rel, lp)
            else:
                lp = torch.where(self.gm, lpb + LAM * (hf_std(xh) - self.e0).abs(), lp)
        with ac():
            fh = self.j.feats(xh).float()
        dn = 1 - (fh * fx).sum(1)  # [1, 32, 32]
        return lp + F.interpolate(dn[:, None], scale_factor=2, mode="nearest")[:, 0]


def windows(d):
    return [F.avg_pool2d(d[:, None], k, stride=k // 2)[:, 0] for k in WIN]


def culprits(cex, m, eps_w, frac=.25):
    hot = m & (cex > eps_w)
    if hot.any():
        return hot
    v = cex[m]
    return m & (cex >= torch.quantile(v.float(), 1 - frac)) & (cex > 0)


def fail(d, dg, eps_w, eps_img, free=None, eps_loc=None):
    """Failing cells.  eps_loc (R1.1, judge v2 only): also cap the masked mean excess over every 2 x 2 cells (16 px,
    stride 1) at eps_loc; the cells of a hot block that exceed eps_w become bad.  Returns (bad, worst, img), plus the
    worst 2 x 2 block when eps_loc is not None; eps_loc=None is bit-identical to the shipped v1 path."""
    cex = (d - dg)[0]
    fr = torch.ones_like(cex) if free is None else free.float()
    bad = torch.zeros(G, G, dtype=torch.bool, device=d.device)
    worst = -1e9
    for k in WIN:
        num = F.avg_pool2d((cex * fr)[None, None], k, stride=k // 2)[0, 0]
        den = F.avg_pool2d(fr[None, None], k, stride=k // 2)[0, 0]
        ex = torch.where(den >= .25, num / den.clamp_min(1e-6), torch.full_like(num, -1e9))
        worst = max(worst, float(ex.max()))
        s = k // 2
        for i, j in (ex > eps_w).nonzero().tolist():
            m = torch.zeros_like(bad)
            m[i * s:i * s + k, j * s:j * s + k] = True
            bad |= culprits(cex, m & (fr > 0), eps_w)
    img = float((cex * fr).sum() / fr.sum().clamp_min(1))
    if img > eps_img:
        bad |= culprits(cex, fr > 0, eps_w, .1)
    if eps_loc is None:
        return bad, worst, img
    num = F.avg_pool2d((cex * fr)[None, None], 2, stride=1)
    den = F.avg_pool2d(fr[None, None], 2, stride=1)
    ratio = num / den.clamp_min(1e-6)
    hot = (ratio > eps_loc) & (den >= .5)
    cover = F.max_pool2d(F.pad(hot.float(), (1, 1, 1, 1)), 2, stride=1)[0, 0] > 0  # [G, G]: cells of a hot block
    bad |= cover & (cex > eps_w)
    worst_block = float(torch.where(den >= .5, ratio, torch.full_like(ratio, -1e9)).max())
    return bad, worst, img, worst_block


def block_any(mask, s):
    return F.max_pool2d(mask[None, None].float(), s)[0, 0] > 0


def refine(lvl, two, bad):
    nl, nt = lvl.clone(), two.clone()
    for l, m in enumerate(masks(lvl)[:-1]):
        s = size(l)
        hit = (m[0] & block_any(bad, s)).repeat_interleave(s, 0).repeat_interleave(s, 1)[None]
        up = hit & ~two
        sp = hit & ~up
        nt = (nt | up) & ~sp
        nl = torch.where(sp, lvl + 1, nl)
    return nl, nt


def candidates(lvl, two, learned_levels=()):
    out = []
    ms = masks(lvl)
    for l in range(L - 1):
        s = size(l)
        for i, j in (ms[l][0] & two[0, ::s, ::s]).nonzero().tolist():
            out.append(("drop", l, i, j))
        child = ms[l + 1][0]
        if l + 1 < L - 1:
            child = child & ~two[0, ::s // 2, ::s // 2]
        for i, j in (F.avg_pool2d(child[None, None].float(), 2)[0, 0] == 1).nonzero().tolist():
            if l in learned_levels:
                out.append(("merge1", l, i, j))
            out.append(("merge", l, i, j))
    return out


def coarsen(lvl, two, cands):
    nl, nt = lvl.clone(), two.clone()
    for kind, l, i, j in sorted(cands, key=lambda c: c[0] == "merge1"):
        s = size(l)
        sl = (0, slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s))
        if kind == "drop":
            nt[sl] = False
        else:
            nl[sl] = l
            nt[sl] = kind != "merge1"
    return nl, nt


def learned_cells(tab, lvl, two):
    """[G, G] cells covered by a learned/pretrained leaf code (LeafAE, DC-AE): the only cells where the structure guard
    still applies in judge v2 (R1.5).  All False when the codec has no learned levels."""
    out = torch.zeros(G, G, dtype=torch.bool, device=lvl.device)
    for l, m in enumerate(masks(lvl)[:-1]):
        if tab[l][2] is None:
            continue
        s = size(l)
        tw = two[0, ::s, ::s] & m[0]
        sel = torch.where(tw, tab[l][2][2], tab[l][2][1])
        lm = (sel > 0).repeat_interleave(s, 0).repeat_interleave(s, 1)  # [G, G] cells covered by a learned node
        mm = m[0].repeat_interleave(s, 0).repeat_interleave(s, 1)
        out |= mm & lm
    return out


def fixed_cells(fixed, dev):
    """Texture blocks [(y, x, side) in px] -> (cell mask [G, G], leaf level per cell [G, G]). Each block is one quadtree
    leaf of its own level (512 px = 0, 256 = 1, 128 = 2), coded by a texture token instead of the exact codes."""
    fm = torch.zeros(G, G, dtype=torch.bool, device=dev)
    fl = torch.zeros(G, G, dtype=torch.long, device=dev)
    for y, x, side in fixed:
        l = int(round(math.log2(S / side)))
        assert size(l) * P == side and y % side == 0 and x % side == 0, (y, x, side)
        fm[y // P:(y + side) // P, x // P:(x + side) // P] = True
        fl[y // P:(y + side) // P, x // P:(x + side) // P] = l
    return fm, fl


def adopt_lowres(out, codec, x):
    """T3: replace a fixed-code layout with the whole-image lowres leaf when it is exact and cheaper.

    The lowres raster is an exact copy of the original at native resolution, so it passes the judge trivially
    (fine-energy ratio 1, and its judge excess over the VAE reference is not worse than the reference's own)."""
    enc = getattr(codec, "lowres", None)
    if enc is not None:
        lt = lowres_tokens(enc)
        if lt < out["tokens"]:
            out.update(lowres=enc, lowres_tokens=lt, lowres_replaced=True,
                       xh_fixed=out["xh"], xh=Codec.decode_lowres(enc, S, x.device), tokens=lt, passed=True)
    return out


def run_oracle(codec, judge, x, g, eps_w, eps_img, polish=200, rounds=30, down_rounds=20, grain=False, fixed=(),
               eps_loc=None, v2=False):
    """fixed: texture blocks [(y, x, side) px]. Each stays one leaf, never split or merged; the aligned judge ignores its
    cells, since a texture token is judged on statistics instead.
    tokens then include one texture token per block on top of the leaf's own token."""
    fm, fl = fixed_cells(fixed, x.device) if fixed else (None, None)
    gm = grain_mask(x, v2=v2) if grain else None
    judge.gm = None
    if grain:
        judge.set_grain(x, gm)
    if getattr(judge, "v3", False):
        judge.set_floor(x, g) if getattr(judge, "reference_relative", False) else judge.set_floor(x)
    fx = judge.j.feats(x)
    dg = judge.cellmap(g, x, fx)
    tab = codec.prepare(x, gm, judge, fx, dg)

    has_learned = any(tab[l][2] is not None for l in range(L - 1))

    def score(lv, tw):
        xh = Codec.decode(tab, lv, tw)
        d = judge.cellmap(xh, x, fx)
        r = fail(d, dg, eps_w, eps_img, free=None if fm is None else ~fm, eps_loc=eps_loc)
        bad = r[0]
        if getattr(judge, "v3", False):  # T1 sharpness floor: blurred 16 px blocks are culprits like any other
            bad = bad | judge.floor_bad(xh)
        if getattr(judge, "win", False):  # U1 window rule: soft 64 px windows are culprits too
            bad = bad | judge.window_bad(xh)
        if not v2 or has_learned:  # R1.5: guard only on learned leaves; off for fixed/FLUX leaves (and default off)
            structure_bad, _ = structure_fail(xh, x, g, free=None if fm is None else ~fm)
            if v2:
                structure_bad = structure_bad & learned_cells(tab, lv, tw)
            bad = bad | structure_bad
        return xh, bad, r[1], r[2]

    lvl = torch.zeros(1, G, G, dtype=torch.long, device=x.device)
    if fm is not None:  # start every other cell at the finest block level, so each block is a leaf of its own
        lvl[0] = torch.where(fm, fl, fl[fm].max())
    two = torch.zeros_like(lvl, dtype=torch.bool)
    nev = 0
    learned_levels = {l for l, models in codec.learned.items() if 1 in models}
    cands = lambda lv, tw: candidates(lv, tw, learned_levels)
    if fm is not None:  # no merge or drop may touch a texture block
        cands = lambda lv, tw: [c for c in candidates(lv, tw, learned_levels) if not bool(block_any(fm, size(c[1]))[c[2], c[3]])]
    for _ in range(rounds):
        _, bad, _, _ = score(lvl, two)
        nev += 1
        if not bad.any():
            break
        nl, nt = refine(lvl, two, bad)
        if torch.equal(nl, lvl) and torch.equal(nt, two):
            break
        lvl, two = nl, nt
    ok = lambda c, b: not bool(block_any(b, size(c[1]))[c[2], c[3]])
    for _ in range(down_rounds):
        cand = cands(lvl, two)
        if not cand:
            break
        _, bad, _, _ = score(*coarsen(lvl, two, cand))
        keep = [c for c in cand if ok(c, bad)]
        nev += 1
        for _ in range(10):
            if not keep:
                break
            nl, nt = coarsen(lvl, two, keep)
            _, bad, _, _ = score(nl, nt)
            nev += 1
            if not bad.any():
                break
            near = F.max_pool2d(bad[None, None].float(), 17, 1, 8)[0, 0] > 0
            k2 = [c for c in keep if ok(c, near)]
            keep = k2 if len(k2) < len(keep) else keep[: len(keep) // 2]
        if not keep or bad.any():
            break
        lvl, two = nl, nt
    pre = count(lvl, two)
    tried, npol = set(), 0
    fast_pol = (hasattr(judge, "evaluate_batch") and not getattr(judge, "v3", False)
                and not getattr(judge, "win", False) and (v2 or not has_learned) and fm is None)
    pass_only = fast_pol and hasattr(judge, "passes_batch") and eps_w < 100 and eps_img < 100 and not (v2 and has_learned)
    key_fn = lambda c: (c[1], {"merge1": 0, "merge": 1, "drop": 2}[c[0]])
    while npol < polish:  # needless splits: each remaining step down alone, biggest leaves first, fully verified
        cand = [c for c in sorted(cands(lvl, two), key=key_fn) if c not in tried]
        if not cand:
            break
        if fast_pol:
            if pass_only:
                c = cand[0]
                tried.add(c)
                nl, nt = coarsen(lvl, two, [c])
                xh = Codec.decode(tab, nl, nt)
                npol += 1
                if judge.passes_batch([xh])[0]:
                    lvl, two = nl, nt
            else:
                batch = cand[:min(8, polish - npol)]
                layouts = [coarsen(lvl, two, [c]) for c in batch]
                xhs = [Codec.decode(tab, nl, nt) for nl, nt in layouts]
                res_b = judge.evaluate_batch(xhs)
                used = len(batch)
                for idx, (c, (nl, nt), (_, _, bad8)) in enumerate(zip(batch, layouts, res_b)):
                    tried.add(c)
                    d_b = torch.where(bad8.to(x.device), 1e4, 0.0)[None]
                    bad_b = fail(d_b, dg, eps_w, eps_img, eps_loc=eps_loc)[0]
                    if v2 and has_learned:
                        sb, _ = structure_fail(xhs[idx], x, g)
                        bad_b = bad_b | (sb & learned_cells(tab, nl, nt))
                    if not bad_b.any():
                        lvl, two = nl, nt
                        used = idx + 1
                        break
                npol += used
            continue
        else:
            c = cand[0]
            tried.add(c)
            nl, nt = coarsen(lvl, two, [c])
            _, bad, _, _ = score(nl, nt)
            npol += 1
            if not bad.any():
                lvl, two = nl, nt
    xh, bad, worst, img = score(lvl, two)
    judge.gm = None
    out = dict(lvl=lvl, two=two, xh=xh, tab=tab, gm=gm, tokens=count(lvl, two) + len(fixed), pre_polish=pre,
               evals=nev + npol, passed=not bool(bad.any()), worst=worst, img=img, fx=fx, dg=dg)
    return adopt_lowres(out, codec, x)


def overlay(xh, lvl, two):
    a = ((xh[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for l, m in enumerate(masks(lvl)[:-1]):
        s = size(l) * P
        for i, j in m[0].nonzero().tolist():
            dbl = bool(two[0, i * size(l), j * size(l)])
            d.rectangle((j * s, i * s, (j + 1) * s - 1, (i + 1) * s - 1), outline=(40, 220, 255) if dbl else (255, 40, 160), width=2 if dbl else 1)
    return im


@cli.command()
@click.argument("data")
@click.argument("bases")
@click.argument("out")
@click.option("--mode", type=click.Choice(["edge", "plain"]), default="edge")
@click.option("--eps_w", type=float, default=.3)
@click.option("--eps_img", type=float, default=.1)
@click.option("--per_group", type=int, default=8)
@click.option("--polish", type=int, default=200)
@click.option("--leafae", default="", help="leafae.pt: learned 1-token codec for its level, chosen per leaf by LPIPS")
@click.option("--photo/--no_photo", default=True, help="also test the held-out photographed Mondrians (group mondrian_photo)")
@click.option("--grain", is_flag=True, help="grain as statistics: mostly-grain leaves send 9 grain floats and synthesise it; grain cells judged on blur + energy")
def oracle(data, bases, out, mode, eps_w, eps_img, per_group, polish, leafae, photo, grain):
    from ..stream.fluxleaf import Flux2VAE as Flux2

    from .synth import items as synth_items
    from .train import group_label

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    judge = Judge(dev)
    la = None
    if leafae:
        from .leafae import load as load_leafae

        la = load_leafae(leafae, dev)
    codec = Codec(torch.load(bases, map_location="cpu", weights_only=False), dev, mode, la, judge.j.lp, grain)
    flux = Flux2(dev)
    he = torch.load(pathlib.Path(data) / "heldout.pt", weights_only=False)
    seen = {}
    idx = [i for i in range(len(he["ids"])) if (photo or he["group"][i] != "mondrian") and (seen.setdefault(he["group"][i], []).append(i) or len(seen[he["group"][i]]) <= per_group)]
    gname = lambda g: "mondrian_photo" if g == "mondrian" else group_label("heldout", g)
    items = [(he["ids"][i], Image.open(io.BytesIO(he["jpg"][i])).convert("RGB"), gname(he["group"][i])) for i in idx]
    items += synth_items()
    rows = []
    for k, (name, im, grp) in enumerate(items):
        x = torch.from_numpy(np.asarray(im.resize((S, S), Image.LANCZOS)).copy()).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
        with torch.no_grad():
            g = flux.dec(flux.enc(x)).clamp(-1, 1)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        r = run_oracle(codec, judge, x, g, eps_w, eps_img, polish, grain=grain)
        if dev == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        with torch.no_grad():
            fo = r["fx"]
            dsim = lambda a: float((judge.j.feats(a) * fo).sum(1).mean())
            dl, rl = dsim(r["xh"]), dsim(g)
            lpf = lambda a: float(judge.j.lp(a, x).mean())  # strict whole-image LPIPS, grain judged exactly
            lp_dec, lp_ref = lpf(r["xh"]), lpf(g)
        row = dict(name=name, group=grp, tokens=r["tokens"], pre_polish=r["pre_polish"], evals=r["evals"], sec=round(sec, 2), passed=r["passed"], worst=round(r["worst"], 4),
                   img=round(r["img"], 4), dino_patch=round(dl, 4), dino_patch_ref=round(rl, 4), lpips=round(lp_dec, 4), lpips_ref=round(lp_ref, 4),
                   grain_cells=round(float(r["gm"].float().mean()), 3) if r["gm"] is not None else 0., edge_share=round(Codec.edge_share(r["tab"], r["lvl"], r["two"]), 3), learned=Codec.learned_share(r["tab"], r["lvl"], r["two"]),
                   leaves={size(l) * P: int(m.sum()) for l, m in enumerate(masks(r["lvl"]))}, lvl=r["lvl"][0].tolist(), two=r["two"][0].int().tolist())
        rows.append(row)
        print(grp, name[-30:], row["tokens"], "pre", row["pre_polish"], "pass" if row["passed"] else "FAIL", f"{sec:.1f}s", "edge", row["edge_share"], flush=True)
        side = Image.new("RGB", (S * 3, S))
        side.paste(im.resize((S, S)), (0, 0))
        side.paste(Image.fromarray(((r["xh"][0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()), (S, 0))
        side.paste(overlay(r["xh"], r["lvl"], r["two"]), (2 * S, 0))
        side.resize((S * 3 // 2, S // 2)).save(out / "img" / f"{k:03d}_{grp}_{int(row['tokens'])}.jpg", quality=88)
        (out / "full").mkdir(exist_ok=True)  # full-resolution decode, for looking at grain and seams
        Image.fromarray(((r["xh"][0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(out / "full" / f"{k:03d}_{grp}.png")
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    med = {g: float(np.median([r["tokens"] for r in v])) for g, v in by.items()}
    summary = dict(mode=mode, side=S, eps_w=eps_w, eps_img=eps_img, polish=polish, median_tokens=med, native_tokens=NATIVE,
                   native_over_median={g: round(NATIVE / v, 1) for g, v in med.items()}, pass_rate=float(np.mean([r["passed"] for r in rows])),
                   waldo_over_mondrian=round(med.get("waldo", 0) / max(med.get("mondrian", 1), 1), 2),
                   polish_saved_frac=float(np.mean([1 - r["tokens"] / r["pre_polish"] for r in rows])),
                   dino_patch={g: round(float(np.mean([r["dino_patch"] for r in v])), 4) for g, v in by.items()},
                   dino_patch_ref={g: round(float(np.mean([r["dino_patch_ref"] for r in v])), 4) for g, v in by.items()},
                   edge_share={g: round(float(np.mean([r["edge_share"] for r in v])), 3) for g, v in by.items()},
                   lpips={g: round(float(np.mean([r["lpips"] for r in v])), 4) for g, v in by.items()},
                   lpips_ref={g: round(float(np.mean([r["lpips_ref"] for r in v])), 4) for g, v in by.items()},
                   grain_cells={g: round(float(np.mean([r["grain_cells"] for r in v])), 3) for g, v in by.items()}, grain=grain,
                   learned_share={g: round(sum(r["learned"][0] for r in v) / max(1, sum(r["learned"][1] for r in v)), 3) for g, v in by.items()},
                   median_sec=float(np.median([r["sec"] for r in rows])))
    print("summary", summary, flush=True)
    (out / "oracle.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    cli()
