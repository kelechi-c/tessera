"""Judge v5, step J1: canonical feature extractor (section W.3 of docs/reviews/20261110-fable-review.md).

    features(x_u8, g_u8, y_u8) -> {name_k: float32 array [nw_y, nw_x]}   k in (16, 64, 256)

x = original, g = grid-VAE decode, y = candidate.  All three are uint8 [H, W, 3] arrays (H, W multiples of 16, at
least 256).  The function is pure: float32, CPU, eval mode, no autocast, no randomness, no layout or latent input.
Windows are k x k px, stride k / 2, in raster order; window (i, j) covers rows [i k/2, i k/2 + k).  No thresholds
live here: every "excess" is  term(y, x) - term(g, x)  (positive = the candidate is worse than the grid decode), so
y = g gives exactly 0.  Gates and tolerances are J3.

Per-window arrays (name_k):
  lpips        LPIPS-VGG spatial map, y-excess over g, window mean
  dino         DINOv3-B/16 patch distance (1 - cos) excess; DIAGNOSTIC (it does not pass locality, see the results)
  dists_tex    local DISTS texture term (feature means, alpha weights): sum_l alpha (1 - s1), window statistics, excess
  dists_struct local DISTS structure term (covariance, beta weights): sum_l beta (1 - s2), window statistics, excess
               Layers whose window would be a single feature cell (16 px window on the 32 px-resolution layer) have
               no within-window statistics and are skipped (weights not renormalised).
  fine         fine-energy deficit: window mean over detail blocks (original fine std > sharp.ACT) of
               ratio_g - ratio_y, ratio = block fine std / (original block fine std + 1e-4); blocks are 16 px at
               stride 8; 0 where the window has no detail block (fine_n holds the block count)
  seam         (16 and 64 px windows only) blockiness excess on the 8 px grid, see `seam_terms`
  color        low-pass colour: Lab, gaussian sigma 4 px, mean over the window of dE(y, x) - dE(g, x)
Original-side maps (functions of x only, window fractions in [0, 1]):
  grain        8 px cells that are grain (hires.grain_mask v2 rule, constants copied) minus text and thin strokes (round 3)
  texture      16 px blocks that are stochastic texture: fine-scale energy above grain level, gradient
               orientation coherence low over a 32 px neighbourhood, not text
  text         16 px blocks that look like type: high local contrast, bimodal luminance, dilated one block
  fine_n       number of detail blocks per window (original-side)

`features_many(x, g, ys)` shares the x / g computations over several candidates (identical results to features).
`digest(report)` is the SHA-256 of a canonical serialisation, used by the replay test.
"""
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import sharp

WINDOWS = (16, 64, 256)
DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# grain rule constants (TAU, E_CLEAN, CREST; images in [-1, 1])
TAU, E_CLEAN, CREST = .045, .006, 4.5
SIG_LAB = 4.0

_M = {}
DEV = os.environ.get("JUDGE5_DEVICE", "cpu")      # "cuda": the networks (LPIPS, DISTS, DINO) run on the GPU in fp32, everything else on CPU


# ---------------------------------------------------------------- models ------------------------------------------------
def _models(dino=True):
    if "lp" not in _M:
        import lpips
        import os
        import DISTS_pytorch
        from DISTS_pytorch import DISTS

        if DEV != "cpu":                    # exact fp32 and deterministic kernels
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        _M["lp"] = lpips.LPIPS(net="vgg", spatial=True, verbose=False).float().eval().requires_grad_(False).to(DEV)
        m = DISTS(load_weights=False)
        w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
        m.alpha.data, m.beta.data = w["alpha"], w["beta"]
        m = m.float().eval().requires_grad_(False).to(DEV)
        ws = m.alpha.sum() + m.beta.sum()
        _M["dists"] = m
        _M["alpha"] = [a.cpu() for a in torch.split(m.alpha.data / ws, m.chns, dim=1)]
        _M["beta"] = [a.cpu() for a in torch.split(m.beta.data / ws, m.chns, dim=1)]
    if dino and "dino" not in _M:
        from transformers import AutoModel

        d = AutoModel.from_pretrained(DINO, torch_dtype=torch.float32).float().eval().requires_grad_(False).to(DEV)
        _M["dino"] = d
        _M["nreg"] = 1 + getattr(d.config, "num_register_tokens", 0)
    return _M


# ---------------------------------------------------------------- helpers -----------------------------------------------
def _t01(a):
    """uint8 [H, W, 3] -> float32 [1, 3, H, W] in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255.0


def _pool(m, k):
    """[1, 1, H, W] pixel map -> [nw_y, nw_x] window means (k x k, stride k / 2)."""
    return F.avg_pool2d(m, k, stride=k // 2)[0, 0]


def _grid_ok(h, w):
    if h % 16 or w % 16 or min(h, w) < 256:
        raise ValueError(f"image {h}x{w}: sides must be multiples of 16 and at least 256")


def _up(m, f):
    return m.repeat_interleave(f, -2).repeat_interleave(f, -1)


def _gauss_kernel(sigma):
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1).float() ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def _blur(x, sigma):
    """Separable gaussian, reflect padding, as a fixed-order sum over the taps (elementwise adds): the result does not
    depend on the thread count, unlike conv2d."""
    k = [float(v) for v in _gauss_kernel(sigma)]
    r = (len(k) - 1) // 2
    H, W = x.shape[-2:]
    xp = F.pad(x, (r, r, 0, 0), mode="reflect")
    out = 0
    for i, w in enumerate(k):
        out = out + w * xp[..., :, i:i + W]
    xp = F.pad(out, (0, 0, r, r), mode="reflect")
    out = 0
    for i, w in enumerate(k):
        out = out + w * xp[..., i:i + H, :]
    return out


def _lab(x01):
    """sRGB [1, 3, H, W] in [0, 1] -> Lab (D65), float32.  Computed in numpy float64 (single-threaded, so the result is
    independent of torch's thread partition: vectorised pow / cbrt kernels differ in the tail elements)."""
    a = x01.numpy().astype(np.float64)
    lin = np.where(a > .04045, np.power((a + .055) / 1.055, 2.4), a / 12.92)
    M = [[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]]
    xyz = np.concatenate([(M[i][0] * lin[:, 0:1] + M[i][1] * lin[:, 1:2]) + M[i][2] * lin[:, 2:3] for i in range(3)], 1)
    xyz = xyz / np.array([.95047, 1., 1.08883]).reshape(1, 3, 1, 1)
    f = np.where(xyz > .008856, np.cbrt(np.maximum(xyz, 1e-8)), 7.787 * xyz + 16 / 116)
    lab = np.concatenate([116 * f[:, 1:2] - 16, 500 * (f[:, 0:1] - f[:, 1:2]), 200 * (f[:, 1:2] - f[:, 2:3])], 1)
    return torch.from_numpy(lab.astype(np.float32))


# ---------------------------------------------------------------- original-side maps ------------------------------------
def _fine_m11(x01):
    xm = x01 * 2 - 1
    return xm - _blur(xm, 1.5)


def _grain_px(x01):
    """Pixel map of grain 8 px cells (hires.grain_mask v2 rule on the fine band of x)."""
    f = _fine_m11(x01)
    h = F.avg_pool2d(f.square(), 8).mean(1, keepdim=True).sqrt()
    peak = F.max_pool2d(f.abs(), 8).amax(1, keepdim=True)
    m = (h < TAU) & (h > E_CLEAN) & (peak / h.clamp_min(1e-6) < CREST)
    return _up(m.float(), 8)


def _text_px(x01):
    """Pixel map of text-like 16 px blocks: p95 - p5 luminance > .35, and > 80% of pixels within .2 of the block's
    min or max (bimodal), dilated by one block."""
    lum = (x01 * torch.tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    b = F.unfold(lum, 16, stride=16)[0]                              # [256, nb]
    lo, hi = b.quantile(.05, dim=0), b.quantile(.95, dim=0)
    rng = hi - lo
    near = (((b - lo) < .2 * rng) | ((hi - b) < .2 * rng)).float().mean(0)
    m = ((rng > .35) & (near > .8)).float()
    H, W = lum.shape[-2:]
    m = m.view(1, 1, H // 16, W // 16)
    m = F.max_pool2d(m, 3, stride=1, padding=1)
    return _up(m, 16)


def _thin_px(x01):
    """Pixel map of thin-stroke blocks (line art): luma closing / opening with a 5 px element moves some pixel of the 16 px
    block by more than 30 grey levels (the strokes are thinner than 5 px); dilated by one block.  Original only."""
    lum = (x01 * 255 * torch.tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    mx = lambda a: F.max_pool2d(a, 5, 1, 2)
    mn = lambda a: -F.max_pool2d(-a, 5, 1, 2)
    d = torch.maximum(mn(mx(lum)) - lum, lum - mx(mn(lum)))
    m = (F.max_pool2d((d > 30).float(), 16) > 0).float()
    m = F.max_pool2d(m, 3, stride=1, padding=1)
    return _up(m, 16)


def _grain_px_judge(x01, text_px=None):
    """Grain map used by the judge: the hires v2 rule (round-2 rule).  The round-3 text/line-art exclusion was reverted
    (reviewer decision after H2: it did not fix doc_5 and lowered grain_reseed pass rates); `_thin_px` is kept unused."""
    return _grain_px(x01)


def grain_windows_16(x_u8):
    """grain_16 of the judge for an original (numpy [63, 63]); what `features` reports as `grain_16`."""
    return _pool(_grain_px_judge(_t01(x_u8)), 16).numpy()


def _texture_px(x01, text_px):
    """Pixel map of stochastic-texture 16 px blocks: fine-scale std above grain level (TAU), gradient orientation
    coherence of the 32 px neighbourhood below .25, and not text."""
    lum = (x01 * torch.tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    gx = lum[..., :, 1:] - lum[..., :, :-1]
    gy = lum[..., 1:, :] - lum[..., :-1, :]
    gx, gy = F.pad(gx, (0, 1, 0, 0)), F.pad(gy, (0, 0, 0, 1))
    Jxx, Jyy, Jxy = (F.avg_pool2d(t, 32, stride=16, padding=8, count_include_pad=False) for t in (gx * gx, gy * gy, gx * gy))
    tr = Jxx + Jyy
    coh = torch.sqrt(((Jxx - Jyy) ** 2 + 4 * Jxy ** 2).clamp_min(0)) / tr.clamp_min(1e-8)
    coh = coh[..., :x01.shape[-2] // 16, :x01.shape[-1] // 16]
    f = _fine_m11(x01)
    h = F.avg_pool2d(f.square(), 16).mean(1, keepdim=True).sqrt()
    tb = F.avg_pool2d(text_px, 16) > .5
    m = (h >= TAU) & (coh < .25) & ~tb
    return _up(m.float(), 16)


def _fine_std_grid(x01):
    """sharp.fine on the uint8-rounded image (PIL-identical), block std (16 px) at stride 8 over the 3 channels.
    -> [1, 1, nbh, nbw], nb = (H - 16) / 8 + 1."""
    xq = (x01 * 255).round() / 255
    f = sharp.fine(xq, uint8=True)
    mean = F.avg_pool2d(f, 16, stride=8)
    sq = F.avg_pool2d(f.square(), 16, stride=8)
    var = sq.mean(1, keepdim=True) - mean.mean(1, keepdim=True).square()
    return var.clamp_min(0).sqrt()


# ---------------------------------------------------------------- reference-side bundles --------------------------------
def _dino_map(M, x01):
    m = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    s = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    h = M["dino"](pixel_values=((x01 - m) / s).to(DEV)).last_hidden_state[:, M["nreg"]:].float().cpu()
    n, mm = x01.shape[-2] // 16, x01.shape[-1] // 16
    return F.normalize(h, dim=-1).transpose(1, 2).reshape(1, -1, n, mm)


def _prep(M, x01, dino):
    """Per-image network outputs reused across pairs."""
    d = dict(x01=x01, dists=[f.float().cpu() for f in M["dists"].forward_once(x01.to(DEV))])
    if dino:
        d["dino"] = _dino_map(M, x01)
    return d


def _dists_terms(M, fa, fb, k):
    """Window excess terms (tex, struct) maps [nw, nw] for window size k, per DISTS layer (see module doc)."""
    tex = struct = 0
    C1 = C2 = 1e-6
    H = fa[0].shape[-2]
    for i, (a, b) in enumerate(zip(fa, fb)):
        s = H // a.shape[-2]
        kk = k // s
        if kk < 2:
            continue
        st = max(1, kk // 2)
        P = lambda t: F.avg_pool2d(t, kk, stride=st)
        ma, mb = P(a), P(b)
        va, vb = (P(a * a) - ma * ma).clamp_min(0), (P(b * b) - mb * mb).clamp_min(0)
        cov = P(a * b) - ma * mb
        s1 = (2 * ma * mb + C1) / (ma * ma + mb * mb + C1)
        s2 = (2 * cov + C2) / (va + vb + C2)
        tex = tex + (M["alpha"][i] * (1 - s1)).sum(1)
        struct = struct + (M["beta"][i] * (1 - s2)).sum(1)
    return tex[0], struct[0]


def _pair_terms(M, X, Y, dino):
    """Distance maps / terms of candidate bundle Y against original bundle X; everything a window feature needs."""
    out = {}
    out["lpips"] = M["lp"]((X["x01"] * 2 - 1).to(DEV), (Y["x01"] * 2 - 1).to(DEV)).cpu()               # [1, 1, H, W]
    if dino:
        out["dino"] = _up(1 - (X["dino"] * Y["dino"]).sum(1, keepdim=True), 16)
    for k in WINDOWS:
        out[f"dists_{k}"] = _dists_terms(M, X["dists"], Y["dists"], k)
    lx = _blur(_lab(X["x01"]), SIG_LAB)
    ly = _blur(_lab(Y["x01"]), SIG_LAB)
    out["color"] = (ly - lx).square().sum(1, keepdim=True).sqrt()
    return out


def _bundle(x_u8):
    return _t01(x_u8)


class Context:
    """The (x, g) side of `features`, computed once so a search can score many candidates y against it:
    `Context(x_u8, g_u8, dino).features(y_u8)` equals `features(x_u8, g_u8, y_u8, dino)` bit for bit."""

    def __init__(self, x_u8, g_u8, dino=True):
        with _scope():
            _grid_ok(*x_u8.shape[:2])
            self.dino = dino
            self.x_u8, self.g_u8 = x_u8, g_u8
            self.M = _models(dino)
            self.X = _prep(self.M, _t01(x_u8), dino)
            G = _prep(self.M, _t01(g_u8), dino)
            self.tg = _pair_terms(self.M, self.X, G, dino)
            x01 = self.X["x01"]
            text_px = _text_px(x01)
            grain_px = _grain_px_judge(x01, text_px)
            tex_px = _texture_px(x01, text_px)
            self.sx = _fine_std_grid(x01)
            self.act = (self.sx > sharp.ACT).float()
            self.ratio_g = _fine_std_grid(G["x01"]) / (self.sx + 1e-4)
            base = {}
            for k in WINDOWS:
                base[f"grain_{k}"] = _pool(grain_px, k).numpy()
                base[f"texture_{k}"] = _pool(tex_px, k).numpy()
                base[f"text_{k}"] = _pool(text_px, k).numpy()
                kk, st = (k - 16) // 8 + 1, k // 16
                den = F.avg_pool2d(self.act, kk, stride=st)[0, 0]
                base[f"fine_n_{k}"] = (den * kk * kk).numpy()
            self.base = base

    def features(self, y_u8):
        with _scope():
            M, X, tg, dino, act, sx = self.M, self.X, self.tg, self.dino, self.act, self.sx
            Y = _prep(M, _t01(y_u8), dino)
            ty = _pair_terms(M, X, Y, dino)
            r = dict(self.base)
            ratio_y = _fine_std_grid(Y["x01"]) / (sx + 1e-4)
            for k in WINDOWS:
                r[f"lpips_{k}"] = _pool(ty["lpips"] - tg["lpips"], k).numpy()
                if dino:
                    r[f"dino_{k}"] = _pool(ty["dino"] - tg["dino"], k).numpy()
                r[f"dists_tex_{k}"] = (ty[f"dists_{k}"][0] - tg[f"dists_{k}"][0]).numpy()
                r[f"dists_struct_{k}"] = (ty[f"dists_{k}"][1] - tg[f"dists_{k}"][1]).numpy()
                r[f"color_{k}"] = _pool(ty["color"] - tg["color"], k).numpy()
                kk, st = (k - 16) // 8 + 1, k // 16
                num = F.avg_pool2d(((self.ratio_g - ratio_y) * act), kk, stride=st)[0, 0]
                den = F.avg_pool2d(act, kk, stride=st)[0, 0]
                r[f"fine_{k}"] = torch.where(den > 0, num / den.clamp_min(1e-9), torch.zeros_like(num)).numpy()
            r.update(seam_terms(self.x_u8, self.g_u8, y_u8))
            return {n: np.ascontiguousarray(a, dtype=np.float32) for n, a in sorted(r.items())}


def _scope():
    """no_grad plus oneDNN off (oneDNN convolutions vary at 1e-7 between processes; native kernels do not); no global side effects."""
    import contextlib

    st = contextlib.ExitStack()
    st.enter_context(torch.no_grad())
    st.enter_context(torch.backends.mkldnn.flags(enabled=False))
    return st


def _compute(x_u8, g_u8, y_list, dino=True):
    ctx = Context(x_u8, g_u8, dino)
    return [ctx.features(y) for y in y_list]


# ---------------------------------------------------------------- seam term (round 2, R2) -------------------------------
SEAM_PERIOD = 8     # boundaries at multiples of 8 px; the periods 8, 16 and 32 of the spec are all multiples of 8, so
                    # the boundary set of p = 8 contains the others and one pass covers them


def _luma(a):
    a = np.asarray(a, np.float32)
    return (a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114).astype(np.float32)


def _jumps(L):
    """Mean luma jump across every pixel boundary, per 16 px segment.
    -> (Jv [H/16, W-1]: boundary between columns c-1 and c at index c-1, Jh [H-1, W/16])."""
    H, W = L.shape
    dv = np.abs(L[:, 1:] - L[:, :-1]).reshape(H // 16, 16, W - 1).mean(1)
    dh = np.abs(L[1:, :] - L[:-1, :]).reshape(H - 1, W // 16, 16).mean(2)
    return dv.astype(np.float32), dh.astype(np.float32)


def _blockiness(J, axis):
    """b = max(0, J(c) - .5 (J(c-1) + J(c+1))) along `axis` (the boundary axis); same shape, zero at the ends."""
    b = np.zeros_like(J)
    if axis == 1:
        b[:, 1:-1] = np.maximum(0, J[:, 1:-1] - .5 * (J[:, :-2] + J[:, 2:]))
    else:
        b[1:-1, :] = np.maximum(0, J[1:-1, :] - .5 * (J[:-2, :] + J[2:, :]))
    return b


def seam_terms(x_u8, g_u8, y_u8):
    """Blockiness excess b(y) - b(g) on the 8 px grid, luma, natural edges cancelled by subtracting x's jumps from both.
    Pixel-local (reach: the 16 px segment and +-1 px across the boundary).  -> {seam_16, seam_64}: window maxima
    (16 px and 64 px windows, stride half)."""
    H, W = x_u8.shape[:2]
    _grid_ok(H, W)
    jx = _jumps(_luma(x_u8))
    jg = _jumps(_luma(g_u8))
    jy = _jumps(_luma(y_u8))
    S = np.zeros((H, W), np.float32)
    for ax, (a, b, c) in enumerate(zip(jx, jg, jy)):       # ax 0: vertical boundaries (columns), 1: horizontal
        d = _blockiness(c - a, 1 - ax) - _blockiness(b - a, 1 - ax)    # [H/16, W-1] or [H-1, W/16]
        if ax == 0:
            cols = np.arange(1, W)                           # boundary index c-1 -> pixel column c
            keep = (cols % SEAM_PERIOD == 0)
            for i in range(H // 16):
                S[16 * i:16 * i + 16, cols[keep]] = np.maximum(S[16 * i:16 * i + 16, cols[keep]], d[i, keep])
        else:
            rows = np.arange(1, H)
            keep = (rows % SEAM_PERIOD == 0)
            for j in range(W // 16):
                S[np.ix_(rows[keep], np.arange(16 * j, 16 * j + 16))] = np.maximum(
                    S[np.ix_(rows[keep], np.arange(16 * j, 16 * j + 16))], d[keep, j][:, None])
    t = torch.from_numpy(S)[None, None]
    return {f"seam_{k}": np.ascontiguousarray(F.max_pool2d(t, k, stride=k // 2)[0, 0].numpy(), dtype=np.float32)
            for k in (16, 64)}


# ---------------------------------------------------------------- public API --------------------------------------------
def features(x_u8, g_u8, y_u8, dino=True):
    """Per-window features of candidate y against original x and grid decode g (see module doc)."""
    return _compute(x_u8, g_u8, [y_u8], dino)[0]


def features_many(x_u8, g_u8, ys, dino=True):
    """`features` for several candidates of one (x, g) pair; each result equals features(x, g, y)."""
    return _compute(x_u8, g_u8, list(ys), dino)


def digest(report):
    """SHA-256 of a canonical serialisation of a features() report (names sorted, shapes, float32 bytes)."""
    h = hashlib.sha256()
    for n in sorted(report):
        a = np.ascontiguousarray(report[n], dtype=np.float32)
        h.update(n.encode() + b"|" + str(a.shape).encode() + b"|" + a.tobytes())
    return h.hexdigest()


EXCESS = ("lpips", "dino", "dists_tex", "dists_struct", "fine", "color", "seam")
