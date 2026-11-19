"""Sharpness floor (revision 5, T1 + T2p step 0): fine-detail energy per 16 px block.

The measure is the reviewer's `docs/reviews/fable-results/t0/blur_blocks.py` as torch code.  `blur_blocks.py` builds
its band-pass with PIL's `GaussianBlur(1.5)`, which is a three-pass box blur with the fractional radius Pillow derives
from sigma 1.5 (`_gaussian_blur_radius`: r = 1.0384615...).  We reproduce exactly that filter so the reference count
(68 +/- 3 blurred blocks on the Waldo pair) holds; a true 9x9 Gaussian differs on the same pair (63 blocks) because
the count is sensitive to the tail of the kernel.

Images are handled as 8-bit rasters, as the reference does: `uint8=True` rounds the image and every blur pass to
[0, 1] in 1/255 steps (PIL's own behaviour).  The differentiable path (`uint8=False`, no per-pass rounding, the three
box passes folded into one separable kernel) is what T2p's `L_sharp` uses on decoded pixels.

  fine(x)         [B, 3, H, W] in [0, 1] -> x minus the band-pass blur
  block_std(x)    -> [B, H // 16, W // 16]: std of fine(x) over each 16 px block AND the 3 channels
  blurred(xh, x)  -> 16 px blocks with real detail in x (std > .02) where the decode keeps < .6 of the energy

The floor rejects nothing the FLUX.2 VAE round trip does (0 of about 14,000 detail blocks on the S5 decodes), while
the shipped decode blurs 153 and the adopted rule 294, in clusters.  T1 adds `blurred` to the judge as culprits; T2p
trains `L_sharp` on `block_std` (differentiable path).

usage (CPU check, the T.7 step 0 test):
  python -m qtok.sharp check DIR   # e.g. docs/reviews/fable-results/s5
"""
import math
import pathlib

import click
import numpy as np
import torch
import torch.nn.functional as F

SIG = 1.5     # Gaussian sigma the reviewer's PIL filter approximates
BLOCK = 16    # px per detail block
ACT = .02     # a block "has real detail" when its original fine std is above this
FLOOR = .6    # the decode must keep at least this share of the original fine-detail energy
PASSES = 3    # Pillow's GaussianBlur pass count

_k_cache = {}


def _radius():
    """Pillow's fractional box radius for GaussianBlur(sigma=1.5), passes=3 (src/libImaging/BoxBlur.c)."""
    sigma2 = SIG * SIG / PASSES
    L = math.sqrt(12.0 * sigma2 + 1.0)
    n = math.floor((L - 1.0) / 2.0)
    a = (2 * n + 1) * (n * (n + 1) - 3 * sigma2)
    a /= 6 * (sigma2 - (n + 1) * (n + 1))
    return n + a


R = _radius()  # 1.0384615...


def _kernels(dev, dtype):
    """The one-pass fractional box kernel of the PIL filter."""
    key = (str(dev), dtype)
    if key not in _k_cache:
        n = int(math.floor(R))
        ww = 1.0 / (2 * R + 1)
        fw = (R - n) / (2 * R + 1)
        _k_cache[key] = torch.tensor([fw] + [ww] * (2 * n + 1) + [fw], device=dev, dtype=dtype)
    return _k_cache[key]


def _conv1d(x, k, dim):
    """Separable convolution along `dim` with replicate padding and per-channel weights; x [B, C, H, W]."""
    c = x.shape[1]
    p = (k.numel() - 1) // 2
    if dim == 3:
        return F.conv2d(F.pad(x, (p, p, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def blur(x, uint8=False):
    """Band-pass blur.  uint8=True: Pillow-identical (input quantised, every pass rounded to 8 bits)."""
    if uint8:
        k = _kernels(x.device, x.dtype)
        y = (x * 255).round().clamp(0, 255) / 255
        for _ in range(PASSES):
            y = _conv1d(y, k, 3)
            y = (y * 255).round().clamp(0, 255) / 255
        for _ in range(PASSES):
            y = _conv1d(y, k, 2)
            y = (y * 255).round().clamp(0, 255) / 255
        return y
    k = _kernels(x.device, x.dtype)
    y = x
    for _ in range(PASSES):  # per-pass replicate padding matches PIL's clamp-to-edge border
        y = _conv1d(y, k, 3)
        y = _conv1d(y, k, 2)
    return y


def fine(x, uint8=False):
    """x [B, 3, H, W] in [0, 1] -> x minus the band-pass blur (the reviewer's `fine`)."""
    return x - blur(x, uint8=uint8)


def block_std(x, block=BLOCK, uint8=False):
    """x [B, 3, H, W] in [0, 1] -> [B, H // block, W // block]: std of fine(x) over the block and the 3 channels.

    Matches numpy `fine.reshape(h, 16, w, 16, 3).std(axis=(1, 3, 4))` (population std over 3 * 256 values).
    The input is clamped to [0, 1] first: the reference measure works on 8-bit rasters, so out-of-range decode
    values clip exactly as they do when the decode is written to PNG (`to_png`); without the clamp a decode that
    overshoots inflates its own fine energy and can hide real blur from the floor."""
    x = x.clamp(0, 1)
    f = fine(x, uint8=uint8)
    mean = F.avg_pool2d(f, block)               # per-channel block means [B, 3, h, w]
    sq = F.avg_pool2d(f.square(), block)        # E[x^2] per channel
    var = sq.mean(1, keepdim=True) - mean.mean(1, keepdim=True).square()
    return var.clamp_min(0).sqrt()[:, 0]


def block_std_m11(x, block=BLOCK, uint8=False):
    """As `block_std`, for images in [-1, 1] (the codec and the VAE work in that range)."""
    return block_std((x.clamp(-1, 1) + 1) / 2, block, uint8)


def detail_ratio(xh, x, block=BLOCK, act=ACT, uint8=True):
    """Per-block fine-energy ratio (decode / original), the original detail mask, and both std maps."""
    so = block_std(x, block, uint8)
    sd = block_std(xh, block, uint8)
    return sd / (so + 1e-4), so > act, so, sd


def blurred(xh, x, block=BLOCK, act=ACT, floor=FLOOR, uint8=True):
    """x, xh [B, 3, H, W] in [0, 1] -> [B, H // block, W // block] bool: detail block lost more than (1 - floor)."""
    ratio, act_m, _, _ = detail_ratio(xh, x, block, act, uint8)
    return act_m & (ratio < floor)


def blurred_m11(xh, x, block=BLOCK, act=ACT, floor=FLOOR, uint8=True):
    """`blurred` for images in [-1, 1] (the codec's convention)."""
    return blurred((xh + 1) / 2, (x + 1) / 2, block, act, floor, uint8)


def n_blurred(xh, x, **kw):
    return int(blurred(xh, x, **kw).sum())


def median_ratio(xh, x, block=BLOCK, act=ACT, uint8=True):
    """Median fine-energy ratio over detail blocks, per image [B] (1.0 when an image has no detail block)."""
    ratio, act_m, _, _ = detail_ratio(xh, x, block, act, uint8)
    out = []
    for b in range(ratio.shape[0]):
        v = ratio[b][act_m[b]]
        out.append(float(v.median()) if int(act_m[b].sum()) else 1.0)
    return torch.tensor(out, device=ratio.device)


def window_ratio(xh, x, block=BLOCK, act=ACT, win=64, uint8=True):
    """Mean fine-energy ratio per `win` px window -> [B, H // win, W // win] (NaN where no detail block; T1 window rule)."""
    ratio, act_m, _, _ = detail_ratio(xh, x, block, act, uint8)
    k = win // block
    num = F.avg_pool2d((ratio * act_m)[:, None], k)[:, 0]
    den = F.avg_pool2d(act_m[:, None].float(), k)[:, 0]
    return torch.where(den > 0, num / den.clamp_min(1e-6), torch.full_like(num, float("nan")))


def window_stats(xh, x, block=BLOCK, act=ACT, floor=.9, win=64, stride=32, min_blocks=4, uint8=True, reference=None,
                 reference_tol=0.0):
    """U1 window rule on one image pair.

    In every ``win`` px window (overlapping, ``stride`` px apart) that holds at least ``min_blocks`` detail blocks
    (original fine std > ``act``), the mean fine-energy ratio of those blocks must be at least ``floor``.  Returns
    (fail_windows, window_total, culprit_blocks, median, detail, blocks_below):
      fail_windows   how many windows break the rule (target 0)
      window_total   windows with enough detail to be judged
      culprit_blocks [h // block, w // block] bool: detail blocks below ``floor`` inside a failing window
      median         median fine-energy ratio over the image's detail blocks
      detail         number of detail blocks
      blocks_below   detail blocks with ratio < floor
    """
    ratio, act_m, _, _ = detail_ratio(xh, x, block, act, uint8)
    B = ratio.shape[-1]
    k, s = win // block, stride // block
    num = F.unfold((ratio * act_m)[:, None], k, stride=s)          # [B, k*k, nW]
    den = F.unfold(act_m[:, None].float(), k, stride=s)            # [B, k*k, nW]
    refnum = None
    if reference is not None:
        rr = detail_ratio(reference, x, block, act, uint8)[0]
        refnum = F.unfold((rr * act_m)[:, None], k, stride=s)
    idx = F.unfold(torch.arange(B * B, device=ratio.device).view(1, 1, B, B).float(), k, stride=s)[0].long()
    out = []
    for b in range(ratio.shape[0]):
        cnt = den[b].sum(0)
        mean = num[b].sum(0) / cnt.clamp_min(1)
        det = cnt >= min_blocks
        threshold = torch.full_like(mean, floor)
        if refnum is not None:  # reference-relative (judge v4): never stricter than the actual grid VAE reference
            threshold = torch.minimum(threshold, refnum[b].sum(0) / cnt.clamp_min(1) - float(reference_tol))
        fail = det & (mean < threshold - 1e-6)
        bad = torch.zeros(B * B, dtype=torch.bool, device=ratio.device)
        if bool(fail.any()):
            sel = idx[:, fail]                                      # [k*k, n_fail]
            hit = sel[(act_m[b].reshape(-1)[sel]) & (ratio[b].reshape(-1)[sel] < threshold[fail][None, :])]
            bad[hit] = True
        m = act_m[b]
        med = float(ratio[b][m].median()) if bool(m.any()) else 1.0
        out.append(dict(fail_windows=int(fail.sum()), window_total=int(det.sum()),
                        culprit_blocks=bad.view(B, B), median=round(med, 4), detail=int(m.sum()),
                        blocks_below=int(((ratio[b] < floor) & m).sum())))
    return out


def summary(xh, x, **kw):
    """Blur audit of one decode against its original: (blurred blocks, detail blocks, median ratio, window-rule fails)."""
    b = blurred(xh, x, **kw)
    ratio, act_m, _, _ = detail_ratio(xh, x, kw.get("block", BLOCK), kw.get("act", ACT), kw.get("uint8", True))
    win = window_ratio(xh, x, kw.get("block", BLOCK), kw.get("act", ACT), 64, kw.get("uint8", True))[0]
    return dict(blurred=int(b.sum()), detail=int(act_m.sum()), median=float(torch.nanmedian(win).nan_to_num(nan=1.)),
                window_fail=int(((win < .9) & ~torch.isnan(win)).sum()))


# ---------------------------------------------------------------- CPU check (T.7 step 0) ----------------------------------
def _load(p):
    from PIL import Image

    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def check_pair(orig, dec, uint8=True):
    """(blurred count, detail blocks, median ratio) of a decode PNG against its original PNG."""
    o, d = _load(orig), _load(dec)
    ratio, act_m, _, _ = detail_ratio(d, o, uint8=uint8)
    n = int((act_m & (ratio < FLOOR)).sum())
    med = float(ratio[act_m].median()) if bool(act_m.any()) else 1.0
    return n, int(act_m.sum()), med


@click.group()
def cli():
    pass


@cli.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False))
def check(root):
    """T.7 step 0: 68 +/- 3 blurred blocks on the shipped Waldo decode, 0 on its VAE decode."""
    root = pathlib.Path(root)
    pair = root / "decodes_sh" / "39_waldo_altdorfer_crucifixion_"
    sh = check_pair(str(pair) + "orig.png", str(pair) + "shipped_e80.png")
    va = check_pair(str(pair) + "orig.png", str(pair) + "vae.png")
    print(f"shipped: blurred {sh[0]} / detail {sh[1]}, median ratio {sh[2]:.3f}")
    print(f"vae:     blurred {va[0]} / detail {va[1]}, median ratio {va[2]:.3f}")
    ok = abs(sh[0] - 68) <= 3 and va[0] == 0
    print(f"CHECK {'PASS' if ok else 'FAIL'} (shipped {sh[0]} expected 68 +/- 3; vae {va[0]} expected 0)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
