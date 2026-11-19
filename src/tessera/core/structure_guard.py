"""Aligned structure guard: windowed correlation of aligned luminance gradients between a decode and the
original, relative to a strong reference (the FLUX.2 VAE round trip).

The judge used for token selection (LPIPS-VGG + DINO) accepts a decode whose small text has been replaced
by letter-like marks.  This guard is the independent aligned-structure term that closes that hole.  The
older guard version lives below and is unchanged unless the v2 keyword arguments are passed, so the
oracle keeps its current behaviour until it opts in.

v1 (in use in the oracle, threshold .30):
    `excess(rec, orig, reference)` -> windowed correlation loss map in 64 px windows, stride 32.
    A decode is rejected when the worst window loss is above the threshold.

v2 (step P0.8; opt in with `guard=`/`windows=`/`mask=`):
    * grain / texture exemption: the correlation is computed over unmasked (non-grain, non-texture-block)
      pixels only -- both gradient fields are multiplied by the valid-pixel mask and every pooled term is
      divided by the pooled mask.  Windows with less than `min_frac` unmasked pixels are skipped.
    * two window sizes, 32 px and 64 px: a decode is rejected when the worst window at either size is above
      the threshold.  A 64 px window averages a line of garbled text with the blank paper around it; 32 px
      keeps the line.
    * the threshold is set from the P0.8 validation set (qtok/p08_guard_analyze.py), not from examples.

Nothing here is enabled by default: `failing_cells()` called the way qtok/hires.py calls it (positional
args, `free=` as a cell mask) runs the unchanged v1 path.  See `guard_verdict()` for the v2 entry point.

Mask convention: `mask` is a float pixel mask [B, 1, S, S] or [1, 1, S, S] with 1 = valid pixel (not grain,
not a texture block).  `free` keeps its old meaning for backwards compatibility in the v2 path too: a cell
mask [G, G] (True = exempt cell, e.g. a texture block); it is expanded to pixels and intersected with `mask`.
"""
import torch
import torch.nn.functional as F

P = 8            # px per cell
G = 64           # cells per side at 512 px
TAU = .045       # grain threshold: fine-scale std below this is a grain cell (matches qtok/hires.py)
MIN_FRAC = .25   # v2: skip windows with less than this fraction of unmasked pixels
WINDOWS_V2 = (32, 64)


def _gradients(x):
    """dx, dy forward differences with replicate padding at the right/bottom border (v1 formula)."""
    return (F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0), mode="replicate"),
            F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode="replicate"))


def _pool(x, window):
    return F.avg_pool2d(x, window, window // 2)


def gradient_correlation(a, b, window=64, mask=None, min_frac=MIN_FRAC):
    """Windowed correlation of aligned luminance gradients, stride half a window.

    a, b: images [-1, 1], [B, 3, S, S].  With `mask=None` this is byte-for-byte the v1 formula.  With a
    pixel mask, both gradient fields are masked and each pooled moment is divided by the pooled mask;
    windows with a masked fraction below `min_frac` are returned as NaN (skipped).
    """
    a, b = a.float().mean(1, keepdim=True), b.float().mean(1, keepdim=True)
    u, v = _gradients(a), _gradients(b)
    values = []
    if mask is None:
        for p, q in zip(u, v):
            mu, mv = _pool(p, window), _pool(q, window)
            cov = _pool(p * q, window) - mu * mv
            var_u = (_pool(p * p, window) - mu * mu).clamp_min(1e-8)
            var_v = (_pool(q * q, window) - mv * mv).clamp_min(1e-8)
            values.append(cov / (var_u.sqrt() * var_v.sqrt() + 1e-3))
    else:
        m = mask.float()
        if m.shape[0] == 1 and a.shape[0] != 1:
            m = m.expand(a.shape[0], -1, -1, -1)
        if m.shape[-2:] != a.shape[-2:]:
            raise ValueError(f"mask spatial shape {tuple(m.shape[-2:])} != image {tuple(a.shape[-2:])}")
        n = _pool(m, window)
        valid = n >= min_frac
        for p, q in zip(u, v):
            mu = _pool(p * m, window) / n.clamp_min(1e-6)
            mv = _pool(q * m, window) / n.clamp_min(1e-6)
            cov = _pool(p * q * m, window) / n.clamp_min(1e-6) - mu * mv
            var_u = (_pool(p * p * m, window) / n.clamp_min(1e-6) - mu * mu).clamp_min(1e-8)
            var_v = (_pool(q * q * m, window) / n.clamp_min(1e-6) - mv * mv).clamp_min(1e-8)
            corr = cov / (var_u.sqrt() * var_v.sqrt() + 1e-3)
            values.append(torch.where(valid, corr, torch.full_like(corr, float("nan"))))
    return sum(values) / 2


def excess(rec, orig, reference, window=64, mask=None, min_frac=MIN_FRAC):
    """v1: correlation-loss map of `rec` against `orig` relative to `reference` (larger = worse aligned structure)."""
    return gradient_correlation(reference, orig, window, mask, min_frac) - \
        gradient_correlation(rec, orig, window, mask, min_frac)


def worst_window(rec, orig, reference, window=64, mask=None, min_frac=MIN_FRAC):
    """Worst single-window correlation loss (>0 = worse aligned structure than the reference). NaN if no window is valid."""
    ex = excess(rec, orig, reference, window, mask, min_frac)
    flat = ex.flatten()
    if torch.isnan(flat).all():
        return float("nan")
    return float(torch.nan_to_num(flat, nan=-1e9).max())


def cell_mask_to_pixels(cell_mask, px=P):
    """[G, G] bool cell mask -> [1, 1, S, S] float pixel mask (1 = valid pixel)."""
    cm = cell_mask[None, None].float()
    return cm.repeat_interleave(px, -2).repeat_interleave(px, -1)


def grain_cells(x, tau=TAU, px=P):
    """Fine-scale std per 8 px cell, thresholded: True where the cell is 'grain' (same rule as qtok/hires.grain_mask).

    Uses the codec's own blur (torchvision gaussian_blur, kernel 9, sigma 1.5), so the v2 grain mask matches the
    mask the judge and the grain synthesiser use.
    """
    from torchvision.transforms.functional import gaussian_blur
    std = F.avg_pool2d((x.float() - gaussian_blur(x.float(), 9, 1.5)).square(), px).mean(1).sqrt()[0]
    return std < tau


def non_grain_mask(x, tau=TAU, px=P, free=None):
    """Pixel mask [1, 1, S, S] of non-grain (and, if `free` is given, non-exempt) pixels: 1 = valid for correlation.

    `free` keeps v1's meaning: a [G, G] cell mask of exempt cells (texture blocks); it is expanded to pixels.
    """
    valid = ~grain_cells(x, tau, px)
    if free is not None:
        valid = valid & free.to(valid.device).bool()
    return cell_mask_to_pixels(valid, px)


def failing_cells(rec, orig, reference, free=None, threshold=.3, windows=None, mask=None, min_frac=MIN_FRAC):
    """Flag 8 px cells whose aligned structure deviates more than `threshold` from the reference.

    v1 (no `windows`/`mask`): the current oracle call.  64 px windows, stride 32, a window is only marked
    when all its cells are outside the `free` cell mask (texture blocks), exactly the behaviour before P0.8.

    v2 (`windows=(32, 64)` and/or `mask=`): masked correlation, a window is marked when its masked fraction
    is at least `min_frac`, and the union over every window size is returned.  `worst` is the maximum loss
    over all valid windows and sizes (NaN if none is valid).
    """
    v2 = windows is not None or mask is not None
    if not v2:
        losses = excess(rec, orig, reference)[0, 0]
        bad = torch.zeros(orig.shape[-2] // P, orig.shape[-1] // P, dtype=torch.bool, device=orig.device)
        for i, j in (losses > threshold).nonzero().tolist():
            patch = (slice(i * 4, i * 4 + 8), slice(j * 4, j * 4 + 8))
            if free is None or bool(free[patch].all()):
                bad[patch] = True
        return bad, float(torch.nan_to_num(losses, nan=-1e9).max())

    windows = tuple(windows) if windows is not None else (64,)
    bad = torch.zeros(orig.shape[-2] // P, orig.shape[-1] // P, dtype=torch.bool, device=orig.device)
    worst = float("nan")
    for k in windows:
        if mask is None and free is not None:
            m = cell_mask_to_pixels(~free.bool(), P).to(orig.device)
        elif mask is not None and free is not None:
            m = mask.float() * cell_mask_to_pixels(free.bool(), P).to(orig.device)
        else:
            m = mask
        ex = excess(rec, orig, reference, k, m, min_frac)[0, 0]
        step, patch = (k // 2) // P, k // P
        for i, j in (ex > threshold).nonzero().tolist():
            bad[step * i:step * i + patch, step * j:step * j + patch] = True
        w = float(torch.nan_to_num(ex, nan=-1e9).max())
        worst = w if worst != worst else max(worst, w)
    return bad, worst


def guard_verdict(rec, orig, reference, threshold=.30, windows=WINDOWS_V2, mask=None, min_frac=MIN_FRAC):
    """v2 image-level verdict: `reject` when the worst valid window at either window size is above `threshold`.

    Returns a dict with the per-window worst losses so a threshold sweep can be done offline.
    """
    per = {int(k): worst_window(rec, orig, reference, k, mask, min_frac) for k in windows}
    vals = [v for v in per.values() if v == v]
    worst = max(vals) if vals else float("-inf")
    return dict(reject=bool(worst > threshold), worst=worst, per_window=per,
                windows=list(windows), threshold=threshold)
