"""Judge v5 as a function (frozen).

verdict = max over 18 gates of (excess / frozen scale), pass iff <= T (1.7759), plus the `mae32` gate (limit 2.5 on a 0..255
scale, 32 px windows).  The gates compare the candidate with the original *relative to the FLUX.2 VAE round trip* of that
original (`vae`), so all three images are needed: ORIG, VAE (= decode(encode(ORIG)) by the FLUX.2 VAE) and CAND.  Window
features at 16 / 64 / 256 px: LPIPS-VGG (grain windows weighted out), DISTS texture, colour, fine energy, block seams.

Reference device: the scales and T were frozen on an NVIDIA L4, CUDA float32 with deterministic kernels and TF32 off; the CPU
path computes the same quantities in float32 but is not guaranteed bit-identical, so near-threshold verdicts can differ in
the last digits.  Needs the `encoder` extras (lpips, DISTS-pytorch; their weights are downloaded/shipped by those packages).
"""
import importlib.resources as ir
import os

import numpy as np
from PIL import Image


def frozen_path():
    return str(ir.files("tessera") / "data" / "frozen_v5.json")


def _u8(a):
    if isinstance(a, (str, os.PathLike)):
        a = Image.open(a)
    if isinstance(a, Image.Image):
        a = a.convert("RGB")
        if a.size != (512, 512):
            a = a.resize((512, 512), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(a, np.uint8))


def judge(orig, vae, cand, device=None, attribute=False):
    """Verdict of record on three images (paths, PIL images or uint8 arrays; resized to 512 px if needed).
    Returns dict(score, passed, argmax, T, mae32[, bad8])."""
    from .core import judge5_fast as Fa
    from .core import judge5_sweep  # noqa: F401  (activates the mae32 gate on the fast path)
    from .core import judge5_search as S

    r = Fa.evaluate(_u8(orig), _u8(vae), _u8(cand), S.Frozen(frozen_path()), dev=device, attribute=attribute)
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v) for k, v in r.items()}
