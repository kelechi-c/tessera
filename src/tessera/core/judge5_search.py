"""Judge v5 as a search evaluator.

  evaluate(x_u8, g_u8, y_u8)  -> verdict of record on 8-bit images: score = max_gate excess / scale, pass iff score <= T
                                  (T, scales, grain threshold frozen in a json file, see `Frozen`)
  attribution                 -> a [64, 64] bool map of 8 px cells for the search to refine (rules below)

Attribution (per failing gate g, i.e. excess_g / scale_g > T; cells are 8 px):
  16 px max gates     the cells of every failing 16 px window
  64 / 256 px max     the cells of every failing window whose own 16 px normalised excess for that term is above the median
                      of the 16 px values inside the window
  mean256 gates       the top 10% of the image's cells by that term's 16 px normalised excess
  lpips cells in grain windows count as zero (the same weights as the verdict).
"""
import json
import os
import pathlib

import numpy as np

from . import judge5 as J
from . import judge5_r2 as R

G8 = 64
BIG = 1e4


class Frozen:
    """T, per-gate scales and the grain threshold, loaded from the frozen json (`frozen` key or flat)."""

    def __init__(self, path):
        d = json.loads(pathlib.Path(path).read_text())
        d = d.get("frozen", d)
        self.T = float(d["T"])
        if os.environ.get("J5_T"):                 # tightening sweep: override the threshold, scales unchanged
            self.T = float(os.environ["J5_T"])
        self.thr = d["grain_thr"]
        self.scale = np.array([d["scale"][k] for k in R.KEYS], np.float64)
        self.path = str(path)


def quantise(xm11):
    """[1, 3, H, W] in [-1, 1] -> uint8 [H, W, 3] exactly as the decode is written to PNG."""
    return ((xm11[0].detach().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def _cells(a16):
    """16 px window map [63, 63] (stride 8) -> value per 8 px cell [64, 64]: max over the windows that cover the cell."""
    v = np.full((G8, G8), -np.inf, np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            v[di:di + 63, dj:dj + 63] = np.maximum(v[di:di + 63, dj:dj + 63], a16)
    return v


def _gate_arrays(f, thr):
    """{(term, k): window map used by the verdict} (lpips with grain weights, as in judge5_r2.gate_vector)."""
    out = {}
    for t in R.TERMS:
        a16, a64, a256 = f[f"{t}_16"], f[f"{t}_64"], f[f"{t}_256"]
        if t == "lpips" and thr is not None:
            w = (f["grain_16"] < thr).astype(np.float32)
            out[(t, 16)] = np.where(w > 0, a16, -np.inf)
            out[(t, 64)], out[(t, 256)] = R._block(a16, 64, w), R._block(a16, 256, w)
        else:
            out[(t, 16)], out[(t, 64)], out[(t, 256)] = a16, a64, a256
    out[("seam", 16)], out[("seam", 64)] = f["seam_16"], f["seam_64"]
    return out


def attribution(f, z, fr):
    """Bad 8 px cells [64, 64] bool from the features f and the normalised gate vector z (see module doc)."""
    bad = np.zeros((G8, G8), bool)
    arrs = _gate_arrays(f, fr.thr)
    sc = dict(zip(R.KEYS, fr.scale))
    for key, zg in zip(R.KEYS, z):
        if zg <= fr.T:
            continue
        term, kind = key.rsplit("_", 1)[0], key.rsplit("_", 1)[1]
        if kind == "max16":
            m = (arrs[(term, 16)] / sc[key]) > fr.T
            for i, j in zip(*np.nonzero(m)):
                bad[i:i + 2, j:j + 2] = True
            continue
        # per-cell normalised excess of this term at 16 px
        a16 = arrs[(term, 16)] if term != "lpips" else np.where(np.isfinite(arrs[("lpips", 16)]), arrs[("lpips", 16)], 0.0)
        cv = _cells(a16) / sc[f"{term}_max16"]
        if kind in ("max64", "max256"):
            k = int(kind[3:])
            wins = arrs[(term, k)] / sc[key]
            n, st = (8, 4) if k == 64 else (32, 16)
            for i, j in zip(*np.nonzero(wins > fr.T)):
                blk = cv[st * i:st * i + n, st * j:st * j + n]
                bad[st * i:st * i + n, st * j:st * j + n] |= blk > np.median(blk)
        else:  # mean256
            thr = np.quantile(cv, .9)
            bad |= cv >= thr
    return bad


class V5:
    """Evaluator bound to one (x, g) pair; per-candidate cost is one `Context.features` call."""

    def __init__(self, fr, x_u8, g_u8):
        self.fr = fr
        self.ctx = J.Context(x_u8, g_u8, dino=False)

    def evaluate(self, y_u8, attribute=True):
        f = self.ctx.features(y_u8)
        v = R.gate_vector(f, f, self.fr.thr)
        z = v / self.fr.scale
        i = int(z.argmax())
        out = dict(score=float(z[i]), argmax=R.KEYS[i], passed=bool(z[i] <= self.fr.T), T=self.fr.T)
        if attribute:
            out["bad8"] = attribution(f, z, self.fr) if not out["passed"] else np.zeros((G8, G8), bool)
        return out


def evaluate(x_u8, g_u8, y_u8, frozen, attribute=False):
    """Verdict of record on three 8-bit images (convenience: builds the context each time)."""
    fr = frozen if isinstance(frozen, Frozen) else Frozen(frozen)
    return V5(fr, x_u8, g_u8).evaluate(y_u8, attribute)
