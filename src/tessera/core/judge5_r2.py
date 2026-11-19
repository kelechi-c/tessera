"""Judge v5 gate maths (frozen scales are applied by the caller): the 18 gates and the block means.

Trimmed copy of qtok/judge5_r2.py: only TERMS, KEYS, _block and gate_vector (the bank-evaluation code stays in qtok)."""
import numpy as np

TERMS = ("lpips", "fine", "color", "dists_tex")
KEYS = [f"{t}_{g}" for t in TERMS for g in ("max16", "max64", "max256", "mean256")] + ["seam_max16", "seam_max64"]


def _block(a16, k, w):
    """Weighted mean of the 16 px window map a16 [63, 63] (stride 8) inside each k px window (stride k/2); w = weights."""
    n = {64: 7, 256: 31}[k]
    st = k // 16
    out = []
    for i in range(0, a16.shape[0] - n + 1, st):
        row = []
        for j in range(0, a16.shape[1] - n + 1, st):
            ww = w[i:i + n, j:j + n]
            d = ww.sum()
            row.append(float((a16[i:i + n, j:j + n] * ww).sum() / d) if d > 0 else 0.0)
        out.append(row)
    return np.array(out, np.float32)


def gate_vector(f, sm, gthr=None):
    """dict key -> value for one control (f: judge5 features, sm: seam terms)."""
    v = {}
    for t in TERMS:
        a16, a64, a256 = f[f"{t}_16"], f[f"{t}_64"], f[f"{t}_256"]
        if t == "lpips" and gthr is not None:
            w = (f["grain_16"] < gthr).astype(np.float32)
            m16 = a16[w > 0]
            v[f"{t}_max16"] = float(m16.max()) if m16.size else 0.0
            b64, b256 = _block(a16, 64, w), _block(a16, 256, w)
            v[f"{t}_max64"] = float(b64.max())
            v[f"{t}_max256"] = float(b256.max())
            v[f"{t}_mean256"] = float(b256.mean())
        else:
            v[f"{t}_max16"], v[f"{t}_max64"], v[f"{t}_max256"] = float(a16.max()), float(a64.max()), float(a256.max())
            v[f"{t}_mean256"] = float(a256.mean())
    v["seam_max16"], v["seam_max64"] = float(sm["seam_16"].max()), float(sm["seam_64"].max())
    return np.array([v[k] for k in KEYS], np.float64)
