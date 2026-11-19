"""Judge v5 as a search evaluator (round 3, H3).  Opt-in: nothing in hires.py / vae16.py changes and v3 stays the default.

  evaluate(x_u8, g_u8, y_u8)  -> verdict of record on 8-bit images: score = max_gate excess / scale, pass iff score <= T
                                  (T, scales, grain threshold frozen in a json file, see `Frozen`)
  attribution                 -> a [64, 64] bool map of 8 px cells for the search to refine (rules below)
  Judge5(dev, frozen_json)    -> a drop-in `judge` for hires.run_oracle / vae16.run_oracle_vae16: its `cellmap` hands the
                                  search a map that is 0 where v5 passes and huge on the attribution cells, so the existing
                                  `fail()` returns exactly those cells as culprits.  No DINO, no v3 floor, no U1 window rule
                                  (v3 = win = False).  The structure guard is off in run_oracle only with v2=True and no
                                  learned leaves; the S2 text gate of route512 is untouched.
  python -m qtok.judge5_search replay CASE.pt   decode in float32, save PNG, re-evaluate, assert the stored score to 1e-6

Attribution (per failing gate g, i.e. excess_g / scale_g > T; cells are 8 px):
  16 px max gates     the cells of every failing 16 px window
  64 / 256 px max     the cells of every failing window whose own 16 px normalised excess for that term is above the median
                      of the 16 px values inside the window
  mean256 gates       the top 10% of the image's cells by that term's 16 px normalised excess
  lpips cells in grain windows count as zero (the same weights as the verdict).
"""
import json
import pathlib
import time

import click
import numpy as np
import torch

from . import hires as H
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
        if __import__("os").environ.get("J5_T"):                 # tightening sweep: override the threshold, scales unchanged
            self.T = float(__import__("os").environ["J5_T"])
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


class Judge5(H.Judge):
    """Drop-in `judge` for hires.run_oracle (opt-in).  `cellmap` is called first with the grid decode g (the reference, it
    binds the image) and then with every candidate decode."""

    reference_relative = False
    frozen_cls = Frozen

    def __init__(self, dev, frozen_json):
        from types import SimpleNamespace

        from .metrics import Judge as M

        m = M(dev, dino=False)                       # LPIPS only: the codec picks leaves with `judge.j.lp`
        self.j = SimpleNamespace(lp=m.lp, feats=lambda x: torch.zeros(x.shape[0], 1, 32, 32, device=x.device), dev=dev)
        self.dev = dev
        self.fr = self.frozen_cls(frozen_json)
        self.v2, self.v3, self.win, self.floor = True, False, False, .6
        self.gm = self.s_orig = self.x01 = None
        self._key = None
        self.v5 = None
        self.last = None
        self.n_eval = 0
        self.sec = 0.0

    def contract(self):
        return dict(policy="v5", T=self.fr.T, grain_thr=self.fr.thr, frozen=self.fr.path, version=5)

    @torch.no_grad()
    def cellmap(self, xh, x, fx):
        key = (tuple(x.shape), round(float(x.sum()), 3), round(float(x[..., ::37, ::41].sum()), 3))
        if key != self._key:                          # first call for this image: xh is the grid decode g
            self._key = key
            self.v5 = V5(self.fr, quantise(x), quantise(xh))
            self.last = None
            return torch.zeros(1, G8, G8, device=xh.device)
        t0 = time.time()
        r = self.v5.evaluate(quantise(xh))
        self.sec += time.time() - t0
        self.n_eval += 1
        self.last = r
        d = torch.zeros(1, G8, G8, device=xh.device)
        d[0][torch.from_numpy(r["bad8"]).to(xh.device)] = BIG
        return d


# ---------------------------------------------------------------- replay -------------------------------------------------
def save_case(path, out, x_u8, g_u8, frozen_path, score, name=""):
    """Store what a replay needs: the layout, the codec table, the images and the verdict of record."""
    torch.save(dict(name=name, lvl=out["lvl"].cpu(), two=out["two"].cpu(), tab=_cpu(out["tab"]), xh=out["xh"].cpu(),
                    x_u8=x_u8, g_u8=g_u8, frozen=str(frozen_path), score=score, tokens=int(out["tokens"])), path)


def _cpu(o):
    if torch.is_tensor(o):
        return o.cpu()
    if isinstance(o, (list, tuple)):
        return type(o)(_cpu(i) for i in o)
    if isinstance(o, dict):
        return {k: _cpu(v) for k, v in o.items()}
    return o


def replay_case(path, png=None):
    """Decode from the saved layout and table in float32, save a PNG, re-read it, evaluate.  Returns (score, stored, dxh)."""
    from PIL import Image

    c = torch.load(path, weights_only=False)
    xh = H.Codec.decode(c["tab"], c["lvl"], c["two"])
    dxh = float((xh.cpu() - c["xh"]).abs().max())
    y = quantise(xh)
    png = png or str(pathlib.Path(path).with_suffix(".png"))
    Image.fromarray(y).save(png)
    y2 = np.asarray(Image.open(png).convert("RGB"), np.uint8)
    assert np.array_equal(y, y2)
    r = evaluate(c["x_u8"], c["g_u8"], y2, c["frozen"])
    return r["score"], c["score"], dxh


@click.group()
def cli():
    pass


@cli.command()
@click.argument("case")
def replay(case):
    s, stored, dxh = replay_case(case)
    ok = abs(s - stored) <= 1e-6 and dxh == 0
    print(f"replay {case}: score {s:.9f} stored {stored:.9f} |dscore| {abs(s - stored):.2e} decode max diff {dxh:.2e} -> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
