"""The all-grid fallback and the per-image choice among {arm A, arm vae16, all-grid}.

All-grid = every 16 px tile of the image is a FLUX.2 grid latent (1,024 payload tokens + the tree header; `H.count` says
1,026): the decode is the grid VAE's own decode of the latents.  It is a candidate for EVERY image, not only when nothing
else passes: the rule is the cheapest layout that passes, so on an image whose fixed-code layout passes at 2,952 tokens the
all-grid layout wins.  Judge it from the decode of the float16 latents, exactly what a `.tsr` stores (the in-memory float32
decode scores 0 by construction and would overstate it).

  grid_layout(dev)                     -> (lvl, two, sel) of the all-grid layout
  grid_tokens()                        -> its token count by the codec's own bill (H.count)
  grid_decode(flux, zt)                -> decode of zt rounded to float16, [1, 3, 512, 512] in [-1, 1]
  grid_row(judge, flux, x, g, zt)      -> dict(arm="grid", tokens, passed, score, argmax) judged by a v5 judge (JudgeFast / Judge5)
  choose(rows)                         -> the winning row: passing first, then fewer tokens, then A < vae16 < grid
"""
import torch

from . import hires as H
from . import vae16 as V

ORDER = {"A": 0, "vae16": 1, "grid": 2}


def grid_layout(dev="cpu"):
    lvl = torch.full((1, H.G, H.G), V.LEVEL16, dtype=torch.long, device=dev)
    two = torch.zeros_like(lvl, dtype=torch.bool)
    sel = torch.ones(V.N16, V.N16, dtype=torch.bool, device=dev)
    return lvl, two, sel


def grid_tokens():
    lvl, two, _ = grid_layout()
    return int(H.count(lvl, two))


@torch.no_grad()
def grid_decode(flux, zt):
    return flux.dec(zt.half().float()).float().clamp(-1, 1)


@torch.no_grad()
def grid_row(judge, flux, x, g, zt):
    """Judge the float16-latent decode.  `judge` must be a v5 judge that keeps its last verdict in `judge.last`."""
    y = grid_decode(flux, zt)
    judge.cellmap(g, x, None)             # binds / re-reads the reference decode of this image (cached after the arms)
    judge.cellmap(y, x, None)
    r = judge.last
    return dict(arm="grid", tokens=grid_tokens(), passed=bool(r["passed"]), score=float(r["score"]), argmax=r.get("argmax"))


def choose(rows):
    """rows: dicts with arm in ORDER, tokens, passed.  Cheapest passing wins; if none passes, the cheapest overall."""
    rows = [r for r in rows if r is not None]
    if not rows:
        raise ValueError("no candidate layouts")
    return min(rows, key=lambda r: (not bool(r["passed"]), r["tokens"], ORDER[r["arm"]]))
