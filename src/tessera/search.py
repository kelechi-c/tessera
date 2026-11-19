"""Encoding: pick a layout (search or router), choose the cheaper passing arm, write real tokens.

Pipeline for one 512 px image x:
  zt = FLUX.2 latents of x, g = FLUX.2 decode(zt)                (the reference the judge compares against)
  arm A      layout search over fixed codes (core.hires.run_oracle), judged by judge v5 + mae32
  arm vae16  layout search whose 16 px leaves may carry the image's own FLUX.2 latent (core.vae16.run_oracle_vae16)
  choice     the cheaper arm that passes (a passing arm beats a failing one); an exact pixel-art raster ("lowres") replaces
             arm A when it is cheaper
  tokens     the chosen layout is packed into a real stream (stream.tokens.pack / pack_flux) with the stream codec's own
             coefficients, decoded back, and judged again: that final verdict, on the decode a user will actually get, is
             what the file records (`meta.verdict`)
The router path (layout="router") replaces the search by `router.layout_for` and then the same pack/decode/verdict steps.

Needs weights: `bases.pt` (PCA bases, ~550 MB), the FLUX.2 VAE (diffusers; black-forest-labs/FLUX.2-dev) and the judge's
LPIPS/DISTS weights.  Options below are the ones of the judge-v5 runs (eps_w .3, eps_img .1, eps_loc .8, polish 150, v2 judge).
"""
import dataclasses
import pathlib
import time

import numpy as np
import torch
from PIL import Image

from .layout import tile_list
from .router import NotTrainedError, Router, layout_for  # noqa: F401
from .stream.grid import S
from .tsr import Tsr

OPTS = dict(eps_w=.3, eps_img=.1, eps_loc=.8, polish=150, search_frac=.9, grain=True)


def to_x(im, dev):
    im = im.convert("RGB") if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im, np.uint8))
    return torch.from_numpy(np.asarray(im.resize((S, S), Image.LANCZOS)).copy()).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1


def to_u8(y):
    return ((y[0].detach().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def weights_dir(weights=None):
    import os

    w = weights or os.environ.get("TESSERA_WEIGHTS")
    if not w:
        raise FileNotFoundError("weights directory not given: pass --weights DIR or set TESSERA_WEIGHTS (DIR/bases.pt, optional DIR/router.pt)")
    return pathlib.Path(w)


def load_bases(weights=None):
    p = weights_dir(weights)
    p = p / "bases.pt" if p.is_dir() else p
    if not p.is_file():
        raise FileNotFoundError(f"bases not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


@dataclasses.dataclass
class Encoded:
    tsr: Tsr
    y: np.ndarray            # uint8 decode from the tokens
    g: np.ndarray            # uint8 FLUX.2 round trip (the judge's reference)
    verdict: dict
    info: dict


class Encoder:
    def __init__(self, weights=None, device=None, layout="search", router=None, allow_untrained=False):
        from .core import hires as H
        from .core import judge5_fast as Fa
        from .core import judge5_sweep  # noqa: F401  (mae32 gate)
        from .judge import frozen_path
        from .stream.codec import LeafCodec
        from .stream.fluxleaf import Flux2VAE

        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layout = layout
        self.router = None
        if layout == "router":
            rp = pathlib.Path(router) if router else (weights_dir(weights) / "router.pt" if (weights or __import__("os").environ.get("TESSERA_WEIGHTS")) else None)
            try:
                if rp is None:
                    raise NotTrainedError("no router path given: the router is not trained yet; use --layout search")
                self.router = Router.load(rp, self.dev)
            except NotTrainedError:
                if not allow_untrained:
                    raise
                self.router = Router.untrained(self.dev)
        elif layout != "search":
            raise ValueError(f"layout must be 'search' or 'router', not {layout!r}")
        ck = load_bases(weights)
        self.H, self.frozen = H, frozen_path()
        self.judge = Fa.JudgeFast(self.dev, self.frozen)
        self.codec = H.Codec(ck, self.dev, "edge", None, self.judge.j.lp, True, v2=True)   # search-side codec
        self.leaf = LeafCodec(ck, self.dev)                                               # token-stream codec
        self.flux = Flux2VAE(self.dev)

    @torch.no_grad()
    def encode(self, im, arm="both", opts=None):
        from .core import judge5_fast as Fa
        from .core import judge5_search as Sr
        from .core import vae16 as V
        from .core import route as R
        from .core import allgrid as GR
        from .stream.tokens import pack, pack_flux, unpack

        H, dev, o = self.H, self.dev, {**OPTS, **(opts or {})}
        t0 = time.time()
        x = to_x(im, dev)
        zt = self.flux.enc(x)
        g = self.flux.dec(zt).float().clamp(-1, 1)
        mw = round(o["eps_w"] * (1 - o["search_frac"]), 6)
        mi = round(o["eps_img"] * (1 - o["search_frac"]), 6)
        cand = {}
        if self.layout == "router":
            ctx = R.Ctx(self.codec, self.judge, x, g, o["grain"], o["eps_w"], o["eps_img"],
                        vae16=dict(flux=self.flux, zt=zt, placeholder="latent"), eps_loc=o["eps_loc"])
            lvl, two, xh, ok, n = layout_for(self.router, ctx)
            cand["vae16"] = dict(lvl=lvl, two=two, sel=ctx.sel, passed=ok, tokens=int(H.count(lvl, two)), y=xh)
        else:
            if arm in ("both", "A"):
                a = H.run_oracle(self.codec, self.judge, x, g, o["eps_w"], o["eps_img"], o["polish"], grain=o["grain"],
                                 eps_loc=o["eps_loc"], v2=True)
                cand["A"] = a
            if arm in ("both", "vae16"):
                cand["vae16"] = V.run_oracle_vae16(self.codec, self.judge, x, g, zt, self.flux, o["eps_w"], o["eps_img"],
                                                   polish=o["polish"], search="crop", iters=2, max_iters=4, tol=2e-3,
                                                   placeholder="latent", repair="cheap", repair_rounds=8, repair_max_leaves=12,
                                                   boundary_max_delta=2, grain=o["grain"], margin_w=mw, margin_img=mi,
                                                   frozen_repairs=True, force_native_after=2, eps_loc=o["eps_loc"])
        if arm in ("both", "grid") or self.layout == "router":     # all-grid: every 16 px tile a grid latent, judged from fp16 latents
            yg = GR.grid_decode(self.flux, zt)
            ev = Fa.evaluate(to_u8(x), to_u8(g), to_u8(yg), self.frozen, dev=dev)
            lg, tg, sg = GR.grid_layout(dev)
            cand["grid"] = dict(lvl=lg, two=tg, sel=sg, passed=bool(ev["passed"]), tokens=GR.grid_tokens(), y=yg)
        if "A" in cand and cand["A"].get("lowres_replaced"):
            enc = cand["A"]["lowres"]
            y = H.Codec.decode_lowres(enc, S, dev)
            verdict = Fa.evaluate(to_u8(x), to_u8(g), to_u8(y), self.frozen, dev=dev)
            meta = dict(arm="lowres", tokens=int(cand["A"]["lowres_tokens"]), verdict=_plain(verdict), seconds=time.time() - t0)
            return Encoded(Tsr(None, "lowres", meta, lowres=enc), to_u8(y), to_u8(g), verdict, meta)
        # cheapest passing layout among {A, vae16, grid} (not "grid only if nothing passes")
        arm_ch = choose_arm([dict(arm=k, tokens=v["tokens"], passed=v["passed"]) for k, v in cand.items()])["arm"]
        c = cand[arm_ch]
        lvl, two = c["lvl"], c["two"]
        tab = None if arm_ch == "grid" else self.leaf.prepare(x)      # an all-grid stream has no fixed-code leaf
        tok = pack(tab, lvl, two) if arm_ch == "A" else pack_flux(tab, lvl, two, zt, sel=c["sel"])
        y = unpack(tok, self.leaf, vae=self.flux if tok.fluxleaf else None)
        verdict = Fa.evaluate(to_u8(x), to_u8(g), to_u8(y), self.frozen, dev=dev)
        meta = dict(arm=arm_ch, tokens=float(tok.n_tokens), search_tokens=int(c["tokens"]), search_passed=bool(c["passed"]),
                    verdict=_plain(verdict), layout=self.layout, seconds=time.time() - t0,
                    tokens_by_arm={k: int(v["tokens"]) for k, v in cand.items()})
        self.last = dict(lvl=lvl.cpu(), two=two.cpu(), sel=None if arm_ch == "A" else c["sel"].cpu(), x=to_u8(x), arm=arm_ch)
        return Encoded(Tsr(tok, arm_ch, meta), to_u8(y), to_u8(g), verdict, meta)


def choose_arm(rows):
    """Cheapest passing layout among the candidate arms (core.allgrid.choose): passing first, then fewer tokens, then A < vae16 < grid."""
    from .core.allgrid import choose

    return choose(rows)


def _plain(v):
    return {k: (float(a) if isinstance(a, (float, np.floating)) else a) for k, a in v.items() if isinstance(a, (int, float, str, bool, np.floating, np.bool_))}


def decode_tsr(t: Tsr, weights=None, device=None):
    """Tsr -> uint8 [512, 512, 3].  Fixed-code files need `bases.pt` only; FLUX-leaf files also need the FLUX.2 VAE."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if t.lowres is not None:
        from .core import lowres as LR

        return LR.decode(t.lowres, t.side)
    from .stream.codec import LeafCodec
    from .stream.tokens import unpack

    leaf = LeafCodec(load_bases(weights), dev)
    vae = None
    if t.tokens.fluxleaf:
        from .stream.fluxleaf import Flux2VAE

        vae = Flux2VAE(dev)
    return to_u8(unpack(t.tokens, leaf, vae=vae))
