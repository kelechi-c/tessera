"""The pure-latent router: predicts a whole layout (size_map, 16-256 px FLUX.2 pyramid tiles) in one pass.

Replaces the fixed-code router of `tessera.router` / `tessera.core.route` for the pure-latent codec (plan 10.5, S7).
Nothing here is trained; the network, its features, the layout record it learns from and the verify loop are defined.

  feats = pyramid_features(lat_pyr, x)                 # [B, NF, 64, 64], needs no TileUp (spec "pyr-res-v1")
  rungs = router.predict(feats)                        # [B, 64, 64] largest tile each cell allows (0..4)
  out   = encode_layout(router, feats, check_fn)       # project + verify against the judge; never fails the judge

`check_fn(size_map) -> dict(passed: bool, bad8: [64, 64] bool or absent)` decodes the layout (TileUp + FLUX.2
decoder) and runs the judge of record; it is injected so this module needs neither.  Verification splits failing tiles
for at most `rounds` rounds (the last round drops them straight to 16 px) and falls back to the all-grid layout, which
reproduces the grid VAE's own latent and costs 1,024 tokens.

Labels: `LayoutRecord2` lines written by the search (`merge_quadtree_pure` + judge), one per image, tagged with the
TileUp checkpoint that produced them: a layout is only right for the decoder that drew it.
"""
import dataclasses
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..router import NotTrainedError
from . import layout as LY

FORMAT = "tessera/layout-record/2"
CKPT_FORMAT = "tessera/latent-router/1"


# ---------------------------------------------------------------- features ----------------------------------------------
def _cell_mean(x, cell=8):
    return F.avg_pool2d(x, cell)


def pixel_stats(x):
    """[B, 3, 512, 512] in [-1, 1] -> [B, 4, 64, 64]: log fine-scale std (detail energy) and mean colour per 8 px cell."""
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
    hf = (x - blur).pow(2).mean(1, keepdim=True)
    return torch.cat((torch.log(_cell_mean(hf).sqrt() + 1e-3), _cell_mean(x)), 1)


def pyramid_residuals(lat_pyr, channel_std=None):
    """{s: [B, 32, 128/s, 128/s]} -> [B, 4, 64, 64]: per cell, log mean |z512 - bilinear token canvas of level s| for
    s = 32, 64, 128, 256 px.  How much a tile of that size would have to invent; free (no decode).

    Pass global dataset `channel_std` [32] when available; otherwise clamps per-image std at 0.25 so plain/flat
    images with near-zero channel variance do not amplify tiny numerical noise."""
    z = lat_pyr[LY.LEAF].float()
    if channel_std is not None:
        sd = torch.as_tensor(channel_std, device=z.device, dtype=torch.float32).clamp_min(1e-3).view(1, LY.CH, 1, 1)
    else:
        sd = z.flatten(2).std(2).clamp_min(0.25)[:, :, None, None]
    out = []
    for s in LY.SIZES[1:]:
        base = LY.level_canvas(lat_pyr[s].float(), s, "bilinear")
        out.append(torch.log(((z - base).abs() / sd).mean(1, keepdim=True) + 1e-3))
    return torch.cat(out, 1)


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    name: str
    nf: int


PYR_RES_V1 = FeatureSpec("pyr-res-v1", 8)          # 4 pyramid residuals + 4 pixel stats
PYR_RES_TILEUP_V1 = FeatureSpec("pyr-res-tileup-v1", 12)  # + 4 judge excess maps of uniform TileUp layouts
SPECS = {s.name: s for s in (PYR_RES_V1, PYR_RES_TILEUP_V1)}


def pyramid_features(lat_pyr, x, excess_fn=None, channel_std=None):
    """Router input [B, NF, 64, 64].  Without `excess_fn`: spec pyr-res-v1 (no TileUp needed).  With
    `excess_fn(size_map) -> [B, 64, 64]` (judge excess per cell of the decode of a uniform layout; needs a trained
    TileUp): spec pyr-res-tileup-v1, adding the excess at uniform 32, 64, 128 and 256 px tiles."""
    f = [pyramid_residuals(lat_pyr, channel_std=channel_std), pixel_stats(x.float())]
    if excess_fn is not None:
        b = x.shape[0]
        f.append(torch.stack([excess_fn(torch.full((b, LY.SIDE, LY.SIDE), s, dtype=torch.long, device=x.device)).float()
                              for s in LY.SIZES[1:]], 1))
    return torch.cat(f, 1)


# ---------------------------------------------------------------- network -----------------------------------------------
class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d),
                               nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d))

    def forward(self, x):
        return x + self.b(x)


class LatentRouter(nn.Module):
    """Dilated FCN over per-cell features -> NR - 1 = 4 ordinal logits per cell, logit P(rung > k): 'this cell allows a
    tile larger than SIZES[k]'.  Receptive field ~60 cells, so each cell sees most of the image."""

    def __init__(self, spec=PYR_RES_V1, c=64, dil=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.spec = spec
        self.register_buffer("mu", torch.zeros(1, spec.nf, 1, 1))
        self.register_buffer("sd", torch.ones(1, spec.nf, 1, 1))
        self.inp = nn.Conv2d(spec.nf, c, 3, padding=1)
        self.body = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, LY.NR - 1, 1))
        self.cfg = dict(c=c, dil=list(dil))
        self.tileup_id = ""

    def forward(self, f):
        if f.shape[1] != self.spec.nf:
            raise ValueError(f"router expects {self.spec.nf} feature channels ({self.spec.name}), got {f.shape[1]}")
        return self.out(self.body(self.inp((f - self.mu) / self.sd)))

    @torch.no_grad()
    def fit_stats(self, feats):
        """Set the input normalisation from a [N, NF, 64, 64] feature sample."""
        self.mu.copy_(feats.mean((0, 2, 3), keepdim=True))
        self.sd.copy_(feats.std((0, 2, 3), keepdim=True).clamp_min(1e-4))

    @torch.no_grad()
    def predict(self, f, tau=.5):
        """[B, 64, 64] rung per cell: the number of logits above tau (monotone by construction).
        `tau` may be a scalar float or a length-4 sequence of per-rung thresholds."""
        probs = torch.sigmoid(self(f))
        if isinstance(tau, (list, tuple)):
            thr = torch.tensor(tau, device=probs.device, dtype=probs.dtype).view(1, LY.NR - 1, 1, 1)
        else:
            thr = float(tau)
        return (probs > thr).sum(1)

    def save(self, path, tileup_id=""):
        torch.save(dict(format=CKPT_FORMAT, model=self.state_dict(), spec=self.spec.name, nf=self.spec.nf, cfg=self.cfg,
                        tileup_id=tileup_id or self.tileup_id), path)

    @classmethod
    def load(cls, path, spec=None, tileup_id=None, device="cpu"):
        """Load a trained router.  NotTrainedError if the file is missing or not a latent-router checkpoint; ValueError
        if its feature spec or TileUp id differs from the one asked for (labels go stale when TileUp changes)."""
        p = pathlib.Path(path)
        if not p.is_file():
            raise NotTrainedError(f"no trained latent router at {p}: the router is not trained yet; use the search")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or ck.get("format") != CKPT_FORMAT:
            raise NotTrainedError(f"{p} is not a latent-router checkpoint")
        if ck["spec"] not in SPECS or SPECS[ck["spec"]].nf != ck["nf"]:
            raise ValueError(f"{p}: unknown feature spec {ck['spec']} / nf {ck['nf']}")
        if spec is not None and ck["spec"] != spec.name:
            raise ValueError(f"{p} was trained on features {ck['spec']}, not {spec.name}")
        if tileup_id is not None and ck.get("tileup_id", "") != tileup_id:
            raise ValueError(f"{p} was trained for TileUp {ck.get('tileup_id')!r}, not {tileup_id!r}")
        m = cls(SPECS[ck["spec"]], c=ck["cfg"]["c"], dil=tuple(ck["cfg"]["dil"]))
        m.load_state_dict(ck["model"])
        m.tileup_id = ck.get("tileup_id", "")
        return m.to(device).eval()


def router_loss(logits, size_map, weight=None):
    """Ordinal BCE: target_k = (rung > k) per cell.  `weight` [B] (e.g. 0 for records whose verdict failed)."""
    r = LY.rungs_of(size_map)
    k = torch.arange(LY.NR - 1, device=logits.device)[None, :, None, None]
    target = (r[:, None] > k).float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()


# ---------------------------------------------------------------- inference ---------------------------------------------
@torch.no_grad()
def verify(size_map, check_fn, rounds=3, blocked=None):
    """Refine a layout until the judge passes.  Each round splits the tiles that touch bad cells one level; the last
    round drops them straight to 16 px.  If it still fails, or the judge gives no bad-cell map, fall back to the
    all-grid layout.  Returns dict(size_map [1,64,64], passed, rounds, tokens, fallback, verdict)."""
    sm = size_map.clone()
    if blocked is not None:
        sm = torch.where(blocked.to(sm.device).bool().reshape(sm.shape), torch.full_like(sm, LY.LEAF), sm)
        sm = LY.project(LY.rungs_of(sm))        # re-validate: a blocked cell inside a big tile splits it
    for n in range(1, rounds + 2):
        ev = check_fn(sm)
        if ev["passed"]:
            return dict(size_map=sm, passed=True, rounds=n, tokens=int(LY.tokens_of(sm)[0]), fallback=False, verdict=ev)
        bad = ev.get("bad8")
        if bad is None or n > rounds or bool((sm == LY.LEAF).all()):
            break
        bad = bad.reshape(1, LY.SIDE, LY.SIDE).to(sm.device).bool()
        new = LY.split_bad(sm, bad, to_leaf=(n == rounds))
        if torch.equal(new, sm):               # bad cells only on 16 px leaves: blame the neighbouring tiles too
            wide = F.max_pool2d(bad[:, None].float(), 3, 1, 1)[:, 0] > 0
            new = LY.split_bad(sm, wide, to_leaf=(n == rounds))
        if torch.equal(new, sm):               # nothing left to split near the failure
            break
        sm = new
    grid = LY.all_grid(1, sm.device)
    ev = check_fn(grid)
    return dict(size_map=grid, passed=bool(ev["passed"]), rounds=rounds + 1, tokens=LY.grid_tokens(), fallback=True,
                verdict=ev)


@torch.no_grad()
def encode_layout(router, feats, check_fn, tau=.5, rounds=3, blocked=None):
    """One image: router -> project -> verify.  feats [1, NF, 64, 64]."""
    sm = LY.project(router.predict(feats, tau), blocked=None if blocked is None else blocked.reshape(1, LY.SIDE, LY.SIDE))
    return verify(sm, check_fn, rounds=rounds, blocked=blocked)


# ---------------------------------------------------------------- training data -----------------------------------------
@dataclasses.dataclass
class LayoutRecord2:
    """One router label: the pure-latent search's layout for an image and the judge's verdict on its decode, tagged
    with the TileUp checkpoint that drew it.  size_map is [64, 64] (tile side in cells, values in SIZES)."""
    name: str
    size_map: np.ndarray
    tokens: int
    verdict: dict
    tileup_id: str
    image_md5: str = ""

    def to_json(self):
        r = LY.rungs_of(torch.from_numpy(self.size_map.astype(np.int64))[None])[0].numpy().astype(np.uint8)
        return json.dumps(dict(format=FORMAT, name=self.name, image_md5=self.image_md5, tileup_id=self.tileup_id,
                               tokens=int(self.tokens), rungs="".join(map(str, r.ravel().tolist())),
                               verdict={k: v for k, v in self.verdict.items() if isinstance(v, (int, float, str, bool))}))

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} record (got {d.get('format')!r})")
        r = np.frombuffer(d["rungs"].encode(), np.uint8) - ord("0")
        sm = LY.sizes_of(torch.from_numpy(r.astype(np.int64)).reshape(1, LY.SIDE, LY.SIDE))[0].numpy()
        return cls(d["name"], sm, d["tokens"], d["verdict"], d["tileup_id"], d.get("image_md5", ""))


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(rec.to_json() + "\n")


def read_records(path, tileup_id=None):
    """All v2 records in a file (v1 lines from the fixed-code router are skipped); with `tileup_id`, only those."""
    out = []
    for q in pathlib.Path(path).read_text().splitlines():
        if not q.strip() or f'"{FORMAT}"' not in q:
            continue
        rec = LayoutRecord2.from_json(q)
        if tileup_id is None or rec.tileup_id == tileup_id:
            out.append(rec)
    return out
