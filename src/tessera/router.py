"""The router: predicts a whole layout in one pass instead of searching for it.  NOT TRAINED in this package.

Interface
  Router.load(path)            -> trained router from a `router.pt` (dict model/vae16/nr/nf); NotTrainedError if the file is
                                  missing or not a router checkpoint
  Router.untrained()           -> random-initialised network, emits a UserWarning; its layouts are noise (smoke tests only)
  router.predict(features)     -> per-cell rungs [1, 64, 64]; `layout_for(...)` -> (lvl, two) after `project`
  layout_for(router, ctx)      -> router layout verified by the same judge as the search (ctx = core.route.Ctx)

Model: dilated FCN over per-cell features (NF = 16 channels: per-rung cell excess (12), reference score, log fine-scale std,
grain mask, mean colour -> see core.route.Ctx.features), 8 residual blocks (dilations 1 2 4 8 1 2 4 8), NR - 1 = 12 ordinal
logits per cell; rung = number of logits above tau.  `project` turns rungs into a valid quadtree layout.

Training data it will be distilled from: one `LayoutRecord` per image = the layout the search found + the judge's verdict on
it (docs/router_training_format.md).  The target is `rungs_of(lvl, two, vae16=True)`; no training code is shipped.
"""
import dataclasses
import json
import pathlib
import warnings

import numpy as np
import torch

from .core import route as R


class NotTrainedError(RuntimeError):
    """Raised when a router layout is requested but no trained router is available."""


class Router(R.Router):
    @classmethod
    def load(cls, path, device="cpu"):
        p = pathlib.Path(path)
        if not p.is_file():
            raise NotTrainedError(f"no trained router at {p}: the router is not trained yet; use --layout search")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "model" not in ck:
            raise NotTrainedError(f"{p} is not a router checkpoint (expected a dict with a 'model' state dict)")
        if ck.get("nr", R.NR) != R.NR or ck.get("nf", R.NF) != R.NF:
            raise NotTrainedError(f"{p}: router shape nr={ck.get('nr')} nf={ck.get('nf')} does not match NR={R.NR} NF={R.NF}")
        m = cls()
        m.load_state_dict(ck["model"])
        m.vae16 = bool(ck.get("vae16", True))
        return m.to(device).eval()

    @classmethod
    def untrained(cls, device="cpu"):
        warnings.warn("tessera Router is randomly initialised (untrained): its layouts are noise", UserWarning, stacklevel=2)
        m = cls()
        m.vae16 = True
        return m.to(device).eval()

    @torch.no_grad()
    def predict(self, features, tau=.5):
        return self.rungs(features, tau)


@torch.no_grad()
def layout_for(router, ctx, tau=.5):
    """features -> rungs -> projected layout -> the judge's verify loop.  Returns (lvl, two, xh, passed, evals)."""
    f = ctx.features()
    r = router.predict(f, tau)
    lvl, two = R.project(r, vae16=getattr(router, "vae16", True) and ctx.use_vae16)
    return R.verify(ctx, lvl, two)


# ---------------------------------------------------------------- training data format ---------------------------------
@dataclasses.dataclass
class LayoutRecord:
    """One training example for the router: the search's layout and the judge's verdict on its decode.
    lvl / two are [64, 64] arrays (one entry per 8 px cell), sel [32, 32] bool or None (no FLUX leaves)."""
    name: str
    lvl: np.ndarray
    two: np.ndarray
    sel: np.ndarray | None
    tokens: float
    arm: str
    verdict: dict
    image_md5: str = ""

    def rungs(self):
        return R.rungs_of(torch.from_numpy(self.lvl.astype(np.int64))[None], torch.from_numpy(self.two.astype(bool))[None],
                          vae16=self.sel is not None)[0].numpy()

    def to_json(self):
        pk = lambda a: None if a is None else dict(shape=list(a.shape), hex=np.packbits(np.asarray(a, np.uint8).ravel()).tobytes().hex())
        return json.dumps(dict(format="tessera/layout-record/1", name=self.name, image_md5=self.image_md5, arm=self.arm,
                               tokens=self.tokens, lvl=self.lvl.astype(np.uint8).ravel().tolist(), two=pk(self.two), sel=pk(self.sel),
                               verdict={k: v for k, v in self.verdict.items() if isinstance(v, (int, float, str, bool))}))

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        if d.get("format") != "tessera/layout-record/1":
            raise ValueError("not a tessera layout record")

        def unpk(p):
            if p is None:
                return None
            n = int(np.prod(p["shape"]))
            return np.unpackbits(np.frombuffer(bytes.fromhex(p["hex"]), np.uint8))[:n].reshape(p["shape"]).astype(bool)

        return cls(d["name"], np.array(d["lvl"], np.uint8).reshape(64, 64), unpk(d["two"]), unpk(d["sel"]), d["tokens"], d["arm"],
                   d["verdict"], d.get("image_md5", ""))


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(rec.to_json() + "\n")


def read_records(path):
    return [LayoutRecord.from_json(q) for q in pathlib.Path(path).read_text().splitlines() if q.strip()]
