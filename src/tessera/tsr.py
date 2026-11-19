"""The `.tsr` file: one image as quadtree tokens.

A torch pickle (torch.save) holding: format "tessera/1", side (512), the header bits (bit-packed), the leaves' floats (fp16),
`fluxleaf` (stream carries FLUX.2 latent leaves; decoding needs the VAE), `arm` ("A" fixed codes, "vae16" FLUX 16 px leaves,
"lowres" exact pixel-art raster) and `meta` (encoder verdict, token count, judge T; informational, not needed to decode).
A "lowres" file has no bits: its payload is the stored raster dict.  Load with `torch.load(weights_only=False)` semantics, so
only open files you trust."""
import dataclasses

import numpy as np
import torch

from . import FORMAT
from .stream.grid import S
from .stream.tokens import Tokens


@dataclasses.dataclass
class Tsr:
    tokens: Tokens | None
    arm: str = "A"
    meta: dict = dataclasses.field(default_factory=dict)
    lowres: dict | None = None
    side: int = S

    def n_tokens(self):
        if self.lowres is not None:
            return float(self.meta.get("tokens", 0))
        return float(self.tokens.n_tokens)


def save(path, t: Tsr):
    d = dict(format=FORMAT, side=t.side, arm=t.arm, meta=t.meta, lowres=t.lowres)
    if t.tokens is not None:
        d.update(bits=torch.from_numpy(np.packbits(t.tokens.bits.cpu().numpy())), n_bits=len(t.tokens.bits),
                 floats=t.tokens.floats.half().cpu(), fluxleaf=bool(t.tokens.fluxleaf))
    torch.save(d, path)


def load(path) -> Tsr:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file (format={d.get('format')!r})")
    tok = None
    if "bits" in d:
        bits = torch.from_numpy(np.unpackbits(d["bits"].numpy())[: d["n_bits"]].copy())
        tok = Tokens(bits, d["floats"].float(), d["side"], fluxleaf=bool(d["fluxleaf"]))
    return Tsr(tok, d["arm"], d.get("meta", {}), d.get("lowres"), d["side"])
