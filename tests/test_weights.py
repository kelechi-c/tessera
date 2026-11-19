"""Tests that need `bases.pt` (set TESSERA_WEIGHTS=DIR).  Skipped otherwise.  FLUX.2 / GPU paths are not covered here."""
import os

import numpy as np
import pytest
import torch

from tessera import layout as LY, tsr
from tessera.search import decode_tsr, load_bases, to_u8, to_x
from tessera.stream.codec import LeafCodec
from tessera.stream.tokens import pack, unpack

pytestmark = pytest.mark.skipif(not os.environ.get("TESSERA_WEIGHTS"), reason="TESSERA_WEIGHTS not set")


def test_fixed_code_stream_roundtrip(tmp_path):
    """prepare -> pack -> .tsr -> decode_tsr reproduces the table's own decode to fp16 precision (arm A, CPU)."""
    from conftest import random_layout
    from tessera.stream.codec import paste

    yy, xx = np.mgrid[:512, :512]
    im = np.stack([(xx > 200) * 200, (yy > 300) * 120, np.full((512, 512), 60)], -1).astype(np.uint8)
    x = to_x(im, "cpu")
    leaf = LeafCodec(load_bases(), "cpu")
    tab = leaf.prepare(x)
    lvl, two = random_layout(5, p_split=.5, max_level=5)
    tok = pack(tab, lvl, two)
    want = paste(tab, lvl, two)
    got = unpack(tok, leaf)
    assert (got - want).abs().max() < 5e-2
    p = tmp_path / "a.tsr"
    tsr.save(p, tsr.Tsr(tok, "A"))
    y = decode_tsr(tsr.load(p), device="cpu")
    assert y.shape == (512, 512, 3) and np.abs(y.astype(int) - to_u8(want).astype(int)).mean() < 2
    assert LY.tiles_from_tokens(tsr.load(p).tokens) == LY.tile_list(lvl, two)
