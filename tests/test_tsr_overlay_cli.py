import json

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from conftest import fake_tab, random_layout
from tessera import layout as LY, overlay, tsr
from tessera.cli import main
from tessera.stream.tokens import pack, pack_flux


def _sample(flux=False, seed=1):
    lvl, two = random_layout(seed, p_split=.7, max_level=6)
    if flux:
        two &= lvl != 5
        sel = torch.ones(32, 32, dtype=torch.bool)
        sel[::2] = False
        return lvl, two, sel, pack_flux(fake_tab(), lvl, two, torch.randn(1, 32, 64, 64), sel=sel)
    return lvl, two, None, pack(fake_tab(), lvl, two)


@pytest.mark.parametrize("flux", [False, True])
def test_tsr_roundtrip(tmp_path, flux):
    lvl, two, sel, tok = _sample(flux)
    p = tmp_path / "a.tsr"
    tsr.save(p, tsr.Tsr(tok, "vae16" if flux else "A", dict(tokens=float(tok.n_tokens))))
    t = tsr.load(p)
    assert torch.equal(t.tokens.bits, tok.bits) and t.tokens.fluxleaf == flux and t.arm == ("vae16" if flux else "A")
    assert torch.allclose(t.tokens.floats, tok.floats.half().float())          # floats are stored as fp16
    assert LY.tiles_from_tokens(t.tokens) == LY.tile_list(lvl, two, sel)


def test_tsr_rejects_foreign_file(tmp_path):
    torch.save(dict(format="retina_codec/1"), tmp_path / "x.tsr")
    with pytest.raises(ValueError):
        tsr.load(tmp_path / "x.tsr")


def test_overlay_draws_outlines_and_key():
    lvl, two, sel, tok = _sample(True)
    tiles = LY.tiles_from_tokens(tok)
    img = Image.fromarray(np.full((512, 512, 3), 128, np.uint8))
    o = overlay.draw(img, tiles)
    assert o.size == (512, 540)
    a = np.asarray(o)
    for k in {t[3] for t in tiles}:
        col = np.array(LY.COLOURS[k])
        assert (np.abs(a.astype(int) - col).sum(-1) == 0).any(), k            # every tile kind is drawn in its colour
    assert (a[512:] != 128).any() and overlay.draw(img, tiles, key=False).size == (512, 512)


def test_cli_overlay_info_and_decode_needs_weights(tmp_path):
    lvl, two, sel, tok = _sample(False)
    p = tmp_path / "t.tsr"
    tsr.save(p, tsr.Tsr(tok, "A", dict(tokens=1.0, verdict=dict(passed=True))))
    Image.fromarray(np.zeros((300, 200, 3), np.uint8)).save(tmp_path / "i.png")
    r = CliRunner()
    res = r.invoke(main, ["overlay", str(tmp_path / "i.png"), str(p), "-o", str(tmp_path / "o.png")])
    assert res.exit_code == 0, res.output
    assert Image.open(tmp_path / "o.png").size == (512, 540)
    res = r.invoke(main, ["info", str(p)])
    assert res.exit_code == 0 and json.loads(res.output)["arm"] == "A"
    res = r.invoke(main, ["decode", str(p), "-o", str(tmp_path / "d.png")], env={"TESSERA_WEIGHTS": ""})
    assert res.exit_code != 0 and not (tmp_path / "d.png").exists()          # no weights: fails, does not write garbage
