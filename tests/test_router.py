import warnings

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from conftest import random_layout
from tessera import layout as LY
from tessera.cli import main
from tessera.core import route as R
from tessera.router import LayoutRecord, NotTrainedError, Router, append_record, read_records


def test_missing_router_raises_not_trained(tmp_path):
    with pytest.raises(NotTrainedError, match="not trained"):
        Router.load(tmp_path / "router.pt")


def test_foreign_checkpoint_raises(tmp_path):
    torch.save({"weights": 1}, tmp_path / "r.pt")
    with pytest.raises(NotTrainedError):
        Router.load(tmp_path / "r.pt")


def test_untrained_router_warns_and_has_right_shape():
    with pytest.warns(UserWarning, match="untrained"):
        m = Router.untrained()
    f = torch.randn(1, R.NF, 64, 64)
    assert m.predict(f).shape == (1, 64, 64) and m(f).shape == (1, R.NR - 1, 64, 64)


def test_trained_checkpoint_roundtrip(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Router.untrained()
    torch.save(dict(model=m.state_dict(), vae16=True, nr=R.NR, nf=R.NF), tmp_path / "router.pt")
    m2 = Router.load(tmp_path / "router.pt")
    f = torch.randn(1, R.NF, 64, 64)
    assert torch.equal(m(f), m2(f))


@pytest.mark.parametrize("vae16", [False, True])
def test_project_makes_valid_layouts(vae16):
    g = torch.Generator().manual_seed(0)
    r = torch.randint(0, R.NR, (1, 64, 64), generator=g)
    lvl, two = R.project(r, vae16=vae16)
    assert LY.validate(lvl)
    if vae16:
        assert not bool(two[lvl == 5].any())


def test_cli_router_without_weights_is_a_clean_error(tmp_path):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "i.png")
    res = CliRunner().invoke(main, ["encode", str(tmp_path / "i.png"), "-o", str(tmp_path / "o.tsr"), "--layout", "router",
                                    "--router", str(tmp_path / "none.pt")])
    assert res.exit_code != 0 and "not trained" in res.output and not (tmp_path / "o.tsr").exists()


def test_layout_record_roundtrip_and_targets(tmp_path):
    lvl, two = random_layout(4, p_split=.7, max_level=6)
    sel = np.zeros((32, 32), bool)
    sel[::3, ::2] = True
    rec = LayoutRecord("a.png", lvl[0].numpy().astype(np.uint8), two[0].numpy(), sel, 321.5, "vae16", dict(score=1.2, passed=True, T=1.78))
    p = tmp_path / "r.jsonl"
    append_record(p, rec)
    append_record(p, rec)
    back = read_records(p)
    assert len(back) == 2 and np.array_equal(back[0].lvl, rec.lvl) and np.array_equal(back[0].two, rec.two)
    assert np.array_equal(back[0].sel, sel) and back[0].tokens == 321.5 and back[0].verdict["passed"] is True
    r = back[0].rungs()
    assert r.shape == (64, 64) and r.max() <= 2 * 5            # vae16 ladder: 16 px is terminal (rung 10)
