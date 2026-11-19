import numpy as np
import pytest
import torch

from test_latent_layout import rand_pyramid, rand_size_map
from tessera.latent import layout as LY
from tessera.latent import router as LR
from tessera.router import NotTrainedError


def _feats(b=1, seed=0, excess=False):
    pyr = rand_pyramid(seed, b=b)
    x = torch.rand(b, 3, 512, 512, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    fn = (lambda sm: sm.float() / 32) if excess else None
    return LR.pyramid_features(pyr, x, fn)


def test_features_shapes_and_specs():
    f = _feats(2)
    assert f.shape == (2, LR.PYR_RES_V1.nf, 64, 64) and torch.isfinite(f).all()
    f2 = _feats(1, excess=True)
    assert f2.shape == (1, LR.PYR_RES_TILEUP_V1.nf, 64, 64)


def test_residual_feature_is_zero_where_the_level_explains_the_latent():
    """If the 512 px latent is exactly the bilinear canvas of a level, that level's residual is at the floor."""
    pyr = rand_pyramid(1, b=1)
    pyr[2] = LY.level_canvas(pyr[8], 8, "bilinear")
    r = LR.pyramid_residuals(pyr)
    assert torch.allclose(r[:, 1], torch.full_like(r[:, 1], np.log(1e-3)), atol=1e-3)   # channel 1 = s 8 (64 px)
    assert bool((r[:, 3] > r[:, 1] + 1).all())                                          # 256 px explains less


def test_router_shapes_monotone_rungs_and_wrong_nf():
    m = LR.LatentRouter()
    f = _feats(2)
    assert m(f).shape == (2, LY.NR - 1, 64, 64)
    r = m.predict(f)
    assert r.shape == (2, 64, 64) and int(r.min()) >= 0 and int(r.max()) <= LY.NR - 1
    assert LY.validate(LY.project(r))
    with pytest.raises(ValueError, match="feature channels"):
        m(torch.zeros(1, 5, 64, 64))


def test_loss_targets_and_a_tiny_overfit():
    """The router can fit one layout: after a few hundred steps the projected prediction equals the label."""
    torch.manual_seed(0)
    f = _feats(1, seed=2)
    sm = rand_size_map(5)[None]
    m = LR.LatentRouter(c=32, dil=(1, 2, 4, 8))
    m.fit_stats(f)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    with torch.no_grad():
        l0 = float(LR.router_loss(m(f), sm))
    for _ in range(250):
        opt.zero_grad()
        loss = LR.router_loss(m(f), sm)
        loss.backward()
        opt.step()
    assert loss.item() < .2 * l0
    assert torch.equal(LY.project(m.predict(f)), sm)
    w = torch.tensor([0.])
    with torch.no_grad():
        assert float(LR.router_loss(m(f), sm, weight=w)) == 0.


def test_checkpoint_roundtrip_and_refusals(tmp_path):
    with pytest.raises(NotTrainedError, match="not trained"):
        LR.LatentRouter.load(tmp_path / "none.pt")
    torch.save({"model": {}}, tmp_path / "old.pt")
    with pytest.raises(NotTrainedError):
        LR.LatentRouter.load(tmp_path / "old.pt")
    m = LR.LatentRouter(c=32, dil=(1, 2))
    m.save(tmp_path / "r.pt", tileup_id="tileup-s5-abc")
    m2 = LR.LatentRouter.load(tmp_path / "r.pt", spec=LR.PYR_RES_V1, tileup_id="tileup-s5-abc")
    f = _feats(1)
    assert torch.equal(m(f), m2(f)) and m2.tileup_id == "tileup-s5-abc"
    with pytest.raises(ValueError, match="TileUp"):
        LR.LatentRouter.load(tmp_path / "r.pt", tileup_id="tileup-s4-old")
    with pytest.raises(ValueError, match="features"):
        LR.LatentRouter.load(tmp_path / "r.pt", spec=LR.PYR_RES_TILEUP_V1)


class FakeJudge:
    """Fails every tile larger than `limit` cells inside `hot` (bad cells = those tiles); everything else passes."""

    def __init__(self, hot, limit=LY.LEAF, give_bad=True):
        self.hot, self.limit, self.give_bad, self.calls = hot, limit, give_bad, 0

    def __call__(self, sm):
        self.calls += 1
        bad = (sm[0] > self.limit) & self.hot
        out = dict(passed=not bool(bad.any()))
        if self.give_bad:
            out["bad8"] = bad
        return out


def test_verify_passes_straight_away_when_the_layout_is_good():
    j = FakeJudge(torch.zeros(64, 64, dtype=torch.bool))
    out = LR.verify(torch.full((1, 64, 64), 32), j)
    assert out["passed"] and not out["fallback"] and out["tokens"] == 4 and j.calls == 1


def test_verify_refines_then_drops_to_leaves_on_the_last_round():
    hot = torch.zeros(64, 64, dtype=torch.bool)
    hot[3, 3] = True                                   # one cell needs 16 px
    j = FakeJudge(hot)
    out = LR.verify(torch.full((1, 64, 64), 32), j, rounds=3)
    assert out["passed"] and not out["fallback"]
    assert int(out["size_map"][0, 3, 3]) == LY.LEAF and LY.validate(out["size_map"])
    # three single-level splits would stop at 32 px; the last round goes straight to 16 px
    assert j.calls == 4 and int(out["size_map"][0, 40, 40]) == 32


def test_verify_falls_back_to_the_grid():
    j = FakeJudge(torch.ones(64, 64, dtype=torch.bool), give_bad=False)    # no bad map -> fallback at once
    out = LR.verify(torch.full((1, 64, 64), 32), j)
    assert out["fallback"] and out["passed"] and out["tokens"] == 1024 and j.calls == 2

    class Never:
        def __call__(self, sm):
            return dict(passed=bool((sm == LY.LEAF).all()), bad8=torch.ones(64, 64, dtype=torch.bool))
    out = LR.verify(torch.full((1, 64, 64), 8), Never(), rounds=2)
    # every cell bad: the last round drops every tile to 16 px, which is the grid and passes without the fallback
    assert out["passed"] and not out["fallback"] and out["tokens"] == 1024

    class Broken:                                    # nothing passes, not even the grid: reported, never hidden
        def __call__(self, sm):
            return dict(passed=False, bad8=torch.ones(64, 64, dtype=torch.bool))
    out = LR.verify(torch.full((1, 64, 64), 32), Broken(), rounds=3)
    assert out["fallback"] and not out["passed"] and out["tokens"] == 1024


def test_verify_blames_the_neighbour_when_the_bad_cell_is_a_leaf():
    """A seam: the judge flags 16 px leaf cells next to a large tile.  Splitting the neighbour fixes it; the layout must
    not jump to the 1,024-token grid."""
    sm = torch.full((1, 64, 64), 32)
    sm[0, :32, :32] = LY.LEAF                            # top-left quadrant already at 16 px

    class Seam:
        def __init__(self):
            self.calls = 0

        def __call__(self, s):
            self.calls += 1
            bad = torch.zeros(64, 64, dtype=torch.bool)
            if int(s[0, 10, 32]) > 8:                    # the tile right of column 31 is too big
                bad[10, 31] = True                       # ... but the judge marks the leaf side of the seam
            return dict(passed=not bool(bad.any()), bad8=bad)

    out = LR.verify(sm, Seam(), rounds=3)
    # 256 leaves in the top-left quadrant, the top-right 256 px tile split to 3 x 128 px + 4 x 64 px, 2 untouched roots
    assert out["passed"] and not out["fallback"] and out["tokens"] == 256 + 7 + 2
    assert int(out["size_map"][0, 10, 32]) <= 8 and int(out["size_map"][0, 60, 60]) == 32


def test_verify_respects_blocked_cells():
    j = FakeJudge(torch.zeros(64, 64, dtype=torch.bool))
    blocked = torch.zeros(64, 64, dtype=torch.bool)
    blocked[50:52, 10:20] = True
    out = LR.verify(torch.full((1, 64, 64), 32), j, blocked=blocked)
    assert LY.validate(out["size_map"]) and bool((out["size_map"][0][blocked] == LY.LEAF).all())


def test_encode_layout_end_to_end_never_fails_the_judge():
    m = LR.LatentRouter(c=32, dil=(1, 2))
    hot = torch.zeros(64, 64, dtype=torch.bool)
    hot[20:30, 20:30] = True
    out = LR.encode_layout(m, _feats(1), FakeJudge(hot))
    assert out["passed"] and LY.validate(out["size_map"])
    assert out["tokens"] == int(LY.tokens_of(out["size_map"]))


def test_layout_record_v2_roundtrip_and_v1_lines_skipped(tmp_path):
    sm = rand_size_map(9).numpy()
    rec = LR.LayoutRecord2("a.png", sm, int(LY.tokens_of(torch.from_numpy(sm)[None])), dict(passed=True, score=1.1), "tu-1")
    p = tmp_path / "r.jsonl"
    p.write_text('{"format": "tessera/layout-record/1", "name": "old"}\n')
    LR.append_record(p, rec)
    LR.append_record(p, LR.LayoutRecord2("b.png", sm, 7, dict(passed=False), "tu-2"))
    back = LR.read_records(p)
    assert [r.name for r in back] == ["a.png", "b.png"]
    assert np.array_equal(back[0].size_map, sm) and back[0].verdict["passed"] is True and back[0].tileup_id == "tu-1"
    assert [r.name for r in LR.read_records(p, tileup_id="tu-2")] == ["b.png"]
    with pytest.raises(ValueError):
        LR.LayoutRecord2.from_json('{"format": "tessera/layout-record/1"}')


def test_pyramid_residuals_global_std_and_per_rung_tau():
    """Global channel_std avoids flat-image variance blowup, and predict() accepts per-rung tuple thresholds."""
    pyr = {s: torch.zeros(1, LY.CH, 128 // s, 128 // s) for s in LY.SIZES}
    pyr[2] = pyr[2] + 1e-4 * torch.randn_like(pyr[2])  # nearly flat image
    r_global = LR.pyramid_residuals(pyr, channel_std=torch.ones(LY.CH))
    r_clamp = LR.pyramid_residuals(pyr)
    assert float(r_global.max()) < -6.0 and float(r_clamp.max()) < -6.0  # stays near log(1e-3) = -6.9
    m = LR.LatentRouter(c=16, dil=(1, 2))
    f = _feats(1)
    r_lo = m.predict(f, tau=(0.2, 0.2, 0.2, 0.2))
    r_hi = m.predict(f, tau=(0.8, 0.8, 0.8, 0.8))
    assert bool((r_lo >= r_hi).all())
