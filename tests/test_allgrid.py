"""All-grid fallback: the per-image rule (cheapest passing of A / vae16 / grid) and the all-grid .tsr round trip (CPU, stub VAE)."""
import random
import types

import pytest
import torch
import torch.nn.functional as F

from tessera import layout as LY, tsr
from tessera.core import allgrid as GR
from tessera.search import choose_arm
from tessera.stream.tokens import pack_flux, unpack


class FakeVAE:
    """enc/dec stand-in with a clamp like the real wrapper; dec reads the first 3 channels, nearest-upsampled to 8 px cells."""

    def enc(self, x):
        z = torch.zeros(x.shape[0], 32, 64, 64)
        z[:, :3] = F.avg_pool2d(x, 8)
        return z

    def dec(self, z):
        return F.interpolate(z[:, :3], scale_factor=8, mode="nearest").clamp(-1, 1)


def row(arm, tokens, passed):
    return dict(arm=arm, tokens=tokens, passed=passed)


def test_grid_wins_when_arm_a_passes_above_1024():
    # the dog: A passes at 2,952, vae16 fails at 981, all-grid passes at ~1,026 -> grid, although A "passes"
    w = choose_arm([row("A", 2952, True), row("vae16", 981, False), row("grid", GR.grid_tokens(), True)])
    assert w["arm"] == "grid"


def test_grid_does_not_beat_a_cheaper_passing_layout():
    assert choose_arm([row("A", 300, True), row("vae16", 403, True), row("grid", 1026, True)])["arm"] == "A"
    assert choose_arm([row("A", 1200, True), row("vae16", 900, True), row("grid", 1026, True)])["arm"] == "vae16"


def test_passing_beats_cheaper_failing_and_ties_go_to_a():
    assert choose_arm([row("A", 3000, True), row("vae16", 100, False), row("grid", 1026, False)])["arm"] == "A"
    assert choose_arm([row("A", 500, True), row("vae16", 500, True), row("grid", 500, True)])["arm"] == "A"
    assert choose_arm([row("A", 500, False), row("vae16", 900, False)])["arm"] == "A"            # nothing passes: cheapest


def test_without_grid_the_old_rule_is_unchanged():
    def old(a, b):
        if not a["passed"] and b["passed"]:
            return "vae16"
        if a["passed"] and not b["passed"]:
            return "A"
        return "A" if a["tokens"] <= b["tokens"] else "vae16"

    rng = random.Random(0)
    for _ in range(500):
        a, b = row("A", rng.randint(1, 3000), rng.random() < .6), row("vae16", rng.randint(1, 3000), rng.random() < .6)
        assert choose_arm([a, b])["arm"] == old(a, b)


def test_grid_layout_and_token_bill():
    lvl, two, sel = GR.grid_layout()
    assert LY.validate(lvl) and bool(sel.all()) and not bool(two.any())
    assert GR.grid_tokens() == 1026          # 1,024 payload tokens + the 2 header tokens of H.count


def test_all_grid_tsr_roundtrip_equals_grid_decode_within_fp16(tmp_path):
    vae = FakeVAE()
    torch.manual_seed(0)
    zt = torch.randn(1, 32, 64, 64)
    lvl, two, sel = GR.grid_layout()
    tok = pack_flux(None, lvl, two, zt, sel=sel)                      # no fixed-code table needed: every leaf is a latent
    assert tok.fluxleaf and len(tok.floats) == 1024 * 128
    assert LY.tiles_from_tokens(tok) == LY.tile_list(lvl, two, sel) and all(t[3] == "flux16" for t in LY.tiles_from_tokens(tok))
    p = tmp_path / "g.tsr"
    tsr.save(p, tsr.Tsr(tok, "grid", dict(tokens=float(tok.n_tokens))))
    back = tsr.load(p)
    y = unpack(back.tokens, types.SimpleNamespace(dev="cpu"), vae=vae)
    assert y.shape == (1, 3, 512, 512)
    # exactly the decode of the fp16 latents (what the file stores) ...
    assert torch.equal(y, GR.grid_decode(vae, zt))
    # ... which differs from the float32 decode only by fp16 rounding
    assert float((y - vae.dec(zt)).abs().max()) < 3e-3
    assert abs(float(back.tokens.n_tokens) - GR.grid_tokens()) <= 1       # the stream's own bill agrees with the codec's


def test_grid_row_is_judged_from_the_fp16_decode():
    vae = FakeVAE()
    zt = torch.randn(1, 32, 64, 64)
    seen = []

    class StubJudge:
        last = None

        def cellmap(self, xh, x, fx):
            seen.append(xh.clone())
            self.last = dict(passed=True, score=0.045, argmax="color_mean256")
            return torch.zeros(1, 64, 64)

    g = vae.dec(zt)
    r = GR.grid_row(StubJudge(), vae, torch.zeros(1, 3, 512, 512), g, zt)
    assert r == dict(arm="grid", tokens=1026, passed=True, score=0.045, argmax="color_mean256")
    assert torch.equal(seen[-1], GR.grid_decode(vae, zt)) and not torch.equal(seen[-1], g)      # the candidate is the fp16 decode, not g
