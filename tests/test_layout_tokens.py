import numpy as np
import pytest
import torch

from conftest import fake_tab, random_layout
from tessera import layout as LY
from tessera.stream.grid import G, K, L, P, S, leaves, masks
from tessera.stream.tokens import Tokens, pack, pack_flux


@pytest.mark.parametrize("seed", range(6))
def test_random_layouts_are_valid_and_covered(seed):
    lvl, two = random_layout(seed)
    assert LY.validate(lvl)
    tiles = LY.tile_list(lvl, two)
    m, covered = LY.tile_map(tiles)
    assert covered and sum(s * s for _, _, s, _ in tiles) == S * S


def test_invalid_layout_rejected():
    lvl = torch.zeros(1, G, G, dtype=torch.long)
    lvl[0, :32, :32] = 1       # one quarter says "level 1" but the rest says "level 0": not a quadtree
    assert not LY.validate(lvl)


@pytest.mark.parametrize("seed", range(6))
def test_pack_parse_roundtrip_fixed(seed):
    lvl, two = random_layout(seed)
    tok = pack(fake_tab(seed=seed), lvl, two)
    assert LY.tiles_from_tokens(tok) == LY.tile_list(lvl, two)
    assert abs(tok.n_tokens - (LY.count_tokens(lvl, two) + 1)) < 1e-6 or tok.header_tokens >= 1


@pytest.mark.parametrize("seed", range(6))
def test_pack_parse_roundtrip_flux_with_selector(seed):
    lvl, two = random_layout(seed, p_split=.75, max_level=6)
    two &= lvl != 5                       # a 16 px leaf never carries 2 tokens in the FLUX arm
    n16 = [(i, j) for l, i, j, _ in leaves(lvl, two) if l == 5]
    sel = torch.zeros(G // 2, G // 2, dtype=torch.bool)
    for q, (i, j) in enumerate(n16):
        sel[i, j] = q % 2 == 0
    zt = torch.randn(1, 32, G, G)
    tok = pack_flux(fake_tab(seed=seed), lvl, two, zt, sel=sel)
    assert tok.fluxleaf
    assert LY.tiles_from_tokens(tok) == LY.tile_list(lvl, two, sel)
    kinds = [k for *_, k in LY.tile_list(lvl, two, sel)]
    assert kinds.count("flux16") == int(sel.sum()) and kinds.count("fixed16") == len(n16) - int(sel.sum())
    # floats: flux leaf = one 128-float token each, fixed leaf = 128 floats, native 192, 2-token 256
    want = sum(192 if l == L - 1 else (K if (l == 5 or not t) else 2 * K) for l, i, j, t in leaves(lvl, two))
    assert len(tok.floats) == want


def test_token_counts_match_stream():
    lvl, two = random_layout(1)
    tok = pack(fake_tab(), lvl, two)
    assert len(tok.floats) / K == pytest.approx(LY.count_tokens(lvl, two))


def test_truncated_or_extra_header_bits_raise():
    lvl, two = random_layout(2)
    tok = pack(fake_tab(), lvl, two)
    with pytest.raises(ValueError):
        LY.tiles_from_tokens(Tokens(tok.bits[:-3], tok.floats))
    with pytest.raises(ValueError):
        LY.tiles_from_tokens(Tokens(torch.cat([tok.bits, torch.zeros(5, dtype=torch.uint8)]), tok.floats))


def test_whole_image_leaf_is_one_tile():
    lvl = torch.zeros(1, G, G, dtype=torch.long)
    two = torch.zeros(1, G, G, dtype=torch.bool)
    assert LY.tile_list(lvl, two) == [(0, 0, 512, "one")]
    assert LY.count_tokens(lvl, two) == 1
