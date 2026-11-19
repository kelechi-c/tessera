import pathlib
import random
import sys

import pytest
import torch

from tessera.latent import layout as LY


def rand_size_map(seed=0, p_split=.6, sizes=LY.SIZES):
    """A random valid pure-latent layout [64, 64]."""
    rng = random.Random(seed)
    sm = torch.zeros(LY.SIDE, LY.SIDE, dtype=torch.long)

    def walk(r0, c0, s):
        if s == LY.LEAF or (s in sizes and rng.random() > p_split):
            sm[r0:r0 + s, c0:c0 + s] = s
            return
        h = s // 2
        for dr in (0, h):
            for dc in (0, h):
                walk(r0 + dr, c0 + dc, h)

    for r0 in (0, 32):
        for c0 in (0, 32):
            walk(r0, c0, 32)
    return sm


def rand_pyramid(seed=0, b=None):
    g = torch.Generator().manual_seed(seed)
    shape = lambda s: ((b,) if b else ()) + LY.level_shape(s)
    return {s: torch.randn(shape(s), generator=g) for s in LY.SIZES}


@pytest.mark.parametrize("seed", range(6))
def test_random_layouts_are_valid_and_tree_roundtrips(seed):
    sm = rand_size_map(seed, p_split=.3 + .1 * seed)
    assert LY.validate(sm[None])
    bits, nodes = LY.tree_bits(sm)
    assert torch.equal(LY.from_tree_bits(bits), sm)
    assert len(LY.leaves(sm)) == int(LY.tokens_of(sm[None])[0])
    assert all(s > LY.LEAF for _, _, s in nodes)


def test_grid_and_coarsest_layouts():
    g = LY.all_grid(2)
    assert LY.validate(g) and LY.tokens_of(g).tolist() == [1024, 1024]
    bits, _ = LY.tree_bits(g[0])
    assert len(bits) == 4 * (1 + 4 + 16 + 64) and all(bits)
    big = torch.full((1, 64, 64), 32)
    assert LY.tokens_of(big).item() == 4 and LY.tree_bits(big[0])[0] == [0, 0, 0, 0]


def test_invalid_layouts_are_rejected():
    sm = rand_size_map(1)[None].clone()
    bad = sm.clone()
    bad[0, 0, 0] = 3
    assert not LY.validate(bad)
    half = torch.full((1, 64, 64), 2)
    half[0, :32, :16] = 32                     # a 256 px label over only half a 256 px block
    assert not LY.validate(half)
    with pytest.raises(ValueError):
        LY.from_tree_bits([1, 0])


def test_project_respects_every_cell_and_blocked_cells():
    g = torch.Generator().manual_seed(3)
    r = torch.randint(0, LY.NR, (2, 64, 64), generator=g)
    blocked = torch.zeros(2, 64, 64, dtype=torch.bool)
    blocked[:, 10:14, 40:50] = True
    sm = LY.project(r, blocked)
    assert LY.validate(sm)
    assert bool((sm <= LY.sizes_of(r)).all())               # no tile larger than any of its cells allows
    assert bool((sm[blocked] == LY.LEAF).all())
    # projecting a valid layout's own rungs gives the layout back
    v = rand_size_map(4)[None]
    assert torch.equal(LY.project(LY.rungs_of(v)), v)


def test_split_bad_splits_only_touching_tiles():
    sm = torch.full((1, 64, 64), 32)
    bad = torch.zeros(1, 64, 64, dtype=torch.bool)
    bad[0, 5, 5] = True
    one = LY.split_bad(sm, bad)
    assert LY.validate(one) and int(LY.tokens_of(one)) == 3 + 4
    leaf = LY.split_bad(sm, bad, to_leaf=True)
    assert LY.validate(leaf) and int(LY.tokens_of(leaf)) == 3 + 256
    assert bool((leaf[0, 32:, :] == 32).all())


@pytest.mark.parametrize("seed", range(3))
def test_pack_unpack_roundtrip(seed):
    sm = rand_size_map(seed)
    pyr = rand_pyramid(seed)
    tok = LY.pack_tokens(pyr, sm)
    assert tok.shape == (int(LY.tokens_of(sm[None])), LY.TOK)
    back = LY.unpack_tokens(tok, LY.leaves(sm))
    assert torch.equal(LY.pack_tokens(back, sm), tok)


def test_tree_walker_matches_tree_bits():
    sm = rand_size_map(7)
    bits, nodes = LY.tree_bits(sm)
    w = LY.TreeWalker()
    seen = []
    for b in bits:
        seen.append(w.node)
        w.push(b)
    assert w.done and seen == nodes and w.leaves == LY.leaves(sm) and torch.equal(w.size_map, sm)


# ---------------------------------------------------------------- the cross-check with qtok ------------------------------
def _qtok():
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from qtok import tileup
    except Exception as e:  # qtok lives beside tessera in the retina checkout only
        pytest.skip(f"qtok not importable: {e}")
    return tileup


@pytest.mark.parametrize("seed", range(4))
def test_token_packing_matches_qtok_canvas(seed):
    """Scatter packed tokens back into a pyramid: qtok's TileUp canvas built from it must equal the canvas built
    from the original pyramid, in both nearest and bilinear modes.  This pins the token order to qtok's."""
    T = _qtok()
    sm = rand_size_map(seed)
    pyr = rand_pyramid(seed, b=1)
    tok = LY.pack_tokens({s: v[0] for s, v in pyr.items()}, sm)
    back = {s: v[None] for s, v in LY.unpack_tokens(tok, LY.leaves(sm)).items()}
    for mode in ("nearest", "bilinear"):
        a = T.build_pyramid_canvas(pyr, sm[None], mode=mode)[0]
        b = T.build_pyramid_canvas(back, sm[None], mode=mode)[0]
        assert torch.equal(a, b), mode


@pytest.mark.parametrize("s", LY.SIZES[1:])
def test_level_canvas_matches_qtok(s):
    T = _qtok()
    z = rand_pyramid(s, b=2)[s]
    for mode in ("nearest", "bilinear"):
        assert torch.allclose(LY.level_canvas(z, s, mode), T._pyramid_blocks(z, s, mode=mode))


def test_counts_and_validity_match_qtok_quadtrees():
    T = _qtok()
    sm = T.draw_quadtree_mask(8, gen=torch.Generator().manual_seed(0))
    assert LY.validate(sm)
    for i in range(8):
        assert len(LY.leaves(sm[i])) == int(LY.tokens_of(sm[i:i + 1]))
        assert torch.equal(LY.from_tree_bits(LY.tree_bits(sm[i])[0]), sm[i])
