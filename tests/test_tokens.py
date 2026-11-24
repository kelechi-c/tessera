"""Unit tests for the versioned `TSR2` binary token format, k-d tree bitstream, validation, and token-only decode."""
import pathlib
import random
import struct

import numpy as np
import pytest
import torch

from tessera.latent import rect as RL
from tessera.latent import rect_tiledown as RTD
from tessera.rect.model import TileUp
from tessera.rect.search import IN_CH_RECT, predict_rect_canvas_td
from tessera.rect.tokens import (
    DTYPE_FLOAT32,
    FORMAT_VERSION,
    GRID_SIDE,
    HEADER_SIZE,
    MAGIC,
    ORDER_RASTER_TOP_LEFT,
    TOKEN_DIM,
    CheckpointMismatchError,
    RectTokenFile,
    TokenFormatError,
    _HEADER,
    decode_kdtree_bits,
    decode_tokens_to_latent,
    dumps_tokens,
    encode_kdtree_bits,
    extract_leaf_tokens,
    file_sha256,
    kdtree_leaves_to_leaf_order,
    leaf_order_to_kdtree_leaves,
    load_tokens,
    loads_tokens,
    save_tokens,
    tokens_to_rect_canvas,
    verify_checkpoint_sha256,
)


def _random_kdtree_shape_map(seed: int, p_split: float = 0.65) -> torch.Tensor:
    rng = random.Random(seed)
    sm = torch.zeros((2, GRID_SIDE, GRID_SIDE), dtype=torch.long)

    def walk(r0, c0, sh, sw):
        acts = RL.valid_actions(sh, sw)
        if len(acts) == 1 or rng.random() > p_split:
            sm[0, r0:r0 + sh, c0:c0 + sw] = sh
            sm[1, r0:r0 + sh, c0:c0 + sw] = sw
            return
        act = rng.choice(acts[1:])
        if act == RL.ACT_SPLIT_H:
            walk(r0, c0, sh // 2, sw)
            walk(r0 + sh // 2, c0, sh // 2, sw)
        else:
            walk(r0, c0, sh, sw // 2)
            walk(r0, c0 + sw // 2, sh, sw // 2)

    for r0, c0 in ((0, 0), (0, 32), (32, 0), (32, 32)):
        walk(r0, c0, 32, 32)
    assert RL.validate_rect(sm)
    return sm


def _make_sample_token_file(sm: torch.Tensor, seed: int = 7) -> RectTokenFile:
    leaves = tuple(RL.leaves_rect(sm))
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn((len(leaves), TOKEN_DIM), generator=g, dtype=torch.float32)
    return RectTokenFile(
        shape_map=sm,
        leaves=leaves,
        tokens=tokens,
        ckpt_sha256="ab" * 32,
        seed=0,
        tu_spike_k=3.0,
        tu_spike_min_area=64,
        meta={"ckpt_name": "step_20000.pt", "mode": "router", "passed": True, "calls": 2},
    )


def test_kdtree_bitstream_roundtrip_min_max_and_random():
    # Minimal: four 32x32 roots (4 tokens)
    sm_min = torch.full((2, GRID_SIDE, GRID_SIDE), 32, dtype=torch.long)
    b_min, n_min = encode_kdtree_bits(sm_min)
    assert n_min == 8 and len(b_min) == 1
    assert torch.equal(decode_kdtree_bits(b_min, n_min), sm_min)

    # Maximal: all 2x2 leaves (1,024 tokens)
    sm_max = RL.all_grid_rect(1)[0]
    b_max, n_max = encode_kdtree_bits(sm_max)
    assert torch.equal(decode_kdtree_bits(b_max, n_max), sm_max)
    assert len(b_max) <= 512

    # Mixed random 2:1 k-d trees covering all 13 shapes
    seen_shapes = set()
    for s in range(20):
        sm = _random_kdtree_shape_map(seed=100 + s, p_split=0.72)
        seen_shapes.update((sh, sw) for _, _, sh, sw in RL.leaves_rect(sm))
        packed, n_bits = encode_kdtree_bits(sm)
        assert torch.equal(decode_kdtree_bits(packed, n_bits), sm)
    assert seen_shapes == set(RL.SHAPES)


def test_binary_roundtrip_lossless_and_exact_size(tmp_path):
    sm = _random_kdtree_shape_map(seed=42, p_split=0.7)
    tf = _make_sample_token_file(sm)
    raw = dumps_tokens(tf)

    # Verify exact expected size formula
    tree_bytes, _ = encode_kdtree_bits(sm)
    unpacked = _HEADER.unpack_from(raw, 0)
    meta_n_bytes = unpacked[13]
    expected_size = HEADER_SIZE + meta_n_bytes + len(tree_bytes) + tf.n_tokens * TOKEN_DIM * 4
    assert len(raw) == expected_size

    out_path = save_tokens(tmp_path / "sample.tsr", tf)
    loaded = load_tokens(out_path)
    assert loaded.version == FORMAT_VERSION
    assert loaded.dtype_code == DTYPE_FLOAT32
    assert loaded.order_code == ORDER_RASTER_TOP_LEFT
    assert torch.equal(loaded.shape_map, tf.shape_map)
    assert loaded.leaves == tf.leaves
    assert torch.equal(loaded.tokens, tf.tokens)  # lossless float32 bit-exact
    assert loaded.ckpt_sha256 == tf.ckpt_sha256
    assert loaded.seed == tf.seed
    assert loaded.tu_spike_k == pytest.approx(tf.tu_spike_k)
    assert loaded.tu_spike_min_area == tf.tu_spike_min_area
    assert loaded.meta == tf.meta


def test_leaf_order_kdtree_mapping_roundtrip():
    sm = _random_kdtree_shape_map(seed=19, p_split=0.75)
    tf = _make_sample_token_file(sm)
    kd_leaves, kd_toks = leaf_order_to_kdtree_leaves(sm, tf.tokens)
    expected_kd = tuple((r0, c0, sh, sw) for r0, c0, sh, sw, _, a in RL.kdtree_trace(sm) if a == RL.ACT_LEAF)
    assert kd_leaves == expected_kd
    raster_leaves, back_toks = kdtree_leaves_to_leaf_order(sm, kd_toks)
    assert raster_leaves == tf.leaves
    assert torch.equal(back_toks, tf.tokens)


def test_malformed_inputs_rejected(tmp_path):
    sm = _random_kdtree_shape_map(seed=5, p_split=0.6)
    tf = _make_sample_token_file(sm)
    valid = dumps_tokens(tf)
    fields = list(_HEADER.unpack_from(valid, 0))
    body = valid[HEADER_SIZE:]

    # Truncated header
    with pytest.raises(TokenFormatError, match="Truncated"):
        loads_tokens(valid[:10])

    # Bad magic
    bad = list(fields)
    bad[0] = b"TSR1"
    with pytest.raises(TokenFormatError, match="magic"):
        loads_tokens(_HEADER.pack(*bad) + body)

    # Unsupported version
    bad = list(fields)
    bad[1] = 2
    with pytest.raises(TokenFormatError, match="version"):
        loads_tokens(_HEADER.pack(*bad) + body)

    # Non-zero reserved flags
    bad = list(fields)
    bad[2] = 1
    with pytest.raises(TokenFormatError, match="flags"):
        loads_tokens(_HEADER.pack(*bad) + body)

    # Bad grid_side / token_dim / dtype / order
    for idx, val, pat in ((3, 32, "grid_side"), (4, 64, "token_dim"), (5, 1, "dtype"), (6, 1, "order")):
        bad = list(fields)
        bad[idx] = val
        with pytest.raises(TokenFormatError, match=pat):
            loads_tokens(_HEADER.pack(*bad) + body)

    # Invalid n_tokens
    bad = list(fields)
    bad[7] = 3
    with pytest.raises(TokenFormatError, match="n_tokens"):
        loads_tokens(_HEADER.pack(*bad) + body)

    # Truncated payload & trailing garbage
    with pytest.raises(TokenFormatError, match="size mismatch"):
        loads_tokens(valid[:-1])
    with pytest.raises(TokenFormatError, match="size mismatch"):
        loads_tokens(valid + b"\x00")

    # Corrupted metadata JSON (not a dict)
    meta_n = fields[13]
    bad_meta = b"[]" + b" " * (meta_n - 2)
    with pytest.raises(TokenFormatError, match="Metadata JSON"):
        loads_tokens(valid[:HEADER_SIZE] + bad_meta + valid[HEADER_SIZE + meta_n:])

    # NaN in token payload
    mut = bytearray(valid)
    mut[-4:] = struct.pack("<f", float("nan"))
    with pytest.raises(TokenFormatError, match="NaN or Inf"):
        loads_tokens(bytes(mut))

    # Non-zero trailing padding bits in k-d bitstream
    tree_n_bits = fields[8]
    tree_n_bytes = fields[9]
    if tree_n_bits % 8 != 0:
        mut2 = bytearray(valid)
        mut2[HEADER_SIZE + meta_n + tree_n_bytes - 1] |= 1
        with pytest.raises(TokenFormatError, match="padding"):
            loads_tokens(bytes(mut2))

    # Invalid square action 0b11 in bitstream
    with pytest.raises(TokenFormatError, match="0b11"):
        decode_kdtree_bits(bytes([0b11000000]), 8)

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_tokens(tmp_path / "missing.tsr")


def test_checkpoint_sha256_verification(tmp_path):
    ckpt_a = tmp_path / "ckpt_a.pt"
    ckpt_b = tmp_path / "ckpt_b.pt"
    ckpt_a.write_bytes(b"checkpoint-a-bytes")
    ckpt_b.write_bytes(b"checkpoint-b-bytes")

    sm = _random_kdtree_shape_map(seed=3, p_split=0.5)
    leaves = tuple(RL.leaves_rect(sm))
    tf = RectTokenFile(
        shape_map=sm,
        leaves=leaves,
        tokens=torch.zeros((len(leaves), TOKEN_DIM), dtype=torch.float32),
        ckpt_sha256=file_sha256(ckpt_a),
    )
    assert verify_checkpoint_sha256(tf, ckpt_a) == file_sha256(ckpt_a)
    with pytest.raises(CheckpointMismatchError, match="mismatch"):
        verify_checkpoint_sha256(tf, ckpt_b)


def test_token_only_decode_matches_in_memory_reference_without_source_z():
    """Persisted leaf tokens alone must reproduce `predict_rect_canvas_td` without passing `z512`."""
    torch.manual_seed(123)
    z_fit = torch.randn(4, RL.CH, RL.SIDE, RL.SIDE)
    basis = RL.fit_rect_basis(z_fit)
    mean = torch.randn(RL.CH) * 0.2
    std = torch.rand(RL.CH) * 0.5 + 0.5
    mv, sv = mean.view(1, RL.CH, 1, 1), std.view(1, RL.CH, 1, 1)

    tu = TileUp(c=(32, 48, 64), in_ch=IN_CH_RECT, attn_blocks=1).eval()
    # Non-zero output conv so residual, zero-mean, spike clamp, and ac_gate are all exercised
    with torch.no_grad():
        tu.out[-1].weight.normal_(0, 0.05)
        tu.out[-1].bias.normal_(0, 0.02)
    tu._zero_mean_resid = True
    tu._ac_floor, tu._ac_tau = 0.26, 0.20
    tu._spike_k, tu._spike_min_area = 3.0, 64

    td = RTD.RectTileDown(c=32, hid=64, use_1d_pool=True, prof_c=8, prof_len=16, dc_lock=True).eval()
    td.ac_floor, td.ac_tau = 0.26, 0.20
    tl = RTD.RectTileLift(hid=64, rank=4, prof_len=16).eval()
    with torch.no_grad():
        for p in tl.parameters():
            p.normal_(0, 0.02)
    object.__setattr__(tu, "_tilelift", tl)

    z1 = torch.randn(1, RL.CH, RL.SIDE, RL.SIDE)
    zn = (z1 - mv) / sv
    rect_pyr_n = RTD.encode_rect_pyramid_td(zn, basis, tiledown=td)
    sm = _random_kdtree_shape_map(seed=77, p_split=0.72)

    # In-memory reference path (uses z1 and full 13-shape pyramid)
    with torch.no_grad():
        ref_canvas, ref_mask, ref_sf, ref_pos, ref_gate = RTD.build_rect_canvas_tl(
            rect_pyr_n, sm[None], basis, tilelift=tl, floor=0.26, tau=0.20
        )
    z_ref = predict_rect_canvas_td(tu, z1, rect_pyr_n, sm[None], basis, mean, std, seed=0)

    # Persisted token-only path (serialize -> deserialize -> decode with NO z1 and NO full pyramid)
    leaves, leaf_tokens = extract_leaf_tokens(rect_pyr_n, sm)
    raw = dumps_tokens(
        RectTokenFile(
            shape_map=sm,
            leaves=leaves,
            tokens=leaf_tokens,
            ckpt_sha256="12" * 32,
            seed=0,
            tu_spike_k=3.0,
            tu_spike_min_area=64,
        )
    )
    tf = loads_tokens(raw)

    tok_canvas, tok_mask, tok_sf, tok_pos, tok_gate = tokens_to_rect_canvas(
        tf.shape_map, tf.tokens, basis, tilelift=tl, floor=0.26, tau=0.20
    )
    assert torch.equal(tok_mask, ref_mask)
    assert torch.equal(tok_sf, ref_sf)
    assert torch.equal(tok_pos, ref_pos)
    assert float((tok_canvas - ref_canvas).abs().max()) == 0.0
    assert float((tok_gate - ref_gate).abs().max()) == 0.0

    z_tok = decode_tokens_to_latent(
        tf.shape_map, tf.tokens, tu, basis, mean, std,
        seed=tf.seed, tu_spike_k=tf.tu_spike_k, tu_spike_min_area=tf.tu_spike_min_area,
    )
    # Coarse tiles must be bit-identical; (2, 2) tiles differ only by float32 ((z - m)/s)*s + m (~1e-7)
    coarse_diff = float(((z_tok - z_ref) * ref_mask.float()).abs().max())
    all_diff = float((z_tok - z_ref).abs().max())
    assert coarse_diff == 0.0
    assert all_diff < 2e-6
