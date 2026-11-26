"""CPU oracles and standalone codec tests for native-resolution FLUX.2."""
import errno
import json
import mmap
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from diffusers import AutoencoderKLFlux2
from PIL import Image
from torch import nn
from torch.nn import functional as F

from tessera.latent import rect as RL
from tessera.rect import (
    FORMAT_VERSION,
    FORMAT_VERSION_GRID,
    CheckpointMismatchError,
    ExactTiledFluxVAE,
    RectGridTokenFile,
    RectTokenFile,
    TokenFormatError,
    decode_exact_tiled,
    decode_grid_tokens_to_latent,
    decode_window_with_global_gn,
    dumps_grid_tokens,
    encode_exact_tiled,
    encode_kdtree_bits,
    file_sha256,
    load_any_tokens,
    load_grid_tokens,
    loads_grid_tokens,
    save_grid_tokens,
    save_tokens,
    verify_checkpoint_sha256,
)
from tessera.rect import encode as E
from tessera.rect import tiled_vae as V
from tessera.rect import tokens as T
from test_cli_and_hub import _install_fast_mocks, _make_tiny_ckpt


@pytest.fixture(scope="module")
def vae():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        return AutoencoderKLFlux2(block_out_channels=(64, 64, 64, 64), norm_num_groups=32).eval()


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _latent(h, w):
    return torch.randn(1, 32, h, w, generator=torch.Generator().manual_seed(31))


@pytest.mark.parametrize("shape", [(128, 128), (96, 144)])
@torch.inference_mode()
def test_encode_and_decode_agree_with_native(vae, shape):
    h, w = shape
    u8 = np.random.default_rng(17).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    x = torch.from_numpy(u8).permute(2, 0, 1)[None].float() / 127.5 - 1
    opts = dict(device="cpu", tile=32, max_tile=13, chunk_q=29)
    native_z = vae.encode(x).latent_dist.mean
    z = encode_exact_tiled(vae, u8, **opts)
    assert z.shape == (1, 32, h // 8, w // 8)
    assert (z - native_z).abs().max() < 1e-5
    native_y = vae.decode(native_z).sample
    y = decode_exact_tiled(vae, native_z, **opts)
    assert y.shape == x.shape
    assert (y - native_y).abs().max() < 1e-5


@torch.inference_mode()
def test_tiny_rectangle_with_channel_changing_shortcuts():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 64, 64), norm_num_groups=32).eval()
    # Test an odd latent width and the channel-changing shortcut on the virtual
    # upsample's consumer as well as on the decoder's materialized stages.
    x = torch.rand((1, 3, 8, 24), generator=torch.Generator().manual_seed(17)) * 2 - 1
    tiled = ExactTiledFluxVAE(vae, tile=8, max_tile=5, chunk_q=2)
    z = vae.encode(x).latent_dist.mean
    assert (tiled.encode(x) - z).abs().max() < 1e-5
    assert (tiled.decode(z) - vae.decode(z).sample).abs().max() < 1e-5


def test_plane_budgets_account_for_live_planes_and_reusable_cuda_cache(vae, monkeypatch, tmp_path):
    mib = 2**20
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (3 * 1024 * mib, 6 * 1024 * mib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 3 * 1024 * mib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 512 * mib)
    auto = ExactTiledFluxVAE(vae, device="cuda:0")._state(decoder=False)
    assert auto.gpu_cap == int(0.45 * 5.5 * 1024 * mib)
    assert ExactTiledFluxVAE(vae, device="cuda:0", gpu_budget_mib=0)._state(decoder=False).gpu_cap == 0

    state = ExactTiledFluxVAE(vae, cpu_budget_mib=1, spill_dir=tmp_path)._state(decoder=False)
    try:
        plane = state.allocate(32, 64, 128)
        spill = state.allocate(32, 64, 128)
        assert spill.mapping is not None
        assert (state.live_cpu_bytes, state.spill_bytes) == (mib, mib)
        state.release(plane)
        after = state.allocate(32, 64, 128)
        assert after.mapping is None and state.live_cpu_bytes == mib
    finally:
        state.close()
    assert state.live_cpu_bytes == 0 and state.live_gpu_bytes == 0


@torch.inference_mode()
def test_spill_mappings_close_on_success_and_failure(vae, monkeypatch, tmp_path):
    mappings = []
    original_mmap = mmap.mmap

    def track_mapping(*args, **kwargs):
        result = original_mmap(*args, **kwargs)
        mappings.append(result)
        return result

    monkeypatch.setattr(V.mmap, "mmap", track_mapping)
    tiled = ExactTiledFluxVAE(vae, tile=32, max_tile=13, cpu_budget_mib=0, spill_dir=tmp_path)
    z = _latent(12, 18)
    x = torch.rand((1, 3, 96, 144), generator=torch.Generator().manual_seed(17))
    assert (tiled.encode(x) - vae.encode(x).latent_dist.mean).abs().max() < 1e-5
    assert (tiled.decode(z) - vae.decode(z).sample).abs().max() < 1e-5
    assert mappings and all(mapping.closed for mapping in mappings)
    assert list(tmp_path.iterdir()) == []

    real_conv = V._conv_tile

    def fail_conv(state, source, conv, box, before=None):
        if box[2] > 0:
            raise RuntimeError("injected convolution failure")
        return real_conv(state, source, conv, box, before)

    count = len(mappings)
    with monkeypatch.context() as patch:
        patch.setattr(V, "_conv_tile", fail_conv)
        with pytest.raises(RuntimeError, match="injected convolution failure"):
            tiled.encode(x)
    assert len(mappings) > count
    assert all(mapping.closed for mapping in mappings)
    assert list(tmp_path.iterdir()) == []

    if hasattr(V.os, "posix_fallocate"):
        real_fallocate = V.os.posix_fallocate
        allocations = 0

        def fail_second_allocation(fd, start, size):
            nonlocal allocations
            allocations += 1
            if allocations == 2:
                raise OSError(errno.ENOSPC, "injected spill storage exhaustion")
            return real_fallocate(fd, start, size)

        with monkeypatch.context() as patch:
            patch.setattr(V.os, "posix_fallocate", fail_second_allocation)
            with pytest.raises(OSError, match="spill storage exhaustion"):
                tiled.encode(x)
        assert allocations == 2
        assert all(mapping.closed for mapping in mappings)
        assert list(tmp_path.iterdir()) == []


@torch.inference_mode()
def test_global_groupnorm_is_complete_and_window_leaves_model_unchanged(vae, monkeypatch):
    z = _latent(12, 18)
    norms = {name: norm for name, norm in vae.decoder.named_modules() if isinstance(norm, nn.GroupNorm)}
    original_forwards = {name: norm.forward for name, norm in norms.items()}
    original_processors = tuple(m.processor for m in vae.modules() if hasattr(m, "processor"))
    original_y = vae.decode(z).sample
    tiled = ExactTiledFluxVAE(vae, tile=32, chunk_q=17)
    y, stats = tiled.decode(z, return_gn_stats=True)
    assert stats.keys() == norms.keys()
    for name, (a, b) in stats.items():
        assert a.shape == b.shape == (norms[name].num_channels,)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (y - original_y).abs().max() < 1e-5
    torch.testing.assert_close(tiled.decode(z), y, atol=0, rtol=0)
    assert (decode_window_with_global_gn(vae, z, stats) - original_y).abs().max() < 1e-5

    # Use distinguishable stats to prove the forward actually consumes them.
    name = "conv_norm_out"
    edited_stats = dict(stats)
    edited_stats[name] = tuple(torch.zeros_like(v) for v in stats[name])
    window = z[..., :9, :10]
    modified = decode_window_with_global_gn(vae, window, edited_stats)
    expected = vae.decoder.conv_out.bias.view(1, 3, 1, 1).expand_as(modified)
    torch.testing.assert_close(modified, expected, atol=0, rtol=0)

    def fail_forward(*args):
        raise RuntimeError("injected decoder failure")

    with monkeypatch.context() as patch:
        patch.setattr(vae.decoder.conv_out, "forward", fail_forward)
        with pytest.raises(RuntimeError, match="injected decoder failure"):
            decode_window_with_global_gn(vae, window, stats)
    with pytest.raises(ValueError, match="stats do not match"):
        decode_window_with_global_gn(vae, window, {})
    assert all(norm.forward == original_forwards[name] for name, norm in norms.items())
    assert all("forward" not in norm.__dict__ for norm in norms.values())
    assert original_processors == tuple(m.processor for m in vae.modules() if hasattr(m, "processor"))
    torch.testing.assert_close(vae.decode(z).sample, original_y, atol=0, rtol=0)


def test_shared_executor_keeps_concurrent_runs_independent(vae):
    tiled = ExactTiledFluxVAE(vae, tile=32, chunk_q=17)
    inputs = (_latent(7, 9), _latent(9, 7))
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(lambda z: tiled.decode(z, return_gn_stats=True), inputs))
    for z, (y, stats) in zip(inputs, parallel, strict=True):
        serial_y, serial_stats = tiled.decode(z, return_gn_stats=True)
        torch.testing.assert_close(y, serial_y, atol=1e-6, rtol=0)
        for name in stats:
            torch.testing.assert_close(stats[name], serial_stats[name], atol=1e-6, rtol=0)


def _grid(sha="ab" * 32):
    sm = torch.full((2, 64, 64), 32, dtype=torch.long)
    leaves = tuple(RL.leaves_rect(sm))
    blocks = [RectTokenFile(sm, leaves, torch.full((4, 128), float(i)), sha, seed=7) for i in range(4)]
    return RectGridTokenFile(835, 513, 2, 2, "reflect", blocks, sha, seed=7, meta={"mode": "router"})


def test_v2_roundtrip_size_and_dispatch(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.write_bytes(b"checkpoint")
    grid = _grid(file_sha256(ckpt))
    data = dumps_grid_tokens(grid)
    metadata = json.dumps(grid.meta, separators=(",", ":")).encode()
    tree_bytes = sum(len(encode_kdtree_bits(block.shape_map)[0]) for block in grid.blocks)
    assert len(data) == 68 + len(metadata) + 6 * 4 + tree_bytes + 4 * 128 * 16
    assert struct.unpack_from("<4sH", data) == (b"TSR2", FORMAT_VERSION_GRID)

    grid_path = save_grid_tokens(tmp_path / "native" / "grid.tsr", grid)
    v1_path = save_tokens(tmp_path / "block.tsr", grid.blocks[0])
    assert isinstance(load_any_tokens(v1_path), RectTokenFile)
    assert load_any_tokens(v1_path).version == FORMAT_VERSION
    actual = load_any_tokens(grid_path)
    assert isinstance(actual, RectGridTokenFile)
    assert (actual.orig_height, actual.orig_width, actual.grid_h, actual.grid_w, actual.pad_mode) == (835, 513, 2, 2, "reflect")
    assert actual.block_tokens == [4] * 4
    assert (actual.n_blocks, actual.n_tokens, actual.meta, actual.seed) == (4, 16, grid.meta, 7)
    assert dumps_grid_tokens(load_grid_tokens(grid_path)) == data
    for original, loaded in zip(grid.blocks, actual.blocks, strict=True):
        torch.testing.assert_close(loaded.shape_map, original.shape_map, atol=0, rtol=0)
        torch.testing.assert_close(loaded.tokens, original.tokens, atol=0, rtol=0)
    assert verify_checkpoint_sha256(actual, ckpt) == grid.ckpt_sha256
    ckpt.write_bytes(b"other checkpoint")
    with pytest.raises(CheckpointMismatchError, match="mismatch"):
        verify_checkpoint_sha256(actual, ckpt)

    grid.blocks[2].meta["judge_calls"] = 7
    assert loads_grid_tokens(dumps_grid_tokens(grid)).blocks[2].meta == {"judge_calls": 7}


@pytest.mark.parametrize("offset, fmt, value", [
    (4, "<H", 1),     # version
    (6, "<H", 2),     # unknown flag
    (8, "<I", 0),     # original height
    (16, "<H", 0),    # grid height
    (20, "<B", 255),  # unknown padding
    (20, "<B", 0),    # none with nonaligned dimensions
    (21, "<B", 1),    # unsupported leaf order
    (28, "<f", float("nan")),
    (32, "<I", 70000),
])
def test_v2_rejects_malformed_header(offset, fmt, value):
    data = bytearray(dumps_grid_tokens(_grid()))
    struct.pack_into(fmt, data, offset, value)
    with pytest.raises(TokenFormatError):
        loads_grid_tokens(data)


def test_v2_rejects_truncation_invalid_tree_and_non_finite_tokens():
    data = dumps_grid_tokens(_grid())
    for bad_size in (0, 67, 100, len(data) - 1):
        with pytest.raises(TokenFormatError):
            loads_grid_tokens(data[:bad_size])
    with pytest.raises(TokenFormatError):
        loads_grid_tokens(data + b"\x00")

    metadata_size, = struct.unpack_from("<I", data, 32)
    descriptor_offset = 68 + metadata_size
    tree_offset = descriptor_offset + 6 * 4
    for offset, fmt, value in [
        (descriptor_offset, "<H", 3),
        (descriptor_offset + 2, "<H", 9),
        (tree_offset, "<B", 0b11000000),
        (len(data) - 4, "<f", float("inf")),
    ]:
        bad = bytearray(data)
        struct.pack_into(fmt, bad, offset, value)
        with pytest.raises(TokenFormatError):
            loads_grid_tokens(bad)


def test_grid_assembles_latents_in_row_major_order(monkeypatch):
    grid = _grid()

    def fake_decode(shape_map, tokens, model, basis_dev, mean, std, **settings):
        assert settings == dict(seed=7, tu_spike_k=3.0, tu_spike_min_area=64)
        return tokens[0, 0].expand(1, 32, 64, 64)

    monkeypatch.setattr(T, "decode_tokens_to_latent", fake_decode)
    dummy = torch.zeros(1)
    z = decode_grid_tokens_to_latent(grid, None, {(2, 2): {"U": dummy}}, dummy, dummy)
    assert z.shape == (1, 32, 128, 128)
    for i in range(4):
        row, col = divmod(i, 2)
        assert (z[..., row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] == i).all()


def _fake_sample(z):
    # Mean deliberately depends on the window context, like a global layer.
    return F.interpolate(z[:, :3].float(), scale_factor=8, mode="nearest") + 0.07 * z[:, 0].mean()


def _install_grid_codec_mocks(monkeypatch):
    window_shapes = []
    image_shapes = []
    _install_fast_mocks(monkeypatch)
    legacy_decoder = E.decoder_only_flux

    class FakeVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Identity()
            self.quant_conv = None

    class FakeTiled:
        def __init__(self, vae, *, device):
            self.vae, self.device = vae, device

        def encode(self, x_u8):
            image_shapes.append(x_u8.copy())
            x = torch.from_numpy(x_u8).permute(2, 0, 1)[None].float() / 127.5 - 1
            return F.avg_pool2d(x, 8, 8).repeat(1, 11, 1, 1)[:, :32].contiguous()

        def decode(self, z, *, return_gn_stats=False):
            y = _fake_sample(z)
            return (y, {"mock": ()}) if return_gn_stats else y

    def fake_decoder(dev, vae=None, repo_id=None):
        dec = legacy_decoder(dev, vae, repo_id)
        dec.vae = vae if vae is not None else FakeVAE()
        return dec

    def fake_window(vae, z, stats, *, device):
        assert stats == {"mock": ()}
        window_shapes.append(z.shape[-2:])
        return _fake_sample(z)

    monkeypatch.setattr(E, "ExactTiledFluxVAE", FakeTiled)
    monkeypatch.setattr(E, "load_flux_vae", lambda **kwargs: FakeVAE())
    monkeypatch.setattr(E, "decoder_only_flux", fake_decoder)
    monkeypatch.setattr(E, "decode_window_with_global_gn", fake_window)
    return image_shapes, window_shapes


@pytest.mark.parametrize("shape, flag", [((513, 549), "--native-res"), ((1, 17), "--tiled"), ((512, 512), "--native-res")])
def test_native_res_cli_pads_and_crops_roundtrip(tmp_path, monkeypatch, shape, flag):
    images, windows = _install_grid_codec_mocks(monkeypatch)
    ckpt = _make_tiny_ckpt(tmp_path / "ckpt.pt")
    photo = tmp_path / "photo.png"
    photo_u8 = np.random.default_rng(22).integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    Image.fromarray(photo_u8).save(photo)
    tokens = tmp_path / "out.tsr"
    result = tmp_path / "decoded.png"
    opts = ["--device", "cpu", "--ckpt", str(ckpt)]
    runner = CliRunner()
    encoded = runner.invoke(E.main, [
        "encode", str(photo), str(tokens), "--mode", "topdown", flag, "--halo-cells", "7", *opts,
    ])
    assert encoded.exit_code == 0, (encoded.output, encoded.exception)
    tf = load_any_tokens(tokens)
    if shape == (512, 512):
        assert isinstance(tf, RectTokenFile)
        assert not images and not windows
    else:
        assert isinstance(tf, RectGridTokenFile)
        assert (tf.orig_height, tf.orig_width) == shape
        assert tf.pad_mode == "reflect" and tf.meta["halo_cells"] == 7
        assert tf.n_blocks == (4 if shape[0] > 512 else 1)
        assert len(images) == 1 and len(windows) == 2 * tf.n_blocks
        expected = np.pad(photo_u8, ((0, -shape[0] % 512), (0, -shape[1] % 512), (0, 0)), mode="reflect")
        np.testing.assert_array_equal(images[0], expected)
        assert all(window == (71, 71) if tf.n_blocks == 4 else window == (64, 64) for window in windows)

    reference_u8 = None
    if shape == (1, 17):
        _, reference_u8 = E.encode_image(
            photo, tmp_path / "api.tsr", device="cpu", ckpt=ckpt, mode="topdown",
            native_res=True, return_reference_u8=True,
        )
    photo.unlink()
    decoded = runner.invoke(E.main, ["decode", str(tokens), str(result), *opts])
    assert decoded.exit_code == 0, (decoded.output, decoded.exception)
    with Image.open(result) as image:
        assert image.size == shape[::-1]
        if reference_u8 is not None:
            np.testing.assert_array_equal(reference_u8, np.array(image))


def test_copythrough_uses_global_reference_and_candidates_use_paired_halo(monkeypatch):
    _, windows = _install_grid_codec_mocks(monkeypatch)
    tiled = E.ExactTiledFluxVAE(object(), device="cpu")
    z = _latent(64, 128)
    golden = _fake_sample(z)
    original, decode = E._block_judge_decoder(tiled, z, golden, {"mock": ()}, 0, 64, 7)
    torch.testing.assert_close(decode(original.clone()), golden[..., 512:], atol=0, rtol=0)
    assert not windows
    candidate = original + 0.05
    window = z[..., 57:]
    patched = window.clone()
    patched[..., 7:] = candidate
    expected = golden[..., 512:] + (_fake_sample(patched) - _fake_sample(window))[..., 56:]
    torch.testing.assert_close(decode(candidate), expected, atol=0, rtol=0)
    decode(candidate + 0.05)
    assert windows == [(64, 71)] * 3
