"""Tests for device resolution, HF hub/cache resolution, standalone encode/decode CLI, and legacy eval compatibility."""
import json
import pathlib

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from tessera.latent import rect as RL
from tessera.latent import rect_tiledown as RTD
from tessera.rect import encode as E
from tessera.rect import hub
from tessera.rect.model import TileUp
from tessera.rect.search import IN_CH_RECT
from tessera.rect.tokens import RectTokenFile, file_sha256, load_tokens, save_tokens


def test_resolve_device_cpu_and_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hub.resolve_device(None) == "cpu"
    assert hub.resolve_device("auto") == "cpu"
    assert hub.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA is not available"):
        hub.resolve_device("cuda")
    with pytest.raises(ValueError, match="CUDA is not available"):
        hub.resolve_device("cuda:0")
    with pytest.raises(ValueError, match="Invalid CPU device"):
        hub.resolve_device("cpu:0")
    with pytest.raises(ValueError, match="Unsupported|Invalid"):
        hub.resolve_device("mps")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert hub.resolve_device("auto") == "cuda:0"
    assert hub.resolve_device("cuda") == "cuda:0"
    assert hub.resolve_device("cuda:0") == "cuda:0"
    assert hub.resolve_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="only 2 CUDA device"):
        hub.resolve_device("cuda:2")


def test_hub_resolution_and_offline_cache_fallback(tmp_path, monkeypatch):
    cached = tmp_path / "step_20000.pt"
    cached.write_bytes(b"cached-weights")
    calls = []

    def fake_download(*, repo_id, filename, local_files_only=False):
        calls.append((repo_id, filename, local_files_only))
        if not local_files_only:
            raise OSError("network offline")
        return str(cached)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    got = hub.resolve_ckpt_path(None, repo_id="custom/repo")
    assert got == cached
    assert calls == [
        ("custom/repo", hub.DEFAULT_CKPT_FILENAME, False),
        ("custom/repo", hub.DEFAULT_CKPT_FILENAME, True),
    ]

    # Explicit local file bypasses hub completely
    calls.clear()
    assert hub.resolve_ckpt_path(cached, repo_id="ignored/repo") == cached
    assert calls == []


def test_flux_vae_uses_default_public_repo_and_offline_fallback(monkeypatch):
    calls = []
    sentinel = object()

    def fake_pretrained(repo_id, *, subfolder=None, torch_dtype=None, local_files_only=False):
        calls.append((repo_id, subfolder, local_files_only))
        if not local_files_only:
            raise OSError("network offline")
        return sentinel

    monkeypatch.setattr("diffusers.AutoencoderKLFlux2.from_pretrained", fake_pretrained)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert hub.load_flux_vae() is sentinel
    assert calls == [
        (hub.DEFAULT_REPO_ID, "vae", False),
        (hub.DEFAULT_REPO_ID, "vae", True),
    ]

    def fail_pretrained(*args, **kwargs):
        raise OSError("missing repo")

    monkeypatch.setattr("diffusers.AutoencoderKLFlux2.from_pretrained", fail_pretrained)
    with pytest.raises(RuntimeError) as exc_info:
        hub.load_flux_vae(repo_id="custom/repo")
    assert "custom/repo" in str(exc_info.value)


def _make_tiny_ckpt(path: pathlib.Path) -> pathlib.Path:
    torch.manual_seed(0)
    z_fit = torch.randn(2, RL.CH, RL.SIDE, RL.SIDE)
    basis = RL.fit_rect_basis(z_fit)
    tu = TileUp(c=(16, 16, 16), in_ch=IN_CH_RECT, attn_blocks=1).eval()
    td = RTD.RectTileDown(c=16, hid=32, use_1d_pool=True, prof_c=4, prof_len=8, dc_lock=True).eval()
    tl = RTD.RectTileLift(hid=32, rank=2, prof_len=8).eval()
    torch.save(
        {
            "model": tu.state_dict(),
            "tiledown": td.state_dict(),
            "tilelift": tl.state_dict(),
            "basis": basis,
            "mean": torch.zeros(RL.CH),
            "std": torch.ones(RL.CH),
            "widths": (16, 16, 16),
            "attn_blocks": 1,
            "td_c": 16,
            "td_hid": 32,
            "td_1d_pool": True,
            "td_prof_c": 4,
            "td_prof_len": 8,
            "td_dc_lock": True,
            "tl_hid": 32,
            "tl_rank": 2,
            "tl_prof_len": 8,
            "zero_mean_resid": True,
            "ac_floor": 0.26,
            "ac_tau": 0.20,
        },
        path,
    )
    return path


def _install_fast_mocks(monkeypatch, *, pass_judge=True):
    """Patch only the heavy FLUX.2 VAE + Judge inside encode.py so CLI flows run in <1s on CPU."""

    def fake_load_images(images_path, limit, dev, paths=None, return_vae=False, repo_id=None):
        if paths is None:
            paths = E._resolve_image_paths(images_path, limit)
        entries = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
            u8 = np.asarray(im, dtype=np.uint8).copy()
            # Deterministic latent from image pixels (downsampled 8x, repeated across 32 channels)
            t = torch.from_numpy(u8).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            z3 = torch.nn.functional.avg_pool2d(t, 8, 8)
            z = z3.repeat(1, 11, 1, 1)[:, :32].contiguous()
            entries.append((p.stem, z.cpu(), u8))
        return (entries, object()) if return_vae else entries

    def fake_decoder(dev, vae=None, repo_id=None):
        def dec(z):
            rgb = z[:, :3].float()
            return torch.nn.functional.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=False).clamp(-1, 1)
        return dec

    judge_calls = []

    class FakeJudge:
        def __init__(self, x_u8, g_u8, dev, **kwargs):
            self.kwargs = kwargs

        def __call__(self, y):
            judge_calls.append(self.kwargs)
            return {"passed": pass_judge, "bad8": torch.zeros((64, 64), dtype=torch.bool), "score": 0.1}

    monkeypatch.setattr(E, "_load_images", fake_load_images)
    monkeypatch.setattr(E, "decoder_only_flux", fake_decoder)
    monkeypatch.setattr(E, "Judge", FakeJudge)
    return judge_calls


def test_cli_standalone_encode_decode_no_source_dependency_and_mismatch(tmp_path, monkeypatch):
    _install_fast_mocks(monkeypatch)
    ckpt = _make_tiny_ckpt(tmp_path / "ckpt.pt")
    other_ckpt = _make_tiny_ckpt(tmp_path / "other.pt")
    # Make other_ckpt have a different sha256
    with other_ckpt.open("ab") as f:
        f.write(b"\x00")

    photo = tmp_path / "photo.png"
    arr = np.linspace(0, 255, 512, dtype=np.uint8)[None, :, None].repeat(512, axis=0).repeat(3, axis=2)
    Image.fromarray(arr).save(photo)

    tok_path = tmp_path / "photo.tsr"
    recon_path = tmp_path / "recon.png"
    runner = CliRunner()

    # Encode (topdown avoids router ckpt in unit test; router default is also tested via mock)
    res_enc = runner.invoke(
        E.main,
        ["encode", str(photo), str(tok_path), "--ckpt", str(ckpt), "--mode", "topdown", "--device", "cpu"],
    )
    assert res_enc.exit_code == 0, res_enc.output
    assert tok_path.is_file()

    # Delete source photo completely to prove decode has zero source dependency
    photo.unlink()
    assert not photo.exists()

    # Decode with mismatched checkpoint must fail cleanly and not write output
    res_bad = runner.invoke(
        E.main,
        ["decode", str(tok_path), str(recon_path), "--ckpt", str(other_ckpt), "--device", "cpu"],
    )
    assert res_bad.exit_code != 0
    assert "mismatch" in res_bad.output.lower()
    assert not recon_path.exists()

    # Decode with matching checkpoint succeeds
    res_dec = runner.invoke(
        E.main,
        ["decode", str(tok_path), str(recon_path), "--ckpt", str(ckpt), "--device", "cpu"],
    )
    assert res_dec.exit_code == 0, res_dec.output
    assert recon_path.is_file()
    assert Image.open(recon_path).size == (512, 512)

    # Corrupted token file fails cleanly
    corrupt = tmp_path / "corrupt.tsr"
    corrupt.write_bytes(b"NOT_TSR2_DATA")
    res_corrupt = runner.invoke(
        E.main,
        ["decode", str(corrupt), str(tmp_path / "bad.png"), "--ckpt", str(ckpt), "--device", "cpu"],
    )
    assert res_corrupt.exit_code != 0
    assert not (tmp_path / "bad.png").exists()


def test_legacy_eval_cli_direct_dot_out_and_no_metrics_preserves_judge(tmp_path, monkeypatch):
    judge_calls = _install_fast_mocks(monkeypatch)
    ckpt = _make_tiny_ckpt(tmp_path / "ckpt.pt")
    photo = tmp_path / "photo.png"
    Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8)).save(photo)

    # Ensure summary LPIPS/DISTS are NOT loaded when --no_metrics is passed
    def fail_dists(*args, **kwargs):
        raise AssertionError("load_dists_model should not be called with --no_metrics")

    monkeypatch.setattr(E, "load_dists_model", fail_dists)

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    # Direct `./out` positional path (legacy CLI syntax!) + --no_metrics + --device cpu
    res = runner.invoke(
        E.main,
        ["./out", "--images", str(photo), "--ckpt", str(ckpt), "--mode", "topdown", "--device", "cpu", "--no_metrics"],
    )
    assert res.exit_code == 0, res.output
    assert len(judge_calls) >= 1  # Real Judge still ran!
    results = json.loads(pathlib.Path("./out/results.json").read_text())
    assert results["meta"]["metrics"] is False
    assert results["meta"]["device"] == "cpu"
    row = results["rows"][0]
    assert row["passed"] is True
    assert "tg" not in row and "to" not in row and "go" not in row
    assert list(pathlib.Path("./out/strips").glob("*.png"))

    # Legacy invocation with output folder literally named "encode" still routes via --images
    res2 = runner.invoke(
        E.main,
        ["encode", "--images", str(photo), "--ckpt", str(ckpt), "--mode", "topdown", "--device", "cpu", "--no-metrics"],
    )
    assert res2.exit_code == 0, res2.output
    assert pathlib.Path("encode/results.json").is_file()


def test_cli_help_and_invalid_device():
    runner = CliRunner()
    for args in (["--help"], ["encode", "--help"], ["decode", "--help"], ["eval", "--help"], ["--version"]):
        res = runner.invoke(E.main, args)
        assert res.exit_code == 0, (args, res.output)
    res_bad_dev = runner.invoke(E.main, ["decode", __file__, "out.png", "--device", "invalid_dev"])
    assert res_bad_dev.exit_code != 0
    assert "invalid device" in res_bad_dev.output.lower() or "unsupported device" in res_bad_dev.output.lower()
