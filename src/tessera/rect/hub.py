"""Checkpoint resolution, verified defaults, device validation, and FLUX.2 VAE loading for Tessera."""
import os
import pathlib

import torch

DEFAULT_REPO_ID = "tensorkelechi/tessera-checkpoints"
DEFAULT_CKPT_FILENAME = "phase3_large_tl/step_20000.pt"
DEFAULT_ROUTER_FILENAME = "router/router_rect.pt"

FLUX_VAE_REPO_ID = DEFAULT_REPO_ID
FLUX_VAE_SUBFOLDER = "vae"

# Shipped verified-router defaults (matches the published 90-image benchmark: 100% Judge pass, 369.6 mean tokens).
VERIFIED_ENCODE_DEFAULTS = dict(
    mode="router",
    leaf_bias=0.5,
    verify=True,
    verify_rounds=8,
    judge_scale=1.0,
    struct_scale=1.5,
    color_floor_de=1.0,
    color_max_floor_de=1.8,
    tu_spike_k=3.0,
    tu_spike_min_area=64,
)


def resolve_device(device: str | torch.device | None = None) -> str:
    """Resolve and validate a device string (`None`/`'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'`).

    Returns a concrete normalized string (`'cpu'` or `'cuda:<index>'`) so multi-GPU selection is unambiguous
    across TileUp, FLUX.2 VAE, and Judge v5.
    """
    if device is None:
        raw = "auto"
    elif isinstance(device, torch.device):
        raw = str(device)
    else:
        raw = str(device).strip()
        if not raw:
            raw = "auto"

    if raw == "auto":
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    try:
        parsed = torch.device(raw)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {device!r}; expected 'auto', 'cpu', 'cuda', or 'cuda:<index>'") from exc

    if parsed.type == "cpu":
        if parsed.index is not None:
            raise ValueError(f"Invalid CPU device {device!r}; use 'cpu'")
        return "cpu"

    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {device!r}, but CUDA is not available on this machine")
        count = torch.cuda.device_count()
        idx = torch.cuda.current_device() if parsed.index is None else int(parsed.index)
        if idx < 0 or idx >= count:
            raise ValueError(
                f"Requested CUDA device index {idx}, but only {count} CUDA device(s) are available (0..{count - 1})"
            )
        return f"cuda:{idx}"

    raise ValueError(f"Unsupported device {device!r}; expected 'auto', 'cpu', 'cuda', or 'cuda:<index>'")


def _hf_download_with_cache_fallback(repo_id: str, filename: str, *, local_files_only: bool = False) -> pathlib.Path:
    """Download a file via Hugging Face Hub, falling back to local HF cache when offline."""
    from huggingface_hub import hf_hub_download

    offline = local_files_only or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        return pathlib.Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=offline))
    except Exception as first_exc:
        if not offline:
            try:
                return pathlib.Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except Exception:
                pass
        raise RuntimeError(
            f"Could not resolve '{filename}' from Hugging Face repo '{repo_id}'. "
            f"Check your network connection, repo ID, or local Hugging Face cache. ({first_exc})"
        ) from first_exc


def resolve_ckpt_path(
    ckpt: str | os.PathLike | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_CKPT_FILENAME,
    local_files_only: bool = False,
) -> pathlib.Path:
    """Resolve the tokenizer checkpoint (`step_20000.pt`) from an explicit path or the HF cache/hub."""
    if ckpt is not None:
        p = pathlib.Path(ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {p}")
        return p
    return _hf_download_with_cache_fallback(repo_id, filename, local_files_only=local_files_only)


def resolve_router_path(
    router: str | os.PathLike | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_ROUTER_FILENAME,
    local_files_only: bool = False,
) -> pathlib.Path:
    """Resolve the router checkpoint (`router_rect.pt`) from an explicit path or the HF cache/hub."""
    if router is not None:
        p = pathlib.Path(router)
        if not p.is_file():
            raise FileNotFoundError(f"Router checkpoint not found: {p}")
        return p
    return _hf_download_with_cache_fallback(repo_id, filename, local_files_only=local_files_only)


def load_flux_vae(
    repo_id: str = DEFAULT_REPO_ID,
    subfolder: str = FLUX_VAE_SUBFOLDER,
    *,
    local_files_only: bool = False,
):
    """Load the FLUX.2 VAE from `repo_id` (default: public `tensorkelechi/tessera-checkpoints`, subfolder `vae`).

    Falls back to the local Hugging Face cache when offline.
    """
    from diffusers import AutoencoderKLFlux2

    offline = local_files_only or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        return AutoencoderKLFlux2.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=torch.float32, local_files_only=offline
        )
    except Exception as first_exc:
        if not offline:
            try:
                return AutoencoderKLFlux2.from_pretrained(
                    repo_id, subfolder=subfolder, torch_dtype=torch.float32, local_files_only=True
                )
            except Exception:
                pass
        raise RuntimeError(
            f"Could not load FLUX.2 VAE from Hugging Face repo '{repo_id}' (subfolder '{subfolder}'). "
            f"Check your network connection, repo ID, or local Hugging Face cache. ({first_exc})"
        ) from first_exc
