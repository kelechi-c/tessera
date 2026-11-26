"""Layer-synchronized inference for the FLUX.2 VAE.

Convolutions read exact halos from completed feature planes. Each layer
accumulates the next GroupNorm's image-wide moments as it writes its cores.
The stride-8 bottleneck retains global attention with chunked queries.
"""

from __future__ import annotations

import copy
import math
import mmap
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

_MIB = 2**20
_CL = torch.channels_last
_Box = tuple[int, int, int, int]
_Affine = tuple[Tensor, Tensor]
GNStats = dict[str, _Affine]
_Budget = float | Literal["auto"]


def _memory_format(device: torch.device) -> torch.memory_format:
    return _CL if device.type == "cuda" else torch.contiguous_format


def _cut(x: Tensor, box: _Box) -> Tensor:
    y0, y1, x0, x1 = box
    return x[..., y0:y1, x0:x1]


def _tiles(height: int, width: int, side: int) -> Iterator[_Box]:
    for y in range(0, height, side):
        for x in range(0, width, side):
            yield y, min(y + side, height), x, min(x + side, width)


def _support(box: _Box, height: int, width: int, stride: int = 1, padding: int = 1) -> tuple[_Box, _Box]:
    y0, y1, x0, x1 = box
    ya, yb = y0 * stride - padding, (y1 - 1) * stride - padding + 3
    xa, xb = x0 * stride - padding, (x1 - 1) * stride - padding + 3
    return (max(0, ya), min(height, yb), max(0, xa), min(width, xb)), (
        max(0, -xa), max(0, xb - width), max(0, -ya), max(0, yb - height)
    )


@dataclass(eq=False)
class _Plane:
    tensor: Tensor
    mapping: mmap.mmap | None = None
    moments: _Moments | None = None

    @property
    def height(self) -> int:
        return self.tensor.size(-2)

    @property
    def width(self) -> int:
        return self.tensor.size(-1)

    def read(self, box: _Box, device: torch.device) -> Tensor:
        # A CPU view must never outlive a mapped plane.
        return _cut(self.tensor, box).to(device, memory_format=_memory_format(device),
                                         copy=self.mapping is not None)

    def write(self, box: _Box, value: Tensor) -> None:
        _cut(self.tensor, box).copy_(value)
        if self.moments is not None:
            self.moments.add(value)

    def discard_pages(self) -> None:
        if self.mapping is not None and hasattr(self.mapping, "madvise"):
            self.mapping.madvise(mmap.MADV_DONTNEED)


def _affine(norm: nn.GroupNorm, mean: Tensor, inv: Tensor) -> _Affine:
    repeat = norm.num_channels // norm.num_groups
    mean = mean.float().repeat_interleave(repeat)
    a = inv.float().repeat_interleave(repeat)
    if norm.weight is not None:
        a = a * norm.weight.float()
    bias = norm.bias.float() if norm.bias is not None else torch.zeros_like(a)
    return a, torch.addcmul(bias, -mean, a)


def _normalize(x: Tensor, affine: _Affine, dtype: torch.dtype) -> Tensor:
    shape = (1, -1) + (1,) * (x.ndim - 2)
    a, b = affine
    normalized = torch.addcmul(b.view(shape), x.float(), a.view(shape)).to(dtype)
    return normalized.contiguous() if x.device.type == "cpu" else normalized


class _Moments:
    def __init__(self, norm: nn.GroupNorm, device: torch.device) -> None:
        self.norm = norm
        self.sum = torch.zeros(norm.num_groups, device=device, dtype=torch.float64)
        self.squares = torch.zeros_like(self.sum)
        self.count = 0

    def add(self, x: Tensor) -> None:
        groups = x.permute(0, 2, 3, 1).unflatten(-1, (self.norm.num_groups, -1))
        axes = (0, 1, 2, 4)
        self.sum += groups.sum(axes, dtype=torch.float64)
        # For half-precision activations, FP32 retains the squared significand.
        square_dtype = torch.float64 if x.dtype == torch.float32 else torch.float32
        self.squares += groups.to(square_dtype).square_().sum(axes, dtype=torch.float64)
        self.count += x.numel() // self.norm.num_groups

    def finish(self, height: int, width: int, dtype: torch.dtype) -> _Affine:
        if self.count * self.norm.num_groups != height * width * self.norm.num_channels:
            raise RuntimeError("Incomplete tiled GroupNorm moments")
        mean = self.sum / self.count
        var = (self.squares / self.count - mean.square()).clamp_min_(0)
        # CUDA's FP16 encoder kernel rounds epsilon to its input dtype.
        eps = torch.tensor(self.norm.eps, dtype=dtype).item()
        return _affine(self.norm, mean, (var.float() + eps).rsqrt())


class _GroupNormMode(TorchFunctionMode):
    """Thread-local functional interception of specific affine GroupNorms."""

    def __init__(
        self,
        norms: Mapping[str, nn.GroupNorm],
        dtype: torch.dtype,
        *,
        frozen: Mapping[str, _Affine] | None = None,
        capture: GNStats | None = None,
    ) -> None:
        if any(norm.weight is None for norm in norms.values()):
            raise ValueError("Exact FLUX.2 inference requires affine GroupNorm")
        self.norms = {id(norm.weight): (name, norm) for name, norm in norms.items()}
        self.dtype = dtype
        self.frozen = frozen
        self.capture = capture

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.group_norm and id(kwargs.get("weight")) in self.norms:
            x, groups = args
            weight, bias = kwargs["weight"], kwargs["bias"]
            name, norm = self.norms[id(weight)]
            if self.frozen is not None:
                return _normalize(x, self.frozen[name], self.dtype)
            # GroupNorm is an FP32 autocast op on CUDA. Use the same kernel
            # and capture its mean and reciprocal std with no extra scan.
            y, mean, inv = torch.native_group_norm(
                x.float().contiguous(), weight.float(), None if bias is None else bias.float(),
                x.size(0), x.size(1), x.numel() // (x.size(0) * x.size(1)), groups, norm.eps,
            )
            assert self.capture is not None
            self.capture[name] = _affine(norm, mean[0], inv[0])
            return y
        return func(*args, **kwargs)


def _named_norms(module: nn.Module) -> dict[str, nn.GroupNorm]:
    return {name: norm for name, norm in module.named_modules() if isinstance(norm, nn.GroupNorm)}


def _run_mid_block(block: nn.Module, x: Tensor, chunk_q: int) -> Tensor:
    """Run the FLUX.2 mid-block with all keys and values for each query chunk."""
    x = block.resnets[0](x, None)
    for attn, resnet in zip(block.attentions, block.resnets[1:], strict=True):
        if attn is not None:
            batch, channels, height, width = x.shape
            seq = attn.group_norm(x.flatten(2)).transpose(1, 2)
            q, k, v = [
                linear(seq).view(batch, -1, attn.heads, channels // attn.heads).transpose(1, 2)
                for linear in (attn.to_q, attn.to_k, attn.to_v)
            ]
            # The normalized input and each projection are full latent planes.
            # Release them as soon as their last consumer has been queued.
            del seq
            attended = torch.empty_like(q)
            for start in range(0, height * width, chunk_q):
                end = min(start + chunk_q, height * width)
                attended[:, :, start:end].copy_(F.scaled_dot_product_attention(
                    q[:, :, start:end].contiguous(), k, v, dropout_p=0.0,
                ))
            del q, k, v
            seq = attended.transpose(1, 2).reshape(batch, -1, channels)
            value = attn.to_out[1](attn.to_out[0](seq)).transpose(1, 2).reshape_as(x)
            del attended, seq
            x = ((value + x) if attn.residual_connection else value) / attn.rescale_output_factor
            del value
        x = resnet(x, None)
    return x


@dataclass
class _RunState:
    device: torch.device
    dtype: torch.dtype
    decoder: bool
    gpu_cap: int
    cpu_cap: int
    spill_dir: Path | None
    norm_names: dict[nn.GroupNorm, str] = field(default_factory=dict)
    gn_stats: GNStats | None = None
    macs: dict[str, int] = field(default_factory=lambda: dict(convolution=0, recomputed_upsample=0, attention=0))
    spill_bytes: int = 0
    live_gpu_bytes: int = 0
    live_cpu_bytes: int = 0
    planes: set[_Plane] = field(default_factory=set)

    def own(self, plane: _Plane) -> _Plane:
        size = plane.tensor.nbytes
        if plane.tensor.is_cuda:
            self.live_gpu_bytes += size
        elif plane.mapping is None:
            self.live_cpu_bytes += size
        self.planes.add(plane)
        return plane

    def allocate(self, channels: int, height: int, width: int, next_norm: nn.GroupNorm | None = None) -> _Plane:
        size = channels * height * width * (4 if self.dtype == torch.float32 else 2)
        shape = (1, channels, height, width)
        moments = None if next_norm is None else _Moments(next_norm, self.device)
        if self.device.type == "cuda" and self.live_gpu_bytes + size <= self.gpu_cap:
            return self.own(_Plane(torch.empty(shape, device=self.device, dtype=self.dtype, memory_format=_CL),
                                   moments=moments))
        if self.live_cpu_bytes + size <= self.cpu_cap:
            return self.own(_Plane(torch.empty(shape, device="cpu", dtype=self.dtype,
                                               memory_format=_memory_format(self.device)),
                                   moments=moments))
        with tempfile.TemporaryFile(dir=self.spill_dir) as fd:
            if hasattr(os, "posix_fallocate"):
                # Reserve backing storage up front so exhaustion raises an
                # ordinary I/O error before the first mapped write.
                os.posix_fallocate(fd.fileno(), 0, size)
            else:
                fd.truncate(size)
            mapping = mmap.mmap(fd.fileno(), size)
        try:
            buffer = torch.frombuffer(mapping, dtype=self.dtype)
            tensor = (buffer.view(1, height, width, channels).permute(0, 3, 1, 2)
                      if self.device.type == "cuda" else buffer.view(shape))
            plane = self.own(_Plane(tensor, mapping, moments))
        except BaseException:
            mapping.close()
            raise
        self.spill_bytes += size
        return plane

    def release(self, plane: _Plane) -> None:
        self.planes.remove(plane)
        if plane.tensor.is_cuda:
            self.live_gpu_bytes -= plane.tensor.nbytes
        elif plane.mapping is None:
            self.live_cpu_bytes -= plane.tensor.nbytes
        del plane.tensor
        plane.moments = None
        if plane.mapping is not None:
            plane.mapping.close()
            plane.mapping = None

    def close(self) -> None:
        while self.planes:
            self.release(next(iter(self.planes)))

    def count_conv(self, conv: nn.Conv2d, h: int, w: int, kind: str = "convolution") -> None:
        self.macs[kind] += h * w * conv.weight.numel()

    def count_resnet(self, block: nn.Module, h: int, w: int) -> None:
        for conv in (block.conv1, block.conv2, block.conv_shortcut):
            if conv is not None:
                self.count_conv(conv, h, w)

    def count_mid(self, block: nn.Module, h: int, w: int) -> None:
        for resnet in block.resnets:
            self.count_resnet(resnet, h, w)
        for attn in block.attentions:
            if attn is not None:
                n, c = h * w, attn.to_q.in_features
                self.macs["attention"] += 2 * n * n * c + 4 * n * c**2

    def finish_norm(self, source: _Source, stats: _Moments) -> _Affine:
        dtype = torch.float32 if self.decoder else self.dtype
        affine = stats.finish(source.height, source.width, dtype)
        if self.gn_stats is not None:
            self.gn_stats[self.norm_names[stats.norm]] = affine
        return affine


@dataclass
class _Upsampled:
    """Virtual last upsample, which would otherwise hold a very wide RGB-scale plane."""

    source: _Plane
    conv: nn.Conv2d
    norm: nn.GroupNorm
    moments: _Moments | None = None

    @property
    def height(self) -> int:
        return self.source.height * 2

    @property
    def width(self) -> int:
        return self.source.width * 2

    def read(self, box: _Box, device: torch.device) -> Tensor:
        bounds, pad = _support(box, self.height, self.width)
        y0, y1, x0, x1 = bounds
        ya, xa = y0 // 2, x0 // 2
        x = F.interpolate(self.source.read((ya, (y1 + 1) // 2, xa, (x1 + 1) // 2), device),
                          scale_factor=2, mode="nearest")
        x = _cut(x, (y0 - ya * 2, y1 - ya * 2, x0 - xa * 2, x1 - xa * 2))
        return F.conv2d(F.pad(x, pad) if any(pad) else x, self.conv.weight, self.conv.bias)

    def discard_pages(self) -> None:
        self.source.discard_pages()


_Source = _Plane | _Upsampled


def _conv_tile(
    state: _RunState,
    source: _Source,
    conv: nn.Conv2d,
    box: _Box,
    before: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    kernel, stride, padding = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    bounds, pad = (box, (0, 0, 0, 0)) if kernel == 1 else _support(
        box, source.height, source.width, stride, padding,
    )
    x = source.read(bounds, state.device)
    if before is not None:
        x = before(x)
    if any(pad):
        x = F.pad(x, pad)
    return F.conv2d(x, conv.weight, conv.bias, stride=stride)


def _activation(
    state: _RunState, nonlinear: Callable[[Tensor], Tensor], source: _Source,
) -> Callable[[Tensor], Tensor]:
    assert source.moments is not None
    affine = state.finish_norm(source, source.moments)
    dtype = torch.float32 if state.decoder else state.dtype

    def apply(x: Tensor) -> Tensor:
        normalized = _normalize(x, affine, dtype)
        return nonlinear(normalized if state.decoder else normalized.contiguous())

    return apply


def _run_conv(
    state: _RunState,
    source: _Plane,
    conv: nn.Conv2d,
    side: int,
    next_norm: nn.GroupNorm | None = None,
    before: Callable[[Tensor], Tensor] | None = None,
) -> _Plane:
    stride = conv.stride[0]
    h, w = source.height // stride, source.width // stride
    out = state.allocate(conv.out_channels, h, w, next_norm)
    for box in _tiles(h, w, side):
        out.write(box, _conv_tile(state, source, conv, box, before))
        if box[3] == w:
            source.discard_pages()
            out.discard_pages()
    state.count_conv(conv, h, w)
    return out


def _run_resnet(
    state: _RunState, source: _Source, block: nn.Module, side: int, next_norm: nn.GroupNorm | None,
) -> _Plane:
    h, w = source.height, source.width
    virtual = isinstance(source, _Upsampled)
    if virtual:
        # The only standalone tiled moment pass. Recompute this upsample
        # while consuming its halos to avoid storing the full plane.
        source.moments = _Moments(source.norm, state.device)
        for box in _tiles(h, w, side):
            source.moments.add(source.read(box, state.device))
            if box[3] == w:
                source.discard_pages()
    before = _activation(state, block.nonlinearity, source)
    shortcut = block.conv_shortcut
    reuse = isinstance(source, _Plane) and shortcut is None
    out = source if reuse else state.allocate(block.conv2.out_channels, h, w)
    middle = state.allocate(block.conv1.out_channels, h, w, block.norm2)
    for box in _tiles(h, w, side):
        if virtual:
            bounds, pad = _support(box, h, w)
            x = source.read(bounds, state.device)
            oy, ox = box[0] - bounds[0], box[2] - bounds[2]
            core = _cut(x, (oy, oy + box[1] - box[0], ox, ox + box[3] - box[2]))
            out.write(box, shortcut(core) if shortcut is not None else core)
            x = before(x)
            if any(pad):
                x = F.pad(x, pad)
            value = F.conv2d(x, block.conv1.weight, block.conv1.bias)
            del x, core
            state.count_conv(source.conv, bounds[1] - bounds[0], bounds[3] - bounds[2], "recomputed_upsample")
        else:
            value = _conv_tile(state, source, block.conv1, box, before)
        middle.write(box, value)
        del value
        if box[3] == w:
            middle.discard_pages()
            source.discard_pages()
            if virtual:
                out.discard_pages()
    if virtual:
        state.release(source.source)
    skip_source = out if virtual else source
    assert isinstance(skip_source, _Plane)
    out.moments = None if next_norm is None else _Moments(next_norm, state.device)
    before = _activation(state, block.nonlinearity, middle)
    for box in _tiles(h, w, side):
        value = _conv_tile(state, middle, block.conv2, box, before)
        skip = skip_source.read(box, state.device)
        if shortcut is not None and not virtual:
            skip = shortcut(skip)
        out.write(box, (skip + value) / block.output_scale_factor)
        del skip, value
        if box[3] == w:
            middle.discard_pages()
            skip_source.discard_pages()
            if out is not skip_source:
                out.discard_pages()
    state.count_resnet(block, h, w)
    state.release(middle)
    if skip_source is not out:
        state.release(skip_source)
    return out


def _run_upsample(
    state: _RunState, source: _Plane, conv: nn.Conv2d, side: int, next_norm: nn.GroupNorm, *, virtual: bool,
) -> _Source:
    up = _Upsampled(source, conv, next_norm)
    state.count_conv(conv, up.height, up.width)
    if virtual:
        return up
    out = state.allocate(conv.out_channels, up.height, up.width, next_norm)
    for box in _tiles(up.height, up.width, side):
        out.write(box, up.read(box, state.device))
        if box[3] == up.width:
            source.discard_pages()
            out.discard_pages()
    state.release(source)
    return out


def _inference_module(module: nn.Module, device: torch.device, dtype: torch.dtype) -> nn.Module:
    param = next(module.parameters())
    if param.device == device and param.dtype == dtype and not module.training:
        return module
    # Only the required submodule is copied when precision or placement differs.
    return copy.deepcopy(module).eval().to(device=device, dtype=dtype)


def _device(vae: nn.Module, device: str | torch.device | None) -> torch.device:
    dev = next(vae.parameters()).device if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"Exact tiled FLUX.2 supports CPU and CUDA, got {dev}")
    return dev


def _validate_latent(z: Tensor) -> None:
    if z.ndim != 4 or z.shape[:2] != (1, 32) or min(z.shape[-2:]) < 1:
        raise ValueError("Expected a raw FLUX.2 latent with shape [1, 32, H, W]")


class ExactTiledFluxVAE:
    """Batch-one FLUX.2 encoder/decoder with image-wide GroupNorm and attention.

    Input tensors are RGB NCHW in [-1, 1], or uint8 RGB in HWC/CHW/NCHW.
    NumPy RGB images use HWC. The encoder returns raw float32 latents on the
    compute device. The decoder returns unclamped samples, which may reside
    on the CPU when the output plane exceeds the available GPU plane budget.

    Budgets cover live feature planes. The full stride-8 native stages,
    attention, and per-tile scratch use working memory beyond these budgets.
    A VAE already in inference mode at the required precision and placement
    (FP16 encoder on CUDA, FP32 otherwise) avoids per-call submodule copies.
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        device: str | torch.device | None = None,
        tile: int = 512,
        max_tile: int = 512,
        gpu_budget_mib: _Budget = "auto",
        cpu_budget_mib: float = 1536,
        chunk_q: int = 2048,
        spill_dir: str | Path | None = None,
    ) -> None:
        if (not all(isinstance(size, int) for size in (tile, max_tile, chunk_q))
                or tile < 8 or tile % 8 or max_tile < 1 or chunk_q < 1):
            raise ValueError("tile must be a positive multiple of 8 and max_tile and chunk_q must be positive")
        for budget in (cpu_budget_mib, 0 if gpu_budget_mib == "auto" else gpu_budget_mib):
            if not math.isfinite(budget) or budget < 0:
                raise ValueError("Plane budgets must be finite and non-negative")
        self.vae = vae
        self.device = _device(vae, device)
        self.tile = tile
        self.max_tile = max_tile
        self.chunk_q = chunk_q
        self.gpu_budget_mib = gpu_budget_mib if self.device.type == "cuda" else 0
        self.cpu_budget_mib = cpu_budget_mib
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None

    def _state(self, *, decoder: bool) -> _RunState:
        on_cuda = self.device.type == "cuda"
        dtype = (torch.bfloat16 if decoder else torch.float16) if on_cuda else torch.float32
        gpu_budget = self.gpu_budget_mib
        if gpu_budget == "auto":
            free, total = torch.cuda.mem_get_info(self.device)
            # The CUDA driver counts PyTorch's reusable cache as occupied.
            cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            available = min(total, free + max(cached, 0))
            gpu_budget = max(512, min(4096, 0.45 * available / _MIB))
        return _RunState(self.device, dtype, decoder, int(gpu_budget * _MIB),
                         int(self.cpu_budget_mib * _MIB), self.spill_dir)

    def _side(self, stride: int) -> int:
        return min(self.max_tile, self.tile // stride)

    @torch.inference_mode()
    def encode(self, x: Tensor | np.ndarray) -> Tensor:
        """Encode a positive multiple-of-8 RGB image to the posterior mean."""
        if isinstance(x, np.ndarray):
            if x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError("Expected an HWC NumPy RGB image")
            x = torch.from_numpy(np.array(x, copy=True)).permute(2, 0, 1)
        elif x.ndim == 3 and x.dtype == torch.uint8 and x.shape[0] != 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4 or x.shape[:2] != (1, 3) or any(s < 8 or s % 8 for s in x.shape[-2:]):
            raise ValueError("Expected one RGB image with positive height and width divisible by 8")
        if x.dtype == torch.uint8:
            x = x.float().div_(127.5).sub_(1.0)
        elif not x.is_floating_point():
            raise ValueError("Expected uint8 pixels or floating-point RGB samples in [-1, 1]")

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        enc = _inference_module(self.vae.encoder, self.device, dtype)
        quant = self.vae.quant_conv
        if quant is not None:
            quant = _inference_module(quant, self.device, dtype)
        state = self._state(decoder=False)
        try:
            with torch.autocast(self.device.type, enabled=False):
                return self._encode(state, enc, quant, x.to(dtype))
        finally:
            state.close()

    def _encode(self, state: _RunState, enc: nn.Module, quant: nn.Module | None, x: Tensor) -> Tensor:
        first, *_, bottleneck = enc.down_blocks
        p = _run_conv(state, _Plane(x), enc.conv_in, self._side(1), first.resnets[0].norm1)
        for i, block in enumerate(enc.down_blocks[:-1]):
            side = self._side(2**i)
            for j, resnet in enumerate(block.resnets):
                next_norm = block.resnets[j + 1].norm1 if j + 1 < len(block.resnets) else None
                # Downsample has no normalization. The last residual can
                # hand off directly to it without accumulating moments.
                p = _run_resnet(state, p, resnet, side, next_norm)
            down, = block.downsamplers
            next_block = enc.down_blocks[i + 1]
            downsampled = _run_conv(state, p, down.conv, self._side(2**(i + 1)), next_block.resnets[0].norm1)
            state.release(p)
            p = downsampled

        h, w = p.height, p.width
        assert p.moments is not None
        norm = bottleneck.resnets[0].norm1
        frozen = state.finish_norm(p, p.moments)
        x = p.tensor.to(state.device, copy=p.mapping is not None)
        state.release(p)
        for resnet in bottleneck.resnets:
            state.count_resnet(resnet, h, w)
        with _GroupNormMode({"first": norm}, state.dtype, frozen={"first": frozen}):
            x = bottleneck(x)
        state.count_mid(enc.mid_block, h, w)
        x = _run_mid_block(enc.mid_block, x, self.chunk_q)
        state.count_conv(enc.conv_out, h, w)
        x = enc.conv_out(enc.conv_act(enc.conv_norm_out(x)))
        if quant is not None:
            state.count_conv(quant, h, w)
            x = quant(x)
        return x.chunk(2, dim=1)[0].to(torch.float32, copy=True)

    @overload
    def decode(self, z: Tensor, *, return_gn_stats: Literal[False] = False) -> Tensor: ...

    @overload
    def decode(self, z: Tensor, *, return_gn_stats: Literal[True]) -> tuple[Tensor, GNStats]: ...

    @torch.inference_mode()
    def decode(self, z: Tensor, *, return_gn_stats: bool = False) -> Tensor | tuple[Tensor, GNStats]:
        """Decode raw latents. Optionally return every decoder GroupNorm's channel-wise (a, b)."""
        _validate_latent(z)
        dec = _inference_module(self.vae.decoder, self.device, torch.float32)
        post_quant = self.vae.post_quant_conv
        if post_quant is not None:
            post_quant = _inference_module(post_quant, self.device, torch.float32)
        state = self._state(decoder=True)
        norms = _named_norms(dec) if return_gn_stats else {}
        if return_gn_stats:
            state.norm_names = {norm: name for name, norm in norms.items()}
            state.gn_stats = {}
        try:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                result = self._decode(state, dec, post_quant, z, norms)
            if state.gn_stats is not None:
                if state.gn_stats.keys() != norms.keys():
                    raise RuntimeError(f"Missing decoder GroupNorm stats: {norms.keys() - state.gn_stats.keys()}")
                return result, state.gn_stats
            return result
        finally:
            state.close()

    def _decode(
        self, state: _RunState, dec: nn.Module, post_quant: nn.Module | None, z: Tensor,
        norms: Mapping[str, nn.GroupNorm],
    ) -> Tensor:
        h, w = z.shape[-2:]
        x = z.to(state.device, dtype=torch.float32)
        if post_quant is not None:
            state.count_conv(post_quant, h, w)
            x = post_quant(x)
        state.count_conv(dec.conv_in, h, w)
        x = dec.conv_in(x)
        state.count_mid(dec.mid_block, h, w)
        native_norms = (name for name in norms if name.startswith(("mid_block.", "up_blocks.0.")))
        mode = (_GroupNormMode({name: norms[name] for name in native_norms}, torch.float32, capture=state.gn_stats)
                if state.gn_stats is not None else nullcontext())
        first = dec.up_blocks[0]
        with mode:
            x = _run_mid_block(dec.mid_block, x, self.chunk_q)
            for resnet in first.resnets:
                state.count_resnet(resnet, h, w)
                x = resnet(x, None)
        p: _Source = state.own(_Plane(x))
        del x

        for i, block in enumerate(dec.up_blocks):
            stride = 8 // 2**i
            if i > 0:
                for j, resnet in enumerate(block.resnets):
                    if j + 1 < len(block.resnets):
                        next_norm = block.resnets[j + 1].norm1
                    else:
                        next_norm = None if block.upsamplers else dec.conv_norm_out
                    p = _run_resnet(state, p, resnet, self._side(stride), next_norm)
            if block.upsamplers:
                up, = block.upsamplers
                assert isinstance(p, _Plane)
                p = _run_upsample(state, p, up.conv, self._side(stride // 2),
                                  dec.up_blocks[i + 1].resnets[0].norm1, virtual=stride == 2)
        assert isinstance(p, _Plane)
        before = _activation(state, dec.conv_act, p)
        out = _run_conv(state, p, dec.conv_out, self._side(1), before=before)
        return out.tensor if out.mapping is None else out.tensor.clone()


@torch.inference_mode()
def decode_window_with_global_gn(
    vae: nn.Module, z_window: Tensor, gn_stats: Mapping[str, _Affine], *, device: str | torch.device | None = None,
) -> Tensor:
    """Native decode of a latent window using frozen image-wide GroupNorm.

    Stats come from `decode(..., return_gn_stats=True)`. The forward uses the
    window's own attention context. Compare candidates to the same unchanged
    halo window to isolate edits from shifts caused by cropping the context.
    """
    _validate_latent(z_window)
    dev = _device(vae, device)
    dec = _inference_module(vae.decoder, dev, torch.float32)
    post_quant = vae.post_quant_conv
    if post_quant is not None:
        post_quant = _inference_module(post_quant, dev, torch.float32)
    norms = _named_norms(dec)
    if norms.keys() != gn_stats.keys():
        raise ValueError(f"GroupNorm stats do not match decoder modules: {norms.keys() ^ gn_stats.keys()}")
    frozen = {}
    for name, (a, b) in gn_stats.items():
        if a.shape != (norms[name].num_channels,) or a.shape != b.shape:
            raise ValueError(f"Expected channel affine vectors for decoder GroupNorm {name!r}")
        frozen[name] = a.to(dev), b.to(dev)
    z = z_window.to(dev, dtype=torch.float32)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        with _GroupNormMode(norms, torch.float32, frozen=frozen):
            if dec is vae.decoder and post_quant is vae.post_quant_conv and not getattr(vae, "use_tiling", False):
                return vae.decode(z).sample
            return dec(post_quant(z) if post_quant is not None else z)


def encode_exact_tiled(vae: nn.Module, x: Tensor | np.ndarray, **options) -> Tensor:
    """Convenience wrapper for `ExactTiledFluxVAE(...).encode(x)`."""
    return ExactTiledFluxVAE(vae, **options).encode(x)


def decode_exact_tiled(
    vae: nn.Module, z: Tensor, *, return_gn_stats: bool = False, **options,
) -> Tensor | tuple[Tensor, GNStats]:
    """Convenience wrapper for `ExactTiledFluxVAE(...).decode(z)`."""
    tiled = ExactTiledFluxVAE(vae, **options)
    return tiled.decode(z, return_gn_stats=True) if return_gn_stats else tiled.decode(z)
