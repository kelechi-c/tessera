"""Versioned binary token container (`TSR2`) and token API contract for the rectangular Tessera codec.

Token Contract (for standalone encode/decode and downstream generators such as `tessera.gen.rect_model`)
--------------------------------------------------------------------------------------------------------
An encoded 512 x 512 Tessera image consists solely of:
  1. A valid 2:1 k-d tree tiling `shape_map` of the 64 x 64 FLUX.2 latent grid (4 to 1,024 disjoint leaves from
     `tessera.latent.rect.SHAPES`, where each latent cell = 8 x 8 px).
  2. One lossless `float32` 128-d wire token per leaf (`tokens` of shape `[N, 128]`), in normalized latent space
     `zn = (z - mean) / std`:
       - Finest `(2, 2)` leaves (16 x 16 px): exact 128-d flattened `(32, 2, 2)` normalized latent patch, identical to
         `RL.project_rect_tokens(zn, 2, 2, None)` (copy-through on decode without storing source `z`).
       - Coarse leaves `(sh, sw) != (2, 2)`: exact 128-d wire token produced by
         `RTD.encode_rect_pyramid_td(zn, basis, tiledown=td)[(sh, sw)]` (linear basis projection + `RectTileDown`
         residual when the checkpoint includes `tiledown`).
  3. Checkpoint identification (`ckpt_sha256`, 32 raw SHA-256 bytes of the tokenizer checkpoint) and deterministic
     TileUp decode parameters (`seed`, `tu_spike_k`, `tu_spike_min_area`).
  4. Optional informational JSON metadata (`meta`, UTF-8 JSON object, <= 64 KiB). No pixels, full latents, or source
     file paths are required or embedded for decoding.

Canonical Ordering
------------------
  - `ORDER_RASTER_TOP_LEFT = 0`: leaves and rows of `tokens` `[N, 128]` are ordered in canonical top-left raster order
    matching `tessera.latent.rect.leaves_rect(shape_map)` — i.e., sorted by `(r0, c0)` ascending.
  - Breadth-first k-d tree node order (used by `BreadthGenRect` / `RL.kdtree_trace`) is mapped deterministically to and
    from `ORDER_RASTER_TOP_LEFT` via `leaf_order_to_kdtree_leaves` and `kdtree_leaves_to_leaf_order`.

Binary Layout (`TSR2`, Version 1, little-endian)
------------------------------------------------
  Fixed 66-byte header (`struct.Struct("<4sHHHHBBHHHHIfI32s")`):
    - magic             : 4s   = b"TSR2"
    - version           : u16  = 1
    - flags             : u16  = 0 (reserved; non-zero rejected)
    - grid_side         : u16  = 64
    - token_dim         : u16  = 128
    - dtype_code        : u8   = 0 (IEEE-754 little-endian float32)
    - order_code        : u8   = 0 (`ORDER_RASTER_TOP_LEFT`)
    - n_tokens          : u16  = N in [4, 1024]
    - tree_n_bits       : u16  = exact bit length of the k-d tree bitstream
    - tree_n_bytes      : u16  = ceil(tree_n_bits / 8)
    - tu_spike_min_area : u16  = TileUp spike clamp minimum tile area in latent cells
    - seed              : u32  = TileUp noise seed
    - tu_spike_k        : f32  = TileUp spike clamp multiplier (0.0 = disabled)
    - meta_n_bytes      : u32  = byte length of UTF-8 JSON object (0..65536)
    - ckpt_sha256       : 32s  = raw SHA-256 digest of the tokenizer checkpoint
  Followed immediately by:
    - meta_bytes  : `meta_n_bytes` bytes (UTF-8 JSON object; empty if `meta_n_bytes == 0`)
    - tree_bytes  : `tree_n_bytes` bytes (MSB-first k-d tree bitstream; unused pad bits MUST be 0)
    - token_bytes : `n_tokens * token_dim * 4` bytes (contiguous `<f4` `[N, 128]`, all values finite)

Binary Layout (`TSR2`, Version 2, little-endian)
------------------------------------------------
  68-byte header with the original image dimensions, block grid dimensions, padding mode and shared
  checkpoint/TileUp settings. Its optional JSON object is file metadata. If flag 1 is set, the JSON
  wraps file metadata under "grid" and per-block metadata under "blocks".
  Row-major blocks share v1 tree and token encoding. After the JSON come all 6-byte block
  descriptors (token count, tree bit count, tree byte count), then all trees, then all tokens.

k-d Tree Bitstream (`encode_kdtree_bits` / `decode_kdtree_bits`)
----------------------------------------------------------------
  Breadth-first traversal starting from the 4 roots `[(0,0,32,32), (0,32,32,32), (32,0,32,32), (32,32,32,32)]`:
    - `(2, 2)` node: implicit leaf, 0 bits.
    - Square `(s, s)` with `s > 2`: 2 bits (`00` = leaf, `01` = split-H, `10` = split-V; `11` invalid).
    - Rectangular `(sh, sw)` with `sh != sw`: 1 bit (`0` = leaf, `1` = split along longer axis).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import struct
from typing import Any

import numpy as np
import torch

from tessera.latent import rect as RL
from tessera.latent import rect_tiledown as RTD
from tessera.rect.search import spike_clamp_resid

MAGIC = b"TSR2"
FORMAT_VERSION = 1
FORMAT_VERSION_GRID = 2
GRID_SIDE = RL.SIDE              # 64 FLUX.2 latent cells per side (512 px)
TOKEN_DIM = RL.TOK               # 128 floats per leaf token
DTYPE_FLOAT32 = 0
ORDER_RASTER_TOP_LEFT = 0
MAX_META_BYTES = 65536

DEFAULT_SEED = 0
DEFAULT_SPIKE_K = 3.0
DEFAULT_SPIKE_MIN_AREA = 64

_HEADER = struct.Struct("<4sHHHHBBHHHHIfI32s")
HEADER_SIZE = _HEADER.size       # 66 bytes

# v2 uses the v1 token dtype, dimensions, tree encoding and raster leaf order.
# Header: magic, version, flags, original H/W, grid H/W, padding, order,
# min spike area, seed, spike k, JSON byte count, and checkpoint digest.
# Layout: header, JSON, all block descriptors, all trees, all float32 tokens.
_GRID_HEADER = struct.Struct("<4sHHIIHHBBHIfI32s")
_GRID_BLOCK = struct.Struct("<HHH")  # token count, tree bits, tree bytes
_GRID_BLOCK_META = 1
_PAD_MODES = ("none", "reflect")
_BLOCK_PIXELS = GRID_SIDE * 8

_ROOTS = ((0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32))
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


class TokenFormatError(ValueError):
    """Raised when a `.tsr` binary token stream is malformed, unsupported, or fails validation."""


class CheckpointMismatchError(ValueError):
    """Raised when the loaded tokenizer checkpoint does not match a token file's recorded SHA-256."""


def file_sha256(path: str | pathlib.Path) -> str:
    """Compute (with mtime/size cache) the 64-character lowercase hex SHA-256 of a file."""
    p = pathlib.Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _SHA_CACHE.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


def _normalize_sha256_bytes(ckpt_sha256: str | bytes) -> bytes:
    if isinstance(ckpt_sha256, bytes):
        if len(ckpt_sha256) != 32:
            raise TokenFormatError(f"ckpt_sha256 bytes must be 32 bytes, got {len(ckpt_sha256)}")
        return ckpt_sha256
    if isinstance(ckpt_sha256, str):
        s = ckpt_sha256.strip().lower()
        if len(s) != 64:
            raise TokenFormatError(f"ckpt_sha256 hex string must be 64 hex chars, got length {len(s)}")
        try:
            return bytes.fromhex(s)
        except ValueError as exc:
            raise TokenFormatError(f"Invalid ckpt_sha256 hex string: {ckpt_sha256!r}") from exc
    raise TokenFormatError(f"ckpt_sha256 must be 64-char hex str or 32-byte bytes, got {type(ckpt_sha256).__name__}")


def _normalize_shape_map(shape_map: torch.Tensor) -> torch.Tensor:
    sm = torch.as_tensor(shape_map, dtype=torch.long).detach().cpu()
    if sm.ndim == 4:
        if sm.shape[0] != 1:
            raise TokenFormatError(f"Expected single-image shape_map [2,64,64] or [1,2,64,64], got {tuple(sm.shape)}")
        sm = sm[0]
    if tuple(sm.shape) != (2, GRID_SIDE, GRID_SIDE):
        raise TokenFormatError(f"Expected shape_map shape (2, {GRID_SIDE}, {GRID_SIDE}), got {tuple(sm.shape)}")
    if not RL.validate_rect(sm):
        raise TokenFormatError("shape_map is not a valid disjoint 2:1 k-d tree tiling of the 64x64 grid")
    return sm.contiguous()


@dataclasses.dataclass
class RectTokenFile:
    """In-memory representation of a persisted Tessera `TSR2` token file."""

    shape_map: torch.Tensor                                 # [2, 64, 64] int64
    leaves: tuple[tuple[int, int, int, int], ...]           # N tuples (r0, c0, sh, sw) in ORDER_RASTER_TOP_LEFT
    tokens: torch.Tensor                                    # [N, 128] float32 normalized wire tokens
    ckpt_sha256: str                                        # 64-char lowercase hex SHA-256 of tokenizer ckpt
    seed: int = DEFAULT_SEED
    tu_spike_k: float = DEFAULT_SPIKE_K
    tu_spike_min_area: int = DEFAULT_SPIKE_MIN_AREA
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    version: int = FORMAT_VERSION
    dtype_code: int = DTYPE_FLOAT32
    order_code: int = ORDER_RASTER_TOP_LEFT

    def __post_init__(self) -> None:
        self.shape_map = _normalize_shape_map(self.shape_map)
        expected_leaves = tuple(RL.leaves_rect(self.shape_map))
        self.leaves = tuple(tuple(int(v) for v in t) for t in self.leaves)
        if self.leaves != expected_leaves:
            raise TokenFormatError("leaves do not match canonical ORDER_RASTER_TOP_LEFT leaves of shape_map")
        tok = torch.as_tensor(self.tokens, dtype=torch.float32).detach().cpu().contiguous()
        if tok.ndim != 2 or tok.shape != (len(self.leaves), TOKEN_DIM):
            raise TokenFormatError(
                f"Expected tokens shape ({len(self.leaves)}, {TOKEN_DIM}), got {tuple(tok.shape)}"
            )
        if not bool(torch.isfinite(tok).all()):
            raise TokenFormatError("tokens contain NaN or Inf values")
        self.tokens = tok
        self.ckpt_sha256 = _normalize_sha256_bytes(self.ckpt_sha256).hex()
        if self.version != FORMAT_VERSION:
            raise TokenFormatError(f"Unsupported format version: {self.version}")
        if self.dtype_code != DTYPE_FLOAT32:
            raise TokenFormatError(f"Unsupported token dtype_code: {self.dtype_code}")
        if self.order_code != ORDER_RASTER_TOP_LEFT:
            raise TokenFormatError(f"Unsupported leaf order_code: {self.order_code}")
        if self.seed < 0 or self.seed >= (1 << 32):
            raise TokenFormatError(f"seed must be in [0, 2^32 - 1], got {self.seed}")
        self.seed = int(self.seed)
        if not math.isfinite(float(self.tu_spike_k)) or float(self.tu_spike_k) < 0.0:
            raise TokenFormatError(f"tu_spike_k must be finite and >= 0, got {self.tu_spike_k}")
        self.tu_spike_k = float(self.tu_spike_k)
        if self.tu_spike_min_area < 0 or self.tu_spike_min_area > 65535:
            raise TokenFormatError(f"tu_spike_min_area must be in [0, 65535], got {self.tu_spike_min_area}")
        self.tu_spike_min_area = int(self.tu_spike_min_area)
        if not isinstance(self.meta, dict):
            raise TokenFormatError(f"meta must be a dict, got {type(self.meta).__name__}")

    @property
    def n_tokens(self) -> int:
        return len(self.leaves)


def _validate_grid_dimensions(orig_height: int, orig_width: int, grid_h: int, grid_w: int, pad_mode: str) -> None:
    if not all(isinstance(size, int) and 1 <= size <= 65535 for size in (grid_h, grid_w)):
        raise TokenFormatError("Grid height and width must be in [1, 65535]")
    if not all(isinstance(size, int) and size > 0 for size in (orig_height, orig_width)):
        raise TokenFormatError("Original grid image dimensions must be positive integers")
    if pad_mode not in _PAD_MODES:
        raise TokenFormatError(f"Unsupported grid padding mode {pad_mode!r}")
    for size, blocks in ((orig_height, grid_h), (orig_width, grid_w)):
        if (size + _BLOCK_PIXELS - 1) // _BLOCK_PIXELS != blocks:
            raise TokenFormatError("Original image dimensions do not match the 512-pixel block grid")
        if pad_mode == "none" and size != blocks * _BLOCK_PIXELS:
            raise TokenFormatError("Unpadded grid requires original dimensions divisible by 512")


@dataclasses.dataclass
class RectGridTokenFile:
    """TSR2 v2 image grid with v1 block layouts and shared codec settings.

    Blocks appear in row-major order. Each block uses the grid's checkpoint,
    seed, and TileUp spike settings. File and optional block metadata are JSON.
    """

    orig_height: int
    orig_width: int
    grid_h: int
    grid_w: int
    pad_mode: str
    blocks: list[RectTokenFile]
    ckpt_sha256: str
    seed: int = DEFAULT_SEED
    tu_spike_k: float = DEFAULT_SPIKE_K
    tu_spike_min_area: int = DEFAULT_SPIKE_MIN_AREA
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_grid_dimensions(self.orig_height, self.orig_width, self.grid_h, self.grid_w, self.pad_mode)
        self.ckpt_sha256 = _normalize_sha256_bytes(self.ckpt_sha256).hex()
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise TokenFormatError("Grid seed must be in [0, 2^32 - 1]")
        if not math.isfinite(self.tu_spike_k) or not 0 <= self.tu_spike_k <= np.finfo(np.float32).max:
            raise TokenFormatError("Grid tu_spike_k must be a finite non-negative float32")
        if not isinstance(self.tu_spike_min_area, int) or not 0 <= self.tu_spike_min_area <= 65535:
            raise TokenFormatError("Grid tu_spike_min_area must be in [0, 65535]")
        if not isinstance(self.meta, dict):
            raise TokenFormatError("Grid meta must be a dict")
        if len(self.blocks) != self.n_blocks:
            raise TokenFormatError(f"Grid expects {self.n_blocks} blocks, got {len(self.blocks)}")
        for block in self.blocks:
            if not isinstance(block, RectTokenFile) or (
                block.ckpt_sha256, block.seed, block.tu_spike_k, block.tu_spike_min_area
            ) != (self.ckpt_sha256, self.seed, self.tu_spike_k, self.tu_spike_min_area):
                raise TokenFormatError("Each grid block must share the grid's checkpoint, seed, and spike settings")

    @property
    def n_tokens(self) -> int:
        return sum(block.n_tokens for block in self.blocks)

    @property
    def n_blocks(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def block_tokens(self) -> list[int]:
        return [block.n_tokens for block in self.blocks]


# ---------------------------------------------------------------- k-d tree bitstream ----------------------------------
def encode_kdtree_bits(shape_map: torch.Tensor) -> tuple[bytes, int]:
    """Encode a valid 2:1 k-d tree `shape_map` [2, 64, 64] into `(packed_bytes, n_bits)`."""
    sm = _normalize_shape_map(shape_map)
    nodes = RL.kdtree_trace(sm)
    bits: list[int] = []
    for _r0, _c0, sh, sw, _lv, act in nodes:
        if sh == 2 and sw == 2:
            if act != RL.ACT_LEAF:
                raise TokenFormatError("Internal error: (2, 2) node must be leaf")
            continue
        if sh == sw:
            if act == RL.ACT_LEAF:
                bits.extend((0, 0))
            elif act == RL.ACT_SPLIT_H:
                bits.extend((0, 1))
            elif act == RL.ACT_SPLIT_V:
                bits.extend((1, 0))
            else:
                raise TokenFormatError(f"Invalid square action {act}")
        else:
            if act == RL.ACT_LEAF:
                bits.append(0)
            elif (sw > sh and act == RL.ACT_SPLIT_V) or (sh > sw and act == RL.ACT_SPLIT_H):
                bits.append(1)
            else:
                raise TokenFormatError(f"Invalid rectangular action {act} for shape ({sh}, {sw})")

    n_bits = len(bits)
    out = bytearray((n_bits + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (7 - (i & 7))
    # Sanity round-trip check: every encoded bitstream must reconstruct the exact shape_map
    decoded = decode_kdtree_bits(bytes(out), n_bits)
    if not torch.equal(decoded, sm):
        raise TokenFormatError("shape_map cannot be represented as a 2:1 k-d tree")
    return bytes(out), n_bits


def decode_kdtree_bits(data: bytes, n_bits: int) -> torch.Tensor:
    """Decode `(packed_bytes, n_bits)` into a validated `shape_map` [2, 64, 64] int64."""
    expected_bytes = (int(n_bits) + 7) // 8
    if n_bits < 0 or len(data) != expected_bytes:
        raise TokenFormatError(f"k-d bitstream byte length {len(data)} does not match n_bits={n_bits}")

    # Trailing padding bits in the final byte must all be zero.
    rem = int(n_bits) & 7
    if rem != 0:
        pad_mask = (1 << (8 - rem)) - 1
        if data[-1] & pad_mask:
            raise TokenFormatError("Non-zero trailing padding bits in k-d tree bitstream")

    pos = 0

    def read_bit() -> int:
        nonlocal pos
        if pos >= n_bits:
            raise TokenFormatError("Truncated k-d tree bitstream")
        b = (data[pos >> 3] >> (7 - (pos & 7))) & 1
        pos += 1
        return b

    sm = torch.zeros((2, GRID_SIDE, GRID_SIDE), dtype=torch.long)
    frontier = list(_ROOTS)
    while frontier:
        nxt: list[tuple[int, int, int, int]] = []
        for r0, c0, sh, sw in frontier:
            if sh == 2 and sw == 2:
                act = RL.ACT_LEAF
            elif sh == sw:
                code = (read_bit() << 1) | read_bit()
                if code == 0:
                    act = RL.ACT_LEAF
                elif code == 1:
                    act = RL.ACT_SPLIT_H
                elif code == 2:
                    act = RL.ACT_SPLIT_V
                else:
                    raise TokenFormatError("Invalid 2-bit square split code 0b11 in k-d tree bitstream")
            else:
                bit = read_bit()
                if bit == 0:
                    act = RL.ACT_LEAF
                else:
                    act = RL.ACT_SPLIT_V if sw > sh else RL.ACT_SPLIT_H

            if act == RL.ACT_LEAF:
                sm[0, r0:r0 + sh, c0:c0 + sw] = sh
                sm[1, r0:r0 + sh, c0:c0 + sw] = sw
            elif act == RL.ACT_SPLIT_H:
                nxt.append((r0, c0, sh // 2, sw))
                nxt.append((r0 + sh // 2, c0, sh // 2, sw))
            else:
                nxt.append((r0, c0, sh, sw // 2))
                nxt.append((r0, c0 + sw // 2, sh, sw // 2))
        frontier = nxt

    if pos != n_bits:
        raise TokenFormatError(f"k-d tree bitstream has {n_bits - pos} unconsumed trailing bit(s)")
    if not RL.validate_rect(sm):
        raise TokenFormatError("Decoded k-d tree bitstream did not produce a valid tiling")
    return sm


# ---------------------------------------------------------------- binary serialization --------------------------------
def dumps_tokens(tf: RectTokenFile) -> bytes:
    """Serialize a `RectTokenFile` into lossless `TSR2` v1 bytes."""
    ckpt_bytes = _normalize_sha256_bytes(tf.ckpt_sha256)
    tree_bytes, tree_n_bits = encode_kdtree_bits(tf.shape_map)
    if tf.meta:
        meta_bytes = json.dumps(tf.meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        meta_bytes = b""
    if len(meta_bytes) > MAX_META_BYTES:
        raise TokenFormatError(f"Metadata JSON exceeds maximum {MAX_META_BYTES} bytes ({len(meta_bytes)} bytes)")
    tok_np = np.ascontiguousarray(tf.tokens.detach().cpu().numpy(), dtype="<f4")
    header = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        0,  # flags
        GRID_SIDE,
        TOKEN_DIM,
        DTYPE_FLOAT32,
        ORDER_RASTER_TOP_LEFT,
        tf.n_tokens,
        tree_n_bits,
        len(tree_bytes),
        tf.tu_spike_min_area,
        tf.seed,
        float(tf.tu_spike_k),
        len(meta_bytes),
        ckpt_bytes,
    )
    return header + meta_bytes + tree_bytes + tok_np.tobytes()


def loads_tokens(data: bytes | bytearray | memoryview) -> RectTokenFile:
    """Parse and strictly validate a `TSR2` v1 byte stream into a `RectTokenFile`."""
    buf = bytes(data)
    if len(buf) < HEADER_SIZE:
        raise TokenFormatError(f"Truncated token file: {len(buf)} bytes (minimum header is {HEADER_SIZE} bytes)")

    (
        magic,
        version,
        flags,
        grid_side,
        token_dim,
        dtype_code,
        order_code,
        n_tokens,
        tree_n_bits,
        tree_n_bytes,
        tu_spike_min_area,
        seed,
        tu_spike_k,
        meta_n_bytes,
        ckpt_bytes,
    ) = _HEADER.unpack_from(buf, 0)

    if magic != MAGIC:
        raise TokenFormatError(f"Invalid magic {magic!r}; expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise TokenFormatError(f"Unsupported TSR2 format version {version} (supported: {FORMAT_VERSION})")
    if flags != 0:
        raise TokenFormatError(f"Unsupported non-zero header flags: 0x{flags:04x}")
    if grid_side != GRID_SIDE:
        raise TokenFormatError(f"Unsupported grid_side {grid_side} (expected {GRID_SIDE})")
    if token_dim != TOKEN_DIM:
        raise TokenFormatError(f"Unsupported token_dim {token_dim} (expected {TOKEN_DIM})")
    if dtype_code != DTYPE_FLOAT32:
        raise TokenFormatError(f"Unsupported dtype_code {dtype_code} (expected {DTYPE_FLOAT32} for float32)")
    if order_code != ORDER_RASTER_TOP_LEFT:
        raise TokenFormatError(f"Unsupported order_code {order_code} (expected {ORDER_RASTER_TOP_LEFT})")
    if n_tokens < 4 or n_tokens > (GRID_SIDE // 2) ** 2:
        raise TokenFormatError(f"Invalid n_tokens {n_tokens}; must be in [4, 1024]")
    if tree_n_bytes != (tree_n_bits + 7) // 8:
        raise TokenFormatError(f"Header tree_n_bytes={tree_n_bytes} disagrees with tree_n_bits={tree_n_bits}")
    if meta_n_bytes > MAX_META_BYTES:
        raise TokenFormatError(f"Header meta_n_bytes={meta_n_bytes} exceeds limit {MAX_META_BYTES}")
    if not math.isfinite(tu_spike_k) or tu_spike_k < 0.0:
        raise TokenFormatError(f"Invalid tu_spike_k={tu_spike_k!r} in header")

    token_n_bytes = n_tokens * token_dim * 4
    expected_total = HEADER_SIZE + meta_n_bytes + tree_n_bytes + token_n_bytes
    if len(buf) != expected_total:
        raise TokenFormatError(
            f"Token file size mismatch: got {len(buf)} bytes, expected {expected_total} bytes "
            f"(header={HEADER_SIZE}, meta={meta_n_bytes}, tree={tree_n_bytes}, tokens={token_n_bytes})"
        )

    off = HEADER_SIZE
    meta_raw = buf[off:off + meta_n_bytes]
    off += meta_n_bytes
    tree_raw = buf[off:off + tree_n_bytes]
    off += tree_n_bytes
    tok_raw = buf[off:off + token_n_bytes]

    if meta_n_bytes == 0:
        meta: dict[str, Any] = {}
    else:
        try:
            meta_obj = json.loads(meta_raw.decode("utf-8"))
        except Exception as exc:
            raise TokenFormatError(f"Malformed metadata JSON: {exc}") from exc
        if not isinstance(meta_obj, dict):
            raise TokenFormatError(f"Metadata JSON must be an object, got {type(meta_obj).__name__}")
        meta = meta_obj

    shape_map = decode_kdtree_bits(tree_raw, tree_n_bits)
    leaves = tuple(RL.leaves_rect(shape_map))
    if len(leaves) != n_tokens:
        raise TokenFormatError(f"Header n_tokens={n_tokens} disagrees with decoded layout leaf count {len(leaves)}")

    tok_np = np.frombuffer(tok_raw, dtype="<f4").reshape(n_tokens, token_dim).copy()
    tokens = torch.from_numpy(tok_np.astype(np.float32, copy=False))
    if not bool(torch.isfinite(tokens).all()):
        raise TokenFormatError("Token payload contains NaN or Inf values")

    return RectTokenFile(
        shape_map=shape_map,
        leaves=leaves,
        tokens=tokens,
        ckpt_sha256=ckpt_bytes.hex(),
        seed=int(seed),
        tu_spike_k=float(tu_spike_k),
        tu_spike_min_area=int(tu_spike_min_area),
        meta=meta,
        version=version,
        dtype_code=dtype_code,
        order_code=order_code,
    )


def save_tokens(path: str | pathlib.Path, tf: RectTokenFile) -> pathlib.Path:
    """Write a `RectTokenFile` to disk in `TSR2` v1 binary format."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(dumps_tokens(tf))
    return p


def load_tokens(path: str | pathlib.Path) -> RectTokenFile:
    """Load and validate a `TSR2` v1 binary token file from disk."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Token file not found: {p}")
    return loads_tokens(p.read_bytes())


def dumps_grid_tokens(tf: RectGridTokenFile) -> bytes:
    """Serialize a grid to compact TSR2 v2 bytes."""
    flags = 0
    meta = tf.meta
    if any(block.meta for block in tf.blocks):
        flags = _GRID_BLOCK_META
        meta = {"grid": tf.meta, "blocks": [block.meta for block in tf.blocks]}
    try:
        meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") if meta else b""
    except (ValueError, TypeError) as exc:
        raise TokenFormatError(f"Invalid grid metadata: {exc}") from exc
    if len(meta_bytes) > MAX_META_BYTES:
        raise TokenFormatError(f"Grid metadata exceeds {MAX_META_BYTES} bytes")
    header = _GRID_HEADER.pack(
        MAGIC, FORMAT_VERSION_GRID, flags, tf.orig_height, tf.orig_width, tf.grid_h, tf.grid_w,
        _PAD_MODES.index(tf.pad_mode), ORDER_RASTER_TOP_LEFT, tf.tu_spike_min_area, tf.seed, tf.tu_spike_k,
        len(meta_bytes), _normalize_sha256_bytes(tf.ckpt_sha256),
    )
    descriptors, trees, payloads = [], [], []
    for block in tf.blocks:
        tree, n_bits = encode_kdtree_bits(block.shape_map)
        tokens = np.ascontiguousarray(block.tokens.detach().cpu().numpy(), dtype="<f4")
        if tokens.shape != (block.n_tokens, TOKEN_DIM) or not np.isfinite(tokens).all():
            raise TokenFormatError("Invalid grid block token payload")
        descriptors.append(_GRID_BLOCK.pack(block.n_tokens, n_bits, len(tree)))
        trees.append(tree)
        payloads.append(tokens.tobytes())
    return b"".join((header, meta_bytes, *descriptors, *trees, *payloads))


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number {value}")
    return number


def loads_grid_tokens(data: bytes | bytearray | memoryview) -> RectGridTokenFile:
    """Parse and strictly validate a TSR2 v2 image grid."""
    buf = memoryview(data)
    if not buf.c_contiguous:
        buf = memoryview(bytes(buf))
    buf = buf.cast("B")
    if len(buf) < _GRID_HEADER.size:
        raise TokenFormatError("Truncated TSR2 grid header")
    (magic, version, flags, orig_h, orig_w, grid_h, grid_w, pad_code, order, spike_area,
     seed, spike_k, meta_size, ckpt_bytes) = _GRID_HEADER.unpack_from(buf)
    if magic != MAGIC or version != FORMAT_VERSION_GRID:
        raise TokenFormatError(f"Invalid grid magic or version: {magic!r}, {version}")
    if flags & ~_GRID_BLOCK_META or pad_code >= len(_PAD_MODES) or order != ORDER_RASTER_TOP_LEFT:
        raise TokenFormatError("Unsupported grid flags, padding mode, or leaf order")
    if meta_size > MAX_META_BYTES or not math.isfinite(spike_k) or spike_k < 0:
        raise TokenFormatError("Invalid grid metadata length or spike settings")
    pad_mode = _PAD_MODES[pad_code]
    _validate_grid_dimensions(orig_h, orig_w, grid_h, grid_w, pad_mode)

    desc_start = _GRID_HEADER.size + meta_size
    n_blocks = grid_h * grid_w
    tree_offset = desc_start + n_blocks * _GRID_BLOCK.size
    if len(buf) < tree_offset + n_blocks * (1 + 4 * TOKEN_DIM * 4):
        raise TokenFormatError("Truncated TSR2 grid descriptors or payload")
    descriptors = list(_GRID_BLOCK.iter_unpack(buf[desc_start:tree_offset]))
    for n_tokens, tree_bits, tree_bytes in descriptors:
        if not 4 <= n_tokens <= (GRID_SIDE // 2) ** 2 or not 8 <= tree_bits <= 4096:
            raise TokenFormatError("Invalid grid block token or tree bit count")
        if tree_bytes != (tree_bits + 7) // 8:
            raise TokenFormatError("Grid block tree byte count disagrees with tree bit count")
    token_offset = tree_offset + sum(n_bytes for _, _, n_bytes in descriptors)
    expected_size = token_offset + sum(n_tokens for n_tokens, _, _ in descriptors) * TOKEN_DIM * 4
    if len(buf) != expected_size:
        raise TokenFormatError(f"Grid file size mismatch: got {len(buf)} bytes, expected {expected_size}")

    meta = {}
    if meta_size:
        try:
            meta = json.loads(bytes(buf[_GRID_HEADER.size:desc_start]).decode("utf-8"),
                              parse_constant=_finite_json_number, parse_float=_finite_json_number)
        except (ValueError, RecursionError) as exc:
            raise TokenFormatError(f"Malformed grid metadata JSON: {exc}") from exc
        if not isinstance(meta, dict):
            raise TokenFormatError("Grid metadata JSON must be an object")
    block_meta = None
    if flags & _GRID_BLOCK_META:
        block_meta = meta.get("blocks")
        if (not isinstance(meta.get("grid"), dict) or not isinstance(block_meta, list)
                or len(block_meta) != n_blocks or not all(isinstance(item, dict) for item in block_meta)):
            raise TokenFormatError("Invalid grid block metadata")
        meta = meta["grid"]

    shared = dict(ckpt_sha256=ckpt_bytes.hex(), seed=seed, tu_spike_k=spike_k, tu_spike_min_area=spike_area)
    blocks = []
    for i, (n_tokens, tree_bits, tree_bytes) in enumerate(descriptors):
        shape_map = decode_kdtree_bits(buf[tree_offset:tree_offset + tree_bytes], tree_bits)
        tree_offset += tree_bytes
        leaves = tuple(RL.leaves_rect(shape_map))
        if len(leaves) != n_tokens:
            raise TokenFormatError(f"Grid block {i} token count disagrees with decoded layout")
        values = np.frombuffer(buf, dtype="<f4", count=n_tokens * TOKEN_DIM, offset=token_offset)
        token_offset += values.nbytes
        tokens = torch.from_numpy(values.astype(np.float32).reshape(n_tokens, TOKEN_DIM))
        blocks.append(RectTokenFile(shape_map, leaves, tokens, **shared,
                                    meta={} if block_meta is None else block_meta[i]))
    return RectGridTokenFile(orig_h, orig_w, grid_h, grid_w, pad_mode, blocks, **shared, meta=meta)


def save_grid_tokens(path: str | pathlib.Path, tf: RectGridTokenFile) -> pathlib.Path:
    """Write a TSR2 v2 image grid to disk."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(dumps_grid_tokens(tf))
    return p


def load_grid_tokens(path: str | pathlib.Path) -> RectGridTokenFile:
    """Load a TSR2 v2 image grid from disk."""
    return loads_grid_tokens(pathlib.Path(path).read_bytes())


def load_any_tokens(path: str | pathlib.Path) -> RectTokenFile | RectGridTokenFile:
    """Load a TSR2 file by dispatching on the version in its header."""
    data = pathlib.Path(path).read_bytes()
    if len(data) < 6 or data[:4] != MAGIC:
        raise TokenFormatError("Missing or invalid TSR2 magic/version")
    version, = struct.unpack_from("<H", data, 4)
    if version == FORMAT_VERSION:
        return loads_tokens(data)
    if version == FORMAT_VERSION_GRID:
        return loads_grid_tokens(data)
    raise TokenFormatError(f"Unsupported TSR2 format version {version}")


def verify_checkpoint_sha256(tf: RectTokenFile | RectGridTokenFile, ckpt_path: str | pathlib.Path) -> str:
    """Verify that `ckpt_path` matches `tf.ckpt_sha256`; returns the digest or raises `CheckpointMismatchError`."""
    actual = file_sha256(ckpt_path)
    if actual != tf.ckpt_sha256:
        hint = f" ({tf.meta.get('ckpt_name')})" if tf.meta.get("ckpt_name") else ""
        raise CheckpointMismatchError(
            f"Tokenizer checkpoint mismatch: token file expects sha256={tf.ckpt_sha256[:16]}...{hint}, "
            f"but loaded checkpoint {ckpt_path} has sha256={actual[:16]}..."
        )
    return actual


# ---------------------------------------------------------------- token extraction / ordering / decode ----------------
def extract_leaf_tokens(
    rect_pyr_n: dict[tuple[int, int], torch.Tensor],
    shape_map: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor]:
    """Extract `(leaves, tokens [N, 128] float32)` in `ORDER_RASTER_TOP_LEFT` from a normalized token pyramid.

    `rect_pyr_n` is the output of `RTD.encode_rect_pyramid_td(zn, basis, tiledown=td)` for a single image (B=1).
    """
    sm = _normalize_shape_map(shape_map)
    leaves = tuple(RL.leaves_rect(sm))
    rows = []
    for r0, c0, sh, sw in leaves:
        grid = rect_pyr_n[(sh, sw)]
        rows.append(grid[0, r0 // sh, c0 // sw].detach().float().cpu())
    tokens = torch.stack(rows, dim=0).contiguous()
    return leaves, tokens


def leaf_order_to_kdtree_leaves(
    shape_map: torch.Tensor,
    tokens: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor]:
    """Reorder `[N, 128]` tokens from `ORDER_RASTER_TOP_LEFT` to breadth-first k-d leaf order (`RL.kdtree_trace`)."""
    sm = _normalize_shape_map(shape_map)
    raster_leaves = RL.leaves_rect(sm)
    if tokens.ndim != 2 or tokens.shape != (len(raster_leaves), TOKEN_DIM):
        raise TokenFormatError(f"Expected tokens shape ({len(raster_leaves)}, {TOKEN_DIM}), got {tuple(tokens.shape)}")
    idx_by_leaf = {leaf: i for i, leaf in enumerate(raster_leaves)}
    kd_leaves = tuple((r0, c0, sh, sw) for r0, c0, sh, sw, _lv, act in RL.kdtree_trace(sm) if act == RL.ACT_LEAF)
    order = torch.tensor([idx_by_leaf[leaf] for leaf in kd_leaves], dtype=torch.long, device=tokens.device)
    return kd_leaves, tokens.index_select(0, order)


def kdtree_leaves_to_leaf_order(
    shape_map: torch.Tensor,
    kd_tokens: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor]:
    """Reorder `[N, 128]` tokens from breadth-first k-d leaf order back to canonical `ORDER_RASTER_TOP_LEFT`."""
    sm = _normalize_shape_map(shape_map)
    kd_leaves = [(r0, c0, sh, sw) for r0, c0, sh, sw, _lv, act in RL.kdtree_trace(sm) if act == RL.ACT_LEAF]
    if kd_tokens.ndim != 2 or kd_tokens.shape != (len(kd_leaves), TOKEN_DIM):
        raise TokenFormatError(f"Expected tokens shape ({len(kd_leaves)}, {TOKEN_DIM}), got {tuple(kd_tokens.shape)}")
    idx_by_leaf = {leaf: i for i, leaf in enumerate(kd_leaves)}
    raster_leaves = tuple(RL.leaves_rect(sm))
    order = torch.tensor([idx_by_leaf[leaf] for leaf in raster_leaves], dtype=torch.long, device=kd_tokens.device)
    return raster_leaves, kd_tokens.index_select(0, order)


@torch.no_grad()
def tokens_to_rect_canvas(
    shape_map: torch.Tensor,
    tokens: torch.Tensor,
    basis_dev: dict[tuple[int, int], dict[str, Any]],
    *,
    tilelift=None,
    floor: float = 0.20,
    tau: float = 0.20,
    border_open: bool = False,
    gate_scale: str = "token",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `(canvas, mask, shape_feat, pos, ac_gate)` solely from `(shape_map, leaf_tokens)` (no source `z`).

    Scatters the persisted `[N, 128]` leaf tokens into sparse per-shape grids matching `RTD.build_rect_canvas_tl`
    so every active tile's lift and AC gate matches the dense encode-time computation identically.
    """
    sm_cpu = _normalize_shape_map(shape_map)
    leaves = RL.leaves_rect(sm_cpu)
    tok_f = torch.as_tensor(tokens, dtype=torch.float32)
    if tok_f.ndim != 2 or tok_f.shape != (len(leaves), TOKEN_DIM):
        raise TokenFormatError(f"Expected tokens shape ({len(leaves)}, {TOKEN_DIM}), got {tuple(tok_f.shape)}")

    first_entry = next(iter(basis_dev.values()))
    dev = first_entry["U"].device
    tok_dev = tok_f.to(device=dev, dtype=torch.float32)
    sm_dev = sm_cpu[None].to(dev)

    # Group leaf indices by shape and populate only the shapes actually present (plus (2, 2) for build_rect_canvas).
    by_shape: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for idx, (r0, c0, sh, sw) in enumerate(leaves):
        by_shape.setdefault((sh, sw), []).append((idx, r0 // sh, c0 // sw))

    sparse_pyr: dict[tuple[int, int], torch.Tensor] = {}
    for sh, sw in ((2, 2), *tuple(by_shape.keys())):
        if (sh, sw) in sparse_pyr:
            continue
        nh, nw = GRID_SIDE // sh, GRID_SIDE // sw
        grid = torch.zeros((1, nh, nw, TOKEN_DIM), device=dev, dtype=torch.float32)
        items = by_shape.get((sh, sw))
        if items:
            t_idx = torch.tensor([it[0] for it in items], dtype=torch.long, device=dev)
            gi = torch.tensor([it[1] for it in items], dtype=torch.long, device=dev)
            gj = torch.tensor([it[2] for it in items], dtype=torch.long, device=dev)
            grid[0, gi, gj] = tok_dev.index_select(0, t_idx)
        sparse_pyr[(sh, sw)] = grid

    return RTD.build_rect_canvas_tl(
        sparse_pyr,
        sm_dev,
        basis_dev,
        tilelift=tilelift,
        floor=floor,
        tau=tau,
        border_open=border_open,
        gate_scale=gate_scale,
    )


@torch.no_grad()
def decode_tokens_to_latent(
    shape_map: torch.Tensor,
    tokens: torch.Tensor,
    model,
    basis_dev: dict[tuple[int, int], dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    seed: int = DEFAULT_SEED,
    tu_spike_k: float | None = None,
    tu_spike_min_area: int | None = None,
) -> torch.Tensor:
    """Reconstruct raw `[1, 32, 64, 64]` FLUX.2 latent `z_hat` purely from persisted `(shape_map, tokens)`.

    Has zero dependency on the source image or source latent `z`: `(2, 2)` leaves copy through directly from their
    persisted 128-float normalized leaf tokens (`canvas * sv + mv`).
    """
    first_entry = next(iter(basis_dev.values()))
    dev = first_entry["U"].device
    use_cuda = dev.type == "cuda"
    mv = mean.view(1, RL.CH, 1, 1).to(dev).float()
    sv = std.view(1, RL.CH, 1, 1).to(dev).float()

    sm_cpu = _normalize_shape_map(shape_map)
    sm_dev = sm_cpu[None].to(dev)
    canvas, mask, sf, pos, ac_gate = tokens_to_rect_canvas(
        sm_cpu,
        tokens,
        basis_dev,
        tilelift=getattr(model, "_tilelift", None) if model is not None else None,
        floor=getattr(model, "_ac_floor", 0.20) if model is not None else 0.20,
        tau=getattr(model, "_ac_tau", 0.20) if model is not None else 0.20,
    )
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    nz = torch.randn((1, 1, GRID_SIDE, GRID_SIDE), device=dev, generator=gen, dtype=canvas.dtype)
    if model is not None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            resid = model.residual(canvas, sf, pos, nz)
        resid = resid.float()
        if getattr(model, "_zero_mean_resid", False):
            resid = RTD.project_coarse_tile_zero_mean(resid, sm_dev)
        spike_k = float(getattr(model, "_spike_k", 0.0) or 0.0) if tu_spike_k is None else float(tu_spike_k)
        spike_min = (
            int(getattr(model, "_spike_min_area", DEFAULT_SPIKE_MIN_AREA))
            if tu_spike_min_area is None
            else int(tu_spike_min_area)
        )
        if spike_k > 0.0:
            resid = spike_clamp_resid(resid, sm_dev, spike_k, spike_min)
            if getattr(model, "_zero_mean_resid", False):
                resid = RTD.project_coarse_tile_zero_mean(resid, sm_dev)
        z_n = torch.where(mask, canvas + ac_gate * resid, canvas)
    else:
        z_n = canvas
    return (z_n * sv + mv).float()


@torch.no_grad()
def decode_grid_tokens_to_latent(
    grid_tf: RectGridTokenFile,
    model,
    basis_dev: dict[tuple[int, int], dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct and assemble the raw FLUX.2 latent in row-major block order."""
    dev = next(iter(basis_dev.values()))["U"].device
    z = torch.empty((1, RL.CH, grid_tf.grid_h * GRID_SIDE, grid_tf.grid_w * GRID_SIDE),
                    device=dev, dtype=torch.float32)
    for i, block in enumerate(grid_tf.blocks):
        row, col = divmod(i, grid_tf.grid_w)
        y, x = row * GRID_SIDE, col * GRID_SIDE
        z[..., y:y + GRID_SIDE, x:x + GRID_SIDE] = decode_tokens_to_latent(
            block.shape_map, block.tokens, model, basis_dev, mean, std,
            seed=grid_tf.seed, tu_spike_k=grid_tf.tu_spike_k, tu_spike_min_area=grid_tf.tu_spike_min_area,
        )
    return z
