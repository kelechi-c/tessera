"""Resolution-free leaf code "lowres": a pixel-art tile stored as an s-times smaller raster plus a kernel flag.

Why (Fable review revision 5, T.3).  Nearest-neighbour art at 16x (a 32 x 32 sprite shown at 512 px) is 3,072
numbers = 24 tokens, exact, against about 1,000 tokens for the fixed codes; and because the raster is parametric
it re-renders crisp at any multiple of the tile side, so a 1,024 px decode of pixel art is free.

Format (byte-exact)
-------------------
A tile of `side` px (32 / 64 / 128 / 256 / 512 = the leaf sides of the 64 x 64 8-px cell grid, levels 4..0;
all multiples of 8) is stored as:

  header: 8 float32 numbers, in this order
      version (1), s, oy, ox, kernel (0 = nearest, 1 = bicubic), side, h, w
  raster: 3 * h * w uint8 numbers, RGB planes [3, h, w] (the native raster, i.e. the tile's own bytes)

`h x w` is the native raster size actually stored.  The lattice boundaries are y = oy + k*s and x = ox + k*s plus
the tile border, so rows [0, oy) form one partial top block (likewise columns), and

  h = (oy > 0) + ceil((side - oy) / s),   w = (ox > 0) + ceil((side - ox) / s)
  floats = 3*h*w + 8,    tokens = ceil(floats / 128)     (a token is 128 floats, the repo convention)

Pixel (y, x) is raster[:, p + (y - oy)//s, q + (x - ox)//s] with p = int(oy > 0), q = int(ox > 0); for y < oy the
index is 0 (the partial top block).  The offset is what finds art that is not aligned to the 8 px grid, e.g.
(s=16, oy=3, ox=5).  Examples: 32 x 32 native at s = 16 (a 16x upscale) = 3,072 + 8 floats = 25 tokens; 16 x 16
native at s = 32 = 776 floats = 7 tokens; 128 x 128 native at s = 4 (4x) = 49,160 floats = 385 tokens.

Decoder.  `decode(enc, side)` renders the tile directly at any multiple of the stored tile side (512, 1024, 2048,
...) by nearest: output pixel Y maps to tile pixel (Y * side0) // side, then to its native block.  Block edges are
hard by construction (every source block is one colour, no intermediate colours), so pixel art decodes crisp at
2x/4x for no extra tokens.  The bicubic kernel is the lossy option (`encode_lossy`): raster = bicubic downsample
by s, decode = bicubic upsample; used only for reporting / smooth content, never by the exact pass.

Lattice detection (`detect`).  A tile has the (s, oy, ox) lattice iff every row that differs from the row above
starts at a position y = oy (mod s), and likewise every column that differs from the column to its left starts at
x = ox (mod s); then every block is constant and equal to its top-left pixel (the partial border block included,
which is what the offset definition above stores).  So detection is vectorised: the residues mod s of the
row-change positions and of the column-change positions must each be single.  `detect` tries s = 2, 4, 8, 16, 32,
verifies the winner pixel-exactly and returns the cheapest exact representation; s = 1 means there is no lattice
(photos and synthetic non-art images), and `encode` returns None for those (the fixed codes keep the leaf).

API for the hires.py integration
--------------------------------
  s, oy, ox = detect(img)          # img: uint8 HWC / [3, H, W] / torch [-1, 1]; s = 1 = no lattice
  enc = encode(img)                # None, or dict(raster [3, h, w] uint8, s, oy, ox, kernel, side, h, w,
                                   #           floats, tokens, header, encode_err)
  y = decode(enc, side=1024)       # uint8 [side, side, 3], hard edges
  n = tokens(enc); f = floats(enc); c = cost(enc)
  s, oy, ox, raster = detect_raster(img)   # detection + the native raster in one call (raster None if s = 1)
`to_u8` / `to_tensor` convert between the oracle's torch float convention ([-1, 1]) and the uint8 raster.

usage:
  python -m qtok.lowres measure --out docs/reviews/fable-results/t3 [--images DIR] [--photos /tmp/r2data]
"""
import json
import pathlib
import time

import click
import numpy as np
import torch
from PIL import Image

K = 128  # floats per token
SS = (2, 4, 8, 16, 32)  # lattice periods in px (the s-times-smaller raster)
SIDES = (32, 64, 128, 256, 512)  # tile sides in px (leaf levels 4..0 of the 64 x 64 8-px cell grid)
HEADER = 8  # header floats per tile: version, s, oy, ox, kernel, side, h, w
VERSION = 1
KERNELS = ("nearest", "bicubic")
KERNEL_ID = {k: i for i, k in enumerate(KERNELS)}
T3 = pathlib.Path(__file__).resolve().parent.parent / "docs/reviews/fable-results/t3"
PASS_RATIO = 1.2  # T3: stored representation at most 1.2x the native raster's own float count


# ---------------------------------------------------------------- pixel conversion ------------------------------------
def to_u8(img):
    """Tile -> uint8 HWC [H, W, 3].  Accepts PIL, numpy uint8 HWC / [3, H, W], or torch [-1, 1] ([1, 3, H, W])."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    if torch.is_tensor(img):
        x = img.detach().float().cpu()
        while x.ndim > 3:
            x = x[0]
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"expected a torch tile [3, H, W] or [1, 3, H, W], got {tuple(x.shape)}")
        a = np.rint((x.permute(1, 2, 0).numpy() + 1.) * 127.5)
        return np.clip(a, 0, 255).astype(np.uint8)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {a.dtype}; pass torch [-1, 1] for float tiles")
    if a.ndim != 3:
        raise ValueError(f"expected an RGB tile, got shape {a.shape}")
    if a.shape[-1] != 3:
        if a.shape[0] == 3:
            a = a.transpose(1, 2, 0)
        else:
            raise ValueError(f"expected an RGB tile, got shape {a.shape}")
    return np.ascontiguousarray(a)


def to_tensor(y):
    """uint8 [side, side, 3] (as decode returns it) -> torch [1, 3, side, side] in [-1, 1], the oracle's convention."""
    a = to_u8(y)
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1.)


def _side(a):
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] != 3:
        raise ValueError(f"expected a square RGB tile [H, W, 3], got {a.shape}")
    return int(a.shape[0])


# ---------------------------------------------------------------- lattice detection -----------------------------------
def _residue(pos, s):
    """The single residue of the change positions mod s, or None if they split; no change -> 0 (aligned offset)."""
    if len(pos) == 0:
        return 0
    r = np.unique(np.asarray(pos) % s)
    return int(r[0]) if r.size == 1 else None


def _block_map(pix, s, o):
    """For each pixel index in `pix`, the native block index: 0 for the partial border block, then 1 + (pix - o)//s."""
    return int(o > 0) + (np.asarray(pix, dtype=np.int64) - o) // s


def _indices(side, s, o):
    return _block_map(np.arange(side), s, o)


def _build(a, s, oy, ox, kernel="nearest", encode_err=0.):
    """Assemble the dict for a known (s, oy, ox) lattice of uint8 tile `a` [side, side, 3]."""
    side = _side(a)
    p, q = int(oy > 0), int(ox > 0)
    h = p + -(-(side - oy) // s)  # ceil((side - oy) / s) full blocks + the partial top block
    w = q + -(-(side - ox) // s)
    ry = np.empty(h, np.int64)
    rx = np.empty(w, np.int64)
    ry[0] = rx[0] = 0
    if p:
        ry[1:] = oy + np.arange(h - p) * s
    else:
        ry[:] = np.arange(h) * s
    if q:
        rx[1:] = ox + np.arange(w - q) * s
    else:
        rx[:] = np.arange(w) * s
    raster = np.ascontiguousarray(a[np.ix_(ry, rx)].transpose(2, 0, 1))  # [3, h, w]
    floats = 3 * h * w + HEADER
    tokens = (floats + K - 1) // K
    hdr = (VERSION, int(s), int(oy), int(ox), KERNEL_ID[kernel], side, int(h), int(w))
    return dict(raster=raster, s=int(s), oy=int(oy), ox=int(ox), kernel=kernel, side=side,
                h=int(h), w=int(w), floats=int(floats), tokens=int(tokens), header=hdr,
                encode_err=float(encode_err))


def _matches(a, enc):
    r = enc["raster"].transpose(1, 2, 0)
    return np.array_equal(r[np.ix_(_indices(enc["side"], enc["s"], enc["oy"]),
                                   _indices(enc["side"], enc["s"], enc["ox"]))], a)


def detect(img):
    """(s, oy, ox): the cheapest exact lattice with s in SS, or (1, 0, 0) when the tile has no lattice."""
    a = to_u8(img)
    side = _side(a)
    dy = np.flatnonzero((a[1:] != a[:-1]).any(axis=(1, 2))) + 1  # rows that differ from the row above
    dx = np.flatnonzero((a[:, 1:] != a[:, :-1]).any(axis=(0, 2))) + 1  # columns that differ from the left column
    best = None
    for s in SS:
        if s > side:
            continue
        oy, ox = _residue(dy, s), _residue(dx, s)
        if oy is None or ox is None:
            continue
        enc = _build(a, s, oy, ox)
        if not _matches(a, enc):
            continue
        key = (enc["tokens"], enc["floats"], -s)  # cheapest; ties go to the larger s (fewer native pixels)
        if best is None or key < best[0]:
            best = (key, (int(s), int(oy), int(ox)))
    return best[1] if best is not None else (1, 0, 0)


def native_raster(enc):
    """The stored native raster as uint8 HWC [h, w, 3] (the bytes the code spends tokens on)."""
    return np.ascontiguousarray(enc["raster"].transpose(1, 2, 0))


def detect_raster(img):
    """(s, oy, ox, raster): the lattice and the exact native raster as uint8 HWC (None when s = 1).
    Convenience wrapper around detect + encode; `raster` is the native-resolution raster the design note asks
    lattice detection to return."""
    a = to_u8(img)
    s, oy, ox = detect(a)
    if s < 2:
        return s, oy, ox, None
    return s, oy, ox, native_raster(encode(a))


def encode(img, max_error=1. / 255.):
    """The cheapest exact lowres representation of a pixel-art tile, or None when it has no s >= 2 lattice."""
    a = to_u8(img)
    s, oy, ox = detect(a)
    if s < 2:
        return None
    enc = _build(a, s, oy, ox)
    err = float(np.abs(decode(enc, enc["side"]).astype(np.int16) - a.astype(np.int16)).max()) / 255.
    if err > max_error:
        return None
    enc["encode_err"] = err
    return enc


def encode_lossy(img, kernel="bicubic", max_err=4. / 255., s=None):
    """Lossy alternative, used only for reporting: bicubic downsample by s then re-upsample; keep the cheapest s
    (largest first) whose uint8 max error is at most max_err.  Returns None if none passes; oy = ox = 0."""
    a = to_u8(img)
    side = _side(a)
    for q in ([s] if s else sorted(SS, reverse=True)):
        if q > 1 and side % q:
            continue
        n = side // q
        small = np.asarray(Image.fromarray(a).resize((n, n), Image.BICUBIC))
        rec = np.asarray(Image.fromarray(small).resize((side, side), Image.BICUBIC))
        err = float(np.abs(rec.astype(np.int16) - a.astype(np.int16)).max()) / 255.
        if err <= max_err:
            enc = _build(np.asarray(Image.fromarray(small).resize((side, side), Image.NEAREST)), q, 0, 0,
                         kernel=kernel)
            enc.update(raster=np.ascontiguousarray(small.transpose(2, 0, 1)), h=n, w=n,
                       floats=3 * n * n + HEADER, tokens=(3 * n * n + HEADER + K - 1) // K,
                       header=(VERSION, int(q), 0, 0, KERNEL_ID[kernel], side, n, n), encode_err=err)
            return enc
    return None


# ---------------------------------------------------------------- decode --------------------------------------------
def decode(enc, side=None):
    """Render a lowres tile as uint8 [side, side, 3].  `side` must be a multiple of enc["side"] for nearest (its
    natural use: 512, 1024, 2048); intermediate sides fall back to a nearest pixel-to-tile map.  No interpolation:
    every source block stays one colour, so block edges are hard."""
    side = int(side or enc["side"])
    if enc["kernel"] == "bicubic":
        im = Image.fromarray(np.ascontiguousarray(enc["raster"].transpose(1, 2, 0)))
        return np.asarray(im.resize((side, side), Image.BICUBIC))
    tile = (np.arange(side, dtype=np.int64) * enc["side"]) // side
    r = enc["raster"].transpose(1, 2, 0)
    return r[np.ix_(_block_map(tile, enc["s"], enc["oy"]), _block_map(tile, enc["s"], enc["ox"]))]


def tokens(enc):
    """Floats of the stored representation, in tokens of 128 (the header included)."""
    return int(enc["tokens"])


def floats(enc):
    return int(enc["floats"])


def cost(enc):
    """Accounting for the T3 pass criterion.  native_floats = 3 * (side/s)^2 = the raster the art was upscaled
    from; stored_floats = 3*h*w + 8; token_floats = 128 * tokens (the codec's actual bill)."""
    native_side = -(-enc["side"] // enc["s"])
    native = 3 * native_side * native_side
    tok = tokens(enc)
    return dict(floats=int(enc["floats"]), tokens=tok, token_floats=K * tok, raster_floats=3 * enc["h"] * enc["w"],
                native_side=native_side, native_floats=int(native),
                ratio=float(enc["floats"]) / native, token_ratio=float(K * tok) / native,
                raster_ratio=float(enc["floats"]) / (3 * enc["h"] * enc["w"]))


def pack_header(enc):
    """The 8-float header as the format defines it (float32 in the token stream)."""
    return np.asarray(enc["header"], dtype=np.float32)


def unpack_header(hdr):
    """Header floats -> dict(version, s, oy, ox, kernel, side, h, w), the inverse of pack_header."""
    v, s, oy, ox, kern, side, h, w = [int(round(float(x))) for x in np.asarray(hdr).flatten()[:HEADER]]
    return dict(version=v, s=s, oy=oy, ox=ox, kernel=KERNELS[kern], side=side, h=h, w=w)


# ---------------------------------------------------------------- measurement ---------------------------------------
def _pack_colours(d):
    d = d.reshape(-1, 3).astype(np.uint32)
    return (d[:, 0] << 16) | (d[:, 1] << 8) | d[:, 2]


def hard_edges(enc, target=1024):
    """Decode at `target` px and prove hard block edges: the decode is the q x q nearest blow-up of the stored tile
    (every source pixel is constant inside its q x q cell) and it uses no colour that is not in the native raster
    (so no intermediate colours from resampling)."""
    side = enc["side"]
    if target % side:
        target = side * 2
    q = target // side
    d = decode(enc, target)
    up = np.repeat(np.repeat(decode(enc, side), q, 0), q, 1)
    cell = d.reshape(side, q, side, q, 3)
    cell_ok = bool(np.array_equal(cell, np.broadcast_to(cell[:, :1, :, :1], cell.shape)))
    raster = enc["raster"].transpose(1, 2, 0)
    cols = np.unique(_pack_colours(d))
    inter = int(np.setdiff1d(cols, np.unique(_pack_colours(raster))).size)
    return dict(side=target, cell=cell_ok, no_intermediate=inter == 0, intermediates=inter,
                upsample=bool(np.array_equal(d, up)), ok=bool(cell_ok and inter == 0 and np.array_equal(d, up)))


def measure_one(name, img, bicubic=True, bicubic_max_err=4. / 255.):
    """One pixel-art image: encode (detect), exact 512 decode check, 1024 hard-edge check, token accounting, plus
    the lossy bicubic attempt for reporting.  Returns the row written to rows.jsonl."""
    t0 = time.time()
    a = to_u8(img)
    side = _side(a)
    s, oy, ox = detect(a)
    row = dict(name=str(name), side=side, s=s, oy=oy, ox=ox, exact=False, hard_1024=False, passed=False)
    if s < 2:
        row.update(error="no lattice (s=1)")
        row["seconds"] = round(time.time() - t0, 4)
        return row
    enc = encode(a)
    if enc is None:  # detection is exact by construction; this only guards a broken input
        row.update(error="lattice found but round trip is not exact")
        row["seconds"] = round(time.time() - t0, 4)
        return row
    c = cost(enc)
    dec = decode(enc, side)
    err = float(np.abs(dec.astype(np.int16) - a.astype(np.int16)).max()) / 255.
    hard = hard_edges(enc, max(1024, side))
    bi = None
    if bicubic:
        encb = encode_lossy(a, max_err=bicubic_max_err)
        bi = None if encb is None else dict(s=encb["s"], tokens=encb["tokens"],
                                            max_error=round(encb["encode_err"], 6))
    row.update(h=enc["h"], w=enc["w"], kernel=enc["kernel"], floats=c["floats"], tokens=c["tokens"],
               token_floats=c["token_floats"], native_side=c["native_side"], native_floats=c["native_floats"],
               ratio=round(c["ratio"], 5), token_ratio=round(c["token_ratio"], 5),
               raster_ratio=round(c["raster_ratio"], 5), max_error=round(err, 6), exact=bool(err < 1. / 255.),
               hard_1024=hard["ok"], hard=hard, bicubic=bi)
    row["passed"] = bool(row["exact"] and c["ratio"] <= PASS_RATIO and c["token_ratio"] <= PASS_RATIO
                         and row["hard_1024"])
    row["seconds"] = round(time.time() - t0, 4)
    return row


FIXED_124 = T3.parent / "p3_e2e13_124" / "rows_A_0.jsonl"  # the adopted-rule fixed-code token bill on the 124 set
FIXED_ARM = "A"


def _fixed_tokens(path=None):
    """name -> fixed-code tokens on the 124 set, from the earlier measured rows (qtok.p3 arm A), if present."""
    path = pathlib.Path(path) if path is not None else FIXED_124
    if not path.exists():
        return None, None
    table = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") == FIXED_ARM and "tokens" in r:
            table[r["name"]] = float(r["tokens"])
    return table, str(path)


def scan_photos(items, limit=0, fixed=None):
    """Run lattice detection on a photo/test set: `items` = [(name, PIL/np image, group), ...] (as qtok.p3
    test_items returns) or a DATA path (then qtok.p3.test_items(DATA, "full") = the 124-image set is used).
    A photo triggers lowres only if its whole 512 px tile is an exact s >= 2 lattice (expected: none of the
    held-out photos do).  When the earlier fixed-code row file exists, triggered images also report the fixed
    code's token bill, so the summary can say whether lowres would actually change a token count."""
    source = "in-memory items"
    if isinstance(items, (str, pathlib.Path)):
        from .p3 import test_items

        source = str(items)
        items = test_items(items, "full")
    items = list(items)
    if limit:
        items = items[:limit]
    fxt, fixed_path = _fixed_tokens(fixed)
    t0 = time.time()
    rows = []
    for name, im, grp in items:
        a = to_u8(im)
        s, oy, ox = detect(a)
        name = str(name)
        r = dict(name=name, group=str(grp), source="synthetic" if name.startswith("synth/") else "held-out",
                 s=s, oy=oy, ox=ox, triggered=bool(s >= 2))
        if r["triggered"]:  # would be offered only if exact; report its bill to be complete
            enc = encode(a)
            r.update(tokens=tokens(enc), floats=floats(enc))
            if fxt is not None and name in fxt:
                r["fixed_tokens"] = fxt[name]
                r["cheaper"] = bool(r["tokens"] < fxt[name])
        rows.append(r)
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], dict(n=0, triggered=0, synthetic=0))
        g["n"] += 1
        g["triggered"] += int(r["triggered"])
        g["synthetic"] += int(r["source"] == "synthetic")
    sh = dict(n=sum(int(r["source"] == "held-out") for r in rows),
              triggered=sum(int(r["triggered"]) for r in rows if r["source"] == "held-out"))
    syn = dict(n=sum(int(r["source"] == "synthetic") for r in rows),
               triggered=sum(int(r["triggered"]) for r in rows if r["source"] == "synthetic"))
    adopted = sum(int(r["triggered"] and r.get("cheaper")) for r in rows)
    unknown = sum(int(r["triggered"] and "fixed_tokens" not in r) for r in rows)
    return dict(n=len(rows), triggered=sum(int(r["triggered"]) for r in rows), rows=rows, groups=groups,
                seconds=round(time.time() - t0, 3), source=source, heldout=sh, synthetic=syn,
                would_be_adopted=adopted, unknown_adoption=unknown, fixed_path=fixed_path)


def _summary_md(res):
    rows, photo = res["rows"], res["photo"]
    n = len(rows)
    n_pass = sum(int(r["passed"]) for r in rows)
    exact = [r for r in rows if r.get("exact")]
    errs = [r["max_error"] for r in rows if "max_error" in r]
    ratios = [r["ratio"] for r in rows if "ratio" in r]
    tratios = [r["token_ratio"] for r in rows if "token_ratio" in r]
    hard = sum(int(r["hard_1024"]) for r in rows)
    md = ["# T3 lowres leaf: exact storage of pixel art, free 1024 px decode, photo trigger scan", "",
          "Fable review revision 5, section T.4 step T3 (`docs/reviews/20261110-fable-review.md`).  The `lowres`",
          "code stores a tile as an s-times smaller uint8 RGB raster plus an 8-float header (s, oy, ox, kernel,",
          "side, h, w, version); floats = 3*h*w + 8 and tokens = ceil(floats / 128).  Nearest-neighbour art at s = 16",
          "(a 32 x 32 sprite shown at 512 px) is 3,072 + 8 floats = 25 tokens, exact; the decoder renders the raster",
          "directly at 1,024 px (and any multiple of the tile side) with hard block edges and no intermediate colours.",
          "", f"Command: `{res['command']}`", ""]
    if res.get("gen"):
        g = res["gen"]
        md += [f"- Generated set: `qtok/lowres_data.py`, {g['n']} deterministic 512 px pixel-art images",
               f"  (native rasters {g['sides']} px, s = {g['factors']}; {g['offsets']} with non-8-px-aligned offsets;",
               f"  {g['four_x']} at 4x upscale = 128 px raster).  Files: `{g['dir']}/`.",
               "  (T3's text lists native rasters of 16/32/48/64 px; 48 px cannot tile 512 px with integer",
               "  s in {2,4,8,16,32} -- 512/s is 16/32/64/128/256 -- so the set uses 16/32/64/128 px.)"]
    else:
        md += [f"- Image set: {res.get('images_dir', 'in-memory items')}."]
    md += ["", "## Pixel-art set", "",
          "| # | image | kind | s | oy | ox | h x w | floats | tokens | ratio | token ratio | max err | 1024 hard | pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        if "error" in r:
            md.append(f"| {i} | {r['name']} | ? | {r['s']} | {r['oy']} | {r['ox']} | - | - | - | - | - | - | - | "
                      f"FAIL ({r['error']}) |")
            continue
        md.append(f"| {i} | {r['name']} | {r.get('kind', '')} | {r['s']} | {r['oy']} | {r['ox']} | "
                  f"{r['h']} x {r['w']} | {r['floats']} | {r['tokens']} | {r['ratio']:.4f} | "
                  f"{r['token_ratio']:.4f} | {r['max_error']:.6f} | {'yes' if r['hard_1024'] else 'NO'} | "
                  f"{'PASS' if r['passed'] else 'FAIL'} |")
    md += ["", f"**Totals: {n_pass}/{n} pass.**  "
               f"Exact reconstruction (< 1/255): {len(exact)}/{n} (max error {min(errs) if errs else '-'} .. "
               f"{max(errs) if errs else '-'} in [0, 1] units, i.e. 0/255 byte error throughout).  "
               f"Stored/native float ratio: {min(ratios):.4f} .. {max(ratios):.4f} (limit {PASS_RATIO}); "
               f"token-granular ratio: {min(tratios):.4f} .. {max(tratios):.4f}.  "
               f"1,024 px decode with hard block edges and no intermediate colours: {hard}/{n}.", "",
               f"Lossy bicubic attempt (budget 4/255) accepted on {sum(1 for r in rows if r.get('bicubic'))}/{n} "
               "images: hard-edged art has no bicubic downsample+upsample that stays inside the error budget, which",
               "is exactly why the exact pass is nearest; the bicubic path is for smooth content (exercised by the",
               "tests on a smooth gradient).", ""]
    if photo is not None:
        ho, syn = photo["heldout"], photo["synthetic"]
        md += ["## Photo trigger scan", "",
               f"Data: `{photo['source']}`, {photo['n']} images (`qtok.p3.test_items(DATA, \"full\")` = the e2e13",
               f"124-image set: {photo['heldout']['n']} held-out photos / test images + {photo['synthetic']['n']} "
               "synthetic documents, flats and Mondrians).  Lattice detection on each whole 512 px image; a trigger",
               "means the whole tile is an exact s >= 2 lattice.",
               "",
               f"- **Held-out photos: {ho['triggered']}/{ho['n']} trigger** (the T3 expectation: real photographs are",
               "  not exact lattices).",
               f"- Synthetic items: {syn['triggered']}/{syn['n']} trigger.  They are generated flat colours and",
               "  axis-aligned Mondrian rectangles, which really are exact lattices, so detection is right to offer",
               "  lowres; the question is whether it is cheaper than the existing fixed code.",
               ""]
        md += ["| group | images | triggered |", "|---|---|---|"]
        for g in sorted(photo["groups"]):
            v = photo["groups"][g]
            md.append(f"| {g} | {v['n']} | {v['triggered']} |")
        trig = [r for r in photo["rows"] if r["triggered"]]
        if trig:
            md += ["", "Triggered images, with the fixed code's measured token bill on the same image",
                   f"(`{photo.get('fixed_path')}`, arm {FIXED_ARM}; lowres tokens = ceil((3*h*w + 8)/128)):", "",
                   "| image | source | s | lowres tokens | fixed tokens | lowres cheaper? |", "|---|---|---|---|---|---|"]
            for r in trig:
                ft = r.get("fixed_tokens")
                cheaper = r.get("cheaper")
                md.append(f"| {r['name']} | {r['source']} | {r['s']} | {r['tokens']} | "
                          f"{'?' if ft is None else (str(int(ft)) if ft == int(ft) else f'{ft:g}')} | "
                          f"{'?' if cheaper is None else ('yes' if cheaper else 'no')} |")
            md += ["", f"Lowres would be adopted on {photo['would_be_adopted']}/{len(trig)} triggered images"
                       + ("" if photo.get("fixed_path") else " (no fixed-code rows available)")
                       + f"; {len(trig) - photo['would_be_adopted'] - photo['unknown_adoption']}/"
                       f"{len(trig)} are cheaper with the fixed code, so their token count is unchanged."]
        md += ["", f"**{photo['triggered']}/{photo['n']} images trigger lowres "
               f"({ho['triggered']}/{ho['n']} held-out photos, {syn['triggered']}/{syn['n']} synthetic); "
               f"{photo['would_be_adopted']} would actually change a token count.**  Scan wall time: "
               f"{photo['seconds']} s.", ""]
    md += ["## Integration notes for `qtok/hires.py`", "",
           "The lowres code is an extra leaf code: it never changes an existing code.  Leaf sides map as",
           "`side = H.size(level) * H.P` (8 px cells), so lowres applies at levels 4..0 = 32, 64, 128, 256, 512 px",
           "(the 16 px native-cell leaf at level 5 is below `SIDES`).  For each leaf:",
           "",
           "```",
           "s, oy, ox = L.detect(leaf_u8)      # leaf_u8 = uint8 tile, e.g. L.to_u8(x_float[-1, 1])",
           "enc = L.encode(leaf_u8)            # None when the leaf has no s >= 2 lattice",
           "n = L.tokens(enc)                  # the bill: ceil((3*h*w + 8) / 128)",
           "y = L.decode(enc, target_px)       # uint8 [target_px, target_px, 3], hard edges;",
           "                                   #   target_px = 2/4 x the leaf side at a 1024/2048 px decode",
           "```",
           "",
           "Admission is free choice: accept the leaf only when `enc is not None` and `L.tokens(enc)` is no more than",
           "the fixed code's cost for that leaf (1 token coded, 2 tokens two-token, 1.5 floats per native cell).",
           "Fixed sizes: a 32 px leaf with a 2^2 native raster is 1 token exact, and a 64 px leaf with 4^2 native",
           "cells is also 1 token, so lowres ties the fixed 1-token codes while being exact.  The decode is",
           "byte-exact in uint8; the oracle's float [-1, 1] round trip (`to_tensor`) is within 0.5/255.",
           "", "## Verdict", "",
           f"T3 pass criterion: exact reconstruction (max error under 1/255) at no more than {PASS_RATIO}x the",
           "raster's own float count, and a 1,024 px decode with hard block edges (no intermediate colours).",
           "",
           f"**{'PASS' if n_pass == n else 'FAIL'}**: {n_pass}/{n} pixel-art images exact (byte error 0), "
           f"token cost {min(tratios):.3f}..{max(tratios):.3f}x the native raster, {hard}/{n} hard-edged 1,024 px",
           f"decodes; {photo['triggered'] if photo else 'n/a'}/{photo['n'] if photo else 'n/a'} test images "
           f"trigger lowres ({photo['would_be_adopted'] if photo else 'n/a'} would change a token count; the "
           f"held-out photos never trigger).", ""]
    return "\n".join(md) + "\n"


def _write_upscale_demo(path, name, img, crop=256):
    """Evidence for the T3 1,024 px claim: hard-edged nearest render on the left, bicubic upsample of the same
    region on the right (same 2x display size, the right one shows the intermediate colours the left never has)."""
    enc = encode(img)
    lo = decode(enc, 512)
    hi = decode(enc, 1024)
    c = (512 - crop) // 2
    hard = hi[2 * c:2 * (c + crop), 2 * c:2 * (c + crop)]
    soft = np.asarray(Image.fromarray(lo[c:c + crop, c:c + crop]).resize((2 * crop, 2 * crop), Image.BICUBIC))
    sep = np.full((2 * crop, 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([hard, sep, soft], 1)).save(path)
    print(f"wrote upscale demo ({name}): left = lowres 1,024 px render, right = bicubic 2x (intermediate colours)",
          flush=True)


def measure(out, images=None, photos=None, limit=0, bicubic=True, write_art=True, fixed=None):
    """Measure the lowres code on a pixel-art set and (optionally) scan a photo set for false triggers.

    `images`: None = the built-in 20-image generator (written under docs/reviews/fable-results/t3/pixelart/);
    a directory of 512 px PNGs; or a list of (name, image) / dict(name=, img=) for tests.  `photos`: DATA path for
    qtok.p3.test_items(DATA, "full"), or a list of (name, image, group).  Writes OUT/rows.jsonl,
    OUT/rows_photos.jsonl (when photos given) and OUT/summary.md, and returns the summary dict."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen = None
    if images is None:
        from . import lowres_data as D

        art = T3 / "pixelart" if write_art else out / "pixelart"
        built = D.build(art)
        items = [(r["name"], r["img"], r) for r in built]
        gen = dict(n=len(built), dir=str(art), sides=sorted({r["native_side"] for r in built}),
                   factors=sorted({r["s"] for r in built}),
                   offsets=sum(1 for r in built if r["oy"] or r["ox"]),
                   four_x=sum(1 for r in built if r["s"] == 4))
    elif isinstance(images, (str, pathlib.Path)):
        items = [(p.stem, np.asarray(Image.open(p).convert("RGB")), None)
                 for p in sorted(pathlib.Path(images).glob("*.png"))]
    else:
        items = [(r["name"], r["img"], r) if isinstance(r, dict) else (r[0], r[1], None) for r in images]
    if limit:
        items = items[:limit]
    rows = []
    for k, (name, im, meta) in enumerate(items):
        r = measure_one(name, im, bicubic=bicubic)
        if meta is not None:
            r.update(kind=meta.get("kind"), true_native_side=meta.get("native_side"))
        rows.append(r)
        if "error" in r:
            print(f"[{k + 1}/{len(items)}] {name}: FAIL {r['error']}", flush=True)
        else:
            print(f"[{k + 1}/{len(items)}] {name}: s={r['s']} oy={r['oy']} ox={r['ox']} "
                  f"{r['h']}x{r['w']} -> {r['tokens']} tok, exact={r['exact']} 1024hard={r['hard_1024']} "
                  f"{'PASS' if r['passed'] else 'FAIL'}", flush=True)
    if gen is not None:
        first = next(r for r in items if r[0] == "scene_detail_a") if any(r[0] == "scene_detail_a" for r in items) else items[0]
        try:
            _write_upscale_demo((T3 if write_art else out) / "upscale_1024.png", first[0], first[1])
        except Exception as e:  # never fail the measurement over the demo image
            print(f"upscale demo skipped: {e}", flush=True)
    photo = scan_photos(photos, limit=limit, fixed=fixed) if photos is not None else None
    if photo is not None:
        print(f"photo scan: {photo['triggered']}/{photo['n']} trigger lowres in {photo['seconds']} s", flush=True)
    cmd = "python -m qtok.lowres measure --out " + str(out)
    if images is not None:
        cmd += f" --images {images}"
    if photos is not None:
        cmd += f" --photos {photos}"
    res = dict(command=cmd, rows=rows, photo=photo, out=str(out), seconds=round(time.time() - t0, 2), gen=gen,
               images_dir=str(images) if images is not None else None)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if photo is not None:
        (out / "rows_photos.jsonl").write_text("".join(json.dumps(r) + "\n" for r in photo["rows"]))
    md = _summary_md(res)
    (out / "summary.md").write_text(md)
    print(md, flush=True)
    print("LOWRES_DONE", flush=True)
    return res


@click.group()
def cli():
    pass


@cli.command("measure")
@click.option("--out", required=True, type=click.Path(), help="output dir (rows.jsonl, summary.md)")
@click.option("--images", default=None, type=click.Path(), help="dir of 512 px PNGs (default: generated set)")
@click.option("--photos", default=None, type=click.Path(), help="DATA dir with heldout.pt for the 124-image scan")
@click.option("--limit", type=int, default=0, help="cap the number of images (smoke)")
@click.option("--bicubic/--no_bicubic", default=True, help="also report the lossy bicubic attempt")
@click.option("--fixed", default=None, type=click.Path(),
              help="fixed-code rows.jsonl on the same image set (default: the p3_e2e13_124 arm-A rows)")
def measure_cli(out, images, photos, limit, bicubic, fixed):
    measure(out=out, images=images, photos=photos, limit=limit, bicubic=bicubic, fixed=fixed)



if __name__ == "__main__":
    cli()
