# tessera

Content-adaptive image tokens: a 512 px image becomes a quadtree of tiles, each a few 128-float tokens (flat areas get one
big tile, dense areas many small ones).  Two kinds of leaf: fixed codes (PCA / edge / polynomial / grain) and the image's own
FLUX.2 latent for 16 px tiles; the encoder picks the cheaper arm that passes per image.  A frozen perceptual judge (v5:
LPIPS-VGG, DISTS texture, colour, fine energy, seams, plus a 32 px mean-error gate) decides whether a layout is good enough.
The layout is found by judge-driven search; a router that predicts it in one pass is defined here but **not trained**.

## Install

```bash
cd tessera
uv sync --extra encoder          # torch, lpips, DISTS-pytorch, diffusers (FLUX.2), click; add --extra dev for pytest
export TESSERA_WEIGHTS=/path/to/dir   # dir with bases.pt (PCA bases, ~550 MB); router.pt later
```

Encoding also needs the FLUX.2-dev VAE (`black-forest-labs/FLUX.2-dev`, gated on Hugging Face) and runs far faster on a GPU
(minutes per dense image on CPU).  `overlay` and `info` need no weights; `decode` needs only `bases.pt` for fixed-code files.

## Commands

```bash
tessera encode IMG -o tokens.tsr [--layout search|router] [--arm both|A|vae16] [--weights DIR] [--record layouts.jsonl]
tessera decode tokens.tsr -o out.png
tessera judge ORIG VAE CAND            # v5 verdict as JSON; exit 0 = pass.  VAE = FLUX.2 round trip of ORIG
tessera overlay IMG tokens.tsr -o overlay.png   # tiles over the image + colour key (pink 1-token, cyan 2-token,
                                                # orange fixed 16 px, green FLUX 16 px, yellow native 8 px)
tessera info tokens.tsr                # arm, tokens, tile counts, recorded verdict
```

## Not trained / not done yet

- **Router**: `--layout router` needs a trained `router.pt`; without it the command stops with `NotTrainedError` ("not trained
  yet; use --layout search").  `--allow-untrained` runs a random-init network with a warning (noise, for plumbing tests only).
  It will be distilled from the search's layouts plus the judge's verdicts: `tessera encode --record layouts.jsonl` writes that
  data (format: `docs/router_training_format.md`).
- **Tile decoder**: decoding is still per-tile pasting; a learned tile decoder is not part of this package.

## Pure-latent path (defined, not trained)

The plan dropped fixed codes for the learned decoder (TileUp, in `qtok/tileup.py`): every tile is one FLUX.2 pyramid
token, 16 px tiles hold the true latent, 32-256 px tiles read the image encoded at the size where the tile is 16 px.
CPU-tested definitions, no weights:

- `tessera.latent.layout`: the layout (`size_map` [B, 64, 64], tile side in cells 2..32), validity, token counts, the
  tree code (split bits, depth first) and token packing in qtok's order (tests cross-check it against
  `qtok.tileup.build_pyramid_canvas` when qtok is importable).
- `tessera.latent.router`: the one-pass router for that layout (dilated FCN, ordinal logits per cell), features that
  need no TileUp (`pyr-res-v1`) or use its judge excess (`pyr-res-tileup-v1`), the verify loop (split failing tiles,
  last round to 16 px, else the all-grid layout), checkpoints tagged with feature spec and TileUp id, and
  `LayoutRecord2` labels (`tessera/layout-record/2`).  `tessera.router` / `core/route.py` remain the fixed-code router
  of the shipped encoder.
- `tessera.gen`: a MAR-style generator: `TreeAR` (split bits, always a valid tree) + `TokenMAR` (masked
  autoregression over leaves with a diffusion head, largest tiles revealed first), per-level token normalisation, and
  the grid baseline through the same code (`TesseraGen(..., fixed_layout=all_grid)`).  `CONFIGS["base"]` is a ~142M
  transformer plus a ~35M diffusion head for both; the tree model is a separate ~3M.  The head predicts v (stable at high noise); labels are
  dropped 10% of the time so the unconditional class is trained; checkpoints carry the per-level token statistics.

Not written yet: the label, train and sampling commands (they go in `qtok/`, once the pyramid-latent shard format of
plan S2 exists), classifier-free guidance in the sampler (the unconditional class is trained, not yet used), and the
`tessera encode --layout router` wiring for this path.

## Layout

`src/tessera/`: `cli.py`, `search.py` (encode/decode drivers), `judge.py` (+ `data/frozen_v5.json`, the frozen T and scales),
`router.py`, `layout.py`, `tsr.py`, `overlay.py`; `core/` = the search and judge code (hires, vae16, judge5*, route);
`stream/` = the token stream codec (grid, leaf codes, tokens, FLUX leaves).  `tests/`: `uv run pytest` (CPU, no weights;
`tests/test_weights.py` runs with `TESSERA_WEIGHTS` set).
