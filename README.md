# tessera

Content-adaptive image tokens.  A 512 px image is encoded by the FLUX.2 VAE into a 64 x 64 latent (one cell = 8 px).
Tessera covers that latent with a **2:1 k-d tree of rectangular tiles** (16 px to 256 px, 13 shapes); each tile becomes one
128-float token, so flat regions cost a few tokens and detailed regions many.  A learned upsampler (**TileUp**, with
**TileDown** / **TileLift** on the encoder side) turns the tokens back into the latent, FLUX.2 decodes it, and a frozen
perceptual **Judge** (v5: LPIPS-VGG, DISTS texture, colour, fine energy, seams, a 32 px mean-error gate) accepts a layout
only if the decode is indistinguishable enough from the plain FLUX.2 round trip.  The layout is found per image by search.

## Install

```bash
git clone https://github.com/kelechi-c/tessera && cd tessera
pip install .            # Python >= 3.11; installs the CUDA build of torch from PyPI - for CPU only, install torch first from
                         # https://download.pytorch.org/whl/cpu
```

## Weights and access

| what | where | sha256 |
| --- | --- | --- |
| tokenizer checkpoint (TileUp + TileDown + TileLift + basis + latent statistics, 121 MB) | `tensorkelechi/tessera-checkpoints`, `phase3_large_tl/step_20000.pt` | `6b092cfc9806424f6353f8141882a49715aedb2186bbccc3f9e9dd8438a5673d` |
| router (inference weights and config only, 4.2 MB) | `tensorkelechi/tessera-checkpoints`, `router/router_rect.pt` | `64885f0fcae11445ec531260626a291689cea7c3e748e3c0b88a1504ff7e9e93` |
| FLUX.2 VAE | `black-forest-labs/FLUX.2-dev` (`vae/` only is used) | gated: accept the licence on the model page with the same account |
| LPIPS and DISTS weights | downloaded on first use by the `lpips` and `DISTS-pytorch` packages | open |

`tensorkelechi/tessera-checkpoints` is **private**: ask the repository owner for read access, then `hf auth login` (or `export HF_TOKEN=...`).

```bash
hf download tensorkelechi/tessera-checkpoints phase3_large_tl/step_20000.pt router/router_rect.pt --local-dir weights
sha256sum weights/phase3_large_tl/step_20000.pt weights/router/router_rect.pt        # must match the table
```

The tokenizer checkpoint is a pickled `torch.load(weights_only=False)` file: load only a file whose hash you have checked.  The router
file holds tensors and plain numbers only and is loaded with `weights_only=True`.

## Example

The flagship encoder: the router predicts the layout in one pass (`--leaf_bias 0.5`), the Judge verifies it at the strict bar, and
failing leaves are split until it passes.  No pruning.

```bash
python - <<'PY'
from PIL import Image, ImageDraw
im = Image.new("RGB", (512, 512), (245, 243, 235)); d = ImageDraw.Draw(im)
for box, c in [((0, 0, 170, 200), (200, 30, 30)), ((170, 0, 512, 60), (20, 40, 150)), ((300, 60, 512, 300), (240, 200, 20)),
               ((300, 300, 512, 512), (20, 40, 150))]:
    d.rectangle(box, fill=c)
for x in (170, 300): d.line((x, 0, x, 512), fill=(10, 10, 10), width=6)
for y in (60, 200, 300): d.line((0, y, 512, y), fill=(10, 10, 10), width=6)
import os; os.makedirs("images", exist_ok=True); im.save("images/mondrian.png")
PY

tessera out --images images --ckpt weights/phase3_large_tl/step_20000.pt \
  --mode router --router weights/router/router_rect.pt --leaf_bias 0.5 \
  --judge_scale 1.0 --struct_scale 1.5 --color_floor_de 1.0 --color_max_floor_de 1.8 --tu_spike_k 3
```

Expected (CPU; the CUDA path is untested in this release):

```
[tessera] dev=cpu n=1 mode=router TileUp=22.18M RectTileDown=4.55M RectTileLift=0.78M
  [1/1] mondrian tok=102 pass=True calls=4 <time>ms
[tessera] 1/1 pass the Judge, mean 102.0 tokens -> out/results.json
```

The 512 x 512 image costs 102 tokens: the router proposed 72, verification needed 4 Judge calls to reach a passing layout.  `results.json`
reports PSNR 36.78 dB against the FLUX.2 round trip and 34.97 dB against the original (the round trip itself is 41.15 dB from the original).
It takes about 1.5-4 minutes on 8 CPU cores.

`out/results.json` has, per image, the token count, Judge verdict, number of Judge calls (`calls`; `route_tok` is the router's layout
before verification) and PSNR / SSIM / LPIPS / DISTS of the decode against the FLUX.2 round trip (`tg`), against the original (`to`) and of the
round trip against the original (`go`).  `out/strips/*.png` shows original | decode | tile borders (colour = tile shape).
`--save_layouts` also writes each layout (`shape_map` uint8 [2, 64, 64]: tile height and width in cells at every cell), the original, the
decode and the round trip.

On 24 images of the project's final evaluation set (GPU run, same flags, same router), all 24 passed the Judge with a mean of 384 tokens and
2.7 Judge calls per image; the router's own layouts averaged 328 tokens before verification.  A plain FLUX.2 latent is 1,024 tokens.

## Options

`tessera OUT --images DIR --ckpt PATH [options]` (`tessera --help` lists all; defaults shown; the defaults are the Judge's default tolerance
`--judge_scale 3.0` and `topdown`, so the flagship flags above are given explicitly).

- `--mode router`: the learned router (`--router`, `--leaf_bias`) predicts the layout, then `--verify` (default on, `--verify_rounds 8`) splits
  failing leaves, falling back to all 16 px tiles if rounds run out; `--no_verify` returns the router's layout and reports whether it passes.
  `--mode topdown` (the default mode, no router needed): start from the four 32 x 32 cell roots and split every leaf the Judge flags until
  the decode passes (about 15x more Judge calls).  `--mode rdtree`: one-shot rate-distortion tree (`--rd_lambda 0.05 --rd_beta 0.5`), same verification.
- `--prune_merges`: after a pass, merge sibling leaves while the Judge still passes (`--prune_max_calls 48` Judge calls per image,
  `--prune_max_rounds 8`, `--prune_fast` skips bisecting failing chunks).  Fewer tokens, more Judge calls.
- `--judge_scale 3.0`: tolerance multiplier of every Judge gate (lower is stricter).  Stricter variants, all off by default:
  `--struct_scale`, `--color_floor_de`, `--color_max_floor_de`, `--mae32_limit`; `--tu_spike_k` clips isolated TileUp spikes.
  The strict bar of the flagship example is `--judge_scale 1.0 --struct_scale 1.5 --color_floor_de 1.0 --color_max_floor_de 1.8 --tu_spike_k 3`.

## Limits

- 512 px only: other sizes are centre-cropped and resized to 512 x 512.
- Tokens are layouts plus 128-float vectors in memory and `results.json`; there is no bitstream or entropy coder, and no standalone decoder
  command (the decode is made inside `tessera` from the tokens, TileUp and the FLUX.2 decoder).
- A CPU run takes minutes per image (each Judge call decodes a 512 px image and runs LPIPS-VGG / DISTS); use a GPU for more than a few images.
- The Judge passes a layout when the decode matches the *FLUX.2 round trip*, not the original: PSNR against the original is bounded by the VAE
  (`go`), and a pass is a perceptual gate, not a guarantee of pixel fidelity.  Fine text and photographed paintings keep many tokens.
- The router is a pilot trained on 279 labelled images (27 for validation).  Alone it is not usable: on 45 calibration images its layouts passed the
  Judge on 4.4 % (`--leaf_bias 0`) and 0 % (`0.5`) of images.  The result above is router + verification: verification repairs the layout with
  about 2-3 Judge calls instead of about 40 for `topdown` + pruning, at about the same token count.  Without `--verify` expect mostly failures.
  One-shot `rdtree` layouts passed 0-76 % depending on the setting.
- Flagship layouts are not pruned (`--prune_merges` would lower the token count at the cost of many more Judge calls).
- Training code, datasets and the evaluation write-ups are not part of this package.
