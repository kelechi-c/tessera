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

| what | where | access |
| --- | --- | --- |
| tokenizer checkpoint (TileUp + TileDown + TileLift + basis + latent statistics, 121 MB) | Hugging Face `tensorkelechi/tessera-checkpoints`, file `phase3_large_tl/step_20000.pt`, sha256 `6b092cfc9806424f6353f8141882a49715aedb2186bbccc3f9e9dd8438a5673d` | **private**: ask the repository owner for read access, then `hf auth login` (or `export HF_TOKEN=...`) |
| FLUX.2 VAE | `black-forest-labs/FLUX.2-dev` (`vae/` only is used) | **gated**: accept the licence on the model page with the same account |
| LPIPS and DISTS weights | downloaded on first use by the `lpips` and `DISTS-pytorch` packages | open |

```bash
hf download tensorkelechi/tessera-checkpoints phase3_large_tl/step_20000.pt --local-dir weights
sha256sum weights/phase3_large_tl/step_20000.pt        # must match the table
```

The checkpoint is a pickled `torch.load(weights_only=False)` file: load only a file whose hash you have checked.

## Example

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

tessera out --images images --ckpt weights/phase3_large_tl/step_20000.pt --prune_merges --prune_fast
```

Expected (CPU, about 6-8 minutes on 8 cores; a GPU takes seconds):

```
[tessera] dev=cpu n=1 mode=topdown TileUp=22.18M RectTileDown=4.55M RectTileLift=0.78M
  [1/1] mondrian tok=86 pass=True calls=13 <time>ms
[tessera] 1/1 pass the Judge, mean 86.0 tokens -> out/results.json
```

The 512 x 512 image costs 86 tokens (a plain FLUX.2 latent is 1,024), 5 sibling merges were accepted, and `out/results.json` reports
PSNR 35.97 dB against the FLUX.2 round trip and 34.24 dB against the original.  It has, per image, the token count, Judge verdict, number of Judge calls and PSNR / SSIM / LPIPS / DISTS of
the decode against the FLUX.2 round trip (`tg`), against the original (`to`) and of the round trip against the original
(`go`).  `out/strips/*.png` shows original | decode | tile borders (colour = tile shape).  `--save_layouts` also writes each
layout (`shape_map` uint8 [2, 64, 64]: tile height and width in cells at every cell), the original, the decode and the round trip.

## Options

`tessera OUT --images DIR --ckpt PATH [options]` (`tessera --help` lists all; defaults shown).

- `--mode topdown` (default): start from the four 32 x 32 cell roots and split every leaf the Judge flags until the decode passes.
  `--mode rdtree`: one-shot rate-distortion tree (`--rd_lambda 0.05 --rd_beta 0.5`), then `--verify` (default on, `--verify_rounds 8`)
  splits failing leaves; `--no_verify` returns the one-shot tree and reports whether it passes.
- `--prune_merges`: after a pass, merge sibling leaves while the Judge still passes (`--prune_max_calls 48` Judge calls per image,
  `--prune_max_rounds 8`, `--prune_fast` skips bisecting failing chunks).  Fewer tokens, more Judge calls.
- `--judge_scale 3.0`: tolerance multiplier of every Judge gate (lower is stricter).  Stricter variants, all off by default:
  `--struct_scale`, `--color_floor_de`, `--color_max_floor_de`, `--mae32_limit`; `--tu_spike_k` clips isolated TileUp spikes.
  The project's stricter label bar was `--judge_scale 1.0 --struct_scale 1.5 --color_floor_de 1.0 --color_max_floor_de 1.8 --tu_spike_k 3`.

## Limits

- 512 px only: other sizes are centre-cropped and resized to 512 x 512.
- Tokens are layouts plus 128-float vectors in memory and `results.json`; there is no bitstream or entropy coder, and no standalone decoder
  command (the decode is made inside `tessera` from the tokens, TileUp and the FLUX.2 decoder).
- A CPU run takes minutes per image (each Judge call decodes a 512 px image and runs LPIPS-VGG / DISTS); use a GPU for more than a few images.
- The Judge passes a layout when the decode matches the *FLUX.2 round trip*, not the original: PSNR against the original is bounded by the VAE
  (`go`), and a pass is a perceptual gate, not a guarantee of pixel fidelity.  Fine text and photographed paintings keep many tokens.
- No router weights are published.  In a pilot (306 labelled images) a trained one-shot router passed the Judge on 4.4 % of images alone and was
  only usable with verification; one-shot `rdtree` layouts (`--no_verify`) passed 0-76 % of images depending on the setting.
  Use `topdown` unless you want to study that.
- Training code, datasets and the evaluation write-ups are not part of this package.
