# Tessera

Tessera represents a 512 × 512 image with adaptive rectangular tiles in FLUX.2 latent space. Each tile carries one 128-float token. On 90 held-out images, Tessera averaged **369.6 tokens** per image against a uniform grid of 1,024. All 90 reconstructions passed the quality check, averaging 2.9 checks per image.

[Interactive demo](https://artifacts.kelechi.cc/tessera_demo_v2/) · [Article](https://artifacts.kelechi.cc/articles/tessera-content-adaptive-tokenization) · [Lab report](https://artifacts.kelechi.cc/articles/tessera-v2-lab-report)

## Quickstart

Use Python 3.11+ and a Hugging Face account with access to the [Tessera checkpoints](https://huggingface.co/tensorkelechi/tessera-checkpoints) and the [FLUX.2-dev VAE](https://huggingface.co/black-forest-labs/FLUX.2-dev).

```bash
git clone https://github.com/kelechi-c/tessera && cd tessera
pip install .
hf auth login
hf download tensorkelechi/tessera-checkpoints phase3_large_tl/step_20000.pt router/router_rect.pt --local-dir weights
```

Run the verified router on your own image. Replace `photo.png` with its path.

```bash
image_dir=$(mktemp -d)
cp photo.png "$image_dir/"
tessera out --images "$image_dir" --ckpt weights/phase3_large_tl/step_20000.pt \
  --mode router --router weights/router/router_rect.pt --leaf_bias 0.5 \
  --judge_scale 1.0 --struct_scale 1.5 --color_floor_de 1.0 --color_max_floor_de 1.8 --tu_spike_k 3
```

Open `out/strips/` to see the reconstruction. Token counts and quality metrics are in `out/results.json`. The CLI centre-crops and resizes inputs to 512 × 512. It evaluates the reconstruction in memory and writes comparison images and metrics. See `tessera --help` for options.

## How it works

The router proposes tiles on the FLUX.2 latent grid. Tessera reconstructs the image, checks the extra error relative to a plain FLUX.2 round trip, and splits tiles where more detail is needed.

![Tessera architecture showing the FLUX.2 latent grid, router proposal, tile reconstruction, and encode-time verification loop](docs/images/architecture.png)

The painting detail below took **164 tokens** after two quality checks.

![Veronese island input, final tile boundaries, grid reference, and verified reconstruction with matching detail crops](docs/images/encoding-example.png)
