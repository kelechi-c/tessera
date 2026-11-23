"""`tessera`: encode 512 px images into 2:1 k-d tree layouts of FLUX.2 latent tokens, judged by the frozen Judge v5.

Per image: FLUX.2 encoder -> latent z; a layout (`shape_map`) is found so that the decoded tile canvas passes the
Judge against the plain FLUX.2 round trip (`--mode topdown`: split Judge-failing leaves until it passes; `--mode
rdtree`: one-shot rate-distortion tree, optionally verified by splitting).  `--prune_merges` then merges sibling
leaves while the Judge still passes.  Output: `results.json`, comparison strips (original | decode | tiles) and with
`--save_layouts` the layouts themselves.

    tessera OUT --images DIR --ckpt tileup_rect.pt
"""
import json
import math
import pathlib
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tessera.latent import rect as RL
from tessera.latent import rect_router as RR
from tessera.latent import rect_tiledown as RTD
from tessera.rect.judge import Judge
from tessera.rect.model import decoder_only_flux, nparams
from tessera.rect.search import (
    best_hv_split_fn,
    compute_lift_err_maps,
    load_tokenizer_ckpt,
    predict_rect_canvas_td,
    prune_kdtree_siblings_td,
    rd_layout,
    split_kdtree_topdown_td,
)

_PALETTE = {
    (2, 2): (255, 110, 0), (2, 4): (255, 155, 40), (4, 2): (255, 155, 40), (4, 4): (245, 195, 75),
    (4, 8): (185, 220, 85), (8, 4): (185, 220, 85), (8, 8): (120, 215, 100), (8, 16): (70, 185, 110),
    (16, 8): (70, 185, 110), (16, 16): (45, 160, 70), (16, 32): (45, 145, 165), (32, 16): (45, 145, 165),
    (32, 32): (45, 130, 215),
}


def _overlay(u8, shape_map_1):
    """Draw the tile borders of a layout over an image (colour by tile shape)."""
    a = u8.copy()
    for r0, c0, sh, sw in RL.leaves_rect(shape_map_1):
        y0, x0, h, w = r0 * 8, c0 * 8, sh * 8, sw * 8
        col = np.array(_PALETTE.get((sh, sw), (255, 255, 255)), dtype=np.uint8)
        a[y0, x0:x0 + w] = col
        a[y0:y0 + h, x0] = col
    return a


def _ssim_torch(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = 11
    coords = torch.arange(k, device=a.device, dtype=a.dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(3, 1, k, k)
    mu1 = F.conv2d(a, win, padding=k // 2, groups=3)
    mu2 = F.conv2d(b, win, padding=k // 2, groups=3)
    s1 = F.conv2d(a * a, win, padding=k // 2, groups=3) - mu1 * mu1
    s2 = F.conv2d(b * b, win, padding=k // 2, groups=3) - mu2 * mu2
    s12 = F.conv2d(a * b, win, padding=k // 2, groups=3) - mu1 * mu2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
    return float((num / den).mean())


def _load_images(images_dir, limit, dev):
    """Centre-crop and resize each image to 512 px, encode it with the FLUX.2 encoder -> [(name, z, u8)]."""
    exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = sorted(p for p in pathlib.Path(images_dir).iterdir() if p.suffix.lower() in exts)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise click.ClickException(f"No images found in {images_dir}")

    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.float32)
    cdt = torch.half if str(dev).startswith("cuda") else torch.float32   # fp16 convs are impractically slow on CPU
    enc = vae.encoder.to(cdt).eval().to(dev)
    qconv = getattr(vae, "quant_conv", None)
    qconv = qconv.to(cdt).eval().to(dev) if qconv is not None else None
    del vae
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (512, 512):
            s = min(im.size)
            l, u = (im.width - s) // 2, (im.height - s) // 2
            im = im.crop((l, u, l + s, u + s)).resize((512, 512), Image.LANCZOS)
        u8 = np.asarray(im, dtype=np.uint8).copy()
        xt = (torch.from_numpy(u8).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 127.5 - 1.0).to(cdt)
        with torch.no_grad():
            h = enc(xt)
            if qconv is not None:
                h = qconv(h)
            z = h.chunk(2, dim=1)[0].float().cpu()
        out.append((p.stem, z, u8))
    del enc, qconv
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


@click.command()
@click.argument("out", type=click.Path())
@click.option("--images", type=click.Path(exists=True, file_okay=False), required=True, help="folder of images (png/jpg/webp)")
@click.option("--ckpt", type=click.Path(exists=True), required=True, help="tileup_rect.pt tokenizer checkpoint")
@click.option("--mode", type=click.Choice(["topdown", "rdtree"]), default="topdown", show_default=True)
@click.option("--verify/--no_verify", default=True, show_default=True,
              help="--mode rdtree: split Judge-failing leaves of the one-shot tree until it passes")
@click.option("--verify_rounds", type=int, default=8, show_default=True,
              help="--mode rdtree --verify: max split rounds before the all-2x2 fallback")
@click.option("--rd_lambda", type=float, default=0.05, show_default=True, help="--mode rdtree: per-token cost lambda")
@click.option("--rd_beta", type=float, default=0.5, show_default=True, help="--mode rdtree: weight of max(e) in node distortion")
@click.option("--rd_area_pow", type=float, default=0.0, show_default=True, help="--mode rdtree: scale node D by (sh*sw/4)**p")
@click.option("--rd_tau", type=float, default=0.0, show_default=True, help="--mode rdtree: noise floor, D = relu(mean + beta*max - tau)")
@click.option("--judge_scale", type=float, default=3.0, show_default=True, help="Judge tolerance multiplier (lower is stricter)")
@click.option("--struct_scale", type=float, default=None,
              help="scale for all non-colour Judge gates and the mae32 limit (colour keeps --judge_scale); default off")
@click.option("--color_floor_de", type=float, default=None, help="floor color_mean256's pass threshold at this Lab dE; default off")
@click.option("--color_max_floor_de", type=float, default=None, help="floor color_max256's pass threshold at this Lab dE; default off")
@click.option("--mae32_limit", type=float, default=None,
              help="absolute mae32 limit (RGB levels) overriding 2.5*struct_scale; default off")
@click.option("--tu_spike_k", type=float, default=0.0, show_default=True,
              help="soft-clip TileUp residual spikes above k * median|hp| per tile/channel; 0 = off")
@click.option("--tu_spike_min_area", type=int, default=64, show_default=True, help="--tu_spike_k: smallest tile area (cells) clipped")
@click.option("--prune_merges", is_flag=True, default=False, help="Judge-verified sibling merge pruning of every passing layout")
@click.option("--prune_fast", is_flag=True, default=False, help="skip bisecting Judge-dirty sibling chunks during pruning")
@click.option("--prune_max_calls", type=int, default=48, show_default=True, help="max Judge calls per image while pruning (0 = unlimited)")
@click.option("--prune_max_rounds", type=int, default=8, show_default=True, help="max sibling-merge rounds per image while pruning")
@click.option("--limit", type=int, default=0, show_default=True, help="encode only the first N images (0 = all)")
@click.option("--strips", type=int, default=12, show_default=True, help="write comparison strips for the first N images")
@click.option("--save_layouts", is_flag=True, default=False,
              help="also write OUT/layouts/<idx>_<id>.npz (shape_map uint8 [2,64,64], original, decode, FLUX.2 round trip)")
def main(out, images, ckpt, mode, verify, verify_rounds, rd_lambda, rd_beta, rd_area_pow, rd_tau, judge_scale, struct_scale,
         color_floor_de, color_max_floor_de, mae32_limit, tu_spike_k, tu_spike_min_area, prune_merges, prune_fast,
         prune_max_calls, prune_max_rounds, limit, strips, save_layouts):
    """Encode the images in --images to Judge-verified 2:1 k-d tree layouts; results go to OUT."""
    jx = {k: v for k, v in dict(color_floor_de=color_floor_de, color_max_floor_de=color_max_floor_de,
                                struct_scale=struct_scale, mae32_limit=mae32_limit).items() if v is not None}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = dev.startswith("cuda")
    out_path = pathlib.Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    strips_dir = out_path / "strips"
    if strips > 0:
        strips_dir.mkdir(parents=True, exist_ok=True)

    tu, td, basis_dev, mean, std, _ = load_tokenizer_ckpt(ckpt, dev)
    mv, sv = mean.view(1, RL.CH, 1, 1).to(dev), std.view(1, RL.CH, 1, 1).to(dev)
    if tu_spike_k > 0:
        tu._spike_k, tu._spike_min_area = float(tu_spike_k), int(tu_spike_min_area)

    import lpips
    from DISTS_pytorch import DISTS

    dec = decoder_only_flux(dev)
    lpips_fn = lpips.LPIPS(net="alex").to(dev).eval().requires_grad_(False)
    dists_fn = DISTS().to(dev).eval().requires_grad_(False)

    def calc_metrics(a01, b01):
        mse = float(((a01 - b01) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim = _ssim_torch(a01, b01)
        with torch.no_grad():
            lp = float(lpips_fn(a01 * 2.0 - 1.0, b01 * 2.0 - 1.0))
            ds = float(dists_fn(a01, b01))
        return dict(psnr=psnr, ssim=ssim, lpips=lp, dists=ds)

    entries = _load_images(images, limit, dev)
    tl_m = getattr(tu, "_tilelift", None)
    print(
        f"[tessera] dev={dev} n={len(entries)} mode={mode} "
        f"TileUp={nparams(tu) / 1e6:.2f}M RectTileDown={(nparams(td) / 1e6 if td is not None else 0.0):.2f}M "
        f"RectTileLift={(nparams(tl_m) / 1e6 if tl_m is not None else 0.0):.2f}M",
        flush=True,
    )

    rows = []
    with torch.no_grad():
        for idx, (rid, z1_cpu, x_u8) in enumerate(entries):
            z1 = z1_cpu.to(dev).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                g_m11 = dec(z1).float().clamp(-1, 1)
            g_t = (g_m11 + 1.0) * 0.5
            g_u8 = (g_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            x_t = torch.from_numpy(x_u8).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
            judge = Judge(x_u8, g_u8, dev, judge_scale=judge_scale, **jx)

            t_img0 = time.time()
            extra = {}
            if mode == "topdown":
                res = split_kdtree_topdown_td(tu, td, z1, basis_dev, mean, std, dec, judge, max_rounds=48)
                sm, y_m11, passed, calls = res["shape_map"], res["y"], res["passed"], res["calls"]
                toks = int(res["tokens"])
            else:
                zn = (z1 - mv) / sv
                rect_pyr_n = RTD.encode_rect_pyramid_td(zn, basis_dev, tiledown=td)
                err_maps = compute_lift_err_maps(z1, rect_pyr_n, basis_dev, mv, sv, tilelift=getattr(tu, "_tilelift", None),
                                                 ac_floor=getattr(tu, "_ac_floor", 0.20), ac_tau=getattr(tu, "_ac_tau", 0.20))
                sm0 = rd_layout(err_maps, rd_lambda, rd_beta, area_pow=rd_area_pow, tau=rd_tau).to(dev)[None]
                last_y = [None]
                n_check = [0]
                route_tok = int(RL.tokens_of_rect(sm0)[0])

                def _check(sm_cand):
                    n_check[0] += 1
                    zc = predict_rect_canvas_td(tu, z1, rect_pyr_n, sm_cand, basis_dev, mean, std, seed=0)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                        yc = dec(zc).float().clamp(-1, 1)
                    last_y[0] = yc
                    return judge(yc)

                if verify:
                    vres = RR.verify_rect(sm0, _check, max_rounds=verify_rounds, split_choice_fn=best_hv_split_fn(err_maps))
                    sm, y_m11, passed, calls = vres["shape_map"], last_y[0], bool(vres["passed"]), int(vres["rounds"])
                    toks = int(vres["tokens"])
                    extra = dict(route_tok=route_tok, fallback=bool(vres["fallback"]), judge_calls=n_check[0])
                else:
                    ev = _check(sm0)
                    sm, y_m11, passed, calls = sm0, last_y[0], bool(ev["passed"]), 1
                    toks = route_tok
                    extra = dict(route_tok=route_tok, fallback=False, judge_calls=n_check[0])
            n_merges = 0
            if prune_merges and passed:
                pr = prune_kdtree_siblings_td(
                    tu, td, z1, basis_dev, mean, std, dec, judge, sm,
                    bisect_dirty=(not prune_fast), max_calls=prune_max_calls, max_rounds=prune_max_rounds,
                )
                sm, y_m11, passed = pr["shape_map"], pr["y"], bool(pr["passed"])
                calls, toks, n_merges = int(calls) + int(pr["prune_calls"]), int(pr["tokens"]), int(pr["merges"])
                if extra:
                    extra["prune_calls"] = int(pr["prune_calls"])
            dt_ms = (time.time() - t_img0) * 1000.0

            y_t = (y_m11.float().clamp(-1, 1) + 1.0) * 0.5
            rows.append(dict(
                id=rid, tokens=toks, passed=passed, calls=calls, merges=n_merges, ms=round(dt_ms, 1),
                tg=calc_metrics(y_t, g_t),      # decode vs the FLUX.2 round trip (what the Judge gates)
                to=calc_metrics(y_t, x_t),      # decode vs the original
                go=calc_metrics(g_t, x_t),      # FLUX.2 round trip vs the original
                **extra,
            ))
            y_u8 = (y_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            if save_layouts:
                lay_dir = out_path / "layouts"
                lay_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    lay_dir / f"{idx:03d}_{rid[:40]}.npz", sm=sm[0].cpu().to(torch.uint8).numpy(), x=x_u8, y=y_u8, g=g_u8,
                    id=np.array(rid), tokens=np.array(toks), passed=np.array(bool(passed)))

            if idx < strips:
                strip = np.concatenate([x_u8, y_u8, _overlay(y_u8, sm[0])], axis=1)
                Image.fromarray(strip).save(strips_dir / f"{idx:03d}_{rid[:24]}_{toks}tok.png")

            print(f"  [{idx + 1}/{len(entries)}] {rid[:24]} tok={toks} pass={passed} calls={calls} {dt_ms:.0f}ms", flush=True)

    meta = dict(mode=mode, verify=verify, judge_scale=judge_scale, prune_merges=bool(prune_merges),
                prune_fast=bool(prune_fast), prune_max_calls=int(prune_max_calls), prune_max_rounds=int(prune_max_rounds),
                verify_rounds=int(verify_rounds), rd_lambda=rd_lambda, rd_beta=rd_beta, rd_area_pow=rd_area_pow, rd_tau=rd_tau,
                **jx)
    if tu_spike_k > 0:
        meta.update(tu_spike_k=float(tu_spike_k), tu_spike_min_area=int(tu_spike_min_area))
    (out_path / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))
    n_pass = sum(bool(r["passed"]) for r in rows)
    print(f"[tessera] {n_pass}/{len(rows)} pass the Judge, mean {np.mean([r['tokens'] for r in rows]):.1f} tokens "
          f"-> {out_path / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
