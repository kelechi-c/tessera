"""`tessera`: standalone 2:1 k-d tree image codec (`encode` / `decode`) and batch Judge-verified evaluation CLI.

Standalone codec workflow (zero extra flags — resolves released checkpoints from Hugging Face cache on first use):

    tessera encode assets/veronese.png veronese.tsr
    tessera decode veronese.tsr reconstructed.png

Legacy batch evaluation CLI (preserved intact, including direct positional `OUT` paths like `./out`):

    tessera ./out --images PATH --ckpt tileup_rect.pt [--mode router --router router_rect.pt ...]
"""
import json
import math
import pathlib
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from tessera import __version__
from tessera.core.judge5 import load_dists_model
from tessera.latent import rect as RL
from tessera.latent import rect_router as RR
from tessera.latent import rect_tiledown as RTD
from tessera.rect.hub import (
    DEFAULT_REPO_ID,
    VERIFIED_ENCODE_DEFAULTS,
    load_flux_vae,
    resolve_ckpt_path,
    resolve_device,
    resolve_router_path,
)
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
from tessera.rect.tokens import (
    DEFAULT_SEED,
    CheckpointMismatchError,
    RectTokenFile,
    TokenFormatError,
    decode_tokens_to_latent,
    extract_leaf_tokens,
    file_sha256,
    load_tokens,
    save_tokens,
    verify_checkpoint_sha256,
)

_PALETTE = {
    (2, 2): (255, 110, 0), (2, 4): (255, 155, 40), (4, 2): (255, 155, 40), (4, 4): (245, 195, 75),
    (4, 8): (185, 220, 85), (8, 4): (185, 220, 85), (8, 8): (120, 215, 100), (8, 16): (70, 185, 110),
    (16, 8): (70, 185, 110), (16, 16): (45, 160, 70), (16, 32): (45, 145, 165), (32, 16): (45, 145, 165),
    (32, 32): (45, 130, 215),
}

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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


def _resolve_image_paths(images_path, limit=0):
    """Resolve `--images PATH` (a single image file or a directory of images) into a non-empty list of Paths."""
    src = pathlib.Path(images_path)
    if src.is_file():
        try:
            with Image.open(src) as im:
                im.verify()
        except Exception as exc:
            raise click.ClickException(f"Not a readable image file: {src} ({exc})") from exc
        paths = [src]
    elif src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    else:
        raise click.ClickException(f"Not a file or directory: {src}")
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise click.ClickException(f"No images found in {src}")
    return paths


def _load_images(images_path, limit, dev, paths=None, return_vae=False, repo_id=DEFAULT_REPO_ID):
    """Centre-crop and resize each image to 512 px, encode it with the FLUX.2 encoder -> [(name, z, u8)]."""
    if paths is None:
        paths = _resolve_image_paths(images_path, limit)

    dev_t = torch.device(dev)
    use_cuda = dev_t.type == "cuda"
    vae = load_flux_vae(repo_id=repo_id)
    cdt = torch.half if use_cuda else torch.float32   # fp16 convs are impractically slow on CPU
    enc = vae.encoder.to(cdt).eval().to(dev_t)
    vae.encoder = None
    qconv = getattr(vae, "quant_conv", None)
    if qconv is not None:
        qconv = qconv.to(cdt).eval().to(dev_t)
        vae.quant_conv = None
    if not return_vae:
        del vae
    out = []
    for p in paths:
        with Image.open(p) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
        if im.size != (512, 512):
            s = min(im.size)
            l, u = (im.width - s) // 2, (im.height - s) // 2
            im = im.crop((l, u, l + s, u + s)).resize((512, 512), Image.LANCZOS)
        u8 = np.asarray(im, dtype=np.uint8).copy()
        xt = (torch.from_numpy(u8).permute(2, 0, 1).unsqueeze(0).float().to(dev_t) / 127.5 - 1.0).to(cdt)
        with torch.no_grad():
            h = enc(xt)
            if qconv is not None:
                h = qconv(h)
            z = h.chunk(2, dim=1)[0].float().cpu()
        out.append((p.stem, z, u8))
    del enc, qconv
    if use_cuda:
        torch.cuda.empty_cache()
    return (out, vae) if return_vae else out


def _load_router(router_path, dev):
    ck_r = torch.load(router_path, map_location="cpu", weights_only=True)
    router_mod = RR.LatentRouterRect(
        width=int(ck_r["width"]),
        dilations=tuple(int(x) for x in ck_r["dilations"]),
        maxpool_feats=bool(ck_r.get("maxpool_feats", False)),
    ).to(dev).eval()
    router_mod.load_state_dict(ck_r["model"])
    return router_mod


def _configure_cuda_determinism(dev) -> None:
    """Match the deterministic, non-TF32 CUDA settings used by Judge v5 so encode and fresh-process decode agree."""
    if torch.device(dev).type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


@torch.no_grad()
def _encode_single_latent(
    z1_cpu,
    x_u8,
    *,
    dev,
    tu,
    td,
    basis_dev,
    mean,
    std,
    dec,
    mode,
    router_mod=None,
    leaf_bias=0.0,
    verify=True,
    verify_rounds=8,
    rd_lambda=0.05,
    rd_beta=0.5,
    rd_area_pow=0.0,
    rd_tau=0.0,
    judge_scale=3.0,
    jx=None,
    prune_merges=False,
    prune_fast=False,
    prune_max_calls=48,
    prune_max_rounds=8,
    seed=DEFAULT_SEED,
):
    """Run Judge-verified rectangular tokenization on one pre-encoded latent `z1_cpu` [1,32,64,64]."""
    dev_t = torch.device(dev)
    use_cuda = dev_t.type == "cuda"
    _configure_cuda_determinism(dev_t)
    mv = mean.view(1, RL.CH, 1, 1).to(dev_t)
    sv = std.view(1, RL.CH, 1, 1).to(dev_t)
    z1 = z1_cpu.to(dev_t).float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
        g_m11 = dec(z1).float().clamp(-1, 1)
    g_t = (g_m11 + 1.0) * 0.5
    g_u8 = (g_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    x_t = torch.from_numpy(x_u8).permute(2, 0, 1).unsqueeze(0).float().to(dev_t) / 255.0
    judge = Judge(x_u8, g_u8, dev, judge_scale=judge_scale, **(jx or {}))

    t_img0 = time.time()
    zn = (z1 - mv) / sv
    rect_pyr_n = RTD.encode_rect_pyramid_td(zn, basis_dev, tiledown=td)
    extra = {}
    if mode == "topdown":
        res = split_kdtree_topdown_td(tu, td, z1, basis_dev, mean, std, dec, judge, max_rounds=48)
        sm, y_m11, passed, calls = res["shape_map"], res["y"], bool(res["passed"]), int(res["calls"])
        toks = int(res["tokens"])
    else:
        err_maps = compute_lift_err_maps(
            z1,
            rect_pyr_n,
            basis_dev,
            mv,
            sv,
            tilelift=getattr(tu, "_tilelift", None),
            ac_floor=getattr(tu, "_ac_floor", 0.20),
            ac_tau=getattr(tu, "_ac_tau", 0.20),
        )
        if mode == "rdtree":
            sm0 = rd_layout(err_maps, rd_lambda, rd_beta, area_pow=rd_area_pow, tau=rd_tau).to(dev_t)[None]
        else:
            sm0 = router_mod.route(
                RR.rect_router_features(zn, basis_dev, tiledown=td),
                leaf_bias=leaf_bias,
                leaf_bias_slope=0.0,
            )
        last_y = [None]
        n_check = [0]
        route_tok = int(RL.tokens_of_rect(sm0)[0])

        def _check(sm_cand):
            n_check[0] += 1
            zc = predict_rect_canvas_td(tu, z1, rect_pyr_n, sm_cand, basis_dev, mean, std, seed=seed)
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
    y_u8 = (y_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return dict(
        shape_map=sm,
        rect_pyr_n=rect_pyr_n,
        y_m11=y_m11,
        y_t=y_t,
        y_u8=y_u8,
        g_t=g_t,
        g_u8=g_u8,
        x_t=x_t,
        passed=passed,
        calls=calls,
        toks=toks,
        n_merges=n_merges,
        ms=dt_ms,
        extra=extra,
    )


def encode_image(
    photo,
    output,
    *,
    repo=DEFAULT_REPO_ID,
    ckpt=None,
    router=None,
    device=None,
    mode=VERIFIED_ENCODE_DEFAULTS["mode"],
    leaf_bias=VERIFIED_ENCODE_DEFAULTS["leaf_bias"],
    verify=VERIFIED_ENCODE_DEFAULTS["verify"],
    verify_rounds=VERIFIED_ENCODE_DEFAULTS["verify_rounds"],
    rd_lambda=0.05,
    rd_beta=0.5,
    rd_area_pow=0.0,
    rd_tau=0.0,
    judge_scale=VERIFIED_ENCODE_DEFAULTS["judge_scale"],
    struct_scale=VERIFIED_ENCODE_DEFAULTS["struct_scale"],
    color_floor_de=VERIFIED_ENCODE_DEFAULTS["color_floor_de"],
    color_max_floor_de=VERIFIED_ENCODE_DEFAULTS["color_max_floor_de"],
    mae32_limit=None,
    tu_spike_k=VERIFIED_ENCODE_DEFAULTS["tu_spike_k"],
    tu_spike_min_area=VERIFIED_ENCODE_DEFAULTS["tu_spike_min_area"],
    prune_merges=False,
    prune_fast=False,
    prune_max_calls=48,
    prune_max_rounds=8,
    seed=DEFAULT_SEED,
    return_reference_u8=False,
):
    """Encode a single image file into a standalone versioned binary `.tsr` token file (no embedded pixels/latents)."""
    src = pathlib.Path(photo)
    if not src.is_file():
        raise FileNotFoundError(f"Input photo not found: {src}")
    paths = _resolve_image_paths(src, limit=1)

    dev = resolve_device(device)
    ckpt_path = resolve_ckpt_path(ckpt, repo_id=repo)
    router_path = resolve_router_path(router, repo_id=repo) if mode == "router" else None
    ckpt_sha = file_sha256(ckpt_path)

    entries, vae = _load_images(src, 1, dev, paths=paths, return_vae=True, repo_id=repo)
    tu, td, basis_dev, mean, std, _ = load_tokenizer_ckpt(ckpt_path, dev)
    if tu_spike_k > 0:
        tu._spike_k, tu._spike_min_area = float(tu_spike_k), int(tu_spike_min_area)
    router_mod = _load_router(router_path, dev) if router_path is not None else None
    dec = decoder_only_flux(dev, vae=vae, repo_id=repo)

    jx = {
        k: v
        for k, v in dict(
            color_floor_de=color_floor_de,
            color_max_floor_de=color_max_floor_de,
            struct_scale=struct_scale,
            mae32_limit=mae32_limit,
        ).items()
        if v is not None
    }
    _, z1_cpu, x_u8 = entries[0]
    res = _encode_single_latent(
        z1_cpu,
        x_u8,
        dev=dev,
        tu=tu,
        td=td,
        basis_dev=basis_dev,
        mean=mean,
        std=std,
        dec=dec,
        mode=mode,
        router_mod=router_mod,
        leaf_bias=leaf_bias,
        verify=verify,
        verify_rounds=verify_rounds,
        rd_lambda=rd_lambda,
        rd_beta=rd_beta,
        rd_area_pow=rd_area_pow,
        rd_tau=rd_tau,
        judge_scale=judge_scale,
        jx=jx,
        prune_merges=prune_merges,
        prune_fast=prune_fast,
        prune_max_calls=prune_max_calls,
        prune_max_rounds=prune_max_rounds,
        seed=seed,
    )

    leaves, leaf_tokens = extract_leaf_tokens(res["rect_pyr_n"], res["shape_map"])
    meta = dict(
        ckpt_name=ckpt_path.name,
        mode=mode,
        passed=bool(res["passed"]),
        calls=int(res["calls"]),
        verify=bool(verify),
    )
    if repo and ckpt is None:
        meta["repo"] = repo
    tf = RectTokenFile(
        shape_map=res["shape_map"][0],
        leaves=leaves,
        tokens=leaf_tokens,
        ckpt_sha256=ckpt_sha,
        seed=seed,
        tu_spike_k=float(tu_spike_k),
        tu_spike_min_area=int(tu_spike_min_area),
        meta=meta,
    )
    save_tokens(output, tf)
    return (tf, res["y_u8"]) if return_reference_u8 else tf


@torch.no_grad()
def decode_file(
    token_file,
    output,
    *,
    repo=DEFAULT_REPO_ID,
    ckpt=None,
    device=None,
    allow_ckpt_mismatch=False,
):
    """Decode a persisted `TSR2` token file to an RGB PNG/image without the source image, router, or Judge."""
    dev = resolve_device(device)
    dev_t = torch.device(dev)
    use_cuda = dev_t.type == "cuda"
    _configure_cuda_determinism(dev_t)

    tf = load_tokens(token_file)
    ckpt_path = resolve_ckpt_path(ckpt, repo_id=repo)
    if not allow_ckpt_mismatch:
        verify_checkpoint_sha256(tf, ckpt_path)

    tu, _, basis_dev, mean, std, _ = load_tokenizer_ckpt(ckpt_path, dev, load_tiledown=False)
    dec = decoder_only_flux(dev, repo_id=repo)
    z_hat = decode_tokens_to_latent(
        tf.shape_map,
        tf.tokens,
        tu,
        basis_dev,
        mean,
        std,
        seed=tf.seed,
        tu_spike_k=tf.tu_spike_k,
        tu_spike_min_area=tf.tu_spike_min_area,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
        y_m11 = dec(z_hat).float().clamp(-1, 1)
    y_t = (y_m11 + 1.0) * 0.5
    y_u8 = (y_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(y_u8).save(out_path)
    return y_u8, tf


# ---------------------------------------------------------------- CLI ------------------------------------------------
class _TesseraCLI(click.Group):
    """Click group preserving legacy positional `tessera OUT --images ...` invocations alongside `encode`/`decode`."""

    def parse_args(self, ctx, args):
        if args:
            first = args[0]
            if first not in ("encode", "decode", "eval", "-h", "--help", "--version"):
                args = ["eval", *args]
            elif first in ("encode", "decode") and "--images" in args:
                args = ["eval", *args]
        return super().parse_args(ctx, args)


@click.group(cls=_TesseraCLI, context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__, "--version")
def main():
    """Tessera: content-adaptive 2:1 k-d tree image tokenizer over FLUX.2 latents.

    \b
    Standalone codec:
      tessera encode PHOTO OUTPUT
      tessera decode FILE OUTPUT

    \b
    Batch evaluation (legacy compatible):
      tessera OUT --images PATH [--ckpt tileup_rect.pt ...]
    """


@main.command("encode")
@click.argument("photo", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--repo", default=DEFAULT_REPO_ID, show_default=True,
              help="Hugging Face repo ID for default tokenizer and router checkpoints")
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), default=None,
              help="local tokenizer checkpoint (overrides --repo)")
@click.option("--router", type=click.Path(exists=True, dir_okay=False), default=None,
              help="local router checkpoint (overrides --repo)")
@click.option("--device", default="auto", show_default=True,
              help="device: auto, cpu, cuda, or cuda:<index>")
@click.option("--mode", type=click.Choice(["router", "topdown", "rdtree"]),
              default=VERIFIED_ENCODE_DEFAULTS["mode"], show_default=True)
@click.option("--leaf_bias", type=float, default=VERIFIED_ENCODE_DEFAULTS["leaf_bias"], show_default=True,
              help="--mode router: added to leaf logit (more leaves, fewer tokens)")
@click.option("--verify/--no_verify", default=VERIFIED_ENCODE_DEFAULTS["verify"], show_default=True,
              help="--mode router/rdtree: split Judge-failing leaves until accepted")
@click.option("--verify_rounds", type=int, default=VERIFIED_ENCODE_DEFAULTS["verify_rounds"], show_default=True,
              help="--mode router/rdtree --verify: max split rounds before full-grid fallback")
@click.option("--rd_lambda", type=float, default=0.05, show_default=True, help="--mode rdtree: per-token cost lambda")
@click.option("--rd_beta", type=float, default=0.5, show_default=True, help="--mode rdtree: weight of max(e)")
@click.option("--rd_area_pow", type=float, default=0.0, show_default=True, help="--mode rdtree: area scaling power")
@click.option("--rd_tau", type=float, default=0.0, show_default=True, help="--mode rdtree: distortion noise floor")
@click.option("--judge_scale", type=float, default=VERIFIED_ENCODE_DEFAULTS["judge_scale"], show_default=True,
              help="Judge tolerance multiplier")
@click.option("--struct_scale", type=float, default=VERIFIED_ENCODE_DEFAULTS["struct_scale"], show_default=True,
              help="scale for non-colour Judge gates and mae32")
@click.option("--color_floor_de", type=float, default=VERIFIED_ENCODE_DEFAULTS["color_floor_de"], show_default=True,
              help="floor color_mean256 pass threshold in Lab dE")
@click.option("--color_max_floor_de", type=float, default=VERIFIED_ENCODE_DEFAULTS["color_max_floor_de"],
              show_default=True, help="floor color_max256 pass threshold in Lab dE")
@click.option("--mae32_limit", type=float, default=None, help="absolute mae32 limit overriding 2.5*struct_scale")
@click.option("--tu_spike_k", type=float, default=VERIFIED_ENCODE_DEFAULTS["tu_spike_k"], show_default=True,
              help="TileUp residual spike clamp multiplier (0 = off)")
@click.option("--tu_spike_min_area", type=int, default=VERIFIED_ENCODE_DEFAULTS["tu_spike_min_area"], show_default=True,
              help="smallest tile area (cells) clipped by --tu_spike_k")
@click.option("--prune_merges", is_flag=True, default=False, help="Judge-verified sibling merge pruning")
@click.option("--prune_fast", is_flag=True, default=False, help="skip bisecting dirty chunks during pruning")
@click.option("--prune_max_calls", type=int, default=48, show_default=True, help="max Judge calls during pruning")
@click.option("--prune_max_rounds", type=int, default=8, show_default=True, help="max sibling-merge rounds")
def encode_cmd(
    photo, output, repo, ckpt, router, device, mode, leaf_bias, verify, verify_rounds,
    rd_lambda, rd_beta, rd_area_pow, rd_tau, judge_scale, struct_scale, color_floor_de,
    color_max_floor_de, mae32_limit, tu_spike_k, tu_spike_min_area, prune_merges,
    prune_fast, prune_max_calls, prune_max_rounds,
):
    """Encode PHOTO into a standalone versioned binary token file OUTPUT (.tsr)."""
    try:
        t0 = time.time()
        tf = encode_image(
            photo,
            output,
            repo=repo,
            ckpt=ckpt,
            router=router,
            device=device,
            mode=mode,
            leaf_bias=leaf_bias,
            verify=verify,
            verify_rounds=verify_rounds,
            rd_lambda=rd_lambda,
            rd_beta=rd_beta,
            rd_area_pow=rd_area_pow,
            rd_tau=rd_tau,
            judge_scale=judge_scale,
            struct_scale=struct_scale,
            color_floor_de=color_floor_de,
            color_max_floor_de=color_max_floor_de,
            mae32_limit=mae32_limit,
            tu_spike_k=tu_spike_k,
            tu_spike_min_area=tu_spike_min_area,
            prune_merges=prune_merges,
            prune_fast=prune_fast,
            prune_max_calls=prune_max_calls,
            prune_max_rounds=prune_max_rounds,
        )
    except click.ClickException:
        raise
    except (ValueError, FileNotFoundError, RuntimeError, TokenFormatError) as exc:
        raise click.ClickException(str(exc)) from exc

    size_kb = pathlib.Path(output).stat().st_size / 1024.0
    dt = time.time() - t0
    click.echo(
        f"[tessera encode] {tf.n_tokens} tokens, pass={tf.meta.get('passed')}, "
        f"calls={tf.meta.get('calls')}, {size_kb:.1f} KiB, {dt:.1f}s -> {output}"
    )


@main.command("decode")
@click.argument("token_file", metavar="FILE", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--repo", default=DEFAULT_REPO_ID, show_default=True,
              help="Hugging Face repo ID for the default tokenizer checkpoint")
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), default=None,
              help="local tokenizer checkpoint (overrides --repo)")
@click.option("--device", default="auto", show_default=True,
              help="device: auto, cpu, cuda, or cuda:<index>")
@click.option("--allow-ckpt-mismatch", "--allow_ckpt_mismatch", is_flag=True, default=False,
              help="decode even if the tokenizer checkpoint SHA-256 differs from the token file header")
def decode_cmd(token_file, output, repo, ckpt, device, allow_ckpt_mismatch):
    """Decode a versioned Tessera token FILE (.tsr) into a reconstructed image OUTPUT."""
    try:
        t0 = time.time()
        _, tf = decode_file(
            token_file,
            output,
            repo=repo,
            ckpt=ckpt,
            device=device,
            allow_ckpt_mismatch=allow_ckpt_mismatch,
        )
    except click.ClickException:
        raise
    except (ValueError, FileNotFoundError, RuntimeError, TokenFormatError, CheckpointMismatchError) as exc:
        raise click.ClickException(str(exc)) from exc

    dt = time.time() - t0
    click.echo(f"[tessera decode] {tf.n_tokens} tokens, {dt:.1f}s -> {output}")


@main.command("eval")
@click.argument("out", type=click.Path())
@click.option("--images", type=click.Path(exists=True, file_okay=True, dir_okay=True), required=True,
              help="image file or folder of images (png/jpg/webp)")
@click.option("--repo", default=DEFAULT_REPO_ID, show_default=True,
              help="Hugging Face repo ID used when --ckpt or --router is omitted")
@click.option("--ckpt", type=click.Path(exists=True), default=None,
              help="tileup_rect.pt tokenizer checkpoint (default: resolved from --repo)")
@click.option("--mode", type=click.Choice(["topdown", "router", "rdtree"]), default="topdown", show_default=True)
@click.option("--router", type=click.Path(exists=True), default=None,
              help="router_rect.pt (inference weights) for --mode router (default: resolved from --repo)")
@click.option("--device", default="auto", show_default=True,
              help="device: auto, cpu, cuda, or cuda:<index>")
@click.option("--metrics/--no-metrics", default=True, show_default=True,
              help="compute summary PSNR/SSIM/LPIPS/DISTS metrics (Judge verification always runs)")
@click.option("--no_metrics", "metrics", flag_value=False, hidden=True)
@click.option("--leaf_bias", type=float, default=0.0, show_default=True, help="--mode router: added to the leaf logit (more leaves, fewer tokens)")
@click.option("--verify/--no_verify", default=True, show_default=True,
              help="--mode router/rdtree: split Judge-failing leaves of the one-shot layout until it passes")
@click.option("--verify_rounds", type=int, default=8, show_default=True,
              help="--mode router/rdtree --verify: max split rounds before the all-2x2 fallback")
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
def eval_cmd(
    out, images, repo, ckpt, mode, router, device, metrics, leaf_bias, verify, verify_rounds, rd_lambda, rd_beta,
    rd_area_pow, rd_tau, judge_scale, struct_scale, color_floor_de, color_max_floor_de, mae32_limit, tu_spike_k,
    tu_spike_min_area, prune_merges, prune_fast, prune_max_calls, prune_max_rounds, limit, strips, save_layouts,
):
    """Evaluate the image or images at --images to Judge-verified 2:1 k-d tree layouts; results go to OUT."""
    try:
        dev = resolve_device(device)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    paths = _resolve_image_paths(images, limit)
    try:
        ckpt_path = resolve_ckpt_path(ckpt, repo_id=repo)
        router_path = resolve_router_path(router, repo_id=repo) if mode == "router" else None
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    jx = {
        k: v
        for k, v in dict(
            color_floor_de=color_floor_de,
            color_max_floor_de=color_max_floor_de,
            struct_scale=struct_scale,
            mae32_limit=mae32_limit,
        ).items()
        if v is not None
    }
    out_path = pathlib.Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    strips_dir = out_path / "strips"
    if strips > 0:
        strips_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries, vae = _load_images(images, limit, dev, paths=paths, return_vae=True, repo_id=repo)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    tu, td, basis_dev, mean, std, _ = load_tokenizer_ckpt(ckpt_path, dev)
    router_mod = _load_router(router_path, dev) if router_path is not None else None
    if tu_spike_k > 0:
        tu._spike_k, tu._spike_min_area = float(tu_spike_k), int(tu_spike_min_area)

    dec = decoder_only_flux(dev, vae=vae, repo_id=repo)
    lpips_fn = dists_fn = None
    if metrics:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
        dists_fn = load_dists_model(dev)

    def calc_metrics(a01, b01):
        mse = float(((a01 - b01) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim = _ssim_torch(a01, b01)
        with torch.no_grad():
            lp = float(lpips_fn(a01 * 2.0 - 1.0, b01 * 2.0 - 1.0))
            ds = float(dists_fn(a01, b01))
        return dict(psnr=psnr, ssim=ssim, lpips=lp, dists=ds)

    tl_m = getattr(tu, "_tilelift", None)
    print(
        f"[tessera] dev={dev} n={len(entries)} mode={mode} metrics={metrics} "
        f"TileUp={nparams(tu) / 1e6:.2f}M RectTileDown={(nparams(td) / 1e6 if td is not None else 0.0):.2f}M "
        f"RectTileLift={(nparams(tl_m) / 1e6 if tl_m is not None else 0.0):.2f}M",
        flush=True,
    )

    rows = []
    with torch.no_grad():
        for idx, (rid, z1_cpu, x_u8) in enumerate(entries):
            res = _encode_single_latent(
                z1_cpu,
                x_u8,
                dev=dev,
                tu=tu,
                td=td,
                basis_dev=basis_dev,
                mean=mean,
                std=std,
                dec=dec,
                mode=mode,
                router_mod=router_mod,
                leaf_bias=leaf_bias,
                verify=verify,
                verify_rounds=verify_rounds,
                rd_lambda=rd_lambda,
                rd_beta=rd_beta,
                rd_area_pow=rd_area_pow,
                rd_tau=rd_tau,
                judge_scale=judge_scale,
                jx=jx,
                prune_merges=prune_merges,
                prune_fast=prune_fast,
                prune_max_calls=prune_max_calls,
                prune_max_rounds=prune_max_rounds,
            )
            sm, y_t, g_t, x_t = res["shape_map"], res["y_t"], res["g_t"], res["x_t"]
            y_u8, g_u8 = res["y_u8"], res["g_u8"]
            toks, passed, calls, n_merges, dt_ms = res["toks"], res["passed"], res["calls"], res["n_merges"], res["ms"]

            row = dict(
                id=rid,
                tokens=toks,
                passed=passed,
                calls=calls,
                merges=n_merges,
                ms=round(dt_ms, 1),
                **res["extra"],
            )
            if metrics:
                row["tg"] = calc_metrics(y_t, g_t)   # decode vs the FLUX.2 round trip (what the Judge gates)
                row["to"] = calc_metrics(y_t, x_t)   # decode vs the original
                row["go"] = calc_metrics(g_t, x_t)   # FLUX.2 round trip vs the original
            rows.append(row)

            if save_layouts:
                lay_dir = out_path / "layouts"
                lay_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    lay_dir / f"{idx:03d}_{rid[:40]}.npz",
                    sm=sm[0].cpu().to(torch.uint8).numpy(),
                    x=x_u8,
                    y=y_u8,
                    g=g_u8,
                    id=np.array(rid),
                    tokens=np.array(toks),
                    passed=np.array(bool(passed)),
                )

            if idx < strips:
                strip = np.concatenate([x_u8, y_u8, _overlay(y_u8, sm[0])], axis=1)
                Image.fromarray(strip).save(strips_dir / f"{idx:03d}_{rid[:24]}_{toks}tok.png")

            print(f"  [{idx + 1}/{len(entries)}] {rid[:24]} tok={toks} pass={passed} calls={calls} {dt_ms:.0f}ms", flush=True)

    meta = dict(
        mode=mode,
        device=dev,
        metrics=bool(metrics),
        verify=verify,
        judge_scale=judge_scale,
        prune_merges=bool(prune_merges),
        leaf_bias=leaf_bias,
        prune_fast=bool(prune_fast),
        prune_max_calls=int(prune_max_calls),
        prune_max_rounds=int(prune_max_rounds),
        verify_rounds=int(verify_rounds),
        rd_lambda=rd_lambda,
        rd_beta=rd_beta,
        rd_area_pow=rd_area_pow,
        rd_tau=rd_tau,
        **jx,
    )
    if tu_spike_k > 0:
        meta.update(tu_spike_k=float(tu_spike_k), tu_spike_min_area=int(tu_spike_min_area))
    (out_path / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))
    n_pass = sum(bool(r["passed"]) for r in rows)
    print(
        f"[tessera] {n_pass}/{len(rows)} pass the Judge, mean {np.mean([r['tokens'] for r in rows]):.1f} tokens "
        f"-> {out_path / 'results.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
