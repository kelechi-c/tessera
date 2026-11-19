"""tessera command line: encode / decode / judge / overlay / info."""
import json
import pathlib

import click
from PIL import Image


@click.group()
def main():
    """tessera: quadtree image tokens."""


_w = click.option("--weights", type=click.Path(), default=None, help="directory with bases.pt (and router.pt); or set TESSERA_WEIGHTS")
_d = click.option("--device", default=None, help="cuda or cpu (default: cuda if available)")


@main.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path(), help="tokens file (.tsr)")
@click.option("--layout", type=click.Choice(["search", "router"]), default="search", show_default=True,
              help="how the layout is found; 'router' needs a trained router.pt (none ships yet)")
@click.option("--router", type=click.Path(), default=None, help="router checkpoint (default WEIGHTS/router.pt)")
@click.option("--allow-untrained", is_flag=True, help="router: fall back to a random-init network with a warning (noise; smoke tests only)")
@click.option("--arm", type=click.Choice(["both", "A", "vae16", "grid"]), default="both", show_default=True)
@click.option("--record", type=click.Path(), default=None, help="append a router training record (layout + verdict) to this jsonl")
@_w
@_d
def encode(image, out, layout, router, allow_untrained, arm, record, weights, device):
    """Encode IMAGE (resized to 512 px) into tokens."""
    from .router import LayoutRecord, NotTrainedError, append_record
    from .search import Encoder
    from .tsr import save

    try:
        enc = Encoder(weights, device, layout, router, allow_untrained)
    except NotTrainedError as e:
        raise click.ClickException(str(e))
    im = Image.open(image)
    r = enc.encode(im, arm)
    save(out, r.tsr)
    v = r.verdict
    click.echo(json.dumps(dict(out=str(out), arm=r.tsr.arm, tokens=round(r.tsr.n_tokens(), 2), passed=bool(v["passed"]),
                               score=round(float(v["score"]), 4), T=round(float(v["T"]), 4), argmax=v["argmax"])))
    if record and getattr(enc, "last", None) is not None:
        import hashlib

        L = enc.last
        append_record(record, LayoutRecord(pathlib.Path(image).name, L["lvl"][0].numpy(), L["two"][0].numpy(),
                                           None if L["sel"] is None else L["sel"].numpy(), float(r.tsr.n_tokens()), L["arm"],
                                           r.tsr.meta["verdict"], hashlib.md5(L["x"].tobytes()).hexdigest()))


@main.command()
@click.argument("tokens", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path(), help="output image (.png)")
@_w
@_d
def decode(tokens, out, weights, device):
    """Decode a .tsr file to an image."""
    from .search import decode_tsr
    from .tsr import load

    Image.fromarray(decode_tsr(load(tokens), weights, device)).save(out)
    click.echo(out)


@main.command("judge")
@click.argument("orig", type=click.Path(exists=True))
@click.argument("vae", type=click.Path(exists=True))
@click.argument("cand", type=click.Path(exists=True))
@_d
def judge_cmd(orig, vae, cand, device):
    """Judge v5 verdict of CAND against ORIG, relative to VAE (the FLUX.2 round trip of ORIG). Exit 0 = pass, 1 = fail."""
    from .judge import judge

    r = judge(orig, vae, cand, device)
    click.echo(json.dumps(r))
    raise SystemExit(0 if r["passed"] else 1)


@main.command()
@click.argument("image", type=click.Path(exists=True))
@click.argument("tokens", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path())
@click.option("--no-key", is_flag=True, help="omit the colour key strip")
def overlay(image, tokens, out, no_key):
    """Draw the tiles of a .tsr file over IMAGE, with a colour key.  Needs no weights."""
    from .layout import tiles_from_tokens
    from .overlay import draw
    from .tsr import load

    t = load(tokens)
    if t.tokens is None:
        raise click.ClickException("lowres files are one whole-image raster: no tile layout to draw")
    draw(Image.open(image), tiles_from_tokens(t.tokens), key=not no_key).save(out)
    click.echo(out)


@main.command()
@click.argument("tokens", type=click.Path(exists=True))
def info(tokens):
    """Print a .tsr file's arm, token count, tile counts and recorded verdict (no weights)."""
    from .layout import tiles_from_tokens
    from .tsr import load

    t = load(tokens)
    d = dict(arm=t.arm, tokens=round(t.n_tokens(), 2), meta=t.meta)
    if t.tokens is not None:
        ts = tiles_from_tokens(t.tokens)
        d["tiles"] = {k: sum(1 for q in ts if q[3] == k) for k in sorted({q[3] for q in ts})}
    click.echo(json.dumps(d, default=str))


if __name__ == "__main__":
    main()
