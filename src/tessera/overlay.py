"""Tile overlay with a colour key (the demo pack's look): outlines of every tile on the image, the key below it."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .layout import COLOURS, KINDS, LABELS
from .stream.grid import S


def draw(img, tiles, key=True, scale=1):
    """img: PIL image or [H, W, 3] uint8 array (resized to 512 px); tiles: [(y, x, size_px, kind)].  Returns a PIL image:
    the picture with every tile outlined, and (key=True) a strip underneath naming the colours that occur."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, np.uint8))
    im = img.convert("RGB").resize((S, S), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for y, x, s, k in tiles:
        d.rectangle((x, y, x + s - 1, y + s - 1), outline=COLOURS[k], width=2 if s >= 32 else 1)
    if scale != 1:
        im = im.resize((int(S * scale),) * 2, Image.NEAREST)
    if not key:
        return im
    used = [k for k in KINDS if any(t[3] == k for t in tiles)]
    h = 28
    out = Image.new("RGB", (im.width, im.height + h), (20, 20, 20))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    f = ImageFont.load_default()
    x = 8
    for k in used:
        n = sum(1 for t in tiles if t[3] == k)
        lab = f"{LABELS[k]} ({n})"
        d.rectangle((x, im.height + 9, x + 12, im.height + 19), outline=COLOURS[k], width=2)
        d.text((x + 17, im.height + 8), lab, fill=(225, 225, 225), font=f)
        x += 17 + int(d.textlength(lab, font=f)) + 14
    return out
