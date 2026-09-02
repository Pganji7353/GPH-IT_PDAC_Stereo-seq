"""Shared helper imported by every step0N script: copies the specific PNG/PDF
pairs that end up as figures in the paper out of each script's own results
subfolder into one flat results/ folder at the top of the working directory,
renamed to match the paper's figure numbering (fig6_a.png, suppl9_sham.png,
etc.). All of a script's other diagnostic/exploratory output stays where it
already was; nothing extra is deleted, this only adds copies of the figures
that matter.
"""
import os
import shutil
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def collect(base_dir, mapping):
    """mapping: {source path relative to base_dir, without extension: dest stem}"""
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    for src_rel, dest_stem in mapping.items():
        found_any = False
        for ext in ("png", "pdf"):
            src = os.path.join(base_dir, f"{src_rel}.{ext}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, f"{dest_stem}.{ext}"))
                found_any = True
        if found_any:
            print(f"  Collected paper figure: {dest_stem}")
        else:
            print(f"  NOTE: expected paper figure not found (skipped): {src_rel}")


def _autocrop(im, pad=14, thresh=248):
    import numpy as np
    a = np.array(im.convert("L"))
    mask = a < thresh
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    l, r = max(xs.min() - pad, 0), min(xs.max() + pad, im.width)
    t, b = max(ys.min() - pad, 0), min(ys.max() + pad, im.height)
    return im.crop((l, t, r, b))


def stitch_grid_2x2(base_dir, sources, dest_stem, gap=20):
    """sources: list of 4 source paths (relative to base_dir, no extension),
    top-left/top-right/bottom-left/bottom-right order. Saves a PNG only
    (the four source PDFs stay selectable individually; a combined vector
    isn't produced here)."""
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(base_dir, f"{s}.png") for s in sources]
    if not all(os.path.exists(p) for p in paths):
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"  NOTE: {dest_stem} grid missing source(s), skipped: {missing}")
        return
    imgs = [_autocrop(Image.open(p).convert("RGB")) for p in paths]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    canvas = Image.new("RGB", (2 * w + gap, 2 * h + gap), "white")
    positions = [(0, 0), (w + gap, 0), (0, h + gap), (w + gap, h + gap)]
    for im, (x, y) in zip(imgs, positions):
        canvas.paste(im, (x, y))
    canvas.save(os.path.join(out_dir, f"{dest_stem}.png"))
    print(f"  Collected paper figure (2x2 grid): {dest_stem}")


def stitch_vertical(base_dir, sources, dest_stem, gap=40):
    """sources: list of source paths (relative to base_dir, no extension),
    stacked top to bottom. Saves both PNG and a PDF wrapping the same raster
    image (source PDFs stay selectable individually; this combined PDF is a
    flattened image, not a re-composed vector)."""
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(base_dir, f"{s}.png") for s in sources]
    if not all(os.path.exists(p) for p in paths):
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"  NOTE: {dest_stem} stack missing source(s), skipped: {missing}")
        return
    imgs = [_autocrop(Image.open(p).convert("RGB")) for p in paths]
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + gap
    canvas.save(os.path.join(out_dir, f"{dest_stem}.png"))
    canvas.save(os.path.join(out_dir, f"{dest_stem}.pdf"))
    print(f"  Collected paper figure (vertical stack): {dest_stem}")
