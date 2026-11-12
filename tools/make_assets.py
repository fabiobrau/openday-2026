"""Generate the one non-photographic asset: the noise decoy.

assets/decoys/noise.png is a random colourful scribble -- busy, and does
nothing. It is a third control used by tools/train_patch.py and
tools/validate_patch.py, never shown at the open day.

    .venv/bin/python tools/make_assets.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def noise_poster(px: int = 1024, blocks: int = 10, seed: int = 11) -> Image.Image:
    """Random colourful scribble. Looks busy, does nothing."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 255, (blocks, blocks, 3), dtype=np.uint8)
    img = Image.fromarray(small, "RGB").resize((px, px), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for _ in range(18):  # a few scribbled strokes so it reads as "drawing"
        pts = [(int(rng.integers(0, px)), int(rng.integers(0, px))) for _ in range(5)]
        d.line(pts, fill=tuple(int(v) for v in rng.integers(0, 255, 3)),
               width=int(rng.integers(8, 26)), joint="curve")
    return img


def main() -> None:
    dec = os.path.join(ROOT, "assets/decoys")
    os.makedirs(dec, exist_ok=True)
    p = os.path.join(dec, "noise.png")
    noise_poster().save(p)
    print("decoy:\n  wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
