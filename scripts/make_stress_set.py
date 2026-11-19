"""Build a degraded copy of the labelled set to probe failure modes.

The labelled documents are clean enough that the baseline makes no errors,
which leaves nothing to learn from. This script simulates the "phone scan"
conditions from the brief by re-rendering every labelled document as a
degraded image. The *content* is unchanged, so the original ground truth
stays valid and is copied verbatim (never edited).

Degradation levels (applied in order, deterministic with a fixed seed):
  mild:   long edge 1100 px, 1.5 deg skew, gaussian blur r=0.8, JPEG q=45
  severe: long edge 800 px,  3 deg skew, gaussian blur r=1.3, sensor noise, JPEG q=25

Usage:  python scripts/make_stress_set.py [--level mild|severe]
Output: data/stress_<level>/<id>.jpg + <id>.json
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEVELS = {
    "mild": {"edge": 1100, "skew": 1.5, "blur": 0.8, "noise": 0, "quality": 45},
    "severe": {"edge": 800, "skew": 3.0, "blur": 1.3, "noise": 18, "quality": 25},
    # Held-out set: created AFTER all design decisions, different seed and parameters; used only to
    # compare final configurations, never to choose thresholds or rules.
    "mild_b": {"edge": 1000, "skew": 2.2, "blur": 1.0, "noise": 8, "quality": 38, "seed": 2027},
    # Clean pages turned sideways, to test whether rotation itself hurts extraction.
    "rot90": {"rotate": 90},
    "rot270": {"rotate": 270},
}


def render_first_pages(path: Path) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        pages = []
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=150, alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    return [Image.open(path).convert("RGB")]


def degrade(img: Image.Image, cfg: dict, rng: random.Random) -> Image.Image:
    if "rotate" in cfg:
        # PIL rotates counter-clockwise; long edge capped at 1568 like the pipeline does.
        img = img.rotate(cfg["rotate"], expand=True)
        scale = min(1.0, 1568 / max(img.size))
        return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    scale = cfg["edge"] / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BILINEAR)
    angle = cfg["skew"] * rng.choice([-1, 1])
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(205, 200, 190))
    img = img.filter(ImageFilter.GaussianBlur(cfg["blur"]))
    if cfg["noise"]:
        noise = Image.effect_noise(img.size, cfg["noise"]).convert("RGB")
        img = ImageChops.add(img, noise, scale=1.0, offset=-128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=cfg["quality"])
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def stack_vertically(pages: list[Image.Image]) -> Image.Image:
    # Multi-page docs become one tall image, like a stitched phone scan.
    width = max(p.width for p in pages)
    out = Image.new("RGB", (width, sum(p.height for p in pages)), "white")
    y = 0
    for p in pages:
        out.paste(p, (0, y))
        y += p.height
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=list(LEVELS), default="severe")
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "labelled")
    args = parser.parse_args()

    cfg = LEVELS[args.level]
    out_dir = ROOT / "data" / f"stress_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.get("seed", 42))
    for doc in sorted(p for p in args.src.iterdir() if p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}):
        gt = doc.with_suffix(".json")
        if not gt.exists():
            continue
        pages = [degrade(p, cfg, rng) for p in render_first_pages(doc)]
        image = pages[0] if len(pages) == 1 else stack_vertically(pages)
        if "rotate" in cfg and len(pages) > 1:
            # stack the *upright* pages first, then rotate the whole sheet once
            image = degrade(stack_vertically(render_first_pages(doc)), cfg, rng)
        image.save(out_dir / f"{doc.stem}.jpg", format="JPEG", quality=92)
        shutil.copyfile(gt, out_dir / gt.name)  # ground truth copied byte-for-byte
        print(f"{doc.name} -> {out_dir.name}/{doc.stem}.jpg {image.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
