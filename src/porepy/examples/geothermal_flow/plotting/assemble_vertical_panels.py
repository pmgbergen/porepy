from __future__ import annotations

"""Assemble a figure from two pre-rendered images stacked vertically.

This script takes two PNG images, typically the two rows of the
manufactured-solution convergence figure:
- the convergence plot on Cartesian meshes (top row)
- the convergence plot on the simplex mesh ensemble (bottom row)

and stacks them into one final figure.

Typical usage
-------------
python geothermal_flow/plotting/assemble_vertical_panels.py \
    --top geothermal_flow/cf_convergence_analysis/mms_halite_output/convergence_cartesian_slide.png \
    --bottom geothermal_flow/cf_convergence_analysis/mms_halite_output/convergence_ensemble_simplex_slide.png \
    --out figures/mms/figure6.png
"""

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble a figure from two images stacked vertically."
    )
    parser.add_argument(
        "--top",
        required=True,
        help="Path to the top image (for example the Cartesian convergence plot).",
    )
    parser.add_argument(
        "--bottom",
        required=True,
        help="Path to the bottom image (for example the simplex convergence plot).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output assembled PNG.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=40,
        help="Vertical gap in pixels between the two rows.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=20,
        help="Outer padding in pixels around the full figure.",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background color for the output canvas.",
    )
    return parser.parse_args()


def resize_to_common_width(
    top_img: Image.Image, bottom_img: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Resize the two images so they share the same width.

    The narrower image is scaled to match the wider image while preserving
    aspect ratio. If both widths already match, the originals are returned.
    """
    top_w, top_h = top_img.size
    bottom_w, bottom_h = bottom_img.size

    target_width = max(top_w, bottom_w)

    if top_w != target_width:
        new_height = int(round(top_h * target_width / top_w))
        top_img = top_img.resize((target_width, new_height), Image.LANCZOS)

    if bottom_w != target_width:
        new_height = int(round(bottom_h * target_width / bottom_w))
        bottom_img = bottom_img.resize((target_width, new_height), Image.LANCZOS)

    return top_img, bottom_img


def assemble_top_bottom(
    top_path: str | Path,
    bottom_path: str | Path,
    output_path: str | Path,
    *,
    gap: int = 40,
    pad: int = 20,
    background: str = "white",
) -> None:
    """Stack two images vertically and save the result."""
    top_path = Path(top_path).resolve()
    bottom_path = Path(bottom_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not top_path.exists():
        raise FileNotFoundError(f"Top image not found: {top_path}")
    if not bottom_path.exists():
        raise FileNotFoundError(f"Bottom image not found: {bottom_path}")

    top_img = Image.open(top_path).convert("RGB")
    bottom_img = Image.open(bottom_path).convert("RGB")

    top_img, bottom_img = resize_to_common_width(top_img, bottom_img)

    top_w, top_h = top_img.size
    bottom_w, bottom_h = bottom_img.size

    canvas_width = max(top_w, bottom_w) + 2 * pad
    canvas_height = top_h + bottom_h + gap + 2 * pad

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=background)

    canvas.paste(top_img, (pad, pad))
    canvas.paste(bottom_img, (pad, pad + top_h + gap))

    canvas.save(output_path, dpi=(600, 600))
    print(f"Saved assembled: {output_path}")


def main() -> None:
    """Run the figure assembly from the command line."""
    args = parse_args()

    assemble_top_bottom(
        top_path=args.top,
        bottom_path=args.bottom,
        output_path=args.out,
        gap=args.gap,
        pad=args.pad,
        background=args.background,
    )


if __name__ == "__main__":
    main()
