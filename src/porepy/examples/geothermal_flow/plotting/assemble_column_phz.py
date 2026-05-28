from __future__ import annotations

"""Assemble Figure from two pre-rendered ParaView column images.

This script takes two PNG images, typically:
- a 3-row column rendered at t = 10 days
- a 3-row column rendered at t = 74 days

and combines them side by side into one final figure.

Typical usage
-------------
python geothermal_flow/plotting/assemble_column_phz.py \
    --left figures/example1/figure8_panels/phz_column_t_10_days.png \
    --right figures/example1/figure8_panels/phz_column_t_74_days.png \
    --out figures/example1/figure8.png
"""

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble Figure from two ParaView column images."
    )
    parser.add_argument(
        "--left",
        required=True,
        help="Path to the left column image (for example t = 10 days).",
    )
    parser.add_argument(
        "--right",
        required=True,
        help="Path to the right column image (for example t = 74 days).",
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
        help="Horizontal gap in pixels between the two columns.",
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


def resize_to_common_height(
    left_img: Image.Image, right_img: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Resize the two images so they share the same height.

    The smaller image is scaled to match the taller image while preserving
    aspect ratio. If both heights already match, the originals are returned.
    """
    left_w, left_h = left_img.size
    right_w, right_h = right_img.size

    target_height = max(left_h, right_h)

    if left_h != target_height:
        new_width = int(round(left_w * target_height / left_h))
        left_img = left_img.resize((new_width, target_height), Image.LANCZOS)

    if right_h != target_height:
        new_width = int(round(right_w * target_height / right_h))
        right_img = right_img.resize((new_width, target_height), Image.LANCZOS)

    return left_img, right_img


def assemble_side_by_side(
    left_path: str | Path,
    right_path: str | Path,
    output_path: str | Path,
    *,
    gap: int = 40,
    pad: int = 20,
    background: str = "white",
) -> None:
    """Assemble two images side by side and save the result."""
    left_path = Path(left_path).resolve()
    right_path = Path(right_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not left_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_path}")

    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

    left_img, right_img = resize_to_common_height(left_img, right_img)

    left_w, left_h = left_img.size
    right_w, right_h = right_img.size

    canvas_width = left_w + right_w + gap + 2 * pad
    canvas_height = max(left_h, right_h) + 2 * pad

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=background)

    left_x = pad
    left_y = pad + (canvas_height - 2 * pad - left_h) // 2

    right_x = pad + left_w + gap
    right_y = pad + (canvas_height - 2 * pad - right_h) // 2

    canvas.paste(left_img, (left_x, left_y))
    canvas.paste(right_img, (right_x, right_y))

    canvas.save(output_path, dpi=(700, 700))
    print(f"Saved assembled: {output_path}")


def main() -> None:
    """Run the Figure assembly from the command line."""
    args = parse_args()

    assemble_side_by_side(
        left_path=args.left,
        right_path=args.right,
        output_path=args.out,
        gap=args.gap,
        pad=args.pad,
        background=args.background,
    )


if __name__ == "__main__":
    main()
