"""Assemble ParaView-rendered s_halite panels into a 2x2 figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    """Read panel PNGs and assemble a 2x2 figure."""
    args = parse_args()

    panel_dir = Path(args.panel_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("s_halite_t_1_days.png", r"$t = 1$ day"),
        ("s_halite_t_10_days.png", r"$t = 10$ days"),
        ("s_halite_t_30_days.png", r"$t = 30$ days"),
        ("s_halite_t_74_days.png", r"$t = 74$ days"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8))

    for ax, (filename, title) in zip(axes.ravel(), panels):
        image_path = panel_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Panel not found: {image_path}")

        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.07,
            0.90,
            title,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            ha="left",
            va="top",
        )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.98,
        wspace=0.03,
        hspace=0.12,
    )

    fig.text(
        0.5,
        0.02,
        "Figure 12: Progression of halite saturation at 1, 10, 30, and 74 days.",
        ha="center",
        va="bottom",
        fontsize=16,
    )

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Saved assembled figure: {out_path}")


if __name__ == "__main__":
    main()