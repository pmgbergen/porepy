from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble ParaView-rendered s_halite panels."
    )
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--colorbar-label",
        default=r"$\mathbf{s}^{\mathbf{hal}}$",
        help="Shared colorbar label.",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.21)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--dpi", type=int, default=600)

    return parser.parse_args()


def crop_background(img: np.ndarray, tol: int = 6) -> np.ndarray:
    """Crop uniform background from an image.

    The background color is estimated from the top-left pixel. Pixels within
    `tol` intensity of that color are treated as background.
    """
    if img.ndim == 2:
        bg = img[0, 0]
        mask = np.abs(img.astype(float) - float(bg)) > tol / 255.0
    else:
        rgb = img[..., :3].astype(float)
        bg = rgb[0, 0, :]
        diff = np.abs(rgb - bg[None, None, :])
        mask = np.any(diff > tol / 255.0, axis=-1)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return img[r0:r1, c0:c1]


def main() -> None:
    """Read panel PNGs, crop margins, and assemble a 2x2 figure."""
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

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 4.0))

    for ax, (filename, title) in zip(axes.ravel(), panels):
        image_path = panel_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Panel not found: {image_path}")

        img = mpimg.imread(image_path)
        img = crop_background(img, tol=6)

        ax.imshow(img, aspect="equal")
        ax.axis("off")
        ax.text(
            0.03,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=18,
            color="white",
            ha="left",
            va="top",
            # fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.02,
        right=0.86,
        bottom=0.03,
        top=0.98,
        wspace=0.04,
        hspace=-0.15,
    )
    # Shared colorbar goes here
    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)
    cmap = mpl.colormaps[args.cmap]
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.89, 0.24, 0.02, 0.54])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(args.colorbar_label, fontsize=18)
    cbar.set_ticks([args.vmin, args.vmax])
    cbar.ax.tick_params(labelsize=15, length=0)

    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved assembled figure: {out_path}")


if __name__ == "__main__":
    main()
