from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble near-well halite comparison panels."
    )
    parser.add_argument("--left", required=True, help="Left panel image.")
    parser.add_argument("--right", required=True, help="Right panel image.")
    parser.add_argument("--out", required=True, help="Output figure path.")

    parser.add_argument(
        "--left-label",
        default=r"$q_{\text{inj}} = 0.28~\text{kg}\,\text{m}^{-3}\text{s}^{-1}$",
    )
    parser.add_argument(
        "--right-label",
        default=r"$q_{\text{inj}} = 0.364~\text{kg}\,\text{m}^{-3}\text{s}^{-1}$",
    )
    parser.add_argument("--colorbar-label", default=r"$s^{\mathrm{hal}}$")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not left_path.exists():
        raise FileNotFoundError(f"Left panel not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right panel not found: {right_path}")

    left_img = mpimg.imread(left_path)
    right_img = mpimg.imread(right_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    panels = [
        (axes[0], left_img, args.left_label),
        (axes[1], right_img, args.right_label),
    ]

    for ax, img, label in panels:
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.06,
            0.94,
            label,
            transform=ax.transAxes,
            color="white",
            fontsize=18,
            va="top",
            ha="left",
        )

    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)
    cmap = mpl.cm.coolwarm
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.88, 0.27, 0.025, 0.50])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(args.colorbar_label, fontsize=18, fontweight="bold")
    cbar.set_ticks([args.vmin, args.vmax])
    cbar.ax.tick_params(labelsize=14)

    # Remove black box around colorbar
    cbar.outline.set_visible(False)

    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    fig.subplots_adjust(
        left=0.02,
        right=0.84,
        top=0.96,
        bottom=0.08,
        wspace=0.08,
    )

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved near-well comparison figure: {out_path}")


if __name__ == "__main__":
    main()
