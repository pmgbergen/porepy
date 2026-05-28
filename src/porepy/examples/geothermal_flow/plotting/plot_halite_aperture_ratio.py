"""Plot halite saturation and aperture ratio along a fracture from CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import os

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{bm}"


def build_file_map(times_days: list[float]) -> dict[str, str]:
    """Build CSV filename-to-legend-label mapping from requested times."""
    file_map: dict[str, str] = {}

    for time in times_days:
        time_label = f"{time:g}"
        filename = f"fracture_profile_t_{time_label}_days.csv"

        if time == 0:
            label = r"$t = 0$ days"
        elif time == 0.5:
            label = r"$t = 0.5$ days"
        elif time == 1 or time == 0.6:
            label = r"$t = 1$ day"
        else:
            label = rf"$t = {time_label}$ days"

        file_map[filename] = label

    return file_map


def load_fracture_data(
    folder_path: str | Path,
    file_map: Mapping[str, str],
    *,
    producer_coords: tuple[float, float] = (85.0, 15.0),
    x_col: str = "Points:0",
    y_col: str = "Points:1",
) -> dict[str, pd.DataFrame]:
    """Load fracture CSVs and add distance from the production point."""
    folder_path = Path(folder_path)
    px, py = producer_coords
    data: dict[str, pd.DataFrame] = {}

    for filename, label in file_map.items():
        full_path = folder_path / filename

        if not full_path.exists():
            print(f"Warning: {full_path} not found; skipping.")
            continue

        df = pd.read_csv(full_path)

        required = {x_col, y_col, "s_halite"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {full_path}: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

        df["dist"] = np.sqrt((df[x_col] - px) ** 2 + (df[y_col] - py) ** 2)
        df = df.sort_values("dist").reset_index(drop=True)

        data[label] = df

    if not data:
        raise RuntimeError(f"No fracture CSV data loaded from {folder_path}")

    return data


def style_axes(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(left=0)
    ax.tick_params(axis="both", which="major", width=2.0, length=6, labelsize=14)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def compute_aperture_ratio(
    s_halite: np.ndarray,
    *,
    clogging_exponent: float,
    minimum_aperture: float,
    reference_aperture: float,
) -> np.ndarray:
    """Compute aperture ratio from halite saturation.

    The aperture ratio is

        a / a0 = (1 - s_halite)**phi

    For the t = 0 curve, the caller passes
    s_halite = S_HALITE_INITIAL, so the initial curve is

        (1 - S_HALITE_INITIAL)**phi

    This is intentionally different from the reference line a/a0 = 1.
    """
    ratio_floor = minimum_aperture / reference_aperture

    raw_ratio = (1.0 - np.asarray(s_halite, dtype=float)) ** clogging_exponent

    return np.maximum(raw_ratio, ratio_floor)


def plot_halite_and_aperture(
    folder_path: str | Path,
    output_path: str | Path,
    *,
    times_days: list[float],
    producer_coords: tuple[float, float] = (85.0, 15.0),
    clogging_exponent: float = 0.1,
    minimum_aperture: float = 1.0e-4,
    reference_aperture: float = 1.0e-3,
    aperture_ylim: tuple[float, float] = (0.9935, 1.0001),
    halite_ylim: tuple[float, float] | None = None,
    s_halite_initial: np.ndarray | float = 0.01055,
    x_col: str = "Points:0",
    y_col: str = "Points:1",
    show: bool = False,
) -> None:
    """Plot halite saturation and aperture ratio along the production fracture."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_map = build_file_map(times_days)
    data = load_fracture_data(
        folder_path,
        file_map,
        producer_coords=producer_coords,
        x_col=x_col,
        y_col=y_col,
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"wspace": 0.3},
    )
    ax_hal, ax_ap = axes

    for color, (label, df) in zip(colors, data.items()):
        is_initial_time = label == r"$t = 0$ days"
        linestyle = "--" if is_initial_time else "-"

        s_halite_raw = df["s_halite"].to_numpy(dtype=float)
        dist = df["dist"].to_numpy(dtype=float)

        if is_initial_time:
            s_halite_plot = np.full_like(
                s_halite_raw,
                s_halite_initial,
                dtype=float,
            )
            s_halite_for_aperture = np.full_like(
                s_halite_raw,
                s_halite_initial,
                dtype=float,
            )
        else:
            s_halite_plot = s_halite_raw
            s_halite_for_aperture = s_halite_raw

        ax_hal.plot(
            dist,
            s_halite_plot,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.9,
            linestyle=linestyle,
        )

        a_ratio = compute_aperture_ratio(
            s_halite_for_aperture,
            clogging_exponent=clogging_exponent,
            minimum_aperture=minimum_aperture,
            reference_aperture=reference_aperture,
        )

        ax_ap.plot(
            dist,
            a_ratio,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.9,
            linestyle=linestyle,
        )
    ax_ap.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        label=r"$a/a^0 = 1$",
    )

    ax_hal.set_xlabel(r"\textbf{Distance from producer [m]}", fontsize=19)
    ax_hal.set_ylabel(r"$\bm{s^{\mathrm{hal}}}$ [$-$]", fontsize=19)
    ax_hal.legend(loc="best", fontsize=15, frameon=True)
    style_axes(ax_hal)

    if halite_ylim is not None:
        ax_hal.set_ylim(*halite_ylim)

    ax_ap.set_xlabel(r"\textbf{Distance from producer [m]}", fontsize=19)
    ax_ap.set_ylabel(r"$\bm{a/a^0}$ [$-$]", fontsize=19)
    ax_ap.legend(loc="best", fontsize=15, frameon=True)
    style_axes(ax_ap)
    ax_ap.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_ap.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax_ap.set_ylim(*aperture_ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=700, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved halite/aperture figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot halite saturation and aperture ratio from fracture CSV files."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--times-days", nargs="+", type=float, required=True)
    parser.add_argument("--producer-x", type=float, default=85.0)
    parser.add_argument("--producer-y", type=float, default=15.0)
    parser.add_argument("--clogging-exponent", type=float, default=0.1)
    parser.add_argument("--minimum-aperture", type=float, default=1.0e-4)
    parser.add_argument("--reference-aperture", type=float, default=1.0e-3)
    parser.add_argument("--aperture-ymin", type=float, default=0.9935)
    parser.add_argument("--aperture-ymax", type=float, default=1.0001)
    parser.add_argument("--halite-ymin", type=float, default=None)
    parser.add_argument("--halite-ymax", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--x-col", default="Points:0")
    parser.add_argument("--y-col", default="Points:1")
    return parser.parse_args()


def main() -> None:
    """Run the command-line plotting entry point."""
    args = parse_args()

    halite_ylim = None
    if args.halite_ymin is not None and args.halite_ymax is not None:
        halite_ylim = (args.halite_ymin, args.halite_ymax)

    plot_halite_and_aperture(
        folder_path=args.input_dir,
        output_path=args.out,
        times_days=args.times_days,
        producer_coords=(args.producer_x, args.producer_y),
        clogging_exponent=args.clogging_exponent,
        minimum_aperture=args.minimum_aperture,
        reference_aperture=args.reference_aperture,
        aperture_ylim=(args.aperture_ymin, args.aperture_ymax),
        halite_ylim=halite_ylim,
        x_col=args.x_col,
        y_col=args.y_col,
        show=args.show,
    )


if __name__ == "__main__":
    main()
