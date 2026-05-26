"""Plot Figure 9 centerline profiles from a ParaView-extracted CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import os
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{bm}"


def _safe_savgol(values: np.ndarray, window_length: int = 17, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing while handling short arrays safely."""
    n = len(values)
    if n < 5:
        return values

    # Window length must be odd and <= n.
    window = min(window_length, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        return values

    return savgol_filter(values, window_length=window, polyorder=polyorder)


def plot_centerline_profile_p_T_h_s_vap_s_hal_z(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    arc_length_col: str = "arc_length",
    smooth: bool = False,
    smooth_range: tuple[float, float] = (0.0, 10.0),
    variables_to_smooth: tuple[str, ...] = ("enthalpy", "s_gas"),
    starting_values_smoothing: dict[str, float] | None = {"enthalpy": 1.197e6, "s_gas": 0.388},
    legend_loc: str = "upper right",
    show: bool = False,
) -> None:
    """Plot centerline profiles for pressure, temperature, enthalpy, saturations, and salinity.

    Parameters
    ----------
    csv_path
        CSV file exported from the ParaView centerline/PlotOverLine state.
    output_path
        Path where the figure is saved.
    arc_length_col
        Name of the distance coordinate column in the CSV file.
    smooth
        If True, smooth selected variables over ``smooth_range``.
    smooth_range
        Distance interval over which smoothing is applied.
    variables_to_smooth
        Names of CSV columns to smooth.
    starting_values_smoothing
        Optional dictionary of values imposed at distance zero after smoothing.
    show
        If True, display the figure interactively after saving.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values(arc_length_col).reset_index(drop=True)

    required_columns = {
        arc_length_col,
        "pressure",
        "temperature",
        "enthalpy",
        "s_gas",
        "s_halite",
        "z_NaCl",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}. "
            f"Available columns are: {list(df.columns)}"
        )

    x = df[arc_length_col].to_numpy(dtype=float)

    if smooth:
        mask = (x >= smooth_range[0]) & (x <= smooth_range[1])

        for var in variables_to_smooth:
            if var in df.columns:
                df.loc[mask, var] = _safe_savgol(df.loc[mask, var].to_numpy(dtype=float))

        if starting_values_smoothing:
            first_idx = df.index[np.argmin(np.abs(x - 0.0))]
            for var, value in starting_values_smoothing.items():
                if var in df.columns:
                    df.loc[first_idx, var] = value

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"wspace": 0.4},
    )

    # ------------------------------------------------------------------
    # Left panel: pressure and temperature
    # ------------------------------------------------------------------
    color_p = "tab:blue"
    color_T = "tab:red"

    ax_left.plot(
        x,
        df["pressure"].to_numpy(dtype=float) / 1.0e6,
        color=color_p,
        linewidth=2.5,
        label=r"$p$",
    )
    ax_left.set_xlabel(r"\textbf{Distance along centerline [m]}", fontsize=19)
    ax_left.set_ylabel(r"$\bm{p}$ \textbf{[MPa]}", fontsize=19, color=color_p)
    ax_left.tick_params(axis="y", labelcolor=color_p)

    ax_left_T = ax_left.twinx()
    ax_left_T.plot(
        x,
        df["temperature"].to_numpy(dtype=float) - 273.15,
        color=color_T,
        linewidth=2.5,
        label=r"$T$",
    )
    ax_left_T.set_ylabel(r"$\bm{T}$ \textbf{[$^\circ$C]}", fontsize=19, color=color_T)
    ax_left_T.tick_params(axis="y", labelcolor=color_T)

    # ------------------------------------------------------------------
    # Right panel: enthalpy, vapor saturation, halite saturation, NaCl
    # ------------------------------------------------------------------
    color_h = "tab:blue"
    color_vap = "tab:green"
    color_hal = "tab:purple"
    color_z = "tab:orange"

    ax_right.plot(
        x,
        df["enthalpy"].to_numpy(dtype=float) / 1.0e3,
        color=color_h,
        linewidth=2.5,
        label=r"$h$",
    )
    ax_right.set_xlabel(r"\textbf{Distance along centerline [m]}", fontsize=19)
    ax_right.set_ylabel(r"$\bm{h}$ \textbf{[kJ/kg]}", fontsize=19, color=color_h)
    ax_right.tick_params(axis="y", labelcolor=color_h)

    ax_right_sat = ax_right.twinx()
    ax_right_sat.plot(
        x,
        df["s_gas"].to_numpy(dtype=float),
        color=color_vap,
        linewidth=2.5,
        linestyle="--",
        label=r"$s^{\mathrm{vap}}$",
    )
    ax_right_sat.plot(
        x,
        df["s_halite"].to_numpy(dtype=float),
        color=color_hal,
        linewidth=2.5,
        linestyle=":",
        label=r"$s^{\mathrm{hal}}$",
    )
    ax_right_sat.plot(
        x,
        df["z_NaCl"].to_numpy(dtype=float),
        color=color_z,
        linewidth=2.5,
        linestyle="-.",
        label=r"$z_{\mathrm{NaCl}}$",
    )
    ax_right_sat.set_ylabel(
        r"\textbf{Mass fraction / Saturation [$-$]}",
        fontsize=19,
        color="black",
    )
    ax_right_sat.tick_params(axis="y", labelcolor="black")

    # Combined legend for right panel.
    handles_1, labels_1 = ax_right.get_legend_handles_labels()
    handles_2, labels_2 = ax_right_sat.get_legend_handles_labels()
    ax_right.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        fontsize=13,
        loc=legend_loc,
        frameon=True,
    )

    # ------------------------------------------------------------------
    # Common styling
    # ------------------------------------------------------------------
    for ax in (ax_left, ax_left_T, ax_right, ax_right_sat):
        ax.tick_params(axis="both", which="major", width=2.0, length=6, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    for ax in (ax_left, ax_right):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        # ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved centerline profile figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Figure 9 centerline profiles from a ParaView CSV file."
    )
    parser.add_argument("--input-csv", required=True, help="Input centerline CSV file.")
    parser.add_argument("--out", required=True, help="Output figure path.")
    parser.add_argument("--arc-length-col", default="arc_length")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smooth-range", nargs=2, type=float, default=[0.0, 10.0])
    parser.add_argument("--smooth-vars", nargs="+", default=["enthalpy", "s_gas"])
    parser.add_argument("--smooth-start", nargs="*", default=[])
    parser.add_argument("--legend-loc", default="upper right")
    return parser.parse_args()

def parse_key_value_pairs(items: list[str]) -> dict[str, float]:
    values = {}
    for item in items:
        key, value = item.split("=", 1)
        values[key] = float(value)
    return values


def main() -> None:
    """Run the command-line plotting entry point."""
    args = parse_args()

    plot_centerline_profile_p_T_h_s_vap_s_hal_z(
        csv_path=args.input_csv,
        output_path=args.out,
        arc_length_col=args.arc_length_col,
        smooth=args.smooth,
        smooth_range=tuple(args.smooth_range),
        variables_to_smooth=tuple(args.smooth_vars),
        starting_values_smoothing=parse_key_value_pairs(args.smooth_start),
        legend_loc=args.legend_loc,
        show=args.show,
    )


if __name__ == "__main__":
    main()