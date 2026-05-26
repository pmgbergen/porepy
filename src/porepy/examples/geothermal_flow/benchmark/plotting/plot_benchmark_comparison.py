"""Plot benchmark CSMP--PorePy comparison.

This reproduces the benchmark figure with two panels:

Left:
    Temperature T [deg C] and pressure p [MPa]

Right:
    Liquid saturation s_liq and halite saturation s_hal

Solid lines are PorePy results.
Dashed lines are CSMP reference data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Plot style configuration
# -----------------------------------------------------------------------------
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{bm}"


def load_reference_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one CSMP reference curve."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]


def load_csmp_reference(reference_dir: str | Path) -> dict[str, np.ndarray]:
    """Load and align CSMP Fig. 6c reference data."""
    reference_dir = Path(reference_dir)

    x_p, p = load_reference_curve(reference_dir / "fig_6c_pressure.csv")
    x_t, t = load_reference_curve(reference_dir / "fig_6c_temperature.csv")
    x_l, s_liq = load_reference_curve(reference_dir / "fig_6c_liquid_saturation.csv")
    x_h, s_hal = load_reference_curve(reference_dir / "fig_6c_halite_saturation.csv")

    # Use the liquid-saturation x-grid as the common CSMP plotting grid.
    x = x_l

    return {
        "x_km": x,
        "pressure_MPa": np.interp(x, x_p, p),
        "temperature_C": np.interp(x, x_t, t),
        "s_liq": s_liq,
        "s_halite": np.interp(x, x_h, s_hal),
    }


def contiguous_region(
    x: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float] | None:
    """Return min and max x for a boolean mask."""
    if not np.any(mask):
        return None

    values = x[mask]
    return float(np.min(values)), float(np.max(values))


def infer_reference_regions(
    x: np.ndarray,
    s_liq: np.ndarray,
    s_hal: np.ndarray,
    *,
    residual_liquid_saturation: float = 0.3,
    threshold: float = 1.0e-3,
) -> dict[str, tuple[float, float] | None]:
    """Infer vapor-liquid and liquid-halite regions from CSMP saturation data."""
    s_vap = 1.0 - (s_liq + s_hal)

    vapor_liquid_mask = (s_vap > threshold) & (s_liq > residual_liquid_saturation)
    liquid_halite_mask = s_hal > threshold

    return {
        "vapor_liquid": contiguous_region(x, vapor_liquid_mask),
        "liquid_halite": contiguous_region(x, liquid_halite_mask),
    }


def parse_region(values: list[float] | None) -> tuple[float, float] | None:
    """Parse optional region limits."""
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Region must contain exactly two values.")
    return float(values[0]), float(values[1])


def shade_regions(
    ax: plt.Axes,
    vapor_liquid_region: tuple[float, float] | None,
    liquid_halite_region: tuple[float, float] | None,
) -> None:
    """Shade the two-phase regions."""
    if vapor_liquid_region is not None:
        ax.axvspan(
            vapor_liquid_region[0],
            vapor_liquid_region[1],
            color="tab:gray",
            alpha=0.5,
            zorder=0,
        )

    if liquid_halite_region is not None:
        ax.axvspan(
            liquid_halite_region[0],
            liquid_halite_region[1],
            color="tab:gray",
            alpha=0.5,
            zorder=0,
        )


def style_spines(*axes: plt.Axes) -> None:
    """Apply consistent spine and tick styling."""
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=14, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)


def plot_benchmark_comparison(
    porepy_csv: str | Path,
    reference_dir: str | Path,
    output_path: str | Path,
    *,
    vapor_liquid_region: tuple[float, float] | None = None,
    liquid_halite_region: tuple[float, float] | None = None,
    residual_liquid_saturation: float = 0.3,
    simulation_time_years: float = 2000.0,
    show_annotations: bool = True,
    show: bool = False,
) -> None:
    """Plot the full CSMP--PorePy benchmark comparison figure."""
    porepy_csv = Path(porepy_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not porepy_csv.exists():
        raise FileNotFoundError(f"PorePy benchmark CSV not found: {porepy_csv}")

    porepy = pd.read_csv(porepy_csv)
    csmp = load_csmp_reference(reference_dir)

    required = {
        "x_km",
        "pressure_MPa",
        "temperature_C",
        "s_liq",
        "s_halite",
    }
    missing = required.difference(porepy.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {porepy_csv}: {sorted(missing)}. "
            f"Available columns: {list(porepy.columns)}"
        )

    x_porepy = porepy["x_km"].to_numpy(dtype=float)
    pressure_porepy = porepy["pressure_MPa"].to_numpy(dtype=float)
    temperature_porepy = porepy["temperature_C"].to_numpy(dtype=float)
    s_liq_porepy = porepy["s_liq"].to_numpy(dtype=float)
    s_hal_porepy = porepy["s_halite"].to_numpy(dtype=float)

    x_csmp = csmp["x_km"]
    pressure_csmp = csmp["pressure_MPa"]
    temperature_csmp = csmp["temperature_C"]
    s_liq_csmp = csmp["s_liq"]
    s_hal_csmp = csmp["s_halite"]

    if vapor_liquid_region is None or liquid_halite_region is None:
        inferred = infer_reference_regions(
            x_csmp,
            s_liq_csmp,
            s_hal_csmp,
            residual_liquid_saturation=residual_liquid_saturation,
        )
        vapor_liquid_region = vapor_liquid_region or inferred["vapor_liquid"]
        liquid_halite_region = liquid_halite_region or inferred["liquid_halite"]

    linewidth_porepy = 3.5
    linewidth_csmp = 3.5
    fontsize = 23

    fig, (ax_tp, ax_sat) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.7),
        gridspec_kw={"wspace": 0.38},
    )

    # -------------------------------------------------------------------------
    # Left panel: Temperature and pressure
    # -------------------------------------------------------------------------
    shade_regions(ax_tp, vapor_liquid_region, liquid_halite_region)

    color_t = "tab:red"
    color_p = "tab:blue"

    ax_tp.set_xlabel(r"\textbf{Distance [km]}", fontsize=fontsize)
    ax_tp.set_ylabel(r"$\bm{T}$ [$^\circ$C]", color=color_t, fontsize=fontsize)
    ax_tp.plot(
        x_porepy,
        temperature_porepy,
        "-",
        color=color_t,
        linewidth=linewidth_porepy,
    )
    ax_tp.plot(
        x_csmp,
        temperature_csmp,
        "--",
        color=color_t,
        linewidth=linewidth_csmp,
    )
    ax_tp.tick_params(axis="y", labelcolor=color_t, direction="in")
    ax_tp.tick_params(axis="x", direction="in")

    ax_tp.set_xlim(0.0, 2.0)
    ax_tp.set_ylim(150.0, 300.0)
    ax_tp.set_xticks(np.linspace(0.0, 2.0, 3))
    ax_tp.set_yticks([150.0, 300.0])
    ax_tp.grid(False)

    ax_p = ax_tp.twinx()
    ax_p.set_ylabel(r"$\bm{p}$ [MPa]", color=color_p, fontsize=fontsize)
    ax_p.plot(
        x_porepy,
        pressure_porepy,
        "-",
        color=color_p,
        linewidth=linewidth_porepy,
    )
    ax_p.plot(
        x_csmp,
        pressure_csmp,
        "--",
        color=color_p,
        linewidth=linewidth_csmp,
    )
    ax_p.tick_params(axis="y", labelcolor=color_p, direction="in")
    ax_p.set_ylim(1.0, 4.0)
    ax_p.set_yticks([1.0, 2.0, 3.0, 4.0])

    ax_tp.text(
        0.68,
        0.82,
        rf"{simulation_time_years:g} years",
        transform=ax_tp.transAxes,
        fontsize=16,
        color="black",
    )

    # -------------------------------------------------------------------------
    # Right panel: Liquid and halite saturation
    # -------------------------------------------------------------------------
    shade_regions(ax_sat, vapor_liquid_region, liquid_halite_region)

    color_liq = "tab:green"
    color_hal = "black"

    ax_sat.set_xlabel(r"\textbf{Distance [km]}", fontsize=fontsize)
    ax_sat.set_ylabel(
        r"$\bm{s^{\mathrm{liq}}}$ [$-$]",
        color=color_liq,
        fontsize=fontsize,
    )
    ax_sat.plot(
        x_porepy,
        s_liq_porepy,
        "-",
        color=color_liq,
        linewidth=linewidth_porepy,
    )
    ax_sat.plot(
        x_csmp,
        s_liq_csmp,
        "--",
        color=color_liq,
        linewidth=linewidth_csmp,
    )

    ax_sat.tick_params(axis="y", labelcolor=color_liq, direction="in")
    ax_sat.tick_params(axis="x", direction="in")
    ax_sat.set_xlim(0.0, 2.0)
    ax_sat.set_ylim(-0.02, 1.01)
    ax_sat.set_xticks(np.linspace(0.0, 2.0, 3))
    ax_sat.set_yticks(np.linspace(0.0, 1.0, 3))
    ax_sat.grid(False)

    ax_hal = ax_sat.twinx()
    ax_hal.set_ylabel(
        r"$\bm{s^{\mathrm{hal}}}$ [$-$]",
        color=color_hal,
        fontsize=fontsize,
    )
    ax_hal.plot(
        x_porepy,
        s_hal_porepy,
        "-",
        color=color_hal,
        linewidth=linewidth_porepy,
    )
    ax_hal.plot(
        x_csmp,
        s_hal_csmp,
        "--",
        color=color_hal,
        linewidth=linewidth_csmp,
    )
    ax_hal.tick_params(axis="y", labelcolor=color_hal, direction="in")
    ax_hal.set_ylim(-0.02, 1.01)
    ax_hal.set_yticks(np.linspace(0.0, 1.0, 3))

    ax_sat.axhline(
        y=residual_liquid_saturation,
        color="black",
        linestyle="-",
        linewidth=1.8,
    )

    if show_annotations:
        ax_sat.text(
            0.08,
            0.65,
            "Vapor\n+\nHalite",
            transform=ax_sat.transAxes,
            fontsize=15,
            ha="center",
            va="center",
            color="black",
        )
        ax_sat.text(
            0.42,
            0.61,
            "Vapor + Liquid",
            transform=ax_sat.transAxes,
            fontsize=15,
            ha="center",
            va="center",
            color="black",
        )
        ax_sat.text(
            0.68,
            0.58,
            "Liquid",
            transform=ax_sat.transAxes,
            fontsize=14,
            rotation=90,
            ha="center",
            va="center",
            color="black",
        )
        ax_sat.text(
            0.86,
            0.65,
            "Liquid\n+\nHalite",
            transform=ax_sat.transAxes,
            fontsize=15,
            ha="center",
            va="center",
            color="black",
        )

    style_spines(ax_tp, ax_p, ax_sat, ax_hal)

    fig.savefig(output_path, dpi=700, bbox_inches="tight", transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved benchmark comparison figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot CSMP--PorePy benchmark comparison."
    )
    parser.add_argument("--porepy-csv", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vl-region", nargs=2, type=float, default=None)
    parser.add_argument("--lh-region", nargs=2, type=float, default=None)
    parser.add_argument("--residual-liquid-saturation", type=float, default=0.3)
    parser.add_argument("--simulation-time-years", type=float, default=2000.0)
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run benchmark plotting."""
    args = parse_args()

    plot_benchmark_comparison(
        porepy_csv=args.porepy_csv,
        reference_dir=args.reference_dir,
        output_path=args.out,
        vapor_liquid_region=parse_region(args.vl_region),
        liquid_halite_region=parse_region(args.lh_region),
        residual_liquid_saturation=args.residual_liquid_saturation,
        simulation_time_years=args.simulation_time_years,
        show_annotations=not args.no_annotations,
        show=args.show,
    )


if __name__ == "__main__":
    main()