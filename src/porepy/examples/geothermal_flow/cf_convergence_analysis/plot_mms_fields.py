"""Plot exact against computed fields of the manufactured-solution convergence test on the finest level.

Usage::

    python -m porepy.examples.geothermal_flow.cf_convergence_analysis.plot_mms_fields <results.json>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

FIELD_CMAP = LinearSegmentedColormap.from_list(
    "navy_teal_khaki_red",
    [
        "#25397d",
        "#35639e",
        "#468db2",
        "#79acb0",
        "#b5b88e",
        "#bda96c",
        "#b97b3d",
        "#a0471f",
        "#750f16",
    ],
)
FIELDS = (("pressure", "p"), ("enthalpy", "h"), ("z_NaCl", "z"))
FRACTURE_COLOUR = "#009900"


def plot_exact_vs_computed(
    results: list,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (4.6, 3.05),
):
    """Plot exact and computed matrix fields and fracture profiles on the finest level.

    Parameters:
        results: Per-level results carrying fields and geometry.
        save_path: File to write; nothing is saved if None.
        figsize: Figure size in inches.

    Returns:
        The figure.

    """
    finest = min(results, key=lambda r: r.cell_diameter)
    values, geometry = finest.values, finest.geometry
    polygons = geometry["matrix"]["polygons"]
    h = max(
        np.max(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)) for p in polygons
    )
    label = rf"$\Delta \zeta = {h:.3f}$"

    with plt.rc_context(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": "#52514e",
        }
    ):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2, 3, height_ratios=[2.0, 0.95], hspace=0.32, wspace=0.42
        )
        for j, (name, symbol) in enumerate(FIELDS):
            exact = values[f"matrix_{name}"]["exact"]
            computed = values[f"matrix_{name}"]["approximate"]
            low, high = (
                min(exact.min(), computed.min()),
                max(exact.max(), computed.max()),
            )
            top = grid[0, j].subgridspec(
                2, 2, width_ratios=[1, 0.07], wspace=0.08, hspace=0.12
            )
            for i, field in enumerate((exact, computed)):
                ax = fig.add_subplot(top[i, 0])
                cells = PolyCollection(
                    polygons,
                    array=field,
                    cmap=FIELD_CMAP,
                    edgecolors="face",
                    linewidths=0.1,
                )
                cells.set_clim(low, high)
                ax.add_collection(cells)
                ax.plot(
                    [0.5, 0.5],
                    [0.25, 0.75],
                    color=FRACTURE_COLOUR,
                    lw=1.0,
                    path_effects=[
                        patheffects.Stroke(linewidth=2.2, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"${symbol}_h$ (matrix)", pad=3)
                if j == 0:
                    ax.set_ylabel(
                        "exact" if i == 0 else f"computed\n({label})", fontsize=7
                    )
            colorbar = fig.colorbar(cells, cax=fig.add_subplot(top[:, 1]))
            colorbar.ax.tick_params(labelsize=5.5, length=2)
            colorbar.locator = matplotlib.ticker.MaxNLocator(4)
            colorbar.update_ticks()

        y = geometry["fracture"]["centers"][:, 1]
        order = np.argsort(y)
        for j, (name, symbol) in enumerate(FIELDS):
            ax = fig.add_subplot(grid[1, j])
            ax.plot(
                y[order],
                values[f"fracture_{name}"]["exact"][order],
                color="k",
                lw=1.0,
                label="exact",
            )
            ax.plot(
                y[order],
                values[f"fracture_{name}"]["approximate"][order],
                ls="none",
                marker="o",
                ms=3.0,
                color="C1",
                mec="none",
                label=f"computed ({label})",
            )
            ax.set_title(f"${symbol}_l$ (fracture)", pad=2)
            ax.set_xlabel(r"$\zeta_2$", labelpad=0)
            ax.set_xticks([0.25, 0.5, 0.75])
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            fontsize=7,
        )
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return fig


if __name__ == "__main__":
    from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_results_io import (
        load_results_json,
    )
    from porepy.examples.geothermal_flow.cf_convergence_analysis.plot_convergence_util import (
        PlotConvergenceUtil,
    )

    json_path = Path(sys.argv[1])
    results = load_results_json(json_path)
    stem = json_path.stem.removeprefix("mms_results_")
    plot_exact_vs_computed(results, json_path.parent / f"exact_vs_computed_{stem}.png")
    PlotConvergenceUtil(results).plot(
        save_path=json_path.parent / f"convergence_{stem}.png"
    )
    if stem.startswith("cartesian"):
        PlotConvergenceUtil(results).plot(
            save_path=json_path.parent / f"convergence_{stem}.png",
            xlabel=None,
            title=None,
            primary_title=None,
            interface_title=None,
            show_legend=False,
            **PlotConvergenceUtil.SLIDE_STYLE,
        )
    else:
        PlotConvergenceUtil(results).plot(
            save_path=json_path.parent / f"convergence_{stem}.png",
            **PlotConvergenceUtil.SLIDE_STYLE,
        )
