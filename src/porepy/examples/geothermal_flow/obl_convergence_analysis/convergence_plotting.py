"""Convergence plots shared by the experiment classes: the relative error of each field group against table refinement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .convergence_results import ConvergenceResult

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class ConvergencePlotMixin:
    """Mixin plotting relative errors against table refinement; override :attr:`FIELD_GROUPS` to regroup the panels."""

    FIELD_GROUPS: Mapping[str, Sequence[str]] = {
        "Saturation": ("S_l", "S_v", "S_h"),
        "Density": ("Rho_l", "Rho_v", "Rho_h"),
        "Specific enthalpy": ("H_l", "H_v", "H_h"),
        "Viscosity": ("mu_l", "mu_v"),
        "Salinity": ("Xl", "Xv"),
        "Temperature": ("Temperature",),
    }

    FIELD_LABELS: Mapping[str, str] = {
        "S_l": r"$s^{\mathrm{liq}}$",
        "S_v": r"$s^{\mathrm{vap}}$",
        "S_h": r"$s^{\mathrm{hal}}$",
        "Rho_l": r"$\rho^{\mathrm{liq}}$",
        "Rho_v": r"$\rho^{\mathrm{vap}}$",
        "Rho_h": r"$\rho^{\mathrm{hal}}$",
        "Rho": r"$\rho$",
        "H_l": r"$h^{\mathrm{liq}}$",
        "H_v": r"$h^{\mathrm{vap}}$",
        "H_h": r"$h^{\mathrm{hal}}$",
        "H": r"$h$",
        "mu_l": r"$\mu^{\mathrm{liq}}$",
        "mu_v": r"$\mu^{\mathrm{vap}}$",
        "Xv": r"$\chi_{\mathrm{vap}}$",
        "Xl": r"$\chi_{\mathrm{liq}}$",
        "Temperature": r"$T$",
        "T": r"$T$",
    }

    SAMPLE_SET_TITLES: Mapping[str, str] = {
        "bulk": "Interior samples",
        "boundary": "Boundary samples",
        "benchmark": "Benchmark states",
    }

    REFERENCE_ORDERS: Sequence[float] = (1.0, 2.0)

    SERIES_COLORS: Sequence[str] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    SERIES_MARKERS: Sequence[str] = ("o", "s", "^", "D")

    _SURFACE = "#ffffff"
    _INK = "#000000"
    _INK_SECONDARY = "#000000"
    _INK_MUTED = "#15201c"
    _GRID = "#FFFFFF"
    _AXIS = "#000000"

    def plot_errors(
        self,
        results: ConvergenceResult,
        category: str = "clean",
        output_dir: str | Path | None = None,
        tag: str = "",
    ) -> dict[str, Figure]:
        """Plot the relative error of every field group against refinement.

        Parameters:
            results: Level results, coarse to fine.
            category: Point category whose errors are plotted.
            output_dir: If given, each figure is saved there as PNG and PDF.
            tag: Appended to the file names.

        Returns:
            Sample set name to its figure; sets without points in the category are skipped.

        """
        figures: dict[str, Figure] = {}
        for sample_set in self.sample_sets:
            figure = self._error_figure(results, sample_set.name, category)
            if figure is None:
                continue
            figures[sample_set.name] = figure
            if output_dir is not None:
                self._save(figure, Path(output_dir), sample_set.name, category, tag)
        return figures

    # ------------------------------------------------------------------ figure
    def _error_figure(
        self, results: ConvergenceResult, set_name: str, category: str
    ) -> Figure | None:
        groups = {
            title: [f for f in names if f in self.fields]
            for title, names in self.FIELD_GROUPS.items()
        }
        groups = {title: names for title, names in groups.items() if names}
        n_points = self._n_points(results, set_name, category)
        if not groups or not results or n_points == 0:
            return None

        n_cols = min(3, len(groups))
        n_rows = int(np.ceil(len(groups) / n_cols))
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.4 * n_rows),
            sharex=True,
            squeeze=False,
            facecolor=self._SURFACE,
        )

        refinement = np.array([2.0**r.level for r in results])
        for ax, (title, names) in zip(axes.flat, groups.items()):
            self._error_axes(ax, results, set_name, category, title, names, refinement)
        for ax in axes.flat[len(groups) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel(
                    "Refinement level", color=self._INK_SECONDARY, fontsize=18
                )
        for row in axes:
            row[0].set_ylabel(r"$l_2$ error ", color=self._INK_SECONDARY, fontsize=18)

        figure.suptitle(
            self._figure_title(results, set_name, category),
            color=self._INK,
            fontsize=12,
            x=0.01,
            ha="left",
        )
        figure.tight_layout()
        return figure

    def _error_axes(
        self,
        ax: Axes,
        results: ConvergenceResult,
        set_name: str,
        category: str,
        title: str,
        names: Sequence[str],
        refinement: np.ndarray,
    ) -> None:
        self._style_axes(ax)
        ax.tick_params(labelsize=16)
        coarsest_errors = []
        for slot, name in enumerate(names):
            errors = np.array(
                [r.errors[set_name][category][name][0] for r in results], dtype=float
            )
            errors[~(errors > 0.0)] = np.nan
            ax.plot(
                refinement,
                errors,
                color=self.SERIES_COLORS[slot % len(self.SERIES_COLORS)],
                marker=self.SERIES_MARKERS[slot % len(self.SERIES_MARKERS)],
                linewidth=1.6,
                markersize=6.5,
                markeredgecolor=self._SURFACE,
                markeredgewidth=1.2,
                label=self.FIELD_LABELS.get(name, name),
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            finite = errors[np.isfinite(errors)]
            if finite.size:
                coarsest_errors.append(finite[0])

        if coarsest_errors:
            self._reference_orders(ax, refinement, max(coarsest_errors))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(refinement)
        ax.set_xticklabels([f"L{r.level}" for r in results])
        ax.minorticks_off()

        # Title on top, legend in one row beneath it, both above the data.
        ax.set_title(title, color=self._INK, fontsize=18, loc="left", pad=30)
        if len(names) > 1:
            legend = ax.legend(
                frameon=False,
                fontsize=15,
                ncol=len(names),
                loc="lower left",
                bbox_to_anchor=(-0.02, 1.0),
                borderaxespad=0.1,
                handlelength=1.6,
                columnspacing=1.0,
            )
            for text in legend.get_texts():
                text.set_color(self._INK_SECONDARY)

    def _reference_orders(
        self, ax: Axes, refinement: np.ndarray, anchor: float
    ) -> None:
        """Draw lines of the reference slopes, starting above the coarsest error.

        Parameters:
            ax: Axis to draw on.
            refinement: Refinement factors ``2**level``.
            anchor: Largest error on the coarsest level.

        """
        start = 1.6 * anchor
        for order in self.REFERENCE_ORDERS:
            line = start * (refinement / refinement[0]) ** (-order)
            ax.plot(refinement, line, color=self._AXIS, linewidth=1.0, zorder=1)
            ax.annotate(
                f"order {order:g}",
                xy=(refinement[-1], line[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=self._INK_MUTED,
                fontsize=15,
                va="center",
            )

    def _figure_title(
        self, results: ConvergenceResult, set_name: str, category: str
    ) -> str:
        return " "
        title = self.SAMPLE_SET_TITLES.get(set_name, set_name)
        n_points = self._n_points(results, set_name, category)
        description = {
            "clean": "no NaN-repaired cell corners",
            "near repair": "touching NaN-repaired nodes",
        }.get(category, category)
        suffix = description if n_points is None else f"{description}, n = {n_points}"
        return f"{title} ({suffix})\n{getattr(results, 'normalization', 'L2 error')}"

    @staticmethod
    def _n_points(
        results: ConvergenceResult, set_name: str, category: str
    ) -> int | None:
        """Number of points in a category.

        Parameters:
            results: Level results.
            set_name: Sample set name.
            category: Point category.

        Returns:
            The number of points, or None if the results do not record it.

        """
        if not results:
            return None
        return getattr(results[-1], "n_points", {}).get(set_name, {}).get(category)

    def _style_axes(self, ax: Axes) -> None:
        ax.set_facecolor(self._SURFACE)
        ax.grid(True, which="major", color=self._GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self._AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(
            colors=self._INK_MUTED, labelcolor=self._INK_SECONDARY, labelsize=10
        )

    @staticmethod
    def _save(
        figure: Figure, directory: Path, set_name: str, category: str, tag: str
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stem = "_".join(
            part
            for part in ("errors", set_name, category.replace(" ", "_"), tag)
            if part
        )
        for suffix in (".png", ".pdf"):
            figure.savefig(
                directory / f"{stem}{suffix}", dpi=600, facecolor=figure.get_facecolor()
            )
