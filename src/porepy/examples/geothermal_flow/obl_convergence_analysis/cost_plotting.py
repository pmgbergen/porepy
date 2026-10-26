"""The figure of Experiment C: interpolation error against generation time and benchmark run time."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter


from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_plotting import (
    ConvergencePlotMixin,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.cost_experiment import (
    CostResult,
)

ONLINE_LABEL = "benchmark run"
"""str: Legend entry for the simulation wall-clock."""

OFFLINE_LABEL = "table generation"
"""str: Legend entry for the one-off generation cost."""

DIRECT_LABEL = "direct evaluation (projected)"
"""str: Legend entry for the projected cost without a table."""

FLOOR_DECADES = 1.0
"""float: Decades below the smallest error at which the zero-error direct evaluation is drawn."""


class CostFigure(ConvergencePlotMixin):
    """Cost against error, in the style of :class:`ConvergencePlotMixin`."""

    def plot_offline_online(
        self,
        result: CostResult,
        output_dir: str | Path | None = None,
        tag: str = "offline_online",
        floor_decades: float = 0.5,
    ) -> Figure | None:
        """Draw the cost-against-error figure as two panels sharing the error axis.

        Parameters:
            result: The measurements; levels without both an error and a completed run are skipped.
            output_dir: If given, the figure is saved there as PNG and PDF.
            tag: Appended to the file name.
            floor_decades: Decades below the smallest error at which direct evaluation is drawn.

        Returns:
            The figure, or None if no level has both a cost and an error.

        """
        levels = [
            level
            for level in sorted(result.boundary_errors)
            if level in result.simulations
            and result.simulations[level].representative is not None
        ]
        if not levels:
            return None
        errors = np.array([result.boundary_errors[level] for level in levels])
        online = np.array(
            [result.simulations[level].total_seconds.median_seconds for level in levels]
        )
        spread = np.array(
            [result.simulations[level].total_seconds.spread_seconds for level in levels]
        )
        generation = np.array(
            [
                result.tables[level].generation_seconds
                if level in result.tables
                and result.tables[level].generation_seconds is not None
                else np.nan
                for level in levels
            ],
            dtype=float,
        )
        projected = result.projected_direct_seconds(levels[-1])
        floor = float(errors.min()) * 10.0**-floor_decades

        figure, (ax_off, ax_on) = plt.subplots(
            1,
            2,
            figsize=(7.2, 4.2),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.15]},
            facecolor=self._SURFACE,
        )
        for ax in (ax_off, ax_on):
            self._style_axes(ax)
            ax.tick_params(labelsize=8.5)
            ax.set_xscale("log")
            ax.set_yscale("log")

        # Offline: one-off cost of generating each table.
        has_cost = np.isfinite(generation)
        ax_off.plot(
            generation[has_cost],
            errors[has_cost],
            color=self.SERIES_COLORS[1],
            marker=self.SERIES_MARKERS[1],
            markersize=6.5,
            markeredgecolor=self._SURFACE,
            markeredgewidth=1.2,
            linewidth=1.7,
            linestyle="--",
        )
        for level, seconds, error in zip(
            np.array(levels)[has_cost], generation[has_cost], errors[has_cost]
        ):
            spacing = result.tables[level].spacing if level in result.tables else None
            # Above and to the right of each point, clear of the falling line.
            ax_off.annotate(
                self._spacing_label(level, spacing),
                xy=(seconds, error),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                color=self._INK_SECONDARY,
                fontsize=8,
            )
        # Room on the right for the label of the finest table.
        ax_off.set_xlim(np.nanmin(generation) / 1.4, np.nanmax(generation) * 8.0)
        ax_off.set_xlabel("Grid construction time [s]", color=self._INK_SECONDARY)
        ax_off.set_ylabel(r"$l_2$ error", color=self._INK_SECONDARY)
        ax_off.set_title("offline", color=self._INK_SECONDARY, fontsize=10)

        # Online: benchmark run time, and the projected run with direct evaluation.
        ax_on.errorbar(
            online,
            errors,
            xerr=spread / 2.0,
            color=self.SERIES_COLORS[0],
            marker=self.SERIES_MARKERS[0],
            markersize=7.0,
            markeredgecolor=self._SURFACE,
            markeredgewidth=1.2,
            linewidth=1.8,
            capsize=3.0,
            label=ONLINE_LABEL,
        )
        for level, seconds, error in zip(levels, online, errors):
            ax_on.annotate(
                f"L{level}",
                xy=(seconds, error),
                xytext=(8, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                color=self._INK_SECONDARY,
                fontsize=8,
            )
        times = list(online)
        if projected is not None:
            ax_on.axvline(
                projected, color=self.SERIES_COLORS[2], linewidth=1.7, linestyle=":"
            )
            ax_on.plot(
                [projected],
                [floor],
                color=self.SERIES_COLORS[2],
                marker=self.SERIES_MARKERS[2],
                markersize=7.0,
                markeredgecolor=self._SURFACE,
                markeredgewidth=1.2,
                linestyle="none",
                label=DIRECT_LABEL,
            )
            ax_on.annotate(
                "no interpolation\nerror",
                xy=(projected, floor),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="right",
                color="#1baf7a",
                fontsize=8,
            )
            times.append(projected)
            ax_on.set_ylim(bottom=floor / 1.6)
        ax_on.set_xlim(min(times) / 1.2, max(times) * 1.2)
        ax_on.set_xlabel("Benchmark runtime [s]", color=self._INK_SECONDARY)
        ax_on.set_title("online", color=self._INK_SECONDARY, fontsize=10)

        # Ticks at 1, 2 and 5 times powers of ten, written as plain numbers.
        for axis in (ax_on.xaxis, ax_on.yaxis):
            axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())

        legend = ax_on.legend(frameon=False, fontsize=8, loc="lower left")
        for text in legend.get_texts():
            text.set_color(self._INK_SECONDARY)
        figure.tight_layout()

        if output_dir is not None:
            self._save_cost_figure(figure, Path(output_dir), tag)
        return figure

    def plot(
        self,
        result: CostResult,
        output_dir: str | Path | None = None,
        tag: str = "",
    ) -> Figure | None:
        """Draw the cost-against-error figure on one axis.

        Parameters:
            result: The measurements; levels without both an error and a completed run are skipped.
            output_dir: If given, the figure is saved there as PNG and PDF.
            tag: Appended to the file name.

        Returns:
            The figure, or None if no level has both a cost and an error.

        """
        levels = [
            level
            for level in sorted(result.boundary_errors)
            if level in result.simulations
            and result.simulations[level].representative is not None
        ]
        if not levels:
            return None

        errors = np.array([result.boundary_errors[level] for level in levels])
        online = np.array(
            [result.simulations[level].total_seconds.median_seconds for level in levels]
        )
        spread = np.array(
            [result.simulations[level].total_seconds.spread_seconds for level in levels]
        )

        figure, ax = plt.subplots(figsize=(7.2, 5.0), facecolor=self._SURFACE)
        self._style_axes(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.errorbar(
            online,
            errors,
            xerr=spread / 2.0,
            color=self.SERIES_COLORS[0],
            marker=self.SERIES_MARKERS[0],
            markersize=7.0,
            markeredgecolor=self._SURFACE,
            markeredgewidth=1.2,
            linewidth=1.8,
            capsize=3.0,
            label=ONLINE_LABEL,
        )
        for level, seconds, error in zip(levels, online, errors):
            spacing = result.tables[level].spacing if level in result.tables else None
            ax.annotate(
                self._spacing_label(level, spacing),
                xy=(seconds, error),
                # xytext=(6, 6),
                xytext=(8, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                color=self._INK_SECONDARY,
                fontsize=8,
            )

        generation = [
            (result.tables[level].generation_seconds, error)
            for level, error in zip(levels, errors)
            if level in result.tables
            and result.tables[level].generation_seconds is not None
        ]
        if generation:
            ax.plot(
                [seconds for seconds, _ in generation],
                [error for _, error in generation],
                color=self.SERIES_COLORS[1],
                marker=self.SERIES_MARKERS[1],
                markersize=6.5,
                markeredgecolor=self._SURFACE,
                markeredgewidth=1.2,
                linewidth=1.7,
                linestyle="--",
                label=OFFLINE_LABEL,
            )

        floor = float(errors.min()) * 10.0**-FLOOR_DECADES
        projected = result.projected_direct_seconds(levels[-1])
        if projected is not None:
            ax.axvline(
                projected,
                color=self.SERIES_COLORS[2],
                linewidth=1.7,
                linestyle=":",
            )
            ax.plot(
                [projected],
                [floor],
                color=self.SERIES_COLORS[2],
                marker=self.SERIES_MARKERS[2],
                markersize=7.0,
                markeredgecolor=self._SURFACE,
                markeredgewidth=1.2,
                linestyle="none",
                label=DIRECT_LABEL,
            )
            ax.annotate(
                "no interpolation error",
                xy=(projected, floor),
                xytext=(-6, 10),
                textcoords="offset points",
                ha="right",
                color=self._INK_MUTED,
                fontsize=8,
            )
            ax.set_ylim(bottom=floor / 2.0)

        times = (
            list(online)
            + [s for s, _ in generation]
            + ([projected] if projected is not None else [])
        )
        ax.set_xlim(min(times) / 1.25, max(times) * 1.25)

        ax.set_xlabel("Computational time [s]", color=self._INK_SECONDARY)
        ax.set_ylabel(
            r"$L_2$ error, phase-boundary samples (mixture quantities)",
            color=self._INK_SECONDARY,
        )
        legend = ax.legend(frameon=False, fontsize=9, loc="best")
        for text in legend.get_texts():
            text.set_color(self._INK_SECONDARY)
        figure.tight_layout()

        if output_dir is not None:
            self._save_cost_figure(figure, Path(output_dir), tag)
        return figure

    @staticmethod
    def _spacing_label(level: int, spacing: tuple[float, float, float] | None) -> str:
        """Annotation for one point: the level and its spacings.

        Parameters:
            level: Refinement level.
            spacing: Table spacings, or None if unknown.

        Returns:
            The label.

        """
        if spacing is None:
            return f"L{level}"
        return f"L{level}: ({spacing[2]:g}, {spacing[1]:g}, {spacing[0]:g})"

    @staticmethod
    def _save_cost_figure(figure: Figure, directory: Path, tag: str) -> None:
        """Write the figure as PNG and PDF.

        Parameters:
            figure: The figure.
            directory: Output folder.
            tag: Appended to the file name.

        """

        directory.mkdir(parents=True, exist_ok=True)
        stem = "_".join(part for part in ("cost_against_error", tag) if part)
        for suffix in (".png", ".pdf"):
            figure.savefig(
                directory / f"{stem}{suffix}", dpi=600, facecolor=figure.get_facecolor()
            )
