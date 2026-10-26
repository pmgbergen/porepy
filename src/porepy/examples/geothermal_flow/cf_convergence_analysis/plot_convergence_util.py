"""Convergence figure of the manufactured-solution test.

Errors are plotted as ``log2(error)`` against ``log2(1/h)``, so a quantity
converging at order ``p`` is a line of slope ``-p``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


class PlotConvergenceUtil:
    """Two-panel convergence figure: primary variables (left), interface fluxes (right)."""

    SLIDE_STYLE = {
        "figsize": (6.0, 3.4),
        "legend_below": True,
        "fontsize": 10,
        "markersize": 4,
        "linewidth": 1.1,
    }


    PRIMARY_VARIABLES = {
        "pressure": "pressure",
        "enthalpy": "enthalpy",
        "z_NaCl": r"$z_{\mathrm{NaCl}}$",
    }
    INTERFACE_FLUXES = {
        "darcy_flux": "Darcy flux",
        "component_flux_NaCl": "NaCl flux",
        "advective_energy_flux": "advective energy flux",
        "conductive_energy_flux": "conductive energy flux",
        # "energy_flux": "total energy flux",
    }
    SUBDOMAIN_STYLES = {
        "matrix": {"marker": "o", "linestyle": "-"},
        "fracture": {"marker": "s", "linestyle": "--"},
    }

    def __init__(
        self,
        results: Sequence,
        primary_variables: Optional[dict[str, str]] = None,
        interface_fluxes: Optional[dict[str, str]] = None,
        reference_orders: Sequence[float] = (1, 2),
    ):
        if len(results) < 2:
            raise ValueError("At least two refinement levels are needed to plot convergence.")
        # Coarsest first, whatever order the levels were supplied in.
        self.results = sorted(results, key=lambda r: r.cell_diameter, reverse=True)
        self.primary_variables = dict(
            self.PRIMARY_VARIABLES if primary_variables is None else primary_variables
        )
        self.interface_fluxes = dict(
            self.INTERFACE_FLUXES if interface_fluxes is None else interface_fluxes
        )
        self.reference_orders = tuple(reference_orders)

    @property
    def log2_inverse_h(self) -> np.ndarray:
        """``log2(1/h)`` at every level, coarsest first."""
        h = np.array([r.cell_diameter for r in self.results], dtype=float)
        return np.log2(1.0 / h)

    def has_quantity(self, quantity: str) -> bool:
        """Whether the results carry ``error_<quantity>``.

        Parameters:
            quantity: Quantity name without the ``error_`` prefix.

        Returns:
            True if every level carries the quantity.

        """
        return all(hasattr(r, "error_" + quantity) for r in self.results)

    def log2_error(self, quantity: str) -> np.ndarray:
        """``log2`` of the error of one quantity at every level.

        Parameters:
            quantity: Quantity name without the ``error_`` prefix.

        Returns:
            The values, coarsest first; NaN where the error is zero.

        """
        errors = np.array([getattr(r, "error_" + quantity) for r in self.results], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(errors > 0.0, np.log2(errors), np.nan)

    def observed_order(self, quantity: str, data_range: slice = slice(-2, None)) -> float:
        """Least-squares order of convergence over ``data_range``.

        Parameters:
            quantity: Quantity name without the ``error_`` prefix.
            data_range: Levels used, by default the two finest.

        Returns:
            The fitted order.

        """
        x = self.log2_inverse_h[data_range]
        y = self.log2_error(quantity)[data_range]
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return float("nan")
        return float(-np.polyfit(x[valid], y[valid], 1)[0])

    def plot(
        self,
        save_path: Optional[str | Path] = None,
        figsize: tuple[float, float] = (12.0, 5.0),
        title: Optional[str] = None,
        legend_below: bool = False,
        fontsize: float = 10.0,
        markersize: float = 6.0,
        linewidth: float = 1.4,
        xlabel: Optional[str] = r"$\log_2(1/\Delta \zeta)$",
        primary_title: Optional[str] = None,
        interface_title: Optional[str] = None,
        show_legend: bool = True,
    ):
        """Draw both panels and optionally save the figure.

        Parameters:
            save_path: File to write; its parent directories are created.
            figsize: Figure size in inches.
            title: Figure title.
            legend_below: Put each legend below its panel (slide layout).
            fontsize: Base font size in points.
            markersize: Marker size of the data curves.
            linewidth: Line width of the data curves.
            xlabel: Label of both x axes; none if None.
            primary_title: Title of the left panel.
            interface_title: Title of the right panel.
            show_legend: Whether to draw the legends.

        Returns:
            The figure and the ``(primary, interface)`` axes.

        """
        fonts = {
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1.5,
            "ytick.labelsize": fontsize - 1.5,
            "legend.fontsize": fontsize - 2,
            "legend.title_fontsize": fontsize - 2,
        }
        self._markersize, self._linewidth = markersize, linewidth
        
        with plt.rc_context(fonts):
            fig, (ax_primary, ax_interface) = plt.subplots(1, 2, figsize=figsize)
            ylabel = r"$\log_2$ of relative $l_2$ error"
            self.plot_primary_variables(
                ax_primary,
                title=primary_title,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            self.plot_interface_fluxes(
                ax_interface,
                title=interface_title,
                xlabel=xlabel,
                ylabel=None if legend_below else ylabel,
            )

            if title is not None:
                fig.suptitle(title)
        
            fig.tight_layout()
            if legend_below:
                fig.subplots_adjust(wspace=0.25)

            if show_legend:
                if legend_below:
                    fig.canvas.draw()
                for ax in (ax_primary, ax_interface):
                    if legend_below:
                        label_bounds = (
                            ax.xaxis.label.get_window_extent()
                            .transformed(ax.transAxes.inverted())
                        )
                        ax.legend(
                            loc="upper center",
                            bbox_to_anchor=(0.5, label_bounds.y0),
                            ncol=2,
                            frameon=False,
                            handlelength=2.4,
                            columnspacing=1.0,
                        )
                    else:
                        ax.legend(fontsize=fontsize - 2)
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=600, bbox_inches="tight")
        return fig, (ax_primary, ax_interface)


    def plot_primary_variables(self, ax, **panel_options) -> None:
        """Matrix and fracture primary variables on one axis.

        Parameters:
            ax: Axis to draw on.
            panel_options: ``title``, ``xlabel`` and ``ylabel`` of the panel.

        """
        curves = []
        for index, (variable, label) in enumerate(self.primary_variables.items()):
            for subdomain, style in self.SUBDOMAIN_STYLES.items():
                quantity = f"{subdomain}_{variable}"
                if self.has_quantity(quantity):
                    curves.append((quantity, f"{subdomain} {label}", {"color": f"C{index}", **style}))
        self._draw_panel(ax, curves, **panel_options)

    def plot_interface_fluxes(self, ax, **panel_options) -> None:
        """Interface fluxes on one axis.

        Parameters:
            ax: Axis to draw on.
            panel_options: ``title``, ``xlabel`` and ``ylabel`` of the panel.

        """
        curves = [
            (f"interface_{name}", label, {"color": f"C{index}", "marker": "^", "linestyle": "-"})
            for index, (name, label) in enumerate(self.interface_fluxes.items())
            if self.has_quantity(f"interface_{name}")
        ]
        self._draw_panel(ax, curves, **panel_options)

    def _draw_panel(
            self, ax, curves, *, title=None, xlabel=None, ylabel=None
        ) -> None:
        if not curves:
            raise ValueError(f"None of the quantities for '{title}' are in the results.")
        
        x = self.log2_inverse_h
        for quantity, label, style in curves:
            ax.plot(x, self.log2_error(quantity), label=f"{label}", #({order:.2f})
                markersize=getattr(self, "_markersize", 6.0),
                linewidth=getattr(self, "_linewidth", 1.4), **style)
            
        self._draw_reference_orders(ax, [quantity for quantity, _, _ in curves])
        span = x[-1] - x[0]
        ax.set_xlim(x[0] - 0.05 * span, x[-1] + 0.2 * span)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:.2f}" for value in x])
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    
    def _draw_reference_orders(self, ax, quantities, margin: float = 0.70) -> None:
        """Dotted reference slopes over the coarsest interval, drawn below every curve.

        Parameters:
            ax: Axis to draw on.
            quantities: Quantities drawn on the axis.
            margin: Gap below the curves, in ``log2`` units.

        """
        
        x = self.log2_inverse_h[:2]
        first = np.array([self.log2_error(q)[0] for q in quantities])
        second = np.array([self.log2_error(q)[1] for q in quantities])
        ok = np.isfinite(first) & np.isfinite(second)
        if not ok.any():
            return
        anchor = min(first[ok].min(), second[ok].min() + min(self.reference_orders) * (x[1] - x[0]))
        anchor -= margin
        for order in self.reference_orders:
            y = anchor - order * (x - x[0])
            ax.plot(x, y, color="0.4", linestyle=":", linewidth=1.4)
            ax.annotate(f"order {order:g}", xy=(x[-1], y[-1]), xytext=(3, 0),
                        textcoords="offset points", color="0.4", fontsize="small", va="center")
