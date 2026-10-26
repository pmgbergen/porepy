"""Mesh-ensemble averaging for convergence studies on unstructured grids.

Each refinement level is run on several triangulations, with the cell sizes
scaled by ``1 + jitter``, and the errors are averaged, because a single
triangulation gives noisy errors for some quantities.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy import stats

from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_convergence import (
    ParamsAwareConvergenceAnalysis,
)

DEFAULT_JITTERS: tuple[float, ...] = (0.0, -0.16, 0.16, -0.12, 0.12, -0.08, 0.08, -0.04, 0.04)
"""tuple[float, ...]: Relative cell-size perturbations, the unperturbed mesh first."""

AVERAGES: dict[str, Callable[[np.ndarray], float]] = {
    "arithmetic": lambda values: float(np.mean(values)),
    "geometric": lambda values: float(np.exp(np.mean(np.log(values)))),
    "median": lambda values: float(np.median(values)),
}
"""dict: Averages available for combining the members of a level."""


def error_names(result) -> list[str]:
    """Sorted names of the ``error_*`` attributes of a result.

    Parameters:
        result: A result object.

    Returns:
        The attribute names, prefix included.

    """
    return sorted(name for name in vars(result) if name.startswith("error_"))


def run_signature(result, digits: int = 9) -> tuple[float, ...]:
    """Hashable signature of a run's errors, used to detect repeated triangulations.

    Parameters:
        result: A result object.
        digits: Significant figures kept.

    Returns:
        The signature.

    """
    values = [result.cell_diameter] + [getattr(result, name) for name in error_names(result)]
    return tuple(float(f"{value:.{digits}g}") for value in values)


class MeshEnsembleConvergenceMixin:
    """Run several meshes per refinement level and average their errors.

    Place it before a ``ConvergenceAnalysis`` subclass in the bases. Settings come
    from class attributes or the ``"mesh_ensemble"`` entry of the model parameters;
    member results are kept in ``ensemble_results``.
    """

    ensemble_jitters: Union[Sequence[float], dict[int, Sequence[float]]] = DEFAULT_JITTERS
    ensemble_average: str = "arithmetic"
    min_distinct_meshes: int = 3

    # ------------------------------------------------------------------ settings
    def ensemble_settings(self) -> dict:
        """Ensemble settings, with parameter overrides applied.

        Returns:
            The ``jitters``, ``average`` and ``min_distinct_meshes`` in force.

        """
        overrides = self.model_params[0].get("mesh_ensemble", {})
        settings = {
            "jitters": overrides.get("jitters", self.ensemble_jitters),
            "average": overrides.get("average", self.ensemble_average),
            "min_distinct_meshes": overrides.get("min_distinct_meshes", self.min_distinct_meshes),
        }
        if settings["average"] not in AVERAGES:
            raise ValueError(
                f"Unknown average {settings['average']!r}; choose from {sorted(AVERAGES)}."
            )
        return settings

    def jitters_for_level(self, level: int) -> tuple[float, ...]:
        """Cell-size perturbations used on one level.

        Parameters:
            level: Refinement level.

        Returns:
            The perturbations.

        Raises:
            ValueError: If a cell size would be non-positive or overlap the next level.

        """
        jitters = self.ensemble_settings()["jitters"]
        if isinstance(jitters, dict):
            jitters = jitters[level]
        jitters = tuple(float(jitter) for jitter in jitters)
        if not jitters:
            raise ValueError(f"No jitters given for level {level}.")
        if min(jitters) <= -1.0:
            raise ValueError("A jitter of -1 or less gives a non-positive cell size.")
        spread = (1.0 + max(jitters)) / (1.0 + min(jitters))
        if spread >= self.spatial_refinement_rate:
            raise ValueError(
                f"The jitters span a factor {spread:.2f} in cell size, which is not "
                f"below the refinement rate {self.spatial_refinement_rate}: members "
                "of neighbouring levels would overlap."
            )
        return jitters

    # ------------------------------------------------------------------ running
    def run_analysis(self) -> list:
        """Run every member of every level and average the errors per level.

        Returns:
            One averaged result per level, coarsest first.

        """
        average = AVERAGES[self.ensemble_settings()["average"]]
        self.ensemble_results = []
        self.ensemble_duplicates = []
        averaged = []
        for level in range(self.levels):
            members = self.run_level_ensemble(level)
            self.ensemble_results.append(members)
            averaged.append(self.average_results(members, average))
        return averaged

    def run_level_ensemble(self, level: int) -> list:
        """Run the members of one level, dropping repeated triangulations.

        Parameters:
            level: Refinement level.

        Returns:
            The distinct member results.

        Raises:
            ValueError: If the parameters ask for a structured grid.

        """
        base = self.model_params[level]
        grid_type = base.get("grid_type")
        if grid_type is not None and grid_type != "simplex":
            raise ValueError(
                f"Mesh ensembles need an unstructured grid, got grid_type={grid_type!r}. "
                "A perturbed Cartesian cell size no longer aligns with the fracture."
            )
        jitters = self.jitters_for_level(level)
        minimum = self.ensemble_settings()["min_distinct_meshes"]

        members, seen = [], set()
        for jitter in jitters:
            result = self.run_single(self.jittered_params(base, jitter))
            result.mesh_jitter = jitter
            result.ensemble_level = level
            signature = run_signature(result)
            if signature in seen:
                self.ensemble_duplicates.append((level, jitter))
                continue
            seen.add(signature)
            members.append(result)

        if len(members) < minimum:
            warnings.warn(
                f"Level {level} has only {len(members)} distinct meshes out of "
                f"{len(jitters)} runs; gmsh returned the same triangulation for "
                "several cell sizes. Add jitters for this level.",
                stacklevel=2,
            )
        return members

    @staticmethod
    def jittered_params(params: dict, jitter: float) -> dict:
        """Copy of the parameters with every ``cell_size*`` entry scaled by ``1 + jitter``.

        Parameters:
            params: Parameters of one level.
            jitter: Relative perturbation.

        Returns:
            The perturbed parameters; the input is not modified.

        """
        params = deepcopy(params)
        meshing = dict(params["meshing_arguments"])
        for key, value in meshing.items():
            if key.startswith("cell_size"):
                meshing[key] = value * (1.0 + jitter)
        params["meshing_arguments"] = meshing
        return params

    def run_single(self, params: dict):
        """Run one simulation through the parent's ``run_analysis``.

        Parameters:
            params: Complete model parameters of the run.

        Returns:
            The parent's result for the run.

        """
        saved = self.levels, self.model_params
        self.levels, self.model_params = 1, [params]
        try:
            return super().run_analysis()[0]
        finally:
            self.levels, self.model_params = saved

    # ------------------------------------------------------------------ assembling
    def average_results(self, members: Sequence, average: Optional[Callable] = None):
        """Combine the members of one level into one result with averaged errors.

        Parameters:
            members: Member results of one level.
            average: Reduction of an array to a float; the configured average if None.

        Returns:
            The averaged result, with ``ensemble_size``, ``ensemble_jitters`` and
            ``ensemble_spread`` added.

        Raises:
            ValueError: If the members disagree on the time step.

        """
        if average is None:
            average = AVERAGES[self.ensemble_settings()["average"]]
        first = members[0]
        names = error_names(first)
        updates = {
            name: average(np.array([getattr(m, name) for m in members], dtype=float))
            for name in names
        }
        updates["cell_diameter"] = float(np.mean([m.cell_diameter for m in members]))

        if hasattr(first, "dt"):
            steps = {float(m.dt) for m in members}
            if len(steps) > 1:
                raise ValueError(f"Members of one level use different time steps: {sorted(steps)}")

        if is_dataclass(first):
            field_names = {f.name for f in fields(first)}
            changes = {key: value for key, value in updates.items() if key in field_names}
            if "values" in field_names:
                changes["values"] = {}
            if "interface_sides" in field_names:
                changes["interface_sides"] = self._average_sides(members, average)
            result = replace(first, **changes)
            for key, value in updates.items():
                if key not in field_names:
                    setattr(result, key, value)
            if hasattr(first, "dt") and "dt" not in field_names:
                result.dt = first.dt
        else:
            kept = {
                key: value for key, value in vars(first).items()
                if not key.startswith("error_")
                and key not in updates
                and key not in ("mesh_jitter", "ensemble_level")
            }
            result = SimpleNamespace(**kept, **updates)

        # None of these names may start with "error_": order_of_convergence treats
        # every such attribute as a quantity to fit.
        result.ensemble_size = len(members)
        result.ensemble_jitters = [getattr(m, "mesh_jitter", None) for m in members]
        result.ensemble_spread = {
            name.removeprefix("error_"): self._spread(
                np.array([getattr(m, name) for m in members], dtype=float)
            )
            for name in names
        }
        return result

    @staticmethod
    def _spread(values: np.ndarray) -> dict:
        positive = values[values > 0]
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "max_over_min": float(values.max() / positive.min()) if positive.size else float("nan"),
            "cv": float(values.std(ddof=1) / values.mean()) if values.size > 1 else float("nan"),
        }

    @staticmethod
    def _average_sides(members: Sequence, average: Callable) -> dict:
        sides = [getattr(m, "interface_sides", None) for m in members]
        if not sides or any(not isinstance(s, dict) for s in sides):
            return {}
        layout = {side: sorted(values) for side, values in sides[0].items()}
        if any({side: sorted(values) for side, values in s.items()} != layout for s in sides):
            return {}
        return {
            side: {
                name: average(np.array([s[side][name] for s in sides], dtype=float))
                for name in names
            }
            for side, names in layout.items()
        }

    # ------------------------------------------------------------------ uncertainty
    def pooled_order_of_convergence(self, name: str, data_range: slice = slice(None)) -> dict:
        """Least-squares order through every member run.

        Parameters:
            name: Quantity name without the ``error_`` prefix.
            data_range: Levels included.

        Returns:
            The ``order``, ``stderr``, ``r_squared`` and number of ``runs``.

        """
        levels = self.ensemble_results[data_range]
        h = np.array([m.cell_diameter for level in levels for m in level], dtype=float)
        e = np.array([getattr(m, "error_" + name) for level in levels for m in level], dtype=float)
        fit = stats.linregress(np.log2(h), np.log2(e))
        return {"order": float(fit.slope), "stderr": float(fit.stderr),
                "r_squared": float(fit.rvalue**2), "runs": int(e.size)}

    def bootstrap_order_of_convergence(
        self,
        name: str,
        samples: int = 2000,
        confidence: float = 0.95,
        data_range: slice = slice(None),
        average: Optional[str] = None,
        seed: int = 0,
    ) -> dict:
        """Bootstrap confidence interval for the order of the level averages.

        Parameters:
            name: Quantity name without the ``error_`` prefix.
            samples: Number of resamples.
            confidence: Coverage of the interval.
            data_range: Levels used in each fit.
            average: Average to use; the configured one if None.
            seed: Seed of the random generator.

        Returns:
            The ``order``, ``low``, ``high``, ``median`` and ``samples``.

        """
        reduce = AVERAGES[average or self.ensemble_settings()["average"]]
        variable = ["error_" + name]
        key = "ooc_" + name
        full = [self.average_results(level, reduce) for level in self.ensemble_results]
        order = self.order_of_convergence(full, variables=variable, data_range=data_range)[key]

        rng = np.random.default_rng(seed)
        estimates = np.empty(samples)
        for index in range(samples):
            resampled = [
                self.average_results(
                    [level[i] for i in rng.integers(0, len(level), len(level))], reduce
                )
                for level in self.ensemble_results
            ]
            estimates[index] = self.order_of_convergence(
                resampled, variables=variable, data_range=data_range
            )[key]
        tail = 100.0 * (1.0 - confidence) / 2.0
        low, high = np.percentile(estimates, [tail, 100.0 - tail])
        return {"order": float(order), "low": float(low), "high": float(high),
                "median": float(np.median(estimates)), "samples": samples}

    # ------------------------------------------------------------------ reporting
    def report_ensemble(self, names: Optional[Sequence[str]] = None) -> None:
        """Print the members and the spread of each quantity per level.

        Parameters:
            names: Quantities without the ``error_`` prefix; all if None.

        """
        if names is None:
            names = [n.removeprefix("error_") for n in error_names(self.ensemble_results[0][0])]
        print("\nMesh ensemble")
        for level, members in enumerate(self.ensemble_results):
            dropped = [j for lvl, j in self.ensemble_duplicates if lvl == level]
            h = [m.cell_diameter for m in members]
            print(f"level {level}: {len(members)} distinct meshes"
                  + (f" ({len(dropped)} duplicate(s) dropped: {dropped})" if dropped else "")
                  + f", h {min(h):.4g}..{max(h):.4g}")
        header = (f"{'quantity':34s} {'level':>5s} {'mean':>11s} {'min':>11s} "
                  f"{'max':>11s} {'max/min':>8s} {'cv':>6s}")
        print("\n" + header)
        print("-" * len(header))
        for name in names:
            for level, members in enumerate(self.ensemble_results):
                values = np.array([getattr(m, "error_" + name) for m in members], dtype=float)
                spread = self._spread(values)
                print(f"{name if level == 0 else '':34s} {level:5d} {values.mean():11.3e} "
                      f"{spread['min']:11.3e} {spread['max']:11.3e} "
                      f"{spread['max_over_min']:8.1f} {spread['cv']:6.2f}")


class EnsembleConvergenceAnalysis(MeshEnsembleConvergenceMixin, ParamsAwareConvergenceAnalysis):
    """:class:`ParamsAwareConvergenceAnalysis` with several meshes per level, averaged."""
    pass


def plot_ensemble_members(
    ax,
    ensemble_results: Sequence[Sequence],
    name: str,
    color: str = "0.4",
    label: Optional[str] = "individual meshes",
    highlight_jitter: Optional[float] = 0.0,
) -> None:
    """Overlay every member run on an axis drawn by ``PlotConvergenceUtil``.

    Parameters:
        ax: Axis to draw on.
        ensemble_results: Member results per level.
        name: Quantity name without the ``error_`` prefix.
        color: Marker colour.
        label: Legend label of the members; none if None.
        highlight_jitter: Jitter of the members to circle; none if None.

    """
    xs, ys, marked = [], [], []
    for level in ensemble_results:
        for member in level:
            x = float(np.log2(1.0 / member.cell_diameter))
            y = float(np.log2(getattr(member, "error_" + name)))
            xs.append(x)
            ys.append(y)
            if highlight_jitter is not None and getattr(member, "mesh_jitter", None) == highlight_jitter:
                marked.append((x, y))
    ax.scatter(xs, ys, s=16, color=color, alpha=0.4, zorder=1, label=label)
    if marked:
        mx, my = zip(*marked)
        ax.scatter(mx, my, s=70, facecolors="none", edgecolors="k", zorder=3,
                   label=f"jitter {highlight_jitter:g}" if label else None)
    low, high = ax.get_xlim()
    span = max(xs) - min(xs)
    ax.set_xlim(min(low, min(xs) - 0.05 * span), max(high, max(xs) + 0.2 * span))
    if label:
        ax.legend(fontsize=8)