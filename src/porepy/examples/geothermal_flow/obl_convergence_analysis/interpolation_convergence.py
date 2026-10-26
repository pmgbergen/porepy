"""Experiment A, convergence of interpolated property values, and the infrastructure shared with Experiment B.

Contains the refinement sequence, direct evaluation with the ``swEOS`` CLI, table
sampling through ``VTKSampler``, the sample sets and :class:`ConvergenceStudy`.

"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_metrics import (
    variation_normalized_l2_error,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_plotting import (
    ConvergencePlotMixin,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_reporting import (
    report_convergence,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_results import (
    ConvergenceResult,
    RefinementResult,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler


FIELD_MAP_CLI: dict[str, str] = {
    # canonical name -> column name printed by swEOS
    "Temperature": "TC",
    "S_l": "Liquid_Saturation",
    "S_v": "Vapour_Saturation",
    "S_h": "Halite_Saturation",
    "Xl": "Liquid_salinity",
    "Xv": "Vapour_salinity",
    "Rho_l": "Liquid_densitykgm3",
    "Rho_v": "Vapour_densitykgm3",
    "Rho_h": "Halite_densitykgm3",
    "Rho": "Bulk_densitykgm3",
    "mu_l": "Liquid_viscosity",
    "mu_v": "Vapour_viscosity",
    "H_l": "Liquid_enthalpykJkg",
    "H_v": "Vapour_enthalpykJkg",
    "H_h": "Halite_enthalpykJkg",
    "H": "Bulk_enthalpykJkg",
}
"""dict: Canonical field names mapped to CLI column names."""

SATURATION_FIELDS = ("S_l", "S_v", "S_h")
"""tuple: Fields used to classify the phase state of a point."""

REPAIR_MASKS = ("mask_nan", "mask_clip", "mask_floor", "mask_extend")
"""tuple: Per-node indicators of NaN filling, clipping, flooring and extension; only ``mask_nan`` defines the categories."""

ALL = "all points"
"""str: Category holding every point of a sample set."""

CLEAN = "clean"
"""str: Category free of NaN-repaired cell corners; interior samples must also lie in one phase region."""

NEAR_REPAIR = "near repair"
"""str: Category whose cells touch a NaN-repaired node at a selected level."""


@dataclass(frozen=True)
class RefinementSequence:
    """A halving sequence of table resolutions; all three spacings halve together."""

    initial_spacing: tuple[float, float, float]
    num_levels: int
    z_range: tuple[float, float] = (1.0e-4, 1.0)
    h_range: tuple[float, float] = (100.0, 3000.0)
    p_range: tuple[float, float] = (10.0, 160.0)

    RANGE_PADDING: ClassVar[float] = 1.0e-3
    """Fraction of a step added to each maximum passed to the CLI."""

    def spacing_at(self, level: int) -> tuple[float, float, float]:
        """Spacings at one level of the sequence.

        Parameters:
            level: Index into the sequence, zero being coarsest.

        Returns:
            The triple ``(eta_z, eta_h, eta_p)``.

        """
        factor = 2.0 ** (-level)
        return tuple(e * factor for e in self.initial_spacing)  # type: ignore[return-value]

    def node_count(self, level: int) -> int:
        """Number of table nodes at one level.

        Parameters:
            level: Index into the sequence, zero being coarsest.

        Returns:
            The node count.

        """
        count = 1
        for low, eta, high in self.cli_bounds(level):
            count *= int(np.floor((high - low) / eta)) + 1
        return count

    def cli_bounds(self, level: int) -> tuple[tuple[float, float, float], ...]:
        """Bounds of the table at one level, as passed to the CLI.

        Parameters:
            level: Index into the sequence, zero being coarsest.

        Returns:
            ``(minimum, spacing, maximum)`` for z, h and p in turn.

        """
        ez, eh, ep = self.spacing_at(level)
        pad = self.RANGE_PADDING
        return (
            (self.z_range[0], ez, min(self.z_range[1] + pad * ez, 1.0)),
            (self.h_range[0], eh, self.h_range[1] + pad * eh),
            (self.p_range[0], ep, self.p_range[1] + pad * ep),
        )

    def cli_range_string(self, level: int) -> str:
        """Format the table bounds for the CLI's ``-R`` option.

        Parameters:
            level: Index into the sequence, zero being coarsest.

        Returns:
            ``z_min/eta_z/z_max/h_min/eta_h/h_max/p_min/eta_p/p_max``.

        """
        return "/".join(f"{value}" for axis in self.cli_bounds(level) for value in axis)


def write_cli_batch(points: np.ndarray, path: Path) -> None:
    """Write sample points in the CLI batch format, without a trailing newline.

    Parameters:
        points: Array of shape ``(n, 3)`` holding ``(z, h, p)`` per row.
        path: File to write.

    """
    reordered = np.asarray(points, dtype=float)[:, [2, 1, 0]]  # (z,h,p)->(p,h,z)
    lines = [" ".join(f"{v:.12g}" for v in row) for row in reordered]
    path.write_text("\n".join(lines))


def read_cli_output(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    """Parse the CLI batch output; values the CLI could not compute stay NaN.

    Parameters:
        path: CLI output file.
        fields: Canonical field names to extract.

    Returns:
        Field name to its values.

    Raises:
        KeyError: If a requested field is absent from the output.

    """
    raw = np.genfromtxt(path, names=True, dtype=None, encoding=None, delimiter=",")

    available = set(raw.dtype.names or ())
    out: dict[str, np.ndarray] = {}
    for name in fields:
        column = FIELD_MAP_CLI.get(name, name)
        if column not in available:
            raise KeyError(
                f"field {column!r} absent from CLI output; "
                f"available: {sorted(available)}"
            )
        out[name] = np.asarray(raw[column], dtype=float)
    return out


PHASE_FIELDS: dict[str, tuple[str, ...]] = {
    "S_l": ("H_l", "Rho_l", "mu_l", "Xl"),
    "S_v": ("H_v", "Rho_v", "mu_v", "Xv"),
    "S_h": ("H_h", "Rho_h"),
}
"""dict: Saturation field mapped to the properties of that phase."""

VISCOSITY_FIELDS = ("mu_l", "mu_v")
"""tuple: Viscosity fields; swEOS floors their zeros to 2e-5 Pa s."""

TABLE_CONVENTION_INPUTS = ("Rho_l", "Rho_v")
"""tuple: CLI fields needed to recognise the states the correlations do not cover."""

VAPOUR_VISCOSITY_MAX_T = 800.0
"""float: Temperature [deg C] above which the vapour viscosity is left out of the errors."""


def convert_reference_to_table_conventions(
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Express CLI output in the units of the VTK tables, with absent-phase properties set to NaN.

    Parameters:
        values: CLI output by canonical field name, in CLI units.

    Returns:
        A new mapping with the same keys, in table units.

    """
    out = {name: array.copy() for name, array in values.items()}

    if "Temperature" in out:
        out["Temperature"] += 273.15

    for name in ("H", "H_l", "H_v", "H_h"):
        if name in out:
            out[name] *= 1.0e3

    if "H" in out:
        unknown = np.isnan(values["Rho_l"]) & np.isnan(values["Rho_v"])
        out["H"][unknown] = np.nan

    if "mu_v" in out and "Temperature" in out:
        out["mu_v"][out["Temperature"] > VAPOUR_VISCOSITY_MAX_T + 273.15] = np.nan

    for saturation, properties in PHASE_FIELDS.items():
        if saturation not in values:
            continue
        absent = ~(values[saturation] > 0.0)
        for name in properties:
            if name in out:
                out[name][absent] = np.nan

    return out


def evaluate_direct(
    points: np.ndarray,
    fields: Sequence[str],
    cli_path: str,
    cwd: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the correlations directly at scattered points, in the table conventions.

    Parameters:
        points: Array of shape ``(n, 3)`` holding ``(z, h, p)`` per row.
        fields: Canonical field names to return.
        cli_path: Path to the ``swEOS`` executable.
        cwd: Working directory of the call.

    Returns:
        Field name to its values at the points.

    Raises:
        RuntimeError: If the CLI fails, writes nothing, or returns the wrong number of rows.

    """
    requested = list(fields)
    needed = requested + [
        name
        for name in (*TABLE_CONVENTION_INPUTS, *PHASE_FIELDS)
        if name not in requested
    ]

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "points.txt"
        out_path = Path(tmp) / "values.csv"  # csv, per the manual examples
        write_cli_batch(points, in_path)
        command = [
            cli_path,
            "-D",
            "0",
            "-V",
            "PHX",
            "-G",
            str(in_path),
            "-O",
            str(out_path),
        ]
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CLI failed ({result.returncode}).\n"
                f"cmd: {' '.join(command)}\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        if not out_path.exists():
            raise RuntimeError(
                f"CLI exited 0 but wrote nothing.\n"
                f"cmd: {' '.join(command)}\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        values = read_cli_output(out_path, needed)
        n_rows = len(next(iter(values.values())))
        if n_rows != len(points):
            raise RuntimeError(f"CLI returned {n_rows} rows for {len(points)} points")
        converted = convert_reference_to_table_conventions(values)
        return {name: converted[name] for name in requested}


@dataclass(frozen=True)
class TableLattice:
    """Lattice position of each node of a tensor-product table, located by its coordinates."""

    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    index: tuple[np.ndarray, np.ndarray, np.ndarray]

    @classmethod
    def of(cls, grid) -> TableLattice:
        """Build the lattice of a table.

        Parameters:
            grid: Table whose nodes form a tensor-product lattice.

        Returns:
            The lattice.

        Raises:
            ValueError: If the nodes do not form a tensor-product lattice.

        """
        nodes = np.asarray(grid.points, dtype=float)
        axes = tuple(np.unique(nodes[:, k]) for k in range(3))
        shape = tuple(len(axis) for axis in axes)
        if int(np.prod(shape)) != len(nodes):
            raise ValueError(f"{len(nodes)} table nodes do not form a {shape} lattice")
        index = tuple(np.searchsorted(axes[k], nodes[:, k]) for k in range(3))
        return cls(axes, index)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of nodes in each direction."""
        return tuple(len(axis) for axis in self.axes)

    def to_lattice(self, values: np.ndarray) -> np.ndarray:
        """Arrange node values, given in table order, on the lattice.

        Parameters:
            values: One value per node, in the node order of the table.

        Returns:
            Array of shape :attr:`shape`.

        """
        values = np.asarray(values)
        lattice = np.empty(self.shape, dtype=values.dtype)
        lattice[self.index] = values
        return lattice

    def lower_corner(self, points: np.ndarray) -> tuple[np.ndarray, ...]:
        """Lattice index of the lowest corner of the cell containing each point.

        Parameters:
            points: Array of shape ``(n, 3)`` in table coordinates; points outside take the nearest boundary cell.

        Returns:
            One integer array per direction.

        """
        return tuple(
            np.clip(
                np.searchsorted(self.axes[k], points[:, k], side="right") - 1,
                0,
                len(self.axes[k]) - 2,
            )
            for k in range(3)
        )

    def nearest(self, valid: np.ndarray) -> tuple[np.ndarray, ...]:
        """For every node, the nearest node where ``valid`` holds, counted in grid steps.

        Parameters:
            valid: Boolean array of shape :attr:`shape`.

        Returns:
            Lattice index of the nearest valid node, one array per direction, in table order.

        """
        _, indices = distance_transform_edt(~valid, return_indices=True)
        return tuple(indices[k][self.index] for k in range(3))


class TabulatedPropertyEvaluator:
    """Table lookup through the solver's ``VTKSampler``, including its coordinate conversion and out-of-bounds rule."""

    def __init__(
        self,
        path: str | Path,
        conversion_factors: tuple[float, float, float] = (1.0, 1.0, 1.0),
        translation_factors: tuple[float, float, float] = (0.0, 0.0, 0.0),
        extend_q: bool = True,
        first_order_expansion: bool = False,
    ) -> None:
        self.path = Path(path)
        self.sampler = VTKSampler(
            str(self.path),
            extend_q=extend_q,
            first_order_expansion=first_order_expansion,
        )
        self.sampler.conversion_factors = conversion_factors
        self.sampler.translation_factors = translation_factors
        self._lattice: TableLattice | None = None
        self._node_values: dict[str, np.ndarray] = {}

    def has_array(self, name: str) -> bool:
        """Whether the table carries a node array of this name.

        Parameters:
            name: VTK array name.

        Returns:
            True if the array is present.

        """
        return name in self.sampler.search_space.point_data

    def cell_corners(self, points: np.ndarray, name: str) -> np.ndarray:
        """Node values of a table array at the eight corners of each point's cell.

        Parameters:
            points: Array of shape ``(n, 3)`` in solver coordinates, before conversion.
            name: VTK array name.

        Returns:
            Array of shape ``(n, 8)``.

        Raises:
            ValueError: If the table nodes do not form a tensor-product lattice.

        """
        grid = self.sampler.search_space
        if self._lattice is None:
            self._lattice = TableLattice.of(grid)
        if name not in self._node_values:
            self._node_values[name] = self._lattice.to_lattice(
                np.asarray(grid.point_data[name], dtype=float).ravel()
            )
        values = self._node_values[name]
        lower = self._lattice.lower_corner(self._to_table_coordinates(points))
        return np.stack(
            [
                values[lower[0] + i, lower[1] + j, lower[2] + k]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ],
            axis=1,
        )

    def _to_table_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Apply the sampler's conversion and translation to query points.

        Parameters:
            points: Query points in solver coordinates.

        Returns:
            The points in table coordinates.

        """
        return np.asarray(points, dtype=float) * np.asarray(
            self.sampler.conversion_factors
        ) + np.asarray(self.sampler.translation_factors)

    def sample(
        self, points: np.ndarray, fields: Sequence[str], want_gradients: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Interpolate values, gradients and repair masks in one pass.

        Parameters:
            points: Array of shape ``(n, 3)`` in solver coordinates, before conversion.
            fields: Canonical field names.
            want_gradients: Whether to return the ``grad_*`` vectors as well.

        Returns:
            ``(values, gradients, out_of_bounds)``; ``gradients`` is empty unless requested.

        Raises:
            KeyError: If a requested field is absent from the table.

        """
        points = np.asarray(points, dtype=float)

        # Record which points fall outside before sampling, since with
        # extend_q set the sampler will silently repair them and the fact that
        # it did so is part of what this study reports.
        oob = self.sampler._capture_points_outside_bounds(
            self._to_table_coordinates(points)
        )

        self.sampler.sample_at(points)
        cloud = self.sampler.sampled_cloud

        values: dict[str, np.ndarray] = {}
        gradients: dict[str, np.ndarray] = {}
        for name in fields:
            if name not in cloud.array_names:
                raise KeyError(
                    f"field {name!r} absent from {self.path.name}; "
                    f"available: {sorted(cloud.array_names)}"
                )
            values[name] = np.asarray(cloud[name], dtype=float).copy()
            if want_gradients and ("grad_" + name) in cloud.array_names:
                gradients[name] = np.asarray(cloud["grad_" + name], dtype=float).copy()

        # The repair masks ride along as ordinary fields, so they are sampled
        # by the same interpolation and can be read from the same cloud.
        for mask in REPAIR_MASKS:
            if mask in cloud.array_names:
                values[mask] = np.asarray(cloud[mask], dtype=float).copy()

        return values, gradients, oob


@dataclass
class SampleSet:
    """A fixed set of states at which the two evaluations are compared, the same at every resolution."""

    name: str
    points: np.ndarray
    description: str
    single_phase_cells: bool = False

    def __len__(self) -> int:
        return int(self.points.shape[0])


def classify_phase(values: dict[str, np.ndarray], tol: float = 1e-6) -> np.ndarray:
    """Label each point by the phases present.

    Parameters:
        values: Mapping containing the saturation fields.
        tol: Saturation below which a phase counts as absent.

    Returns:
        Integer array whose bits mark liquid, vapour and halite; 0 means no phase.

    """
    label = np.zeros(len(next(iter(values.values()))), dtype=int)
    for bit, name in enumerate(SATURATION_FIELDS):
        if name in values:
            label |= (values[name] > tol).astype(int) << bit
    return label


def build_interior_samples(
    sequence: RefinementSequence,
    evaluate: Callable[[np.ndarray], dict[str, np.ndarray]],
    scan_points_per_axis: int = 40,
    max_interior_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> SampleSet:
    """Points in the interior of single-phase regions.

    Parameters:
        sequence: Supplies the tabulated intervals.
        evaluate: Direct evaluation, used to classify the scan.
        scan_points_per_axis: Points per direction in the classification scan.
        max_interior_samples: Number of samples to retain.
        rng: Random generator.

    Returns:
        The sample set.

    """
    rng = np.random.default_rng(0) if rng is None else rng
    axes = [
        np.linspace(*sequence.z_range, scan_points_per_axis),
        np.linspace(*sequence.h_range, scan_points_per_axis),
        np.linspace(*sequence.p_range, scan_points_per_axis),
    ]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    flat = mesh.reshape(-1, 3)
    label = classify_phase(evaluate(flat)).reshape(
        scan_points_per_axis, scan_points_per_axis, scan_points_per_axis
    )

    interior = np.ones_like(label, dtype=bool)
    for axis in range(3):
        for shift in (-1, 1):
            interior &= label == np.roll(label, shift, axis=axis)
    interior[0, :, :] = interior[-1, :, :] = False
    interior[:, 0, :] = interior[:, -1, :] = False
    interior[:, :, 0] = interior[:, :, -1] = False
    # A block of the unknown region is uniform too, but it is not a phase.
    interior &= label != 0

    candidates = mesh[interior]
    if len(candidates) == 0:
        raise RuntimeError(
            "no single-phase interior found; the scan may be too coarse "
            "to resolve the phase regions"
        )
    take = rng.choice(
        len(candidates), size=min(max_interior_samples, len(candidates)), replace=False
    )
    return SampleSet(
        name="bulk",
        points=candidates[take],
        description="single-phase interior, one scan cell clear of any transition",
        single_phase_cells=True,
    )


def build_phase_boundary_samples(
    sequence: RefinementSequence,
    evaluate: Callable[[np.ndarray], dict[str, np.ndarray]],
    num_boundary_paths: int = 500,
    points_per_path: int = 40,
    rng: np.random.Generator | None = None,
) -> SampleSet:
    """Points on trajectories that cross a phase boundary.

    Parameters:
        sequence: Supplies the tabulated intervals.
        evaluate: Direct evaluation, used to locate the crossings.
        num_boundary_paths: Number of trajectories to attempt.
        points_per_path: Points per trajectory in the locating scan.
        rng: Random generator.

    Returns:
        The sample set.

    Raises:
        RuntimeError: If no crossing is found on any trajectory.

    """
    rng = np.random.default_rng(1) if rng is None else rng
    lo = np.array([sequence.z_range[0], sequence.h_range[0], sequence.p_range[0]])
    hi = np.array([sequence.z_range[1], sequence.h_range[1], sequence.p_range[1]])

    kept: list[np.ndarray] = []
    for _ in range(num_boundary_paths):
        a, b = lo + rng.random(3) * (hi - lo), lo + rng.random(3) * (hi - lo)
        s = np.linspace(0.0, 1.0, points_per_path)[:, None]
        path = a + s * (b - a)
        label = classify_phase(evaluate(path))
        switches = np.flatnonzero(np.diff(label) != 0)
        for i in switches:
            # Entering or leaving the unknown region is not a phase transition.
            if label[i] == 0 or label[i + 1] == 0:
                continue
            # Concentrate around the crossing rather than sampling uniformly.
            s_local = np.linspace(s[i, 0], s[i + 1, 0], 9)[:, None]
            kept.append(a + s_local * (b - a))

    if not kept:
        raise RuntimeError(
            "no phase transition found on any trajectory; widen the ranges "
            "or increase num_boundary_paths"
        )
    return SampleSet(
        name="boundary",
        points=np.vstack(kept),
        description=(
            "trajectories crossing a phase boundary, sampled densely at the crossing"
        ),
    )


def build_benchmark_samples(path: str | Path) -> SampleSet:
    """States visited by the one-dimensional salt-dissolution benchmark.

    Parameters:
        path: File of states, one ``(z, h, p)`` per line.

    Returns:
        The sample set.

    """
    points = np.loadtxt(path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"expected an (n, 3) array of (z, h, p); got shape {points.shape}"
        )
    return SampleSet(
        name="benchmark",
        points=points,
        description="states visited by the Section 4 benchmark",
    )


class ConvergenceStudy(ConvergencePlotMixin, ABC):
    """Shared configuration, samples and cell classification of the convergence studies."""

    def __init__(
        self,
        sequence: RefinementSequence,
        cli_path: str,
        fields: Sequence[str] | None = None,
        cwd: str | None = None,
        conversion_factors: tuple[float, float, float] = (1.0, 1.0, 1.0),
        translation_factors: tuple[float, float, float] = (0.0, 0.0, 0.0),
        extend_q: bool = True,
        first_order_expansion: bool = False,
    ) -> None:
        self.sequence = sequence
        self.cli_path = cli_path
        self.cwd = cwd
        self.fields = list(fields) if fields else list(FIELD_MAP_CLI)
        self.sample_sets: list[SampleSet] = []
        # The factors map the sample coordinates onto the table coordinates.
        # The sample sets are built in table units (kJ/kg, bar), so the
        # identity is correct here, unlike in the solver, which passes SI units.
        self.conversion_factors = conversion_factors
        self.translation_factors = translation_factors
        self.extend_q = extend_q
        self.first_order_expansion = first_order_expansion

    def evaluate_reference_values(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the correlations directly, in the table conventions.

        Parameters:
            points: Array of shape ``(n, 3)`` holding ``(z, h, p)`` per row.

        Returns:
            Field name to its reference values.

        """
        return evaluate_direct(points, self.fields, self.cli_path, self.cwd)

    def build_sample_sets(self, benchmark_path: str | Path | None = None) -> None:
        """Construct the interior, phase-boundary and benchmark sample sets.

        Parameters:
            benchmark_path: File of benchmark states; the benchmark set is omitted if None.

        """
        self.sample_sets = [
            build_interior_samples(self.sequence, self.evaluate_reference_values),
            build_phase_boundary_samples(self.sequence, self.evaluate_reference_values),
        ]
        if benchmark_path is not None:
            self.sample_sets.append(build_benchmark_samples(benchmark_path))
        for s in self.sample_sets:
            print(f"  {s.name:10s} {len(s):6d} points  --  {s.description}")

    def open_table(self, path: str | Path) -> TabulatedPropertyEvaluator:
        """Open the table at one level with the study's sampler settings.

        Parameters:
            path: Generated VTK table.

        Returns:
            The table evaluator.

        """
        return TabulatedPropertyEvaluator(
            path,
            conversion_factors=self.conversion_factors,
            translation_factors=self.translation_factors,
            extend_q=self.extend_q,
            first_order_expansion=self.first_order_expansion,
        )

    def classify_sample_cells(
        self, table: TabulatedPropertyEvaluator, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify the table cell around each point: one phase region, and near a NaN repair.

        Parameters:
            table: The table at one level.
            points: Sample points, before conversion.

        Returns:
            The pair ``(single_region, near_repair)`` of boolean arrays.

        Raises:
            ValueError: If the table lacks ``PhaseRegion`` or ``mask_nan``.

        """
        missing = [
            name for name in ("PhaseRegion", "mask_nan") if not table.has_array(name)
        ]
        if missing:
            raise ValueError(
                f"{table.path.name} has no {' or '.join(missing)} array; "
                "regenerate it with run_experiment_a.py --stage generate"
            )
        regions = table.cell_corners(points, "PhaseRegion")
        single_region = np.all(regions == regions[:, :1], axis=1)
        near_repair = np.any(table.cell_corners(points, "mask_nan") > 0.0, axis=1)
        return single_region, near_repair

    def categorize_samples(
        self,
        sample_set: SampleSet,
        cell_states: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Split a sample set into the point categories.

        Parameters:
            sample_set: The sample set.
            cell_states: Output of :meth:`classify_sample_cells` at each level the categories are fixed on.

        Returns:
            Mask per category, keyed by :data:`ALL`, :data:`CLEAN` and :data:`NEAR_REPAIR`.

        """
        near_repair = np.logical_or.reduce([near for _, near in cell_states])
        clean = ~near_repair
        if sample_set.single_phase_cells:
            clean &= np.logical_and.reduce([single for single, _ in cell_states])
        return {
            ALL: np.ones(len(sample_set), dtype=bool),
            CLEAN: clean,
            NEAR_REPAIR: near_repair,
        }

    def report(self, results: ConvergenceResult) -> None:
        """Print error and order tables using the shared result format.

        Parameters:
            results: The refinement results.

        """
        report_convergence(results, self.sample_sets)

    @abstractmethod
    def run(self, table_paths: dict[int, str | Path]) -> ConvergenceResult:
        """Evaluate a refinement sequence against the study's reference.

        Parameters:
            table_paths: Table path per refinement level.

        Returns:
            The refinement results.

        """


class FunctionConvergenceStudy(ConvergenceStudy):
    """Compare interpolated properties with direct EOS reference values."""

    def run(self, table_paths: dict[int, str | Path]) -> ConvergenceResult:
        """Compare interpolated against direct values at every resolution.

        Parameters:
            table_paths: Table path per refinement level.

        Returns:
            One :class:`RefinementResult` per level, coarse to fine.

        Raises:
            RuntimeError: If :meth:`build_sample_sets` has not been called.
            ValueError: If no tables are supplied.

        """
        if not self.sample_sets:
            raise RuntimeError("call build_sample_sets() first")

        # The reference is evaluated once: the sample sets are fixed across
        # resolutions, so the direct values are too.
        reference = {
            s.name: self.evaluate_reference_values(s.points) for s in self.sample_sets
        }

        levels = sorted(table_paths)
        if not levels:
            raise ValueError("No tables supplied for the convergence study.")
        # Each is keyed by level, then by sample set.
        sampled: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        cell_states: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        repairs: dict[int, dict[str, dict[str, float]]] = {}
        oob: dict[int, dict[str, float]] = {}
        for level in levels:
            table = self.open_table(table_paths[level])
            sampled[level], cell_states[level] = {}, {}
            repairs[level], oob[level] = {}, {}
            for s in self.sample_sets:
                # One sample call per set: sample_at replaces the sampler's
                # cached cloud, so values, masks and the out-of-bounds flags
                # must all be read from the same pass.
                approx, _, outside = table.sample(s.points, self.fields)
                repairs[level][s.name] = {
                    m: float((approx[m] > 0.0).mean())
                    for m in REPAIR_MASKS
                    if m in approx
                }
                oob[level][s.name] = float(outside.mean())
                sampled[level][s.name] = approx
                cell_states[level][s.name] = self.classify_sample_cells(table, s.points)

        # The categories are fixed once, on the two finest levels, and used at
        # every level, so that all errors of a category are measured on the
        # same points and the order between two levels follows from their
        # errors alone.
        finest = levels[-2:]
        subsets = {
            s.name: self.categorize_samples(
                s, [cell_states[level][s.name] for level in finest]
            )
            for s in self.sample_sets
        }

        results: list[RefinementResult] = []
        for level in levels:
            errors = {
                s.name: {
                    category: {
                        f: variation_normalized_l2_error(
                            sampled[level][s.name][f][subset],
                            reference[s.name][f][subset],
                        )
                        for f in self.fields
                    }
                    for category, subset in subsets[s.name].items()
                }
                for s in self.sample_sets
            }
            results.append(
                RefinementResult(
                    level=level,
                    spacing=self.sequence.spacing_at(level),
                    node_count=self.sequence.node_count(level),
                    errors=errors,
                    n_points={
                        name: {
                            category: int(subset.sum())
                            for category, subset in by_category.items()
                        }
                        for name, by_category in subsets.items()
                    },
                    repairs=repairs[level],
                    out_of_bounds=oob[level],
                )
            )
        return ConvergenceResult(
            refinements=results,
            fields=tuple(self.fields),
            normalization=(
                "Variation-normalised L2 error (absolute L2 for constant references)"
            ),
            category_levels=tuple(finest),
        )
