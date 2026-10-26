"""Experiment A: interpolated fluid property values against direct evaluation.

Run in three stages:

    python run_experiment_a.py --stage check      # check the swEOS batch interface
    python run_experiment_a.py --stage generate   # build the tables
    python run_experiment_a.py --stage compare    # measure the interpolation errors

Settings are in ``convergence_config.py``.

"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_config import (
    BENCHMARK_STATES,
    CLI,
    CWD,
    EXTEND_ABSENT_PHASES,
    FIELDS,
    SEQUENCE,
    STUDY_SETTINGS,
    TABLE_DIR,
    benchmark_states_path,
    table_path,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_config import (
    FUNCTION_FIGURE_DIR as FIGURE_DIR,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.interpolation_convergence import (
    CLEAN,
    PHASE_FIELDS,
    SATURATION_FIELDS,
    VISCOSITY_FIELDS,
    FunctionConvergenceStudy,
    TableLattice,
    evaluate_direct,
)

CLIPPED_FIELDS = (*SATURATION_FIELDS, "Xv", "Xl")
"""tuple: Fractions that the post-processing clips to [0, 1]."""

EXPECTED_ORDERS = """
  Expected orders: about 2 on the bulk set, where the data are smooth, and
  below 2 (about 1) on the boundary set, where a cell spans a phase boundary.
  If the two agree, either the boundary samples do not cross a transition, or
  the errors have reached the accuracy of the direct evaluation."""
"""str: Expected orders, printed after each report of stage 3."""


def stage_check() -> None:
    """Compare batch and single-point swEOS output at a few states, to check the CLI interface."""
    print("Stage 1: checking the CLI batch interface\n")

    probe = np.array(
        [
            [0.032, 438.0, 316.0],  # the single-point example from the CLI docs
            [0.100, 800.0, 100.0],
            [0.500, 1500.0, 120.0],
        ]
    )

    print("  single-point invocations, for comparison:")
    for z, h, p in probe:
        cmd = f"{CLI} -D 0 -V PXH -H {h} -P {p} -X {z}"
        print(f"    {cmd}")
        result = subprocess.run(cmd.split(), cwd=CWD, capture_output=True, text=True)
        lines = [line for line in result.stdout.splitlines() if "=" in line]
        for line in lines[:4]:
            print(f"      {line}")

    print("\n  batch invocation on the same three points:")
    values = evaluate_direct(probe, FIELDS, CLI, CWD)
    print(f"    {'field':8s} " + "  ".join(f"{i:>12d}" for i in range(len(probe))))
    for name, v in values.items():
        print(f"    {name:8s} " + "  ".join(f"{x:12.5g}" for x in v))


def stage_generate() -> None:
    """Build one table per resolution."""
    print("Stage 2: generating tables\n")
    TABLE_DIR.mkdir(exist_ok=True)

    for level in range(SEQUENCE.num_levels):
        eta = SEQUENCE.spacing_at(level)
        nodes = SEQUENCE.node_count(level)
        out = table_path(level)
        print(
            f"  level {level}: eta=({eta[0]:.4g}, {eta[1]:.4g}, {eta[2]:.4g})  "
            f"nodes={nodes:,}"
        )
        command = f"{CLI} -D 3 -V XHP -R {SEQUENCE.cli_range_string(level)} -O {out}"
        t0 = time.time()
        result = subprocess.run(
            command.split(),
            cwd=CWD,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=1800,  # 30 min; the finest level is the slow one
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"level {level} failed (rc={result.returncode}).\n"
                f"cmd: {' '.join(command.split())}\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        if not out.exists():
            raise RuntimeError(
                f"level {level} returned 0 but wrote no {out.name}.\n"
                f"cmd: {' '.join(command.split())}\n"
                f"stdout: {result.stdout[:500]}"
            )
        # The CLI also writes a ParaView script next to the table; it is unused.
        Path(f"{out}.py").unlink(missing_ok=True)

        print(f"    generated in {elapsed:.1f} s")

        # The post-processing of the production pipeline is applied here too
        # (NaN extrapolation, saturation clipping, absent-phase extension,
        # floors, then finite-differenced gradients), because the solver reads
        # a repaired table and this study must measure the closure the solver
        # actually evaluates. The NaN and clipping masks are recorded on the
        # raw table, before those repairs remove what they mark.
        raw_nodes, masks = record_repair_masks(out)
        extrapolate_nan_regions(out, FIELDS, out)
        postprocess_with_masks(
            out, raw_nodes, masks, extend_absent_phases=EXTEND_ABSENT_PHASES
        )


def extrapolate_nan_regions(
    vtk_path: str | Path,
    requested_fields: list[str],
    output_path: str | Path,
) -> None:
    """Fill the NaNs of a raw table from the nearest valid node, and rewrite it so pyvista can read it.

    Parameters:
        vtk_path: Table as written by the CLI.
        requested_fields: Fields to fill; the others are copied unchanged.
        output_path: Where the result is written, as a binary structured grid.

    Raises:
        ValueError: If the nodes do not form a tensor-product lattice.

    """
    mesh = meshio.read(str(vtk_path))
    nodes = np.asarray(mesh.points, dtype=float)
    lattice = TableLattice.of(mesh)
    values = {
        name: np.asarray(array).reshape(len(nodes), -1)[:, 0]
        for name, array in mesh.point_data.items()
    }

    filled: dict[str, int] = {}
    for field in requested_fields:
        if field not in values:
            print(f"    NaN fill: field {field!r} not in table, skipped")
            continue
        missing = np.isnan(values[field])
        if not missing.any():
            continue
        # Five neighbours are requested and the first is taken. Asking for one
        # breaks a tie between two equidistant nodes the other way, which would
        # change the tables generated so far.
        donor = cKDTree(nodes[~missing]).query(nodes[missing], k=5)[1][:, 0]
        values[field][missing] = values[field][~missing][donor]
        filled[field] = int(missing.sum())
    if filled:
        print("    NaN fill: " + ", ".join(f"{name} {n}" for name, n in filled.items()))
    else:
        print("    NaN fill: no NaNs, values left unchanged")

    # The nodes are placed by their lattice index, so nothing here depends on
    # the order in which the CLI wrote them. The axes are written back in the
    # precision the CLI used, which for the finest table saves 16 MB over the
    # double precision the lattice works in.
    axes = [np.asarray(axis, dtype=mesh.points.dtype) for axis in lattice.axes]
    grid = pv.StructuredGrid(*np.meshgrid(*axes, indexing="ij"))
    nx, ny, _ = lattice.shape
    i, j, k = lattice.index
    flat = i + nx * (j + ny * k)  # the grid stores its nodes with z fastest
    for name, array in values.items():
        placed = np.empty(len(nodes), dtype=array.dtype)
        placed[flat] = array
        grid.point_data[name] = placed
    grid.save(str(output_path), binary=True)


def record_repair_masks(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mark, on the raw table, the NaN nodes (``mask_nan``) and the fractions outside [0, 1] (``mask_clip``).

    Parameters:
        path: Table as written by the CLI.

    Returns:
        The node coordinates and the masks, both in the node order of the raw file.

    """
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points, dtype=float)
    n = len(nodes)
    data = {
        name: np.asarray(values, dtype=float).reshape(n, -1)[:, 0]
        for name, values in mesh.point_data.items()
    }

    mask_nan = np.zeros(n, dtype=bool)
    mask_clip = np.zeros(n, dtype=bool)

    for array in FIELDS:
        if array in data:
            mask_nan |= np.isnan(data[array])

    for array in CLIPPED_FIELDS:
        if array in data:
            d = data[array]
            mask_clip |= (d < -1e-12) | (d > 1.0 + 1e-12)

    print(f"    mask_nan  : {mask_nan.sum():8d} / {n} ({100 * mask_nan.mean():5.2f}%)")
    print(
        f"    mask_clip : {mask_clip.sum():8d} / {n} ({100 * mask_clip.mean():5.2f}%)"
    )

    return nodes, {
        "mask_nan": mask_nan,
        "mask_clip": mask_clip,
    }


def extend_absent_phase_properties(grid: pv.DataSet) -> np.ndarray:
    """Fill each phase property, where its phase is absent, from the nearest valid node.

    Parameters:
        grid: Table on a tensor-product lattice, modified in place.

    Returns:
        Boolean mask of the nodes where at least one property was filled.

    Raises:
        ValueError: If the nodes do not form a tensor-product lattice.

    """
    lattice = TableLattice.of(grid)
    filled = np.zeros(grid.n_points, dtype=bool)
    for saturation, properties in PHASE_FIELDS.items():
        if saturation not in grid.array_names:
            continue
        present = lattice.to_lattice(np.asarray(grid[saturation]) > 0.0)
        phase_donor = None

        for name in properties:
            if name not in grid.array_names:
                continue
            values = np.asarray(grid[name], dtype=float).copy()
            valid = present
            if name in VISCOSITY_FIELDS:
                valid = present & lattice.to_lattice(values > 0.0)
            if valid.all() or not valid.any():
                continue

            if valid is present:
                if phase_donor is None:
                    phase_donor = lattice.nearest(present)
                donor = phase_donor
            else:
                donor = lattice.nearest(valid)

            missing = ~valid[lattice.index]
            values[missing] = lattice.to_lattice(values)[donor][missing]
            grid.point_data[name] = values
            filled |= missing
    return filled


def postprocess_with_masks(
    path: Path,
    raw_nodes: np.ndarray,
    masks: dict[str, np.ndarray],
    extend_absent_phases: bool = True,
) -> None:
    """Attach the repair masks and apply the remaining repairs: clipping, absent-phase fill, floors and gradients.

    Parameters:
        path: Table to post-process, modified in place.
        raw_nodes: Node coordinates of the raw table.
        masks: Masks in the node order of ``raw_nodes``.
        extend_absent_phases: Whether to fill the properties of absent phases.

    Raises:
        ValueError: If the repaired table does not have the raw table's nodes.

    """
    grid = pv.read(str(path))
    nodes = np.asarray(grid.points, dtype=float)
    order_raw = np.lexsort(raw_nodes.T[::-1])
    order_grid = np.lexsort(nodes.T[::-1])
    if len(nodes) != len(raw_nodes) or not np.allclose(
        raw_nodes[order_raw], nodes[order_grid]
    ):
        raise ValueError(f"{path.name}: repaired table nodes differ from the raw table")
    for name, mask in masks.items():
        aligned = np.empty(len(nodes))
        aligned[order_grid] = mask[order_raw].astype(float)
        grid.point_data[name] = aligned

    for array in CLIPPED_FIELDS:
        if array in grid.array_names:
            grid[array] = np.clip(grid[array], 0.0, 1.0)

    if extend_absent_phases:
        mask_extend = extend_absent_phase_properties(grid)
    else:
        mask_extend = np.zeros(len(nodes), dtype=bool)
    grid.point_data["mask_extend"] = mask_extend.astype(float)

    mask_floor = np.zeros(len(nodes), dtype=bool)
    for array, floor in (
        ("mu_l", 2.0e-5),
        ("mu_v", 2.0e-5),
        ("Rho_l", 1.0e-8),
        ("Rho_v", 1.0e-8),
    ):
        if array in grid.array_names:
            d = np.asarray(grid[array], dtype=float).copy()
            zero = d == 0.0
            d[zero] = floor
            grid[array] = d
            mask_floor |= zero
    grid.point_data["mask_floor"] = mask_floor.astype(float)

    n = len(nodes)
    note = "" if extend_absent_phases else "  (extension off)"
    print(
        f"    mask_extend: {mask_extend.sum():8d} / {n} "
        f"({100 * mask_extend.mean():5.2f}%){note}"
    )
    print(
        f"    floored    : {mask_floor.sum():8d} / {n} ({100 * mask_floor.mean():5.2f}%)"
    )

    for array in FIELDS:
        if array in grid.array_names:
            grad = grid.compute_derivative(array, gradient=True)["gradient"]
            grid.point_data.set_vectors(grad, "grad_" + array)

    grid.save(str(path), binary=True)


def stage_compare() -> None:
    """Measure the interpolation error at every resolution, without and, if needed, with extrapolation."""
    print("Stage 3: comparing interpolated against direct evaluation\n")

    tables = {
        level: table_path(level)
        for level in range(SEQUENCE.num_levels)
        if table_path(level).exists()
    }
    if not tables:
        raise RuntimeError(f"no tables in {TABLE_DIR}; run --stage generate first")
    print(f"  found {len(tables)} tables: levels {sorted(tables)}\n")

    experiment = FunctionConvergenceStudy(**STUDY_SETTINGS)

    print("  building sample sets (this calls the CLI to locate the phase boundaries)")
    benchmark = benchmark_states_path()
    if benchmark is None:
        print(f"  NOTE: {BENCHMARK_STATES} not found; benchmark set omitted")
    experiment.build_sample_sets(benchmark_path=benchmark)

    for extend_q in (False, True):
        label = "with extrapolation" if extend_q else "interpolation only"
        print(f"\n\n{'#' * 76}\n#  {label}  (extend_q={extend_q})\n{'#' * 76}")
        experiment.extend_q = extend_q
        results = experiment.run(tables)
        experiment.report(results)
        print(EXPECTED_ORDERS)
        experiment.plot_errors(
            results, category=CLEAN, output_dir=FIGURE_DIR, tag=f"extend_q_{extend_q}"
        )
        print(f"\n  figures saved to {FIGURE_DIR}")

        if not any(
            fraction > 0.0
            for result in results
            for fraction in result.out_of_bounds.values()
        ):
            print(
                "\n  Every sample lies inside every table, so extend_q=True would "
                "give the same results; skipped."
            )
            break


def main() -> None:
    """Run the stage selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("check", "generate", "compare"), required=True
    )
    args = parser.parse_args()
    {"check": stage_check, "generate": stage_generate, "compare": stage_compare}[
        args.stage
    ]()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--stage", "compare"])
    main()
