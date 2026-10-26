"""Experiment C: what the tabulated closure costs.

Run in stages; each reads and writes ``experiment_c/cost.json``:

    python run_experiment_c.py --stage evaluate   # per-evaluation cost, direct and tabulated
    python run_experiment_c.py --stage tables     # generation time and size of each table
    python run_experiment_c.py --stage simulate   # one benchmark run per table
    python run_experiment_c.py --stage errors     # phase-boundary errors from Experiment A
    python run_experiment_c.py --stage figure     # cost against error
    python run_experiment_c.py --stage all        # all of the above, in order

Settings are in ``cost_config.py``.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from porepy.examples.geothermal_flow.obl_convergence_analysis import (
    cost_config as config,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_config import (
    CLI,
    CONVERSION_FACTORS,
    CWD,
    FIELDS,
    SEQUENCE,
    STUDY_SETTINGS,
    TRANSLATION_FACTORS,
    table_path,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.cost_experiment import (
    CostResult,
    EvaluationComparison,
    EvaluationCost,
    SimulationCost,
    TableCost,
    is_cache_valid,
    load_cost_result,
    load_generation_costs,
    run_benchmark,
    save_cost_result,
    save_generation_costs,
    solution_difference,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.cost_plotting import (
    CostFigure,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.interpolation_convergence import (
    CLEAN,
    FunctionConvergenceStudy,
    TabulatedPropertyEvaluator,
    build_phase_boundary_samples,
    evaluate_direct,
)

COST_FILE = config.COST_DIR / "cost.json"
"""Path: Where every stage reads and writes its measurements."""

VTK_HEADER_BYTES = 2048
"""int: Bytes read from a table header to find its grid dimensions."""


def available_levels() -> list[int]:
    """Refinement levels whose table exists.

    Returns:
        The levels, coarsest first.

    """
    return [level for level in config.LEVELS if table_path(level).exists()]


def node_count_of(path: Path) -> int:
    """Node count of a table, read from its header.

    Parameters:
        path: The table.

    Returns:
        The number of nodes.

    Raises:
        ValueError: If the header carries no grid dimensions.

    """
    with path.open("rb") as stream:
        header = stream.read(VTK_HEADER_BYTES).decode("latin-1")
    for line in header.splitlines():
        if line.upper().startswith("DIMENSIONS"):
            dimensions = [int(value) for value in line.split()[1:4]]
            return int(np.prod(dimensions))
    raise ValueError(f"{path.name}: no DIMENSIONS line in the first bytes")


# -----------------------------------------------------------------------------
# Stage 1: per-evaluation cost.
# -----------------------------------------------------------------------------
def stage_evaluate() -> None:
    """Time direct evaluation against table lookup at the same phase-boundary states."""
    print("Stage 1: per-evaluation cost\n")
    points = boundary_states()
    print(f"  {len(points)} states, from {config.BOUNDARY_STATES.name}\n")

    direct = EvaluationCost.measure(
        "direct CLI",
        lambda batch: evaluate_direct(batch, FIELDS, CLI, CWD),
        points,
        config.EVALUATION_BATCH_SIZES,
        config.EVALUATION_REPEATS,
    )

    lookups = {level: measure_lookup(level, points) for level in available_levels()}

    comparison = EvaluationComparison(
        state_count=len(points), direct=direct, lookups=lookups
    )
    
    print("\n  speedup of one lookup over one direct evaluation")
    for level in lookups:
        print(f"    L{level}: {comparison.speedup(level):10.1f}x")

    result = load_cost_result(COST_FILE)
    result.evaluation = comparison
    save_cost_result(COST_FILE, result)
    print(f"\n  written to {COST_FILE}")


def measure_lookup(level: int, points: np.ndarray) -> EvaluationCost:
    """Time lookups in one table; loading the table is not timed.

    Parameters:
        level: Refinement level of the table.
        points: States to look up.

    Returns:
        The measured cost.

    """
    table = TabulatedPropertyEvaluator(
        table_path(level),
        conversion_factors=CONVERSION_FACTORS,
        translation_factors=TRANSLATION_FACTORS,
        extend_q=False,
    )
    return EvaluationCost.measure(
        f"lookup L{level}",
        lambda batch: table.sample(batch, FIELDS),
        points,
        config.EVALUATION_BATCH_SIZES,
        config.EVALUATION_REPEATS,
    )


def boundary_states() -> np.ndarray:
    """The phase-boundary states, built once and cached.

    Returns:
        Array of shape ``(n, 3)`` holding ``(z, h, p)`` per row.

    """
    if config.BOUNDARY_STATES.exists():
        return np.loadtxt(config.BOUNDARY_STATES)
    print("  locating the phase boundaries with the CLI (done once)")
    samples = build_phase_boundary_samples(
        SEQUENCE, lambda batch: evaluate_direct(batch, FIELDS, CLI, CWD)
    )
    config.BOUNDARY_STATES.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(config.BOUNDARY_STATES, samples.points)
    return samples.points


# -----------------------------------------------------------------------------
# Stage 2: offline cost of the tables.
# -----------------------------------------------------------------------------
def stage_tables(regenerate: bool = True, keep: bool = False) -> None:
    """Record the generation time, size and node count of each table.

    Parameters:
        regenerate: Whether a missing or stale cached time may be measured by rebuilding the table.
        keep: Whether to keep the tables rebuilt for the timing.

    """
    print("Stage 2: offline cost of the tables\n")
    cached = load_generation_costs(config.GENERATION_COST_FILE)
    result = load_cost_result(COST_FILE)
    costs: dict[int, TableCost] = {}

    for level in available_levels():
        path = table_path(level)
        nodes, size = node_count_of(path), path.stat().st_size
        entry = cached.get(level)
        if entry is not None and is_cache_valid(entry, nodes, size):
            costs[level] = entry
            print(
                f"  L{level}: {nodes:,} nodes, {size / 1e6:8.1f} MB, "
                f"{entry.generation_seconds:7.1f} s (cached)"
            )
            continue
        if not regenerate:
            costs[level] = TableCost(
                level=level,
                spacing=SEQUENCE.spacing_at(level),
                node_count=nodes,
                file_bytes=size,
                source="unavailable",
            )
            print(f"  L{level}: {nodes:,} nodes, {size / 1e6:8.1f} MB, time unknown")
            continue
        costs[level] = measure_generation(level, keep=keep)

    result.tables = costs
    save_cost_result(COST_FILE, result)
    save_generation_costs(config.GENERATION_COST_FILE, costs)
    report_scaling(result)
    print(f"\n  written to {COST_FILE} and {config.GENERATION_COST_FILE}")


def measure_generation(level: int, keep: bool) -> TableCost:
    """Build one table again, timing the CLI and the post-processing separately.

    Parameters:
        level: Refinement level to build.
        keep: Whether to keep the table afterwards.

    Returns:
        The measured cost.

    Raises:
        RuntimeError: If the CLI fails or writes nothing.

    """
    import subprocess
    import time

    from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_config import (  # noqa: E501
        EXTEND_ABSENT_PHASES,
    )
    from porepy.examples.geothermal_flow.obl_convergence_analysis.run_experiment_a import (  # noqa: E501
        extrapolate_nan_regions,
        postprocess_with_masks,
        record_repair_masks,
    )

    config.TIMING_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TIMING_TABLE_DIR / f"XHP_level_{level}.vtk"
    command = f"{CLI} -D 3 -V XHP -R {SEQUENCE.cli_range_string(level)} -O {out}"
    print(f"  L{level}: timing generation")

    start = time.perf_counter()
    completed = subprocess.run(
        command.split(),
        cwd=CWD,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    cli_seconds = time.perf_counter() - start
    if completed.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"level {level} generation failed (rc={completed.returncode}): "
            f"{completed.stdout[:500]}"
        )
    Path(f"{out}.py").unlink(missing_ok=True)

    start = time.perf_counter()
    raw_nodes, masks = record_repair_masks(out)
    extrapolate_nan_regions(out, FIELDS, out)
    postprocess_with_masks(
        out, raw_nodes, masks, extend_absent_phases=EXTEND_ABSENT_PHASES
    )
    postprocess_seconds = time.perf_counter() - start

    cost = TableCost(
        level=level,
        spacing=SEQUENCE.spacing_at(level),
        node_count=node_count_of(out),
        file_bytes=out.stat().st_size,
        cli_seconds=cli_seconds,
        postprocess_seconds=postprocess_seconds,
        source="measured",
    )
    print(
        f"    CLI {cli_seconds:7.1f} s, post-processing {postprocess_seconds:7.1f} s, "
        f"{cost.file_bytes / 1e6:.1f} MB"
    )
    if not keep:
        out.unlink(missing_ok=True)
    return cost


def report_scaling(result: CostResult) -> None:
    """Print whether the offline quantities grow as the spacing predicts.

    Parameters:
        result: The measurements.

    """
    checks = result.scaling_checks()
    if not checks:
        return
    print("\n  growth under refinement")
    for check in checks:
        print(f"    {check}")


# -----------------------------------------------------------------------------
# Stage 3: the benchmark, once per resolution.
# -----------------------------------------------------------------------------
def stage_simulate() -> None:
    """Run the benchmark :data:`cost_config.SIMULATION_REPEATS` times per table; failed runs are recorded."""
    print("Stage 3: benchmark cost per resolution\n")
    end_time = config.END_TIME_YEARS
    print(
        "  end time: "
        + (
            "as configured in benchmark.yaml"
            if end_time is None
            else f"{end_time:g} years (overridden for every resolution alike)"
        )
    )
    if config.MODEL_OVERRIDES:
        print("  model overrides: ", end="")
        print(", ".join(f"{k}={v}" for k, v in config.MODEL_OVERRIDES.items()))
    if config.TIME_OVERRIDES:
        print("  time overrides, applied to every resolution: ", end="")
        print(", ".join(f"{k}={v:g}" for k, v in config.TIME_OVERRIDES.items()))
    result = load_cost_result(COST_FILE)
    solutions: dict[int, dict[str, np.ndarray]] = {}

    for level in available_levels():
        runs = []
        for repeat in range(config.SIMULATION_REPEATS):
            print(f"\n  L{level}, run {repeat + 1} of {config.SIMULATION_REPEATS}")
            run, solution = run_benchmark(
                level=level,
                table_path=table_path(level),
                config_path=config.BENCHMARK_CONFIG,
                end_time_years=end_time,
                seconds_per_year=config.SECONDS_PER_YEAR,
                time_overrides=config.TIME_OVERRIDES,
                model_overrides=config.MODEL_OVERRIDES,
            )
            runs.append(run)
            if run.completed:
                solutions[level] = solution
            report_run(run)
        result.simulations[level] = SimulationCost(level=level, runs=tuple(runs))

    finest = max(solutions) if solutions else None
    if finest is not None:
        for level, solution in solutions.items():
            if level == finest:
                continue
            result.simulations[level].solution_error = solution_difference(
                solution, solutions[finest]
            )
        print(f"\n  solution differences taken against L{finest}")
    save_cost_result(COST_FILE, result)
    report_simulations(result)
    print(f"\n  written to {COST_FILE}")


def report_run(run) -> None:
    """Print what one benchmark run cost.

    Parameters:
        run: The run.

    """
    split = run.split
    status = "completed" if run.completed else f"INCOMPLETE ({run.message})"
    print(f"    {status}")
    print(
        f"    wall-clock {split.total_seconds:8.1f} s"
        f"  (load {run.table_load_seconds:.1f} s, prepare {run.prepare_seconds:.1f} s,"
        f" both excluded)"
    )
    print(
        f"    split: thermodynamics {split.thermodynamics_seconds:8.1f} s"
        f" ({split.thermodynamics_fraction:5.1%}),"
        f" assembly {split.assembly_seconds:7.1f} s,"
        f" linear solve {split.linear_solve_seconds:7.1f} s,"
        f" other {split.other_seconds:7.1f} s"
    )
    print(
        f"    rejected attempts cost {run.rejected_seconds:7.1f} s "
        f"({run.rejected_fraction:5.1%} of the run), "
        f"{run.rejected_iterations} iterations, "
        f"{run.rejected_evaluations:,} evaluations"
    )
    print(
        f"    steps {run.time_steps:4d}, rejected {run.rejected_steps:3d}, "
        f"iterations mean {run.mean_iterations:5.1f} max {run.max_iterations:3d}, "
        f"closure evaluations {run.closure_evaluations:,} "
        f"in {run.closure_calls:,} calls"
    )


def report_simulations(result: CostResult) -> None:
    """Print the per-resolution summary and the projected direct-evaluation cost.

    Parameters:
        result: The measurements.

    """
    print("\n  summary")
    print(
        f"  {'level':>5s} {'wall-clock':>22s} {'thermo':>8s} {'rejected':>9s} "
        f"{'steps':>6s} {'failed':>7s} {'mean it.':>9s} {'evaluations':>14s}"
    )
    for level, cost in sorted(result.simulations.items()):
        run = cost.representative
        if run is None:
            print(f"  {level:5d}  no completed run")
            continue
        print(
            f"  {level:5d} {str(cost.total_seconds):>22s} "
            f"{run.split.thermodynamics_fraction:7.1%} "
            f"{run.rejected_fraction:8.1%} {run.time_steps:6d} "
            f"{run.rejected_steps:7d} {run.mean_iterations:9.1f} "
            f"{run.closure_evaluations:14,d}"
        )
        if cost.solution_error:
            worst = max(cost.solution_error.items(), key=lambda item: item[1][0])
            print(
                f"        solution difference against the finest run: "
                f"worst variable {worst[0]} at {worst[1][0]:.3e}"
            )

    if result.evaluation is None:
        print("\n  per-evaluation cost not measured; run --stage evaluate for the")
        print("  projected cost of the same run without a table")
        return
    for level in sorted(result.simulations):
        projected = result.projected_direct_seconds(level)
        measured = result.simulations[level].representative
        if projected is None or measured is None:
            continue
        print(
            f"\n  L{level}: the same run with every lookup evaluated directly "
            f"would take about {projected:,.0f} s"
        )
        print(
            f"        against {measured.split.total_seconds:,.0f} s measured, "
            f"a factor of {projected / measured.split.total_seconds:,.0f}"
        )
        print(
            "        projected from the fit, not measured: "
            f"{result.evaluation.direct.model}"
        )


# -----------------------------------------------------------------------------
# Stage 4: the accuracy the cost buys.
# -----------------------------------------------------------------------------
def stage_errors() -> None:
    """Record the largest phase-boundary error over :data:`cost_config.ERROR_FIELDS` per table."""
    print("Stage 4: accuracy from Experiment A\n")
    levels = available_levels()
    tables = {level: table_path(level) for level in levels}
    conv_study = FunctionConvergenceStudy(**STUDY_SETTINGS, extend_q=False)
    print("  building sample sets (this calls the CLI)")
    conv_study.build_sample_sets()
    results = conv_study.run(tables)

    errors: dict[int, float] = {}
    for refinement in results:
        # by_field = refinement.errors["boundary"][CLEAN]
        # worst_field = max(by_field, key=lambda name: by_field[name][0])
        by_field ={
            name: value
            for name, value in refinement.errors["boundary"][CLEAN].items()
            if name in config.ERROR_FIELDS
        }
        worst_field = max(by_field, key=lambda name: by_field[name][0])

        errors[refinement.level] = float(by_field[worst_field][0])
        print(
            f"  L{refinement.level}: {errors[refinement.level]:.4e} "
            f"(worst field {worst_field})"
        )

    result = load_cost_result(COST_FILE)
    result.boundary_errors = errors
    save_cost_result(COST_FILE, result)
    print(f"\n  written to {COST_FILE}")


# -----------------------------------------------------------------------------
# Stage 5: the figure.
# -----------------------------------------------------------------------------
def stage_figure() -> None:
    """Draw cost against error, one point per resolution."""

    print("Stage 5: cost against error\n")
    result = load_cost_result(COST_FILE)
    report_scaling(result)
    
    figure = CostFigure().plot_offline_online(result, output_dir=config.COST_FIGURE_DIR)
    if figure is None:
        print("  nothing to draw: no level has both a completed run and an error")
        print("  run --stage simulate and --stage errors first")
        return
    print(f"  figure saved to {config.COST_FIGURE_DIR}")


def main() -> None:
    """Run the stage selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("evaluate", "tables", "simulate", "errors", "figure", "all"),
        required=True,
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="do not rebuild a table to measure its generation time",
    )
    parser.add_argument(
        "--keep-timing-tables",
        action="store_true",
        help="keep the tables built for the generation timing",
    )
    args = parser.parse_args()

    def tables() -> None:
        stage_tables(regenerate=not args.no_regenerate, keep=args.keep_timing_tables)

    stages = {
        "evaluate": stage_evaluate,
        "tables": tables,
        "simulate": stage_simulate,
        "errors": stage_errors,
        "figure": stage_figure,
    }
    if args.stage == "all":
        for name, stage in stages.items():
            print(f"\n{'#' * 76}\n#  {name}\n{'#' * 76}")
            stage()
        return
    stages[args.stage]()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--stage", "figure"])
    main()
