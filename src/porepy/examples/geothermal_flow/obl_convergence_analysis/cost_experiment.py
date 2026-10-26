"""Experiment C: computational cost of the tabulated closure.

Measures the per-evaluation cost, direct against tabulated (C1), and the cost of
the one-dimensional benchmark per table resolution (C2). The cost of a simulation
with direct evaluation is projected from the two rather than run.

"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Sequence

import numpy as np

from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_metrics import (
    variation_normalized_l2_error,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler


@dataclass(frozen=True)
class RepeatedTiming:
    """Wall-clock of one measurement, repeated; ``seconds`` holds one time per repeat."""

    seconds: tuple[float, ...]

    @property
    def median_seconds(self) -> float:
        """Median over the repeats, the number quoted elsewhere."""
        return float(np.median(self.seconds))

    @property
    def minimum_seconds(self) -> float:
        """Fastest repeat."""
        return float(np.min(self.seconds))

    @property
    def maximum_seconds(self) -> float:
        """Slowest repeat."""
        return float(np.max(self.seconds))

    @property
    def spread_seconds(self) -> float:
        """Slowest repeat minus the fastest."""
        return self.maximum_seconds - self.minimum_seconds

    @property
    def relative_spread(self) -> float:
        """Spread as a fraction of the median, or NaN for a zero median."""
        median = self.median_seconds
        return self.spread_seconds / median if median > 0.0 else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.median_seconds:.4g} s "
            f"(+/- {self.spread_seconds:.2g} over {len(self.seconds)} runs)"
        )


def measure_repeated(action: Callable[[], Any], repeats: int) -> RepeatedTiming:
    """Time an action several times.

    Parameters:
        action: Called once per repeat; its return value is discarded.
        repeats: Number of repeats, at least one.

    Returns:
        The timing.

    Raises:
        ValueError: If fewer than one repeat is requested.

    """
    if repeats < 1:
        raise ValueError(f"need at least one repeat, got {repeats}")
    seconds: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        seconds.append(time.perf_counter() - start)
    return RepeatedTiming(tuple(seconds))


@dataclass(frozen=True)
class LinearCostModel:
    """Cost of an evaluation call: a fixed part plus a part per evaluation."""

    fixed_seconds: float
    per_evaluation_seconds: float

    def seconds_for(self, evaluations: int) -> float:
        """Projected cost of evaluating a number of states in one call.

        Parameters:
            evaluations: Number of states.

        Returns:
            The projected wall-clock in seconds.

        """
        return self.fixed_seconds + self.per_evaluation_seconds * evaluations

    def __str__(self) -> str:
        return (
            f"{self.fixed_seconds:.4g} s + "
            f"{self.per_evaluation_seconds:.4g} s per evaluation"
        )


def fit_linear_cost(
    evaluations: Sequence[int], seconds: Sequence[float]
) -> LinearCostModel:
    """Fit cost against batch size by least squares.

    Parameters:
        evaluations: Batch sizes, at least two distinct values.
        seconds: Measured wall-clock, one per batch size.

    Returns:
        The fitted model.

    Raises:
        ValueError: If fewer than two distinct batch sizes are given, or the lengths differ.

    """
    sizes = np.asarray(evaluations, dtype=float)
    times = np.asarray(seconds, dtype=float)
    if sizes.shape != times.shape:
        raise ValueError(f"{sizes.size} batch sizes but {times.size} timings")
    if np.unique(sizes).size < 2:
        raise ValueError("need at least two distinct batch sizes to separate the costs")
    slope, intercept = np.polyfit(sizes, times, 1)
    if intercept < 0.0:
        # A call cannot take negative time; a negative intercept is scatter in the
        # timings. Refit with the fixed cost held at zero (least squares through the origin).
        slope, intercept = float(np.dot(sizes, times) / np.dot(sizes, sizes)), 0.0

    return LinearCostModel(
        fixed_seconds=float(intercept), per_evaluation_seconds=float(slope)
    )


@dataclass
class EvaluationCost:
    """Cost of one way of evaluating the closure, with the fit through the median timings."""

    name: str
    timings: dict[int, RepeatedTiming]
    model: LinearCostModel

    @classmethod
    def measure(
        cls,
        name: str,
        evaluate: Callable[[np.ndarray], Any],
        points: np.ndarray,
        batch_sizes: Sequence[int],
        repeats: int,
    ) -> EvaluationCost:
        """Time one evaluation path over several batch sizes.

        Parameters:
            name: What is timed.
            evaluate: Called with an ``(n, 3)`` array of states.
            points: States to draw the batches from.
            batch_sizes: Batch sizes; sizes beyond the number of states are skipped.
            repeats: Timing repeats per batch size.

        Returns:
            The measured cost.

        Raises:
            ValueError: If no batch size fits in the given states.

        """
        points = np.asarray(points, dtype=float)
        usable = [int(n) for n in batch_sizes if n <= len(points)]
        if not usable:
            raise ValueError(
                f"{len(points)} states are fewer than the smallest batch size "
                f"{min(batch_sizes)}"
            )
        timings: dict[int, RepeatedTiming] = {}
        for size in usable:
            batch = points[:size]
            timings[size] = measure_repeated(lambda: evaluate(batch), repeats)
            print(f"    {name}: {size:6d} states  {timings[size]}")
        model = fit_linear_cost(
            list(timings), [t.median_seconds for t in timings.values()]
        )
        print(f"    {name}: {model}")
        return cls(name=name, timings=timings, model=model)


@dataclass
class EvaluationComparison:
    """Direct evaluation against table lookup at the same states, with the lookup cost per level."""

    state_count: int
    direct: EvaluationCost
    lookups: dict[int, EvaluationCost]

    def speedup(self, level: int) -> float:
        """How many times cheaper one lookup is than one direct evaluation.

        Parameters:
            level: Refinement level of the table.

        Returns:
            The ratio of the per-evaluation costs.

        """
        lookup = self.lookups[level].model.per_evaluation_seconds
        return self.direct.model.per_evaluation_seconds / lookup

    def projected_direct_seconds(self, evaluations: int) -> float:
        """Projected cost of a number of direct evaluations, from the fitted model.

        Parameters:
            evaluations: Number of states.

        Returns:
            The projected wall-clock in seconds.

        """
        return self.direct.model.seconds_for(evaluations)


@dataclass
class TableCost:
    """What one table costs to build and store; ``source`` is ``measured``, ``cached`` or ``unavailable``."""

    level: int
    spacing: tuple[float, float, float]
    node_count: int
    file_bytes: int
    cli_seconds: float | None = None
    postprocess_seconds: float | None = None
    source: str = "unavailable"

    @property
    def generation_seconds(self) -> float | None:
        """Total generation time, or None if either part is missing."""
        if self.cli_seconds is None or self.postprocess_seconds is None:
            return None
        return self.cli_seconds + self.postprocess_seconds


@dataclass(frozen=True)
class ScalingCheck:
    """Whether a quantity grows with refinement as expected."""

    name: str
    exponent: float
    expected_exponent: float
    tolerance: float

    @property
    def holds(self) -> bool:
        """Whether the measured exponent is within tolerance."""
        return abs(self.exponent - self.expected_exponent) <= self.tolerance

    def __str__(self) -> str:
        verdict = "as expected" if self.holds else "DEPARTS from the expected scaling"
        return (
            f"{self.name}: exponent {self.exponent:.2f} "
            f"(expected {self.expected_exponent:.2f}, {verdict})"
        )


def check_power_law(
    refinement_factors: Sequence[float],
    values: Sequence[float],
    name: str,
    expected_exponent: float,
    tolerance: float,
) -> ScalingCheck:
    """Fit ``value ~ refinement**exponent`` and compare with the expected exponent.

    Parameters:
        refinement_factors: Refinement factor of each level, one for the coarsest.
        values: The quantity at each level, all strictly positive.
        name: Quantity name, for reporting.
        expected_exponent: The exponent the quantity should follow.
        tolerance: Allowed departure from it.

    Returns:
        The check.

    Raises:
        ValueError: If fewer than two levels are given, or a value is not positive.

    """
    factors = np.asarray(refinement_factors, dtype=float)
    quantities = np.asarray(values, dtype=float)
    if factors.size < 2 or factors.size != quantities.size:
        raise ValueError("need at least two levels, and one value per level")
    if np.any(quantities <= 0.0) or np.any(factors <= 0.0):
        raise ValueError("power-law fit needs strictly positive values")
    exponent = np.polyfit(np.log(factors), np.log(quantities), 1)[0]
    return ScalingCheck(
        name=name,
        exponent=float(exponent),
        expected_exponent=expected_exponent,
        tolerance=tolerance,
    )


def load_generation_costs(path: str | Path) -> dict[int, TableCost]:
    """Read cached generation times.

    Parameters:
        path: The cache file; a missing file is not an error.

    Returns:
        Refinement level to its cached cost.

    """
    path = Path(path)
    if not path.exists():
        return {}
    stored = json.loads(path.read_text())
    costs: dict[int, TableCost] = {}
    for level, entry in stored.items():
        entry = dict(entry)
        entry["spacing"] = tuple(entry["spacing"])
        costs[int(level)] = TableCost(**entry)
    return costs


def save_generation_costs(path: str | Path, costs: dict[int, TableCost]) -> None:
    """Write generation times to the cache.

    Parameters:
        path: The cache file, created with its parent if absent.
        costs: Refinement level to its cost.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {str(level): asdict(cost) for level, cost in sorted(costs.items())},
            indent=2,
        )
    )


def is_cache_valid(cost: TableCost, node_count: int, file_bytes: int) -> bool:
    """Whether a cached cost still describes the table on disk.

    Parameters:
        cost: The cached cost.
        node_count: Nodes in the table now.
        file_bytes: Size of the table now.

    Returns:
        True if the cached entry can be reused.

    """
    return (
        cost.generation_seconds is not None
        and cost.node_count == node_count
        and cost.file_bytes == file_bytes
    )


class TimedSampler(VTKSampler):
    """A sampler that records the time and number of its lookups; the table load is timed separately."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        start = time.perf_counter()
        super().__init__(*args, **kwargs)
        self.load_seconds = time.perf_counter() - start
        self.lookup_seconds = 0.0
        self.calls = 0
        self.evaluations = 0

    def sample_at(self, points: np.ndarray) -> Any:
        """Look up the table, recording the time and the number of states.

        Parameters:
            points: Array of shape ``(n, 3)``.

        Returns:
            The result of the base class.

        """
        start = time.perf_counter()
        result = super().sample_at(points)
        self.lookup_seconds += time.perf_counter() - start
        self.calls += 1
        self.evaluations += int(np.asarray(points).shape[0])
        return result


@dataclass
class RuntimeSplit:
    """Where the wall-clock of a run went; table lookups are excluded from the assembly time."""

    total_seconds: float = 0.0
    thermodynamics_seconds: float = 0.0
    assembly_seconds: float = 0.0
    linear_solve_seconds: float = 0.0

    @property
    def other_seconds(self) -> float:
        """Whatever the three measured parts do not account for."""
        return (
            self.total_seconds
            - self.thermodynamics_seconds
            - self.assembly_seconds
            - self.linear_solve_seconds
        )

    @property
    def thermodynamics_fraction(self) -> float:
        """Share of the run spent in table lookups, NaN for an empty run."""
        if self.total_seconds <= 0.0:
            return float("nan")
        return self.thermodynamics_seconds / self.total_seconds


@dataclass
class SimulationRun:
    """One benchmark run with one table; the ``rejected_*`` fields count discarded attempts, included in ``split.total_seconds``."""

    level: int
    completed: bool
    split: RuntimeSplit
    prepare_seconds: float
    table_load_seconds: float
    closure_evaluations: int
    closure_calls: int
    iterations: tuple[int, ...]
    rejected_steps: int
    final_time_seconds: float
    rejected_seconds: float = 0.0
    rejected_evaluations: int = 0
    rejected_iterations: int = 0
    message: str = ""

    @property
    def rejected_fraction(self) -> float:
        """Share of the run spent on attempts that were thrown away."""
        total = self.split.total_seconds
        return self.rejected_seconds / total if total > 0.0 else float("nan")

    @property
    def time_steps(self) -> int:
        """Number of accepted time steps."""
        return len(self.iterations)

    @property
    def mean_iterations(self) -> float:
        """Mean Newton iterations per accepted step, NaN if there were none."""
        return float(np.mean(self.iterations)) if self.iterations else float("nan")

    @property
    def max_iterations(self) -> int:
        """Largest iteration count of any accepted step, zero if there were none."""
        return int(np.max(self.iterations)) if self.iterations else 0


@dataclass
class SimulationCost:
    """The repeated runs of one table resolution, with the solution difference against the finest run."""

    level: int
    runs: tuple[SimulationRun, ...]
    solution_error: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def completed_runs(self) -> tuple[SimulationRun, ...]:
        """The runs that reached the end time."""
        return tuple(run for run in self.runs if run.completed)

    @property
    def total_seconds(self) -> RepeatedTiming:
        """Wall-clock of the completed runs."""
        return RepeatedTiming(
            tuple(run.split.total_seconds for run in self.completed_runs)
        )

    @property
    def representative(self) -> SimulationRun | None:
        """The completed run whose wall-clock is the median, or None."""
        runs = self.completed_runs
        if not runs:
            return None
        median = float(np.median([r.split.total_seconds for r in runs]))
        return min(runs, key=lambda r: abs(r.split.total_seconds - median))


class CostInstrumentation:
    """Mixin recording assembly, solve, iteration and rejection counts; place it ahead of the model class."""

    def assemble_linear_system(self) -> None:
        """Assemble, and record the time that is not a table lookup."""
        record = self._cost_record
        lookups_before = record.lookup_seconds()
        start = time.perf_counter()
        super().assemble_linear_system()  # type: ignore[misc]
        elapsed = time.perf_counter() - start
        record.split.assembly_seconds += elapsed - (
            record.lookup_seconds() - lookups_before
        )

    def solve_linear_system(self) -> np.ndarray:
        """Solve, and record the time.

        Returns:
            The solution of the linear system.

        """
        record = self._cost_record
        start = time.perf_counter()
        solution = super().solve_linear_system()  # type: ignore[misc]
        record.split.linear_solve_seconds += time.perf_counter() - start
        return solution

    def before_nonlinear_loop(self) -> None:
        """Mark the start of an attempt at a time step."""
        self._cost_record.begin_attempt()
        super().before_nonlinear_loop()  # type: ignore[misc]

    def after_nonlinear_convergence(self) -> None:
        """Record the iteration count of an accepted time step."""
        iterations = int(self.nonlinear_solver_statistics.num_iteration)  # type: ignore[attr-defined]
        self._cost_record.iterations.append(iterations)
        self._cost_record.end_attempt(rejected=False, iterations=iterations)
        super().after_nonlinear_convergence()  # type: ignore[misc]

    def after_nonlinear_failure(self) -> None:
        """Record a time step the solver failed on, before the model's own method raises."""
        self._cost_record.end_attempt(
            rejected=True,
            iterations=int(self.nonlinear_solver_statistics.num_iteration),  # type: ignore[attr-defined]
        )
        super().after_nonlinear_failure()  # type: ignore[misc]


@dataclass
class _CostRecord:
    """What the instrumentation accumulates during one run."""

    split: RuntimeSplit = field(default_factory=RuntimeSplit)
    iterations: list[int] = field(default_factory=list)
    rejected_steps: int = 0
    rejected_seconds: float = 0.0
    rejected_evaluations: int = 0
    rejected_iterations: int = 0
    samplers: list[TimedSampler] = field(default_factory=list)
    attempt_start_seconds: float | None = None
    attempt_start_evaluations: int = 0

    def begin_attempt(self) -> None:
        """Mark the start of one attempt at a time step."""
        self.attempt_start_seconds = time.perf_counter()
        self.attempt_start_evaluations = self.evaluations()

    def end_attempt(self, rejected: bool, iterations: int) -> None:
        """Close the attempt, keeping what it cost if it was thrown away.

        Parameters:
            rejected: Whether the solver failed on this attempt.
            iterations: Newton iterations the attempt used.

        """
        if self.attempt_start_seconds is None:
            return
        if rejected:
            self.rejected_steps += 1
            self.rejected_seconds += time.perf_counter() - self.attempt_start_seconds
            self.rejected_evaluations += (
                self.evaluations() - self.attempt_start_evaluations
            )
            self.rejected_iterations += iterations
        self.attempt_start_seconds = None

    def lookup_seconds(self) -> float:
        """Time spent in table lookups so far, over all samplers.

        Returns:
            The time in seconds.

        """
        return sum(sampler.lookup_seconds for sampler in self.samplers)

    def evaluations(self) -> int:
        """States looked up so far, over all samplers.

        Returns:
            The number of states.

        """
        return sum(sampler.evaluations for sampler in self.samplers)

    def calls(self) -> int:
        """Lookup calls so far, over all samplers.

        Returns:
            The number of calls.

        """
        return sum(sampler.calls for sampler in self.samplers)

    def load_seconds(self) -> float:
        """Time spent reading tables, over all samplers.

        Returns:
            The time in seconds.

        """
        return sum(sampler.load_seconds for sampler in self.samplers)


@contextmanager
def timed_samplers(driver: Any) -> Generator[list[TimedSampler], None, None]:
    """Make the driver construct :class:`TimedSampler` instances inside the block.

    Parameters:
        driver: The simulation-driver module.

    Yields:
        The samplers constructed inside the block, in construction order.

    """
    constructed: list[TimedSampler] = []
    original = driver.VTKSampler

    class _Recording(TimedSampler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            constructed.append(self)

    driver.VTKSampler = _Recording
    try:
        yield constructed
    finally:
        driver.VTKSampler = original


def primary_variable_values(model: Any) -> dict[str, np.ndarray]:
    """Values of each primary variable at the current time.

    Parameters:
        model: A prepared model.

    Returns:
        Variable name to its values, concatenated over the grids.

    """
    values: dict[str, list[np.ndarray]] = {}
    for variable in model.equation_system.variables:
        values.setdefault(variable.name, []).append(
            model.equation_system.get_variable_values([variable], time_step_index=0)
        )
    return {name: np.concatenate(parts) for name, parts in values.items()}


def solution_difference(
    values: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, tuple[float, int]]:
    """Difference between two final solutions per variable, in the norm of Experiment A.

    Parameters:
        values: Final values of one run.
        reference: Final values of the reference run.

    Returns:
        Variable name to ``(error, excluded)``; variables missing from either run are skipped.

    """
    errors: dict[str, tuple[float, int]] = {}
    for name, array in values.items():
        other = reference.get(name)
        if other is None or other.shape != array.shape:
            continue
        errors[name] = variation_normalized_l2_error(array, other)
    return errors


@dataclass
class _Snapshot:
    """Counters at the moment the time loop starts."""

    lookup_seconds: float
    evaluations: int
    calls: int
    assembly_seconds: float
    linear_solve_seconds: float

    @classmethod
    def of(cls, record: _CostRecord) -> _Snapshot:
        """Take a snapshot of a record.

        Parameters:
            record: The record.

        Returns:
            The snapshot.

        """
        return cls(
            lookup_seconds=record.lookup_seconds(),
            evaluations=record.evaluations(),
            calls=record.calls(),
            assembly_seconds=record.split.assembly_seconds,
            linear_solve_seconds=record.split.linear_solve_seconds,
        )


def overridden(block: dict[str, Any], overrides: dict[str, Any], name: str) -> dict:
    """Copy of a configuration block with entries replaced.

    Parameters:
        block: The block to copy.
        overrides: Entries to replace.
        name: Block name, for the error message.

    Returns:
        The new block.

    Raises:
        KeyError: If an override names an entry the block does not have.

    """
    unknown = sorted(set(overrides) - set(block))
    if unknown:
        raise KeyError(f"{unknown} are not entries of the configuration's {name} block")
    return {**block, **overrides}


def run_benchmark(
    level: int,
    table_path: str | Path,
    config_path: str | Path,
    end_time_years: float | None,
    seconds_per_year: float,
    time_overrides: dict[str, float] | None = None,
    model_overrides: dict[str, Any] | None = None,
    model_class: type | None = None,
    solver_overrides: dict[str, Any] | None = None,
) -> tuple[SimulationRun, dict[str, np.ndarray]]:
    """Run the benchmark once with one table and record what it cost; only the time loop is timed.

    Parameters:
        level: Refinement level of the table, for reporting.
        table_path: The PHZ table to run with.
        config_path: The benchmark configuration.
        end_time_years: End time in years; the configured one if None.
        seconds_per_year: Conversion used for ``end_time_years``.
        time_overrides: Entries of the ``time`` block to replace.
        model_overrides: Entries of the ``model`` block to replace.
        model_class: The model to run; the one the driver builds if None.
        solver_overrides: Entries of the ``solver`` block to replace.

    Returns:
        The run and the final primary-variable values; a failed run has ``completed``
        False and the reason in ``message``.

    Raises:
        KeyError: If an override names an entry the configuration does not have.

    """
    import porepy as pp
    from porepy.examples.geothermal_flow import simulation_driver as driver

    config = driver.load_config(config_path)
    # Joining an absolute path replaces the directory, so only the PHZ table
    # moves and the PTZ table stays where the configuration puts it.
    config["vtk"] = {**config["vtk"], "phz_file": str(Path(table_path).resolve())}
    config["case_name"] = f"{config['case_name']}_cost_level_{level}"
    if end_time_years is not None:
        config["time"] = {**config["time"], "end": end_time_years * seconds_per_year}
    config["time"] = overridden(config["time"], time_overrides or {}, "time")
    config["model"] = overridden(config["model"], model_overrides or {}, "model")
    config["solver"] = overridden(config["solver"], solver_overrides or {}, "solver")

    # The driver creates this directory in its own run(); the exporter needs it.
    Path(driver.visualization_dir(config)).mkdir(parents=True, exist_ok=True)

    if model_class is None:
        model_class = driver.create_model_class(config)
    instrumented = type(
        "InstrumentedBenchmarkModel", (CostInstrumentation, model_class), {}
    )
    params = {**driver.build_model_params(config), **driver.build_solver_params(config)}
    model = instrumented(params)
    record = _CostRecord()
    model._cost_record = record

    with timed_samplers(driver) as samplers:
        driver.attach_vtk_samplers(model, config)
    record.samplers.extend(samplers)

    start = time.perf_counter()
    model.prepare_simulation()
    prepare_seconds = time.perf_counter() - start
    params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()
    driver.configure_schur_if_requested(model, config)

    before = _Snapshot.of(record)
    completed, message = True, ""
    start = time.perf_counter()
    try:
        pp.run_time_dependent_model(model, params)
    except Exception as error:  # noqa: BLE001 - a failed run is a result, not a crash
        completed, message = False, f"{type(error).__name__}: {error}"
    total_seconds = time.perf_counter() - start

    if completed and not model.time_manager.final_time_reached():
        completed, message = False, "the time loop ended before the end time"

    split = RuntimeSplit(
        total_seconds=total_seconds,
        thermodynamics_seconds=record.lookup_seconds() - before.lookup_seconds,
        assembly_seconds=record.split.assembly_seconds - before.assembly_seconds,
        linear_solve_seconds=(
            record.split.linear_solve_seconds - before.linear_solve_seconds
        ),
    )
    run = SimulationRun(
        level=level,
        completed=completed,
        split=split,
        prepare_seconds=prepare_seconds,
        table_load_seconds=record.load_seconds(),
        closure_evaluations=record.evaluations() - before.evaluations,
        closure_calls=record.calls() - before.calls,
        iterations=tuple(record.iterations),
        rejected_steps=record.rejected_steps,
        final_time_seconds=float(model.time_manager.time),
        rejected_seconds=record.rejected_seconds,
        rejected_evaluations=record.rejected_evaluations,
        rejected_iterations=record.rejected_iterations,
        message=message,
    )
    return run, primary_variable_values(model)


@dataclass
class CostResult:
    """Everything Experiment C measures, per refinement level.

    Attributes:
        evaluation: The per-evaluation comparison, or None if not measured.
        tables: Refinement level mapped to what its table cost to build.
        simulations: Refinement level mapped to its repeated runs.
        boundary_errors: Refinement level mapped to the clean boundary-set
            error of Experiment A, the accuracy the cost buys.
    """

    evaluation: EvaluationComparison | None = None
    tables: dict[int, TableCost] = field(default_factory=dict)
    simulations: dict[int, SimulationCost] = field(default_factory=dict)
    boundary_errors: dict[int, float] = field(default_factory=dict)

    def projected_direct_seconds(self, level: int) -> float | None:
        """Projected wall-clock of the same run with direct evaluation, assuming the same solver path.

        Parameters:
            level: Refinement level whose run is used.

        Returns:
            The projected time in seconds, or None if the run or the comparison is missing.

        """
        cost = self.simulations.get(level)
        run = cost.representative if cost is not None else None
        if run is None or self.evaluation is None:
            return None
        model = self.evaluation.direct.model
        direct_thermodynamics = (
            model.fixed_seconds * run.closure_calls
            + model.per_evaluation_seconds * run.closure_evaluations
        )
        return (
            run.split.total_seconds
            - run.split.thermodynamics_seconds
            + direct_thermodynamics
        )

    def scaling_checks(self) -> list[ScalingCheck]:
        """Check that table size, node count and generation time grow as ``eta**-3``.

        Returns:
            One check per quantity with at least two levels available.

        """
        from porepy.examples.geothermal_flow.obl_convergence_analysis import (
            cost_config as config,
        )

        levels = sorted(self.tables)
        if len(levels) < 2:
            return []
        coarsest = self.tables[levels[0]].spacing[0]
        factors = [coarsest / self.tables[level].spacing[0] for level in levels]
        checks: list[ScalingCheck] = []
        quantities: dict[str, list[float | None]] = {
            "node count": [float(self.tables[k].node_count) for k in levels],
            "table size": [float(self.tables[k].file_bytes) for k in levels],
            "generation time": [
                (
                    None
                    if self.tables[k].generation_seconds is None
                    else float(self.tables[k].generation_seconds)
                )
                for k in levels
            ],
        }
        for name, values in quantities.items():
            usable = [
                (f, v) for f, v in zip(factors, values) if v is not None and v > 0.0
            ]
            if len(usable) < 2:
                continue
            checks.append(
                check_power_law(
                    [f for f, _ in usable],
                    [v for _, v in usable],
                    name=name,
                    expected_exponent=config.EXPECTED_SCALING_EXPONENT,
                    tolerance=config.SCALING_EXPONENT_TOLERANCE,
                )
            )
        return checks


def _evaluation_cost_to_json(cost: EvaluationCost) -> dict[str, Any]:
    """Represent one evaluation cost as JSON-compatible data.

    Parameters:
        cost: The evaluation cost.

    Returns:
        The data.

    """
    return {
        "name": cost.name,
        "timings": {str(size): list(t.seconds) for size, t in cost.timings.items()},
        "model": asdict(cost.model),
    }


def _evaluation_cost_from_json(stored: dict[str, Any]) -> EvaluationCost:
    """Rebuild one evaluation cost from stored data.

    Parameters:
        stored: The stored data.

    Returns:
        The evaluation cost.

    """
    return EvaluationCost(
        name=stored["name"],
        timings={
            int(size): RepeatedTiming(tuple(seconds))
            for size, seconds in stored["timings"].items()
        },
        model=LinearCostModel(**stored["model"]),
    )


def _run_from_json(stored: dict[str, Any]) -> SimulationRun:
    """Rebuild one simulation run from stored data.

    Parameters:
        stored: The stored data.

    Returns:
        The run.

    """
    stored = dict(stored)
    stored["split"] = RuntimeSplit(**stored["split"])
    stored["iterations"] = tuple(stored["iterations"])
    return SimulationRun(**stored)


def save_cost_result(path: str | Path, result: CostResult) -> None:
    """Write the measurements to JSON, so that the stages can run separately.

    Parameters:
        path: Destination file, created with its parent if absent.
        result: The measurements.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Any] = {
        "tables": {str(k): asdict(v) for k, v in sorted(result.tables.items())},
        "simulations": {
            str(level): {
                "level": cost.level,
                "runs": [asdict(run) for run in cost.runs],
                "solution_error": {
                    name: list(value) for name, value in cost.solution_error.items()
                },
            }
            for level, cost in sorted(result.simulations.items())
        },
        "boundary_errors": {
            str(k): v for k, v in sorted(result.boundary_errors.items())
        },
    }
    if result.evaluation is not None:
        stored["evaluation"] = {
            "state_count": result.evaluation.state_count,
            "direct": _evaluation_cost_to_json(result.evaluation.direct),
            "lookups": {
                str(level): _evaluation_cost_to_json(cost)
                for level, cost in sorted(result.evaluation.lookups.items())
            },
        }
    path.write_text(json.dumps(stored, indent=2))


def load_cost_result(path: str | Path) -> CostResult:
    """Read measurements written by :func:`save_cost_result`.

    Parameters:
        path: The file; a missing file gives an empty result.

    Returns:
        The measurements.

    """
    path = Path(path)
    if not path.exists():
        return CostResult()
    stored = json.loads(path.read_text())
    evaluation = None
    if "evaluation" in stored:
        evaluation = EvaluationComparison(
            state_count=stored["evaluation"]["state_count"],
            direct=_evaluation_cost_from_json(stored["evaluation"]["direct"]),
            lookups={
                int(level): _evaluation_cost_from_json(cost)
                for level, cost in stored["evaluation"]["lookups"].items()
            },
        )
    tables = {}
    for level, entry in stored.get("tables", {}).items():
        entry = dict(entry)
        entry["spacing"] = tuple(entry["spacing"])
        tables[int(level)] = TableCost(**entry)
    simulations = {
        int(level): SimulationCost(
            level=entry["level"],
            runs=tuple(_run_from_json(run) for run in entry["runs"]),
            solution_error={
                name: (value[0], int(value[1]))
                for name, value in entry["solution_error"].items()
            },
        )
        for level, entry in stored.get("simulations", {}).items()
    }
    return CostResult(
        evaluation=evaluation,
        tables=tables,
        simulations=simulations,
        boundary_errors={
            int(level): float(value)
            for level, value in stored.get("boundary_errors", {}).items()
        },
    )
