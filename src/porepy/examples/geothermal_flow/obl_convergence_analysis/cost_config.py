"""Settings for Experiment C, the cost study; the shared settings are in :mod:`convergence_config`."""

from pathlib import Path

from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_config import (
    CWD,
    SEQUENCE,
)

BENCHMARK_CONFIG = Path(CWD).parent / "configs" / "benchmark.yaml"
"""Path: The one-dimensional benchmark, run once per table; read, never written."""

COST_DIR = Path(CWD) / "experiment_c"
"""Path: Where the measured costs and the sample states are written."""

COST_FIGURE_DIR = Path(CWD) / "figures_experiment_c"
"""Path: Where the cost-against-error figure is written."""

TIMING_TABLE_DIR = COST_DIR / "tables_timing"
"""Path: Scratch folder for the tables rebuilt to time their generation."""

GENERATION_COST_FILE = COST_DIR / "table_generation_cost.json"
"""Path: Cached generation times, reused only while node count and file size match."""

BOUNDARY_STATES = COST_DIR / "boundary_states.txt"
"""Path: Phase-boundary states used for the per-evaluation timings, written on the first run."""

EVALUATION_BATCH_SIZES = (128, 512, 2048)
"""tuple: Batch sizes timed, to separate the per-call from the per-point cost."""

EVALUATION_REPEATS = 3
"""int: Timing repeats per batch size, reported as median and spread."""

SIMULATION_REPEATS = 3
"""int: Benchmark runs per table resolution, reported as median and spread."""

END_TIME_YEARS: float | None = 500 # None
"""float or None: End time of the benchmark runs in years; None keeps the configured one."""

TIME_OVERRIDES: dict[str, float] = {"dt_min": 864000.0}
"""dict: Entries of the configuration's ``time`` block overridden for every resolution."""

MODEL_OVERRIDES: dict[str, object] = {"fractional_flow": False}
"""dict: Entries of the configuration's ``model`` block overridden, e.g. the formulation."""

SECONDS_PER_YEAR = 3.1536e7
"""float: Seconds in a year, as used by the benchmark configuration."""

EXPECTED_SCALING_EXPONENT = 3.0
"""float: Expected growth of node count, size and generation time, as ``eta**-3``."""

SCALING_EXPONENT_TOLERANCE = 0.15
"""float: How far the measured exponent may sit from the expected one."""

LEVELS = tuple(range(SEQUENCE.num_levels))
"""tuple: Refinement levels the cost study covers."""

ERROR_FIELDS: tuple[str, ...] = ("Temperature", "S_l", "S_v", "S_h", "Rho", "H")
"""tuple: Mixture quantities whose worst phase-boundary error is the figure's error axis."""



