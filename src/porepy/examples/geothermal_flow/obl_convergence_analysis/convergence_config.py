"""Settings shared by Experiments A and B.

:data:`SEQUENCE` and :data:`EXTEND_ABSENT_PHASES` take effect only after the
tables are regenerated with ``run_experiment_a.py --stage generate``.

"""

from pathlib import Path

from porepy.examples.geothermal_flow.obl_convergence_analysis.interpolation_convergence import (
    FIELD_MAP_CLI,
    RefinementSequence,
)

CLI = "./sweos_cli/swEOS"
"""str: Path to the EOS executable, relative to CWD."""

CWD = str(Path(__file__).resolve().parent)
"""str: Working directory for CLI calls."""

TABLE_DIR = Path(CWD) / "tables_experiment_a"
"""Path: Where the generated tables are written."""

FUNCTION_FIGURE_DIR = Path(CWD) / "figures_experiment_a"
"""Path: Where the figures of Experiment A (property values) are written."""

GRADIENT_FIGURE_DIR = Path(CWD) / "figures_experiment_b"
"""Path: Where the figures of Experiment B (property gradients) are written."""

BENCHMARK_STATES = Path(CWD) / "benchmark_states.txt"
"""Path: Benchmark states, one ``(z, h, p)`` per line; the benchmark sample set is skipped if missing."""

# The coordinate transform VTKSampler applies to the query points. The sample
# points are built in table units (kJ/kg, bar), so the identity is correct here.
# The solver passes SI units and uses (1.0, 1e-3, 1e-5) instead.
CONVERSION_FACTORS = (1.0, 1.0, 1.0)
"""tuple: Scaling applied to (z, h, p) before probing."""

TRANSLATION_FACTORS = (0.0, 0.0, 0.0)
"""tuple: Offset applied to (z, h, p) after scaling."""

# The refinement sequence. The spacing halves from level to level, so the finest
# of the four levels has the spacing of the production table, (0.01, 10, 10).
# The ranges are those of the production table. Their lengths are multiples of
# the coarsest spacing, so every level has nodes at both ends of each range and
# only the spacing varies between levels.

SEQUENCE = RefinementSequence(
    initial_spacing=(0.08, 80.0, 80.0),
    num_levels=4,
    z_range=(0.0001, 0.9601),
    h_range=(120.0, 3000.0),
    p_range=(10.0, 490.0),
)

EXTEND_ABSENT_PHASES = False
"""bool: Fill absent-phase properties from the nearest node where the phase is present."""


FIELDS = list(FIELD_MAP_CLI.keys())
"""list: Fields to compare, each reported separately."""

STUDY_SETTINGS = {
    "sequence": SEQUENCE,
    "cli_path": CLI,
    "fields": FIELDS,
    "cwd": CWD,
    "conversion_factors": CONVERSION_FACTORS,
    "translation_factors": TRANSLATION_FACTORS,
}
"""dict: Constructor arguments shared by the studies of both experiments."""


def table_path(level: int) -> Path:
    """Path of the generated table at one refinement level.

    Parameters:
        level: Refinement level.

    Returns:
        The table path.

    """
    return TABLE_DIR / f"XHP_level_{level}.vtk"


def benchmark_states_path() -> Path | None:
    """Path of the benchmark states file.

    Returns:
        The path, or None if the file does not exist.

    """
    return BENCHMARK_STATES if BENCHMARK_STATES.exists() else None
