"""
Test all available executable examples.

The tests collect Python example modules from ``porepy.examples`` and execute
their ``run_example()`` functions. Each executable example is expected to return
a list of models. For each returned model, the simulation status is checked to
verify that the simulation completes successfully or stops in a controlled way.

"""

import importlib
from pathlib import Path

import matplotlib
import pytest

import porepy

# Disable plotting during tests.
matplotlib.use("template")

EXAMPLE_DIR = Path(porepy.__file__).parent / "examples"
# Exclude .py files that are not executable examples and example files
# that are temporarily not tested in the folder of examples.
EXCLUDED_EXAMPLE_FILENAMES = [
    "__init__.py",
    "example_params.py",
    "geothermal_reservoir.py",
]
EXAMPLE_FILENAMES = [
    path
    for path in EXAMPLE_DIR.glob("*.py")
    if path.name not in EXCLUDED_EXAMPLE_FILENAMES
]


@pytest.mark.examples
@pytest.mark.parametrize("example_path", EXAMPLE_FILENAMES)
def test_run_examples(example_path: Path):
    """We run the executable examples and check that they didn't raise any error.

    The test imports the example module, verifies the definition of a ``run_example()``
    function, and checks the simulation status of models.

    """
    module_name = f"porepy.examples.{example_path.stem}"
    module = importlib.import_module(module_name)

    # The executable example is required to define a run_example() function.
    if not hasattr(module, "run_example"):
        raise AssertionError(f"{module_name} does not define run_example().")

    models = module.run_example()

    for model in models:
        status = model.nonlinear_solver_statistics
        assert status.simulation_status.is_success(), (
            f"{module_name} did not complete successfully."
            f"Simulation status: {status.simulation_status}."
        )
