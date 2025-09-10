"""We perform a static scan of the model-related files to find uses of `self.params`.
The detected parameter keys are expected to match those in the example config, ensuring
it stays up to date. We also enforce lowercase keys for the PorePy parameters.

"""

import re
from pathlib import Path

import porepy as pp
from porepy.examples.example_params import model_params, solver_params


def search_params_directory(root_dirs: list[Path]) -> set[str]:
    """Scans the files in the provided list of directories and returns the found
    parameter keys.

    """

    pattern_brackets = re.compile(r'params\[\s*"([\S^]+?)"\s*\]')
    """Captures `param_name` in `params["param_name"]`."""
    pattern_get = re.compile(r'params\.get\(\s*"(\S+?)"\s*')
    """Captures `param_name` in `params.get("param_name")."""

    def search_params_file(filepath: Path) -> set[str]:
        """Scans the provided file and returns the found parameter keys."""
        matches: set[str] = set()
        lines = filepath.read_text()
        found = pattern_brackets.findall(lines)
        matches.update(found)
        found = pattern_get.findall(lines)
        matches.update(found)
        return matches

    results: set[str] = set()
    for root_dir in root_dirs:
        assert root_dir.exists()
        for fname in root_dir.rglob("*.py"):
            results.update(search_params_file(fname))
    return results


def compare_parameters_with_expected(found: set[str], expected: set[str]) -> bool:
    """Ensures that the two sets of parameters are the same."""
    # This could be done in one line, but the error message is more visual this way.
    # Useful for debugging, does not hurt for the test.
    print(f"\nParams in config but not in files: {expected - found}")
    print(f"\nParams in files but not in config: {found - expected}")
    return len(found - expected) == 0 and len(expected - found) == 0


def test_example_params_up_to_date():
    """Compares the first-level keys in the example_params with the parameters used in
    the source files.

    """
    pp_path = Path(pp.__file__).parent
    directories_params = [
        pp_path / "models",
        pp_path / "viz",
        pp_path / "numerics/nonlinear",
    ]
    # There are two distinct dictionaries of parameters: for the model and for the
    # solver. There's no way to statically detect what belongs to which dict.
    # So far, we check that they exist at least somewhere.
    params_common = set(model_params.keys()) | set(solver_params.keys())

    # First, we ensure that the parameters are in lower case.
    for param in params_common:
        assert param.lower() == param, (
            f"We enforce only lower case parameters, {param}."
        )

    found = search_params_directory(directories_params)
    assert compare_parameters_with_expected(found=found, expected=params_common)
