"""We perform a static scan of the model-related files to find uses of `self.params`.
The detected parameter keys are expected to match those in the example config, ensuring
it stays up to date. We also enforce lowercase keys for the PorePy parameters.

"""

import os
import re

from porepy.examples.example_params import model_params, solver_params


def search_params_directory(root_dirs: list[str]) -> set[str]:
    """Scans the files in the provided list of directories and returns the found
    parameter keys.

    """

    pattern_brackets = re.compile(r'params\[\s*"([\S^]+?)"\s*\]')
    """Captures `param_name` in `params["param_name"]`."""
    pattern_get = re.compile(r'params\.get\(\s*"(\S+?)"\s*')
    """Captures `param_name` in `params.get("param_name")."""

    def search_params_file(filepath: str) -> set[str]:
        """Scans the provided file and returns the found parameter keys."""
        matches: set[str] = set()
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read()
            found = pattern_brackets.findall(lines)
            matches.update(found)
            found = pattern_get.findall(lines)
            matches.update(found)
        return matches

    results: set[str] = set()
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".py"):
                    path = os.path.join(dirpath, fname)
                    results.update(search_params_file(path))
    return results


def compare_parameters_with_expected(found: set[str], expected: set[str]) -> bool:
    """Ensures that the two sets of parameters are the same."""
    # This could be done in one line, but the error message is more visual this way.
    # Useful for debugging, does not hurt for the test.
    print("\nParams in config but not in files:")
    success = True
    for param in expected:
        if param not in found:
            print(param)
            success = False
    print("\nParams in files but not in config:")
    for param in found:
        if param not in expected:
            print(param)
            success = False
    return success


def test_example_params_up_to_date():
    """Compares the first-level keys in the example_params with the parameters used in
    the source files.

    """
    directories_model_params = [
        "src/porepy/models",
        "src/porepy/viz",
    ]
    directories_solver_params = [
        "src/porepy/numerics/nonlinear",
    ]
    # There are two distinc dictionaries of parameters: for the model and for the
    # solver. There's no way to statically detect what belongs to which dict.
    # So far, we check that they exist at least somewhere.
    directories_common = directories_model_params + directories_solver_params
    params_common = set(model_params.keys()) | set(solver_params.keys())

    # First, we ensure that the parameters are in lower case.
    for param in params_common:
        assert param.lower() == param, (
            f"We enforse only lower case parameters, {param}."
        )

    found = search_params_directory(directories_common)
    assert compare_parameters_with_expected(found=found, expected=params_common)
