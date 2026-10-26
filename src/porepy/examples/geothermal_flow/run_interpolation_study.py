"""Study driver for the verification results of the geothermal model paper.

This module runs a study that is made of several command-line stages, such as
the interpolation study of Section 4.2 in ``obl_convergence_analysis``. That
study keeps its settings in Python modules (``convergence_config.py`` and
``cost_config.py``) and is run stage by stage through its own runners. The
study manifest, for example ``geothermal_flow/configs/obl_study.yaml``, lists
those stages in the order they must run, and records the module settings that
produced the paper results.

Before any stage runs, the recorded settings are compared with the current
values in the modules, and the study stops if one differs. A setting that
changed since the paper would otherwise change Figures 7 and 8 silently.

Typical usage from ``src/porepy/examples`` is

    python -m geothermal_flow.run_interpolation_study \\
        --config geothermal_flow/configs/obl_study.yaml

To run selected stages only, use

    python -m geothermal_flow.run_interpolation_study \\
        --config geothermal_flow/configs/obl_study.yaml \\
        --steps cost_simulate cost_errors

To inspect the commands without executing them, add ``--dry-run``.

The stages write to the folders their modules define, for example
``obl_convergence_analysis/tables_experiment_a/`` and
``obl_convergence_analysis/experiment_c/cost.json``. The figures are drawn
afterwards by ``geothermal_flow.make_figures``.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from porepy.examples.geothermal_flow.io_utils import load_yaml


def as_plain(value: Any) -> Any:
    """Convert a setting to the form YAML loads it in.

    Dataclasses become mappings, tuples become lists and numbers become floats,
    so that ``(0.08, 80.0, 80.0)`` in a module equals ``[0.08, 80, 80]`` in YAML.
    """
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def check_expected_settings(expected: dict[str, dict[str, Any]]) -> None:
    """Compare module settings with the values recorded in the manifest.

    Raises:
        ValueError: If any recorded setting differs from the module value.
    """
    mismatches = []
    for module_name, settings in expected.items():
        module = importlib.import_module(module_name)
        for name, recorded in settings.items():
            current = as_plain(getattr(module, name))
            if current != as_plain(recorded):
                mismatches.append(
                    f"  {module_name}.{name}:\n"
                    f"    manifest: {as_plain(recorded)}\n"
                    f"    module:   {current}"
                )
    if mismatches:
        raise ValueError(
            "Study settings differ from the manifest:\n" + "\n".join(mismatches)
        )
    print("Study settings match the manifest.")


def run_step(name: str, step_cfg: dict[str, Any], *, dry_run: bool = False) -> None:
    """Run one stage as ``python -m <module> <args>``, streaming its output."""
    command = [sys.executable, "-m", step_cfg["module"]]
    command.extend(str(arg) for arg in step_cfg.get("args", []))

    print("\n" + "=" * 80)
    print(f"Step: {name}")
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a geothermal-flow study.")
    parser.add_argument("--config", required=True, help="Path to the study YAML file.")
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Optional list of step keys to run. If omitted, run all, in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the study workflow."""
    args = parse_args()

    config = load_yaml(args.config)
    steps = config["steps"]

    selected = args.steps or list(steps.keys())
    missing = [name for name in selected if name not in steps]
    if missing:
        raise KeyError(f"Unknown step names: {missing}")

    check_expected_settings(config.get("expected_settings", {}))

    # Run in manifest order, whatever order the steps were requested in.
    for name in steps:
        if name in selected:
            run_step(name, steps[name], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
