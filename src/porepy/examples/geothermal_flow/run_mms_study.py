"""Run the manufactured-solution study of Appendix A from a YAML configuration.

The study itself is :func:`geothermal_halite_verification_driver.main`. This
module only builds its parameters from a configuration file, so that the mesh
sequence, the manufactured-solution constants and the output folder of each
paper result are recorded next to the other paper configurations, rather than
in the module-level settings of the driver.

Typical usage from ``src/porepy/examples`` is

    python -m porepy.examples.geothermal_flow.run_mms_study \\
        --config geothermal_flow/configs/mms_cartesian.yaml

    python -m porepy.examples.geothermal_flow.run_mms_study \\
        --config geothermal_flow/configs/mms_simplex_ensemble.yaml

To print the parameters without running the study, add ``--dry-run``.

The driver writes ``mms_results_<grid_type>.json`` (one mesh per level) or
``mms_results_ensemble_<grid_type>.json`` (mesh ensemble) to the output folder.
``geothermal_flow.make_figures`` draws Figures 21 and 22 from these files.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import porepy as pp

from porepy.examples.geothermal_flow.cf_convergence_analysis import (
    mms_halite_convergence as driver,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_constants import (
    MMS_HALITE,
    HaliteMMSConstants,
)
from porepy.examples.geothermal_flow.io_utils import load_yaml

GEOTHERMAL_DIR = Path(__file__).resolve().parent
"""Path: The ``geothermal_flow`` folder, against which relative output folders resolve."""


def build_constants(overrides: dict[str, Any]) -> HaliteMMSConstants:
    """Return the canonical constants with the configured entries replaced.

    Args:
        overrides: Field names of :class:`HaliteMMSConstants` and their values.

    Returns:
        A new constants instance.

    Raises:
        KeyError: If an override names a field the constants do not have. Such an
            entry would change nothing, and a silent no-op in a verification study
            is a wrong result rather than a harmless mistake.
    """
    known = {field.name for field in fields(MMS_HALITE)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise KeyError(f"{unknown} are not fields of HaliteMMSConstants")
    return replace(MMS_HALITE, **{name: float(value) for name, value in overrides.items()})


def output_directory(config: dict[str, Any]) -> Path:
    """Folder the driver writes its results and VTU files to."""
    directory = Path(config["output"]["directory"])
    return directory if directory.is_absolute() else GEOTHERMAL_DIR / directory


def results_path(config: dict[str, Any]) -> Path:
    """The JSON file the driver writes for this configuration."""
    study = config["study"]
    prefix = "mms_results_ensemble_" if study["ensemble"] else "mms_results_"
    return output_directory(config) / f"{prefix}{study['grid_type']}.json"


def build_params(config: dict[str, Any]) -> dict[str, Any]:
    """Build the driver's model parameters from the configuration.

    :func:`driver.model_parameters` reads the canonical constants, so every entry
    it derives from them is rebuilt here from the configured constants. The
    model takes its closure, exact solution and aperture law from
    ``params["mms_constants"]``, so the overrides reach all of them.

    Args:
        config: The loaded YAML configuration.

    Returns:
        The model parameter dictionary passed to :func:`driver.main`.
    """
    study = config["study"]
    solver = config.get("solver", {})

    params = driver.model_parameters(
        grid_type=study["grid_type"],
        levels=int(study["levels"]),
        base_cell_size=float(study["base_cell_size"]),
        dt=float(study["dt"]),
        final_time=float(study["final_time"]),
    )

    constants = build_constants(config.get("mms_constants", {}))
    params["mms_constants"] = constants
    params["material_constants"] = {
        "solid": pp.SolidConstants(
            residual_aperture=constants.a_res,
            permeability=constants.K0,
            normal_permeability=constants.K_k,
            porosity=constants.phi0,
            thermal_conductivity=constants.kappa_r,
            density=constants.rho_r,
            specific_heat_capacity=constants.cp_r,
        )
    }
    params["reference_variable_values"] = pp.ReferenceVariableValues(
        pressure=constants.p0, temperature=constants.T0
    )

    if "nonlinear_tolerance" in solver:
        tolerance = float(solver["nonlinear_tolerance"])
        params["nl_convergence_tol"] = tolerance
        params["nl_convergence_tol_res"] = tolerance
    if "global_line_search" in solver:
        params["global_line_search"] = bool(solver["global_line_search"])

    params["folder_name"] = str(output_directory(config))
    params["convergence"] = True
    return params


def run(config_path: str | Path, *, dry_run: bool = False) -> Path:
    """Run one manufactured-solution study and return its results file."""
    config = load_yaml(config_path)
    params = build_params(config)
    study = config["study"]
    constants = params["mms_constants"]
    target = results_path(config)

    print(f"Study:      {config.get('study_name', Path(config_path).stem)}")
    print(
        f"Grid:       {study['grid_type']}, ensemble={bool(study['ensemble'])}, "
        f"levels={params['levels']}, "
        f"base cell size={params['meshing_arguments']['cell_size']}"
    )
    print(f"Aperture:   a0={constants.a0:g}, theta={constants.theta:g}")
    print(f"Saturation: S0={constants.S0:g}, S1={constants.S1:g}")
    print(f"Tolerance:  {params['nl_convergence_tol']:g}")
    print(f"Results:    {target}")

    if dry_run:
        return target

    # The driver selects the single-mesh or the ensemble study, and fixes the
    # ensemble jitters, inside main(); only the selection is set here.
    driver.use_ensemble_of_mesh = bool(study["ensemble"])
    driver.show_ensemble_members = False
    driver.main(params)

    if not target.exists():
        raise FileNotFoundError(f"The study finished but wrote no {target}")
    print(f"\nResults written to {target}")
    return target


def main() -> None:
    """Run the study selected on the command line."""
    parser = argparse.ArgumentParser(
        description="Run the Appendix A manufactured-solution study."
    )
    parser.add_argument("--config", required=True, help="Path to the study YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parameters without running the study.",
    )
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
