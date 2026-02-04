"""Flow model configuration file for the cold injection examples.

Setting of fluid mixtures, initial and boundary conditions, and other model parameters
such as position of wells and injection rates.

"""

from __future__ import annotations

import numpy as np

import porepy as pp
from porepy.applications.material_values.solid_values import basalt

from .solver import NewtonArmijoAndersonSolver


class ModelConfig(pp.PorePyModel):
    """Helper class to bundle the model configuration and inherit from
    ``PorePyModel``."""

    ### Domain configurations.

    _DOMAIN_DIMENSIONS: list[float] = [100.0, 20.0, 100.0]
    """Domain dimensions in meters."""

    ### Fluid components.

    _COMPONENT_NAMES: list[str] = ["H2O", "CO2"]
    """Names of fluid components used in the model."""

    _IDEAL_COMPONENTS: list[pp.compositional.ideal.IdealFluid] = [
        pp.compositional.ideal.IdealH2O,
        pp.compositional.ideal.IdealCO2,
    ]
    """Ideal fluid components used in the model."""

    ### Initial values.

    _p_INIT: float = 20e6
    """Initial pressure in the whole domain in Pascals."""

    _T_INIT: float = 450.0
    """Initial temperature in the whole domain in Kelvin."""

    _z_INIT: dict[str, float] = dict(
        [(n, z) for n, z in zip(_COMPONENT_NAMES, [0.995, 0.005])]
    )
    """Initial overall fractions of fluid components in the whole domain, given as a
    dictionary mapping component names to values."""

    ### Injection and production conditions.

    _p_OUT: float = _p_INIT - 1e6
    """Pressure at production well in Pascals."""

    _T_IN: float = 300.0
    """Temperature of injected fluid in Kelvin."""

    _z_IN: dict[str, float] = dict(
        [(n, z) for n, z in zip(_COMPONENT_NAMES, [0.9, 0.1])]
    )
    """Overall fractions of injected fluid, given as a dictionary mapping component
    names to values. Total injection is scaled by these values per component."""

    _TOTAL_INJECTED_MASS: float = 10 * 27430.998956110157 / (60 * 60)  # mol / m^3
    """Total injected mass in mol/m^3/s.
    
    Note:
        Density value of the mixture at ``T_IN`` and initial pressure in the domain
        is calculated beforehand using the pT flash and hardcoded here.

        Value is divided by 3600 to convert from per hour to per second, and
        multiplied by 10 to obtain a total injection of 10 m^3 per hour.
    
    """

    ### Boundary conditions.
    _T_BC: float = 640.0
    """Temperature at the heated boundary in Kelvin."""

    ### Well configurations.

    _INJECTION_POINTS: list[np.ndarray] = [np.array([15.0, 10.0])]
    """Coordinates of injection wells in meters."""

    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([85.0, 10.0])]
    """Coordinates of production wells in meters."""

    _T_INJECTION: dict[int, float] = {0: _T_IN}
    """Injection temperature per injection well index (starting with 0)."""

    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
    """Production pressure per production well index (starting with 0)."""

    _INJECTED_MASS: dict[str, dict[int, float]] = {}
    """Injected mass per component name and injection well index, calculated from total
    injected mass and injected overall fractions."""
    for n in _COMPONENT_NAMES:
        _INJECTED_MASS[n] = {0: _TOTAL_INJECTED_MASS * _z_IN[n]}


NUM_MONTHS = 15
time_schedule = [i * 30 * pp.DAY for i in range(NUM_MONTHS + 1)]

phase_property_params = {
    "phase_property_params": [1e-4, 1e-2],
}

basalt_ = basalt.copy()
basalt_["permeability"] = 1e-14
material_params = {"solid": pp.SolidConstants(**basalt_)}  # type:ignore

flash_params = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "atol_res": 1e-3,
        "max_iterations": 80,
        "armijo_step_size": 0.95,
        "armijo_decline": 0.495,
        "npipm_penalty_cc": 1,
        "npipm_penalty_neg": 1,
        "npipm_slack_decline": 0.5,
    },
    "global_iteration_stride": 3,
    "fallback_to_iterate": True,
}
flash_params.update(phase_property_params)

# restart_params = {
#     "restart_options": {
#         "restart": False,
#         "pvd_file": pathlib.Path(".\\visualization\\data.pvd").resolve(),
#         "is_mdg_pvd": False,
#         "vtu_files": None,
#         "times_file": pathlib.Path(".\\visualization\\times.json").resolve(),
#     },
# }

meshing_params = {
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size": 5e-1,
        "cell_size_fracture": 5e-1,
    },
}

solver_params = {
    "max_iterations": 30,
    "nl_convergence_tol": 5e-6,
    "nl_convergence_tol_res": 1e-6,
    "apply_schur_complement_reduction": True,
    "linear_solver": "scipy_sparse",
    "nonlinear_solver": NewtonArmijoAndersonSolver,
    "armijo_line_search": True,
    "armijo_line_search_weight": 0.9,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 15,
    "armijo_stop_after_residual_reaches": 1e0,
    "appplyard_chop": 0.3,
    "newton_chop": 1.0,
    "anderson_acceleration": False,
    "anderson_acceleration_depth": 3,
    "anderson_acceleration_constrained": False,
    "anderson_acceleration_regularization_parameter": 1e-3,
    "anderson_start_after_residual_reaches": 1e2,
    "solver_statistics_file_name": "solver_statistics.json",
    "flag_failure_as_diverged": False,
}

MODEL_PARAMS = {
    "equilibrium_specification": (
        pp.compositional.FlashSpec.ph,
        "persistent-variables",
    ),
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "flash_params": flash_params,
    "fractional_flow": False,
    "material_constants": material_params,
    "prepare_simulation": False,
    "enable_buoyancy_effects": False,
    "compile": True,
    "flash_compiler_args": (
        pp.compositional.FlashSpec.pT,
        pp.compositional.FlashSpec.ph,
    ),
}

MODEL_PARAMS.update(phase_property_params)
MODEL_PARAMS.update(meshing_params)
MODEL_PARAMS.update(solver_params)
