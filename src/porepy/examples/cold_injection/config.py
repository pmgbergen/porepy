"""Flow model configuration file for the cold injection examples.

Setting of fluid mixtures, initial and boundary conditions, and other model parameters
such as position of wells and injection rates.

"""

from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp
import porepy.compositional.flash as pf
from porepy.applications.material_values.solid_values import basalt
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfo,
    ConvergenceStatus,
)

from .solver import CFLESolver


class RelaxedCFLEResidualCriterion(pp.ResidualBasedAbsoluteCriterion):
    """Relaxed residual-based convergence criterion applying a separate tolerance
    for isofugacity constraints."""

    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], ConvergenceInfo],
        tol_isofug: float | None = None,
    ):
        super().__init__(tol, metric)
        self.tol_isofug = tol_isofug if tol_isofug is not None else tol

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence using :attr:`tol_isofug` for isofugacity constraints, and
        :attr:`tol` for all other equations.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.
                - value: The value to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of
                the non-linear iteration and information about the convergence check.

        """
        metric_value = self.metric(kwargs["residual"])
        if isinstance(metric_value, dict):
            status = (
                ConvergenceStatus.CONVERGED
                if all(
                    [
                        v < self.tol_isofug if "isofugacity" in k else v < self.tol
                        for k, v in metric_value.items()
                    ]
                )
                else ConvergenceStatus.NOT_CONVERGED
            )
        else:
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.tol
                else ConvergenceStatus.NOT_CONVERGED
            )

        return status, metric_value


class ModelConfig(pp.PorePyModel):
    """Helper class to bundle the model configuration and inherit from
    ``PorePyModel``."""

    ### Domain configurations.

    _DOMAIN_DIMENSIONS: list[float] = [100.0, 20.0, 100.0]
    """Domain dimensions in meters."""

    _INJECTION_POINTS: list[np.ndarray] = [np.array([10.0, 10.0])]
    """Coordinates of injection wells in meters."""

    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([90.0, 10.0])]
    """Coordinates of production wells in meters."""

    _APERTURE_FACTOR_AFTER_TIME: list[tuple[float, pp.number]] = []
    """2-tuples of time-factor pairs, indicating at which time the aperture is
    multiplied with given factor."""

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

    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}
    """Initial overall fractions of fluid components in the whole domain, given as a
    dictionary mapping component names to values."""

    ### Injection and production conditions.

    _p_OUT: float = _p_INIT - 1e6
    """Pressure at production well in Pascals."""

    _T_IN: float = 300.0
    """Temperature of injected fluid in Kelvin."""

    _z_IN: dict[str, float] = {"H2O": 0.9, "CO2": 0.1}
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

    @property
    def _INJECTED_MASS(self) -> dict[str, dict[int, float]]:
        """Injected mass per component name and injection well index, calculated from
        total injected mass and injected overall fractions."""
        d: dict[str, dict[int, float]] = {}
        for n in self._COMPONENT_NAMES:
            d[n] = {}
            for i in range(len(self._INJECTION_POINTS)):
                if self.fluid.num_components == 1:
                    d[n][i] = self._TOTAL_INJECTED_MASS
                else:
                    d[n][i] = self._TOTAL_INJECTED_MASS * self._z_IN[n]

        return d

    @property
    def _T_INJECTION(self) -> dict[int, float]:
        """Injection temperature per injection well index."""
        d = {}
        for i in range(len(self._INJECTION_POINTS)):
            d[i] = self._T_IN
        return d

    @property
    def _p_PRODUCTION(self) -> dict[int, float]:
        """Production pressure per production well index."""
        d = {}
        for i in range(len(self._PRODUCTION_POINTS)):
            d[i] = self._p_OUT
        return d


def get_default_params(
    *,
    base_permeability: float = 1e-14,
) -> tuple[dict, dict]:
    """Get default parametrization for example.

    Some parametrization is supported by this method, others must be set directly when
    obtaining the dict.

    """
    phase_property_params = {
        "phase_property_params": [1e-4, 1e-2, 0.25, 5.0],
        "gen_arg_params": [1e-4, 1e-2, 0.25, 5.0],
    }

    basalt_ = basalt.copy()
    basalt_["permeability"] = base_permeability
    basalt_["specific_heat_capacity"] = 0.0
    material_params = {"solid": pp.SolidConstants(**basalt_)}  # type:ignore

    flash_params = {
        "mode": "parallel",
        "solver": "npipm",
        "solver_params": {
            "atol_res": 1e-3,
            "max_iterations": 80,
        },
        "global_iteration_stride": 3,
        "compile": True,
        "compile_args": (
            pp.compositional.FlashSpec.pT,
            pp.compositional.FlashSpec.ph,
        ),
        "initializer": pf.HeuristicVLInitializer,
    }
    flash_params.update(phase_property_params)

    meshing_params = {
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": 5e-1,
            "cell_size_fracture": 5e-1,
        },
    }

    MODEL_PARAMS = {
        "solver_statistics_file_name": "solver_statistics.json",
        "equilibrium_specification": (
            pp.compositional.FlashSpec.ph,
            "persistent-variables",
        ),
        "linear_solver": "scipy_sparse",
        "apply_schur_complement_reduction": True,
        "flash_params": flash_params,
        "fractional_flow": False,
        "material_constants": material_params,
        "enable_buoyancy_effects": False,
    }

    MODEL_PARAMS.update(phase_property_params)
    MODEL_PARAMS.update(meshing_params)

    SOLVER_PARAMS = {
        "prepare_simulation": False,
        "nonlinear_solver": CFLESolver,
        # "do_armijo_line_search": True,
        # "armijo_line_search_weight": 0.95,
        # "armijo_line_search_incline": 0.2,
        # "armijo_line_search_max_iterations": 15,
        # "armijo_stop_after_residual_reaches": 1e0,
        # "appleyard_chop": 0.3,
        # "newton_chop": 1.0,
        # "do_anderson_acceleration": False,
        # "anderson_acceleration_depth": 3,
        # "anderson_acceleration_constrained": False,
        # "anderson_acceleration_regularization_parameter": 1e-3,
        # "anderson_start_after_residual_reaches": 1e2,
    }
    SOLVER_PARAMS.update(CFLESolver.default_params())

    return MODEL_PARAMS, SOLVER_PARAMS


def get_default_convergence_criteria(
    model: ModelConfig,
    max_iterations: int,
    atol_res: float,
    atol_inc: float,
    atol_res_isofug: float,
    atol_div: float = 1e8,
) -> dict:
    """Returns the default convergence criteria for the CFLE setup."""

    return {
        "max_iterations": int(max_iterations),
        "nl_convergence_criteria": {
            "inc_abs": pp.IncrementBasedAbsoluteCriterion(
                tol=atol_inc, metric=pp.VariableBasedLebesgueMetric(model)
            ),
            "res_abs": RelaxedCFLEResidualCriterion(
                tol=atol_res,
                metric=pp.EquationBasedLebesgueMetric(model),
                tol_isofug=atol_res_isofug,
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=max_iterations),
            "inc_nan": pp.IncrementBasedNanCriterion(),
            "res_nan": pp.ResidualBasedNanCriterion(),
            "res_div": pp.ResidualBasedAbsoluteDivergenceCriterion(
                tol=atol_div, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
    }
