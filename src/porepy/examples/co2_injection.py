"""Example for 2-phase, 2-component flow using equilibrium calculations.

Simulates the injection of CO2 into an initially water-saturated 2D domain, using a
mD model with points as wells.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

"""

from __future__ import annotations

# GENERAL MODEL CONFIGURATION

FRACTIONAL_FLOW: bool = True
"""Use the fractional flow formulation without upwinding in the diffusive fluxes."""
EQUILIBRIUM_CONDITION: str = "unified-p-h"
"""Define the equilibrium condition to determin the flash type used in the solution
procedure."""
EXPORT_SCHEDULED_TIME_ONLY: bool = False
"""Exports all  time steps produced by the time stepping algorithm, otherwise only
the scheduled times."""
ANALYSIS_MODE: bool = False
"""Uses the Analysis mixin to store some additional values to inspect convergence."""
DISABLE_COMPILATION: bool = False
"""For disabling numba compilation and faster start of simulation."""


import json
import logging
import pathlib
import time
import warnings
from collections import deque
from typing import Any, Callable, Literal, Optional, Sequence, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

if DISABLE_COMPILATION:
    import os

    os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
import porepy.models.compositional_flow as cf
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.applications.test_utils.models import create_local_model_class
from porepy.fracs.wells_3d import _add_interface

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class _FlowConfiguration(pp.PorePyModel):
    """Helper class to bundle the configuration of pressure, temperature and mass
    for in- and outflow."""

    # Mesh size.
    _h_MESH: float = 2.0
    # 4.        308 cells
    # 2.        1204 cells
    # 1.        4636 cells
    # 5e-1      18,464 cells
    # 2.5e-1    73,748 cells
    # 1e-1      462,340 cells

    # Initial values.
    _p_INIT: float = 20e6
    _T_INIT: float = 450.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    # In- and outflow values.
    _T_HEATED: float = 640.0
    _T_IN: float = 300.0
    _z_IN: dict[str, float] = {"H2O": 0.9, "CO2": 0.1}

    _p_OUT: float = _p_INIT - 1e6

    # Value obtained from a p-T flash with values defined above.
    # Divide by 3600 to obtain an injection of unit per hour
    # Multiplied by some number for how many units per hour
    _TOTAL_INJECTED_MASS: float = 10 * 27430.998956110157 / (60 * 60)  # mol / m^3
    # _TOTAL_INJECTED_MASS: float = 10 * 21202.860945350567 / (60 * 60)  # mol / m^3

    # Injection model configuration
    _T_INJECTION: dict[int, float] = {0: _T_IN}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}

    _INJECTED_MASS: dict[str, dict[int, float]] = {
        "H2O": {0: _TOTAL_INJECTED_MASS * _z_IN["H2O"]},
        "CO2": {0: _TOTAL_INJECTED_MASS * _z_IN["CO2"]},
    }

    # Coordinates of injection and production wells in meters
    _INJECTION_POINTS: list[np.ndarray] = [np.array([15.0, 10.0])]
    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([85.0, 10.0])]


class FluidMixture(pp.PorePyModel):
    """2-component, 2-phase fluid with H2O and CO2, and a liquid and gas phase."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.FluidComponent]
    ) -> Sequence[
        tuple[pp.compositional.PhysicalState, str, pp.compositional.EquationOfState]
    ]:
        import porepy.compositional.peng_robinson as pr

        eos = pr.PengRobinsonCompiler(
            components, [pr.h_ideal_H2O, pr.h_ideal_CO2], pr.get_bip_matrix(components)
        )
        return [
            (pp.compositional.PhysicalState.liquid, "L", eos),
            (pp.compositional.PhysicalState.gas, "G", eos),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        return [self.pressure, self.temperature] + [  # type:ignore[return-value]
            phase.extended_fraction_of[comp] for comp in phase
        ]


class SolutionStrategy(cfle.SolutionStrategyCFLE):
    """Provides some pre- and post-processing for flash methods."""

    flash: pp.compositional.Flash

    pressure_variable: str
    temperature_variable: str
    enthalpy_variable: str
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_fraction: Callable[[pp.Component | pp.Phase], bool]
    secondary_variables: list[str]
    secondary_equations: list[str]
    fraction_in_phase_variables: list[str]
    phase_fraction_variables: list[str]
    isofugacity_constraint_for_component_in_phase: Callable[
        [pp.Component, pp.Phase, Sequence[pp.Grid]],
        pp.ad.Operator,
    ]

    _block_row_permutation: np.ndarray
    _block_column_permutation: np.ndarray
    _block_size: int

    _num_flash_iter: np.ndarray

    def __init__(self, params: dict | None = None):
        super().__init__(params)  # type:ignore[safe-super]

        self._residual_norm_history: deque[float] = deque(maxlen=4)
        self._increment_norm_history: deque[float] = deque(maxlen=3)

        self._cum_flash_iter_per_grid: dict[pp.Grid, list[np.ndarray]] = {}
        self._global_iter_per_time: dict[float, tuple[int, ...]] = {}
        self._time_tracker: dict[str, list[float]] = {
            "flash": [],
            "assembly": [],
            "linsolve": [],
        }

    def data_to_export(self):
        data: list = super().data_to_export()

        for sd in self.mdg.subdomains():
            if sd in self._cum_flash_iter_per_grid:
                ni = self._cum_flash_iter_per_grid[sd]
                n = np.array(sum(ni), dtype=int)
            else:
                n = np.zeros(sd.num_cells, dtype=int)

            data.append((sd, "cumulative flash iterations", n))

        return data

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()  # type:ignore[misc]
        self._residual_norm_history.clear()
        self._increment_norm_history.clear()
        model.nonlinear_solver_statistics.num_iteration_armijo = 0  # type:ignore[attr-defined]
        self._cum_flash_iter_per_grid.clear()

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Flags the time step as diverged, if there is a nan in the residual."""
        status = super().check_convergence(  # type:ignore[misc]
            nonlinear_increment, residual, reference_residual, nl_params
        )
        if residual is not None:
            if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
                status = (False, True)

        # Convergence check for individual norms, if the global residual is not yet
        # small enough.
        tol_res = float(self.params["nl_convergence_tol_res"])
        tol_inc = float(self.params["nl_convergence_tol"])
        # Relaxed tolerance for quantities loosing their physical meaning in the unified
        # setting when a phase dissappears.
        # Relaxed tolerance for partial fractions and isofugacity constraints.
        # Saying variations have to drop below 1% (not significant enough to change the
        # state)
        tol_relaxed = 1e-2

        # Additional tracking for analysis
        res_norm_per_eq = {}
        incr_norm_per_var = {}
        condition_numbers: dict[str, float] = {}

        if status == (False, False):
            residuals_converged: list[bool] = []
            increments_converged: list[bool] = []

            # First, perform standard check for all equations except isofugacity
            # constraints, and all variables except partial fractions
            for name, eq in self.equation_system.equations.items():
                rn = self.compute_residual_norm(
                    cast(np.ndarray, self.equation_system.evaluate(eq)),
                    reference_residual,
                )
                res_norm_per_eq[name] = rn

                if "isofugacity" not in name:
                    residuals_converged.append(rn < tol_res)
                else:
                    residuals_converged.append(rn < tol_relaxed)

            partial_frac_vars = self.fraction_in_phase_variables
            for var in self.equation_system.variables:
                rn = self.compute_nonlinear_increment_norm(
                    nonlinear_increment[self.equation_system.dofs_of([var])]
                )
                if var.name not in incr_norm_per_var:
                    incr_norm_per_var[var.name] = rn
                else:
                    incr_norm_per_var[var.name] = np.sqrt(
                        incr_norm_per_var[var.name] ** 2 + rn**2
                    )
                if var.name not in partial_frac_vars:
                    increments_converged.append(rn < tol_inc)
                else:
                    increments_converged.append(rn < tol_relaxed)

            status = (
                all(residuals_converged) and all(increments_converged),
                False,
            )
            if status[0]:
                print("\nConverged with relaxed, unified CF criteria.\n")

        # if ANALYSIS_MODE:
        #     assert isinstance(
        #         self.nonlinear_solver_statistics, ExtendedSolverStatistics
        #     )
        #     self.nonlinear_solver_statistics.extended_log(
        #         self.time_manager.time,
        #         self.time_manager.dt,
        #         res_norm_per_equation=res_norm_per_eq,
        #         incr_norm_per_variable=incr_norm_per_var,
        #         condition_numbers=condition_numbers,
        #     )

        # Keeping residual/ increment norm history and checking for stationary points.
        self._residual_norm_history.append(
            self.compute_residual_norm(residual, reference_residual)
        )
        self._increment_norm_history.append(
            self.compute_nonlinear_increment_norm(nonlinear_increment)
        )

        if len(self._residual_norm_history) == self._residual_norm_history.maxlen:
            residual_stationary = (
                np.allclose(
                    self._residual_norm_history,
                    self._residual_norm_history[-1],
                    rtol=0.0,
                    atol=np.min((tol_res, 1e-6)),
                )
                and tol_res != np.inf
            )
            increment_stationary = (
                np.allclose(
                    self._increment_norm_history,
                    self._increment_norm_history[-1],
                    rtol=0.0,
                    atol=np.min((tol_inc, 1e-6)),
                )
                and tol_inc != np.inf
            )
            if residual_stationary and increment_stationary and not status[0]:
                print("Detected stationary point. Flagging as diverged.")
                status = (False, True)

        return status

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        if residual is None:
            return np.nan
        residual_norm = np.linalg.norm(residual)
        return float(residual_norm)

    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        global_flash_iter = 0
        for ni in self._cum_flash_iter_per_grid.values():
            n = sum(ni).sum()
            global_flash_iter += int(n)
        self._global_iter_per_time[self.time_manager.time] = (
            self.nonlinear_solver_statistics.num_iteration,
            model.nonlinear_solver_statistics.num_iteration_armijo,
            global_flash_iter,
        )

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """Performing pT flash in injection wells, because T is fixed there."""
        start = time.time()
        stride = int(self.params["flash_params"].get("global_iteration_stride", 3))
        for sd in self.mdg.subdomains():
            if sd not in self._cum_flash_iter_per_grid:
                self._cum_flash_iter_per_grid[sd] = []
            if "injection_well" in sd.tags:
                equ_spec = {
                    "p": self.equation_system.evaluate(
                        self.pressure([sd]), state=state
                    ),
                    "T": self.equation_system.evaluate(
                        self.temperature([sd]), state=state
                    ),
                }
                self.local_equilibrium(sd, state=state, equilibrium_specs=equ_spec)  # type:ignore
                self._cum_flash_iter_per_grid[sd].append(self._num_flash_iter)
            elif self.nonlinear_solver_statistics.num_iteration % stride == 0:
                self.local_equilibrium(sd, state=state)  # type:ignore
                self._cum_flash_iter_per_grid[sd].append(self._num_flash_iter)
            else:
                cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
                    self,  # type:ignore[arg-type]
                    state,
                )
        self._time_tracker["flash"].append(time.time() - start)

    def update_dependent_quantities(self):
        subdomains = self.mdg.subdomains()

        fluid_state: pp.compositional.FluidProperties = super().current_fluid_state(  # type:ignore
            subdomains
        )

        # Normalizing fractions in case of overshooting
        eps = 1e-7  # binding overall fractions away from zero
        z = fluid_state.z
        z[z >= 1.0] = 1.0 - eps
        z[z <= 0.0] = 0.0 + eps
        z = pp.compositional.normalize_rows(z.T).T

        s = fluid_state.sat
        s[s >= 1.0] = 1.0
        s[s <= 0.0] = 0.0
        s = pp.compositional.normalize_rows(s.T).T

        y = fluid_state.y
        y[y >= 1.0] = 1.0
        y[y <= 0.0] = 0.0
        y = pp.compositional.normalize_rows(y.T).T

        for z_i, comp in zip(z, self.fluid.components):
            if self.has_independent_fraction(comp):
                self.equation_system.set_variable_values(
                    z_i,
                    [comp.fraction(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )
        for j, data in enumerate(zip(s, y, self.fluid.phases)):
            s_j, y_j, phase = data
            if self.has_independent_saturation(phase):
                self.equation_system.set_variable_values(
                    s_j,
                    [phase.saturation(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )
            if self.has_independent_fraction(phase):
                self.equation_system.set_variable_values(
                    y_j,
                    [phase.fraction(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )

            phase_state = fluid_state.phases[j]
            x_sum = phase_state.x.sum(axis=0)
            idx = x_sum > 1.0
            if np.any(idx):
                phase_state.x[:, idx] = pp.compositional.normalize_rows(
                    phase_state.x[:, idx].T
                ).T
            idx = phase_state.x < 0
            if np.any(idx):
                phase_state.x[idx] = 0.0
            idx = phase_state.x > 1.0
            if np.any(idx):
                phase_state.x[idx] = 1.0
            for i, comp in enumerate(self.fluid.components):
                if self.has_independent_partial_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        phase_state.x[i],
                        [phase.extended_fraction_of[comp](subdomains)],
                        iterate_index=0,
                    )

        super().update_dependent_quantities()

    def current_fluid_state(
        self,
        subdomains: Sequence[pp.Grid] | pp.Grid,
        state: Optional[np.ndarray] = None,
    ) -> pp.compositional.FluidProperties:
        """Method to pre-process the evaluated fractions. Normalizes the extended
        fractions where they violate the unity constraint."""
        fluid_state: pp.compositional.FluidProperties = super().current_fluid_state(  # type:ignore
            subdomains, state
        )

        if isinstance(subdomains, pp.Grid):
            subdomains = [subdomains]

        for phase in fluid_state.phases:
            x_sum = phase.x.sum(axis=0)
            idx = x_sum > 1.0
            if np.any(idx):
                phase.x[:, idx] = pp.compositional.normalize_rows(phase.x[:, idx].T).T
            idx = phase.x < 0
            if np.any(idx):
                phase.x[idx] = 0.0
            idx = phase.x > 1.0
            if np.any(idx):
                phase.x[idx] = 1.0

        return fluid_state

    def postprocess_flash(
        self,
        sd: pp.Grid,
        equilibrium_results: tuple[pp.compositional.FluidProperties, np.ndarray],
        state: Optional[np.ndarray] = None,
    ) -> pp.compositional.FluidProperties:
        """A post-processing where the flash is again attempted where not successful.

        1. Where not successful, it attempts the flash from scratch, with
           initialization procedure.
           If the new attempt did not reach the desired precision (max iter), it
           treats it as success.
        2. Where still not successful, it falls back to the values of the previous
           iterate.

        """
        fluid_state, success = equilibrium_results

        failure = success > 0
        equilibrium_condition = str(
            pp.compositional.get_local_equilibrium_condition(self)
        )
        if np.any(failure):
            logger.debug(
                f"Flash from iterate state failed in {failure.sum()} cells on grid"
                + f" {sd.id}. Performing full flash .."
            )
            z = [
                self.equation_system.evaluate(comp.fraction([sd]), state=state)[failure]  # type:ignore[index]
                for comp in self.fluid.components
            ]
            p = self.equation_system.evaluate(self.pressure([sd]), state=state)[failure]  # type:ignore[index]
            h = self.equation_system.evaluate(self.enthalpy([sd]), state=state)[failure]  # type:ignore[index]
            T = self.equation_system.evaluate(self.temperature([sd]), state=state)[
                failure
            ]  # type:ignore[index]

            # no initial guess, and this model uses only p-h flash.
            flash_kwargs: dict[str, Any] = {
                "z": z,
                "p": p,
                "params": self.params.get("flash_params"),
            }

            if "p-h" in equilibrium_condition:
                flash_kwargs["h"] = h
            elif "p-T" in equilibrium_condition:
                flash_kwargs["T"] = T

            sub_state, sub_success, num_iter = self.flash.flash(**flash_kwargs)

            ni = np.zeros_like(success)
            ni[failure] = num_iter
            self._num_flash_iter += ni

            fallback = sub_success > 1
            max_iter_reached = sub_success == 1
            # treat max iter reached as success, and hope for the best in the PDE iter
            sub_success[max_iter_reached] = 0
            # fall back to previous iter values
            if np.any(fallback):
                logger.debug(
                    f"Full flash failed in {fallback.sum()} cells on grid "
                    + f"{sd.id}. Falling back to previous iterate state."
                )
                sub_state = self.fall_back(sd, sub_state, failure, fallback)
                sub_success[fallback] = 0
            # update parent state with sub state values
            success[failure] = sub_success
            fluid_state.T[failure] = sub_state.T
            fluid_state.h[failure] = sub_state.h
            fluid_state.rho[failure] = sub_state.rho

            for j in range(len(fluid_state.phases)):
                fluid_state.sat[j][failure] = sub_state.sat[j]
                fluid_state.y[j][failure] = sub_state.y[j]

                fluid_state.phases[j].x[:, failure] = sub_state.phases[j].x

                fluid_state.phases[j].rho[failure] = sub_state.phases[j].rho
                fluid_state.phases[j].h[failure] = sub_state.phases[j].h
                fluid_state.phases[j].mu[failure] = sub_state.phases[j].mu
                fluid_state.phases[j].kappa[failure] = sub_state.phases[j].kappa

                fluid_state.phases[j].drho[:, failure] = sub_state.phases[j].drho
                fluid_state.phases[j].dh[:, failure] = sub_state.phases[j].dh
                fluid_state.phases[j].dmu[:, failure] = sub_state.phases[j].dmu
                fluid_state.phases[j].dkappa[:, failure] = sub_state.phases[j].dkappa

                fluid_state.phases[j].phis[:, failure] = sub_state.phases[j].phis
                fluid_state.phases[j].dphis[:, :, failure] = sub_state.phases[j].dphis

        # Parent method performs a check that everything is successful.
        return super().postprocess_flash(sd, (fluid_state, success), state=state)  # type:ignore

    @no_type_check
    def fall_back(
        self,
        grid: pp.Grid,
        sub_state: pp.compositional.FluidProperties,
        failed_idx: np.ndarray,
        fallback_idx: np.ndarray,
    ) -> pp.compositional.FluidProperties:
        """Falls back to previous iterate values for the fluid state, on given
        grid."""
        sds = [grid]
        data = self.mdg.subdomain_data(grid)
        equilibrium_condition = str(
            pp.compositional.get_local_equilibrium_condition(self)
        )
        if "T" not in equilibrium_condition:
            sub_state.T[fallback_idx] = pp.get_solution_values(
                self.temperature_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "h" not in equilibrium_condition:
            sub_state.h[fallback_idx] = pp.get_solution_values(
                self.pressure_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "p" not in equilibrium_condition:
            sub_state.p[fallback_idx] = pp.get_solution_values(
                self.enthalpy_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]

        for j, phase in enumerate(self.fluid.phases):
            if j > 0:
                sub_state.sat[j][fallback_idx] = pp.get_solution_values(
                    phase.saturation(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]
                sub_state.y[j][fallback_idx] = pp.get_solution_values(
                    phase.fraction(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]

            # mypy: ignore[union-attr]
            sub_state.phases[j].rho[fallback_idx] = pp.get_solution_values(
                phase.density.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].h[fallback_idx] = pp.get_solution_values(
                phase.specific_enthalpy.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].mu[fallback_idx] = pp.get_solution_values(
                phase.viscosity.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].kappa[fallback_idx] = pp.get_solution_values(
                phase.thermal_conductivity.name, data, iterate_index=0
            )[failed_idx][fallback_idx]

            sub_state.phases[j].drho[:, fallback_idx] = pp.get_solution_values(
                phase.density._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dh[:, fallback_idx] = pp.get_solution_values(
                phase.specific_enthalpy._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dmu[:, fallback_idx] = pp.get_solution_values(
                phase.viscosity._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dkappa[:, fallback_idx] = pp.get_solution_values(
                phase.thermal_conductivity._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]

            for i, comp in enumerate(phase):
                sub_state.phases[j].x[i, fallback_idx] = pp.get_solution_values(
                    phase.extended_fraction_of[comp](sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]

                sub_state.phases[j].phis[i, fallback_idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp].name, data, iterate_index=0
                )[failed_idx][fallback_idx]
                # NOTE numpy does some weird transpositions when dealing with 3D arrays
                dphi = sub_state.phases[j].dphis[i]
                dphi[:, fallback_idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp]._name_derivatives,
                    data,
                    iterate_index=0,
                )[:, failed_idx][:, fallback_idx]
                sub_state.phases[j].dphis[i, :, :] = dphi

        # reference phase fractions and saturations must be computed, since not stored
        sub_state.y[self.fluid.reference_phase_index, :] = 1 - np.sum(
            sub_state.y[1:, :], axis=0
        )
        sub_state.sat[self.fluid.reference_phase_index, :] = 1 - np.sum(
            sub_state.sat[1:, :], axis=0
        )

        return sub_state

    def assemble_linear_system(self) -> None:
        start = time.time()
        super().assemble_linear_system()
        self._time_tracker["assembly"].append(time.time() - start)

    def solve_linear_system(self) -> np.ndarray:
        start = time.time()
        sol = super().solve_linear_system()
        self._time_tracker["linsolve"].append(time.time() - start)
        return sol


class PointWells2D(_FlowConfiguration):
    """2D matrix with point grids as injection and production points.

    Alternative for the ``WellNetwork3d`` in 2d.

    """

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": self.units.convert_units(100.0, "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(20.0, "m"),
            }
        )

    def set_geometry(self):
        super().set_geometry()

        for i, injection_point in enumerate(self._INJECTION_POINTS):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._PRODUCTION_POINTS):
            self._add_well(production_point, i, "production")

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Helper method to construct a well in 2D as a PointGrid and add respective
        interface.

        Parameters:
            point: Point in space representing well.
            well_index: Assigned number for well of type ``well_type``.
            well_type: Label to add a tag to the point grid labelng as injector or
            producer.

        """
        matrix = self.mdg.subdomains(dim=self.nd)[0]
        assert isinstance(point, np.ndarray)
        p: np.ndarray
        if point.shape == (2,):
            p = np.zeros(3)
            p[:2] = point
        elif point.shape == (3,):
            p = point
        else:
            raise ValueError(
                f"Point for well {(well_type, well_index)} must be 1D array of length "
                + "2 or 3."
            )

        sd_0d = pp.PointGrid(self.units.convert_units(p, "m"))
        # Tag for processing of equations.
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        self.mdg.add_subdomains(sd_0d)

        # Motivated by wells_3d.py#L828
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )

        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)


class AdjustedWellModel2D(_FlowConfiguration):
    """Adjustment of a 2D model which has wells modelled as point grids.

    Two types of point grids are expected: ``'injection_well'`` and
    ``'production_well'``.

    In the injection well, mass is expected to enter the system (mass per time)
    at a given temperature (fixed value).

    At the production well, a given pressure value is required.

    In injection wells, the energy balance is replaced by a simple constraint
    ``T - T_injection = 0``.
    In production wells, the fluid mass balance (pressure equation) is replaced by
    ``p - p_production = 0``.

    In injection wells, a given inflow per fluid component is expected, which enter the
    system as a source term in the respective point grid.

    In production wells, all DOFs except pressure, and all equations are removed.
    The outflow of mass and energy can be computed with respective well fluxes.
    An exact composition of the fluid at the production well can be obtained from the
    values in the matrix grid, using the cell which was used to construct the mortar
    grid for the production well.

    In this sense, production wells and their respective grids are only used to mimic
    an internal, free-flow boundary.

    Important:
        This is a mixin modifying equations and variables. It must be mixed in
        above all other variable and equation mixins.

    Note:
        In injection wells, only the injected mass is defined, not the injected energy.
        This is due to the energy balance equation being replaced by an temperature
        constraint. This can cause trouble if there is temporarily some backflow in
        the production wells due to pressure drop around the wells. TODO

    """

    compute_residual_norm: Callable[[Optional[np.ndarray], np.ndarray], float]
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    pressure_variable: str
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def _filter_wells(
        self,
        subdomains: Sequence[pp.Grid],
        well_type: Literal["production", "injection"],
    ) -> tuple[list[pp.Grid], list[pp.Grid]]:
        """Helper method to return the partitioning of subdomains into wells of defined
        ``well_type`` and other grids.

        Parameters:
            subdomains: A list of subdomains.
            well_type: Well type to filter out (injector or producer).

        Returns:
            A 2-tuple containing

            1. All 0D grids tagged as wells of type ``well_type``.
            2. All other grids found in ``subdomains``.

        """
        tag = f"{well_type}_well"
        wells = [sd for sd in subdomains if sd.dim == 0 and tag in sd.tags]
        other_sds = [sd for sd in subdomains if sd not in wells]
        return wells, other_sds

    # Adjusting PDEs
    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""
        _, no_production_wells = self._filter_wells(subdomains, "production")
        eq: pp.ad.Operator = super().mass_balance_equation(no_production_wells)  # type:ignore[misc]
        name = eq.name
        return eq

        volume_stabilization = self.fluid.density(
            no_production_wells
        ) * pp.ad.sum_operator_list(
            [
                phase.fraction(no_production_wells) / phase.density(no_production_wells)
                for phase in self.fluid.phases
            ],
            "fluid_specific_volume",
        ) - self.porosity(no_production_wells)

        volume_stabilization = self.volume_integral(
            volume_stabilization, no_production_wells, dim=1
        )
        volume_stabilization = pp.ad.time_derivatives.dt(
            volume_stabilization, self.ad_time_step
        )
        eq = eq + volume_stabilization
        eq.set_name(name)
        return eq

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""
        _, no_injection_wells = self._filter_wells(subdomains, "injection")
        return super().energy_balance_equation(no_injection_wells)  # type:ignore[misc]

    # Introducing pressure and temperature constraint at production and injection.
    def set_equations(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        injection_wells, _ = self._filter_wells(subdomains, "injection")
        production_wells, _ = self._filter_wells(subdomains, "production")

        p_constraint = self.pressure_constraint_at_production_wells(production_wells)
        self.equation_system.set_equation(p_constraint, production_wells, {"cells": 1})
        T_constraint = self.temperature_constraint_at_injection_wells(injection_wells)
        self.equation_system.set_equation(T_constraint, injection_wells, {"cells": 1})

    def pressure_constraint_at_production_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns an constraint of form :math:`p - p_p=0` which replaces the
        pressure equation in production wells.

        Parameters:
            subdomains: A list of grids (tagged as production wells).

        Returns:
            The left-hand side of above equation.

        """
        p_production = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells)
                    * self._p_PRODUCTION[sd.tags["production_well"]]
                    for sd in subdomains
                ]
            ),
            name="production_pressure",
        )

        pressure_constraint_production = self.pressure(subdomains) - p_production
        pressure_constraint_production.set_name("production_pressure_constraint")
        return pressure_constraint_production

    def temperature_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        temperature at production wells."""
        T_injection = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells) * self._T_INJECTION[sd.tags["injection_well"]]
                    for sd in subdomains
                ]
            ),
            name="injection_temperature",
        )

        temperature_constraint_injection = self.temperature(subdomains) - T_injection
        temperature_constraint_injection.set_name("injection_temperature_constraint")
        return temperature_constraint_injection

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Augments the source term in the pressure equation to account for the mass
        injected through injection wells."""
        source: pp.ad.Operator = super().fluid_source(subdomains)  # type:ignore[misc]

        injection_wells, _ = self._filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        injected_mass: pp.ad.Operator = pp.ad.sum_operator_list(
            [
                self.volume_integral(
                    self.injected_component_mass(comp, injection_wells),
                    injection_wells,
                    1,
                )
                for comp in self.fluid.components
            ],
            "total_injected_fluid_mass",
        )

        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells) @ injected_mass
        )

        return source

    def component_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Adjusted source term for a component's mass balance equation to account
        for the injected mass in the injection wells, and removing all mass in the
        production wells."""
        source: pp.ad.Operator = super().component_source(component, subdomains)  # type:ignore[misc]

        injection_wells, _ = self._filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        injected_mass = self.volume_integral(
            self.injected_component_mass(component, injection_wells),
            injection_wells,
            1,
        )

        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells) @ injected_mass
        )

        # Removing source term in production well, mimicing outflow of mass.
        production_wells, _ = self._filter_wells(subdomains, "production")
        source -= subdomain_projections.cell_prolongation(production_wells) @ (
            subdomain_projections.cell_restriction(production_wells) @ source
        )

        return source

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Adjusted energy source term removing all energy in the production wells."""
        source = super().energy_source(subdomains)  # type:ignore[misc]

        # Removing source term in production well, mimicing outflow of energy.
        production_wells, _ = self._filter_wells(subdomains, "production")
        _, no_injection_wells = self._filter_wells(subdomains, "injection")
        subdomain_projections = pp.ad.SubdomainProjections(no_injection_wells)
        source -= subdomain_projections.cell_prolongation(production_wells) @ (
            subdomain_projections.cell_restriction(production_wells) @ source
        )
        return source

    def injected_component_mass(
        self, component: pp.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the injected mass of a fluid component in [kg m^-3 s^-1] (or moles).

        This is used as a source term on balance equations in injection wells. Note that
        the volume integral is not performed here, but in the respective method
        assembling the source term for a balance equation.

        Parameters:
            component: A fluid component.
            subdomains: A list of grids (grids tagged as ``'injection_wells'``)

        Returns:
            The source term wrapped as a dens AD array.
        """
        injected_mass: list[np.ndarray] = []
        for sd in subdomains:
            assert "injection_well" in sd.tags, (
                f"Grid {sd.id} not tagged as injection well."
            )
            injected_mass.append(
                np.ones(sd.num_cells)
                * self._INJECTED_MASS[component.name][sd.tags["injection_well"]]
            )

        if injected_mass:
            source = np.hstack(injected_mass)
        else:
            source = np.zeros((0,))

        return pp.ad.DenseArray(source, f"injected_mass_density_{component.name}")


class InitialConditions(_FlowConfiguration):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        # f = lambda x: self._p_IN + x /10 * (self._p_OUT - self._p_IN)
        # vals = np.array(list(map(f, sd.cell_centers[0])))
        # return vals
        p = np.ones(sd.num_cells)
        if sd.dim == 0 and "production_well" in sd.tags:
            return p * self._p_PRODUCTION[sd.tags["production_well"]]
        else:
            return p * self._p_INIT

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        T = np.ones(sd.num_cells)
        if sd.dim == 0 and "injection_well" in sd.tags:
            return T * self._T_INJECTION[sd.tags["injection_well"]]
        else:
            return T * self._T_INIT

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        return np.ones(sd.num_cells) * self._z_INIT[component.name]

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        return np.zeros(sd.num_cells)


class BoundaryConditions(_FlowConfiguration):
    """Boundary conditions defining a ``left to right`` flow in the matrix (2D)

    Mass flux:

    - No flux conditions on top and bottom
    - Dirichlet data (pressure, temperatyre and composition) on left and right faces

    Heat flux:

    - No flux on left, top, right
    - heated bottom side (given by temperature as a Dirichlet-type BC)

    Trivial Neumann conditions for fractures.

    """

    def _central_stripe(self, sd: pp.Grid) -> tuple[float, float]:
        """Returns the left and right boundary of the central, vertical stripe of the
        matrix, which represents roughly a third of the area.

        The x-axis is used to determin what is a third.

        """

        x_min = float(sd.cell_centers[0].min())
        x_max = float(sd.cell_centers[0].max())

        c = (x_min + x_max) / 2.0
        s = (x_max - x_min) / 6.0

        return c - s, c + s

    def _heated_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.south] = True
        left, right = self._central_stripe(sd)
        heated &= sd.face_centers[0] >= left
        heated &= sd.face_centers[0] <= right

        return heated

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            heated = self._heated_boundary_faces(sd)
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if cf.is_fractional_flow(self):
            return self.bc_type_fourier_flux(sd)
        else:
            return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == 2:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            vals[heated_faces] = self._p_INIT

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == 2:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            vals[heated_faces] = self._T_HEATED

        return vals

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == 2:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            vals[heated_faces] = self._z_INIT[component.name]

        return vals


class NewtonArmijoSolver(pp.NewtonSolver):
    """Newton solver with Armijo line search.

    The residual objective function is tailored to models where phase properties are
    assumed to be surrogate factories and require an update before evaluating the
    objective function.

    """

    def iteration(self, model: pp.PorePyModel):
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""
        dx = super().iteration(model)
        alpha = self.armijo_line_search(model, dx)
        return alpha * dx

    def armijo_line_search(self, model: pp.PorePyModel, dx: np.ndarray) -> float:
        """Performs the Armijo line search."""
        if not self.params.get("armijo_line_search", False):
            return 1.0

        rho = float(self.params.get("armijo_line_search_weight", 0.9))
        kappa = float(self.params.get("armijo_line_search_incline", 0.4))
        N = int(self.params.get("armijo_line_search_max_iterations", 50))

        pot_0 = self.armijo_objective_function(model, dx, 0.0)
        rho_i = rho
        n = 0

        for i in range(N):
            n = i
            rho_i = rho**i

            pot_i = self.armijo_objective_function(model, dx, rho_i)
            if pot_i <= (1 - 2 * kappa * rho_i) * pot_0:
                break

        model.nonlinear_solver_statistics.num_iteration_armijo += n  # type:ignore[attr-defined]
        logger.info(f"Armijo line search determined weight: {rho_i} ({n})")
        return rho_i

    def armijo_objective_function(
        self, model: pp.PorePyModel, dx: np.ndarray, weight: float
    ) -> float:
        """The objective function to be minimized is the norm of the residual squared
        and divided by 2."""
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        state = x_0 + weight * dx
        # model.update_thermodynamic_properties_of_phases(state)
        cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
            model,  # type:ignore[arg-type]
            state,
        )
        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return float(np.dot(residual, residual) / 2)


class Permeability(pp.PorePyModel):
    """Custom permeability with a slightly lower permability around the wells and a
    constant permeability of 1 in the wells."""

    total_mass_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return pp.constitutive_laws.DimensionDependentPermeability.permeability(
            self,  # type:ignore[arg-type]
            subdomains,
        )

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Matrix permeability with a higher permeability with factor 1e3 around the
        wells in the matrix."""

        assert len(subdomains) <= 1, "Expecting at most 1 grid as matrix."

        K_vals: list[np.ndarray] = [np.zeros((0,))]

        for sd in subdomains:
            k = np.ones(sd.num_cells)
            if sd.dim == self.nd:
                k *= self.solid.permeability
                l, r = BoundaryConditions._central_stripe(self, sd)  # type:ignore[arg-type]
                k[sd.cell_centers[0] < l] *= 1e1
                k[sd.cell_centers[0] > r] *= 1e1
            K_vals.append(k)

        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            np.concatenate(K_vals), name="base_matrix_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("matrix_permeability")
        return K

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.intersection_permeability(subdomains)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Base permeability of wells is 1."""
        N = sum([sd.num_cells for sd in subdomains])
        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            1.0, size=N, name="base_well_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("well_permeability")
        return K


# region Model class assembly
# mypy: disable-error-code="type-abstract"
model_class: type[pp.PorePyModel]

if FRACTIONAL_FLOW:
    model_class = create_local_model_class(
        cfle.EnthalpyBasedCFFLETemplate,
        [pp.constitutive_laws.FouriersLawAd, pp.constitutive_laws.DarcysLawAd],
    )
else:
    model_class = cfle.EnthalpyBasedCFLETemplate

model_class = create_local_model_class(
    model_class,
    [
        SolutionStrategy,
        BoundaryConditions,
        InitialConditions,
        FluidMixture,
        AdjustedWellModel2D,
        PointWells2D,
        Permeability,
    ],
)

# endregion

# region Model parametrization

time_schedule = [i * 30 * pp.DAY for i in range(25)]

max_iterations = 40 if FRACTIONAL_FLOW else 30
newton_tol = 1e-6
newton_tol_increment = 5e-6

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=10 * pp.MINUTE,
    dt_min_max=(pp.MINUTE, 30 * pp.DAY),
    iter_max=max_iterations,
    iter_optimal_range=(20, 30) if FRACTIONAL_FLOW else (12, 22),
    iter_relax_factors=(0.8, 2.0),
    recomp_factor=0.6,
    recomp_max=15,
    print_info=True,
    rtol=0.0,
)

phase_property_params = {
    "phase_property_params": [0.0],
}

basalt_ = basalt.copy()
basalt_["permeability"] = 1e-14
material_params = {"solid": pp.SolidConstants(**basalt_)}  # type:ignore[arg-type]

flash_params: dict[Any, Any] = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "tolerance": 1e-8,
        "max_iterations": 80,  # 150
        "armijo_rho": 0.99,
        "armijo_kappa": 0.4,
        "armijo_max_iterations": 50 if "p-T" in EQUILIBRIUM_CONDITION else 30,
        "npipm_u1": 10,
        "npipm_u2": 10,
        "npipm_eta": 0.5,
    },
    "global_iteration_stride": 2 if FRACTIONAL_FLOW else 3,
}
flash_params.update(phase_property_params)

restart_params = {
    "restart_options": {
        "restart": False,
        "pvd_file": pathlib.Path(".\\visualization\\data.pvd").resolve(),
        "is_mdg_pvd": False,
        "vtu_files": None,
        "times_file": pathlib.Path(".\\visualization\\times.json").resolve(),
    },
}

meshing_params = {
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size": _FlowConfiguration._h_MESH,
        "cell_size_fracture": 5e-1,
    },
}

solver_params = {
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "linear_solver": "scipy_sparse",
    "nonlinear_solver": NewtonArmijoSolver,
    "armijo_line_search": True,
    "armijo_line_search_weight": 0.95,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 10,
    "solver_statistics_file_name": "solver_statistics.json",
}

# if ANALYSIS_MODE:
#     solver_params["nonlinear_solver_statistics"] = ExtendedSolverStatistics

model_params = {
    "equilibrium_condition": EQUILIBRIUM_CONDITION,
    "has_time_dependent_boundary_equilibrium": False,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "reduce_linear_system": True,
    "flash_params": flash_params,
    "fractional_flow": FRACTIONAL_FLOW,
    "rediscretize_fourier_flux": FRACTIONAL_FLOW,
    "rediscretize_darcy_flux": FRACTIONAL_FLOW,
    "material_constants": material_params,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "compile": True,
    "flash_compiler_args": ("p-T", "p-h"),
}

if EXPORT_SCHEDULED_TIME_ONLY:
    model_params["times_to_export"] = time_schedule

model_params.update(phase_property_params)
model_params.update(restart_params)
model_params.update(meshing_params)
model_params.update(solver_params)

# endregion

# Casting to the most complex model type for typing purposes.
model = cast(cfle.EnthalpyBasedCFFLETemplate, model_class(model_params))

logging.basicConfig(level=logging.INFO)
logging.getLogger("porepy").setLevel(logging.DEBUG)
t_0 = time.time()
model.prepare_simulation()
prep_sim_time = time.time() - t_0
logging.getLogger("porepy").setLevel(logging.INFO)

# Defining sub system with block-diagonal form for efficient Schur complement reduction.
primary_equations = cf.get_primary_equations_cf(model)
primary_equations += [
    eq for eq in model.equation_system.equations.keys() if "flux" in eq
]
primary_equations += [
    "production_pressure_constraint",
    "injection_temperature_constraint",
]
primary_variables = cf.get_primary_variables_cf(model)
primary_variables += list(
    set([v.name for v in model.equation_system.variables if "flux" in v.name])
)

model.schur_complement_primary_equations = primary_equations
model.schur_complement_primary_variables = primary_variables

t_0 = time.time()
try:
    pp.run_time_dependent_model(model, model_params)
except Exception as err:
    print(f"SIMULATION FAILED: {err}")
    # NOTE To avoid recomputation of time step size.
    model.time_manager.is_constant = True
    model.after_nonlinear_convergence()
    model.time_manager.is_constant = False
sim_time = time.time() - t_0

# region Plots
fig, ax1 = plt.subplots()

global_iter_per_time: dict[float, tuple[int, int, int]] = model._global_iter_per_time  # type:ignore[attr-defined]

N = len(global_iter_per_time)
times = np.zeros(N)
newton_iter = np.zeros(N, dtype=int)
cum_armijo_iter = np.zeros(N, dtype=int)
cum_flash_iter = np.zeros(N, dtype=int)
for i, tNI in enumerate(global_iter_per_time.items()):
    t, NI = tNI
    n, a, f = NI
    times[i] = t
    newton_iter[i] = n
    cum_armijo_iter[i] = a
    cum_flash_iter[i] = f

img1 = ax1.plot(
    times,
    newton_iter,
    "k-",
    times,
    cum_armijo_iter,
    "k--",
)
ax1.set_xscale("log")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Global iterations")
ax1.set_ylim(
    0.0,
    30,
)

color = "tab:red"
ax2 = ax1.twinx()
ax2.set_ylabel("Local iterations", color=color)
img2 = ax2.plot(
    times,
    cum_flash_iter,
    "r--",
)
ax2.tick_params(axis="y", labelcolor=color)

ax1.legend(img1 + img2, ["Newton", "Line Search (cum.)", "Flash (cum.)"])

fig.tight_layout()
fig.savefig(pathlib.Path(".\\visualization\\num_iter.png").resolve(), format="png")

times_file = pathlib.Path("./visualization/times.json").resolve().open("r")
times_data = json.load(times_file)
TIMES = times_data["time"]
DT = times_data["dt"]
nT = len(DT)
assert nT == len(TIMES)
Tidx = np.arange(nT)

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("time step index")
ax1.set_ylabel("time (s)", color=color)
ax1.plot(Tidx, TIMES, color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_yscale("log")

ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel("dt (s)", color=color)
ax2.plot(Tidx, DT, color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_yscale("log")

fig.tight_layout()
fig.savefig(pathlib.Path(".\\visualization\\time_progress.png").resolve(), format="png")

# endregion


def min_avg_max(v: np.ndarray) -> tuple[float, float, float]:
    return float(v.min()), float(v.mean()), float(v.max())


print(
    f"equ type: {EQUILIBRIUM_CONDITION}\tfrac flow: {FRACTIONAL_FLOW}\t"
    + "h={model._h_MESH}"
)
print(f"Set-up time (incl. compilation): {prep_sim_time} (s).")
print(f"Simulation run time: {sim_time / 60.0} (min).")
print(f"smalles time step: {np.min(DT) / 3600} hours")
print(f"largest time step: {np.max(DT) / (3600 * 24)} days")
print(f"end time: {np.max(TIMES) / (3600 * 24 * 365)} years")
print(f"global num iter (min, avg, max): {min_avg_max(newton_iter)}")
print(f"cum. line search num iter (min, avg, max): {min_avg_max(cum_armijo_iter)}")
avg_cum_flash_iter = cum_flash_iter / model.mdg.num_subdomain_cells()
print(f"avg. cum. flash num iter (min, avg, max): {min_avg_max(avg_cum_flash_iter)}")
times_assembly = np.array(model._time_tracker["assembly"])
times_linsolve = np.array(model._time_tracker["linsolve"])
times_flash = np.array(model._time_tracker["flash"])
print(f"assembly time (min, avg, max): {min_avg_max(times_assembly)}")
print(f"linsolve time (min, avg, max): {min_avg_max(times_linsolve)}")
print(f"flash time (min, avg, max): {min_avg_max(times_flash)}")
