"""This module implements a multiphase-multicomponent flow model with phase change
using the Soereide model to define the fluid mixture.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

References:
    [1] Ingolf Søreide, Curtis H. Whitson,
        Peng-Robinson predictions for hydrocarbons, CO2, N2, and H2 S with pure water
        and NaCI brine,
        Fluid Phase Equilibria,
        Volume 77,
        1992,
        https://doi.org/10.1016/0378-3812(92)85105-H

"""

from __future__ import annotations

import logging
import os
import pathlib
import time
import warnings
from collections import deque
from typing import Any, Callable, Literal, Optional, Sequence, cast, no_type_check

import numpy as np
import scipy.sparse as sps

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
from porepy.applications.test_utils.models import create_local_model_class
from porepy.fracs.wells_3d import _add_interface
from porepy.numerics.solvers.andersonacceleration import AndersonAcceleration
from porepy.viz.solver_statistics import ExtendedSolverStatistics

logging.basicConfig(level=logging.INFO)
logging.getLogger("porepy").setLevel(logging.DEBUG)

t_0 = time.time()
import porepy.compositional.peng_robinson as pr
import porepy.models.compositional_flow_with_equilibrium as cfle

compile_time = time.time() - t_0


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# region Mathematical model configuration for flow & transport

fractional_flow: bool = False
use_well_model: bool = True
equilibrium_condition: str = "unified-p-h"
use_schur_technique: bool = True
analysis_mode: bool = True


class _FlowConfiguration(pp.PorePyModel):
    """Helper class to bundle the configuration of pressure, temperature and mass
    for in- and outflow."""

    # Initial values.
    _p_INIT: float = 20e6
    _T_INIT: float = 450.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    # In- and outflow values.
    _T_HEATED: float = 640.0
    _p_IN: float = _p_INIT
    _T_IN: float = 300.0
    _z_IN: dict[str, float] = {
        "H2O": _z_INIT["H2O"] - 0.095,
        "CO2": _z_INIT["CO2"] + 0.095,
    }
    _p_OUT: float = _p_INIT - 1e6
    _T_OUT: float = _T_INIT
    _z_OUT: dict[str, float] = {
        "H2O": _z_INIT["H2O"],
        "CO2": _z_INIT["CO2"],
    }

    # Injection model configuration

    _T_INJECTION: dict[int, float] = {0: _T_IN}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}

    # Value obtained from a p-T flash with values defined above.
    # Divide by 3600 to obtain an injection of unit per hour
    # Multiplied by some number for how many units per hour
    _TOTAL_INJECTED_MASS: float = 10 * 27430.998956110157 / (60 * 60)  # mol / m^3

    _INJECTED_MASS: dict[str, dict[int, float]] = {
        "H2O": {0: _TOTAL_INJECTED_MASS * _z_IN["H2O"]},
        "CO2": {0: _TOTAL_INJECTED_MASS * _z_IN["CO2"]},
    }

    # Coordinates of injection and production wells in meters
    _INJECTION_POINTS: list[np.ndarray] = [np.array([15.0, 10.0])]
    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([85.0, 10.0])]


# endregion


class SoereideMixture(pp.PorePyModel):
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.FluidComponent]
    ) -> Sequence[
        tuple[pp.compositional.EoSCompiler, pp.compositional.PhysicalState, str]
    ]:
        eos = pr.PengRobinsonCompiler(
            components, [pr.h_ideal_H2O, pr.h_ideal_CO2], pr.get_bip_matrix(components)
        )
        return [
            (eos, pp.compositional.PhysicalState.liquid, "L"),
            (eos, pp.compositional.PhysicalState.gas, "G"),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        return [self.pressure, self.temperature] + [  # type:ignore[return-value]
            phase.extended_fraction_of[comp] for comp in phase
        ]


class AnalysisMixin(pp.PorePyModel):
    """Storing some information for analysis purposes."""

    @no_type_check
    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Flags the time step as diverged, if there is a nan in the residual."""

        if isinstance(self.nonlinear_solver_statistics, ExtendedSolverStatistics):
            res_norm_per_eq = {}
            incr_norm_per_var = {}
            condition_numbers = {}

            var_names = set([var.name for var in self.equation_system.variables])

            for n, e in self.equation_system.equations.items():
                res_norm_per_eq[n] = self.compute_residual_norm(
                    self.equation_system.evaluate(e), reference_residual
                )
            for n in var_names:
                v = nonlinear_increment[self.equation_system.dofs_of([n])]
                incr_norm_per_var[n] = self.compute_residual_norm(v, reference_residual)

            # A = self.linear_system[0]
            # try:
            #     condition_numbers["global"] = np.linalg.cond(A.todense())
            # except np.linalg.LinAlgError:
            #     condition_numbers["global"] = np.nan

            # A, _ = self.equation_system.assemble(
            #     equations=self.primary_equations, variables=self.primary_variables
            # )
            # try:
            #     condition_numbers["block_pp"] = np.linalg.cond(A.todense())
            # except np.linalg.LinAlgError:
            #     condition_numbers["block_pp"] = np.nan
            # A, _ = self.equation_system.assemble(
            #     equations=self.secondary_equations, variables=self.secondary_variables
            # )
            # try:
            #     condition_numbers["block_ss"] = np.linalg.cond(A.todense())
            # except np.linalg.LinAlgError:
            #     condition_numbers["block_ss"] = np.nan

            self.nonlinear_solver_statistics.extended_log(
                self.time_manager.time,
                self.time_manager.dt,
                res_norm_per_equation=res_norm_per_eq,
                incr_norm_per_variable=incr_norm_per_var,
                condition_numbers=condition_numbers,
            )

        return super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )

    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()

        for sd in self.mdg.subdomains(dim=0):
            p = self.equation_system.evaluate(self.pressure([sd]))
            T = self.equation_system.evaluate(self.temperature([sd]))
            h = self.equation_system.evaluate(self.enthalpy([sd]))
            if "injection_well" in sd.tags:
                s = "injection"
                i = sd.tags["injection_well"]
            elif "production_well" in sd.tags:
                s = "production"
                i = sd.tags["production_well"]
            else:
                continue
            print(f"{s} well {i}:\n\tp={p}\n\tT={T}\n\th={h}")


class SolutionStrategy(pp.PorePyModel):
    """Provides some pre- and post-processing for flash methods."""

    flash: pp.compositional.Flash

    compute_nonlinear_increment_norm: Callable[[np.ndarray], float]

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

    _block_row_permutation: np.ndarray
    _block_column_permutation: np.ndarray
    _block_size: int

    def __init__(self, params: dict | None = None):
        super().__init__(params)  # type:ignore[safe-super]

        self._last_residual_norms: deque[float] = deque(maxlen=4)
        self._last_increment_norms: deque[float] = deque(maxlen=3)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()  # type:ignore[misc]
        self._last_residual_norms.clear()

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
        tol = float(self.params["nl_convergence_tol_res"])
        tol_inc = float(self.params["nl_convergence_tol"])
        if status == (False, False):
            status_per_eq = []
            rn_per_eq = {}
            for name, eq in self.equation_system.equations.items():
                rn = self.compute_residual_norm(
                    cast(np.ndarray, self.equation_system.evaluate(eq)),
                    reference_residual,
                )
                if "isofugacity" in name:
                    status_per_eq.append(rn < 1e-2)
                    rn_per_eq[name] = rn
                else:
                    status_per_eq.append(rn < tol)

            residuals_converged = all(status_per_eq)

            status_per_var = []
            rn_per_var = {}
            partial_frac_vars = self.fraction_in_phase_variables
            for var in self.equation_system.variables:
                rn = self.compute_nonlinear_increment_norm(
                    nonlinear_increment[self.equation_system.dofs_of([var])]
                )
                if var.name in partial_frac_vars:
                    status_per_var.append(rn < 1e-2)
                    rn_per_var[(var.name, var.domain.id)] = rn
                else:
                    status_per_var.append(rn < tol_inc)

            increments_converged = all(status_per_var)

            status = (
                residuals_converged and increments_converged,
                False,
            )
            if status[0]:
                print(
                    "Converged with custom criteria.\n"
                    + f"\tNon-satisfying residuals:\n{rn_per_eq}\n"
                    + f"\tNon-satisfying Increments:\n{rn_per_var}\n"
                )

        self._last_residual_norms.append(
            self.compute_residual_norm(residual, reference_residual)
        )
        self._last_increment_norms.append(
            self.compute_nonlinear_increment_norm(nonlinear_increment)
        )

        if len(self._last_residual_norms) == self._last_residual_norms.maxlen:
            residual_stationary = (
                np.allclose(
                    self._last_residual_norms,
                    self._last_residual_norms[-1],
                    rtol=0.0,
                    atol=tol,
                )
                and tol != np.inf
            )
            increment_stationary = (
                np.allclose(
                    self._last_increment_norms,
                    self._last_increment_norms[-1],
                    rtol=0.0,
                    atol=tol_inc,
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

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """To avoid nonphysical overall fractions due to over-shooting with large
        time step sizes, we introduce a normalization here after the Newton increment is
        added to the global solution vector.

        """
        super().after_nonlinear_iteration(nonlinear_increment)  # type:ignore

        sds = self.mdg.subdomains()

        # Normalizing feed fractions in case of overshooting
        eps = 1e-7  # binding overall fractions away from zero
        z = np.array(
            [
                self.equation_system.evaluate(comp.fraction(sds))
                for comp in self.fluid.components
            ]
        )
        z[z >= 1.0] = 1.0 - eps
        z[z <= 0.0] = 0.0 + eps

        z = pp.compositional.normalize_rows(z.T).T

        for z_i, comp in zip(z, self.fluid.components):
            if self.has_independent_fraction(comp):
                self.equation_system.set_variable_values(
                    z_i,
                    [comp.fraction(sds)],  # type:ignore[arg-type]
                    iterate_index=0,
                )

    def after_nonlinear_failure(self):
        self.exporter.write_pvd()
        super().after_nonlinear_failure()  # type:ignore

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """Performing pT flash in injection wells, because T is fixed there."""
        for sd in self.mdg.subdomains():
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
            elif self.nonlinear_solver_statistics.num_iteration % 3 == 0:
                self.local_equilibrium(sd, state=state)  # type:ignore

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

        unity_tolerance = 0.05

        # sat_sum = np.sum(fluid_state.sat, axis=0)
        # if np.any(sat_sum > 1 + unity_tolerance):
        #     raise ValueError("Saturations violate unity constraint")
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = fluid_state.y < 0
        if np.any(idx):
            fluid_state.y[idx] = 0.0
        idx = fluid_state.y > 1.0
        if np.any(idx):
            fluid_state.y[idx] = 1.0
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = y_sum > 1.0
        if np.any(idx):
            fluid_state.y[:, idx] = pp.compositional.normalize_rows(
                fluid_state.y[:, idx].T
            ).T

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

            sub_state, sub_success, _ = self.flash.flash(**flash_kwargs)

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

    def schur_complement_inverter(self) -> Callable[[sps.spmatrix], sps.spmatrix]:
        """Parallelized block diagonal inverter for local equilibrium equations,
        assuming they are defined on all subdomains in each cell."""

        # return super().schur_complement_inverter()
        if not hasattr(self, "_block_permutation"):
            # Local system size of p-h flash including saturations and phase mass
            # constraints.
            ncomp = self.fluid.num_components
            nphase = self.fluid.num_phases
            block_size: int = ncomp - 1 + ncomp * (nphase - 1) + nphase + 1 + nphase - 1
            assert block_size == len(self.secondary_equations)
            assert block_size == len(self.secondary_variables)
            self._block_size = block_size

            # Permutation of DOFs in porepy, assuming the order is equ1, equ2,...
            # With an subdomain-minor order per variable.
            N = self.mdg.num_subdomain_cells()
            shift = np.kron(np.arange(block_size), np.ones(N))
            stride = np.arange(N) * block_size
            permutation = (np.kron(np.ones(block_size), stride) + shift).astype(
                np.int32
            )
            identity = np.arange(N * block_size, dtype=np.int32)
            assert np.allclose(np.sort(permutation), identity, rtol=0.0, atol=1e-16)

            self._block_row_permutation = permutation

            # Above permutation can be used for both column and blocks, if there is
            # no point grid. If there is a point grid, above permutation assembles the
            # blocks belonging to the point grid cells already fully, due to how AD is
            # implemented in PorePy. We then need a special column permutation, which
            # permutes only on the those grids which are not point grids.
            # Note that the subdomains are sorted by default ascending w.r.t. their
            # dimension. Point grids come last
            subdomains = self.mdg.subdomains()
            if subdomains[-1].dim == 0:
                non_point_grids = [g for g in subdomains if g.dim > 0]
                sub_N = sum([g.num_cells for g in non_point_grids])
                shift = np.kron(np.arange(block_size), np.ones(sub_N))
                stride = np.arange(sub_N) * block_size
                sub_permutation = (np.kron(np.ones(block_size), stride) + shift).astype(
                    np.int32
                )
                sub_identity = np.arange(sub_N * block_size, dtype=np.int32)
                assert np.allclose(
                    np.sort(sub_permutation), sub_identity, rtol=0.0, atol=1e-16
                )
                permutation = np.arange(N * block_size, dtype=np.int32)
                permutation[: sub_N * block_size] = sub_permutation
                self._block_column_permutation = permutation

        def inverter(A: sps.csr_matrix) -> sps.csr_matrix:
            row_perm = pp.matrix_operations.ArraySlicer(
                range_indices=self._block_row_permutation
            )
            inv_col_perm = pp.matrix_operations.ArraySlicer(
                domain_indices=self._block_row_permutation
            )
            if hasattr(self, "_block_column_permutation"):
                col_perm = pp.matrix_operations.ArraySlicer(
                    range_indices=self._block_column_permutation
                )
                inv_row_perm = pp.matrix_operations.ArraySlicer(
                    domain_indices=self._block_column_permutation
                )
            else:
                col_perm = row_perm
                inv_row_perm = inv_col_perm

            A_block = (col_perm @ (row_perm @ A).transpose()).transpose()

            # The local p-h flash system has a size of
            inv_A_block = pp.matrix_operations.invert_diagonal_blocks(
                A_block,
                (np.ones(self.mdg.num_subdomain_cells()) * self._block_size).astype(
                    np.int32
                ),
                method="numba",
            )

            inv_A = (
                inv_col_perm @ (inv_row_perm @ inv_A_block).transpose()
            ).transpose()
            # Because the matrix has a condition number of order 5 to 6, and we use
            # float64 (precision order -16), the last entries are useless, if not
            # problematic.
            treat_as_zero = np.abs(inv_A.data) < 1e-10
            inv_A.data[treat_as_zero] = 0.0
            inv_A.eliminate_zeros()
            return inv_A

        return inverter


class Geometry(_FlowConfiguration):
    """2D matrix representing a rectangular domain."""

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": self.units.convert_units(100.0, "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(20.0, "m"),
            }
        )


class PointWells2D(Geometry):
    """2D matrix with point grids as injection and production points.

    Alternative for the ``WellNetwork3d`` in 2d.

    """

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

    def _inlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define inlet."""
        sides = self.domain_boundary_sides(sd)

        inlet = np.zeros(sd.num_faces, dtype=bool)
        inlet[sides.west] = True

        return inlet

    def _outlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define outlet."""

        sides = self.domain_boundary_sides(sd)

        outlet = np.zeros(sd.num_faces, dtype=bool)
        outlet[sides.east] = True

        return outlet

    def _heated_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.south] = True
        left, right = self._central_stripe(sd)
        heated &= sd.face_centers[0] >= left
        heated &= sd.face_centers[0] <= right

        return heated

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Setting only conditions on matrix
        if sd.dim == 2:
            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, inout, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            heated = self._heated_faces(sd) | inout
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)
        # if sd.dim == 2:
        #     return pp.BoundaryCondition(sd, self._inlet_faces(sd), "dir")
        # else:
        #     return pp.BoundaryCondition(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_fluid_flux(sd)

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Flag the inflow and outflow faces for the flash."""
        return self.bc_type_fourier_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._p_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            vals[inlet] = self._p_IN
            vals[outlet] = self._p_OUT

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._T_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)
            heated = self._heated_faces(sd)

            vals[inlet] = self._T_IN
            vals[outlet] = self._T_OUT
            vals[heated] = self._T_HEATED

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._z_INIT[component.name]

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            vals[inlet] = self._z_IN[component.name]
            vals[outlet] = self._z_OUT[component.name]

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals


class BoundaryConditionsOnlyConduction(BoundaryConditions):
    """Declares all faces to be Neumann, except the heated boundary."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            heated = self._heated_faces(sd)
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # return pp.BoundaryCondition(sd)
        return self.bc_type_fourier_flux(sd)


class NewtonAndersonArmijoSolver(pp.NewtonSolver, AndersonAcceleration):
    """Newton solver with Armijo line search and Anderson acceleration.

    The residual objective function is tailored to models where phase properties are
    assumed to be surrogate factories and require an update before evaluating the
    objective function.

    """

    def __init__(self, params: dict | None = None):
        pp.NewtonSolver.__init__(self, params)
        if params is None:
            params = {}
        depth = int(params.get("anderson_acceleration_depth", 3))
        dimension = int(params["anderson_acceleration_dimension"])
        constrain = params.get("anderson_acceleration_constrained", False)
        reg_param = params.get("anderson_acceleration_regularization_parameter", 0.0)
        AndersonAcceleration.__init__(
            self,
            dimension,
            depth,
            constrain_acceleration=constrain,
            regularization_parameter=reg_param,
        )

    def iteration(self, model: pp.PorePyModel):
        """An iteration consists of performing the Newton step, obtaining the step size
        from the line search, and then performing the Anderson acceleration based on
        the iterates which are obtained using the step size."""

        iteration = model.nonlinear_solver_statistics.num_iteration

        dx = pp.NewtonSolver.iteration(self, model)

        res_norm = np.linalg.norm(model.linear_system[1])

        if self.params.get("Anderson_acceleration", False):
            x = model.equation_system.get_variable_values(iterate_index=0)
            x_temp = x + dx
            if not (np.any(np.isnan(x_temp)) or np.any(np.isinf(x_temp))):
                try:
                    xp1 = self.apply(x_temp, dx.copy(), iteration)
                    if res_norm < 10.0:
                        dx = xp1 - x
                except Exception:
                    logger.warning(
                        f"Resetting Anderson acceleration at"
                        f" T={model.time_manager.time}; i={iteration} due to failure."
                    )
                    self.reset()
        alpha = self.nonlinear_line_search(model, dx)
        return alpha * dx

    def nonlinear_line_search(
        self, model: pp.PorePyModel, dx: np.ndarray
    ) -> np.ndarray:
        """Performs the Armijo line search."""
        if not self.params.get("Global_line_search", False):
            return np.ones_like(dx)

        rho = float(self.params.get("armijo_line_search_weight", 0.9))
        kappa = float(self.params.get("armijo_line_search_incline", 0.4))
        N = int(self.params.get("armijo_line_search_max_iterations", 50))

        pot_0 = self.residual_objective_function(model, dx, 0.0)
        rho_i = rho
        n = 0

        for i in range(N):
            n = i
            rho_i = rho**i

            pot_i = self.residual_objective_function(model, dx, rho_i)
            if pot_i <= (1 - 2 * kappa * rho_i) * pot_0:
                break

        logger.info(f"Armijo line search determined weight: {rho_i} ({n})")
        return rho_i * np.ones_like(dx)

    def residual_objective_function(
        self, model: pp.PorePyModel, dx: np.ndarray, weight: float
    ) -> np.floating[Any]:
        """The objective function to be minimized is the norm of the residual squared
        and divided by 2."""
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        state = x_0 + weight * dx
        # model.update_thermodynamic_properties_of_phases(state)
        cfle.cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
            model,  # type:ignore[arg-type]
            state,
        )
        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return np.dot(residual, residual) / 2


# region Model class assembly
model_class: type[pp.PorePyModel]

if equilibrium_condition == "unified-p-h" and fractional_flow:
    model_class = cfle.EnthalpyBasedCFFLETemplate  # type:ignore[type-abstract]
elif equilibrium_condition == "unified-p-h" and not fractional_flow:
    model_class = cfle.EnthalpyBasedCFLETemplate  # type:ignore[type-abstract]
else:
    raise ValueError(
        "Unsupported base configuration: "
        + f"{equilibrium_condition}, "
        + f"fractiona_flow={fractional_flow},"
    )


model_class = create_local_model_class(
    model_class,
    [InitialConditions, SolutionStrategy, SoereideMixture],  # type:ignore[type-abstract]
)

if use_well_model:
    model_class = create_local_model_class(
        model_class,
        [PointWells2D, AdjustedWellModel2D, BoundaryConditionsOnlyConduction],  # type:ignore[type-abstract]
    )
else:
    model_class = create_local_model_class(
        model_class,
        [Geometry, BoundaryConditions],  # type:ignore[type-abstract]
    )

if fractional_flow:
    model_class = create_local_model_class(
        model_class,
        [pp.constitutive_laws.FouriersLawAd, pp.constitutive_laws.DarcysLawAd],  # type:ignore[type-abstract]
    )

if analysis_mode:
    model_class = create_local_model_class(model_class, [AnalysisMixin])  # type:ignore[type-abstract]

# endregion

# Model parametrization

time_schedule = [i * pp.DAY for i in range(31)] + [
    i * 15 * pp.DAY for i in range(3, 25)
]

max_iterations = 30
newton_tol = 1.5e-4
newton_tol_increment = 1e-6

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=2 * pp.HOUR,
    dt_min_max=(10.0 * pp.MINUTE, 15 * pp.DAY),
    iter_max=max_iterations,
    iter_optimal_range=(12, 24),
    iter_relax_factors=(0.8, 2.0),
    recomp_factor=0.6,
    recomp_max=15,
    print_info=True,
    rtol=0.0,
)

phase_property_params = {
    "phase_property_params": [0.0],
}

material_params = {
    "solid": pp.SolidConstants(
        permeability=1e-13,
        porosity=0.2,
        thermal_conductivity=1.6736,
        specific_heat_capacity=3.0,
    )
}

flash_params: dict[Any, Any] = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "tolerance": 1e-8,
        "max_iterations": 80,  # 150
        "armijo_rho": 0.99,
        "armijo_kappa": 0.4,
        "armijo_max_iterations": 50 if "p-T" in equilibrium_condition else 30,
        "npipm_u1": 10,
        "npipm_u2": 10,
        "npipm_eta": 0.5,
    },
}
flash_params.update(phase_property_params)

restart_params = {
    "restart_options": {
        "restart": False,
        # "pvd_file": pathlib.Path('.\\visualization\\data.pvd').resolve(),
        "is_mdg_pvd": False,
        "vtu_files": None,
        # "times_file": pathlib.Path('.\\visualization\\times.json').resolve(),
        "time_index": -1,
    },
}

meshing_params = {
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size": 5e-1,
        # "cell_size": 1.0,
        "cell_size_fracture": 5e-1,
    },
}

solver_params = {
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "linear_solver": "scipy_sparse",
    "Global_line_search": True,
    "nonlinear_solver": NewtonAndersonArmijoSolver,
    "armijo_line_search_weight": 0.95,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 10,
    "Anderson_acceleration": False,
    "anderson_acceleration_depth": 3,
    "anderson_acceleration_constrained": False,
    "anderson_acceleration_regularization_parameter": 1e-3,
    "solver_statistics_file_name": "solver_statistics.json",
    "nonlinear_solver_statistics": ExtendedSolverStatistics,
}

model_params = {
    "equilibrium_condition": equilibrium_condition,
    "has_time_dependent_boundary_equilibrium": False,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "reduce_linear_system": use_schur_technique,
    "flash_params": flash_params,
    "fractional_flow": fractional_flow,
    "rediscretize_fourier_flux": fractional_flow,
    "rediscretize_darcy_flux": fractional_flow,
    "material_constants": material_params,
    "time_manager": time_manager,
    "times_to_export": time_schedule,
    "prepare_simulation": False,
    "compile": True,
    "flash_compiler_args": ("p-T", "p-h"),
}

model_params.update(phase_property_params)
model_params.update(restart_params)
model_params.update(meshing_params)
model_params.update(solver_params)

# Casting to the most complex model type for typing purposes.
model = cast(cfle.EnthalpyBasedCFFLETemplate, model_class(model_params))

t_0 = time.time()
model.prepare_simulation()
prep_sim_time = time.time() - t_0
compile_time += prep_sim_time
model_params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()

if use_schur_technique:
    primary_equations = model.get_primary_equations_cf()
    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    if use_well_model:
        primary_equations += [
            "production_pressure_constraint",
            "injection_temperature_constraint",
        ]
    primary_variables = model.get_primary_variables_cf()
    primary_variables += list(
        set([v.name for v in model.equation_system.variables if "flux" in v.name])
    )

    model.primary_equations = primary_equations
    model.primary_variables = primary_variables

logging.getLogger("porepy").setLevel(logging.INFO)
t_0 = time.time()
pp.run_time_dependent_model(model, model_params)
sim_time = time.time() - t_0

print(f"Set-up time: {prep_sim_time} (s).")
print(f"Approximate compilation time: {compile_time} (s).")
print(f"Simulation run time: {sim_time / 60.0} (min).")
