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

REFINEMENT_LEVEL: Literal[0, 1, 2, 3, 4] = 3
"""Chose mesh size with h = 4 * 0.5 ** i, with i being the refinement level."""
EQUILIBRIUM_CONDITION: str = "unified-p-h"
"""Define the equilibrium condition to determin the flash type used in the solution
procedure."""
FLASH_TOL_CASE: Literal[0, 1, 2, 3, 4] = 4
"""Define the flash tolerance used in the solution procedure."""
EXPORT_SCHEDULED_TIME_ONLY: bool = False
"""Exports all  time steps produced by the time stepping algorithm, otherwise only
the scheduled times."""
BUOYANCY_ON: bool = False
"""Turn on buoyancy. NOTE: This is still under development."""

MESH_SIZES: dict[int, float] = {
    0: 4.0,  # 308 cells
    1: 2.0,  # 1204 cells
    2: 1.0,  # 4636 cells
    3: 0.5,  # 18,464 cells
    4: 0.25,  # 73,748 cells
}
"""Tested mesh sizes in meters."""

FLASH_TOLERANCES: dict[int, float] = {
    0: 1e-1,
    1: 1e-2,
    2: 1e-3,
    3: 1e-5,
    4: 1e-8,
}
"""Tested flash tolerances."""

h_MESH = MESH_SIZES[REFINEMENT_LEVEL]
tol_flash = FLASH_TOLERANCES[FLASH_TOL_CASE]

FRACTIONAL_FLOW: bool = False
"""Use the fractional flow formulation without upwinding in the diffusive fluxes."""

DISABLE_COMPILATION: bool = False
"""For disabling numba compilation and faster start of simulation."""
USE_ADTPFA_FLUX_DISCRETIZATION: bool = False
"""Uses the adaptive flux discretization for both Darcy and Fourier flux."""


import json
import logging
import pathlib
import time
import warnings
from collections import deque
from typing import Any, Callable, Literal, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sps
from scipy.linalg import lstsq

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

    pressure_variable: str
    temperature_variable: str
    enthalpy_variable: str
    fraction_in_phase_variables: list[str]

    def __init__(self, params: dict | None = None):
        super().__init__(params)  # type:ignore[safe-super]

        self._residual_norm_history: deque[float] = deque(maxlen=4)
        self._increment_norm_history: deque[float] = deque(maxlen=3)

        # Data saving for plotting for paper.
        self._time_steps: list[float] = []
        self._time_step_sizes: list[float] = []
        self._time_tracker: dict[
            Literal["flash", "assembly", "linsolve"], list[float]
        ] = {
            "flash": [],
            "assembly": [],
            "linsolve": [],
        }
        self._recomputations: list[int] = []
        """Number of recomputations of dt at a time due to convergence failure."""
        self._num_global_iter: list[int] = []
        """Number of global iterations per successful time step."""
        self._num_cell_averaged_flash_iter: list[int] = []
        """Number of cell-averaged flash iterations per successful time step."""
        self._num_linesearch_iter: list[int] = []
        """Number of linesearch iterations per successful time step."""

        self._flash_iter_counter: int = 0
        """Counter for cell-averaged flash iterations per time step."""

        self._cum_flash_iter_per_grid: dict[pp.Grid, list[np.ndarray]] = {}

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
        self._cum_flash_iter_per_grid.clear()
        self._flash_iter_counter = 0
        model.nonlinear_solver_statistics.num_iteration_armijo = 0

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
        # Get number of recomputations from time manager before it is reset.
        self._recomputations.append(self.time_manager._recomp_num)
        super().after_nonlinear_convergence()
        self._num_global_iter.append(self.nonlinear_solver_statistics.num_iteration)
        self._num_cell_averaged_flash_iter.append(self._flash_iter_counter)
        self._num_linesearch_iter.append(
            self.nonlinear_solver_statistics.num_iteration_armijo
        )
        self._time_step_sizes.append(self.time_manager.dt)
        # NOTE the time manager always returns the time at the end of the time step,
        # The one for which we solve.
        self._time_steps.append(self.time_manager.time - self.time_manager.dt)

    def after_nonlinear_failure(self):
        # Do not include clock times of failed attempts.
        n = self.nonlinear_solver_statistics.num_iteration
        self._time_tracker["linsolve"] = self._time_tracker["linsolve"][:-n]
        self._time_tracker["assembly"] = self._time_tracker["assembly"][:-n]
        self._time_tracker["flash"] = self._time_tracker["flash"][:-n]
        return super().after_nonlinear_failure()

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """Performing pT flash in injection wells, because T is fixed there."""
        start = time.time()
        stride = int(self.params["flash_params"].get("global_iteration_stride", 3))
        nfi_mean = 0
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
                nfi = self.local_equilibrium(
                    sd,
                    state=state,
                    equilibrium_specs=equ_spec,
                    return_num_iter=True,
                )  # type:ignore
                self._cum_flash_iter_per_grid[sd].append(nfi)
            elif self.nonlinear_solver_statistics.num_iteration % stride == 0:
                nfi = self.local_equilibrium(sd, state=state, return_num_iter=True)  # type:ignore
                self._cum_flash_iter_per_grid[sd].append(nfi)
            else:
                cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
                    self,
                    state,
                )
                nfi = np.zeros(sd.num_cells, dtype=int)
            nfi_mean += nfi.mean()
        self._flash_iter_counter += int(nfi_mean)
        self._time_tracker["flash"].append(time.time() - start)

    def assemble_linear_system(self) -> None:
        start = time.time()
        super().assemble_linear_system()
        self._time_tracker["assembly"].append(time.time() - start)

    def solve_linear_system(self) -> np.ndarray:
        start = time.time()
        sol = super().solve_linear_system()
        self._time_tracker["linsolve"].append(time.time() - start)
        return sol

    # def add_nonlinear_fourier_flux_discretization(self) -> None:
    #     pass


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


class BuoyancyModel(pp.PorePyModel):
    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.update_buoyancy_driven_fluxes()

    def set_nonlinear_discretizations(self):
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field


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
    """No flow BC, with the exception of a stripe on the bottom boundary where
    temperature Dirichlet-BC are given."""

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


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(
        self,
        dimension: int,
        depth: int,
        constrain_acceleration: bool = False,
        regularization_parameter: float = 0.0,
    ) -> None:
        self._dimension = int(dimension)
        self._depth = int(depth)
        self._constrain_acceleration: bool = bool(constrain_acceleration)
        self._reg_param: float = float(regularization_parameter)

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = np.zeros(self._dimension)
        self._gkm1: np.ndarray = np.zeros(self._dimension)

    def reset(self) -> None:
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Parameters:
            gk: application of some fixed point iteration onto approximation xk, i.e.,
                g(xk).
            fk: residual g(xk) - xk; in general some increment.
            iteration: current iteration count.

        Returns:
            Modified application of fixed point approximation after acceleration, i.e.,
            the new iterate xk+1.

        """

        if iteration == 0:
            self.reset()

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem.
            A = self._Fk[:, 0:mk]
            b = fk
            if self._constrain_acceleration:
                A = np.vstack((A, np.ones((1, self._depth))))
                b = np.concatenate((b, np.ones(1)))

            direct_solve = False

            if self._reg_param > 0:
                b = A.T @ b
                A = A.T @ A + self._reg_param * np.eye(A.shape[1])
                direct_solve = np.linalg.matrix_rank(A) >= A.shape[1]

            if direct_solve:
                gamma_k = np.linalg.solve(A, b)
            else:
                gamma_k = lstsq(A, b)[0]

            # Do the mixing
            x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        # Store values for next iteration.
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return x_k_plus_1


class NewtonArmijoAndersonSolver(pp.NewtonSolver, AndersonAcceleration):
    """Newton solver with Armijo line search.

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
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""
        # dx = super().iteration(model)
        iteration = model.nonlinear_solver_statistics.num_iteration

        dx = pp.NewtonSolver.iteration(self, model)

        if self.params.get("anderson_acceleration", False):
            x = model.equation_system.get_variable_values(iterate_index=0)
            x_temp = x + dx
            if not (np.any(np.isnan(x_temp)) or np.any(np.isinf(x_temp))):
                try:
                    xp1 = self.apply(x_temp, dx.copy(), iteration)
                    res = model.equation_system.assemble(evaluate_jacobian=False)
                    # TODO Wrong reference residual
                    res_norm = model.compute_residual_norm(res, res)
                    if res_norm <= self.params.get(
                        "anderson_start_after_residual_reaches", np.inf
                    ):
                        dx = xp1 - x
                except Exception:
                    logger.warning(
                        f"Resetting Anderson acceleration at"
                        f" T={model.time_manager.time}; i={iteration} due to failure."
                    )
                    self.reset()

        alpha = self.armijo_line_search(model, dx)
        return alpha * dx

    def armijo_line_search(self, model: pp.PorePyModel, dx: np.ndarray) -> float:
        """Performs the Armijo line search."""
        res = model.equation_system.assemble(evaluate_jacobian=False)
        # TODO Wrong reference residual
        res_norm = model.compute_residual_norm(res, res)
        if not self.params.get(
            "armijo_line_search", False
        ) or res_norm <= self.params.get("armijo_stop_after_residual_reaches", 0.0):
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


# mypy: disable-error-code="type-abstract,attr-defined"
model_class: type[pp.PorePyModel]

if FRACTIONAL_FLOW:
    model_class = create_local_model_class(
        cfle.EnthalpyBasedCFFLETemplate,
        [pp.constitutive_laws.DarcysLawAd],
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
        # BuoyancyModel,
    ],
)

if USE_ADTPFA_FLUX_DISCRETIZATION:
    model_class = create_local_model_class(
        model_class,
        [
            pp.constitutive_laws.DarcysLawAd,
            pp.constitutive_laws.FouriersLawAd,
        ],
    )

if __name__ == "__main__":
    time_schedule = [i * 30 * pp.DAY for i in range(25)]

    max_iterations = 40 if FRACTIONAL_FLOW else 30
    newton_tol = 1e-6
    newton_tol_increment = 5e-6

    time_manager = pp.TimeManager(
        schedule=time_schedule,
        dt_init=10 * pp.MINUTE,
        dt_min_max=(pp.MINUTE, 30 * pp.DAY),
        iter_max=max_iterations,
        iter_optimal_range=(20, 30) if FRACTIONAL_FLOW else (10, 20),
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
            "tolerance": tol_flash,
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
            "cell_size": h_MESH,
            "cell_size_fracture": 5e-1,
        },
    }

    solver_params = {
        "max_iterations": max_iterations,
        "nl_convergence_tol": newton_tol_increment,
        "nl_convergence_tol_res": newton_tol,
        "apply_schur_complement_reduction": True,
        "linear_solver": "scipy_sparse",
        "nonlinear_solver": NewtonArmijoAndersonSolver,
        "armijo_line_search": True,
        "armijo_line_search_weight": 0.95,
        "armijo_line_search_incline": 0.2,
        "armijo_line_search_max_iterations": 10,
        "armijo_stop_after_residual_reaches": 1e0,
        "anderson_acceleration": False,
        "anderson_acceleration_depth": 3,
        "anderson_acceleration_constrained": False,
        "anderson_acceleration_regularization_parameter": 1e-3,
        "anderson_start_after_residual_reaches": 1e2,
        "solver_statistics_file_name": "solver_statistics.json",
    }

    model_params = {
        "equilibrium_condition": EQUILIBRIUM_CONDITION,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "flash_params": flash_params,
        "fractional_flow": FRACTIONAL_FLOW,
        "material_constants": material_params,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "buoyancy_on": BUOYANCY_ON,
        "compile": True,
        "flash_compiler_args": ("p-T", "p-h"),
    }

    if EXPORT_SCHEDULED_TIME_ONLY:
        model_params["times_to_export"] = time_schedule

    model_params.update(phase_property_params)
    model_params.update(restart_params)
    model_params.update(meshing_params)
    model_params.update(solver_params)

    # Casting to the most complex model type for typing purposes.
    model = cast(cfle.EnthalpyBasedCFFLETemplate, model_class(model_params))
    model.nonlinear_solver_statistics.num_iteration_armijo = 0  # type:ignore[attr-defined]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    model_params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()

    # Defining sub system for Schur complement reduction.
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
    if "p-T" in EQUILIBRIUM_CONDITION:
        try:
            pp.run_time_dependent_model(model, model_params)
        except Exception as err:
            print(f"SIMULATION FAILED: {err}")
            # NOTE To avoid recomputation of time step size.
            model.time_manager.is_constant = True
            model.after_nonlinear_convergence()
            model.time_manager.is_constant = False
    else:
        pp.run_time_dependent_model(model, model_params)
    sim_time = time.time() - t_0

    # Dump simulation data for visualization.
    model = cast(SolutionStrategy, model)  # type:ignore[assignment]
    data = {
        "refinement_level": REFINEMENT_LEVEL,
        "equilibrium_condition": EQUILIBRIUM_CONDITION,
        "tol_flash_case": FLASH_TOL_CASE,
        "num_cells": model.mdg.num_subdomain_cells(),
        "t": model._time_steps,
        "dt": model._time_step_sizes,
        "recomputations": model._recomputations,
        "num_global_iter": model._num_global_iter,
        "num_flash_iter": model._num_cell_averaged_flash_iter,
        "num_linesearch_iter": model._num_linesearch_iter,
        "clock_time_global_solver": (
            np.mean(model._time_tracker["linsolve"]),
            np.sum(model._time_tracker["linsolve"]),
        ),
        "clock_time_assembly": (
            np.mean(model._time_tracker["assembly"]),
            np.sum(model._time_tracker["assembly"]),
        ),
        "clock_time_flash_solver": (
            np.mean(model._time_tracker["flash"]),
            np.sum(model._time_tracker["flash"]),
        ),
        "setup_time": prep_sim_time,
        "simulation_time": sim_time,
    }

    with open(
        pathlib.Path(
            f"stats_{EQUILIBRIUM_CONDITION}_h{REFINEMENT_LEVEL}_ftol{FLASH_TOL_CASE}.json"
        ),
        "w",
    ) as result_file:
        json.dump(data, result_file)
