"""Example for 2-phase, 2-component flow using equilibrium calculations.

Simulates the injection of CO2 into an initially water-saturated 2D domain, using a
mD model with points as wells.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Literal, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.models.compositional_flow as cf
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.fracs.wells_3d import _add_interface


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

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()  # type:ignore[misc]
        self._residual_norm_history.clear()
        self._increment_norm_history.clear()

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
                print("\nConverged with relaxed CFLE criteria.\n")

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

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """Performing pT flash in injection wells, because T is fixed there."""
        stride = self.params.get("flash_params", {}).get("global_iteration_stride", 1)  # type:ignore
        do_flash = False
        if isinstance(stride, int):
            assert stride > 0, "Global iteration stride must be positive."
            n = self.nonlinear_solver_statistics.num_iteration
            do_flash = (n + 1) % stride == 0 or n == 0
        elif stride is not None:
            raise ValueError(
                f"Global iteration stride for local equilibrium solver must be integer"
                f" or None, got {type(stride)}."
            )
        for sd in self.mdg.subdomains():
            if "injection_well" in sd.tags:  # and stride is not None:
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
            elif do_flash:
                nfi = self.local_equilibrium(sd, state=state, return_num_iter=True)  # type:ignore
            else:
                cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
                    self,
                    state,
                )

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


class Permeability(pp.PorePyModel):
    """Custom permeability with a higher absolute permability around the wells and a
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


class _ModelMixins(
    Permeability,
    PointWells2D,
    AdjustedWellModel2D,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SolutionStrategy,
):
    """Collection of used mixins, including a quadratic relative permeability law."""


class ColdCO2InjectionModel(_ModelMixins, cfle.EnthalpyBasedCFLETemplate):  # type:ignore
    """2-phase 2-component model simulating the injection of a cold water-co2 mixture
    into an initially hot and water saturated domain."""


class ColdCO2InjectionModelFF(_ModelMixins, cfle.EnthalpyBasedCFFLETemplate):  # type:ignore
    """Analogous to class:`ColdCO2InjectionModel` but based on the fractional flow
    template."""
