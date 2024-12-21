"""Module containing a simple tracer flow setup, modelled as a single phase, 2-component
flow."""

from __future__ import annotations

from typing import Literal, Sequence, TypedDict, cast

import numpy as np

import porepy as pp
from porepy.applications.material_values.fluid_values import water as _water
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    BoundaryConditionsMulticomponent,
    ComponentMassBalanceEquations,
    InitialConditionsFractions,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow


class WaterDict(TypedDict):
    """For mypy when using **water in combination with FluidComponent."""

    compressibility: float
    density: float
    specific_heat_capacity: float
    thermal_conductivity: float
    thermal_expansion: float
    viscosity: float
    name: str


# Casting this to help mypy
water = cast(WaterDict, _water)


class DomainX(pp.PorePyModel):
    """Unit square domain with two line fractures forming a X."""

    def set_domain(self) -> None:
        """Setting a 2d unit square as matrix."""
        size = self.units.convert_units(1, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting 2 fractures in x shape."""
        frac_1_points = self.units.convert_units(
            np.array([[0.2, 0.8], [0.2, 0.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        frac_2_points = self.units.convert_units(
            np.array([[0.2, 0.8], [0.8, 0.2]]), "m"
        )
        frac_2 = pp.LineFracture(frac_2_points)
        self._fractures = [frac_1, frac_2]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.05, "m")
        cell_size_fracture = self.units.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        }
        return mesh_args


class TracerFluid:
    """Setting up a 2-component fluid."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Mixed in method defining water as the reference component and a simple
        tracer as the second component."""

        component_1 = pp.FluidComponent(**water)
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


def inlet_faces(bg: pp.BoundaryGrid, sides: pp.domain.DomainSides) -> np.ndarray:
    """Helper function to define a snippet of the western boundary as the inlet."""

    inlet = np.zeros(bg.num_cells, dtype=bool)
    inlet[sides.north] = True
    inlet &= bg.cell_centers[0] >= 0.4
    inlet &= bg.cell_centers[0] <= 0.6

    return inlet


def outlet_faces(bg: pp.BoundaryGrid, sides: pp.domain.DomainSides) -> np.ndarray:
    """Helper function to define a snippet of the eastern boundary as the inlet."""

    outlet = np.zeros(bg.num_cells, dtype=bool)
    outlet[sides.south] = True
    outlet &= bg.cell_centers[0] >= 0.4
    outlet &= bg.cell_centers[0] <= 0.6

    return outlet


class TracerIC(InitialConditionsFractions):
    """Initial conditions for pressure and tracer fraction.

    Mixes in the initial pressure values, and inherits the IC treatment for
    multi-component fluids.

    """

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Setting initial pressure equal to pressure on outflow boundary."""
        # Initial and outlet pressure are the same.
        return np.ones(sd.num_cells)

    def initial_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """Setting initial tracer overall fraction to zero."""

        assert component.name == "tracer", "Only the tracer is independent."

        # No tracer in the domain at the beginning.
        return np.zeros(sd.num_cells)


class TracerBC(BoundaryConditionsMulticomponent):
    """Boundary conditions for pressure, flow and tracer.

    Mixes in the BC for pressure and the boundary type definition, and inherits the
    BC treatment for multi-component fluids.

    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Flagging the inlet and outlet faces as Dirichlet boundary, where pressure
        is given."""
        dirichlet_faces = np.zeros(sd.num_faces, dtype=bool)
        # Define boundary faces on grids which are not points:
        if sd.dim > 0:
            sides = self.domain_boundary_sides(sd)
            # need to cast, bg exists if sd is not point
            bg = cast(pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(sd))
            bg_sides = self.domain_boundary_sides(bg)
            inlet = inlet_faces(bg, bg_sides)
            outlet = outlet_faces(bg, bg_sides)

            dirichlet = np.zeros(bg.num_cells, dtype=bool)
            dirichlet[inlet | outlet] = True

            # broadcast to proper size
            dirichlet_faces[sides.all_bf] = dirichlet

        return pp.BoundaryCondition(sd, dirichlet_faces, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns for the upwinding discretization the same inlet and outlet faces
        marked as 'dir' as for the elliptic discretization."""
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Defines some non-trivial values on inlet and outlet faces of the matrix."""

        p = np.zeros(bg.num_cells)

        # defining BC values only on matrix
        if bg.parent.dim == 2:
            sides = self.domain_boundary_sides(bg)
            inlet = inlet_faces(bg, sides)
            outlet = outlet_faces(bg, sides)

            p[inlet] = 1.5
            p[outlet] = 1.0

        return p

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Defines some non-trivial inflow of the tracer component on the inlet."""

        z = np.zeros(bg.num_cells)

        assert component.name == "tracer", "Only the tracer is independent."

        # 10% tracer inflow into matrix
        if bg.parent.dim == 2:
            sides = self.domain_boundary_sides(bg)
            inlet = inlet_faces(bg, sides)

            z[inlet] = 0.1

        return z


class TracerFlowSetup(  # type: ignore[misc]
    DomainX,
    TracerFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    TracerBC,
    TracerIC,
    SinglePhaseFlow,
):
    """Complete set-up for tracer flow modelled as a single phase, 2-component flow
    problem."""


# If executed as main, run simulation
if __name__ == "__main__":
    # initial time step 60 seconds
    dt_init = 60
    # Simulation time 2 hour
    T_end = 2 * 60 * 60
    # min max time step size is 6 seconds and 10 minutes respectively
    dt_min_max = (0.1 * dt_init, 10 * 60)
    # parameters for Newton solver
    max_iterations = 80
    newton_tol = 1e-6
    newton_tol_increment = newton_tol

    time_manager = pp.TimeManager(
        schedule=[0, T_end],
        dt_init=dt_init,
        dt_min_max=dt_min_max,
        iter_max=max_iterations,
        iter_optimal_range=(2, 10),
        iter_relax_factors=(0.8, 1.2),
        recomp_factor=0.8,
        recomp_max=5,
    )

    params = {
        "material_constants": {
            # solid with impermeable fractures
            "solid": pp.SolidConstants(
                porosity=0.1, permeability=1e-7, normal_permeability=1e-19
            ),
        },
        # The respective DOFs are eliminated by default. These flags are for
        # demonstration.
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        # We use upwinding in the pressure equation, no fractional flow (default).
        "fractional_flow": False,
        "time_manager": time_manager,
        "max_iterations": max_iterations,
        "nl_convergence_tol": newton_tol_increment,
        "nl_convergence_tol_res": newton_tol,
        "progressbars": True,
    }

    # NOTE mypy has some issues with the composed CF protocol, which is part of some of
    # the mixins.
    model = TracerFlowSetup(params)  # type:ignore[abstract]
    pp.run_time_dependent_model(model, params)
    pp.plot_grid(
        model.mdg,
        "pressure",
        figsize=(10, 8),
        linewidth=0.2,
        title="Pressure distribution",
        plot_2d=True,
    )
    pp.plot_grid(
        model.mdg,
        "z_tracer",
        figsize=(10, 8),
        linewidth=0.2,
        title="Tracer distribution after 2 hours",
        plot_2d=True,
    )
