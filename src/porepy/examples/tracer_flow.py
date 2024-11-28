"""Module containing a simple tracer flow setup, modelled as a single phase, 2-component
flow."""

from __future__ import annotations

import pathlib
from typing import Literal, Sequence, TypedDict, cast

import numpy as np

import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    _BoundaryConditionsAdvection,  # TODO remove once fixed
)
from porepy.models.compositional_flow import (
    BoundaryConditionsFractions,
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


class DomainX(pp.PorePyModel):

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

        component_1 = pp.FluidComponent(name="water", **cast(WaterDict, water))
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


def inlet_faces(bg: pp.BoundaryGrid, sides: pp.domain.DomainSides) -> np.ndarray:
    """Helper function to define a snippet of the western boundary as the inlet."""

    inlet = np.zeros(bg.num_cells, dtype=bool)
    inlet[sides.west] = True
    inlet &= bg.cell_centers[1] >= 0.4
    inlet &= bg.cell_centers[1] <= 0.6

    return inlet


def outlet_faces(bg: pp.BoundaryGrid, sides: pp.domain.DomainSides) -> np.ndarray:
    """Helper function to define a snippet of the eastern boundary as the inlet."""

    outlet = np.zeros(bg.num_cells, dtype=bool)
    outlet[sides.east] = True
    outlet &= bg.cell_centers[1] >= 0.4
    outlet &= bg.cell_centers[1] <= 0.6

    return outlet


# defining inlet and outlet pressure
p_OUTLET = 1.0
p_INLET = 1.5
# defining the amount of tracer comming into the domain
z_inlet = 0.1


class TracerIC(InitialConditionsFractions):

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Setting initial pressure equal to pressure on outflow boundary."""
        return np.ones(sd.num_cells) * p_OUTLET

    def initial_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        if component.name == "tracer":
            return np.zeros(sd.num_cells)
        else:
            assert False, "This will never happen since water is a dependent component."


class TracerBC(BoundaryConditionsFractions, _BoundaryConditionsAdvection):

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

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Defines some non-trivial values on inlet and outlet faces of the matrix."""

        p = np.zeros(bg.num_cells)

        # defining BC values only on matrix
        if bg.parent.dim == 2:
            sides = self.domain_boundary_sides(bg)
            inlet = inlet_faces(bg, sides)
            outlet = outlet_faces(bg, sides)

            p[inlet] = p_INLET
            p[outlet] = p_OUTLET

        return p

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Defines some non-trivial inflow of the tracer component on the inlet."""

        z = np.zeros(bg.num_cells)

        if bg.parent.dim == 2:
            if component.name == "tracer":
                sides = self.domain_boundary_sides(bg)
                inlet = inlet_faces(bg, sides)

                z[inlet] = z_inlet
            else:
                assert (
                    False
                ), "This will never happen since water is a dependent component."

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


solid = pp.SolidConstants(porosity=0.1, permeability=1e-11, normal_permeability=1e-19)

# initial time step 1 second
dt_init = 1
# Simulation time 1 hour
T_end = dt_init * 60 * 60  # simulating 1 hour
# min max time step size is 1/10 second and 1 minute respectively
dt_min_max = (0.1 * dt_init, 60 * dt_init)
# parameters for Newton solver
max_iterations = 80
newton_tol = 1e-8
newton_tol_increment = newton_tol

time_manager = pp.TimeManager(
    schedule=np.arange(0, T_end, 10 * 60),  # schedule every 10 minutes
    dt_init=dt_init,
    dt_min_max=(0.1 * dt_init, 60 * dt_init),
    iter_max=max_iterations,
    iter_optimal_range=(2, 10),
    iter_relax_factors=(0.9, 1.1),
    recomp_factor=0.5,
    recomp_max=5,
)

params = {
    "material_constants": {
        # solid with impermeable fractures
        "solid": pp.SolidConstants(
            porosity=0.1, permeability=1e-11, normal_permeability=1e-19
        ),
    },
    # The respective DOFs are eliminated by default. These flags are for demonstration.
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    # We use an inconsistent discretization in the pressure equation, no fractional flow
    "fractional_flow": False,
    "time_manager": time_manager,
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "restart_options": {
        "restart": True,
        "is_mdg_pvd": True,
        "pvd_file": pathlib.Path("./visualization/data.pvd"),
    },
}

# NOTE mypy has some issues with the composed CF protocol, which is part of some of the
# mixins. For some reasons it thinks this class has tracer_transport_equation_names
# as an attribute, even though it does not have that mixin among the base classes.
model = TracerFlowSetup(params)  # type:ignore[abstract]
pp.run_time_dependent_model(model, params)
pp.plot_grid(
    model.mdg,
    "pressure",
    figsize=(10, 8),
    linewidth=0.2,
    title="Pressure distribution",
    plot_2d=True,
    if_plot=True,
)
pp.plot_grid(
    model.mdg,
    "z_tracer",
    figsize=(10, 8),
    linewidth=0.2,
    title="Pressure distribution",
    plot_2d=True,
    if_plot=True,
)
