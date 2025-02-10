"""Module containing a simple tracer flow setup, modelled as a single phase, 2-component
flow."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMassDirNorthSouth,
)
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    BoundaryConditionsMulticomponent,
    ComponentMassBalanceEquations,
    InitialConditionsFractions,
)


class TracerFluid:
    """Setting up a 2-component fluid."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Mixed in method defining water as the reference component and a simple
        tracer as the second component."""

        component_1 = pp.FluidComponent(**water)
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


class TracerIC(InitialConditionsFractions):
    """Initial conditions for pressure and tracer fraction.

    Mixes in the initial pressure values, and inherits the IC treatment for
    multi-component fluids.

    """

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Setting initial pressure equal to pressure on outflow boundary."""
        # Initial and outlet pressure are the same.
        return self.reference_variable_values.pressure * np.ones(sd.num_cells)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """Setting initial tracer overall fraction to zero."""

        assert component.name == "tracer", "Only the tracer is independent."

        # No tracer in the domain at the beginning.
        return np.zeros(sd.num_cells)


class TracerBC(BoundaryConditionsMassDirNorthSouth, BoundaryConditionsMulticomponent):
    """Boundary conditions for pressure, flow and tracer.

    Mixes in the BC for pressure and the boundary type definition, and inherits the
    BC treatment for multi-component fluids.

    """

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        The pressure at most of the boundary is inherited from
        BoundaryConditionsMassDirNorthSouth, hence constant. On the north side, add a
        pressure equal to the x-coordinate along the boundary.
        """
        values = super().bc_values_pressure(boundary_grid)
        domain_sides = self.domain_boundary_sides(boundary_grid)
        values[domain_sides.north] = (
            self.reference_variable_values.pressure
            + boundary_grid.cell_centers[0, domain_sides.north]
        )

        return values

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Defines some non-trivial inflow of the tracer component on the inlet
        (north)."""

        z = np.zeros(boundary_grid.num_cells)

        assert component.name == "tracer", "Only the tracer is independent."

        # Set the tracer concentration to 0.1 on the left half of the north boundary,
        # and 0.2 on the right half.
        if boundary_grid.parent.dim == 2:
            domain_sides = self.domain_boundary_sides(boundary_grid)
            z[domain_sides.north] = 0.1 + 0.1 * (
                boundary_grid.cell_centers[0, domain_sides.north] > 0.5
            )

        return z


class TracerFlowSetup(  # type: ignore[misc]
    SquareDomainOrthogonalFractures,
    TracerFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    TracerBC,
    TracerIC,
    pp.SinglePhaseFlow,
):
    """Complete set-up for tracer flow modelled as a single phase, 2-component flow
    problem."""


# If executed as main, run simulation
if __name__ == "__main__":
    # Initial time step 60 seconds.
    dt_init = pp.MINUTE
    # Simulation time 20 minutes.
    T_end = 20 * pp.MINUTE
    # min max time step size is 6 seconds and 10 minutes respectively
    dt_min_max = (0.1 * dt_init, 10 * pp.MINUTE)
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
            # Solid with impermeable fractures.
            "solid": pp.SolidConstants(
                porosity=0.1, permeability=1e-7, normal_permeability=1e-19
            ),
        },
        "fracture_indices": [0, 1],
        # The respective DOFs are eliminated by default. These flags are for
        # demonstration.
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "time_manager": time_manager,
        "max_iterations": max_iterations,
        "nl_convergence_tol": newton_tol_increment,
        "nl_convergence_tol_res": newton_tol,
        "meshing_arguments": {"cell_size": 0.05},
        "grid_type": "simplex",
    }

    model = TracerFlowSetup(params)
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
        title="Tracer distribution after 20 minutes",
        plot_2d=True,
    )
