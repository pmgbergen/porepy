"""Profiling the cost of assembling equations with empty domains of definition.

Assembles a single-phase flow model on a non-fractured 2D domain, where
interface and well equations have zero DOFs. Compares full assembly time
against assembly with empty equations removed, across varying grid
resolutions.

"""

from typing import Literal
import time
import numpy as np

import porepy as pp

solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * 1e-6,
    density=2700.0,
    specific_heat_capacity=880.0 * 1e-6,
)
material_constants = {"solid": solid_constants}


class NonFracturedDomain(pp.PorePyModel):
    """Geometry specification for a non-fractured domain."""

    def set_domain(self) -> None:
        """Domain."""
        x_extent = self.units.convert_units(700, "m")
        y_extent = self.units.convert_units(600, "m")
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": x_extent, "ymin": 0, "ymax": y_extent}
        )
    
    def grid_type(self) -> Literal["simplex"]:
        """Set a simplex grid, which is the only grid type that can represent this
        fracture geometry.
        """
        return "simplex"

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for grid generation.
        """
        # Default value of 10.0, scaled by the length unit.
        cell_size = self.units.convert_units(10.0, "m")
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        return self.params.get("meshing_arguments", default_meshing_args)


class BoundaryConditions(pp.PorePyModel):
    """Use default BC implementation
    """
    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Pressure value of one atmosphere (101325 Pa) on west side.
        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.west] = self.units.convert_units(101325, "Pa")
        return values

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the east and west boundary. The rest are Neumann by
        default.
        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.east + domain_sides.west, "dir")
        return bc

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Inflow on the west boundary.
        """
        values = np.zeros(bg.num_cells)
        return values


class InitialConditions(pp.PorePyModel):
    """Use default IC implementation
    """
    pass


class SinglePhaseFlowNonFractured(  # type: ignore[misc]
    NonFracturedDomain,
    BoundaryConditions,
    InitialConditions,
    pp.SinglePhaseFlow,
):
    """Single-phase flow model on a non-fractured domain for profiling."""
    pass


# Model parameters
model_params = {
    "material_constants": material_constants,
    "grid_type": "simplex",
    "time_manager": pp.TimeManager(
        dt_init=1,
        schedule=[0, 4],
        constant_dt=True,
    ),
}

# Set different gridcell sizes.
cell_sizes = [1, 2, 5, 10]

# Inspect registered equations and their sizes
# Identify which equations are empty
print("\n--- Assembly cost of empty equations (non-fractured domain) ---")
for cell_size in cell_sizes:
    cell_size_m = pp.Units().convert_units(cell_size, "m")
    model_params_i = {
        **model_params,
        "meshing_arguments": {"cell_size": cell_size_m},
    }
    model_i = SinglePhaseFlowNonFractured(params=model_params_i)
    model_i.prepare_simulation()
    eq_system = model_i.equation_system

    # Identify which equations are empty
    empty_eqs = []
    for name, eq in eq_system.equations.items():
        # Each equation has associated subdomains — check their total DOFs
        image_info = eq_system.equation_image_space_composition[name]
        total_dofs = sum(len(indices) for indices in image_info.values())
        if total_dofs == 0:
            empty_eqs.append(name)

    # Time spent in assembling full equation system.
    t0 = time.perf_counter()
    A_full, b_full = eq_system.assemble()
    t_full = time.perf_counter() - t0

    # Remove empty equations, reassemble.
    for name in empty_eqs:
        eq_system._equations.pop(name)

    # Time spent in assembling reduced equation system with empty equations removed.
    t0 = time.perf_counter()
    A_reduced, b_reduced = eq_system.assemble()
    t_reduced = time.perf_counter() - t0

    num_cells = sum(sd.num_cells for sd in model_i.mdg.subdomains())
    print(f"cell_size={cell_size:>3}, cells={num_cells:>7}, DOFs={A_full.shape[0]:>7}, "
          f"time_full={t_full:.3f}s, time_reduced={t_reduced:.3f}s, "
          f"savings={t_full-t_reduced:.3f}s ({(t_full-t_reduced)/t_full*100:.1f}%)")