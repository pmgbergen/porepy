import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.applications.md_grids.domains import nd_cube_domain

import os
import sys
import pdb

os.system("clear")


class ModifiedGeometry:
    def set_domain(self) -> None:
        """ """
        size = self.solid.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """ """
        frac_1_points = self.solid.convert_units(
            np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        """ """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """ """
        cell_size = self.solid.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class ModifiedBC:
    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """ """
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_darcy(self, subdomains: list[pp.Grid]):
        """ """
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            val_loc[bounds.west] = self.fluid.convert_units(5, "Pa")
            val_loc[bounds.east] = self.fluid.convert_units(2, "Pa")
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


class SinglePhaseFlowGeometryBC(ModifiedGeometry, ModifiedBC, SinglePhaseFlow):
    """ """

    ...


fluid_constants = pp.FluidConstants({"viscosity": 0.1, "density": 0.2})
solid_constants = pp.SolidConstants({"permeability": 0.5, "porosity": 0.25})
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
params = {"material_constants": material_constants}


model = SinglePhaseFlowGeometryBC(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(
    model.mdg,
    "pressure",
    figsize=(10, 8),
    linewidth=0.25,
    title="Pressure distribution",
)

pdb.set_trace()

print("\n\nDone!")
