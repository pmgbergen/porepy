import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import SinglePhaseFlow
import logging
from porepy.applications.md_grids.domains import nd_cube_domain

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = 2 / self.units.m
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1 = pp.LineFracture(np.array([[0.2, 1.8], [0.2, 1.8]]) / self.units.m)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method to assign simplex instead.

        """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        mesh_args: dict[str, float] = {"cell_size": 0.05 / self.units.m}
        return mesh_args

    # def before_nonlinear_loop(self) -> None:
    #     """Method to be called before entering the non-linear solver, thus at the start
    #     of a new time step.

    #     Possible usage is to update time-dependent parameters, discretizations etc.

    #     """
    #     print("test")
    #     super().bef


class ModifiedBC:
    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_darcy(self, subdomains: list[pp.Grid]):
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        # Define boundary regions
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            # See section on scaling for explanation of the conversion.
            val_loc[bounds.west] = self.fluid.convert_units(5, "Pa")
            val_loc[bounds.east] = self.fluid.convert_units(2, "Pa")
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


class ModifiedSource:
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assign unitary fracture source"""
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []

        for sd in subdomains:
            if sd.dim == self.mdg.dim_max():
                values.append(np.zeros(sd.num_cells))
            else:
                values.append(np.ones(sd.num_cells))

        external_sources = pp.wrap_as_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


class CompressibleFlow(ModifiedGeometry, ModifiedBC, ModifiedSource, SinglePhaseFlow):
    """Combining modified geometry, boundary conditions and the source term with the default model."""

    ...


fluid_constants = pp.FluidConstants({"compressibility": 0.01})
material_constants = {"fluid": fluid_constants}

time_manager = pp.TimeManager(
    schedule=[0, 0.05],
    dt_init=1e-4,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "time_manager": time_manager,
    "progressbars": True,
}
model = CompressibleFlow(params)


pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), linewidth=0.2)
