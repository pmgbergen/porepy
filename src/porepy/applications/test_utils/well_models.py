"""Contains code for setting up a simple but non-trivial model with a well.
"""

import numpy as np

import porepy as pp


class OneVerticalWell:
    domain: pp.Domain
    """Domain for the model."""

    solid: pp.SolidConstants

    def set_well_network(self) -> None:
        """Assign well network class."""
        points = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 1]])
        mesh_size = self.solid.convert_units(1 / 10.0, "m")
        self.well_network = pp.WellNetwork3d(
            domain=self.domain,
            wells=[pp.Well(points)],
            parameters={"mesh_size": mesh_size},
        )

    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")
        h = 0.15 * ls
        mesh_sizes = {
            "cell_size_fracture": h,
            "cell_size_boundary": h,
            "cell_size_min": 0.2 * h,
        }

        return mesh_sizes

    def grid_type(self) -> str:
        return "simplex"


class BoundaryConditionsWellSetup(pp.BoundaryConditionMixin):
    """Boundary conditions for the well setup."""

    fluid: pp.FluidConstants

    params: dict[str, float]
    """Model parameters."""

    def _bc_type(self, sd: pp.Grid, well_cond: str) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        If `sd` has dimension 1, `well_cond` will be assigned on the top and bottom
        faces of `sd`. If `sd` has a different dimension, Dirichlet conditions are
        assigned on the top and bottom faces.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        cond = well_cond if sd.dim == 1 else "dir"

        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.top + domain_sides.bottom, cond)

    def _bc_values(self, boundary_grid: pp.BoundaryGrid, value: float) -> np.ndarray:
        """
        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.
            value: Value to assign.

        Returns:
            bc: Boundary condition array.

        """

        vals_loc = np.zeros(boundary_grid.num_cells)
        if boundary_grid.dim == 0:
            domain_sides = self.domain_boundary_sides(boundary_grid)
            # Inflow for the top boundary of the well.
            vals_loc[domain_sides.top] = value
        return vals_loc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "neu")

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value of 0 unless fluid's reference pressure is changed.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        value = self.fluid.convert_units(
            self.params.get("well_flux", -1), "kg * m ^ 3 * s ^ -1"
        )
        return self._bc_values(boundary_grid, value)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type(sd, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "dir")

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """
        Parameters:
            boundary_grids: A boundary grid in the domain.

        Returns:
            Numeric enthalpy flux values for a Neumann-type BC.

        """
        val = self.fluid.convert_units(self.params.get("well_enthalpy", 1e7), "K")
        return self._bc_values(boundary_grid, val)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "neu")


class WellPermeability(pp.constitutive_laws.CubicLawPermeability):
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        This function is an extension of the CubicLawPermeability class which includes
        well permeability.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=9)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures_and_intersections: list[pp.Grid] = [
            sd
            for sd in subdomains
            if sd.dim < self.nd and ("parent_well_index" not in sd.tags)
        ]
        wells = [sd for sd in subdomains if "parent_well_index" in sd.tags]

        permeability = (
            projection.cell_prolongation(matrix) @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(fractures_and_intersections)
            @ self.cubic_law_permeability(fractures_and_intersections)
            + projection.cell_prolongation(wells) @ self.well_permeability(wells)
        )
        return permeability

    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(1, size, name="well permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)
