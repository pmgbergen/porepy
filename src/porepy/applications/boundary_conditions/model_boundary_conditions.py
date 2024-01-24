"""Custom classes for model boundary conditions."""

from __future__ import annotations

from typing import Any

import numpy as np

import porepy as pp


class BoundaryConditionsMassDirWestEast(pp.BoundaryConditionMixin):
    """Boundary conditions for the flow problem.

    Dirichlet boundary conditions are defined on the west and east boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    Usage: two-dimensional flow benchmark models.

    """

    fluid: pp.FluidConstants

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the west and east boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        vals_loc = np.zeros(boundary_grid.num_cells)
        vals_loc[domain_sides.west + domain_sides.east] = self.fluid.pressure()
        return vals_loc

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the density-mobility product.

        Dirichlet boundary conditions are defined on the west and east boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")


class BoundaryConditionsMassDirNorthSouth(pp.BoundaryConditionMixin):
    """Boundary conditions for the flow problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    fluid: pp.FluidConstants

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        vals_loc = np.zeros(boundary_grid.num_cells)
        vals_loc[domain_sides.north + domain_sides.south] = self.fluid.pressure()
        return vals_loc

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the density-mobility product.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")


class BoundaryConditionsEnergyDirNorthSouth(pp.BoundaryConditionMixin):
    """Boundary conditions for the thermal problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")


class BoundaryConditionsMechanicsDirNorthSouth(pp.BoundaryConditionMixin):
    """Boundary conditions for the mechanics with Dirichlet conditions on north and
    south boundaries.

    """

    params: dict[str, Any]
    """Model parameters."""
    solid: pp.SolidConstants
    """Solid parameters."""
    fluid: pp.FluidConstants
    """Fluid parameters."""
    nd: int
    """Number of dimensions."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, domain_sides.north + domain_sides.south, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Values for north and south faces are set to zero unless otherwise specified
        through items ux_north, uy_north, ux_south, uy_south in the parameter dictionary
        passed on model initialization.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
        values[1, domain_sides.north] = self.solid.convert_units(
            self.params.get("uy_north", 0), "m"
        )
        values[1, domain_sides.south] = self.solid.convert_units(
            self.params.get("uy_south", 0), "m"
        )
        values[0, domain_sides.north] = self.solid.convert_units(
            self.params.get("ux_north", 0), "m"
        )
        values[0, domain_sides.south] = self.solid.convert_units(
            self.params.get("ux_south", 0), "m"
        )
        return values.ravel("F")


class TimeDependentMechanicalBCsDirNorthSouth(BoundaryConditionsMechanicsDirNorthSouth):
    """Time dependent displacement boundary conditions.

    For use in (thermo)poremechanics.

    """

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Initial value is u_y = 0.042 at north boundary. This is a hard-coded value
        corresponding to aperture of horizontal fracture used in various tests. Adding
        it on the boundary ensures a stress-free initial state. For positive times,
        uy_north and uy_south are fetched from parameter dictionary and added,
        defaulting to 0.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
        # Add fracture width on top if there is a fracture.
        if len(self.mdg.subdomains()) > 1:
            frac_val = self.solid.convert_units(0.042, "m")
        else:
            frac_val = 0
        values[1, domain_sides.north] = frac_val
        if self.time_manager.time > 1e-5:
            values[1, domain_sides.north] += self.solid.convert_units(
                self.params.get("uy_north", 0), "m"
            )
            values[1, domain_sides.south] += self.solid.convert_units(
                self.params.get("uy_south", 0), "m"
            )
        return values.ravel("F")
