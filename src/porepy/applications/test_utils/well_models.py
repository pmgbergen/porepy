"""Contains code for setting up a simple but non-trivial model with a well."""

from typing import Literal, Sequence

import numpy as np

import porepy as pp


class OneVerticalWell(pp.PorePyModel):
    def set_wells(self) -> None:
        """Set wells in the well network."""
        self._wells = [pp.Well(np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 1.0]]))]

    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.units.convert_units(1, "m")
        # Set default values, then update with any user-provided meshing arguments.
        h = 0.15 * ls
        mesh_sizes = {
            "cell_size_fracture": h,
            "cell_size_boundary": h,
            "cell_size_min": 0.2 * h,
        }
        mesh_sizes.update(self.params.get("meshing_args", {}))
        return mesh_sizes

    def well_meshing_arguments(self) -> dict:
        mesh_size = self.units.convert_units(1 / 10.0, "m")
        return {"cell_size": mesh_size}

    def grid_type(self) -> Literal["simplex", "cartesian"]:
        return self.params.get("grid_type", "simplex")


class OneSlantedWell(pp.PorePyModel):
    """Model with one slanted well.

    If used with a unit cube domain, the well starts at the top at x=0.25 and ends
    almost at the bottom at x=0.75. The y coordinate is constant at 0.3.

    """

    def set_wells(self) -> None:
        """Set wells in the well network."""
        self._wells = [pp.Well(np.array([[0.25, 0.3], [0.75, 0.3], [1.0, 0.2]]))]

    def well_meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.units.convert_units(1 / 10.0, "m")
        # Set default values, then update with any user-provided meshing arguments.
        mesh_sizes = {"cell_size": ls}
        return self.params.get("well_meshing_args", mesh_sizes)


class BoundaryConditionsWellSetup(pp.PorePyModel):
    """Boundary conditions for the well setup."""

    def _bc_type(self, sd: pp.Grid, well_cond: str) -> pp.BoundaryCondition:
        """Boundary condition type for well-related boundaries.

        If `sd` has dimension 1, `well_cond` will be assigned on the top and bottom
        faces of `sd`. If `sd` has a different dimension, Dirichlet conditions are
        assigned on the top and bottom faces.

        Parameters:
            sd: Subdomain for which to define boundary conditions.
            well_cond: Boundary condition type to assign on well grids.

        Returns:
            Boundary condition object.

        """
        cond = well_cond if sd.dim == 1 else "dir"

        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.top + domain_sides.bottom, cond)

    def _bc_values(self, bg: pp.BoundaryGrid, value: float) -> np.ndarray:
        """Assign a boundary value on the top faces of well boundary grids.

        For 0D boundary grids, values are assigned on the top face of the well.
        All other boundary values are zero.

        Parameters:
            bg: Boundary grid for which to define boundary conditions.
            value: Value to assign.

        Returns:
            Boundary condition values array.

        """

        vals_loc = np.zeros(bg.num_cells)
        if bg.dim == 0:
            domain_sides = self.domain_boundary_sides(bg)
            # Inflow for the top boundary of the well.
            vals_loc[domain_sides.top] = value
        return vals_loc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Neumann boundary conditions are defined on the top and bottom boundaries
        if `sd` has dimension 1. If `sd` has a different dimension, Dirichlet
        conditions are assigned on the top and bottom faces.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        return self._bc_type(sd, "neu")

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for mobility-free Darcy flux.

        For 0D boundary grids, mobility-free Darcy-flux values are assigned
        on the top face of the well. The value is taken from the `well_flux`
        model parameter. All other boundary values are zero.

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        value = self.units.convert_units(self.params.get("well_flux", -1), "Pa * m")
        return self._bc_values(bg, value)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for mass fluid flux.

        Dirichlet boundary conditions are defined on the top and bottom boundaries
        of the subdomain.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        return self._bc_type(sd, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for enthalpy flux.

        Dirichlet boundary conditions are defined on the top and bottom boundaries
        of the subdomain.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        return self._bc_type(sd, "dir")

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """
        Parameters:
            bg: A boundary grid in the domain.

        Returns:
            Numeric enthalpy flux values for a Neumann-type BC.

        """
        val = self.units.convert_units(self.params.get("well_enthalpy", 1e7), "K")
        return self._bc_values(bg, val)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        Neumann boundary conditions are defined on the top and bottom boundaries
        if `sd` has dimension 1. If `sd` has a different dimension, Dirichlet
        conditions are assigned on the top and bottom faces.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        return self._bc_type(sd, "neu")


class BoundaryConditionsOneRateOnePressureWell(BoundaryConditionsWellSetup):
    """Rate control on one well and pressure control on another.

    The rate-controlled well is driven by the ``well_flux`` parameter as in
    :class:`BoundaryConditionsWellSetup`. Use with a model of more than one well, such
    as :class:`~porepy.applications.md_grids.model_geometries.TwoWells3d`. The well
    named by :attr:`pressure_controlled_well` is instead held at the ``well_pressure``
    parameter, and therefore produces wherever the formation around it stands above
    that pressure. The direction of flow across its contacts is thus an outcome of the
    simulation rather than something imposed, which is what makes the setup useful for
    testing the upwind direction of the advected quantities.

    """

    pressure_controlled_well: int = 1
    """Number of the well held at a fixed pressure. The others are rate-controlled."""

    def is_pressure_controlled(self, grid: pp.Grid | pp.BoundaryGrid) -> bool:
        """Whether a grid belongs to the pressure-controlled well.

        Parameters:
            grid: A subdomain, or a boundary grid of one.

        Returns:
            True if the grid is, or bounds, the pressure-controlled well.

        """
        subdomain = grid.parent if isinstance(grid, pp.BoundaryGrid) else grid
        return getattr(subdomain, "well_num", -1) == self.pressure_controlled_well

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        The pressure-controlled well is given a Dirichlet condition where the
        rate-controlled wells are given a Neumann one.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        if self.is_pressure_controlled(sd):
            return self._bc_type(sd, "dir")
        return super().bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for pressure.

        The pressure of the pressure-controlled well is taken from the
        ``well_pressure`` model parameter and assigned at the top of that well, where
        it is open. All other boundary values are left to the superclass.

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        if not self.is_pressure_controlled(bg):
            return super().bc_values_pressure(bg)  # type: ignore[misc]
        value = self.units.convert_units(self.params.get("well_pressure", -1), "Pa")
        return self._bc_values(bg, value)


class TracerWellFluid:
    """A two-component fluid whose second component marks the fluid in the well.

    The components are inert and share unit properties, so the tracer only records
    where fluid has travelled. That is what makes it useful for testing the component
    balance across a well contact: the overall fraction differs sharply between the
    well and the rock, so a component flux advected from the wrong side is far from
    the right answer rather than close to it.

    """

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Mixed in method defining the reference component and an inert tracer."""
        return [
            pp.FluidComponent(name="fluid", compressibility=0, density=1, viscosity=1),
            pp.FluidComponent(name="tracer"),
        ]


class TracerInTheWell(pp.PorePyModel):
    """The tracer of :class:`TracerWellFluid` starts, and is injected, in the wells.

    The rock and the fractures start free of it, so any tracer found there arrived
    through a well contact. Mix in ahead of
    :class:`~porepy.models.compositional_flow.BoundaryConditionsMulticomponent` and
    :class:`~porepy.models.compositional_flow.InitialConditionsFractions`, which supply
    the machinery these values are read by.

    """

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Tracer enters through the well boundaries and nowhere else.

        Parameters:
            component: The component whose overall fraction is asked for.
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary values of the overall fraction.

        """
        assert component.name == "tracer", "Only the tracer is independent."
        return np.full(bg.num_cells, 1.0 if bg.parent.dim == 1 else 0.0)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """The wells start full of tracer and everything else free of it.

        Parameters:
            component: The component whose overall fraction is asked for.
            sd: Subdomain for which to define initial conditions.

        Returns:
            Initial values of the overall fraction.

        """
        assert component.name == "tracer", "Only the tracer is independent."
        return np.full(sd.num_cells, 1.0 if sd.dim == 1 else 0.0)


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
            sd for sd in subdomains if sd.dim < self.nd and (not self.is_well_grid(sd))
        ]
        wells = [sd for sd in subdomains if self.is_well_grid(sd)]

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
