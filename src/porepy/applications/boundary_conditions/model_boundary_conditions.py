"""Custom classes for model boundary conditions."""

from __future__ import annotations

import warnings
from typing import Callable, Literal

import numpy as np

import porepy as pp


class BoundaryConditionsMassDirWestEast(pp.PorePyModel):
    """Boundary conditions for the flow problem.

    Dirichlet boundary conditions are defined on the west and east boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 1d, 2d or 3d.

    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the west and east boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the west and east boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.west + domain_sides.east] = (
            self.reference_variable_values.pressure
        )
        return values

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


class BoundaryConditionsMassDirNorthSouth(pp.PorePyModel):
    """Boundary conditions for the flow problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    """

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

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.north + domain_sides.south] = (
            self.reference_variable_values.pressure
        )
        return values

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


class BoundaryConditionsEnergyDirNorthSouth(pp.PorePyModel):
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


class BoundaryConditionsMechanicsDirNorthSouth(pp.PorePyModel):
    """Boundary conditions for the mechanics with Dirichlet conditions on north and
    south boundaries.

    """

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

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Values for north and south faces are set to zero unless otherwise specified
        through items u_north and u_south in the parameter dictionary passed on model
        initialization.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))
        if bg.dim < self.nd - 1:
            # No displacement is implemented on grids of co-dimension >= 2.
            return values.ravel("F")

        if "uy_north" in self.params or "uy_south" in self.params:
            warnings.warn(
                "uy_north and uy_south are deprecated. Use u_north and u_south instead."
            )
        # Wrap as array for convert_units. Thus, the passed values can be scalar or
        # list. Then tile for correct broadcasting below.
        u_n = np.tile(
            self.params.get("u_north", np.zeros(self.nd)), (bg.num_cells, 1)
        ).T
        u_s = np.tile(
            self.params.get("u_south", np.zeros(self.nd)), (bg.num_cells, 1)
        ).T
        values[:, domain_sides.north] = self.units.convert_units(u_n, "m")[
            :, domain_sides.north
        ]
        values[:, domain_sides.south] = self.units.convert_units(u_s, "m")[
            :, domain_sides.south
        ]
        return values.ravel("F")


class TimeDependentMechanicalBCsDirNorthSouth(BoundaryConditionsMechanicsDirNorthSouth):
    """Time dependent displacement boundary conditions.

    For use in (thermo)poremechanics.

    """

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Initial value is u_y = self.solid.fracture_gap +
        self.solid.maximum_elastic_fracture_opening at north boundary. Adding it on
        the boundary ensures a stress-free initial state, as it compensates for those
        two values corresponding to zero traction contact according to the class
        :class:`~porepy.models.constitutive_laws.FractureGap`. For positive times,
        uy_north and uy_south are fetched from parameter dictionary and added,
        defaulting to 0.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))
        # Add fracture width on top if there is a fracture.
        if len(self.mdg.subdomains()) > 1:
            frac_val = (
                self.solid.fracture_gap + self.solid.maximum_elastic_fracture_opening
            )
        else:
            frac_val = 0
        values[1, domain_sides.north] = frac_val
        if self.time_manager.time > 1e-5:
            return values.ravel("F") + super().bc_values_displacement(bg)
        else:
            return values.ravel("F")


class BoundaryConditionsMechanicsNeumann:
    """Boundary conditions for the mechanics with Neumann conditions on almost all
    boundaries.

    The only exception is that internal boundaries are converted to Dirichlet and three
    points are partly fixed to avoid rigid body motions. We pick maximum z coordinate
    for all three points. The points have
            1) min x and max y coordinate, (fixed in y and z directions)
            2) max x and max y coordinate,  (fixed in y and z directions)
            3) mean x and min y coordinate. (fixed in x and z directions)
    Seen from above, this looks like:

                -------
      (no y) 1  |     | 2 (no y)
                |     |             ^ y
                |     |             |
                -------             +---> x
                   3 (no x)
    """

    domain: pp.domain.Domain
    """Model domain."""
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    """Function returning the domain boundary sides of a given grid."""
    nd: int
    """Number of spatial dimensions."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Neumann boundary conditions are defined on all boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "neu")
        if sd.dim < self.nd:
            # No displacement is implemented on grids of co-dimension >= 1.
            return bc
        bc.internal_to_dirichlet(sd)
        faces_to_fix = self.faces_to_fix(sd)
        # Fix y and z displacements on face 1 and 2, x and z displacements on face 3.
        dir = [
            np.array([False, True, True]),  # Fix y and z on face 1 (west)
            np.array([False, True, True]),  # Fix y and z on face 2 (east)
            np.array([True, False, True]),  # Fix x and z on face 3 (south)
        ]
        for i, face in enumerate(faces_to_fix):
            bc.is_dir[:, face] = dir[i]
            bc.is_neu[:, face] = ~dir[i]  # Negate for Neumann
        return bc

    def faces_to_fix(self, sd: pp.Grid) -> list[np.int64]:
        """Return list of faces to fix to avoid rigid body motions.

        See class documentation for more details on the choice of points.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            List of face indices to fix, ordered as in the class documentation.

        """
        domain_sides = self.domain_boundary_sides(sd)
        box = self.domain.bounding_box

        # Point 1 is on the center top of the west boundary, having min x coordinate and
        # y coordinate slightly smaller than the max y coordinate. This is intended to
        # avoid picking a face along the z-aligned edge.
        x_mean = 0.5 * (box["xmax"] + box["xmin"])
        # Compute a cell size h to place the point slightly inside the domain along the
        # y direction instead of at the very corner.
        h = np.mean(np.sqrt(sd.face_areas[domain_sides.west]))
        y_high = box["ymax"] - 0.5 * h
        z_max = box["zmax"]
        point_1 = np.array([box["xmin"], y_high, z_max])
        pts = sd.face_centers[:, domain_sides.west]
        ind_1 = domain_sides.west.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_1, pts))
        ]
        # Point 2 is on the center top of the east boundary.
        point_2 = np.array([box["xmax"], y_high, z_max])
        pts = sd.face_centers[:, domain_sides.east]
        ind_2 = domain_sides.east.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_2, pts))
        ]
        # Point 3 is on the center top of the south boundary, having min y coordinate
        # and mean x coordinate.
        point_3 = np.array([x_mean, box["ymin"], z_max])
        pts = sd.face_centers[:, domain_sides.south]
        ind_3 = domain_sides.south.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_3, pts))
        ]
        return [ind_1, ind_2, ind_3]


class GravityMagnitude:
    """Mixin class for gravity magnitude computation for different materials.

    See also :class:`~porepy.models.constitutive_laws.GravityForce`, which returns the
    gravity force as an AD operator.

    """

    fluid: pp.Fluid
    """Fluid model associated with the flow problem."""
    solid: pp.SolidConstants
    """Solid model associated with the mechanics problem."""
    units: pp.Units
    """Model units."""

    def gravity_force_magnitude(
        self, material: Literal["fluid", "solid", "bulk"]
    ) -> float:
        """Compute gravity force magnitude for a given region.

        Parameters:
            material: Material for which to compute gravity force. Can be "bulk",
                "fluid" or "solid".

        Returns:
            Gravity force magnitude for the specified material.
        """
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")
        if material == "bulk":
            phi = self.solid.porosity
            gravity = g * (
                self.fluid.reference_component.density * phi
                + (1 - phi) * self.solid.density
            )
        elif material == "fluid":
            gravity = g * self.fluid.reference_component.density
        elif material == "solid":
            gravity = g * self.solid.density
        else:
            raise ValueError(f"Unknown region '{material}' for gravity force.")
        return gravity


class LithostaticBoundaryStressValues(GravityMagnitude):
    """Boundary conditions for the mechanics with lithostatic stress on all boundaries.

    Neumann boundary conditions are defined on all boundaries. Zero stress is assumed at
    time zero. This corresponds to an initial stress-free state, and presumably to zero
    initial displacement as well. For positive times, lithostatic stress is applied
    according to the depth of the boundary faces. The principal stresses are assumed to
    align with the coordinate axes, and the relative magnitudes can be adjusted through
    the parameter "lithostatic_stress_multipliers", which should be an array of three
    values. The default is an array of ones, corresponding to equal stresses in all
    directions.

    """

    params: dict
    """Model parameters."""
    depth: Callable[[np.ndarray], np.ndarray]
    """Function to compute depth of points."""
    equation_system: pp.EquationSystem
    """Equation system associated with the model."""
    gravity_force: Callable[[pp.GridLikeSequence], pp.ad.Operator]
    """Function to compute gravity force."""
    domain_boundary_sides: Callable[[pp.GridLike], pp.domain.DomainSides]
    """Function returning the domain boundary sides of a given grid."""
    time_manager: pp.TimeManager
    """Time manager associated with the model."""

    @property
    def lithostatic_stress_multipliers(self) -> np.ndarray:
        """Return multipliers for lithostatic stress.

        Returns:
            Multipliers for lithostatic stress in the three principal directions.
            Default is an array of ones.

        """
        return self.params.get("lithostatic_stress_multipliers", np.ones(3))

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # Initialize array for stress values.
        values = np.zeros((3, boundary_grid.num_cells))
        # Assume zero initial stress state.
        if self.time_manager.time < 1e-5:
            return values.ravel("F")

        gravity = self.gravity_force_magnitude("bulk")
        # Multiply with lithostatic stress multipliers, which can be used to set
        # different stresses in different directions.
        gradient = self.lithostatic_stress_multipliers * gravity
        # Get domain sides and depth at boundary cell centers.
        domain_sides = self.domain_boundary_sides(boundary_grid)
        depth = self.depth(boundary_grid.cell_centers)

        # The sign of the stress depends on the side of the domain according to the
        # direction of the outer normal vector on the boundary. Loop over directions.
        for i, sides in enumerate(
            [["west", "east"], ["south", "north"], ["bottom", "top"]]
        ):
            # Apply stress on both sides of the domain in direction i.
            for side, sign in zip(sides, [1, -1]):
                # Get indices of faces on the given side.
                ind = getattr(domain_sides, side)
                if np.any(ind):
                    # Set ith component of stress on these faces.
                    values[i, ind] = (
                        gradient[i]
                        * depth[ind]
                        * sign
                        * boundary_grid.cell_volumes[ind]
                    )

        return values.ravel("F")


class HydrostaticPressureValues(GravityMagnitude):
    """Mixin class for hydrostatic pressure values.

    Provides method to compute hydrostatic pressure at given depths.

    """

    def hydrostatic_pressure(self, depth: np.ndarray) -> np.ndarray:
        r"""Compute hydrostatic pressure at given depths.

        The hydrostatic pressure at depth z is given by
        .. math::
            p(z) = \rho g z + p_{atm}
        where :math:`\rho` is the fluid density, :math:`g` is the gravity acceleration,
        and :math:`p_{atm}` is the atmospheric pressure.

        Parameters:
            depth: Array of depths at which to compute hydrostatic pressure.

        Returns:
            Array of hydrostatic pressure values at the given depths.

        """
        gravity = self.gravity_force_magnitude("fluid")
        pressure = gravity * depth + self.units.convert_units(
            pp.ATMOSPHERIC_PRESSURE, units="Pa"
        )
        return pressure


class HydrostaticBoundaryPressureValues(HydrostaticPressureValues):
    """Boundary conditions for the flow with hydrostatic pressure on all boundaries.

    Pressure values corresponding to hydrostatic pressure are defined on all boundaries.
    These will be used in Dirichlet boundary conditions for pressure and any derived
    quantities defined on the boundaries, such as the density-mobility product. The
    boundary types must be defined elsewhere.

    """

    params: dict
    """Model parameters."""
    depth: Callable[[np.ndarray], np.ndarray]
    """Function to compute depth of points."""
    equation_system: pp.EquationSystem
    """Equation system associated with the model."""
    fluid: pp.Fluid
    """Fluid model associated with the flow problem."""
    gravity_force: Callable[[pp.GridLikeSequence, str], pp.ad.Operator]
    """Function to compute gravity force."""

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each face in the subdomain.

        """
        depth = self.depth(boundary_grid.cell_centers)
        values = self.hydrostatic_pressure(depth)
        return values


class ThermalGradientTemperatureValues:
    r"""Mixin class for temperature values corresponding to a linear thermal gradient.

    The temperature at depth z is given by
        .. math::
            T(z) = T_{surface} + G \cdot z
    where :math:`T_{surface}` is the surface temperature and :math:`G` is the thermal
    gradient.

    """

    params: dict
    """Model parameters."""
    depth: Callable[[np.ndarray], np.ndarray]
    """Function to compute depth of points."""
    equation_system: pp.EquationSystem
    """Equation system associated with the model."""
    units: pp.Units
    """Model units."""

    @property
    def thermal_gradient(self) -> float:
        """Return thermal gradient.

        Returns:
            Thermal gradient in K/m. Default is 0.03 K/m.

        """
        val = self.params.get("thermal_gradient", 0.03)  # Default 30 K/km
        return self.units.convert_units(val, "K*m^-1")

    @property
    def surface_temperature(self) -> float:
        """Return surface temperature.

        Returns:
            Surface temperature in K. Default is 283.15 K.

        """
        val = self.params.get("surface_temperature", 293.15)  # Default 20 C
        return self.units.convert_units(val, "K")

    def temperature_at_depth(self, depth: np.ndarray) -> np.ndarray:
        """Compute temperature at given depths.

        Parameters:
            depth: Array of depths at which to compute temperature.

        Returns:
            Array of temperature values at the given depths.
        """
        return self.surface_temperature + self.thermal_gradient * depth


class ThermalGradientBoundaryTemperatureValues(ThermalGradientTemperatureValues):
    """Boundary conditions for the thermal problem with linear temperature gradient
    on all boundaries.

    Temperature values corresponding to a linear thermal gradient are defined on all
    boundaries. These will be used in Dirichlet boundary conditions for temperature and
    any derived quantities defined on the boundaries. The boundary types must be defined
    elsewhere.

    """

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each face in the subdomain.

        """
        depth = self.depth(boundary_grid.cell_centers)
        return self.temperature_at_depth(depth)
