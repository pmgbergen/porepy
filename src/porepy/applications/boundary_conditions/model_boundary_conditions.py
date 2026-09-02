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
        if not self.time_manager.is_at_initial_time():
            return values.ravel("F") + super().bc_values_displacement(bg)
        else:
            return values.ravel("F")


class BoundaryConditionsMechanicsNeumann:
    """Boundary conditions for the mechanics with Neumann conditions on almost all
    boundaries.

    The only exception is that internal boundaries are converted to Dirichlet and three
    points are partly fixed to avoid rigid body motions.

    In both 2D and 3D, the anchor points are ordered west, south, east.

    In 3D, the points are selected at maximum z coordinate:
        1) min x and high y coordinate, (fixed in y and z directions)
        2) mean x and min y coordinate, (fixed in x and z directions)
        3) max x and high y coordinate. (fixed in y and z directions)
    Seen from above, this looks like:

                -------
      (no y) 1  |     | 2 (no y)
                |     |             ^ y
                |     |             |
                -------             +---> x
                   3 (no x)

    In 2D, the points are selected at mean y on west/east and mean x on south, with
    the same ordering and constraints:
        1) west boundary: fix y
        2) south boundary: fix x
        3) east boundary: fix y
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
        dir_masks = self._anchor_dirichlet_component_masks()
        if len(faces_to_fix) != len(dir_masks):
            raise ValueError("Number of anchor faces and component masks must match.")
        for face, mask in zip(faces_to_fix, dir_masks):
            if mask.shape != (self.nd,):
                raise ValueError(
                    f"Anchor mask shape {mask.shape} incompatible with nd={self.nd}."
                )
            bc.is_dir[:, np.atleast_1d(face)] = mask[:, None]
            bc.is_neu[:, np.atleast_1d(face)] = ~mask[:, None]  # Negate for Neumann
        return bc

    def _anchor_dirichlet_component_masks(self) -> list[np.ndarray]:
        """Return Dirichlet component masks for anchor faces.

        Returns:
            A list of boolean arrays, one for each anchor face. `True` indicates the
            component is fixed with Dirichlet data.

        """
        if self.nd == 2:
            # Fix y on face 1 (west), x on face 2 (south), y on face 3 (east).
            return [
                np.array([False, True]),
                np.array([True, False]),
                np.array([False, True]),
            ]
        if self.nd == 3:
            # In 3D, use the same west-south-east ordering as in 2D, with
            # additional z-fixing to remove out-of-plane rigid-body motion.
            return [
                np.array([False, True, True]),
                np.array([True, False, True]),
                np.array([False, True, True]),
            ]
        raise NotImplementedError(
            "BoundaryConditionsMechanicsNeumann is only implemented for nd=2 or nd=3, "
            + f" got nd={self.nd}."
        )

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

        x_mean = 0.5 * (box["xmax"] + box["xmin"])

        def face_index(side, point):
            """For a given side of a domain and a target point, find the index of the
            face closest to the point."""
            side_mask = domain_sides.__getattribute__(side)
            points_on_side = sd.face_centers[: self.nd, side_mask]
            return side_mask.nonzero()[0][
                np.argmin(pp.distances.point_pointset(point, points_on_side))
            ]

        if self.nd == 2:
            y_mean = 0.5 * (box["ymax"] + box["ymin"])
            # Point 1: center on west boundary.
            ind_1 = face_index("west", np.array([box["xmin"], y_mean]))

            # Point 2: center of south boundary.
            ind_2 = face_index("south", np.array([x_mean, box["ymin"]]))

            # Point 3: center on east boundary.
            ind_3 = face_index("east", np.array([box["xmax"], y_mean]))
            return [ind_1, ind_2, ind_3]

        if self.nd == 3:
            # Point 1 is on the center top of the west boundary, having min x
            # coordinate and y coordinate slightly smaller than the max y coordinate.
            # This avoids picking a face along the z-aligned edge.
            h = np.mean(np.sqrt(sd.face_areas[domain_sides.west]))
            y_high = box["ymax"] - 0.5 * h
            z_max = box["zmax"]
            ind_1 = face_index("west", np.array([box["xmin"], y_high, z_max]))

            # Point 2 is on the center top of the south boundary.
            ind_2 = face_index("south", np.array([x_mean, box["ymin"], z_max]))

            # Point 3 is on the center top of the east boundary.
            ind_3 = face_index("east", np.array([box["xmax"], y_high, z_max]))
            return [ind_1, ind_2, ind_3]

        raise NotImplementedError(
            "BoundaryConditionsMechanicsNeumann is only implemented for nd=2 or nd=3, "
            + f" got nd={self.nd}."
        )


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

    nd: int
    """Number of spatial dimensions."""
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
    def lithostatic_stress_offset(self) -> np.ndarray:
        """Return datum lithostatic stress.

        Defaults to 0. For geometries below the datum, typical values
        are lithostatic stress multiplier times effective overburden stress
        at datum of the domain due to gravity.

        Returns:
            Datum lithostatic stress in Pa. Default is 0 Pa.

        """
        datum_lithostatic_stress = self.params.get(
            "datum_lithostatic_stress", np.zeros(self.nd)
        )
        return self.units.convert_units(datum_lithostatic_stress, units="Pa")

    @property
    def lithostatic_stress_multipliers(self) -> np.ndarray:
        """Return multipliers for lithostatic stress.

        Returns:
            Multipliers for lithostatic stress in the three principal directions.
            Default is an array of ones.

        """
        return self.params.get("lithostatic_stress_multipliers", np.ones(self.nd))

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # Initialize array for stress values.
        values = np.zeros((self.nd, boundary_grid.num_cells))
        # Assume zero initial stress state.
        if self.time_manager.is_at_initial_time():
            return values.ravel("F")

        gravity = self.gravity_force_magnitude("bulk")
        # Multiply with lithostatic stress multipliers, which can be used to set
        # different stresses in different directions.
        gradient = self.lithostatic_stress_multipliers * gravity
        # Allow for explicit offset.
        offset = self.lithostatic_stress_offset
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
                        (offset[i] + gradient[i] * depth[ind])
                        * sign
                        * boundary_grid.cell_volumes[ind]
                    )

        return values.ravel("F")


class HydrostaticPressureValues(GravityMagnitude):
    """Mixin class for hydrostatic pressure values.

    Provides method to compute hydrostatic pressure at given depths.

    """

    params: dict
    """Model parameters."""

    @property
    def hydrostatic_pressure_offset(self) -> float:
        """Return hydrostatic pressure at datum.

        Default is atmospheric pressure, assuming the top of the domain
        corresponds to the surface of the Earth. To control the effective
        hydrostatic pressure at the datum, set the parameter "datum_pressure"
        in the model parameters.

        Returns:
            Datum pressure in Pa.

        """
        datum_pressure = self.params.get("datum_pressure", pp.ATMOSPHERIC_PRESSURE)
        return self.units.convert_units(datum_pressure, units="Pa")

    def hydrostatic_pressure(self, depth: np.ndarray) -> np.ndarray:
        r"""Compute hydrostatic pressure at given depths.

        The hydrostatic pressure at depth z is given by
        .. math::
            p(z) = \rho g z + p_{offset}
        where :math:`\rho` is the fluid density, :math:`g` is the gravity acceleration,
        and :math:`p_{offset}` is the offset pressure.

        Parameters:
            depth: Array of depths at which to compute hydrostatic pressure.

        Returns:
            Array of hydrostatic pressure values at the given depths.

        """
        gravity = self.gravity_force_magnitude("fluid")
        pressure = gravity * depth + self.hydrostatic_pressure_offset
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
