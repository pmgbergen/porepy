"""Library of constitutive equations."""
from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np

import porepy as pp

import pdb
import scipy as sp

number = pp.number
Scalar = pp.ad.Scalar


class DimensionReduction:
    """Apertures and specific volumes."""

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    is_well: Callable[[pp.Grid | pp.MortarGrid], bool]
    """Check if a grid is a well. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        """Get the aperture of a single grid.

        Parameters:
            grid: Grid for which to compute the aperture.

        Returns:
            Aperture for each cell in the grid.

        """
        # NOTE: The aperture concept is not well defined for nd. However, we include it
        # for simplified implementation of specific volumes, which are defined as
        # aperture^nd-dim and should be 1 for dim=nd.
        aperture = np.ones(grid.num_cells)
        if grid.dim < self.nd:
            if self.is_well(grid):
                # This is a well. The aperture is the well radius.
                aperture *= self.solid.well_radius()
            else:
                aperture = self.solid.residual_aperture() * aperture
        return aperture

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

        Aperture is a characteristic thickness of a cell, with units [m]. It's value is
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.

        See also:
            :meth:specific_volume.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Ad operator representing the aperture for each cell in each subdomain.

        """
        if len(subdomains) == 0:
            return pp.wrap_as_ad_array(0, size=0)
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # The implementation here is not perfect, but it seems to be what is needed
        # to make the Ad framework happy: Build the global array by looping over
        # subdomains and add the local contributions.
        # Note that the aperture is an array (in the Ad sense) not a matrix, thus there
        # is no risk of the number of columns being wrong (as there would be if we
        # were to wrap the aperture as an Ad matrix).

        for i, sd in enumerate(subdomains):
            # First make the local aperture array.
            a_loc = pp.wrap_as_ad_array(self.grid_aperture(sd))
            # Expand to a global array.
            a_glob = projection.cell_prolongation([sd]) @ a_loc
            if i == 0:
                apertures = a_glob
            else:
                apertures += a_glob
        apertures.set_name("aperture")

        return apertures

    def specific_volume(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]]
    ) -> pp.ad.Operator:
        """Specific volume [m^(nd-d)].

        For subdomains, the specific volume is the cross-sectional area/volume of the
        cell, i.e. aperture to the power :math`nd-dim`. For interfaces, the specific
        volume is inherited from the higher-dimensional subdomain neighbor.

        See also:
            :meth:aperture.

        Parameters:
            subdomains: List of subdomain or interface grids.

        Returns:
            Specific volume for each cell.

        """
        if len(grids) == 0:
            return pp.wrap_as_ad_array(0, size=0)

        if isinstance(grids[0], pp.MortarGrid):
            # For interfaces, the specific volume is inherited from the
            # higher-dimensional subdomain neighbor.
            assert all(isinstance(g, pp.MortarGrid) for g in grids), "Mixed grids"

            interfaces: list[pp.MortarGrid] = [
                g for g in grids if isinstance(g, pp.MortarGrid)
            ]  # appease mypy.
            neighbor_sds = self.interfaces_to_subdomains(interfaces)
            projection = pp.ad.MortarProjections(self.mdg, neighbor_sds, interfaces)
            # Check that all interfaces are of the same co-dimension
            codim = interfaces[0].codim
            assert all(intf.codim == codim for intf in interfaces)
            if codim == 1:
                trace = pp.ad.Trace(neighbor_sds)
                v_h = trace.trace @ self.specific_volume(neighbor_sds)
            else:
                v_h = self.specific_volume(neighbor_sds)
            v = projection.primary_to_mortar_avg @ v_h
            v.set_name("specific_volume")
            return v

        assert all(isinstance(g, pp.Grid) for g in grids), "Mixed grids"
        subdomains: list[pp.Grid] = [g for g in grids if isinstance(g, pp.Grid)]
        # Compute specific volume as the cross-sectional area/volume
        # of the cell, i.e. raise to the power nd-dim
        subdomain_projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        v: pp.ad.Operator = None  # type: ignore

        # Loop over dimensions, and add the contribution from each subdomain within
        # that dimension.
        # NOTE: The loop is reversed, to ensure that the subdomains are processed in the
        # same order as will be returned by an iteration over the subdomains of the
        # mixed-dimensional grid. If the order in input argument subdomains is
        # different, the result will likely be wrong.
        for dim in range(self.nd, -1, -1):
            sd_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(sd_dim) == 0:
                continue
            a_loc = self.aperture(sd_dim)
            v_loc = a_loc ** Scalar(self.nd - dim)
            v_glob = subdomain_projection.cell_prolongation(sd_dim) @ v_loc
            if v is None:
                v = v_glob
            else:
                v = v + v_glob

        v.set_name("specific_volume")

        return v


class DisplacementJumpAperture(DimensionReduction):
    """Fracture aperture from displacement jump."""

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    normal_component: Callable[[list[pp.Grid]], pp.ad.SparseArray]
    """Operator giving the normal component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the displacement jump on fracture grids. Normally defined in a
    mixin instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    is_well: Callable[[pp.Grid | pp.MortarGrid], bool]
    """Check if a grid is a well. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def residual_aperture(self, subdomains: list[pp.Grid]) -> Scalar:
        """Residual aperture [m].

        Parameters:
            subdomains: List of subdomain grids. Not used, but included for consistency
                with other methods.

        Returns:
            Ad operator representing the resdiual aperture for the grids. The value is
            constant for each grid, and is the same for all cells in the grid.

        """
        return Scalar(self.solid.residual_aperture(), name="residual_aperture")

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

        The aperture computation depends on the dimension of the subdomain. For the
        matrix, the aperture is one. For intersections, the aperture is given by the
        average of the apertures of the adjacent fractures. For fractures, the aperture
        equals displacement jump plus residual aperture.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Operator representing apertures.

        """
        # For now, assume no intersections
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        # Subdomains of the top dimension
        nd_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        num_cells_nd_subdomains = sum([sd.num_cells for sd in nd_subdomains])
        one = pp.wrap_as_ad_array(1, size=num_cells_nd_subdomains, name="one")
        # Start with nd, where aperture is one.
        apertures = projection.cell_prolongation(nd_subdomains) @ one

        # NOTE: The loop is reversed, to ensure that the subdomains are processed in the
        # same order as will be returned by an iteration over the subdomains of the
        # mixed-dimensional grid. If the order in input argument subdomains is
        # different, the result will likely be wrong.
        # Only consider subdomains of lower dimension, there is no aperture for the top
        # dimension.
        for dim in range(self.nd - 1, -1, -1):
            subdomains_of_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(subdomains_of_dim) == 0:
                continue
            if dim == self.nd - 1:
                # Fractures. Get displacement jump
                normal_jump = self.normal_component(
                    subdomains_of_dim
                ) @ self.displacement_jump(subdomains_of_dim)
                # The jump should be bounded below by gap function. This is not
                # guaranteed in the non-converged state. As this (especially
                # non-positive values) may give significant trouble in the aperture.
                # Insert safeguard by taking maximum of the jump and a residual
                # aperture.
                f_max = pp.ad.Function(pp.ad.maximum, "maximum_function")

                a_ref = self.residual_aperture(subdomains_of_dim)
                
                apertures_of_dim = f_max(normal_jump + a_ref, a_ref)
                apertures_of_dim.set_name("aperture_maximum_function")
                apertures = (
                    apertures
                    + projection.cell_prolongation(subdomains_of_dim) @ apertures_of_dim
                )
            else:
                if dim == self.nd - 2:
                    well_subdomains = [
                        sd for sd in subdomains_of_dim if self.is_well(sd)
                    ]
                    if len(well_subdomains) > 0:
                        # Wells. Aperture is given by well radius.
                        radii = [self.grid_aperture(sd) for sd in well_subdomains]
                        well_apertures = pp.wrap_as_ad_array(
                            np.hstack(radii), name="well apertures"
                        )
                        apertures = (
                            apertures
                            + projection.cell_prolongation(well_subdomains)
                            @ well_apertures
                        )
                        # Well subdomains need not be considered further.
                        subdomains_of_dim = [
                            sd for sd in subdomains_of_dim if sd not in well_subdomains
                        ]

                # Intersection aperture is average of apertures of intersecting
                # fractures.
                interfaces_dim = self.subdomains_to_interfaces(subdomains_of_dim, [1])
                # Only consider interfaces of the current dimension, i.e. those related
                # to higher-dimensional neighbors.
                interfaces_dim = [intf for intf in interfaces_dim if intf.dim == dim]
                # Get the higher-dimensional neighbors.
                parent_subdomains = self.interfaces_to_subdomains(interfaces_dim)
                # Only consider the higher-dimensional neighbors, i.e. disregard the
                # intersections with the current dimension.
                parent_subdomains = [
                    sd for sd in parent_subdomains if sd.dim == dim + 1
                ]

                # Define the combined set of subdomains of this dimension and the
                # parents. Sort this according to the MixedDimensionalGrid's order of
                # the subdomains.
                parent_and_this_dim_subdomains = self.mdg.sort_subdomains(
                    subdomains_of_dim + parent_subdomains
                )

                # Create projection operator between the subdomains involved in the
                # computation, i.e. the current dimension and the parents.
                mortar_projection = pp.ad.MortarProjections(
                    self.mdg, parent_and_this_dim_subdomains, interfaces_dim
                )
                # Also create projections between the subdomains we act on.
                parent_and_subdomain_projection = pp.ad.SubdomainProjections(
                    parent_and_this_dim_subdomains
                )

                # Get the apertures of the higher-dimensional neighbors by calling this
                # method on the parents. TODO: It should be possible to store the values
                # from the aperture calculation on the previous dimension.
                parent_apertures = self.aperture(parent_subdomains)

                # The apertures on the lower-dimensional subdomains are the mean
                # apertures from the higher-dimensional neighbors. This requires both a
                # projection of the actual apertures and counting the number of
                # higher-dimensional neighbors.

                # Define a trace operator. This is needed to go from the cell-based
                # apertures among the parents to the faces (which are accessible to the
                # mortar projections).
                trace = pp.ad.Trace(parent_subdomains)

                # Projection from parents to intersections via the mortar grid. This is
                # a convoluted operation: Map from the trace (only defined on the
                # parents) to the full set of subdomains. Project first to the mortars
                # and then to the lower-dimensional subdomains. The resulting compound
                # projection is used  to map apertures and to count the number of neighbors.
                parent_cells_to_intersection_cells = (
                    mortar_projection.mortar_to_secondary_avg
                    @ mortar_projection.primary_to_mortar_avg
                    @ parent_and_subdomain_projection.face_prolongation(
                        parent_subdomains
                    )
                    @ trace.trace
                )

                # Average weights are the number of cells in the parent subdomains
                # contributing to each intersection cells.
                average_weights = np.ravel(
                    parent_cells_to_intersection_cells.evaluate(
                        self.equation_system
                    ).sum(axis=1)
                )
                nonzero = average_weights > 0
                average_weights[nonzero] = 1 / average_weights[nonzero]
                # Wrap as a DenseArray
                divide_by_num_neighbors = pp.wrap_as_ad_array(
                    average_weights, name="average_weights"
                )

                # Project apertures from the parents and divide by the number of
                # higher-dimensional neighbors.
                apertures_of_dim = divide_by_num_neighbors * (
                    parent_cells_to_intersection_cells @ parent_apertures
                )
                # Above matrix is defined on intersections and parents. Restrict to
                # intersections.
                intersection_subdomain_projection = pp.ad.SubdomainProjections(
                    parent_and_this_dim_subdomains
                )
                apertures_of_dim = (
                    intersection_subdomain_projection.cell_restriction(
                        subdomains_of_dim
                    )
                    @ apertures_of_dim
                )
                # Set a name for the apertures of this dimension
                apertures_of_dim.set_name(f"Displacement_jump_aperture_dim_{dim}")

                # Add to total aperture.
                apertures += (
                    projection.cell_prolongation(subdomains_of_dim) @ apertures_of_dim
                )

        # Give the operator a name
        apertures.set_name("Displacement_jump_apertures")

        return apertures


class FluidDensityFromPressure:
    """Fluid density as a function of pressure."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def fluid_compressibility(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Fluid compressibility [1/Pa].

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The constant compressibility of the fluid, represented as an Ad operator.
            The value is taken from the fluid constants.

        """
        return Scalar(self.fluid.compressibility(), "fluid_compressibility")

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of pressure.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        with :math:`\\rho_0` the reference density, :math:`p_0` the reference pressure,
        :math:`c_p` the compressibility and :math:`p` the pressure.

        The reference density and the compressibility are taken from the fluid
        constants, while the reference pressure is accessible by mixin; a typical
        implementation will provide this in a variable class.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of pressure.

        """
        # The reference density is taken from the fluid constants..
        rho_ref = Scalar(self.fluid.density(), "reference_fluid_density")
        rho = rho_ref * self.pressure_exponential(subdomains)
        rho.set_name("fluid_density")
        return rho

    def pressure_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of pressure.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """

        # print("\n\n print inside pressure_sxpoential")
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # pdb.set_trace()
        # self.pressure(subdomains)
        # pdb.set_trace()

        # Wrap compressibility from fluid class as matrix (left multiplication with dp)
        c = self.fluid_compressibility(subdomains)

        return exp(c * dp)


class FluidDensityFromTemperature:
    """Fluid density as a function of temperature."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of temperature.

        .. math::
            \\rho = \\rho_0 \\exp \\left[-c_T\\left(T - T_0\\right) \\right]

        with :math:`\\rho_0` the reference density, :math:`T_0` the reference
        temperature, :math:`c_T` the thermal expansion and :math:`T` the pressure.

        The reference density and the thermal expansion are taken from the fluid
        constants, while the reference temperature is accessible by mixin; a typical
        implementation will provide this in a variable class.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of temperature.

        """
        # The reference density is taken from the fluid constants..
        rho_ref = Scalar(self.fluid.density(), "reference_fluid_density")
        rho = rho_ref * self.temperature_exponential(subdomains)
        rho.set_name("fluid_density")
        return rho

    def temperature_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of pressure.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dtemp = self.perturbation_from_reference("temperature", subdomains)

        return exp(Scalar(-1) * Scalar(self.fluid.thermal_expansion()) * dtemp)


class FluidDensityFromPressureAndTemperature(
    FluidDensityFromPressure, FluidDensityFromTemperature
):
    """Fluid density which is a function of pressure and temperature."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of pressure and temperature.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right)
            - c_T\\left(T - T_0\\right) \\right]

        with :math:`\\rho_0` the reference density, :math:`p_0` the reference pressure,
        :math:`c_p` the compressibility, :math:`p` the pressure, :math:`T` the
        temperature, :math:`T_0` the reference temperature, and :math:`c_T` the thermal
        expansion coefficient.

        The reference density, the compressibility and the thermal expansion coefficient
        are all taken from the fluid constants, while the reference pressure and
        temperature are accessible by mixin; a typical implementation will provide this
        in a variable class.

          Parameters:
              subdomains: List of subdomain grids.

          Returns:
              Fluid density as a function of pressure and temperature.

        """
        rho_ref = Scalar(self.fluid.density(), "reference_fluid_density")

        rho = (
            rho_ref
            * self.pressure_exponential(subdomains)
            * self.temperature_exponential(subdomains)
        )
        rho.set_name("fluid_density_from_pressure_and_temperature")
        return rho


class FluidMobility:
    """Class for fluid mobility and its discretization in flow problems."""

    fluid_viscosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the fluid viscosity. Normally provided by a mixin of
    instance :class:`~porepy.models.VariableMixin`.

    """
    mobility_keyword: str
    """Keyword for the discretization of the mobility. Normally provided by a mixin of
    instance :class:`~porepy.models.SolutionStrategy`.

    """

    def mobility(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mobility of the fluid flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mobility.

        """
        return pp.ad.Scalar(1) / self.fluid_viscosity(subdomains)

    def mobility_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.UpwindAd:
        """Discretization of the fluid mobility factor.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_mobility_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the interface mobility.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)


class ConstantViscosity:
    """Constant viscosity."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def fluid_viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid viscosity [Pa s].

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            Operator for fluid viscosity, represented as an Ad operator. The value is
            picked from the fluid constants.

        """
        return Scalar(self.fluid.viscosity(), "viscosity")


class ConstantPermeability:
    """A spatially homogeneous permeability field."""

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.

    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        The permeability is quantity which enters the discretized equations in a form
        that cannot be differentiated by Ad (this is at least true for a subset of the
        relevant discretizations). For this reason, the permeability is not returned as
        an Ad operator, but as a numpy array, to be wrapped as a SecondOrderTensor and
        passed as a discretization parameter.

        Parameters:
            subdomains: Subdomains where the permeability is defined.

        Raises:
            ValueError: If more than one subdomain is provided.

        Returns:
            Cell-wise permeability tensor. The value is picked from the solid constants.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_ad_array(
            self.solid.permeability(), size, name="permeability"
        )
        return permeability

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Normal permeability [m^2].

        Contrary to the permeability, the normal permeability typically enters into the
        discrete equations in a form that can be differentiated by Ad. For this reason,
        the normal permeability is returned as an Ad operator.

        Parameters:
            interfaces: List of interface grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            Scalar normal permeability on the interfaces between subdomains, represented
            as an Ad operator. The value is picked from the solid constants.

        """
        return Scalar(self.solid.normal_permeability())


class CubicLawPermeability(ConstantPermeability):
    """Cubic law permeability for fractures and intersections."""

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Function that returns the specific volume of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    aperture: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the aperture of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        This function combines a matrix permeability with a cubic law permeability for
        fractures and intersections. The combination entails projection between the two
        subdomain subsets and all subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures_and_intersections: list[pp.Grid] = [
            sd for sd in subdomains if sd.dim < self.nd
        ]

        permeability = projection.cell_prolongation(matrix) @ self.matrix_permeability(
            matrix
        ) + projection.cell_prolongation(
            fractures_and_intersections
        ) @ self.cubic_law_permeability(
            fractures_and_intersections
        )
        return permeability

    def cubic_law_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic law permeability for fractures (or intersections).

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        aperture = self.aperture(subdomains)
        perm = (aperture ** Scalar(2)) / Scalar(12)

        return perm

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the matrix.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return super().permeability(subdomains)


class DarcysLaw:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around.
    """

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    bc_values_darcy: Callable[[list[pp.Grid]], pp.ad.DenseArray]
    """Darcy flux boundary conditions. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.

    """
    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Normal permeability. Normally defined in a mixin instance of :
    class:`~porepy.models.constitutive_laws.ConstantPermeability`.

    """
    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    darcy_keyword: str
    """Keyword used to identify the Darcy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.SparseArray]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    internal_boundary_normal_to_outwards: Callable[
        [list[pp.Grid], int], pp.ad.SparseArray
    ]
    """Switch interface normal vectors to point outwards from the subdomain. Normally
    set by a mixin instance of :class:`porepy.models.geometry.ModelGeometry`.
    """
    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str, int], pp.ad.DenseArray]
    """Wrap a grid attribute as a DenseArray. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    outwards_internal_boundary_normals: Callable[
        [list[pp.MortarGrid], bool], pp.ad.Operator
    ]
    """Outwards normal vectors on internal boundaries. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Specific volume. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DimensionReduction` or a subclass thereof.

    """
    aperture: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Aperture. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DimensionReduction` or a subclass thereof.

    """

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )
        p: pp.ad.MixedDimensionalVariable = self.pressure(subdomains)
        pressure_trace = (
            discr.bound_pressure_cell @ p
            + discr.bound_pressure_face
            @ (projection.mortar_to_primary_int @ self.interface_darcy_flux(interfaces))
            + discr.bound_pressure_face @ self.bc_values_darcy(subdomains)
            + discr.vector_source @ self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace

    def darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of Darcy's law.

        Note:
            The fluid mobility is not included in the Darcy flux. This is because we
            discretize it with an upstream scheme. This means that the fluid mobility
            may have to be included when using the flux in a transport equation.
            The units of the Darcy flux are [m^2 Pa / s].

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(subdomains)
            + discr.bound_flux
            @ (
                self.bc_values_darcy(subdomains)
                + projection.mortar_to_primary_int
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source @ self.vector_source(subdomains, material="fluid")
        )
        flux.set_name("Darcy_flux")
        return flux

    def interface_darcy_flux_equation(self, interfaces: list[pp.MortarGrid]):
        """Darcy flux on interfaces.

        The units of the Darcy flux are [m^2 Pa / s], see note in :meth:`darcy_flux`.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # Ignore mypy complaint about unexpected keyword arguments.
        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        trace = pp.ad.Trace(subdomains, dim=1)

        # Gradient operator in the normal direction. The collapsed distance is
        # :math:`\frac{a}{2}` on either side of the fracture.
        # We assume here that :meth:`apeture` is implemented to give a meaningful value
        # also for subdomains of co-dimension > 1.
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg @ self.aperture(subdomains) ** Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        # Project the two pressures to the interface and multiply with the normal
        # diffusivity.
        # The cell volumes are scaled in two stages:
        # The term cell_volumes carries the volume of the cells in the mortar grids,
        # while the volume scaling from reduced dimensions is picked from the
        # specific volumes of the higher dimension (variable `specific_volume`)
        # and projected to the interface via a trace operator.
        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(subdomains)
        specific_volume_intf = (
            projection.primary_to_mortar_avg
            @ trace.trace
            @ self.specific_volume(subdomains)
        )
        specific_volume_intf.set_name("specific_volume_at_interfaces")
        eq = self.interface_darcy_flux(interfaces) - (
            cell_volumes
            * (
                self.normal_permeability(interfaces)
                * normal_gradient
                * specific_volume_intf
                * (
                    pressure_h
                    - pressure_l
                    + self.interface_vector_source(interfaces, material="fluid")
                )
            )
        )

        eq.set_name("interface_darcy_flux_equation")
        return eq

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        # TODO: The ad.Discretizations may be purged altogether. Their current function
        # is very similar to the ad.Geometry in that both basically wrap numpy/scipy
        # arrays in ad arrays and collect them in a block matrix. This similarity could
        # possibly be exploited. Revisit at some point.
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]], material: str
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(0, "m*s^-2")
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        source = pp.wrap_as_ad_array(val, size=size, name="zero_vector_source")
        return source

    def interface_vector_source(
        self, interfaces: list[pp.MortarGrid], material: str
    ) -> pp.ad.Operator:
        """Interface vector source term.

        The term is the dot product of unit normals and vector source values.
        Normalization is needed to balance the integration done in the interface flux
        law.

        Parameters:
            interfaces: List of interfaces where the vector source is defined.

        Returns:
            Cell-wise vector source term.

        """
        # Account for sign of boundary face normals.
        # No scaling with interface cell volumes.
        normals = self.outwards_internal_boundary_normals(
            interfaces, unitary=True  # type: ignore[call-arg]
        )
        # Project vector source from lower-dimensional neighbors to the interfaces.
        # This allows including pressure and temperature dependent density, which would
        # not be defined on the interface.
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )
        vector_source = projection.secondary_to_mortar_avg @ self.vector_source(
            subdomain_neighbors, material=material
        )
        # Make dot product with vector source in two steps. First multiply the vector
        # source with a matrix (though the formal mypy type is Operator, the matrix is
        # composed by summation).
        normals_times_source = normals * vector_source
        # Then sum over the nd dimensions. We need to surpress mypy complaints on  basis
        # having keyword-only arguments. The result will in effect be a matrix.
        nd_to_scalar_sum = pp.ad.sum_operator_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]  # type: ignore[call-arg]
        )
        # Finally, the dot product between normal vectors and the vector source. This
        # must be implemented as a matrix-vector product (yes, this is confusing).
        dot_product = nd_to_scalar_sum @ normals_times_source
        dot_product.set_name("Interface vector source term")
        return dot_product


class PeacemanWellFlux:
    """Well fluxes.

    Relations between well fluxes and pressures are implemented in this class.
    Peaceman 1977 https://doi.org/10.2118/6893-PA

    Assumes permeability is cell-wise scalar.

    """

    volume_integral: Callable[
        [pp.ad.Operator, Union[list[pp.Grid], list[pp.MortarGrid]], int], pp.ad.Operator
    ]
    """Integration over cell volumes, implemented in
    :class:`pp.models.abstract_equations.BalanceEquation`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""
    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure variable."""
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Well flux variable."""
    solid: pp.SolidConstants
    """Solid constants object."""
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the permeability of a subdomain. Normally provided by a
    mixin class with a suitable permeability definition.

    """

    def well_flux_equation(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Equation for well fluxes.

        Parameters:
            interfaces: List of interfaces where the well fluxes are defined.

        Returns:
            Cell-wise well flux operator.

        """

        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        r_w = self.well_radius(subdomains)
        skin_factor = self.skin_factor(interfaces)
        r_e = self.equivalent_well_radius(subdomains)

        f_log = pp.ad.Function(pp.ad.functions.log, "log_function_Piecmann")
        well_index = (
            pp.ad.Scalar(2 * np.pi)
            * projection.primary_to_mortar_avg
            @ (self.permeability(subdomains) / (f_log(r_e / r_w) + skin_factor))
        )
        eq: pp.ad.Operator = self.well_flux(interfaces) - self.volume_integral(
            well_index, interfaces, 1
        ) * (
            projection.primary_to_mortar_avg @ self.pressure(subdomains)
            - projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        )
        eq.set_name("well_flux_equation")
        return eq

    def equivalent_well_radius(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute equivalent radius for Peaceman well model.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise equivalent radius operator.

        """
        # Implementational note: The computation of equivalent radius is highly
        # simplified and ignores discretization and anisotropy effects. For more
        # advanced alternatives, see the MRST book,
        # https://www.cambridge.org/core/books/an-introduction-to-
        # reservoir-simulation-using-matlabgnu-octave/F48C3D8C88A3F67E4D97D4E16970F894
        if len(subdomains) == 0:
            # Set 0.2 as the unused value for equivalent radius. This is a bit arbitrary,
            # but 0 is a bad choice, as it will lead to division by zero.
            return Scalar(0.2, name="equivalent_well_radius")

        h_list = []
        for sd in subdomains:
            if sd.dim == 0:
                # Avoid division by zero for points. The value is not used in calling
                # method well_flux_equation, as all wells are 1d.
                h_list.append(np.array([1]))
            else:
                h_list.append(np.power(sd.cell_volumes, 1 / sd.dim))
        r_e = Scalar(0.2) * pp.wrap_as_ad_array(np.concatenate(h_list))
        r_e.set_name("equivalent_well_radius")
        return r_e

    def skin_factor(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Compute skin factor for Peaceman well model.

        Parameters:
            interfaces: List of interfaces.

        Returns:
            Skin factor operator.

        """
        skin_factor = pp.ad.Scalar(self.solid.skin_factor())
        skin_factor.set_name("skin_factor")
        return skin_factor

    def well_radius(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute well radius for Peaceman well model.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise well radius operator.

        """
        r_w = pp.ad.Scalar(self.solid.well_radius())
        r_w.set_name("well_radius")
        return r_w


class ThermalExpansion:
    """Thermal expansion coefficients for the fluid and the solid."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def fluid_thermal_expansion(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal expansion of the fluid [1/K].

        Parameters:
            subdomains: List of subdomains where the thermal expansion is defined.

        Returns:
            Thermal expansion of the fluid, represented as an Ad operator. The value is
            constant for all subdomains.

        """
        val = self.fluid.thermal_expansion()
        return Scalar(val, "fluid_thermal_expansion")

    def solid_thermal_expansion(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal expansion of the solid [1/K].

        Parameters:
            subdomains: List of subdomains where the thermal expansion is defined.

        Returns:
            Thermal expansion of the fluid, represented as an Ad operator. The value is
            constant for all subdomains.

        """
        val = self.solid.thermal_expansion()
        return Scalar(val, "solid_thermal_expansion")


class ThermalConductivityLTE:
    """Thermal conductivity in the local thermal equilibrium approximation."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    reference_porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.PoroMechanicsPorosity`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid thermal conductivity [W / (m K)].

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar
            representing the constant thermal conductivity of the fluid.

        """
        return Scalar(self.fluid.thermal_conductivity(), "fluid_thermal_conductivity")

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid thermal conductivity [W / (m K)].

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar, wrapped as
            an Ad operator, representing the constant thermal conductivity of the fluid.

        """
        return Scalar(self.solid.thermal_conductivity(), "solid_thermal_conductivity")

    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        The thermal conductivity is computed as the porosity-weighted average of the
        fluid and solid thermal conductivities. In this implementation, both are
        considered constants, however, if the porosity changes with time, the weighting
        factor will also change.

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conducivity operator.

        """
        phi = self.porosity(subdomains)
        # Since thermal conductivity is used as a discretization parameter, it has to be
        # evaluated before the discretization matrices are computed.
        try:
            phi.evaluate(self.equation_system)
        except KeyError:
            # We assume this means that the porosity includes a discretization matrix
            # for div_u which has not yet been computed.
            phi = self.reference_porosity(subdomains)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) + (
            Scalar(1) - phi
        ) * self.solid_thermal_conductivity(subdomains)

        return conductivity

    def normal_thermal_conductivity(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Scalar:
        """Normal thermal conductivity.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        val = self.fluid.normal_thermal_conductivity()
        return Scalar(val, "normal_thermal_conductivity")


class FouriersLaw:
    """This class could be refactored to reuse for other diffusive fluxes. It's somewhat
    cumbersome, though, since potential, discretization, and boundary conditions all
    need to be passed around. Also, gravity effects are not included, as opposed to the
    Darcy flux (see that class).
    """

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Temperature variable. Normally defined in a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interface_fourier_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Fourier flux variable on interfaces. Normally defined in a mixin instance of
   :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    bc_values_fourier: Callable[[list[pp.Grid]], pp.ad.DenseArray]
    """Fourier flux boundary conditions. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """
    normal_thermal_conductivity: Callable[[list[pp.MortarGrid]], pp.ad.Scalar]
    """Conductivity on a mortar grid. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a subclass.

    """
    fourier_keyword: str
    """Keyword used to identify the Fourier flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str, int], pp.ad.DenseArray]
    """Wrap a grid attribute as a DenseArray. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Specific volume. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DimensionReduction` or a subclass thereof.

    """
    aperture: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Aperture. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DimensionReduction` or a subclass thereof.

    """

    def temperature_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Temperature on the subdomain boundaries.

        .. note::
            The below implementation assumes the heat flux is discretized with a finite
            volume method (either Tpfa or Mpfa). Other discretizations may be possible,
            but would likely require a modification of this (and several other) methods.

        Parameters:
            subdomains: List of subdomains where the temperature is defined.

        Returns:
            Temperature on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.fourier_flux_discretization(
            subdomains
        )
        t: pp.ad.MixedDimensionalVariable = self.temperature(subdomains)
        temperature_trace = (
            discr.bound_pressure_cell @ t  # "pressure" is a legacy misnomer
            + discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_int
                @ self.interface_fourier_flux(interfaces)
            )
            + discr.bound_pressure_face @ self.bc_values_fourier(subdomains)
        )
        return temperature_trace

    def fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discrete Fourier flux on subdomains.

        .. note::
            The below implementation assumes the heat flux is discretized with a finite
            volume method (either Tpfa or Mpfa). Other discretizations may be possible,
            but would likely require a modification of this (and several other) methods.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            An Ad-operator representing the Fourier flux on the subdomains.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.fourier_flux_discretization(
            subdomains
        )

        # As opposed to darcy_flux in :class:`DarcyFluxFV`, the gravity term is not
        # included here.
        flux: pp.ad.Operator = discr.flux @ self.temperature(
            subdomains
        ) + discr.bound_flux @ (
            self.bc_values_fourier(subdomains)
            + projection.mortar_to_primary_int @ self.interface_fourier_flux(interfaces)
        )
        flux.set_name("Fourier_flux")
        return flux

    def interface_fourier_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Discrete Fourier flux on interfaces.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Fourier flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # Ignore mypy complaint about unexpected keyword arguments.
        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        specific_volume = self.specific_volume(subdomains)
        trace = pp.ad.Trace(subdomains, dim=1)

        # Gradient operator in the normal direction. The collapsed distance is
        # :math:`\frac{a}{2}` on either side of the fracture.
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg @ self.aperture(subdomains) ** Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        # Project the two temperatures to the interface and multiply with the normal
        # conductivity.
        # See comments in :meth:`interface_darcy_flux_equation` for more information on
        # the terms in the below equation.
        eq = self.interface_fourier_flux(interfaces) - (
            cell_volumes
            * (
                self.normal_thermal_conductivity(interfaces)
                * (
                    normal_gradient
                    * (projection.primary_to_mortar_avg @ trace.trace @ specific_volume)
                    * (
                        projection.primary_to_mortar_avg
                        @ self.temperature_trace(subdomains)
                        - projection.secondary_to_mortar_avg
                        @ self.temperature(subdomains)
                    )
                )
            )
        )
        eq.set_name("interface_fourier_flux_equation")
        return eq

    def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Fourier flux discretization.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization object for the Fourier flux.

        """
        return pp.ad.MpfaAd(self.fourier_keyword, subdomains)


class AdvectiveFlux:
    """Mixin class for discretizing advective fluxes."""

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.
    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.
    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.
    """
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Darcy flux variables on subdomains. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DarcysLaw`.
    """
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Well flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """

    def advective_flux(
        self,
        subdomains: list[pp.Grid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindAd,
        bc_values: pp.ad.Operator,
        interface_flux: Optional[
            Callable[[list[pp.MortarGrid]], pp.ad.Operator]
        ] = None,
    ) -> pp.ad.Operator:
        """An operator represetning the advective flux on subdomains.

        .. note::
            The implementation assumes that the advective flux is discretized using a
            standard upwind discretization. Other discretizations may be possible, but
            this has not been considered.

        Parameters:
            subdomains: List of subdomains.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.
            bc_values: Boundary conditions for the advective flux.
            interface_flux: Interface flux operator/variable. If subdomains have no
                neighboring interfaces, this argument can be omitted.

        Returns:
            Operator representing the advective flux.

        """

        # print("\n inside advective_flux")

        darcy_flux = self.darcy_flux(subdomains)
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux: pp.ad.Operator = (
            darcy_flux * (discr.upwind @ advected_entity)
            - discr.bound_transport_dir @ (darcy_flux * bc_values)
            # Advective flux coming from lower-dimensional subdomains ### EB: comment, you are in the wrong place...
            - discr.bound_transport_neu @ bc_values
        )

        if interface_flux is not None:
            flux -= (
                discr.bound_transport_neu
                @ mortar_projection.mortar_to_primary_int
                @ interface_flux(interfaces)
            )
        else:
            assert len(interfaces) == 0

        return flux

    def interface_advective_flux(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """An operator represetning the advective flux on interfaces.

        .. note::
            The implementation here is tailored for discretization using an upwind
            discretization for the advective interface flux. Other discretizaitons may
            be possible, but this has not been considered.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.

        """
        # If no interfaces are given, make sure to proceed with a non-empty subdomain
        # list if relevant.
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        trace = pp.ad.Trace(subdomains)
        # Project the two advected entities to the interface and multiply with upstream
        # weights and the interface Darcy flux.
        interface_flux: pp.ad.Operator = self.interface_darcy_flux(interfaces) * (
            discr.upwind_primary
            @ mortar_projection.primary_to_mortar_avg
            @ trace.trace
            @ advected_entity
            + discr.upwind_secondary
            @ mortar_projection.secondary_to_mortar_avg
            @ advected_entity
        )
        return interface_flux

    def well_advective_flux(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """An operator represetning the advective flux on interfaces.

        .. note::
            The implementation here is tailored for discretization using an upwind
            discretization for the advective interface flux. Other discretizaitons may
            be possible, but this has not been considered.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.
        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        # Project the two advected entities to the interface and multiply with upstream
        # weights and the interface Darcy flux.
        interface_flux: pp.ad.Operator = self.well_flux(interfaces) * (
            discr.upwind_primary
            @ mortar_projection.primary_to_mortar_avg
            @ advected_entity
            + discr.upwind_secondary
            @ mortar_projection.secondary_to_mortar_avg
            @ advected_entity
        )
        return interface_flux


class SpecificHeatCapacities:
    """Class for the specific heat capacities of the fluid and solid phases."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid specific heat capacity [J/kg/K].

        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity. The value is picked
            from the fluid constants.

        """
        return Scalar(
            self.fluid.specific_heat_capacity(), "fluid_specific_heat_capacity"
        )

    def solid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid specific heat capacity [J/kg/K].

        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the solid specific heat capacity. The value is picked
            from the solid constants.

        """
        return Scalar(
            self.solid.specific_heat_capacity(), "solid_specific_heat_capacity"
        )


class EnthalpyFromTemperature(SpecificHeatCapacities):
    """Class for representing the ethalpy, computed from the perturbation from a
    reference temperature.
    """

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Temperature variable. Normally defined in a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """
    enthalpy_keyword: str
    """Keyword used to identify the enthalpy flux discretization. Normally"
     set by an instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """

    def fluid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid enthalpy [J*kg^-1*m^-nd].

        The enthalpy is computed as a perturbation from a reference temperature as
        .. math::
            h = c_p (T - T_0)

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid enthalpy.

        """
        c = self.fluid_specific_heat_capacity(subdomains)
        enthalpy = c * self.perturbation_from_reference("temperature", subdomains)
        enthalpy.set_name("fluid_enthalpy")
        return enthalpy

    def enthalpy_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.UpwindAd:
        """Discretization of the fluid enthalpy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid enthalpy flux.

        """
        return pp.ad.UpwindAd(self.enthalpy_keyword, subdomains)

    def interface_enthalpy_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the interface enthalpy.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface enthalpy flux.

        """
        return pp.ad.UpwindCouplingAd(self.enthalpy_keyword, interfaces)

    def solid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid enthalpy [J*kg^-1*m^-nd].

        The enthalpy is computed as a perturbation from a reference temperature as
        .. math::
            h = c_p (T - T_0)

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid enthalpy.

        """
        c = self.solid_specific_heat_capacity(subdomains)
        enthalpy = c * self.perturbation_from_reference("temperature", subdomains)
        enthalpy.set_name("solid_enthalpy")
        return enthalpy


class GravityForce:
    """Gravity force.

    The gravity force is defined as the product of the fluid density and the gravity
    vector:

    .. math::
        g = -\\rho \\mathbf{g}= -\\rho \\begin{bmatrix} 0 \\\\ 0 \\\\ G \\end{bmatrix}

    where :math:`\\rho` is the fluid density, and :math:`G` is the magnitude of the
    gravity acceleration.

    To be used in fluid fluxes and as body force in the force/momentum balance equation.

    """

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    e_i: Callable[[Union[list[pp.Grid], list[pp.MortarGrid]], int, int], pp.ad.Operator]
    """Function that returns the unit vector in the i-th direction.

    Normally provided by a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def gravity_force(
        self,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: Literal["fluid", "solid"],
    ) -> pp.ad.Operator:
        """Vector source term on either subdomains or interfaces.

        Represents gravity effects. EK: Let's discuss how to name/think about this term.
        Note that it appears slightly differently in a flux and a force/momentum
        balance.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        size = np.sum([g.num_cells for g in grids])
        gravity = pp.wrap_as_ad_array(val, size=size, name="gravity")
        rho = getattr(self, material + "_density")(grids)
        # Gravity acts along the last coordinate direction (z in 3d, y in 2d)

        # Ignore type error, can't get mypy to understand keyword-only arguments in
        # mixin
        e_n = self.e_i(grids, i=self.nd - 1, dim=self.nd)  # type: ignore[call-arg]
        # e_n is a matrix, thus we need @ for it.
        source = Scalar(-1) * e_n @ (rho * gravity)
        source.set_name("gravity_force")
        return source


class LinearElasticMechanicalStress:
    """Linear elastic stress tensor.

    To be used in mechanical problems, e.g. force balance.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    bc_values_mechanics: Callable[[list[pp.Grid]], pp.ad.DenseArray]
    """Mechanics boundary conditions. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsMomentumBalance`.

    """
    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    interface_displacement: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Displacement variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    local_coordinates: Callable[[list[pp.Grid]], pp.ad.SparseArray]
    """Mapping to local coordinates. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def mechanical_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            subdomains: List of subdomains. Should be of co-dimension 0.

        Returns:
            Ad operator representing the mechanical stress on grid faces of the
                subdomains.

        """
        for sd in subdomains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            assert sd.dim == self.nd

        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = pp.ad.MpsaAd(self.stress_keyword, subdomains)
        # Fractures in the domain
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Boundary conditions on external boundaries
        bc = self.bc_values_mechanics(subdomains)
        proj = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=self.nd)
        # The stress in the subdomanis is the sum of the stress in the subdomain,
        # the stress on the external boundaries, and the stress on the interfaces.
        # The latter is found by projecting the displacement on the interfaces to the
        # subdomains, and let these act as Dirichlet boundary conditions on the
        # subdomains.
        stress = (
            discr.stress @ self.displacement(subdomains)
            + discr.bound_stress @ bc
            + discr.bound_stress
            @ proj.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress

    def fracture_stress(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Fracture stress on interfaces.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Fracture stress operator.

        Raises:
            ValueError: If any interface is not of co-dimension 1.

        """
        for interface in interfaces:
            if any([interface.dim != self.nd - 1]):
                raise ValueError("Interface must be of co-dimension 1.")

        # Subdomains of the interfaces
        subdomains = self.interfaces_to_subdomains(interfaces)
        # Isolate the fracture subdomains
        fracture_subdomains = [sd for sd in subdomains if sd.dim == self.nd - 1]
        # Projection between all subdomains of the interfaces
        subdomain_projection = pp.ad.SubdomainProjections(subdomains, self.nd)
        # Projection between the subdomains and the interfaces
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        # Spelled out, the stress on the interface is found by mapping the
        # contact traction (a primary variable) from local to global coordinates (note
        # the transpose), prolonging the traction from the fracture subdomains to all
        # subdomains (the domain of definition for the mortar projections), projecting
        # to the interface, and switching the sign of the traction depending on the
        # sign of the mortar sides.
        traction = (
            mortar_projection.sign_of_mortar_sides
            @ mortar_projection.secondary_to_mortar_int
            @ subdomain_projection.cell_prolongation(fracture_subdomains)
            @ self.local_coordinates(fracture_subdomains).transpose()
            @ self.contact_traction(fracture_subdomains)
        )
        traction.set_name("mechanical_fracture_stress")
        return traction


class PressureStress(LinearElasticMechanicalStress):
    """Stress tensor from pressure.

    To be used in poromechanical problems.

    .. note::
        The below discretization assumes the stress is discretized with a Mpsa finite
        volume discretization. Other discretizations may be possible, but are not
        available in PorePy at the moment, and would likely require changes to this
        method.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    reference_pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference pressure. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    outwards_internal_boundary_normals: Callable[
        [list[pp.MortarGrid], bool], pp.ad.Operator
    ]
    """Outwards normal vectors on internal boundaries. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.SparseArray]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """

    def pressure_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure contribution to stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Pressure stress operator.

        Raises:
            ValueError: If any subdomain is not of dimension `nd`.

        """
        for sd in subdomains:
            # The stress is only defined in matrix subdomains. The stress from fluid
            # pressure in fracture subdomains is handled in :meth:`fracture_stress`.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of dimension nd.")

        # No need to accommodate different discretizations for the stress tensor, as we
        # have only one.
        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)
        # The stress is simply found by the grad_p operator, multiplied with the
        # pressure perturbation.
        stress: pp.ad.Operator = (
            discr.grad_p @ self.pressure(subdomains)
            # The reference pressure is only defined on sd_primary, thus there is no
            # need for a subdomain projection.
            - discr.grad_p @ self.reference_pressure(subdomains)
        )
        stress.set_name("pressure_stress")
        return stress

    def fracture_stress(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Fracture stress on interfaces.

        The fracture stress is composed of the stress from the contact traction, and
        the stress from the fluid pressure inside the fracture.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Poromechanical stress operator on matrix-fracture interfaces.

        Raises:
            ValueError: If any interface is not of dimension `nd - 1`.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError("Interfaces must be of dimension nd - 1.")

        traction = super().fracture_stress(interfaces) + self.fracture_pressure_stress(
            interfaces
        )
        traction.set_name("poro_mechanical_fracture_stress")
        return traction

    def fracture_pressure_stress(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Pressure contribution to stress tensor on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Pressure stress operator.

        """
        # All subdomains of the interfaces
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)

        # Consistent sign of the normal vectors.
        # Note the unitary scaling here, we will scale the pressure with the area
        # of the interface (equivalently face area in the matrix subdomains) elsewhere.
        outwards_normal = self.outwards_internal_boundary_normals(
            interfaces, unitary=True  # type: ignore[call-arg]
        )

        # Expands from cell-wise scalar to vector. Equivalent to the :math:`\mathbf{I}p`
        # operation.
        # Mypy seems to believe that sum always returns a scalar. Ignore errors.
        scalar_to_nd: pp.ad.Operator = pp.ad.sum_operator_list(  # type: ignore[assignment]
            [
                e_i  # type: ignore[misc]
                for e_i in self.basis(interfaces, dim=self.nd)  # type: ignore[call-arg]
            ]
        )
        # Spelled out, from the right: Project the pressure from the fracture to the
        # mortar, expand to an nd-vector, and multiply with the outwards normal vector.
        stress = outwards_normal * (
            scalar_to_nd
            @ mortar_projection.secondary_to_mortar_avg
            @ self.pressure(subdomains)
        )
        stress.set_name("fracture_pressure_stress")
        return stress


class ThermoPressureStress(PressureStress):
    """Stress tensor from pressure and temperature.

    To be used in thermoporomechanical problems.

    .. note::
        This class assumes both pressure and temperature stresses. To avoid having
        to discretize twice, the pressure stress is discretized with a Mpsa
        discretization, while the temperature stress computed as a scaled version of
        the pressure stress. If pure thermomechanical problems are to be solved, a
        different class must be used for the temperature stress.

    """

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Temperature variable. Normally defined in a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    reference_temperature: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference temperature. Normally defined in a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Biot coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.

    """
    bulk_modulus: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Bulk modulus. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.LinearElasticSolid`.

    """
    solid_thermal_expansion: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Thermal expansion coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalExpansion`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """

    def thermal_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Temperature contribution to stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Temperature stress operator.

        Raises:
            AssertionError: If any subdomain is not of dimension `nd`.

        """
        for sd in subdomains:
            if sd.dim != self.nd:
                raise ValueError("Subdomains must be of dimension nd - 1.")

        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)
        alpha = self.biot_coefficient(subdomains)
        beta = self.solid_thermal_expansion(subdomains)
        k = self.bulk_modulus(subdomains)

        # Check that both are scalar. Else, the scaling may not be correct.
        assert isinstance(alpha, pp.ad.Scalar)
        assert isinstance(beta, pp.ad.Scalar)
        # The thermal stress should be multiplied by beta and k. Divide by alpha to
        # cancel that factor from the discretization matrix.
        stress: pp.ad.Operator = (
            beta
            * k
            / alpha
            * (
                discr.grad_p @ self.temperature(subdomains)
                - discr.grad_p @ self.reference_temperature(subdomains)
            )
        )
        stress.set_name("thermal_stress")
        return stress


class ConstantSolidDensity:
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def solid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Constant solid density.

        Parameters:
            subdomains: List of subdomains where the density is defined.

        Returns:
            Scalar operator representing the solid density. The (constant) value is
                picked from the solid constants.

        """
        return Scalar(self.solid.density(), "solid_density")


class LinearElasticSolid(LinearElasticMechanicalStress, ConstantSolidDensity):
    """Linear elastic properties of a solid.

    Includes "primary" stiffness parameters (lame_lambda, shear_modulus) and "secondary"
    parameters (bulk_modulus, lame_mu, poisson_ratio). The latter are computed from the
    former. Also provides a method for computing the stiffness matrix as a
    FourthOrderTensor.
    """

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus operator. The value is picked from the solid
            constants.

        """
        return Scalar(self.solid.shear_modulus(), "shear_modulus")

    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter operator. The value is picked from the
                solid constants.

        """
        return Scalar(self.solid.lame_lambda(), "lame_lambda")

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the Young's modulus is defined.

        Returns:
            Cell-wise Young's modulus in Pascal. The value is picked from the solid
                constants.

        """
        val = (
            self.solid.shear_modulus()
            * (3 * self.solid.lame_lambda() + 2 * self.solid.shear_modulus())
            / (self.solid.lame_lambda() + self.solid.shear_modulus())
        )
        return Scalar(val, "youngs_modulus")

    def bulk_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Bulk modulus [Pa]."""
        val = self.solid.lame_lambda() + 2 * self.solid.shear_modulus() / 3
        return Scalar(val, "bulk_modulus")

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        lmbda = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        mu = self.solid.shear_modulus() * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu, lmbda)


class FrictionBound:
    """Friction bound for fracture deformation.

    This class is intended for use with fracture deformation models.
    """

    normal_component: Callable[[list[pp.Grid]], pp.ad.SparseArray]
    """Operator giving the normal component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def friction_bound(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction bound [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [Pa].

        """
        t_n: pp.ad.Operator = self.normal_component(subdomains) @ self.contact_traction(
            subdomains
        )
        bound: pp.ad.Operator = (
            Scalar(-1.0) * self.friction_coefficient(subdomains) * t_n
        )
        bound.set_name("friction_bound")
        return bound

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient operator.

        """
        return Scalar(
            self.solid.friction_coefficient(),
            "friction_coefficient",
        )


class ShearDilation:
    """Class for calculating fracture dilation due to tangential shearing.

    The main method of the class is :meth:`shear_dilation_gap`.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    tangential_component: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the tangential component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the displacement jump on fracture grids. Normally defined in a
    mixin instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear dilation [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise shear dilation.

        """
        angle: pp.ad.Operator = self.dilation_angle(subdomains)
        f_norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        f_tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")
        shear_dilation: pp.ad.Operator = f_tan(angle) * f_norm(
            self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        )

        shear_dilation.set_name("shear_dilation")
        return shear_dilation

    def dilation_angle(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Dilation angle [rad].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise dilation angle operator [rad].

        """
        return Scalar(self.solid.dilation_angle(), "dilation_angle")


class BartonBandis:
    """Implementation of the Barton-Bandis model for elastic fracture normal
    deformation.

    The Barton-Bandis model represents a non-linear elastic deformation in the normal
    direction of a fracture. Specifically, the decrease in normal opening,
    :math:``\Delta u_n`` under a force :math:``\sigma_n`` given as

    .. math::

        \Delta u_n =  \frac{\Delta u_n^{max} \sigma_n}{\Delta u_n^{max} K_n + \sigma_n}

    where :math:``\Delta u_n^{max}`` is the maximum fracture closure and the material
    constant :math:``K_n`` is known as the fracture normal stiffness.

    The Barton-Bandis equation is defined in
    :meth:``elastic_normal_fracture_deformation`` while the two parameters
    :math:``\Delta u_n^{max}`` and :math:``K_n`` can be set by the methods
    :meth:``maximum_fracture_closure`` and :meth:``fracture_normal_stiffness``.

    """

    normal_component: Callable[[list[pp.Grid]], pp.ad.SparseArray]
    """Operator giving the normal component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def elastic_normal_fracture_deformation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Barton-Bandis model for elastic normal deformation of a fracture.

        The model computes a *decrease* in the normal opening as a function of the
        contact traction and material constants. See comments in the class documentation
        for how to include the Barton-Bandis effect in the model for fracture
        deformation.

        The returned value depends on the value of the solid constant
        maximum_fracture_closure. If its value is zero, the Barton-Bandis model is
        void, and the method returns a hard-coded pp.ad.Scalar(0) to avoid zero
        division. Otherwise, an operator which implements the Barton-Bandis model is
        returned. The special treatment ammounts to a continuous extension in the limit
        of zero maximum fracture closure.

        The implementation is based on the paper

        References:
            Fundamentals of Rock Joint Deformation, by S.C. Bandis, A.C.Lumdsen, N.R.
            Barton, International Journal of Rock Mechanics & Mining Sciences, 1983,
            Link: https://doi.org/10.1016/0148-9062(83)90595-8.

            See in particular Equations (8)-(9) (page 10) in that paper.

        Parameters:
            subdomains: List of fracture subdomains.

        Raises:
            ValueError: If the maximum fracture closure is negative.

        Returns:
            The decrease in fracture opening, as computed by the Barton-Bandis model.

        """
        # The maximum closure of the fracture.
        maximum_closure = self.maximum_fracture_closure(subdomains)

        # If the maximum closure is zero, the Barton-Bandis model is not valid in the
        # case of zero normal traction. In this case, we return an empty operator.
        #  If the maximum closure is negative, an error is raised.
        val = maximum_closure.evaluate(self.equation_system)
        if (
            (isinstance(val, (float, int)) and val == 0)
            or (isinstance(val, np.ndarray) and np.any(val == 0))
            or isinstance(val, pp.ad.AdArray)
            and np.any(val.val == 0)
        ):
            return Scalar(0)
        elif (
            (isinstance(val, (float, int)) and val < 0)
            or (isinstance(val, np.ndarray) and np.any(val < 0))
            or isinstance(val, pp.ad.AdArray)
            and np.any(val.val < 0)
        ):
            raise ValueError("The maximum closure must be non-negative.")

        nd_vec_to_normal = self.normal_component(subdomains)

        # The effective contact traction. Units: Pa = N/m^(nd-1)
        # The papers by Barton and Bandis assumes positive traction in contact, thus we
        # need to switch the sign.
        contact_traction = Scalar(-1) * self.contact_traction(subdomains)

        # Normal component of the traction.
        normal_traction = nd_vec_to_normal @ contact_traction

        # Normal stiffness (as per Barton-Bandis terminology). Units: Pa / m
        normal_stiffness = self.fracture_normal_stiffness(subdomains)

        # The openening is found from the 1983 paper.
        # Units: Pa * m / Pa = m.
        opening_decrease = (
            normal_traction
            * maximum_closure
            / (normal_stiffness * maximum_closure + normal_traction)
        )

        opening_decrease.set_name("Barton-Bandis_closure")

        return opening_decrease

    def maximum_fracture_closure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The maximum closure of a fracture [m].

        Used in the Barton-Bandis model for normal elastic fracture deformation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The maximum allowed decrease in fracture opening.

        """
        max_closure = self.solid.maximum_fracture_closure()
        return Scalar(max_closure, "maximum_fracture_closure")

    def fracture_normal_stiffness(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The normal stiffness of a fracture [Pa*m^-1].

        Used in the Barton-Bandis model for normal elastic fracture deformation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The fracture normal stiffness.

        """

        normal_stiffness = self.solid.fracture_normal_stiffness()
        return Scalar(normal_stiffness, "fracture_normal_stiffness")


class FractureGap(BartonBandis, ShearDilation):
    """Class for calculating the fracture gap.

    The main method of the class, :meth:``fracture_gap`` incorporates the effect of
    both shear dilation and the Barton-Bandis effect.

    """

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def fracture_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture gap [m].

        Parameters:
            subdomains: List of subdomains where the gap is defined.

        Raises:
            ValueError: If the reference fracture gap is smaller than the maximum
                fracture closure. This can lead to negative openings from the
                Barton-Bandis model.

        Returns:
            Cell-wise fracture gap operator.

        """
        barton_bandis_closure = self.elastic_normal_fracture_deformation(subdomains)

        dilation = self.shear_dilation_gap(subdomains)

        gap = self.reference_fracture_gap(subdomains) + dilation - barton_bandis_closure
        val = (
            self.reference_fracture_gap(subdomains)
            - self.maximum_fracture_closure(subdomains)
        ).evaluate(self.equation_system)

        if (
            (isinstance(val, (float, int)) and val < 0)
            or (isinstance(val, np.ndarray) and np.any(val < 0))
            or (isinstance(val, pp.ad.AdArray) and np.any(val.val < 0))
        ):
            msg = (
                "The reference fracture gap must be larger"
                " than the maximum fracture closure."
            )
            raise ValueError(msg)
        gap.set_name("fracture_gap")
        return gap

    def reference_fracture_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference fracture gap [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise reference fracture gap operator [m].

        """
        return Scalar(self.solid.fracture_gap(), "reference_fracture_gap")


class BiotCoefficient:
    """Biot coefficient."""

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Biot coefficient.

        Parameters:
            subdomains: List of subdomains where the Biot coefficient is defined.

        Returns:
            Biot coefficient operator.

        """
        return Scalar(self.solid.biot_coefficient(), "biot_coefficient")


class SpecificStorage:
    """Specific storage."""

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def specific_storage(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Specific storage [1/Pa], i.e. inverse of the Biot modulus.

        The specific storage :math:`S_\varepsilon` can also be obtained from more
        fundamental quantities, i.e., :math:`S_\varepsilon = (\alpha - \phi_0) K_d^{
        -1} + \phi_0 c_f)`, where :math:`\alpha` is the Biot's coefficient,
        :math:`\phi_0` is the reference porosity, :math:`K_d` is the solid's bulk
        modulus, and :math:`c_f` is the fluid compressibility.

        Parameters:
            subdomains: List of subdomains where the specific storage is defined.

        Returns:
            Specific storage operator.

        Note:
            Only used when :class:`BiotPoroMechanicsPorosity` is used as a part of the
            constitutive laws.

        """
        return Scalar(self.solid.specific_storage(), "specific_storage")


class ConstantPorosity:
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Constant porosity [-].

        The porosity is assumed to be constant, and is defined by the solid constant
        object.

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            The porosity represented as an Ad operator. The value is constant for all
            subdomains.

        """
        return Scalar(self.solid.porosity(), "porosity")


class PoroMechanicsPorosity:
    """Porosity for poromechanical models.

    Note:
        For legacy reasons, the discretization matrices for the
        :math:`\nabla \cdot \mathbf{u}` and stabilization terms include a volume
        integral. That factor is counteracted in :meth:`displacement_divergence` and
        :meth:`biot_stabilization`, respectively. This ensure that the returned
        operators correspond to intensive quantities and are compatible with the rest
        of this class. The assumption is that the porosity will be integrated over
        cell volumes later before entering the equation.

    """

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """
    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Biot coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.

    """
    bulk_modulus: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Bulk modulus. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.LinearElasticSolid`.

    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    darcy_keyword: str
    """Keyword used to identify the Darcy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    bc_values_mechanics: Callable[[list[pp.Grid]], pp.ad.DenseArray]
    """Mechanics boundary conditions. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsMomentumBalance`.

    """
    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    interface_displacement: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Displacement variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str, int], pp.ad.DenseArray]
    """Wrap a grid attribute as a DenseArray. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity.

        Pressure and displacement dependent porosity in the matrix. Unitary in fractures
        and intersections.

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Porosity operator.

        """
        subdomains_nd = [sd for sd in subdomains if sd.dim == self.nd]
        subdomains_lower = [sd for sd in subdomains if sd.dim < self.nd]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        # Constant unitary porosity in fractures and intersections
        size = sum([sd.num_cells for sd in subdomains_lower])
        one = pp.wrap_as_ad_array(1, size=size, name="one")
        rho_nd = projection.cell_prolongation(subdomains_nd) @ self.matrix_porosity(
            subdomains_nd
        )
        rho_lower = projection.cell_prolongation(subdomains_lower) @ one
        rho = rho_nd + rho_lower
        rho.set_name("porosity")
        return rho

    def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity in the nd-dimensional matrix [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].

        """

        # Sanity check
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")

        # Add contributions to poromechanics porosity
        phi = (
            self.reference_porosity(subdomains)
            + self.porosity_change_from_pressure(subdomains)
            + self.porosity_change_from_displacement(subdomains)
            + self.biot_stabilization(subdomains)
        )
        phi.set_name("Stabilized matrix porosity")

        return phi

    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference porosity.

        Parameters:
            subdomains: List of subdomains where the reference porosity is defined.

        Returns:
            Reference porosity operator.

        """
        return Scalar(self.solid.porosity(), "reference_porosity")

    def porosity_change_from_pressure(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Contribution of the pressure changes to the matrix porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise contribution of the pressure changes to the matrix porosity [-].

        """

        # Retrieve material parameters
        alpha = self.biot_coefficient(subdomains)
        phi_ref = self.reference_porosity(subdomains)
        bulk_modulus = self.bulk_modulus(subdomains)

        # Pressure changes
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Compute 1/N as defined in Coussy, 2004, https://doi.org/10.1002/0470092718.
        n_inv = (alpha - phi_ref) * (Scalar(1) - alpha) / bulk_modulus

        # Pressure change contribution
        pressure_contribution = n_inv * dp
        pressure_contribution.set_name("Porosity change from pressure")

        return pressure_contribution

    def porosity_change_from_displacement(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Contribution of the divergence displacement to the matrix porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise contribution of the divergence of the displacement to the
            matrix porosity. Scaling with Biot's coefficient is already included.

        """
        alpha = self.biot_coefficient(subdomains)
        div_u = self.displacement_divergence(subdomains)
        div_u_contribution = alpha * div_u
        div_u_contribution.set_name("Porosity change from displacement")
        return div_u_contribution

    def displacement_divergence(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Divergence of displacement [-].

        This is div(u). Note that opposed to old implementation, the temporal is not
        included here. Rather, it is handled by :meth:`pp.ad.dt`.

        Parameters:
            subdomains: List of subdomains where the divergence is defined.

        Returns:
            Divergence operator accounting from contributions from interior of the
            domain and from internal and external boundaries.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Displacement divergence only defined in nd.")

        # Obtain neighbouring interfaces
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Mock discretization (empty `discretize` method), used to access discretization
        # matrices computed by Biot discretization.
        discr = pp.ad.DivUAd(self.stress_keyword, subdomains, self.darcy_keyword)
        # Projections
        sd_projection = pp.ad.SubdomainProjections(subdomains, dim=self.nd)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=self.nd
        )
        bc_values = self.bc_values_mechanics(subdomains)

        # Compose operator.
        div_u_integrated = discr.div_u @ self.displacement(
            subdomains
        ) + discr.bound_div_u @ (
            bc_values
            + sd_projection.face_restriction(subdomains)
            @ mortar_projection.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        # Divide by cell volumes to counteract integration.
        # The div_u discretization contains a volume integral. Since div u is used here
        # together with intensive quantities, we need to divide by cell volumes.
        cell_volumes_inv = Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        div_u = cell_volumes_inv * div_u_integrated
        div_u.set_name("div_u")
        return div_u

    def biot_stabilization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Biot stabilization term.

        TODO: Determine if this is the correct place to include stabilization.

        Parameters:
            subdomains: List of subdomains where the stabilization is defined.

        Returns:
            Biot stabilization operator.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Biot stabilization only defined in nd.")

        discr = pp.ad.BiotStabilizationAd(self.darcy_keyword, subdomains)
        # The stabilization is based on perturbation. If pressure is used directly,
        # results will not match if the reference state is not zero, see
        # :func:`test_without_fracture` in test_poromechanics.py.
        dp = self.perturbation_from_reference("pressure", subdomains)
        stabilization_integrated = discr.stabilization @ dp

        # Divide by cell volumes to counteract integration.
        # The stabilization discretization contains a volume integral. Since the
        # stabilization term is used here together with intensive quantities, we need to
        # divide by cell volumes.
        cell_volumes_inverse = Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        stabilization = cell_volumes_inverse * stabilization_integrated
        stabilization.set_name("biot_stabilization")
        return stabilization


class BiotPoroMechanicsPorosity(PoroMechanicsPorosity):
    """Porosity for poromechanical models following classical Biot's theory.

    The porosity is defined such that, after the chain rule is applied to the
    accumulation term, the classical conservation equation from the Biot
    equations is recovered.

    Note that we assume constant fluid density and constant specific storage.

    """

    specific_storage: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Specific storage. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.SpecificStorage`.

    """

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def porosity_change_from_pressure(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Contribution of the pressure changes to the matrix porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise contribution of the pressure changes to the matrix porosity [-].

        """
        specific_storage = self.specific_storage(subdomains)
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Pressure change contribution
        pressure_contribution = specific_storage * dp
        pressure_contribution.set_name("Biot's porosity change from pressure")

        return pressure_contribution


class ThermoPoroMechanicsPorosity(PoroMechanicsPorosity):
    """Add thermal effects to matrix porosity."""

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """
    solid_thermal_expansion: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Thermal expansion coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalExpansion`.
    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Biot coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.

    """

    def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].

        """
        # Inherit poromechanical porosity from base class.
        phi = super().matrix_porosity(subdomains)
        # Add thermal contribution.
        phi += self.porosity_change_from_temperature(subdomains)
        phi.set_name("Thermoporomechanics porosity")
        return phi

    def porosity_change_from_temperature(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Thermal contribution to the changes in porosity [-].

        TODO: Discuss cf. Coussy p. 73. Not sure about the interpretation of alpha_phi.

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise thermal porosity expansion operator [-].

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")
        dtemperature = self.perturbation_from_reference("temperature", subdomains)
        phi_ref = self.reference_porosity(subdomains)
        beta = self.solid_thermal_expansion(subdomains)
        alpha = self.biot_coefficient(subdomains)
        # TODO: Figure out why * is needed here, but not in
        # porosity_change_from_pressure.
        phi = Scalar(-1) * (alpha - phi_ref) * beta * dtemperature
        phi.set_name("Porosity change from temperature")
        return phi
