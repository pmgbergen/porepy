"""Library of constitutive equations."""
from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp

number = pp.number
Scalar = pp.ad.Scalar

ArrayType = TypeVar("ArrayType", pp.ad.AdArray, np.ndarray)


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
            return pp.wrap_as_dense_ad_array(0, size=0)
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # The implementation here is not perfect, but it seems to be what is needed
        # to make the Ad framework happy: Build the global array by looping over
        # subdomains and add the local contributions.
        # Note that the aperture is an array (in the Ad sense) not a matrix, thus there
        # is no risk of the number of columns being wrong (as there would be if we
        # were to wrap the aperture as an Ad matrix).

        for i, sd in enumerate(subdomains):
            # First make the local aperture array.
            a_loc = pp.wrap_as_dense_ad_array(self.grid_aperture(sd))
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
            return pp.wrap_as_dense_ad_array(0, size=0)

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
        one = pp.wrap_as_dense_ad_array(1, size=num_cells_nd_subdomains, name="one")
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
                        well_apertures = pp.wrap_as_dense_ad_array(
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
                    parent_cells_to_intersection_cells.value(self.equation_system).sum(
                        axis=1
                    )
                )
                nonzero = average_weights > 0
                average_weights[nonzero] = 1 / average_weights[nonzero]
                # Wrap as a DenseArray
                divide_by_num_neighbors = pp.wrap_as_dense_ad_array(
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

    def fluid_compressibility(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid compressibility.

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The constant compressibility of the fluid [Pa^-1], represented as an Ad
            operator. The value is taken from the fluid constants.

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
            Fluid density as a function of pressure [kg*m^-3].

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
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

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


class SecondOrderTensorUtils:
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.SparseArray]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Function that returns the specific volume of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Solution strategies are normally
    defined in a mixin class.

    """

    def isotropic_second_order_tensor(
        self, subdomains: list[pp.Grid], permeability: pp.ad.Operator
    ) -> pp.ad.Operator:
        """Isotropic permeability [m^2].

        Parameters:
            permeability: Permeability, scalar per cell.

        Returns:
            3d isotropic permeability, with nonzero values on the diagonal and zero
            values elsewhere. K is a second order tensor having 3^2 entries per cell,
            represented as an array of length 9*nc. The values are ordered as
                Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz

        """
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)
        basis = self.basis(subdomains, 9)
        diagonal_indices = [0, 4, 8]
        permeability = pp.ad.sum_operator_list(
            [basis[i] @ permeability for i in diagonal_indices]
        )
        permeability.set_name("isotropic_second_order_tensor")
        return permeability

    def operator_to_SecondOrderTensor(
        self,
        sd: pp.Grid,
        operator: pp.ad.Operator,
        fallback_value: number,
    ) -> pp.SecondOrderTensor:
        """Convert Ad operator to PorePy tensor representation.

        Parameters:
            sd: Subdomain where the operator is defined.
            operator: Operator to convert.

        Returns:
            SecondOrderTensor representation of the operator.

        """
        # Evaluate as 9 x num_cells array
        volume = self.specific_volume([sd]).value(self.equation_system)
        try:
            permeability = operator.value(self.equation_system)
        except KeyError:
            # If the permeability depends on an not yet computed discretization matrix,
            # fall back on reference value.
            permeability = fallback_value * np.ones(sd.num_cells) * volume
            return pp.SecondOrderTensor(permeability)
        evaluated_value = operator.value(self.equation_system)
        if not isinstance(evaluated_value, np.ndarray):
            # Raise error rather than cast for verbosity of function which is not
            # directly exposed to the user, but depends on a frequently user-defined
            # quantity (the tensor being converted).
            raise ValueError(
                f"Operator {operator.name} has type {type(evaluated_value)}, "
                f"expected numpy array for conversion to SecondOrderTensor."
            )
        val = evaluated_value.reshape(9, -1, order="F")
        # SecondOrderTensor's constructor expects up to six entries: kxx, kyy, kzz,
        # kxy, kxz, kyz. These correspond to entries 0, 4, 8, 1, 2, 5 in the 9 x
        # num_cells array.
        diagonal_indices = [0, 4, 8]
        tensor_components = [val[i] for i in diagonal_indices]
        # Check that the operator is indeed symmetric.
        off_diagonal_indices = [1, 2, 5]
        other_indices = [3, 6, 7]
        for i, j in zip(off_diagonal_indices, other_indices):
            if not np.allclose(val[i], val[j]):
                raise ValueError(f"Operator is not symmetric for indices {i} and {j}.")
            tensor_components.append(val[i])

        # Scale by specific volume
        args = [a * volume for a in tensor_components]
        return pp.SecondOrderTensor(*args)


class ConstantPermeability(SecondOrderTensorUtils):
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
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.permeability(), size, name="permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, permeability)

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


class DimensionDependentPermeability(ConstantPermeability):
    """Permeability depending on subdomain dimension.

    The use of sub-methods allows convenient code reuse if the permeability
    for one of the subdomain sets is changed.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        This function combines the permeability of the matrix, fractures and
        intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=9)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures: list[pp.Grid] = [sd for sd in subdomains if sd.dim == self.nd - 1]
        intersections: list[pp.Grid] = [sd for sd in subdomains if sd.dim < self.nd - 1]

        permeability = (
            projection.cell_prolongation(matrix) @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(fractures)
            @ self.fracture_permeability(fractures)
            + projection.cell_prolongation(intersections)
            @ self.intersection_permeability(intersections)
        )
        return permeability

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the matrix.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return super().permeability(subdomains)

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(1, size, name="permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of intersections.

        Note that as permeability is not meaningful in 0d domains, this method will only
        impact the tangential permeability of 1d intersection lines.


        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(1, size, name="permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)


class CubicLawPermeability(DimensionDependentPermeability):
    """Cubic law permeability for fractures and intersections.

    The cubic law is derived from the Navier-Stokes equations, and is valid for
    laminar flow under the so-called parallel plate assumption. This gives a cubic
    relationship between the volumetric flow rate and the pressure drop. Note that in
    PorePy, the permeability is multiplied by the aperture to yield a transmissivity,
    thus the permeability is proportional to the square of the aperture.

    """

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Solution strategies are normally
    defined in a mixin class.

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

    def cubic_law_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic law permeability for fractures or intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        aperture = self.aperture(subdomains)
        permeability = (aperture ** Scalar(2)) / Scalar(12)
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)


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
    bc_data_darcy_flux_key: str
    """See
    :attr:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow.
    bc_data_darcy_flux_key`.

    """
    bc_type_darcy_flux: Callable[[pp.Grid], pp.BoundaryCondition]

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

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
    volume_integral: Callable[
        [pp.ad.Operator, Sequence[pp.Grid] | Sequence[pp.MortarGrid], int],
        pp.ad.Operator,
    ]
    """Integration over cell volumes, implemented in
    :class:`pp.models.abstract_equations.BalanceEquation`.

    """
    gravity_force: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]], Literal["fluid", "solid"]],
        pp.ad.Operator,
    ]
    """Gravity force. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.GravityForce` or
    :class:`~porepy.models.constitutive_laws.ZeroGravityForce`.

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

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name="bc_values_darcy",
        )

        pressure_trace = (
            discr.bound_pressure_cell @ p
            + discr.bound_pressure_face
            @ (projection.mortar_to_primary_int @ self.interface_darcy_flux(interfaces))
            + discr.bound_pressure_face @ boundary_operator
            + discr.bound_pressure_vector_source
            @ self.vector_source_darcy_flux(subdomains)
        )
        return pressure_trace

    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of Darcy's law.

        Note:
            The fluid mobility is not included in the Darcy flux. This is because we
            discretize it with an upstream scheme. This means that the fluid mobility
            may have to be included when using the flux in a transport equation.
            The units of the Darcy flux are [m^2 Pa / s].

        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_darcy_flux_key,
                domains=domains,
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument `domains` should either be a list of grids or a list of
                boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name="bc_values_" + self.bc_data_darcy_flux_key,
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            domains
        )
        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(domains)
            + discr.bound_flux
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source @ self.vector_source_darcy_flux(domains)
        )
        flux.set_name("Darcy_flux")
        return flux

    def interface_darcy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Darcy flux on interfaces.

        The units of the Darcy flux are [m^2 Pa / s], see note in :meth:`darcy_flux`.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # Gradient operator in the normal direction. The collapsed distance is
        # :math:`\frac{a}{2}` on either side of the fracture.
        # We assume here that :meth:`aperture` is implemented to give a meaningful value
        # also for subdomains of co-dimension > 1.
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg @ self.aperture(subdomains) ** Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        # Project the two pressures to the interface and multiply with the normal
        # diffusivity.
        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(subdomains)
        eq = self.interface_darcy_flux(interfaces) - self.volume_integral(
            self.normal_permeability(interfaces)
            * (
                normal_gradient * (pressure_h - pressure_l)
                + self.interface_vector_source_darcy_flux(interfaces)
            ),
            interfaces,
            1,
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
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source_darcy_flux(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]]
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.

        Returns:
            Cell-wise nd-vector source term operator.

        """
        return self.gravity_force(grids, "fluid")

    def interface_vector_source_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
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
        # int? @EK
        vector_source = (
            projection.secondary_to_mortar_avg
            @ self.vector_source_darcy_flux(subdomain_neighbors)
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


class AdTpfaFlux:
    """Differentiable discretization of a diffusive flux.

    The diffusive flux is given by

        q = - K (grad p - g)

    where K is the diffusivity tensor and p is the primary variable/potential. In the
    case of Darcy's law, the diffusivity tensor is the permeability tensor and the
    primary variable is the pressure. The implementation is agnostic to this, and can be
    used for other constitutive laws as well (e.g. Fourier's law).

    To use for a specific constitutive law, the following methods must be used when
    overriding specific methods:
    - diffusive_flux: Discretization of the diffusive flux. This method should be called
        by the overriding method (darcy_flux, fourier_flux etc).
    - potential_trace: Discretization of the potential on the subdomain boundaries. This
        method should be called by the overriding method (pressure_trace,
        temperature_trace etc).

    Note:
        This class implicitly assumes conventions on naming of methods and BC value
        keys. Specifically, the BC values keys are assumed to be of the form
        "bc_values_" + flux_name, where flux_name is the name of the flux (e.g.
        "darcy_flux" or "fourier_flux"). The same goes for "inteface_" + flux_name and
        flux_name + "_discretization". These conventions are used to simplify the
        implementation of the class' methods. TODO: Consider making this more explicit.

    """

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    subdomains_to_boundary_grids: Callable[
        [Sequence[pp.Grid]], Sequence[pp.BoundaryGrid]
    ]
    """Function that maps a sequence of subdomains to a sequence of boundary grids.
    Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    nd: int
    """Number of spatial dimensions. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]
    """Combine Dirichlet and Neumann boundary conditions. Normally defined in a mixin
    instance of :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.SparseArray]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Function that returns the specific volume of a subdomain or interface.

    Normally provided by a mixin of instance
    :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """

    def diffusive_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
        potential: Callable[[list[pp.Grid]], pp.ad.Operator],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
        flux_name: str,
    ) -> pp.ad.Operator:
        """Discretization of a diffusive constitutive law.

        Parameters:
            domains: List of domains where the flux is defined.
            diffusivity_tensor: Function returning the diffusivity tensor as an Ad
                operator. For Darcy's and Fourier's law, this is the permeability and
                thermal conductivity, respectively.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise integrated flux.

        """
        # NOTE (relevant for debugging): The returned derivative of the transmissibility
        # is of the form
        #   dp * dT/d(psi)    (psi is a primary variable)
        # where dp is the pressure difference across the face. Experience has shown that
        # the dp factor is easily overlooked when comparing a computed and a 'known'
        # value, resulting in frustration.

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of an empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=flux_name,
                domains=domains,
            )
        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and boundary grids."""
            )

        domains = cast(list[pp.Grid], domains)
        boundary_grids = self.subdomains_to_boundary_grids(domains)
        interfaces: Sequence[pp.MortarGrid] = self.subdomains_to_interfaces(
            domains, [1]
        )
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        # Compute the transmissibility matrix, see the called function for details. Also
        # obtain various helper objects.
        (
            t_f,
            diff_discr,
            hf_to_f,
            d_vec,
        ) = self.__transmissibility_matrix(domains, diffusivity_tensor)

        # Treatment of boundary conditions.
        one = pp.ad.Scalar(1)
        # Obtain filters that lett pass external boundary faces (divided into Dirichlet
        # and Neumann faces), internal boundary faces (faces between subdomains), and
        # tip faces (on immersed tips of domains).
        external_dir_filter, external_neu_filter = diff_discr.external_boundary_filters(
            self.mdg, domains, boundary_grids, "bc_values_" + flux_name
        )
        internal_boundary_filter = diff_discr.internal_boundary_filter(domains)
        tip_filter = diff_discr.tip_filter(domains)

        # The Tpfa transmissibility of a face with a Neumann condition is zero. Thus
        # eliminate the transmissibilities on external Neumann faces, as well as on
        # internal boundaries and tip faces, both of which by assumption are assigned
        # Neumann conditions.
        t_f = (
            one - (external_neu_filter + internal_boundary_filter + tip_filter)
        ) * t_f

        # Sign of boundary faces.
        bnd_sgn = diff_discr.boundary_sign(domains)
        # Discretization of boundary conditions: On Neumann faces, we will simply add
        # the flux, with a sign change if the normal vector is pointing inwards.
        neu_bnd = external_neu_filter * bnd_sgn
        # On Dirichlet faces, an assigned Dirichlet value (assuming zero potential in
        # the cell adjacent to the face) will induce a flux proportional to t_f. (The
        # actual flux through the face is found by adding the contribution from the cell
        # center potential, this is taken care of).
        dir_bnd = external_dir_filter * (-bnd_sgn * t_f)
        t_bnd = neu_bnd + dir_bnd

        # Discretization of vector source:
        #
        # The flux through a face with normal vector n_j, as seen from cell i, driven by
        # a vector source v_i in cell i, is given by
        #
        #   q_j = n_j^T K_i v_i
        #
        # A Tpfa-style discretization of this term will apply harmonic averaging of the
        # permeabilities (see function __transmissibility_matrix), and multiply with the
        # difference in vector source between the two cells. We have already computed
        # the transmissibility matrix, which computes the product of the permeability
        # tensor, the normal vector and a unit vector from cell to face center. To
        # convert this to a discretizaiton for the vector source, we first need to
        # project the vector source onto the unit vector from cell to face center.
        # Second, the vector source should be scaled by the distance from cell to face
        # center. This can be seen as compensating for the distance in the denominator
        # of the half-face transmissibility, or as converting the vector source into a
        # potential-like quantity before applying the flux calculation.

        # The vector source can be 2d or 3d, but the geometry, thus discretization, is
        # always 3d, thus we need to map from nd to 3d.
        cells_nd_to_3d = diff_discr.nd_to_3d(domains, self.nd)
        # Mapping from cells to half-faces of 3d quantities.
        cells_to_hf_3d = diff_discr.half_face_map(
            domains, from_entity="cells", with_sign=False, dimensions=(3, 3)
        )

        # Build a mapping for the cell-wise vector source, unravelled from the right:
        # First, map the vector source from nd to 3d. Second, map from cells to
        # half-faces. Third, project the vector source onto the vector from cell center
        # to half-face center (this is the vector which Tpfa uses as a proxy for the
        # full gradient, see comments in the method __transmissibility_matrix). As the
        # rows of d_vec have length equal to the distance, this compensates for the
        # distance in the denominator of the half-face transmissibility. Fourth, map
        # from half-faces to faces, using a mapping with signs, thereby taking the
        # difference between the two vector sources.
        vector_source_c_to_f = pp.ad.SparseArray(
            hf_to_f @ d_vec @ cells_to_hf_3d @ cells_nd_to_3d
        )

        # Fetch the constitutive law for the vector source.
        vector_source_cells = getattr(self, "vector_source_" + flux_name)(domains)

        # Compute the difference in pressure and vector source between the two cells on
        # the sides of each face.
        potential_difference = pp.ad.SparseArray(
            diff_discr.face_pairing_from_cell_array(domains)
        ) @ potential(domains)
        vector_source_difference = vector_source_c_to_f @ vector_source_cells

        # Fetch the discretization of the Darcy flux
        base_discr = getattr(self, flux_name + "_discretization")(domains)

        # Compose the discretization of the Darcy flux q = T(k(u)) * p, (where the k(u)
        # dependency can be replaced by other primary variables. The chain rule gives
        #
        #  dT = p * (dT/du) * du + T dp
        #
        # A similar expression holds for the vector source term. If the base
        # discretization (which calculates T in the above equation) is Tpfa, the full
        # expression will be obtained by the Ad machinery and there is no need for
        # special treatment. If the base discretization is Mpfa, we need to mix this
        # T-matrix with the the Tpfa-style approximation of dT/du, as is done in the
        # below if-statement.
        if isinstance(base_discr, pp.ad.MpfaAd):
            # To obtain a mixture of Tpfa and Mpfa, we utilize pp.ad.Function, one for
            # the flux and one for the vector source.

            # Define the Ad function for the flux
            flux_p = pp.ad.Function(
                partial(self.__mpfa_flux_discretization, base_discr),
                "differentiable_mpfa",
            )(t_f, potential_difference, potential(domains))

            # Define the Ad function for the vector source
            vector_source_d = pp.ad.Function(
                partial(self.__mpfa_vector_source_discretization, base_discr),
                "differentiable_mpfa_vector_source",
            )(t_f, vector_source_difference, vector_source_cells)

        else:
            # The base discretization is Tpfa, so we can rely on the Ad machinery to
            # compose the full expression.
            flux_p = t_f * potential_difference
            vector_source_d = t_f * vector_source_difference

        # As the base discretization is only invoked inside a function, and then only by
        # the parse()-method, that is, not on operator form, it will not be found in the
        # search for discretization schemes in the operator tree (implemented in the
        # Operator class), and therefore, it will not actually be discretized. To
        # circumvent this problem, we artifically add a term that involves the base
        # discretization on operator form, and multiply it by zero to avoid it having
        # any real impact on the equation. This is certainly an ugly hack, but it will
        # have to do for now.
        flux_p = flux_p + pp.ad.Scalar(0) * base_discr.flux @ potential(domains)

        # Get boundary condition values
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=potential,
            neumann_operator=getattr(self, flux_name),
            bc_type=getattr(self, "bc_type_" + flux_name),
            name="bc_values_" + flux_name,
        )

        # Compose the full discretization of the Darcy flux, which consists of three
        # terms: The flux due to pressure differences, the flux due to boundary
        # conditions, and the flux due to the vector source.
        flux: pp.ad.Operator = (
            flux_p
            + t_bnd
            * (
                boundary_operator
                + intf_projection.mortar_to_primary_int
                @ getattr(self, "interface_" + flux_name)(interfaces)
            )
            + vector_source_d
        )
        flux.set_name("Differentiable diffusive flux")
        return flux

    def potential_trace(
        self,
        subdomains: list[pp.Grid],
        potential: Callable[[list[pp.Grid]], pp.ad.Operator],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
        flux_name: str,
    ) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=potential,
            neumann_operator=getattr(self, flux_name),
            bc_type=getattr(self, "bc_type_" + flux_name),
            name="bc_values_" + flux_name,
        )
        base_discr = getattr(self, flux_name + "_discretization")(subdomains)
        # Obtain the transmissibilities in operator form. Ignore other outputs.
        t_f_full, *_ = self.__transmissibility_matrix(subdomains, diffusivity_tensor)
        one = pp.ad.Scalar(1)

        # BC filters for Dirichlet and Neumann faces.
        diff_discr = pp.numerics.fv.tpfa.DifferentiableTpfa()
        dir_filter, neu_filter = diff_discr.external_boundary_filters(
            self.mdg, subdomains, boundary_grids, "bc_values_" + flux_name
        )
        # Also a separate filter for internal boundaries, which are always Neumann.
        internal_boundary_filter = diff_discr.internal_boundary_filter(subdomains)

        # Face contribution to boundary potential is 1 on Dirichlet faces, -1/t_f_full
        # on Neumann faces (both external and internal - see Tpfa.discretize). Named
        # "bound_pressure_face" and not "bound_potential_face" to be consistent with the
        # base discretization.
        bound_pressure_face_discr = dir_filter - (
            neu_filter + internal_boundary_filter
        ) * (one / t_f_full)

        # Project the interface flux to the primary grid, preparing for discretization
        # on internal boundaries.
        projected_internal_flux = projection.mortar_to_primary_int @ getattr(
            self, "interface_" + flux_name
        )(interfaces)

        if isinstance(base_discr, pp.ad.MpfaAd):
            # Approximate the derivative of the transmissibility matrix with respect to
            # permeability by a Tpfa-style discretization.
            boundary_value_contribution = pp.ad.Function(
                partial(self.__mpfa_bound_pressure_discretization, base_discr),
                "differentiable_mpfa",
            )(
                bound_pressure_face_discr,
                projected_internal_flux,
                boundary_operator,
            )

        else:
            # The base discretization is Tpfa, so we can rely on the Ad machinery to
            # compose the discretization, treating internal and external boundaries
            # equally.
            boundary_value_contribution = bound_pressure_face_discr * (
                projected_internal_flux + boundary_operator
            )

        # As the base discretization is only invoked inside a function, and then only by
        # the parse()-method, that is, not on operator form, it will not be found in the
        # search for discretization schemes in the operator tree (implemented in the
        # Operator class), and therefore, it will not actually be discretized. To
        # circumvent this problem, we artifically add a term that involves the base
        # discretization on operator form, and multiply it by zero to avoid it having
        # any real impact on the equation. This is certainly an ugly hack, but it will
        # have to do for now.
        # TODO: Do we need this trick here, or is it sufficient to do so in the
        # diffusive flux method?
        boundary_value_contribution = boundary_value_contribution + pp.ad.Scalar(
            0
        ) * base_discr.flux @ potential(subdomains)

        pressure_trace = (
            base_discr.bound_pressure_cell @ potential(subdomains)
            # Contribution from boundaries.
            + boundary_value_contribution
            # the vector source is independent of k
            + base_discr.bound_pressure_vector_source
            @ getattr(self, "vector_source_" + flux_name)(subdomains)
        )
        return pressure_trace

    def __transmissibility_matrix(
        self,
        subdomains: list[pp.Grid],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
    ) -> tuple[
        pp.ad.Operator,
        pp.numerics.fv.tpfa.DifferentiableTpfa,
        sps.spmatrix,
        sps.spmatrix,
    ]:
        """Compute the Tpfa transmissibility matrix for a list of subdomains."""
        # In Tpfa, the Darcy flux through a face with normal vector n_j, as seen from
        # cell i, is given by (subscripts indicate face or cell index)
        #
        #    q_j = n_j^T K_i e_ij (p_i - p_j) / dist_ij
        #
        # Here, K_i is the permeability tensor in cell i, e_ij is the unit vector from
        # cell i to face j, and dist_ij is the distance between the cell center and the
        # face center. Comparing with the continuous formulation, we see that the
        # pressure gradient is approximated by the pressure difference, divided by
        # distance, in the direction between cell and face centers. Writing out the
        # expression for the half-face transmissibility
        #
        #    t = n_r^T K_rs e_s / dist
        #
        # Here, subscripts indicate (Cartesian) dimensions, the summation convention is
        # applied, and dist again represent the distance from cell to face center. (EK:
        # the change of meaning of subscript is unfortunate, but it is very important to
        # understand how the components of the permeability tensor and the normal and
        # distance vectors are multiplied.) This formulation can be reformulated to
        #
        #   t = n_r^T e_s K_rs / dist
        #
        # where the point is that, by right multiplying the permeability tensor, this
        # can be represented as an Ad operator (which upon parsing will be an AdArray
        # which only can be right multiplied). The below code implements this
        # formulation. The full transmissibility matrix is obtained by taking the
        # harmonic mean of the two half-face transmissibilities on each face.

        # The cell-wise permeability tensor is represented as an Ad operator which
        # evaluates to an AdArray with 9 * n_cells entries. Also scale with specific
        # volume.
        basis = self.basis(subdomains, dim=9)  # type: ignore[call-arg]
        volumes = pp.ad.sum_operator_list(
            [e @ self.specific_volume(subdomains) for e in basis]
        )
        k_c = volumes * diffusivity_tensor(subdomains)

        # Create the helper discretization object, which will be used to generate
        # grid-related quantities and mappings.
        diff_discr = pp.numerics.fv.tpfa.DifferentiableTpfa()

        # Get the normal vector, vector from cell center to face center (d_vec), and
        # distance from cell center to face center (dist) for each half-face.
        n, d_vec, dist = diff_discr.half_face_geometry_matrices(subdomains)

        # Compose the geometric part of the half-face transmissibilities. Note that
        # dividing d_vec by dist essentially forms a unit vector from cell to face
        # center.
        d_n_by_dist = sps.diags(1 / dist) * d_vec @ n

        # Form the full half-face transmissibilities and take its reciprocal, preparing
        # for a harmonic mean between the two half-face transmissibilities on ecah side
        # of a face.
        one = pp.ad.Scalar(1)
        t_hf_inv = one / (pp.ad.SparseArray(d_n_by_dist) @ k_c)

        # Compose full-face transmissibilities
        # Sum over half-faces to get transmissibility on faces.
        # Include sign to cancel the effect of the d_vec @ n having opposite signs on
        # the two half-faces.
        hf_to_f = diff_discr.half_face_map(
            subdomains, to_entity="faces", with_sign=True
        )
        # Take the harmonic mean of the two half-face transmissibilities.
        t_f_full = one / (pp.ad.SparseArray(hf_to_f) @ t_hf_inv)
        t_f_full.set_name("transmissibility matrix")
        return t_f_full, diff_discr, hf_to_f, d_vec

    def __mpfa_flux_discretization(
        self, base_discr: pp.ad.MpfaAd, T_f: ArrayType, p_diff: ArrayType, p: ArrayType
    ) -> ArrayType:
        """Approximate the product rule for the expression d(T_MPFA * p), where T_MPFA
        is the transmissibility matrix for an Mpfa discretization.

        The approximation is taken as

            d(T_MPFA * p) ~ T_MPFA * dp + p_diff * d(T_TPFA)

        where, for a single face, p_diff is the difference in pressure between the two
        cells on either side of the face - the method self.diffusive_flux() for
        further explanation.

        Parameters:
            base_discr: Base discretization of the flux. T_f: Transmissibility matrix.
            p_diff: Difference in potential between the two cells on either side of the
                face.
            p: Potential.

        Returns:
            AdArray with value and Jacobian matrix representing the flux associated with
                the potential difference.

        """
        # NOTE: Keep in mind that this functions will be evaluated in forward mode, thus
        # the inputs are not Ad-operators, but numerical values.

        # We know that base_discr.flux is a sparse matrix, so we can call parse
        # directly.
        base_flux = base_discr.flux.parse(self.mdg)
        # If the function has been called using .value, p is a numpy array and we pass
        # only the value.
        if not isinstance(p, pp.ad.AdArray):
            return base_flux @ p
        # Otherwise, at the time of evaluation, p will be an AdArray, thus we can access
        # its val and jac attributes.
        val = base_flux @ p.val
        jac = base_flux @ p.jac

        if hasattr(T_f, "jac"):
            # Add the contribution to the Jacobian matrix from the derivative of the
            # transmissibility matrix times the pressure difference. To see why this is
            # correct, it may be useful to consider the flux over a single face
            # (corresponding to one row in the Jacobian matrix).
            jac += sps.diags(p_diff.val) @ T_f.jac

        return pp.ad.AdArray(val, jac)

    def __mpfa_vector_source_discretization(
        self,
        base_discr: pp.ad.MpfaAd,
        T_f: ArrayType,
        vs_diff: ArrayType,
        vs: ArrayType,
    ) -> ArrayType:
        """Approximate the product rule for the expression d(VS_MPFA * vs), where
        VS_MPFA is the Mpfa discretization of the vector source, and vs is the vector
        source.

         The approximation is taken as

            d(VS_MPFA * vs) ~ VS_MPFA * d(vs) + vs_diff * d(VS_TPFA)

        where, for a single face, vs_diff is the difference in vector source between
        the two cells on either side of the face, see the method self.diffusive_flux()
        for further explanation.

        Parameters:
            base_discr: Base discretization of the vector source.
            T_f: Transmissibility matrix.
            vs_diff: Difference in vector source between the two cells on either side of
                the face.
            vs: Vector source.

        Returns:
            AdArray with value and Jacobian matrix representing the flux associated with
            the vector source.

        """
        # NOTE: Keep in mind that this functions will be evaluated in forward mode, thus
        # the inputs are not Ad-operators, but numerical values.

        # We know that base_discr.vector_source is a sparse matrix, so we can call parse
        # directly.
        base_discr_vector_source = base_discr.vector_source.parse(self.mdg)

        # Composing the full expression for the vector source term is a bit tricky, as
        # any of the three arguments may be either a numpy array or an AdArray. We need
        # to handle all combinations of these cases.
        #
        # First, we check if any of the arguments are numpy arrays, which would
        # correspond to the case where the Operator is evaluated using .value(). In this
        # case, we simply return the product of the base discretization with the vector
        # source.
        if (
            isinstance(T_f, np.ndarray)
            and isinstance(vs, np.ndarray)
            and isinstance(vs_diff, np.ndarray)
        ):
            # The value is a numpy array, and we simply return the product with the base
            # discretization.
            return base_discr_vector_source @ vs

        # We now know that the return value should be an AdArray, and we need to compute
        # the Jacobian as well as the value. However, the type of vs (thus vs_diff) can
        # still be either numpy array or AdArray: The former corresponds to a constant
        # vector source, the latter to a vector source that depends on the primary
        # variable (e.g., a non-constant density in a gravity term). We need to unify
        # these cases:
        if isinstance(vs, np.ndarray):
            # If this is broken, something really weird is going on.
            assert isinstance(vs_diff, np.ndarray)
            vs_val = vs
            vs_diff_val = vs_diff

            num_rows = vs_val.size
            num_cols = self.equation_system.num_dofs()
            vs_jac = sps.csr_matrix((num_rows, num_cols))
        else:
            # The value is an AdArray, and we can access its val and jac attributes.
            vs_val = vs.val
            vs_diff_val = vs_diff.val
            vs_jac = vs.jac

        # The value is an AdArray, and we need to compute the Jacobian as well as the
        # value.
        val = base_discr_vector_source @ vs_val
        # The contribution from differentiating the vector source term to the Jacobian
        # of the flux.
        jac = base_discr.vector_source.parse(self.mdg) @ vs_jac

        if hasattr(T_f, "jac"):
            # Add the contribution to the Jacobian matrix from the derivative of the
            # transmissibility matrix times the vector source difference.
            jac += sps.diags(vs_diff_val) @ T_f.jac

        return pp.ad.AdArray(val, jac)

    def __mpfa_bound_pressure_discretization(
        self,
        base_discr: pp.ad.MpfaAd,
        bound_pressure_face: ArrayType,
        internal_flux: ArrayType,
        external_bc: ArrayType,
    ) -> ArrayType:
        """Approximate the product rule for the expression d(PT_MPFA * bc), where
        PT_MPFA is the Mpfa discretization of the potential trace reconstruction, and bc
        represents boundary conditions (internal and external).

        The approximation is taken as

            d(PT_MPFA * bc) ~ PT_MPFA * d(bc) + bc * d(PT_TPFA)

        where, for a single face, bc is the boundary condition value, and d(bc) is the
        differential of the boundary condition value. The latter will be non-zero *only*
        for internal boundaries (where it will typically represent the derivative of
        an interface flux).

        Parameters:
            base_discr: Base discretization of the pressure trace. bound_pressure_face:
            Pressure trace discretization, computed with
                differentiable tpfa.
            internal_flux: Interface fluxes.
            external_bc: External boundary conditions.

        Returns:
            AdArray with value and Jacobian matrix representing the reconstructed
                pressure trace.

        """
        # NOTE: Keep in mind that this functions will be evaluated in forward mode, thus
        # the inputs are not Ad-operators, but numerical values.

        # We know that base_discr.bound_pressure_face is a sparse matrix, so we can call
        # parse directly. At the time of evaluation, internal_flux will be an AdArray,
        # thus we can access its val and jac attributes, while external_flux is a numpy
        # array.
        base_term = base_discr.bound_pressure_face.parse(self.mdg)
        # The value is the standard product of the matrix and boundary values.

        # If the function has been called using .value, p is a numpy array and we pass
        # only the value.
        if not isinstance(internal_flux, pp.ad.AdArray):
            return base_term @ (internal_flux + external_bc)

        # Otherwise, at the time of evaluation, internal_flux will be an AdArray, thus
        # we can access its val and jac attributes.

        # EK: Testing revealed a case where the external_bc was an AdArray. The precise
        # reason for this is not clear (it could be a straightforward result of the
        # rules of parsing), but to cover all cases, we do a if-else here.
        if isinstance(external_bc, pp.ad.AdArray):
            external_bc_val = external_bc.val
        else:
            external_bc_val = external_bc

        # Use external_bc (both Dirichlet and Neumann) since both enter into the
        # pressure trace reconstruction.
        val = base_term @ (internal_flux.val + external_bc_val)
        # The Jacobian matrix has one term corresponding to the standard (e.g.,
        # non-differentiable FV) discretization. No need to add the external boundary
        # values, as they should not be differentiated.
        jac = base_term @ internal_flux.jac

        if hasattr(bound_pressure_face, "jac"):
            # If the permeability, thus the pressure reconstruction operator, has a
            # Jacobian, add its contribution. For external Dirichlet boundaries, the
            # Jacobian is zero (the element is constant 1), thus these faces give no
            # contribution. There will be a contribution from external Neumann
            # boundaries, as well as from internal boundaries (which are always
            # Neumann).
            jac += (
                sps.diags(internal_flux.val + external_bc_val) @ bound_pressure_face.jac
            )

        return pp.ad.AdArray(val, jac)


class DarcysLawAd(AdTpfaFlux, DarcysLaw):
    """Adaptive discretization of the Darcy flux from generic adaptive flux class."""

    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the permeability of a subdomain. Normally provided by a
    mixin class with a suitable permeability definition.

    """

    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of Darcy's law.


        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """
        flux = self.diffusive_flux(
            domains, self.pressure, self.permeability, "darcy_flux"
        )
        return flux

    def pressure_trace(self, domains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        pressure_trace = self.potential_trace(
            domains, self.pressure, self.permeability, "darcy_flux"
        )
        pressure_trace.set_name("Differentiable pressure trace")
        return pressure_trace


class PeacemanWellFlux:
    """Well fluxes.

    Relations between well fluxes and pressures are implemented in this class.
    Peaceman 1977 https://doi.org/10.2118/6893-PA

    Assumes permeability is cell-wise scalar.

    """

    volume_integral: Callable[
        [pp.ad.Operator, Sequence[pp.Grid] | Sequence["pp.MortarGrid"], int],
        pp.ad.Operator,
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
    e_i: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]], int, int], pp.ad.SparseArray
    ]
    """Function that returns the unit vector in the i-th direction.

    Normally provided by a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

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
        e_i = self.e_i(subdomains, i=0, dim=9).T  # type: ignore[call-arg]
        # We assume isotropic permeability and extract xx component.
        isotropic_permeability = e_i @ self.permeability(subdomains)
        well_index = (
            pp.ad.Scalar(2 * np.pi)
            * projection.primary_to_mortar_avg
            @ (isotropic_permeability / (f_log(r_e / r_w) + skin_factor))
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
        r_e = Scalar(0.2) * pp.wrap_as_dense_ad_array(np.concatenate(h_list))
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


class ThermalConductivityLTE(SecondOrderTensorUtils):
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
            phi.value(self.equation_system)
        except KeyError:
            # We assume this means that the porosity includes a discretization matrix
            # for div_u which has not yet been computed.
            phi = self.reference_porosity(subdomains)
        if isinstance(phi, Scalar):
            size = sum([sd.num_cells for sd in subdomains])
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) + (
            Scalar(1) - phi
        ) * self.solid_thermal_conductivity(subdomains)
        return self.isotropic_second_order_tensor(subdomains, conductivity)

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
    need to be passed around.
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
    normal_thermal_conductivity: Callable[[list[pp.MortarGrid]], pp.ad.Scalar]
    """Conductivity on a mortar grid. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a subclass.

    """
    fourier_keyword: str
    """Keyword used to identify the Fourier flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
    bc_data_fourier_flux_key: str
    """See
    :attr:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance.
    bc_data_fourier_flux_key`.

    """
    bc_type_fourier_flux: Callable[[pp.Grid], pp.ad.Operator]
    """Function that returns the boundary condition type for the Fourier flux. Normally
    defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

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
    volume_integral: Callable[
        [pp.ad.Operator, Sequence[pp.Grid] | Sequence["pp.MortarGrid"], int],
        pp.ad.Operator,
    ]
    """Integration over cell volumes, implemented in
    :class:`pp.models.abstract_equations.BalanceEquation`.

    """
    nd: int
    """Number of spatial dimensions."""
    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

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

        boundary_operator_fourier = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=subdomains,
                dirichlet_operator=self.temperature,
                neumann_operator=self.fourier_flux,
                bc_type=self.bc_type_fourier_flux,
                name="bc_values_fourier",
            )
        )

        t: pp.ad.MixedDimensionalVariable = self.temperature(subdomains)
        temperature_trace = (
            discr.bound_pressure_cell @ t  # "pressure" is a legacy misnomer
            + discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_int
                @ self.interface_fourier_flux(interfaces)
            )
            + discr.bound_pressure_face @ boundary_operator_fourier
            + discr.bound_pressure_vector_source
            @ self.vector_source_fourier_flux(subdomains)
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

        if len(subdomains) == 0 or isinstance(subdomains[0], pp.BoundaryGrid):
            # Given Neumann data prescribed for Fourier flux on boundary.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_fourier_flux_key, domains=subdomains
            )

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.fourier_flux_discretization(
            subdomains
        )

        boundary_operator_fourier = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.temperature,
            neumann_operator=self.fourier_flux,
            bc_type=self.bc_type_fourier_flux,
            name="bc_values_fourier",
        )

        flux: pp.ad.Operator = (
            discr.flux @ self.temperature(subdomains)
            + discr.bound_flux
            @ (
                boundary_operator_fourier
                + projection.mortar_to_primary_int
                @ self.interface_fourier_flux(interfaces)
            )
            + discr.vector_source @ self.vector_source_fourier_flux(subdomains)
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
        temperature_h = projection.primary_to_mortar_avg @ self.temperature_trace(
            subdomains
        )
        temperature_l = projection.secondary_to_mortar_avg @ self.temperature(
            subdomains
        )
        eq = self.interface_fourier_flux(interfaces) - self.volume_integral(
            self.normal_thermal_conductivity(interfaces)
            * (
                normal_gradient * (temperature_h - temperature_l)
                + self.interface_vector_source_fourier_flux(interfaces)
            ),
            interfaces,
            1,
        )
        eq.set_name("interface_fourier_flux_equation")
        return eq

    def vector_source_fourier_flux(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]]
    ) -> pp.ad.Operator:
        """Vector source term. Zero for Fourier flux.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(0, "m*s^-2")  # TODO: Fix units
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        source = pp.wrap_as_dense_ad_array(val, size=size, name="zero_vector_source")
        return source

    def interface_vector_source_fourier_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Vector source term. Zero for Fourier flux.

        Corresponds to the inner product of a nd vector source with normal vectors.

        Parameters:
            interfaces: List of interface grids where the vector source is defined.

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(0, "m*s^-2")  # TODO: Fix units
        size = int(np.sum([g.num_cells for g in interfaces]))
        source = pp.wrap_as_dense_ad_array(val, size=size, name="zero_vector_source")
        return source

    def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Fourier flux discretization.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization object for the Fourier flux.

        """
        return pp.ad.MpfaAd(self.fourier_keyword, subdomains)


class FouriersLawAd(AdTpfaFlux, FouriersLaw):
    """Adaptive discretization of the Fourier flux from generic adaptive flux class."""

    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the thermal conductivity of a subdomain. Provided by a
    mixin class of instance
    :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a subclass
    thereof.

    """

    def fourier_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of Fourier's law.


        Parameters:
            domains: List of domains where the Fourier flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Operator representing face-wise Fourier flux.

        """
        flux = self.diffusive_flux(
            domains, self.temperature, self.thermal_conductivity, "fourier_flux"
        )
        return flux

    def temperature_trace(self, domains: list[pp.Grid]) -> pp.ad.Operator:
        """Temperature on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the temperature is defined.

        Returns:
            Temperature on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        temperature_trace = self.potential_trace(
            domains, self.temperature, self.thermal_conductivity, "fourier_flux"
        )
        temperature_trace.set_name("Differentiable temperature trace")
        return temperature_trace


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
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
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
        darcy_flux = self.darcy_flux(subdomains)
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        flux: pp.ad.Operator = (
            darcy_flux * (discr.upwind @ advected_entity)
            - discr.bound_transport_dir @ (darcy_flux * bc_values)
            # Advective flux coming from lower-dimensional subdomains
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
        """Gravity force term on either subdomains or interfaces.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector representing the gravity force [kg*s^-2*m^-2].

        """
        val = self.fluid.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        size = np.sum([g.num_cells for g in grids]).astype(int)
        gravity = pp.wrap_as_dense_ad_array(val, size=size, name="gravity")
        rho = getattr(self, material + "_density")(grids)

        # Gravity acts along the last coordinate direction (z in 3d, y in 2d). Ignore
        # type error, can't get mypy to understand keyword-only arguments in mixin.
        e_n = self.e_i(grids, i=self.nd - 1, dim=self.nd)  # type: ignore[call-arg]
        # e_n is a matrix, thus we need @ for it.
        gravity = Scalar(-1) * e_n @ (rho * gravity)
        gravity.set_name("gravity_force")
        return gravity


class ZeroGravityForce:
    """Zero gravity force.

    To be used in fluid fluxes and as body force in the force/momentum balance equation.

    """

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

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
        """Gravity force term on either subdomains or interfaces.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector representing the gravity force.

        """
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        return pp.wrap_as_dense_ad_array(0, size=size, name="zero_vector_source")


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
    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    bc_type_mechanics: Callable[[pp.Grid], pp.ad.Operator]

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

    def mechanical_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Raises:
            ValueError: If any grid is not of co-dimension 0.
            ValueError: If any the method is called with a mixture of subdomains and
                boundary grids.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.create_boundary_operator(
                name=self.stress_keyword, domains=domains  # type: ignore[call-arg]
            )

        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        for sd in domains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = self.stress_discretization(domains)
        # Fractures in the domain
        interfaces = self.subdomains_to_interfaces(domains, [1])

        # Boundary conditions on external boundaries
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics",
        )

        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        # The stress in the subdomanis is the sum of the stress in the subdomain,
        # the stress on the external boundaries, and the stress on the interfaces.
        # The latter is found by projecting the displacement on the interfaces to the
        # subdomains, and let these act as Dirichlet boundary conditions on the
        # subdomains.
        stress = (
            discr.stress @ self.displacement(domains)
            + discr.bound_stress @ boundary_operator
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

    def stress_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.BiotAd | pp.ad.MpsaAd:
        """Discretization of the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Discretization operator for the stress tensor.

        """
        return pp.ad.MpsaAd(self.stress_keyword, subdomains)


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

    def stress_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.BiotAd | pp.ad.MpsaAd:
        """Discretization of the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Discretization operator of the stress tensor.

        """
        return pp.ad.BiotAd(self.stress_keyword, subdomains)


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
        val = maximum_closure.value(self.equation_system)
        if np.any(val == 0):
            return Scalar(0)
        elif np.any(val < 0):
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
        ).value(self.equation_system)

        if np.any(val < 0):
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
    bc_type_mechanics: Callable[[pp.Grid], pp.ad.Operator]

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

    mechanical_stress: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

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
        one = pp.wrap_as_dense_ad_array(1, size=size, name="one")
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

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics",
        )

        # Compose operator.
        div_u_integrated = discr.div_u @ self.displacement(
            subdomains
        ) + discr.bound_div_u @ (
            boundary_operator
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

        beta_phi = (alpha - phi_ref) * beta_solid according to Coussy Eq. 4.44.

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
