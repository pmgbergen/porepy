"""Library of constitutive equations."""
from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

number = pp.number
Scalar = pp.ad.Scalar


def ad_wrapper(
    vals: Union[number, np.ndarray],
    array: bool,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[pp.ad.Array, pp.ad.Matrix]:
    """Create ad array or diagonal matrix.

    Utility method.

    Parameters:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        array: Whether to return a matrix or vector.
        size: Size of the array or matrix. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:
        Values wrapped as an Ad object.

    """
    if type(vals) is not np.ndarray:
        assert size is not None, "Size must be set if vals is not an array"
        value_array: np.ndarray = vals * np.ones(size)
    else:
        value_array = vals

    if array:
        return pp.ad.Array(value_array, name)
    else:
        if size is None:
            size = value_array.size
        matrix = sps.diags(vals, shape=(size, size))
        return pp.ad.Matrix(matrix, name)


class DimensionReduction:
    """Apertures and specific volumes."""

    nd: int
    """Ambient dimension of the problem.

    This attribute must be provided by a mixin class. Normally, it is set by
    :class:`porepy.models.geometry.ModelGeometry.

    """
    # TODO: Figure out how to document these class instances that must be decleared for
    # mypy to accept the use of mixin.

    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        """FIXME: Decide on how to treat interfaces."""
        aperture = np.ones(grid.num_cells)
        if grid.dim < self.nd:
            aperture *= 0.1
        return aperture

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Ad operator representing the aperture for each cell in each subdomain.

        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # The implementation here is not perfect, but it seems to be what is needed
        # to make the Ad framework happy: Build the global array by looping over
        # subdomains and add the local contributions.
        # Note that the aperture is an array (in the Ad sense) not a matrix, thus there
        # is no risk of the number of columns being wrong (as there would be if we
        # were to wrap the aperture as an Ad matrix).

        for i, sd in enumerate(subdomains):
            # First make the local aperture array.
            a_loc = ad_wrapper(self.grid_aperture(sd), array=True)
            # Expand to a global array.
            a_glob = projection.cell_prolongation([sd]) * a_loc
            if i == 0:
                apertures = a_glob
            else:
                apertures += a_glob
        apertures.set_name("aperture")

        return apertures

    def specific_volume(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Specific volume [m^(nd-d)]

        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Specific volume for each cell.
        """
        # Compute specific volume as the cross-sectional area/volume
        # of the cell, i.e. raise to the power nd-dim
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        v: pp.ad.Operator = None  # type: ignore

        # Loop over dimensions, and add the contribution from each subdomain within
        # that dimension.
        # TODO: Will looping over subdomains shuffle the order of the cells? If so,
        # this will be a problem.
        for dim in range(self.nd + 1):
            sd_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(sd_dim) == 0:
                continue
            a_loc = self.aperture(sd_dim)
            v_loc = a_loc ** Scalar(self.nd - dim)
            v_glob = projection.cell_prolongation(sd_dim) * v_loc
            if v is None:
                v = v_glob
            else:
                v = v + v_glob

        # If we found no subdomains, we may have a problem. Possibly we should just
        # return a void Operator.
        assert v is not None, "No subdomains found"

        v.set_name("specific_volume")

        return v


class DisplacementJumpAperture:
    """Fracture aperture from displacement jump."""

    nd: int
    """Ambient dimension of the problem.

    This attribute must be provided by a mixin class. Normally, it is set by
    :class:`porepy.models.geometry.ModelGeometry.

    """

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]
    """Method to map from subdomains to the adjacent interfaces.

    This method must be provided by a mixin class. Normally, it is set by
    :class:`porepy.models.geometry.ModelGeometry.

    """

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Method to map from interfaces to the adjacent subdomains.

    This method must be provided by a mixin class. Normally, it is set by
    :class:`porepy.models.geometry.ModelGeometry.

    """

    normal_component: Callable[[list[pp.Grid]], pp.ad.Operator]

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]

    reference_gap: Callable[[list[pp.Grid]], pp.ad.Operator]

    mdg: pp.MixedDimensionalGrid

    system_manager: pp.ad.EquationSystem
    """EquationSystem object for the current model.

    This attribute must be provided by a mixin class. Normally, it is set by the
    relevant solution strategy class.

    """

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

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
        one = ad_wrapper(1, True, size=num_cells_nd_subdomains, name="one")
        # Start with nd, where aperture is one.
        apertures = projection.cell_prolongation(nd_subdomains) * one

        # TODO: Same comments as in DimensionReduction.specific_volume() regarding
        # the order of the subdomains.
        for dim in range(self.nd):
            subdomains_of_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(subdomains_of_dim) == 0:
                continue
            if dim == self.nd - 1:
                # Fractures. Get displacement jump
                normal_jump = self.normal_component(
                    subdomains_of_dim
                ) * self.displacement_jump(subdomains_of_dim)
                # The jump should be bounded below by gap function. This is not
                # guaranteed in the non-converged state. As this (especially
                # non-positive values) may give significant trouble in the aperture.
                # Insert safeguard:
                # EK: Safeguard in which sense?
                f_max = pp.ad.Function(pp.ad.maximum, "maximum_function")

                g_ref = self.reference_gap(subdomains_of_dim)
                apertures_of_dim = f_max(normal_jump, g_ref)
                apertures_of_dim.set_name("aperture_maximum_function")
                apertures = (
                    apertures
                    + projection.cell_prolongation(subdomains_of_dim) * apertures_of_dim
                )
            else:
                # Intersection aperture is average of apertures of intersecting
                # fractures.
                interfaces_dim = self.subdomains_to_interfaces(subdomains_of_dim)
                parent_fractures = self.interfaces_to_subdomains(interfaces_dim)
                mortar_projection = pp.ad.MortarProjections(
                    self.mdg, subdomains_of_dim + parent_fractures, interfaces_dim
                )
                parent_apertures = self.aperture(parent_fractures)
                parent_to_intersection = (
                    mortar_projection.mortar_to_secondary_avg
                    * mortar_projection.primary_to_mortar_avg
                )
                average_weights = parent_to_intersection.evaluate(
                    self.system_manager
                ).sum(axis=1)
                average_mat = ad_wrapper(average_weights, False, name="average_weights")
                apertures_of_dim = (
                    average_mat * parent_to_intersection * parent_apertures
                )
                apertures += (
                    projection.cell_prolongation(subdomains_of_dim) * apertures_of_dim
                )

        return apertures


class ConstantFluidDensity:
    """Constant fluid density."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.

    This attribute must be provided by a mixin class. Normally, it is set by the
    XXX

    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Fluid density [kg/m^3].

        Parameters:
            subdomains: List of subdomain grids.Note used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            Operator for fluid density, represented as an Ad operator. The value is
            picked from the fluid constants.

        """
        return Scalar(self.fluid.density(), "fluid_density")


class FluidDensityFromPressure:
    """Fluid density as a function of pressure."""

    fluid: pp.FluidConstants

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]

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
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Wrap compressibility from fluid class as matrix (left multiplication with dp)
        c = self.fluid_compressibility(subdomains)

        # The reference density is taken from the fluid constants..
        rho_ref = Scalar(self.fluid.density(), "reference_fluid_density")
        rho = rho_ref * exp(c * dp)
        rho.set_name("fluid_density")
        return rho


class FluidDensityFromPressureAndTemperature(FluidDensityFromPressure):
    """Fluid density which is a function of pressure and temperature."""

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of pressure and temperature.

        .. math::
              \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]\\left(-(T - T_0)\\right)/c_T \\right)

          with :math:`\\rho_0` the reference density, :math:`p_0` the reference pressure,
          :math:`c_p` the compressibility, :math:`p` the pressure, :math:`T` the temperature,
          :math:`T_0` the reference temperature, and :math:`c_T` the thermal expansion
          coefficient.

          The reference density, the compressibility and the thermal expansion coefficient
          are all taken from the fluid constants, while the reference pressure and
          temperature are accessible by mixin; a typical implementation will provide this
          in a variable class.

          Parameters:
              subdomains: List of subdomain grids.

          Returns:
              Fluid density as a function of pressure.

        """
        # Get the pressure part of the density function from the super class.
        rho = super().fluid_density(subdomains)

        # Next the temperature part. Note the minus sign, which makes density decrease
        # with increasing temperature.
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        rho = rho * exp(-dtemp / self.fluid.thermal_expansion())
        return rho


class ConstantViscosity:
    """Constant viscosity."""

    fluid: pp.FluidConstants

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

    def permeability(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Permeability [m^2].

        The permeability is quantity which enters the discretized equations in a form
        that cannot be differentiated by Ad (this is at least true for a subset of the
        relevant discretizations). For this reason, the permeability is not returned as
        an Ad operator, but as a numpy array, to be wrapped as a SecondOrderTensor and
        passed as a discretization parameter.

        Parameters:
            subdomain: Subdomain where the permeability is defined.

        Returns:
            Cell-wise permeability tensor. The value is picked from the solid constants.

        """
        assert len(subdomains) == 1, "Only one subdomain is allowed."
        size = subdomains[0].num_cells
        return self.solid.permeability() * np.ones(size)

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


class DarcysLaw:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around.
    """

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    mdg: pp.MixedDimensionalGrid

    bc_values_darcy_flux: Callable[[list[pp.Grid]], np.ndarray]

    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]

    fluid: pp.FluidConstants

    nd: int

    darcy_keyword: str

    basis: Callable[[Sequence[pp.GridLike]], np.ndarray]

    internal_boundary_normal_to_outwards: Callable[
        [list[pp.MortarGrid], int], pp.ad.Matrix
    ]

    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str], pp.ad.Operator]

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: pp.ad.MpfaAd = self.darcy_flux_discretization(subdomains)
        p: pp.ad.MixedDimensionalVariable = self.pressure(subdomains)
        pressure_trace = (
            discr.bound_pressure_cell * p
            + discr.bound_pressure_face
            * (projection.mortar_to_primary_int * self.interface_darcy_flux(interfaces))
            + discr.bound_pressure_face * self.bc_values_darcy_flux(subdomains)
            + discr.vector_source * self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace

    def darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Darcy flux.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: pp.ad.MpfaAd = self.darcy_flux_discretization(subdomains)
        flux: pp.ad.Operator = (
            discr.flux * self.pressure(subdomains)
            + discr.bound_flux
            * (
                self.bc_values_darcy_flux(subdomains)
                + projection.mortar_to_primary_int
                * self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source * self.vector_source(subdomains, material="fluid")
        )
        flux.set_name("Darcy_flux")
        return flux

    def interface_darcy_flux_equation(self, interfaces: list[pp.MortarGrid]):
        """Darcy flux on interfaces.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(interfaces, "cell_volumes")
        # Project the two pressures to the interface and multiply with the normal
        # diffusivity
        eq = self.interface_darcy_flux(
            interfaces
        ) - cell_volumes * self.normal_permeability(interfaces) * (
            projection.primary_to_mortar_avg * self.pressure_trace(subdomains)
            - projection.secondary_to_mortar_avg * self.pressure(subdomains)
            # TODO: Reintroduce the below term?
            # + self.interface_vector_source(interfaces, material="fluid")
        )
        eq.set_name("interface_darcy_flux_equation")
        return eq

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """
        Note:
            The ad.Discretizations may be purged altogether. Their current function is
            very similar to the ad.Geometry in that both basically wrap numpy/scipy
            arrays in ad arrays and collect them in a block matrix. This similarity
            could possibly be exploited. Revisit at some point.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]], material: str
    ) -> pp.ad.Operator:
        """Vector source term.

        Represents gravity effects. EK: Let's discuss how to name/think about this term.
        Note that it appears slightly differently in a flux and a force/momentum
        balance.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
            defined. material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(0, "m*s^-2")
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        source = ad_wrapper(val, array=True, size=size, name="zero_vector_source")
        return source

    def interface_vector_source(
        self, interfaces: list[pp.MortarGrid], material: str
    ) -> pp.ad.Operator:
        """Interface vector source term.

        The term is the product of unit normals and vector source values. Normalization
        is needed to balance the integration done in the interface flux law.

        Parameters:
            interfaces: List of interfaces where the vector source is defined.

        Returns:
            Face-wise vector source term.
        """

        # Account for sign of boundary face normals
        normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
        return normals * self.vector_source(interfaces, material=material)


class ThermalConductivityLTE:
    """Thermal conductivity in the local thermal equilibrium approximation."""

    fluid: pp.FluidConstants
    solid: pp.SolidConstants

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    equation_system: pp.ad.EquationSystem

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid thermal conductivity [W / (m K)]

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar
            representing the constant thermal conductivity of the fluid.

        """
        return Scalar(self.fluid.thermal_conductivity(), "fluid_thermal_conductivity")

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid thermal conductivity [W / (m K)].

        The

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar, wrapped as
            an Ad operator, representing the constant thermal conductivity of the fluid.

        """
        return Scalar(self.solid.thermal_conductivity(), "solid_thermal_conductivity")

    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Thermal conductivity [m^2].

        The thermal conductivity is computed as the porosity-weighted average of the
        fluid and solid thermal conductivities. In this implementation, both are
        considered constants, however, if the porosity changes with time, the weighting
        factor will also change.

        The thermal conductivity is quantity which enters the discretized equations in a
        form that cannot be differentiated by Ad (this is at least true for a subset of
        the relevant discretizations). For this reason, the thermal conductivity is not
        returned as an Ad operator, but as a numpy array, to be wrapped as a
        SecondOrderTensor and passed as a discretization parameter.

        Parameters:
            subdomain: Subdomain where the thurmal conductivity is defined.

        Returns:
            Cell-wise conducivity tensor.

        """
        assert len(subdomains) == 1, "Only one subdomain is allowed."
        size = subdomains[0].num_cells
        phi = self.porosity(subdomains)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) + (
            Scalar(1) - phi
        ) * self.solid_thermal_conductivity(subdomains)
        vals = conductivity.evaluate(self.equation_system)

        # In case vals is proper operator, not a combination of Scalars and Arrays,
        # get the values.
        if isinstance(vals, pp.ad.Ad_array):
            vals = vals.val

        return vals * np.ones(size)

    def normal_thermal_conductivity(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Normal thermal conductivity.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        # Porosity weighting inherited from lower-dimensional neighbor
        subdomains = self.interfaces_to_subdomains(interfaces)

        phi = self.porosity(subdomains)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) + (
            Scalar(1) - phi
        ) * self.solid_thermal_conductivity(subdomains)
        return conductivity


class FouriersLaw:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around. Also, gravity effects are
    not included, as opposed to the Darcy flux (see that class).
    """

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    mdg: pp.MixedDimensionalGrid

    interface_fourier_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    bc_values_fourier_flux: Callable[[list[pp.Grid]], np.ndarray]

    normal_thermal_conductivity: Callable[[list[pp.MortarGrid]], pp.ad.Operator]

    fourier_keyword: str

    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str], pp.ad.Operator]

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
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr = self.fourier_flux_discretization(subdomains)
        t: pp.ad.MixedDimensionalVariable = self.temperature(subdomains)
        temperature_trace = (
            discr.bound_pressure_cell * t  # "pressure" is a legacy misnomer
            + discr.bound_pressure_face
            * (
                projection.mortar_to_primary_int
                * self.interface_fourier_flux(interfaces)
            )
            + discr.bound_pressure_face * self.bc_values_fourier_flux(subdomains)
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
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr = self.fourier_flux_discretization(subdomains)

        # As opposed to darcy_flux in :class:`DarcyFluxFV`, the gravity term is not
        # included here.
        flux: pp.ad.Operator = discr.flux * self.temperature(
            subdomains
        ) + discr.bound_flux * (
            self.bc_values_fourier_flux(subdomains)
            + projection.mortar_to_primary_int * self.interface_fourier_flux(interfaces)
        )
        flux.set_name("Darcy_flux")
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

        cell_volumes = self.wrap_grid_attribute(interfaces, "cell_volumes")
        # Project the two pressures to the interface and multiply with the normal
        # diffusivity
        eq = self.interface_fourier_flux(
            interfaces
        ) - cell_volumes * self.normal_thermal_conductivity(interfaces) * (
            projection.primary_to_mortar_avg * self.temperature_trace(subdomains)
            - projection.secondary_to_mortar_avg * self.temperature(subdomains)
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

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]

    mdg: pp.MixedDimensionalGrid

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    def advective_flux(
        self,
        subdomains: list[pp.Grid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindAd,
        bc_values: pp.ad.Operator,
        interface_flux: Callable[[list[pp.MortarGrid]], pp.ad.Operator],
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
            interface_flux: Interface flux operator/variable.

        Returns:
            Operator representing the advective flux.

        """
        darcy_flux = self.darcy_flux(subdomains)
        interfaces = self.subdomains_to_interfaces(subdomains)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        flux: pp.ad.Operator = (
            darcy_flux * (discr.upwind * advected_entity)
            - discr.bound_transport_dir * darcy_flux * bc_values
            # Advective flux coming from lower-dimensional subdomains
            - discr.bound_transport_neu
            * (
                mortar_projection.mortar_to_primary_int * interface_flux(interfaces)
                + bc_values
            )
        )
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
        # IMPLEMENTATION NOTE: If we ever implement other discretizations than upwind,
        # we may need to change the below definition.
        interface_flux: pp.ad.Operator = self.interface_darcy_flux(interfaces) * (
            discr.upwind_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * advected_entity
            + discr.upwind_secondary
            * mortar_projection.secondary_to_mortar_avg
            * advected_entity
        )
        return interface_flux


class SpecificHeatCapacities:

    fluid: pp.FluidConstants
    solid: pp.SolidConstants

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
    """Class for representing the ethalpy, computed from the perturbation from a reference
    temperature.
    """

    temperature: pp.ad.Operator
    reference_temperature: float

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]

    def fluid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid enthalpy [J / m^3].

        TODO: Check units here, and in solid_specific_heat_capacity.

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
        # TODO: Is there not one more c-scaling here?
        return c * enthalpy

    def solid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid enthalpy [J/kg/K].

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
        # TODO: Is there a scaling factor c too much here?
        return c * enthalpy


class GravityForce:
    """Gravity force.

    The gravity force is defined as the product of the fluid density and the gravity
    vector:

    .. math::
        g = -\\rho \\mathbf{g}= -\\rho \\begin{bmatrix} 0 \\\\ 0 \\\\ G \\end{bmatrix}

    where :math:`\\rho` is the fluid density, and :math:`G` is the magnitude of the
    gravity acceleration.

    To be used in fluid fluxes and as body force in the force/momentum balance equation.

    TODO: Decide whether to use this or zero as default for Darcy fluxes.
    """

    fluid: pp.FluidConstants

    e_i: Callable[[Union[list[pp.Grid], list[pp.MortarGrid]], int, int], pp.ad.Operator]

    nd: int

    def gravity_force(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]], material: str
    ) -> pp.ad.Operator:
        """Vector source term.

        TODO: Is it deliberate that we have Union[list[pp.Grid], list[pp.MortarGrid]] here,
        but list[pp.GridLike] in other places in this module?

        Represents gravity effects. EK: Let's discuss how to name/think about this term.
        Note that it appears slightly differently in a flux and a force/momentum
        balance.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
            defined. material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator.

        """
        val = self.fluid.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        size = np.sum([g.num_cells for g in grids])
        gravity = ad_wrapper(val, array=True, size=size, name="gravity")
        rho = getattr(self, material + "_density")(grids)
        # Gravity acts along the last coordinate direction (z in 3d, y in 2d)
        e_n = self.e_i(grids, i=self.nd - 1, dim=self.nd)
        source = (-1) * rho * e_n * gravity
        source.set_name("gravity_force")
        return source


class LinearElasticMechanicalStress:
    """Linear elastic stress tensor.

    To be used in mechanical problems, e.g. force balance.

    """

    nd: int

    stress_keyword: str

    bc_values_mechanics: Callable[[list[pp.Grid]], pp.ad.Operator]

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    interface_displacement: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    mdg: pp.MixedDimensionalGrid

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
            assert sd.dim == self.nd
        discr = pp.ad.MpsaAd(self.stress_keyword, subdomains)
        interfaces = self.subdomains_to_interfaces(subdomains)
        bc = self.bc_values_mechanics(subdomains)
        proj = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=self.nd)
        stress = (
            discr.stress * self.displacement(subdomains)
            + discr.bound_stress * bc
            + discr.bound_stress
            * proj.mortar_to_primary_avg
            * self.interface_displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress


class PressureStress:
    """Stress tensor from pressure.

    To be used in poromechanical problems.

    .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

    """

    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure variable. Should be defined in the class inheriting from this mixin."""
    reference_pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference pressure. Should be defined in the class inheriting from this mixin."""

    def pressure_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure contribution to stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Pressure stress operator.

        """
        for sd in subdomains:
            assert sd.dim == self.nd
        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)
        stress: pp.ad.Operator = (
            discr.grad_p * self.pressure(subdomains)
            # The reference pressure is only defined on sd_primary, thus there is no
            # need for a subdomain projection.
            - discr.grad_p * self.reference_pressure(subdomains)
        )
        stress.set_name("pressure_stress")
        return stress

    def interface_pressure_stress(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Pressure contribution to stress tensor on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Pressure stress operator.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        # Outwards normals. Scaled by face areas for FV formulation.
        outwards_normal = self.outwards_internal_boundary_normals(
            interfaces, dim=self.nd, unitary=False
        )
        # Expands from cell-wise scalar to vector. Equivalent to the :math:`\mathbf{I}p`
        # operation.
        scalar_to_nd = sum([e_i for e_i in self.basis(interfaces)])
        stress = (
            outwards_normal
            * scalar_to_nd
            * mortar_projection.secondary_to_mortar_avg
            * self.pressure(subdomains)
        )
        stress.set_name("interface_pressure_stress")
        return stress


class ConstantSolidDensity:

    solid: pp.SolidConstants

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
    parameters (bulk_modulus, lame_mu, poisson_ratio). The latter are computed from the former.
    Also provides a method for computing the stiffness matrix as a FourthOrderTensor.
    """

    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus operator. The value is picked from the solid constants.

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
        val = (self.solid.lame_lambda() + 2 * self.solid.shear_modulus()) / 3
        return Scalar(val, "bulk_modulus")

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        TODO: It makes sense to return a fourth order tensor here. Does that mean we
        should return a second order tensor for the permeability and conductivity
        tensors (right now, we give numpy arrays).

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.
        """
        lmbda = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        mu = self.solid.shear_modulus() * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu, lmbda)


class FracturedSolid:
    """Fractured rock properties.

    This class is intended for use with fracture deformation models.
    """

    nd: int

    tangential_component: Callable[[list[pp.Grid]], pp.ad.Operator]

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]

    solid: pp.SolidConstants

    def gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture gap [m].

        Parameters:
            subdomains: List of subdomains where the gap is defined.

        Returns:
            Cell-wise fracture gap operator.

        """
        angle: pp.ad.Operator = self.dilation_angle(subdomains)
        f_norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        f_tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")
        shear_dilation: pp.ad.Operator = f_tan(angle) * f_norm(
            self.tangential_component(subdomains) * self.displacement_jump(subdomains)
        )

        gap = self.reference_gap(subdomains) + shear_dilation
        gap.set_name("gap_with_shear_dilation")
        return gap

    def reference_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference gap [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise reference gap operator [m].

        """
        return Scalar(self.solid.gap(), "reference_gap")

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

    def dilation_angle(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Dilation angle [rad].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise dilation angle operator [rad].

        """
        return Scalar(self.solid.dilation_angle(), "dilation_angle")


class FrictionBound:
    """Friction bound for fracture deformation.

    This class is intended for use with fracture deformation models.
    """

    normal_component: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator extracting normal component of vector. Should be defined in class combined with
    this mixin."""
    traction: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Traction variable. Should be defined in class combined with from this mixin."""
    friction_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction coefficient. Should be defined in class combined with this mixin."""

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]

    def friction_bound(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction bound [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [Pa].

        """
        t_n: pp.ad.Operator = self.normal_component(subdomains) * self.contact_traction(
            subdomains
        )
        bound: pp.ad.Operator = (-1) * self.friction_coefficient(subdomains) * t_n
        bound.set_name("friction_bound")
        return bound


class BiotCoefficient:
    """Biot coefficient."""

    solid: pp.SolidConstants

    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Biot coefficient.

        Parameters:
            subdomains: List of subdomains where the Biot coefficient is defined.

        Returns:
            Biot coefficient operator.

        """
        return Scalar(self.solid.biot_coefficient(), "biot_coefficient")


class ConstantPorosity:

    solid: pp.SolidConstants

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.porosity(), "porosity")


class PoroMechanicsPorosity:

    solid: pp.SolidConstants

    nd: int

    mdg: pp.MixedDimensionalGrid

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    bulk_modulus: Callable[[list[pp.Grid]], pp.ad.Operator]

    stress_keyword: str
    darcy_keyword: str

    bc_values_mechanics: Callable[[list[pp.Grid]], pp.ad.Operator]

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    interface_displacement: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.porosity(), "reference_porosity")

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
        one = ad_wrapper(1, True, size=size, name="one")
        rho_nd = projection.cell_prolongation(subdomains_nd) * self.matrix_porosity(
            subdomains_nd
        )
        rho_lower = projection.cell_prolongation(subdomains_lower) * one
        rho = rho_nd + rho_lower
        rho.set_name("porosity")
        return rho

    def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].
        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")
        phi_ref = self.reference_porosity(subdomains)
        dp = self.perturbation_from_reference("pressure", subdomains)
        alpha = self.biot_coefficient(subdomains)
        bulk = self.bulk_modulus(subdomains)

        # 1/N as defined in Coussy, 2004, https://doi.org/10.1002/0470092718.
        n_inv = (alpha - phi_ref) * (1 - alpha) / bulk

        phi = phi_ref + n_inv * dp + alpha * self.displacement_divergence(subdomains)
        return phi

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
            Divergence operator accounting from contributions from interior of the domain
            and from internal and external boundaries.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Displacement divergence only defined in nd.")

        # Obtain neighbouring interfaces
        interfaces = self.subdomains_to_interfaces(subdomains)
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
        div_u = discr.div_u * self.displacement(subdomains) + discr.bound_div_u * (
            bc_values
            + sd_projection.face_restriction(subdomains)
            * mortar_projection.mortar_to_primary_avg
            * self.interface_displacement(interfaces)
        )
        div_u.set_name("div_u")
        return div_u


def boundary_values_from_operator(
    operator: Callable[[list[pp.Grid]], pp.ad.Operator],
    subdomain: pp.Grid,
    faces: np.ndarray,
    model,
) -> np.ndarray:
    """Extract Dirichlet values from an operator.

    Parameters:
        interior_operator: Operator to extract Dirichlet values from.
        subdomain: Subdomain where the operator is defined.
        faces: Faces where Dirichlet values are to be extracted.
        equation_system: Equation system facilitating evaluation of the operator.

    Returns:
        Dirichlet values (sd.num_faces,).

    """

    # boundary = model.mdg.subdomain_to_boundary_grid(subdomain)
    # # Extract Dirichlet values from the operator
    # vals = np.zeros(subdomain.num_faces)

    # boundary_vals = operator([boundary]).evaluate(model.equation_system)
    # # Unlike Operators, wrapped constants (Scalar, Array) do not have val attribute.
    # if hasattr(boundary_vals, "val"):
    #     boundary_vals = boundary_vals.val

    # if isinstance(boundary_vals, Number):
    #     # Scalar value, simple assignment
    #     vals[faces] = boundary_vals
    # else:
    #     # Array value, assumed to be cell-wise
    #     assert isinstance(boundary_vals, np.ndarray)
    #     face_vals = boundary.projection * boundary_vals
    #     assert face_vals.shape == (subdomain.num_faces,)
    #     vals[faces] = face_vals[faces]

    # FIXME: Revisit BoundaryGrid. This is a hack to get the code to run.
    #
    # TODO: See previous comment! Fare, fare, krigsmann! (Norwegian for "farewell, my
    # friend"). EK, IS. Which tags have I forgotten to mark this as important?
    vals = np.zeros(subdomain.num_faces)
    return vals
