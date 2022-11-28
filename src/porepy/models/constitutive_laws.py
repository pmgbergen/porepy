"""Library of constitutive equations."""
from functools import partial
from typing import Callable, Optional, Union

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

    """
    if type(vals) is not np.ndarray:
        assert size is not None, "Size must be set if vals is not an array"
        value_array: np.ndarray = vals * np.ones(size)

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

    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        """FIXME: Decide on how to treat interfaces."""
        aperture = np.ones(grid.num_cells)
        if grid.dim < self.nd:
            aperture *= 0.1
        return aperture

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        for i, sd in enumerate(subdomains):
            a_loc = ad_wrapper(self.grid_aperture(sd), array=True)
            a_glob = projection.cell_prolongation([sd]) * a_loc
            if i == 0:
                apertures = a_glob
            else:
                apertures = apertures + a_glob
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
        v: pp.ad.Operator = None
        for dim in range(self.nd + 1):
            sd_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(sd_dim) == 0:
                continue
            a_loc = self.aperture(sd_dim)
            v_loc = a_loc ** Scalar(self.nd + 1 - dim)
            v_glob = projection.cell_prolongation(sd_dim) * v_loc
            if v is None:
                v = v_glob
            else:
                v = v + v_glob
        v.set_name("specific_volume")

        return v


class DisplacementJumpAperture:
    """Fracture aperture from displacement jump."""

    nd: int

    subdomains_to_interfaces: Callable[[list[pp.Grid]], list[pp.MortarGrid]]

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]

    normal_component: Callable[[list[pp.Grid]], pp.ad.Operator]

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]

    reference_gap: Callable[[list[pp.Grid]], pp.ad.Operator]

    mdg: pp.MixedDimensionalGrid

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Operator representing apertures.

        """
        # For now, assume no intersections
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        nd_subdomains = [sd for sd in subdomains if sd.dim == self.nd]
        size = sum([sd.num_cells for sd in nd_subdomains])
        one = ad_wrapper(1, True, size=size, name="one")
        # Start with nd, where aperture is one.
        apertures = projection.cell_prolongation(nd_subdomains) * one

        for dim in range(self.nd):
            subdomains_of_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(subdomains_of_dim) == 0:
                continue
            if dim == self.nd - 1:
                # Fractures. Get displacement jump
                normal_jump = self.normal_component(
                    subdomains_of_dim
                ) * self.displacement_jump(subdomains_of_dim)
                # The jump should be bounded below by gap function. This is not guaranteed
                # in the non-converged state. As this (especially non-positive values)
                # may give significant trouble in the aperture. Insert safeguard:
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

    """Underforstått:

    def __init__(self, fluid: UnitFluid):
        self.fluid = ...

    eller tilsvarende. Se SolutionStrategiesSinglePhaseFlow.
    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Fluid density [kg/m^3].

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Operator for fluid density.

        """
        return Scalar(self.fluid.density(), "fluid_density")


class FluidDensityFromPressure:
    """Fluid density as a function of pressure."""

    def fluid_compressibility(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        return Scalar(self.fluid.compressibility(), "fluid_compressibility")

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of pressure.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        with :math:`\\rho_0` the reference density, :math:`p_0` the reference pressure,
        :math:`c_p` the compressibility and :math:`p` the pressure.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        # Reference variables are defined in Variables class.
        dp = self.perturbation_from_reference("pressure", subdomains)
        # Wrap compressibility from fluid class as matrix (left multiplication with dp)
        c = self.fluid_compressibility(subdomains)
        # I suggest using the fluid's constant density as the reference value. While not
        # explicit, this saves us from defining reference properties i hytt og pine. We
        # could consider letting this class inherit from ConstantDensity (and call super
        # to obtain reference value), but I don't see what the benefit would be.
        rho_ref = Scalar(self.fluid.density(), "reference_fluid_density")
        rho = rho_ref * exp(c * dp)
        return rho


class FluidDensityFromPressureAndTemperature(FluidDensityFromPressure):
    """Extend previous case"""

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid density as a function of pressure and temperature."""
        rho = super().fluid_density(subdomains)
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        rho = rho * exp(-dtemp / self.fluid.thermal_expansion())
        return rho


class ConstantViscosity:
    def fluid_viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.fluid.viscosity(), "viscosity")


class ConstantPermeability:
    def permeability(self, subdomains: list[pp.Grid]) -> pp.SecondOrderTensor:
        """Permeability [m^2].

        This will be set as before (pp.PARAMETERS) since it

        Parameters:
            subdomain: Subdomain where the permeability is defined.
                Permeability is a discretization parameter and is assigned to individual
                subdomain data dictionaries. Hence, the list will usually contain only
                one element.

        Returns:
            Cell-wise permeability tensor.
        """
        assert len(subdomains) == 1, "Only one subdomain is allowed."
        size = subdomains[0].num_cells
        return self.solid.permeability() * np.ones(size)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Normal permeability.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Scalar normal permeability on the interfaces.

        """
        return Scalar(self.solid.normal_permeability())


class DarcysLawFV:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around.
    """

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr = self.darcy_flux_discretization(subdomains)
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
        discr: pp.ad.Discretization = self.darcy_flux_discretization(subdomains)
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
            # + self.interface_vector_source(interfaces, material="fluid")
        )
        eq.set_name("interface_darcy_flux_equation")
        return eq

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Discretization:
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
            Cell-wise nd-vector source term operator
        """
        val: np.ndarray = self.fluid.convert_units(0, "m*s^-2")
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        source: pp.ad.Array = ad_wrapper(
            val, array=True, size=size, name="zero_vector_source"
        )
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
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=self.nd
        )
        # Expand cell volumes to nd
        # Fixme: Do we need right multiplication with transpose as well?
        cell_volumes = self.wrap_grid_attribute(interfaces, "cell_volumes")
        face_normals = self.wrap_grid_attribute(subdomains, "face_normals")

        # Expand cell volumes to nd
        scalar_to_nd = sum(self.basis(subdomains))
        cell_volumes_inv = scalar_to_nd * cell_volumes ** (-1)
        # Account for sign of boundary face normals
        flip = self.internal_boundary_normal_to_outwards(interfaces)
        unit_outwards_normals = (
            flip * cell_volumes_inv * projection.primary_to_mortar_avg * face_normals
        )
        return unit_outwards_normals * self.vector_source(interfaces, material=material)


class ThermalConductivityLTE:
    """Thermal conductivity in the local thermal equilibrium approximation."""

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid thermal conductivity.

        SI units: W / (m K)

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of fluid.

        """
        return Scalar(self.fluid.thermal_conductivity(), "fluid_thermal_conductivity")

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid thermal conductivity.

        SI units: W / (m K)

        Parameters:
            grids: List of subdomains where the thermal conductivity is defined.

        Returns:
            Thermal conductivity of solid.

        """
        return Scalar(self.solid.thermal_conductivity(), "solid_thermal_conductivity")

    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.SecondOrderTensor:
        """Permeability [m^2].

        This will be set as before (pp.PARAMETERS) since it is used in the
        discretization of the thermal equation by either the mpfa or the tpfa method.

        Parameters:
            subdomain: Subdomain where the permeability is defined.
                Permeability is a discretization parameter and is assigned to individual
                subdomain data dictionaries. Hence, the list is assumed to contain only
                one element.

        Returns:
            Cell-wise permeability tensor.

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
        if hasattr(vals, "val"):
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


class FouriersLawFV:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around. Also, gravity effects are
    not included, as opposed to the Darcy flux (see that class).
    """

    def temperature_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Temperature on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the temperature is defined.

        Returns:
            Temperature on the subdomain boundaries. Parsing the operator will return a
            face-wise array
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
        """Fourier flux.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Face-wise Fourier flux in cubic meters per second.
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: pp.ad.Discretization = self.fourier_flux_discretization(subdomains)

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

    def interface_fourier_flux_equation(self, interfaces: list[pp.MortarGrid]):
        """Fourier flux on interfaces.

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

    def fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Discretization:
        """Fourier flux discretization.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization of the Fourier flux.

        """
        return pp.ad.MpfaAd(self.fourier_keyword, subdomains)


class AdvectiveFlux:
    def advective_flux(
        self,
        subdomains: list[pp.Grid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.Discretization,
        bc_values: pp.ad.Operator,
        interface_flux: Callable[[list[pp.MortarGrid]], pp.ad.Operator],
    ) -> pp.ad.Operator:
        """Advective flux.

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
        discr: pp.ad.Discretization,
    ) -> pp.ad.Operator:
        """Advective flux on interfaces.

        Parameters:
            interfaces: List of interface grids.

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
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * advected_entity
            + discr.upwind_secondary
            * mortar_projection.secondary_to_mortar_avg
            * advected_entity
        )
        return interface_flux


class SpecificHeatCapacities:
    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid specific heat capacity.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid specific heat capacity.

        """
        return Scalar(
            self.fluid.specific_heat_capacity(), "fluid_specific_heat_capacity"
        )

    def solid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid specific heat capacity.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid specific heat capacity.

        """
        return Scalar(
            self.solid.specific_heat_capacity(), "solid_specific_heat_capacity"
        )


class EnthalpyFromTemperature(SpecificHeatCapacities):
    temperature: pp.ad.Operator
    reference_temperature: float

    def fluid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid enthalpy.

        SI units: J / m^3.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid enthalpy.

        """
        c = self.fluid_specific_heat_capacity(subdomains)
        enthalpy = c * self.perturbation_from_reference("temperature", subdomains)
        enthalpy.set_name("fluid_enthalpy")
        return c * enthalpy

    def solid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid enthalpy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid enthalpy.

        """
        c = self.solid_specific_heat_capacity(subdomains)
        enthalpy = c * self.perturbation_from_reference("temperature", subdomains)
        enthalpy.set_name("solid_enthalpy")
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

    def gravity_force(
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
            Cell-wise nd-vector source term operator
        """
        val: np.ndarray = self.fluid.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        size = np.sum([g.num_cells for g in grids])
        gravity: pp.ad.Array = ad_wrapper(val, array=True, size=size, name="gravity")
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

    def mechanical_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Linear elastic mechanical stress."""
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


# Foregriper litt her for å illustrere utvidelse til poromekanikk.
# Det blir
# PoroConstit(LinearElasticSolid, PressureStress):
#    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#        return self.pressure_stress(subdomains) + self.mechanical_stress(subdomains)
class PressureStress:
    """Stress tensor from pressure.

    To be used in poromechanical problems.
    """

    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure variable. Should be defined in the class inheriting from this mixin."""
    reference_pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference pressure. Should be defined in the class inheriting from this mixin."""

    def pressure_stress(self, subdomains):
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
            # The reference pressure is only defined on sd_primary, thus there is no need
            # for a subdomain projection.
            - discr.grad_p * self.reference_pressure(subdomains)
        )
        stress.set_name("pressure_stress")
        return stress


class ConstantSolidDensity:
    def solid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
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
            Cell-wise shear modulus operator [Pa].
        """
        return Scalar(self.solid.shear_modulus(), "shear_modulus")

    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter operator [Pa].
        """
        return Scalar(self.solid.lame_lambda(), "lame_lambda")

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the Young's modulus is defined.

        Returns:
            Cell-wise Young's modulus in Pascal.
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

    def gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture gap [m].

        Parameters:
            subdomains: List of subdomains where the gap is defined.

        Returns:
            Cell-wise fracture gap operator [m].
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

    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Biot coefficient.

        Parameters:
            subdomains: List of subdomains where the Biot coefficient is defined.

        Returns:
            Biot coefficient operator.
        """
        return Scalar(self.solid.biot_coefficient(), "biot_coefficient")


class ConstantPorosity:
    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.porosity(), "porosity")


class PoroMechanicsPorosity:
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
