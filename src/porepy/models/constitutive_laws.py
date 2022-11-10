"""Library of constitutive equations.

Consists of three types of classes
    Units for scaling
    Materials for constants (Solid, Fluid). These are envisioned as components/attributes of
        model classes, see fluid_mass_balance.SolutionStrategyIncompressibleFlow.set_materials
    Constitutive equations on ad form. This will eventually become the most important part,
        from which a model is assembled based on mixin/inheritance.

See usage_example.py
"""
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

number = pp.number


def ad_wrapper(
    vals: Union[number, np.ndarray],
    array: bool,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[pp.ad.Array, pp.ad.Matrix]:
    """Create ad array or diagonal matrix.

    Utility method.

    Args:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        array: Whether to return a matrix or vector.
        size: Size of the array or matrix. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:

    """
    if type(vals) is not np.ndarray:
        assert size is not None, "Size must be set if vals is not an array"
        vals: np.ndarray = vals * np.ones(size)

    if array:
        return pp.ad.Array(vals, name)
    else:
        if size is None:
            size = vals.size
        matrix = sps.diags(vals, shape=(size, size))
        return pp.ad.Matrix(matrix, name)


"""
Below are some examples of Mixins which are low-level components of a set of constitutive
equations. First three different versions of fluid density, then one for permeability.

FIXME: Choose whether materials or the classes below are responsible for expanding to number
of cells. Probably safest to do that below in case of issues with vector values or cell/face
ambiguity.
"""


class DimensionReduction:
    """Apertures and specific volumes."""

    def grid_aperture(self, grid: pp.Grid):
        """FIXME: Decide on how to treat interfaces."""
        aperture = np.ones(grid.num_cells)
        if grid.dim < self.nd:
            aperture *= 0.1
        return aperture

    def aperture(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        for i, sd in enumerate(subdomains):
            a_loc = ad_wrapper(self.grid_aperture(sd), array=False)
            a_glob = (
                projection.cell_prolongation([sd])
                * a_loc
                * projection.cell_restriction([sd])
            )
            if i == 0:
                apertures = a_glob
            else:
                apertures = a_glob
        apertures.set_name("aperture")
        return apertures

    def specific_volume(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        # Get aperture
        a: pp.ad.Operator = self.aperture(subdomains)
        # Compute specific volume as the cross-sectional area/volume
        # of the cell, i.e. raise to the power nd-dim
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        v: pp.ad.Operator = None
        for dim in range(self.nd + 1):
            sd_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(sd_dim) == 0:
                continue
            a_loc = self.aperture(sd_dim)
            v_loc = a_loc ** pp.ad.Scalar(self.nd + 1 - dim)
            v_glob = (
                projection.cell_prolongation(sd_dim)
                * v_loc
                * projection.cell_restriction(sd_dim)
            )
            if v is None:
                v = v_glob
            else:
                v += v_glob
        v.set_name("specific_volume")
        sz = sum([sd.num_cells for sd in subdomains])
        vals = np.ones(sz)
        vol = ad_wrapper(vals, array=False)
        return vol


class ConstantFluidDensity:

    """Underforstått:

    def __init__(self, fluid: UnitFluid):
        self.fluid = ...

    eller tilsvarende. Se SolutionStrategiesIncompressibleFlow.
    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        val = self.fluid.density(subdomains)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, False, num_cells, "fluid_density")


class FluidDensityFromPressure:
    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        # Reference variables are defined in Variables class.
        dp = self.pressure(subdomains) - self.reference_pressure(self, subdomains)
        # I suggest using the fluid's constant density as the reference value. While not
        # explicit, this saves us from defining reference properties i hytt og pine.
        # We could consider letting this class inherit from ConstantDensity (and call super
        # to obtain reference value), but I don't see what the benefit would be.
        rho_ref = self.fluid.density(subdomains)
        rho = rho_ref * exp(dp / self.fluid.compressibility(subdomains))
        return rho


class FluidDensityFromPressureAndTemperature(FluidDensityFromPressure):
    """Extend previous case"""

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rho = super().fluid_density(subdomains)
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.temperature(subdomains) - self.reference_temperature(
            self, subdomains
        )
        rho = rho * exp(-dtemp / self.fluid.thermal_expansion(subdomains))
        return rho


class ConstantViscosity:
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return ad_wrapper(
            self.fluid.viscosity(subdomains), array=False, name="viscosity"
        )


class DarcyFlux:
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around.
    """

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Args:
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

        Args:
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

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Discretization:
        """
        .. note::
            The ad.Discretizations may be purged altogether. Their current function is very
            similar to the ad.Geometry in that both basically wrap numpy/scipy arrays in ad
            arrays and collect them in a block matrix. This similarity could possibly be
            exploited. Revisit at some point.

        Args:
            subdomains:

        Returns:

        """
        return pp.ad.MpfaAd(self.darcy_discretization_parameter_key, subdomains)

    def vector_source(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]], material: str
    ) -> pp.ad.Operator:
        """Vector source term.

        Represents gravity effects. EK: Let's discuss how to name/think about this term. Note
        that it appears slightly differently in a flux and a force/momentum balance.

        Args:
            grids: List of subdomain or interface grids where the vector source is defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator
        """
        vals: np.ndarray = self.fluid.convert_and_expand(
            0, "m*s^-2", grids, dim=self.nd
        )
        source: pp.ad.Matrix = ad_wrapper(vals, array=False, name="zero_vector_source")
        return source

    def interface_vector_source(self, interfaces):
        """Interface vector source term.

        The term is the product of unit normals and vector source values. Normalization is
        needed to balance the integration done in the interface flux law.

        Args:
            interfaces: List of interfaces where the vector source is defined.

        Returns:
            Face-wise vector source term.
        """
        geometry = pp.ad.Geometry(interfaces, nd=self.nd, matrix_names=["cell_volumes"])
        # Expand cell volumes to nd
        # Fixme: Do we need right multiplication with transpose as well?
        cell_volumes_inv = geometry.scalar_to_nd_cell * geometry.cell_volumes ** (-1)
        # Account for sign of boundary face normals
        flip = self.internal_boundary_normal_to_outwards(interfaces)
        unit_outwards_normals = (
            flip * cell_volumes_inv * geometry.wrap_grid_vector("face_normals")
        )
        return unit_outwards_normals * self.vector_source(interfaces)


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

        Args:
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
            (discr.upwind * advected_entity) * darcy_flux
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

        Args:
            interfaces: List of interface grids.

        Returns:
            Operator representing the advective flux on the interfaces.
        """
        # If no interfaces are given, make sure to proceed with a non-empty subdomain list.
        if not interfaces:
            subdomains = self.mdg.subdomains(dim=self.nd)
        else:
            subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        trace = pp.ad.Trace(subdomains)
        # Project the two advected entities to the interface and multiply with upstream weights
        # and the interface Darcy flux.
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


class GravityForce:
    """Gravity force.

    The gravity force is defined as the product of the fluid density and the gravity vector:

    .. math::
        g = -\\rho \\mathbf{g}= -\\rho \\begin{bmatrix} 0 \\\\ 0 \\\\ G \\end{bmatrix}

    where :math:`\\rho` is the fluid density, and :math:`G` is the magnitude of the gravity
    acceleration.

    To be used in fluid fluxes and as body force in the force/momentum balance equation.

    TODO: Decide whether to use this or zero as default for Darcy fluxes.
    """

    def gravity_force(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]], material: str
    ) -> pp.ad.Operator:
        """Vector source term.

        Represents gravity effects. EK: Let's discuss how to name/think about this term. Note
        that it appears slightly differently in a flux and a force/momentum balance.

        Args:
            grids: List of subdomain or interface grids where the vector source is defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector source term operator
        """
        # Geometry needed for basis vector.
        geometry = pp.ad.Geometry(grids, nd=self.nd)
        val: np.ndarray = self.fluid.convert_and_expand(
            pp.GRAVITY_ACCELERATION, "m*s^-2", grids
        )
        gravity: pp.ad.Matrix = ad_wrapper(val, array=False, name="gravity")
        rho = getattr(self, material + "_density")(grids)
        source = (-1) * geometry.e_i(i=self.nd - 1) * gravity * rho
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
        discr = pp.ad.MpsaAd(subdomains, nd=self.nd)
        interfaces = self.subdomains_to_interfaces(subdomains)
        bc = self.bc_values_mechanics(subdomains)
        proj = pp.ad.MortarProjections(subdomains, interfaces, nd=self.nd)
        stress = (
            discr.stress * self.displacement(subdomains)
            + discr.bound_stress * bc
            + discr.bound_stress
            * self.subdomain_projections(self.nd).face_restriction(subdomains)
            * proj.mortar_to_primary_avg
            * self.displacement(interfaces)
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

        Args:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Pressure stress operator.
        """
        for sd in subdomains:
            assert sd.dim == self.nd
        discr = pp.ad.BiotAd(self.mechanics_parameter_key, subdomains)
        stress: pp.ad.Operator = (
            discr.grad_p * self.pressure(subdomains)
            # The reference pressure is only defined on sd_primary, thus there is no need
            # for a subdomain projection.
            - discr.grad_p * self.reference_pressure(subdomains)
        )
        stress.set_name("pressure_stress")
        return stress


class LinearElasticSolid(LinearElasticMechanicalStress):
    """Linear elastic properties of a solid.

    Includes "primary" stiffness parameters (lame_lambda, shear_modulus) and "secondary"
    parameters (bulk_modulus, lame_mu, poisson_ratio). The latter are computed from the former.
    Also provides a method for computing the stiffness matrix as a FourthOrderTensor.
    """

    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus operator [Pa].
        """
        return ad_wrapper(
            self.solid.shear_modulus(subdomains), False, name="shear_modulus"
        )

    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter operator [Pa].
        """
        return ad_wrapper(self.solid.lame_lambda(subdomains), False, name="lame_lambda")

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa].

        Args:
            subdomains: List of subdomains where the Young's modulus is defined.

        Returns:
            Cell-wise Young's modulus in Pascal.
        """
        val = (
            self.solid.shear_modulus(subdomains)
            * (
                3 * self.solid.lame_lambda(subdomains)
                + 2 * self.solid.shear_modulus(subdomains)
            )
            / (
                self.solid.lame_lambda(subdomains)
                + self.solid.shear_modulus(subdomains)
            )
        )
        return ad_wrapper(val, False, name="youngs_modulus")

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Args:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.
        """
        lmbda = self.solid.lame_lambda([subdomain])
        mu = self.solid.shear_modulus([subdomain])
        return pp.FourthOrderTensor(mu, lmbda)


class FracturedSolid:
    """Fractured rock properties.

    This class is intended for use with fracture deformation models.
    """

    def reference_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference gap [m].

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise reference gap operator [m].
        """
        return ad_wrapper(self.solid.gap(subdomains), True, name="reference_gap")

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient.

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient operator.
        """
        return ad_wrapper(
            self.solid.friction_coefficient(subdomains),
            False,
            name="friction_coefficient",
        )


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

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [Pa].

        """
        t_n: pp.ad.Operator = self.normal_component(subdomains) * self.traction(
            subdomains
        )
        bound: pp.ad.Operator = (-1) * self.friction_coefficient(subdomains) * t_n
        bound.set_name("friction_bound")
        return bound


class ConstantPorousMedium:
    def permeability(self, subdomain: pp.Grid) -> pp.SecondOrderTensor:
        # This will be set as before (pp.PARAMETERS) since it is a discretization parameter
        # Hence not list[subdomains]
        # perm = pp.SecondOrderTensor(
        #    self.solid.permeability(subdomain) * np.ones(subdomain.num_cells)
        # )
        return self.solid.permeability(subdomain)

    def normal_permeability(self, subdomain: pp.Grid) -> pp.ad.Operator:
        return self.solid.normal_permeability(subdomain)

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        poro = self.solid.porosity(subdomains)

        return ad_wrapper(poro, False, name="porosity")


class ConstantSinglePhaseFluid(ConstantFluidDensity, ConstantViscosity):
    """Collection of constant fluid properties.

    This class is intended for use in single-phase flow models.
    """

    ...
