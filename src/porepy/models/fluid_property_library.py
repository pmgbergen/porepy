""""Module containing some constant heuristic fluid property implemenations.

Most of the laws implemented here are meant for 1-phase, 1-component mixtures, using
some fluid component stored in the fluid's reference component.

Note:
    In order to override default implementations of fluid properties in
    :class:`~porepy.compositional.compositional_mixins.FluidMixin`, the classes hererin
    must be mixed into the model *before* the fluid mixin.

    .. code::python

        class MyModel(
            # ...
            ContitutiveLawFromFluidLibrary,
            # ...
            FluidMixin,
            # ...
        )

    E.g., Python must find :meth:`FluidDensityFromPressure.density_of_phase` before
    it finds the default
    :meth:`~porepy.compositional.compositional_mixins.FluidMixin.density_of_phase`.

Note:
    Different constitutive laws, based on analytical expressions, for fluid properties
    can be implemented in the same way as the classes here, see for instance the methods
    in :class:`~porepy.models.fluid_property_library.FluidDensityFromPressure`.

"""

from __future__ import annotations

from typing import Callable, Sequence, cast

import porepy as pp

from .protocol import PorePyModel

__all__ = [
    "FluidDensityFromPressure",
    "FluidDensityFromTemperature",
    "FluidDensityFromPressureAndTemperature",
    "FluidMobility",
    "ConstantViscosity",
    "ConstantFluidThermalConductivity",
    "FluidEnthalpyFromTemperature",
]

Scalar = pp.ad.Scalar
ExtendedDomainFunctionType = pp.ExtendedDomainFunctionType


class FluidDensityFromPressure(PorePyModel):
    """Fluid density as a function of pressure for a single-phase, single-component
    fluid."""

    def fluid_compressibility(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant compressibility [Pa^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The fluid constant wrapped as an AD Scalar.

        """
        return Scalar(
            self.fluid.reference_component.compressibility, "fluid_compressibility"
        )

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a density exponential law for the fluid's phase.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        The reference density and the compressibility are taken from the material
        constants of the reference component, while the reference pressure is accessible
        by mixin; a typical implementation will provide this in a variable class.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )
            rho_ = rho_ref * self.pressure_exponential(cast(list[pp.Grid], domains))
            rho_.set_name("fluid_density")
            return rho_

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

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Wrap compressibility from fluid class as matrix (left multiplication with dp).
        c = self.fluid_compressibility(subdomains)
        return exp(c * dp)


class FluidDensityFromTemperature(PorePyModel):
    """Fluid density as a function of temperature for a single-phase, single-component
    fluid."""

    def fluid_thermal_expansion(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant thermal expansion [K^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            The constant wrapped in as an AD scalar.

        """
        val = self.fluid.reference_component.thermal_expansion
        return Scalar(val, "fluid_thermal_expansion")

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Analogous to :meth:`FluidDensityFromPressure.density_of_phase`, but using
        temperature and the thermal expansion of the reference component.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ - c_T \\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )
            rho_ = rho_ref * self.temperature_exponential(cast(list[pp.Grid], domains))
            rho_.set_name("fluid_density")
            return rho_

        return rho

    def temperature_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of temperature.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        c = self.fluid_thermal_expansion(subdomains)
        return exp(Scalar(-1) * c * dtemp)


class FluidDensityFromPressureAndTemperature(
    FluidDensityFromPressure, FluidDensityFromTemperature
):
    """Fluid density which is a function of pressure and temperature, for a single-phase
    single-component fluid."""

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Returns a combination of the laws in the parent class methods:

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right)
            - c_T\\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )

            rho_ = (
                rho_ref
                * self.pressure_exponential(cast(list[pp.Grid], domains))
                * self.temperature_exponential(cast(list[pp.Grid], domains))
            )
            rho_.set_name("fluid_density_from_pressure_and_temperature")
            return rho_

        return rho


class FluidMobility(PorePyModel):
    """Class for fluid mobility and its discretization in single-phase flow problems."""

    mobility_keyword: str
    """Keyword for the discretization of the mobility. Normally provided by a mixin of
    instance :class:`~porepy.models.SolutionStrategy`.

    """

    def mobility(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Mobility of the fluid flux, given by the reciprocal of the fluid's reference
        phase viscosity.

        Parameters:
            subdomains: List of subdomains or boundaries.

        Returns:
            Operator representing the mobility [m * s * kg^-1].

        """
        return pp.ad.Scalar(1) / self.fluid.reference_phase.viscosity(subdomains)

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


class ConstantViscosity(PorePyModel):
    """Constant viscosity for a single-phase fluid."""

    def viscosity_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant viscosity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing representing the constant phase viscosity on some
            domains.

        """

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(self.fluid.reference_component.viscosity, "viscosity")

        return mu


class ConstantFluidThermalConductivity(PorePyModel):
    """Ïmplementation of a constant thermal conductivity for a single-phase fluid."""

    def thermal_conductivity_of_phase(
        self, phase: pp.Phase
    ) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant thermal conductivity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing the constant phase conductivity on some domains.

        """

        def kappa(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(
                self.fluid.reference_component.thermal_conductivity,
                "fluid_thermal_conductivity",
            )

        return kappa

    def normal_thermal_conductivity(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Scalar:
        """Constant normal thermal conductivity of the fluid given by the fluid
        constants stored in the fluid's reference component.

        Using the fluid value corresponds to assuming a fluid-filled fracture.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        # NOTE this is not really a fluid-related const. law, it is more related to
        # mixed-dimensional problems.
        val = self.fluid.reference_component.normal_thermal_conductivity
        return Scalar(val, "normal_thermal_conductivity")


class FluidEnthalpyFromTemperature(PorePyModel):
    """Implementation of a linearized fluid enthalpy :math:`c(T - T_{ref})` for a
    single-phase, single-component fluid.

    It uses the specific heat capacity of the fluid's reference component as :math:`c`,
    which is constant.

    """

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity  [J/kg/K]. The value
            is picked from the constants of the reference component.

        """
        return Scalar(
            self.fluid.reference_component.specific_heat_capacity,
            "fluid_specific_heat_capacity",
        )

    def specific_enthalpy_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a linear specific enthalpy for the fluid's phase.

        .. math::

            h = c \\left(T - T_0\\right)

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def h(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            c = self.fluid_specific_heat_capacity(cast(list[pp.Grid], domains))
            enthalpy = c * self.perturbation_from_reference(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h


class MobilityCF(FluidMobility):
    """Mixin class defining mobilities for the balance equations in the CF setting.

    Flux discretizations are handled by respective constitutive laws and the parent
    class.

    Provides various methods to assemble total, component and phase mobility, as well as
    fractional mobilities.

    Important:
        Mobility terms are designed to be representable also on boundary grids as user-
        given data.
        Values on the Neumann boundary (especially fractional mobilities) must be
        implemented by the user in :class:`BoundaryConditionsCF`.
        Those values are then consequently multiplied with boundary flux values in
        respective balance equations.

    """

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Provided by some mixin collecting constitutive laws for the solid skeleton."""

    def total_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Total mobility of the fluid mixture, by summing :meth:`phase_mobility`

        .. math::

                \sum_j \dfrac{\rho_j k_r(s_j)}{\mu_j},

        Used as a non-linear part of the diffusive tensor in the
        :class:`TotalMassBalanceEquation`.

        Parameters:
            domains: A list of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        name = "total_mobility"
        mobility = pp.ad.sum_operator_list(
            [self.phase_mobility(phase, domains) for phase in self.fluid.phases],
            name,
        )
        return mobility

    def phase_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the mobility of a phase :math:`j`

        .. math::

            \rho_j \dfrac{k_r(s_j)}{\mu_j}.

        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        name = f"phase_mobility_{phase.name}"
        # distinguish between single-phase case and multi-phase case: usage of rel-perm
        # Makes this class compatible with single-phase models, without requiring some
        # rel-perm mixin
        if self.fluid.num_phases > 1:
            mobility = (
                phase.density(domains)
                * self.relative_permeability(phase.saturation(domains))
                / phase.viscosity(domains)
            )
        else:
            assert phase == self.fluid.reference_phase
            mobility = phase.density(domains) / phase.viscosity(domains)
        mobility.set_name(name)
        return mobility

    def component_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear term in the advective flux in a component mass balance equation,
        or total mobility of a component.

        It is obtained by summing :meth:`phase_mobility` weighed with
        :attr:`~porepy.compositional.base.Phase.partial_fraction_of` the component,
        if the component is present in the phase.

        .. math::

                \sum_j x_{n, ij} \rho_j \dfrac{k_r(s_j)}{\mu_j},

        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        name = f"component_mobility_{component.name}"
        # NOTE this method is kept as general as possible when typing the signature.
        # But the default fluid of the model consists of FluidComponent, not Component.
        # Adding type:ignore for this reason.
        mobility = pp.ad.sum_operator_list(
            [
                phase.partial_fraction_of[component](domains)
                * self.phase_mobility(phase, domains)
                for phase in self.fluid.phases
                if component in phase  # type:ignore[operator]
            ],
            name,
        )
        return mobility

    def fractional_component_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the :meth:`component_mobility` divided by the
        :meth:`total_mobility`.

        To be used in component mass balance equations in a fractional flow set-up,
        where the total mobility is part of the non-linear diffusive tensor in the
        Darcy flux.

        I.e.,

        .. math::

            - \nabla \cdot \left(f_{\eta} D(p, Y) \nabla p\right),

        assuming the tensor :math:`D(p, Y)` contains the total mobility.


        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\eta}` in above expession in operator form.

        """
        name = f"fractional_component_mobility_{component.name}"
        frac_mob = self.component_mobility(component, domains) / self.total_mobility(
            domains
        )
        frac_mob.set_name(name)
        return frac_mob

    def fractional_phase_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the :meth:`phase_mobility` divided by the :meth:`total_mobility`.

        To be used in phase mass balance equations in a fractional flow set-up,
        where the total mobility is part of the non-linear diffusive tensor in the
        Darcy flux.

        I.e.,

        .. math::

            - \nabla \cdot \left(f_{\gamma} D(p, Y) \nabla p\right),

        assuming the tensor :math:`D(p, Y)` contains the total mobility.


        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\gamma}` in above expession in operator form.

        """
        name = f"fractional_phase_mobility_{phase.name}"
        frac_mob = self.phase_mobility(phase, domains) / self.total_mobility(domains)
        frac_mob.set_name(name)
        return frac_mob
