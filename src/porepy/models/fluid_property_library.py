""""Module containing some constant heuristic fluid property implemenations.

Most of the laws implemented here are meant for 1-phase, 1-component mixtures, using some
fluid constants stored in the fluid's reference component.

TODO: Various flid_* methods have tests implemented in
test_fluid_mass_balance, test_constitutive_laws, test_energy_balance
refactor tests since their name space changed to model.fluid

"""

from __future__ import annotations

from typing import Callable, Sequence, cast

import porepy as pp

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


class FluidDensityFromPressure:
    """Fluid density as a function of pressure."""

    fluid: pp.compositional.Fluid
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixin`."""
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def fluid_compressibility(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Fluid compressibility.

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The constant compressibility of the fluid [Pa^-1], represented as an Ad
            operator. The value is taken from the fluid constants.

        """
        return Scalar(
            self.fluid.reference_component.compressibility, "fluid_compressibility"
        )

    # NOTE replaces mixin fluid_density
    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Mixin method for :class:`~porepy.compositional.compositional_mixins.FluidMixin`
        to provide a density exponential law for the fluid's phase..

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        The reference density and the compressibility are taken from the material
        constants of the reference component, while the reference pressure is accessible by
        mixin; a typical implementation will provide this in a variable class.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of pressure [kg * m^-3].

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

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Wrap compressibility from fluid class as matrix (left multiplication with dp)
        c = self.fluid_compressibility(subdomains)
        return exp(c * dp)


class FluidDensityFromTemperature:
    """Fluid density as a function of temperature."""

    fluid: pp.compositional.Fluid
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Function that returns a perturbation from the reference state. Normally
    provided by a mixin of instance :class:`~porepy.models.VariableMixin`.

    """

    def fluid_thermal_expansion(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the thermal expansion  [1/K]. The value is picked from the
            fluid constants of the reference component.

        """
        val = self.fluid.reference_component.thermal_expansion
        return Scalar(val, "fluid_thermal_expansion")

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Analogous to :meth:`FluidDensityFromPressure.density_of_phase`, but using
        temperature and the thermal expansion of the reference component.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ - c_T \\left(T - T_0\\right) \\right]

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of temperature [kg * m^-3].

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

        # Reference variables are defined in a variables class which is assumed
        # to be available by mixin.
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        c = self.fluid_thermal_expansion(subdomains)
        return exp(Scalar(-1) * c * dtemp)


class FluidDensityFromPressureAndTemperature(
    FluidDensityFromPressure, FluidDensityFromTemperature
):
    """Fluid density which is a function of pressure and temperature."""

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Returns a combination of the laws in the parent class methods:

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right)
            - c_T\\left(T - T_0\\right) \\right]

          Parameters:
              subdomains: List of subdomain grids.

          Returns:
              Fluid density as a function of pressure and temperature.

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


class FluidMobility:  # TODO this does not belong here, requires a solution for 1 and m-phase
    """Class for fluid mobility and its discretization in single-phase flow problems."""

    mobility_keyword: str
    """Keyword for the discretization of the mobility. Normally provided by a mixin of
    instance :class:`~porepy.models.SolutionStrategy`.

    """

    fluid: pp.Fluid

    def mobility(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mobility of the fluid flux, given by the reciprocal of the fluid's reference phase
        viscosity.

        Parameters:
            subdomains: List of subdomains.

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


class ConstantViscosity:
    """Constant viscosity for a 1-phase, 1-component fluid."""

    fluid: pp.Fluid
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixin`."""

    # NOTE replaces mixin fluid_viscosity
    def viscosity_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Mixin method for :class:`~porepy.compositional.compositional_mixins.FluidMixin`
        to provide a constant viscosity for the fluid's phase."""

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(self.fluid.reference_component.viscosity, "viscosity")

        return mu


class ConstantFluidThermalConductivity:
    """Ïmplementation of a constant thermal conductivity for a 1-phase, 1-component fluid."""

    fluid: pp.Fluid
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixin`."""

    # NOTE replaces mixin fluid_thermal_conductivity
    def thermal_conductivity_of_phase(
        self, phase: pp.Phase
    ) -> ExtendedDomainFunctionType:
        """ "Mixin method for :class:`~porepy.compositional.compositional_mixins.FluidMixin`
        to provide a constant thermal conductivity for the fluid's phase."""

        def kappa(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(
                self.fluid.reference_component.thermal_conductivity,
                "fluid_thermal_conductivity",
            )

        return kappa

    def normal_thermal_conductivity(  # NOTE this is not really a fluid-related const. law
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Scalar:
        """Constant normal thermal conductivity of the fluid given by the fluid constants
        stored in the fluid's reference component.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        val = self.fluid.reference_component.normal_thermal_conductivity
        return Scalar(val, "normal_thermal_conductivity")


class FluidEnthalpyFromTemperature:
    """Implementation of a linearized fluid enthalpy :math:`c(T - T_{ref})` for a 1-phase,
    1-component fluid.

    It uses the specific heat capacity of the fluid's reference component as :math:`c`,
    which is constant.

    """

    fluid: pp.Fluid
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixin`."""

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """See :class:`~porepy.models.VariableMixin`."""

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity  [J/kg/K]. The value
            is picked from the fluid constants of the reference component.

        """
        return Scalar(
            self.fluid.reference_component.specific_heat_capacity,
            "fluid_specific_heat_capacity",
        )

    # NOTE replaces mixin fluid_enthalpy
    def specific_enthalpy_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """ "Mixin method for :class:`~porepy.compositional.compositional_mixins.FluidMixin`
        to provide a linear specific enthalpy for the fluid's phase."""

        def h(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            c = self.fluid_specific_heat_capacity(cast(list[pp.Grid], domains))
            enthalpy = c * self.perturbation_from_reference(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h
