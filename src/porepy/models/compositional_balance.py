"""Module defining basic equatios for fluid flow with multiple componens/species."""
from __future__ import annotations

import logging
import time
import warnings
from functools import partial
from typing import Callable, Literal, Optional, Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc
from porepy.grids.mortar_grid import MortarGrid
from porepy.numerics.ad.operators import Operator

from . import energy_balance as energy
from . import fluid_mass_balance as mass
from . import mass_and_energy_balance as mass_energy

logger = logging.getLogger(__name__)


class MobilityCF:
    """Mixin class defining mobilities for the balance equations in the CF setting, and
    which discretization is to be used for the magnitude of terms in the compositional
    flow and transport model.

    The flexibility is required due to the varying mathematical nature of the pressure
    equation, energy balance and transport equations.

    They also need all separate instances of the discretization objects to avoid
    false storage access.

    Discretizations handled by this class include those for the non-linear weights in
    various flux terms.

    Flux discretizations are handled by respective constitutive laws.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Provided by :class:`PermeabilityCF`."""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Provided by :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.
    """

    mobility_keyword: str
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`."""
    enthalpy_keyword: str
    """Provided by :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`.
    """

    has_extended_fractions: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    def total_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Returns the non-linear weight in the advective flux in the pressure
        equation.

        Parameters:
            domains: A list of subdomains or boundary grids.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j k_r(s_j)}{\mu_j},

            which is the advected mass in the total mass balance equation.

        """
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains)
                / phase.viscosity(domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            "total_mobility",
        )
        return mobility

    def total_mobility_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the total fluid mobility in the total mass balance on the
        subdomains.

        Important:
            Upwinding in the pressure equation is inconsistent. This method is left for
            convenience, but should not be used.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_total_mobility_discretization(
        self, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Analogous to :meth:`total_mobility_discretization` on interfaces.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)

    def enthalpy_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Scalar weight for the enthalpy flux:

        .. math::

            \sum_j h_j \rho_j \dfrac{k_r(s_j)}{\mu_j}

        Parameters:
            domains: A sequence of subdomains or boundary grids

        Returns:
            Above operator, assuming all terms in above expression are defined on
            subdomains and boundaries.

        """
        weight = pp.ad.sum_operator_list(
            [
                phase.enthalpy(domains)
                * phase.density(domains)
                / phase.viscosity(domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            "enthalpy-mobility",
        )
        return weight

    def fractional_enthalpy_mobility(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Returns the enthalpy mobility divided by the total mobility.

        To be used in a fractional flow set-up, where the total mobility is part of the
        non-linear diffusive tensor in the pressure equation (for consistent
        discretization).

        """
        op = self.enthalpy_mobility(domains) / self.total_mobility(domains)
        op.set_name("fractional-enthalpy-mobility")
        return op

    def enthalpy_mobility_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the enthalpy flux in the energy
        balance on the subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the enthalpy mobility.

        """
        return pp.ad.UpwindAd(self.enthalpy_keyword, subdomains)

    def interface_enthalpy_mobility_discretization(
        self, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Analogous to :meth:`enthalpy_mobility_discretization` on interfaces.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface enthalpy mobility.

        """
        return pp.ad.UpwindCouplingAd(self.enthalpy_keyword, interfaces)

    def component_mobility(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear weight in the advective flux of a component mass balance equation

        Parameters:
            domains: A sequence of subdomains or boundary grids.

        Raises:
            NotImplementedError: If the mixture mixin did not create extended fractions
                of components in phases.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j}{\mu_j} x_{n, ij} k_r(s_j),

            which is the advected mass in the total mass balance equation.
            :math:`x_{n, ij}` denotes the normalized fraction of component i in phase j.

        """

        if not self.has_extended_fractions:
            raise NotImplementedError(
                "Mobilities of a component not available:"
                + " No compositional fractions created."
            )
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains)
                / phase.viscosity(domains)
                * phase.normalized_fraction_of[component](domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            f"{component.name}-mobility",
        )
        return mobility

    def fractional_component_mobility(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Analogously to :meth:`fractional_enthalpy_mobility`, returns the component
        mobility divided by the total mobility for a fractional flow formulation."""
        op = self.component_mobility(component, domains) / self.total_mobility(domains)
        op.set_name(f"{component.name}-fractional-mobility")
        return op

    def component_mobility_discretization(
        self, component: ppc.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the advective flux in the
        component mass balance equations.

        The discretization is should be the same as for
        :meth:`total_mobility_discretization` since it is based on the same flux.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Discretization of the mobility in the component's mass balance.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_component_mobility_discretization(
        self, component: ppc.Component, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the advective flux in the
        component mass balance equations on interfaces.

        Parameters:
            component: A transportable fluid component in the mixture.
            interfaces: List of interfaces.

        Returns:
            Discretization of the mobility in the component's mass balance.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)


class ThermalConductivityCF(pp.constitutive_laws.ThermalConductivityLTE):
    """Fourier's law in the compositional flow setting.

    Atop the parent class methods, it provides means to compute the conductivity of a
    fluid mixture.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assembles the fluid conductivity as a sum of phase conductivities weighed
        with saturations.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar defined in
            each cell.

        """
        conductivity = pp.ad.sum_operator_list(
            [
                phase.conductivity(subdomains) * phase.saturation(subdomains)
                for phase in self.fluid_mixture.phases
            ],
            "fluid_thermal_conductivity",
        )
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
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # this is a constitutive law based on Banshoya 2023
        normal_conductivity = projection.secondary_to_mortar_avg @ (
            self.fluid_thermal_conductivity(subdomains)
        )
        normal_conductivity.set_name("norma_thermal_conductivity")
        return normal_conductivity


class PermeabilityCF(pp.constitutive_laws.ConstantPermeability):
    """A constitutive law for permeabilities and mobilities, based on the fractional
    flow formulation."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`."""

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    def normal_permeability(self, interfaces: list[MortarGrid]) -> Operator:
        """A constitutive law returning the normal permeability as the product of
        total mobility and the permeability on the lower-dimensional subdomain.

        Parameters:
            interfaces: A list of mortar grids.

        Returns:
            The product of total mobility and permeability of the lower-dimensional.

        """

        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        normal_permeability = projection.secondary_to_mortar_avg @ (
            self.total_mobility(subdomains) * self.permeability(subdomains)
        )
        normal_permeability.set_name("norma_permeability")
        return normal_permeability

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        Parameters:
            saturation: Operator representing the saturation of a phase.

        Returns:
            The base class method implements the quadratic law ``saturation ** 2``.

        """
        return saturation**2


class TotalMassBalanceEquation(pp.BalanceEquation):
    """Mixed-dimensional balance of total mass in a fluid mixture.

    Also referred to as *pressure equation*.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    Note:
        This balance equation assumes that the total advected mass is part of the
        diffusive second-order tensor in the non-linear MPFA discretization.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.ConstantPorosity`."""

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    interface_darcy_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""
    well_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.PiecmannWellFlux`."""

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        # TODO: If wells are integrated for nd=2 models, consider refactoring sorting of
        # interfaces into method returning either "normal" or well interfaces.
        codim_2_interfaces = self.mdg.interfaces(codim=2)
        sd_eq = self.mass_balance_equation(subdomains)
        intf_eq = self.interface_darcy_flux_equation(codim_1_interfaces)
        well_eq = self.well_flux_equation(codim_2_interfaces)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_eq, codim_1_interfaces, {"cells": 1})
        self.equation_system.set_equation(well_eq, codim_2_interfaces, {"cells": 1})

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mass balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.fluid_mass(subdomains)
        flux = self.fluid_flux(subdomains)
        source = self.fluid_source(subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("pressure_equation")
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Analogous to
        :meth:`~porepy.models.fluid_mass_balance.MassBalanceEquations.fluid_mass`
        with the only difference that :attr:`~porepy.composite.base.Mixture.density`,
        which by procedure is defined using the pressure, temperature and fractions on
        all subdomains."""
        mass_density = self.fluid_mixture.density(subdomains) * self.porosity(
            subdomains
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name("total_fluid_mass")
        return mass

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """In the basic setting, the total fluid flux is the Darcy flux, where the
        total mobility is included in a diffuse second-order tensor."""
        return self.darcy_flux(domains)

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Analogous to
        :meth:`~porepy.models.fluid_mass_balance.MassBalanceEquations.fluid_source`
        with the only difference that we do not use the interface fluid flux, but the
        interface flux variable.

        Also we do not use the well fluid flux, but the well flux variable.

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        well_interfaces = self.subdomains_to_interfaces(subdomains, [2])
        well_subdomains = self.interfaces_to_subdomains(well_interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        well_projection = pp.ad.MortarProjections(
            self.mdg, well_subdomains, well_interfaces
        )
        subdomain_projection = pp.ad.SubdomainProjections(self.mdg.subdomains())
        source = projection.mortar_to_secondary_int @ self.interface_darcy_flux(
            interfaces
        )
        source.set_name("interface_fluid_flux_source")
        well_fluxes = well_projection.mortar_to_secondary_int @ self.well_flux(
            well_interfaces
        ) - well_projection.mortar_to_primary_int @ self.well_flux(interfaces)
        well_fluxes.set_name("well_fluid_flux_source")
        source += subdomain_projection.cell_restriction(subdomains) @ (
            subdomain_projection.cell_prolongation(well_subdomains) @ well_fluxes
        )
        source.set_name("total_fluid_source")
        return source


class TotalEnergyBalanceEquation_h(energy.EnergyBalanceEquations):
    """Mixed-dimensional balance of total energy in a fluid mixture, formulated with an
    independent enthalpy variable.

    Balance equation for all subdomains and advective and diffusive fluxes
    (Fourier flux) internally and on all interfaces of codimension one and advection on
    interfaces of codimension two (well-fracture intersections).

    Note:
        Since enthalpy is an independent variable, models using this balance need
        something more:

        If temperature is also an independent variable, it needs a constitutive law
        to close the system.

        If temperature is not an independent variable,
        :meth:`porepy.models.energy_balance.VariablesEnergyBalance.temperature` needs to
        be overwritten to provide a secondary expression.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`VariablesCompositionalFlow`."""

    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a
    subclass thereof."""
    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Provided by :class:`ConstitutiveLawsCompositionalFlow`."""

    advective_flux: Callable[
        [
            list[pp.Grid],
            pp.ad.Operator,
            pp.ad.UpwindAd,
            pp.ad.Operator,
            Callable[[list[pp.MortarGrid]], pp.ad.Operator],
        ],
        pp.ad.Operator,
    ]
    """Provided by :class:`~porepy.models.constitutive_laws.AdvectiveFlux`."""
    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`"""
    fractional_enthalpy_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`"""

    enthalpy_mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Provided by :class:`MobilityCF`."""
    interface_enthalpy_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    bc_type_enthalpy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by
    :class:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance`."""

    bc_data_enthalpy_flux_key: str
    """Provided by
    :class:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance`."""

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites the parent method to use the fluix mixture density and the primary
        unknown enthalpy."""
        energy = (
            self.fluid_mixture.density(subdomains) * self.enthalpy(subdomains)
            - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_mixture_internal_energy")
        return energy

    def enthalpy_flux(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """The child method modifies the non-linear weight in the enthalpy flux
        (enthalpy mobility) and the custom discretization for it."""

        if len(subdomains) == 0 or all(
            [isinstance(g, pp.BoundaryGrid) for g in subdomains]
        ):
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_enthalpy_flux_key,
                domains=subdomains,
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in subdomains]):
            raise ValueError(
                "Domains must consist entirely of subdomains for the enthalpy flux."
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        subdomains = cast(list[pp.Grid], subdomains)

        boundary_operator_enthalpy = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=subdomains,
                dirichlet_operator=self.fractional_enthalpy_mobility,
                neumann_operator=self.enthalpy_flux,
                bc_type=self.bc_type_enthalpy_flux,
                name="bc_values_enthalpy",
            )
        )

        discr = self.enthalpy_mobility_discretization(subdomains)
        weight = self.fractional_enthalpy_mobility(subdomains)
        flux = self.advective_flux(
            subdomains,
            weight,
            discr,
            boundary_operator_enthalpy,
            self.interface_enthalpy_flux,
        )
        flux.set_name("enthalpy_flux")
        return flux

    def interface_enthalpy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Uses the enthalpy mobility and the custom discretization implemented by
        :class:`MobilityCF`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_enthalpy_mobility_discretization(interfaces)
        weight = self.fractional_enthalpy_mobility(subdomains)
        flux = self.interface_advective_flux(
            interfaces,
            weight,
            discr,
        )

        eq = self.interface_enthalpy_flux(interfaces) - flux
        eq.set_name("interface_enthalpy_flux_equation")
        return eq

    def well_enthalpy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Uses the enthalpy mobility and the custom discretization implemented by
        :class:`MobilityCF`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_enthalpy_mobility_discretization(interfaces)
        weight = self.fractional_enthalpy_mobility(subdomains)
        flux = self.well_advective_flux(
            interfaces,
            weight,
            discr,
        )

        eq = self.well_enthalpy_flux(interfaces) - flux
        eq.set_name("well_enthalpy_flux_equation")
        return eq


class ComponentMassBalanceEquations(mass.MassBalanceEquations):
    """Mixed-dimensional balance of mass in a fluid mixture for present components.

    The total mass balance is the sum of all component mass balances.

    Since feed fractions per independent component are unknowns, the model requires
    additional transport equations to close the system.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    Note:
        This is a sophisticated version of
        :class:`~porepy.models.fluid_mass_balance.MassBalanceEquation` where the
        non-linear weights in the flux term stem from a multiphase, multicomponent
        mixture.

        There is room for unification and recycling of code.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcyFlux`."""

    fractional_component_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`."""

    component_mobility_discretization: Callable[
        [ppc.Component, list[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`MobilityCF`."""
    interface_component_mobility_discretization: Callable[
        [ppc.Component, list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    bc_data_component_flux_key: Callable[[ppc.Component], str]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""
    bc_type_component_flux: Callable[[ppc.Component, pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        subdomains = self.mdg.subdomains()

        for component in self.fluid_mixture.components:
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue

            sd_eq = self.mass_balance_equation_for_component(component, subdomains)
            self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def mass_balance_equation_for_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Mass balance equation for subdomains for a given component.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.fluid_mass_for_component(component, subdomains)
        flux = self.fluid_flux_for_component(component, subdomains)
        source = self.fluid_source_of_component(component, subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(f"mass_balance_equation_{component.name}")
        return eq

    def fluid_mass_for_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """The cell-wise fluid mass for a given component.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Operator representing the cell-wise component mass.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid_mixture.density(subdomains)
            * component.fraction(subdomains)
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name(f"component_mass_{component.name}")
        return mass

    def fluid_flux_for_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A modified fluid flux, where the advected entity (component mobility)
        accounts for all phases in the mixture for a given component.

        See :meth:`mobility_of_component`.

        It also accounts for custom choices of mobility discretizations (see
        :class:`MobilityCF`).

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_component_flux_key(component), domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        discr = self.component_mobility_discretization(domains)
        weight = self.fractional_component_mobility(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        weight_dirichlet_bc = partial(self.fractional_component_mobility, component)
        weight_dirichlet_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            weight_dirichlet_bc,
        )
        fluid_flux_neumann_bc = partial(self.fractional_component_mobility, component)
        fluid_flux_neumann_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            fluid_flux_neumann_bc,
        )
        bc_type = partial(self.bc_type_component_flux, component)
        bc_type = cast(
            Callable[[pp.Grid], pp.BoundaryCondition],
            bc_type,
        )
        interface_flux = partial(self.interface_flux_for_component, component)
        interface_flux = cast(Callable[[list[pp.MortarGrid]], pp.ad.Operator])

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=weight_dirichlet_bc,
            neumann_operator=fluid_flux_neumann_bc,
            bc_type=bc_type,
            name=f"bc_values_component_flux_{component.name}",
        )
        flux = self.advective_flux(
            domains, weight, discr, boundary_operator, interface_flux
        )
        flux.set_name(f"component_flux_{component.name}")
        return flux

    def interface_flux_for_component(
        self, component: ppc.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface fluid flux using a component's mobility and discretization for it.

        Parameters:
            omponent: A transportable fluid component in the mixture.
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux in a component's mass
            balance.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_component_mobility_discretization(component, interfaces)
        weight = self.fractional_component_mobility(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name(f"interface_component_flux_{component.name}")
        return flux

    def well_flux_for_component(
        self, component: ppc.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well fluid flux using a component's mobility and discretization for it.

        Parameters:
            omponent: A transportable fluid component in the mixture.
            interfaces: List of interface grids.

        Returns:
            Operator representing the well flux in a component's mass
            balance.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_component_mobility_discretization(component, interfaces)
        weight = self.fractional_component_mobility(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, weight, discr)
        flux.set_name(f"well_component_flux_{component.name}")
        return flux

    def fluid_source_of_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Includes:

        - interface flow from neighboring subdomains of higher dimension.
        - well flow from neighboring subdomains of lower and higher dimension.

        .. note::
            When overriding this method to assign internal fluid sources, one is advised
            to call the base class method and add the new contribution, thus ensuring
            that the source term includes the contribution from the interface fluxes.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term.

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        well_interfaces = self.subdomains_to_interfaces(subdomains, [2])
        well_subdomains = self.interfaces_to_subdomains(well_interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        well_projection = pp.ad.MortarProjections(
            self.mdg, well_subdomains, well_interfaces
        )
        subdomain_projection = pp.ad.SubdomainProjections(self.mdg.subdomains())
        source = projection.mortar_to_secondary_int @ self.interface_flux_for_component(
            component, interfaces
        )
        source.set_name(f"interface_component_flux_source_{component.name}")
        well_fluxes = (
            well_projection.mortar_to_secondary_int
            @ self.well_flux_for_component(component, well_interfaces)
            - well_projection.mortar_to_primary_int
            @ self.well_flux_for_component(component, well_interfaces)
        )
        well_fluxes.set_name(f"well_component_flux_source_{component.name}")
        source += subdomain_projection.cell_restriction(subdomains) @ (
            subdomain_projection.cell_prolongation(well_subdomains) @ well_fluxes
        )
        return source


class EquationsCompositionalFlow(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation_h,
    ComponentMassBalanceEquations,
    ppc.SecondaryExpressionsMixin,
    ppc.EquilibriumEquationsMixin,
):
    has_extended_fractions: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    def secondary_equation_names(self) -> list[str]:
        """Returns a complete list of secondary equations introduced by the
        compositional framework.

        These include the equilibrium equations (if any), the density relations and
        custom secondary expressions (if any).

        """
        return ppc.EquilibriumEquationsMixin.get_equilibrium_equation_names(
            self
        ) + ppc.SecondaryExpressionsMixin.get_secondary_equation_names(self)

    def primary_equation_names(self) -> list[str]:
        """Returns the list of primary equation names, defined as the complement of
        secondary equation names within the complete set of equations stored in
        :attr:`equation_system`."""
        all_equations = set(_ for _ in self.equation_system.equations.keys())
        secondary_equations = set(self.secondary_equation_names())
        return list(all_equations.difference(secondary_equations))

    def set_equations(self):
        TotalMassBalanceEquation.set_equations(self)
        TotalEnergyBalanceEquation_h.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)

        # If an equilibrium is defined, introduce the equations as a block of secondary
        # equations
        if self.equilibrium_type is not None:
            ppc.EquilibriumEquationsMixin.set_equations(self)

        # density relations are always defined and the same, hence we use the base class
        ppc.SecondaryExpressionsMixin.set_density_relations_for_phases(self)

        # This might be custom by an inherited class, hence we use the mixin
        self.set_secondary_equations()


class VariablesCompositionalFlow(
    mass_energy.VariablesFluidMassAndEnergy,
    ppc.CompositeVariables,
):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    has_extended_fractions: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    enthalpy_variable: str
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        1. Sets up the pressure and temperature variables from standard mass and energy
           transport models.
        3. Creates the transported enthalpy variable.
        4. Ccreates all compositional variables.

        """
        # pressure and temperature. This covers also the interface variables for
        # Fourier flux, Darcy flux and enthalpy flux.
        mass_energy.VariablesFluidMassAndEnergy.create_variables(self)

        # enthalpy variable
        self.equation_system.create_variables(
            self.enthalpy_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "J"},
        )

        # compositional variables
        ppc.CompositeVariables.create_variables(self)

    def enthalpy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Representation of the enthalpy as an AD-Operator.

        Parameters:
            domains: List of subdomains or list of boundary grids.

        Raises:
            ValueError: If the passed sequence of domains does not consist entirely
                of instances of boundary grid.

        Returns:
            A mixed-dimensional variable representing the enthalpy, if called with a
            list of subdomains.

            If called with a list of boundary grids, returns an operator representing
            boundary values.

        """
        if len(domains) > 0 and all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return self.create_boundary_operator(
                name=self.enthalpy_variable, domains=domains  # type: ignore[call-arg]
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument domains a mixture of subdomain and boundary grids"""
            )

        domains = cast(list[pp.Grid], domains)

        return self.equation_system.md_variable(self.enthalpy_variable, domains)


class ConstitutiveLawsCompositionalFlow(
    MobilityCF,
    ThermalConductivityCF,
    PermeabilityCF,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.DimensionReduction,
):
    """Constitutive laws for CF with thermal conductivity and permeability adapted to
    the compositional setting."""


class BoundaryConditionsCompositionalFlow(
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
):
    """Mixin treating boundary conditions for the compositional flow.

    Atop of inheriting the treatment for single phase flow (which is exploited for the
    total mass balance) and the energy balance (total energy balance), this class has
    a treatment for BC for component mass balances.

    By default, Dirichlet-type BC are implemented for pressure and temperature.

    """

    mobility_keyword: str
    """Prodived by
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`"""
    bc_data_fluid_flux_key: str
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`"""

    def bc_data_component_flux_key(self, component: ppc.Component) -> str:
        """Replaces :attr:`~porepy.models.fluid_mass_balance.
        BoundaryConditionsSinglePhaseFlow.bc_data_fluid_flux_key` to access the
        Neumann data of the advective flux in a components's mas balance.

        Parameters:
            component: A fluid component in the mixture with a mass balance equation.

        Returns:
            The key for storing (time-dependent) Neumann BC data for the advective flux
            in the components mass balance.

            Note that this data must be computed in the solution strategy.

        """
        return f"{self.bc_data_fluid_flux_key}_{component.name}"

    def bc_values_component_flux(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        r"""Replaces :meth:`~porepy.models.fluid_mass_balance.
        BoundaryConditionsSinglePhaseFlow.bc_values_fluid_flux` to compute the
        Neumann data of the advective flux in a components's mas balance.

        Important:
            Override this method to provide custom Neumann data for the flux,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing the mass
            fluid flux values on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_type_component_flux(
        self, component: ppc.Component, grid: pp.Grid
    ) -> pp.BoundaryCondition:
        """Replaces :meth:`~porepy.models.fluid_mass_balance.
        BoundaryConditionsSinglePhaseFlow.bc_type_fluid_flux` in a component's mass
        balance

        Parameters:
            component: A transportable fluid component in the mixture.
            grid: Subdomain on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned.

        """
        # Define boundary faces. TODO this should be the same as for the Darcy flux
        boundary_faces = self.domain_boundary_sides(grid).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(grid, boundary_faces, "dir")


class SolutionStrategyCompositionalFlow(
    mass_energy.SolutionStrategyFluidMassAndEnergy,
):
    """Solution strategy for compositional flow.

    The initialization parameters can contain the following entries:

    - ``'eliminate_reference_phase'``: Defaults to True. If True, the molar fraction
      and saturation of the reference phase are eliminated by unity, reducing the size
      of the system. If False, more work is required by the modeller.
    - ``'eliminate_reference_component'``: Defaults to True. If True, the overall
      fraction of the reference component is eliminated by unity, reducing the number
      of unknowns. Also, the local mass constraint for the reference component is
      removed as an equation. If False, the modelled must close the system.
    - ``'normalize_state_constraints'``: Defaults to True. If True, local state
      constraints in the equilibrium problem are devided by the target value. Equations
      relating state functions become dimensionless.
    - ``'use_semismooth_complementarity'``: Defaults to True. If True, the
      complementarity conditions for each phase are formulated in semi-smooth form using
      a AD-compatible ``min`` operator. The semi-smooth Newton can be applied to solve
      the equilibrium problem. If False, the modeller must adapt the solution strategy.
    - ``'has_extended_fractions'``: See :attr:`has_extended_fractions`.

    The following parameters are required:

    - ``'equilibrium_type'``: See :attr:`equilibrium_type`.

    Raises:
        ValueError: If no equilibrium type is defined and fractions in phases are
            **not** requested by the user in the mixture mixin.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    bc_type_component_flux: Callable[[ppc.Component, pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""

    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`ThermalConductivityCF`."""
    total_mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`MobilityCF`."""
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """

    enthalpy_mobility_discretization: Callable[[Sequence[pp.Grid]], pp.ad.UpwindAd]
    """Provided by :class:`MobilityCF`."""
    interface_enthalpy_mobility_discretization: Callable[
        [Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    fourier_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`MobilityCF`."""
    interface_component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    equilibriate_fluid: Callable[
        [Optional[np.ndarray]], tuple[ppc.FluidState, np.ndarray]
    ]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    postprocess_failures: Callable[[ppc.FluidState, np.ndarray], ppc.FluidState]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    set_up_flasher: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""

    temperature_variable: str
    """Provided by :class:`~porepy.energy_balance.SolutionStrategyEnergyBalance`."""
    pressure_variable: str
    """Provided by :class:~porepy.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.
    """
    feed_fraction_variables: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""
    solute_fraction_variables: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""
    phase_fraction_variables: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""
    saturation_variables: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""
    compositional_variables: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""

    create_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    assign_thermodynamic_properties_to_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations = list()

        self.equilibrium_type: Literal["p-T", "p-h", "v-h"] | None = params[
            "equilibrium_type"
        ]
        """A string denoting the two state functions which are assumed constant at
        equilibrium.

        Important:
            The user **must** provide a value explictly in the input parameters.

        If set to ``None``, the framework assumes there are no equilibrium calculations,
        hence there are **no equilibrium equations** and **no equilibriation of the
        fluid**.
        Also, **no fractions of components in phases** are created as unknowns
        (see :attr:`~porepy.composite.base.Phase.fraction_of`).

        The user must in this case provide secondary equations which provide expressions
        for the unknowns in the equilibrium:

        Examples:
            If no equilibrium is defined, there are dangling variables which need a
            definition or a constitutive law.

            These include:

            1. Temperature
            :meth:`~porepy.models.energy_balance.VariablesEnergyBalance.temperature`
            2. Phase saturations :attr:`~porepy.composite.base.Phase.saturation`
            3. Optionally molar fractions of components in phases
            :attr:`~porepy.composite.base.Phase.fraction_of` if
            :attr:`has_extended_fractions` is True.

            Note that secondary expressions relating molar phase fractions and
            saturations via densities are always included.

        """

        has_extended_fractions = params.get("has_extended_fractions", None)
        if has_extended_fractions is None:
            if self.equilibrium_type is None:
                has_extended_fractions = False
            else:
                has_extended_fractions = True
        has_extended_fractions = cast(bool, has_extended_fractions)

        self.has_extended_fractions: bool = has_extended_fractions
        """A flag indicates whether (extended) fractions of components in phases should
        be created or not. This must be True, if :attr:`equilibrium_type` is not None.

        Note:
            This is optional.
            If ``equilibrium_type`` is None, it defaults to False.
            If ``equilibrium_type`` is set, it defaults to True

        If True and ``equilibrium_type == None``, the user must provide secondary
        expressions for the fractional variables to close the system.

        Note:
            Molar fractions of phases (:attr:`~porepy.composite.base.Phase.fraction`)
            are always created, as well as saturations
            (:attr:`~porepy.composite.base.Phase.saturation`).

            This is due to the definition of a mixture properties as a sum of partial
            phase properties weighed with respective fractions.

            Secondary equations, relating molar fractions and saturations via densities,
            are always included in the compositional flow equation.

        """

        # Performing sanity checks of set-up
        if self.equilibrium_type is not None and self.has_extended_fractions is False:
            raise ValueError(
                f"Conflicting model set-up: No extended fractions requested but"
                + f" {self.equilibrium_type} equilibrium set."
            )
        elif self.equilibrium_type is None and self.has_extended_fractions is True:
            warnings.warn(
                "Unusual model set-up: Extended fractions requested but no"
                + " equilibrium system requested. Check secondary equations."
            )

        self.eliminate_reference_phase: bool = params.get(
            "eliminate_reference_phase", True
        )
        """Flag to eliminate the molar phase fraction and saturation of the reference
        phase as variables from the model."""

        self.eliminate_reference_component: bool = params.get(
            "eliminate_reference_phase", True
        )
        """Flag to eliminate the overall fraction of the reference component as a
        variable and the local mass constraint for said component from the model."""

        self.normalize_state_constraints: bool = params.get(
            "normalize_state_constraints", True
        )
        """Flag to divide state function constraints by some overall value and make
        respective equations non-dimensional in the physical sense."""

        self.use_semismooth_complementarity: bool = params.get(
            "use_semismooth_complementarity", True
        )
        """Flag to use a semi-smooth min operator for the complementarity conditions
        in the equilibrium problem. As a consequence the equilibrium problem
        can be solved locally using a semi-smooth Newton algorithm."""

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid mixture."""

        self._domainprojection: pp.ad.SubdomainProjections(self.mdg.subdomains())

    def prepare_simulation(self) -> None:
        """Introduces some additional elements in between steps performed by the parent
        method.

        1. It creates a mixture before creating any variables.
        2. After creating variables it creates the phase properties defined by the
           mixture mixin.
        3. At the end it instantiates the flash instance if an equilibrium type is
           defined and computes the initial equilibrium in subdomains and on boundaries.

        """
        self.set_materials()
        self.set_geometry()
        self.initialize_data_saving()
        self.set_equation_system_manager()

        # This block is new and the order is critical
        self.create_mixture()
        self.create_variables()
        self.assign_thermodynamic_properties_to_mixture()

        self.initial_condition()
        self.reset_state_from_file()
        self.set_equations()
        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()
        self.save_data_time_step()

        # Set the flash if an equilibrium system is defined
        if self.equilibrium_type is not None:
            self.set_up_flasher()

    def initial_condition(self) -> None:
        """Initiates additionally zero fluxes for the component mobilities."""
        super().initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
            )

    def set_discretization_parameters(self) -> None:
        """Sets the discretization parameters, minding the custom Fourier law."""
        # First, discretizations in single phase flow, which is used for pressure equ.
        mass.SolutionStrategySinglePhaseFlow.set_discretization_parameters(self)

        # Second, discretizations in the energy equation. Slightly different due to
        # having a fluid mixture instead of fluid constants
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.fourier_keyword,
                {
                    "bc": self.bc_type_fourier_flux(sd),
                    # initialize with zero, this will be computed anyways before iter
                    "second_order_tensor": pp.SecondOrderTensor(np.zeros(sd.num_cells)),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_enthalpy_flux(sd),
                    # TODO copy boundary values from boundary operator
                },
            )

        # Third, discretization parameters for mobility discretizations
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {
                    "bc": self.bc_type_darcy_flux(sd),
                },
            )

    def update_discretization_parameters(self) -> None:
        """Method called before non-linear iterations to update discretization
        parameters.

        The base method computes the conductivity tensor by evaluating the fluid's
        thermal conductivity and storing it as a cell-wise isotropic tensor.

        It does the same for the pressure equation, where total mobility is evaluated
        and multiplied with absolute permeability.

        """

        subdomains = self.mdg.subdomains()
        conducivity = self.thermal_conductivity(subdomains)
        total_mobility = self.total_mobility(subdomains)
        projection = pp.ad.SubdomainProjections(subdomains)

        # NOTE The non-linear MPFA discretization for the Conductive flux in the heat
        # equation and the diffusive flux in the pressure equation are missing
        # derivatives w.r.t. their dependencies.. Jacobian is NOT exact.
        # NOTE this is critical if total mobility is formulated as an auxiliary variable

        for sd, data in self.mdg.subdomains(return_data=True):
            mat = projection.cell_restriction([sd])

            conducivity_sd = pp.SecondOrderTensor(
                (mat @ conducivity).value(self.equation_system)
            )
            data[pp.PARAMETERS][self.fourier_keyword].update(
                {"second_order_tensor": conducivity_sd}
            )

            total_mob_sd = pp.SecondOrderTensor(
                ((mat @ total_mobility) * self.permeability([sd])).value(
                    self.equation_system
                )
            )
            data[pp.PARAMETERS][self.darcy_keyword].update(
                {"second_order_tensor": total_mob_sd}
            )

    def set_nonlinear_discretizations(self) -> None:
        """Overwrites parent methods to point to discretizations in
        :class:`MobilityCF`.

        Adds additionally the non-linear MPFA discretizations to a separate list, since
        the updates are performed at different steps in the algorithm.

        """
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()

        # Upwind of enthalpy mobility in energy equation
        self.add_nonlinear_discretization(
            self.enthalpy_mobility_discretization(subdomains).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_enthalpy_mobility_discretization(interfaces).flux,
        )

        # Upwinding of mobilities in component balance equations
        for component in self.fluid_mixture.components:
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue

            self.add_nonlinear_discretization(
                self.component_mobility_discretization(component, subdomains).upwind,
            )
            self.add_nonlinear_discretization(
                self.interface_component_mobility_discretization(
                    component, interfaces
                ).flux,
            )

        # TODO this is experimental and expensive
        self._nonlinear_flux_discretizations.append(
            self.fourier_flux_discretization(subdomains),
        )
        self._nonlinear_flux_discretizations.append(
            self.darcy_flux_discretization(subdomains),
        )

    def rediscretize_fluxes(self) -> None:
        """Discretize nonlinear terms."""
        tic = time.time()
        # Uniquify to save computational time, then discretize.
        unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
            self._nonlinear_flux_discretizations
        )
        pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)
        logger.info(
            "Re-discretized nonlinear fluxes in {} seconds".format(time.time() - tic)
        )

    def before_nonlinear_loop(self) -> None:
        """TODO update boundary conditions for upwinding if non-constant BC
        (boundary equilibrium)"""
        super().before_nonlinear_loop()

    def perform_flash(self) -> ppc.FluidState:
        """Method performing flash calculations and updating the iterative values of
        unknowns in the local equilibrium problem on the whole mixed-dimensional grid.

        This method is called in as a first step in :meth:`before_nonlinear_iteration`
        if ``equilibrium_type`` is defined.

        1. Calls
           :meth:`~porepy.composite.composite_mixins.FlashMixin.equilibriate_fluid`,
        2. Calls
           :meth:`~porepy.composite.composite_mixins.FlashMixin.postprocess_failures`,
        3. Updates all fractional variables using the flash results, as well as pressure
           and temperature, depending on the equilibrium type.
        4. Returns the fluid state.

        """
        state = self.postprocess_failures(*self.equilibriate_fluid(None))

        # Setting equilibrium values for fractional variables
        vars_y = self.phase_fraction_variables
        vars_s = self.phase_fraction_variables
        if vars_y:  # if not empty
            ind = 0  # lagging indexation of ref phase eliminated
            for j in range(self.fluid_mixture.num_phases):
                # skip if not a variable
                if self.eliminate_reference_phase:
                    continue
                self.equation_system.set_variable_values(
                    state.sat[j], [vars_y[ind]], iterate_index=0
                )
                self.equation_system.set_variable_values(
                    state.y[j], [vars_s[ind]], iterate_index=0
                )
                ind += 1

        if self.has_extended_fractions:
            vars_x = self.compositional_variables
            ind = 0
            # NOTE This assumes no-one messed with the order during creation
            for j, phase in enumerate(self.fluid_mixture.phases):
                for i in range(len(phase.components)):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[i], [vars_x[ind]], iterate_index=0
                    )
                    ind += 1

        # setting Temperature and pressure values, depending on equilibrium definition
        if "T" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                state.T, [self.temperature_variable], iterate_index=0
            )
        if "p" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                state.p, [self.pressure_variable], iterate_index=0
            )

        # TODO The resulting fluid enthalpy can change due to numerical
        # precision. Should it be updated as well?

        return state

    def update_thermodynamic_properties(
        self, fluid_state: Optional[ppc.FluidState] = None
    ) -> None:
        """Method to update various thermodynamic properties of present phases, on all
        subdomains.

        If ``fluid_state`` is None, evaluates pressure, temperature and fractions.
        If a ``fluid_state`` is given, uses respective values.

        Calls :meth:`~porepy.composite.base.Mixture.compute_properties` to store the
        values.

        """

        if fluid_state is None:
            subdomains = self.mdg.subdomains()

            p = self.equation_system.get_variable_values([self.pressure_variable])
            T = self.equation_system.get_variable_values([self.temperature_variable])
            xn = [
                [
                    phase.normalized_fraction_of[comp](subdomains).value(
                        self.equation_system
                    )
                    for comp in phase.components
                ]
                for phase in self.fluid_mixture.phases
            ]
        else:
            p = fluid_state.p
            T = fluid_state.T
            xn = [phase.xn for phase in fluid_state.phases]

        self.fluid_mixture.compute_properties(p, T, xn, store=True)

    def update_discretizations(self) -> None:
        """Convenience method to update discretization parameters and non-linear
        discretizations.

        Called in :meth:`before_nonlinear_iteration` after the thermodynamic properties
        are updated.

        """
        # re-discretize
        self.update_discretization_parameters()
        self.rediscretize_fluxes()

        for sd, data in self.mdg.subdomains(return_data=True):
            # Computing Darcy flux and updating it in the mobility dicts for pressure
            # and energy equtaion
            vals = self.darcy_flux([sd]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            # Computing the darcy flux in fractures (given by variable)
            vals = self.interface_darcy_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            # Computing the darcy flux in wells (given by variable)
            vals = self.well_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        # Re-discretize Upwinding
        self.rediscretize()

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform the p-h flash as a predictor step.

        Subsequently it computes the fluxes for various Upwind discretiztions
        (without calling the parent methods of mass and energy though, to save time).

        Finally, it calles the base class' method to update discretization parameters
        and to re-discretize.

        """

        # Flashing the mixture as a predictor step, if equilibrium defined
        if self.equilibrium_type is not None:
            fluid_state = self.perform_flash()
        else:
            fluid_state = None

        self.update_thermodynamic_properties(fluid_state)

        # After updating the fluid properties, update discretizations
        self.update_discretizations()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Expands the Schur complement using ``solution_vector``, to include secondary
        variables."""

        global_solution_vector = ...  # TODO
        super().after_nonlinear_iteration(global_solution_vector)

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        This method performs a Schur complement elimination.

        Primary variables are pressure, enthalpy and overall feed fractions,
        as well as the interface fluxes (Darcy and Fourier)
        Primary equations are the pressure equation, energy equation and component mass
        balance equations, as well as the interface equations for the interface fluxes.

        Secondary variables are the remaining molar fractions, saturations and
        temperature.
        Secondary equations are all flash equations, including the phase density
        relations.

        """
        t_0 = time.time()
        self.linear_system = ...  # TODO
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds.")


class CompositionalFlow(  # type: ignore[misc]
    ppc.FluidMixtureMixin,
    ppc.FlashMixin,
    EquationsCompositionalFlow,
    VariablesCompositionalFlow,
    ConstitutiveLawsCompositionalFlow,
    BoundaryConditionsCompositionalFlow,
    SolutionStrategyCompositionalFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Generic class for setting up a multiphase multi-component flow model.

    The primary, transportable variables are:

    - pressure
    - (specific molar) enthalpy of the mixture
    - ``num_comp - 1 `` overall fractions per independent component

    The secondary, local variables are:

    - ``num_phases - 1`` saturations per independent phase
    - ``num_phases - 1`` molar phase fractions per independent phase
    - ``num_phases * num_comp`` extended fractions of components in phases
    - temperature

    The primary block of equations consists of:

    - pressure equation / transport of total mass
    - energy balance / transport of total energy
    - ``num_comp - 1`` transport equations for each independent component

    The secondary block of equations represents the local equilibrium problem formulated
    as a unified p-h flash:

    - ``num_comp - 1`` local mass conservation equations for each independent component
    - ``num_comp * (num_phase - 1)`` isofugacity constraints
    - ``num_phases`` semi-smooth complementarity conditions for each phase
    - local enthalpy constraint (equating mixture enthalpy with transported enthalpy)
    - ``num_phase - 1`` density relations per independent phase

    In total the model encompasses
    ``3 + num_comp - 1 + num_comp * num_phases + 2 * (num_phases - 1)`` equations und
    DOFs per cell in each subdomains, excluding the unknowns on interfaces.

    Example:
        For the most simple model set-up, make a mixture mixing defining the components
        and phases, and derive a flow model from it.

        Instantiate the model with ```equilbrium_type`=None``, i.e. no local
        thermodynamic equilibrium calculations.
        This is essentially a multiphase flow without phase change.

        .. code:: python
            :linenos:

            class MyMixture(FluidMixtureMixin):

                def get_components(self):
                    ...
                def get_phase_configuration(self, components):
                    ...

            class MyModel(MyMixture, CompositionalFlow):
                ...

    """

    pass
