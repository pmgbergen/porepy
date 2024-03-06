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
from porepy.composite.utils_c import extended_compositional_derivatives_v as _extend
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

    Important:
        Mobility terms are designed to be representable also on boundary grids.
        **This is intended for the Dirichlet boundary where a value is required for e.g.
        upwinding.**

        Values on the Neumann boundary (especially fractional mobilities) must be
        implemented by the user in :class:`BoundaryConditionsCompositionalFlow`.
        Those values are then consequently multiplied with boundary flux values in
        respective balance equations.

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

    def total_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Non-linear term in the Darcyflux in the pressure equation.

        Parameters:
            domains: A list of subdomains or boundary grids.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j k_r(s_j)}{\mu_j},

            which is the advected mass in the total mass balance equation.

        """
        name = "total_mobility"
        # change name if on boundary to help the user in the operator tree
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                name = f"bc_{name}"
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains)
                / phase.viscosity(domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            name,
        )
        return mobility

    def advected_enthalpy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Non-linear term in the advective flux in the energy balance equation.

        Parameters:
            domains: A sequence of subdomains or boundary grids

        Returns:
            An operator representing

            .. math::

                \sum_j h_j \rho_j \dfrac{k_r(s_j)}{\mu_j}

            which is the advected enthalpy in the total energy balance.

        """
        name = "advected_enthalpy"
        # change name if on boundary to help the user in the operator tree
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                name = f"bc_{name}"
        weight = pp.ad.sum_operator_list(
            [
                phase.enthalpy(domains)
                * phase.density(domains)
                / phase.viscosity(domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            name,
        )
        return weight

    def advected_component_mass(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear term in the advective flux in a component mass balance equation.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
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
        name = f"advected_mass_{component.name}"
        # change name if on boundary to help the user in the operator tree
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                name = f"bc_{name}"
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains)
                / phase.viscosity(domains)
                * phase.partial_fraction_of[component](domains)
                * self.relative_permeability(phase.saturation(domains))
                for phase in self.fluid_mixture.phases
            ],
            name,
        )
        return mobility

    def fractional_component_mobility(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the :meth:`advected_component_mass` divided by the
        :meth:`total_mobility`.

        To be used in a fractional flow set-up, where the total mobility is part of the
        non-linear diffusive tensor in Darcy flux.

        I.e.,

        .. math::

            - \nabla \cdot \left(f_i D(p, Y) \nabla p\right),

        assuming the tensor :math:`D(p, Y)` contains the total mobility.


        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_i` in above expession in operator form.

        """
        name = f"fractional_mobility_{component.name}"
        # change name if on boundary to help the user in the operator tree
        if len(domains) > 0:
            if isinstance(domains[0], pp.BoundaryGrid):
                name = f"bc_{name}"
        op = self.advected_component_mass(component, domains) / self.total_mobility(
            domains
        )
        op.set_name(name)
        return op

    def total_mobility_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the total fluid mobility in the total mass balance on the
        subdomains.

        Important:
            Upwinding in the pressure equation is inconsistent. This method is left for
            now, but should not be used.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_total_mobility_discretization(
        self, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the total fluid mobility in the advective flux in fluid
        flux on interfaces.

        Important:
            As for :meth:`total_mobility_discretization`, this should not be used in a
            consistent formulation.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)

    def advected_enthalpy_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the enthalpy flux in the energy
        balance on the subdomains.

        Note:
            Though the same flux is used as in the pressure equation and mass balances
            (Darcy flux), the storage key for this discretization is different than
            f.e. for :meth:`fractional_mobility_discretization` for flexibility reasons.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the enthalpy mobility.

        """
        return pp.ad.UpwindAd(self.enthalpy_keyword, subdomains)

    def interface_advected_enthalpy_discretization(
        self, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the non-linear weight in the enthalpy flux in the
        energy balance equations on interfaces.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface enthalpy mobility.

        """
        return pp.ad.UpwindCouplingAd(self.enthalpy_keyword, interfaces)

    def fractional_mobility_discretization(
        self, component: ppc.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the advective flux in the
        component mass balance equations.

        Note:
            In a fractional flow formulation with a single darcy flux, this
            discretization should be the same for all components.
            The signature is left to include the component for any case.

            This base method uses the same storage key word for all discretizations.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Discretization of the mobility in the component's mass balance.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_fractional_mobility_discretization(
        self, component: ppc.Component, interfaces: Sequence[pp.MortarGrid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the advective flux in the
        component mass balance equations on interfaces.

        Same note applies as for :meth:`fractional_mobility_discretization`.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            interfaces: List of interfaces.

        Returns:
            Discretization of the mobility in the component's mass balance.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)


class ThermalConductivityCF(pp.constitutive_laws.ThermalConductivityLTE):
    """A constitutive law providing the fluid and normal thermal conductivity to be
    used with Fourier's Law."""

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
    """A constitutive law providing relative and normal permeability functions
    in the compositional framework."""

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

    Defines an advective weight to be used in the advective flux, assuming the total
    mobility is part of the diffusive tensor in the pressure equation.

    Note:
        Since enthalpy is an independent variable, models using this balance need
        something more:

        If temperature is also an independent variable, it needs a constitutive law
        to close the system.

        If temperature is not an independent variable,
        :meth:`porepy.models.energy_balance.VariablesEnergyBalance.temperature` needs to
        be overwritten to provide a secondary expression.

    Note:
        Many of the methods here which override parent methods can be omitted with
        a proper generalization of the advective weight in the parent class. TODO

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`VariablesCompositionalFlow`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcyFlux`."""
    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`"""
    advected_enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`"""

    advected_enthalpy_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Provided by :class:`MobilityCF`."""
    interface_advected_enthalpy_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    bc_type_darcy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`."""

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites the parent method to use the fluix mixture density and the primary
        unknown enthalpy."""
        energy = (
            self.fluid_mixture.density(subdomains) * self.enthalpy(subdomains)
            - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_mixture_internal_energy")
        return energy

    def advective_weight_enthalpy_flux(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the enthalpy flux.

        This is computed by dividing the :meth:`~MobilityCF.advected_enthalpy` by the
        :meth:`~MobilityCF.total_mobility`,
        consistent with the definition of the Darcy flux in
        :class:`TotalMassBalanceEquation`, where the total mobility is part of the
        diffusive tensor.

        Note:
            The advective weight must have values on the Dirichlet boundary with influx

        """
        op = self.advected_enthalpy(domains) / self.total_mobility(domains)
        op.set_name("advective_weight_enthalpy_flux")
        return op

    def enthalpy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """The child method modifies the non-linear weight in the enthalpy flux
        (enthalpy mobility) and the custom discretization for it.

        Note:
            The enthalpy flux is also defined on the Neumann boundary.

        """

        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # enthalpy flux values on the boundary (Neumann type) are given by the
            # overall darcy flux and user-provided values for the weight
            fraction_boundary = self.create_boundary_operator(
                self.bc_data_enthalpy_flux_key, domains
            )
            op = fraction_boundary * self.darcy_flux(domains)
            op.set_name(f"bc_enthalpy_flux")
            return op

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                "Domains must consist entirely of subdomains for the enthalpy flux."
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        boundary_operator_enthalpy = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=domains,
                dirichlet_operator=self.advective_weight_enthalpy_flux,
                neumann_operator=self.enthalpy_flux,
                bc_type=self.bc_type_darcy_flux,  # TODO this is inconsistent in parent
                name="bc_values_enthalpy",
            )
        )

        discr = self.advected_enthalpy_discretization(domains)
        weight = self.advective_weight_enthalpy_flux(domains)
        flux = self.advective_flux(
            domains,
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
        discr = self.interface_advected_enthalpy_discretization(interfaces)
        weight = self.advective_weight_enthalpy_flux(subdomains)
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
        discr = self.interface_advected_enthalpy_discretization(interfaces)
        weight = self.advective_weight_enthalpy_flux(subdomains)
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

    This equation is defined on all subdomains. Due to a single pressure and interface
    flux variable, there is no need for additional equations as is the case in the
    pressure equation.

    Note:
        This is a sophisticated version of
        :class:`~porepy.models.fluid_mass_balance.MassBalanceEquation` where the
        non-linear weights in the flux term stem from a multiphase, multicomponent
        mixture.

        There is room for unification and recycling of code. TODO

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcyFlux`."""

    fractional_component_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`."""

    fractional_mobility_discretization: Callable[
        [ppc.Component, list[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`MobilityCF`."""
    interface_fractional_mobility_discretization: Callable[
        [ppc.Component, list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    bc_data_fractional_mobility_key: Callable[[ppc.Component], str]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""

    bc_type_darcy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`."""

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all independent components on all subdomains.

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
        r"""The accumulated fluid mass for a given component in the overall fraction
        formulation, i.e. cell-wise volume integral of

        .. math::

            \Phi \left(\sum_j \rho_j s_j\right) z_j

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Operator representing above expression.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid_mixture.density(subdomains)
            * component.fraction(subdomains)
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name(f"component_mass_{component.name}")
        return mass

    def advective_weight_component_flux(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the advective mass flux in the fractional
        formulation.

        This is computed by dividing the advected component by the total mobility,
        consistent with the definition of the Darcy flux in
        :class:`TotalMassBalanceEquation`, where the total mobility is part of the
        diffusive tensor.

        Note:
            The advective weight must have values on the Dirichlet boundary with influx.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            See :meth:`MobilityCF.fractional_component_mobility`.

        """
        return self.fractional_component_mobility(component, domains)

    def fluid_flux_for_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A fractional fluid flux, where the advective flux consists of the Darcy flux
        multiplied with the fractional mobility of a component.

        Assumes a consistent formulation and discretization of the pressure equation,
        where the total mobility is part of the diffusive tensor in the Darcy flux.

        It also accounts for custom choices of mobility discretizations (see
        :class:`MobilityCF`).

        Note:
            The fluid flux is also defined on the Neumann boundary.
            In this case it returns the BC values for the fractional mobility multiplied
            with the Darcy flux on the Neumann boundary.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            The advective flux in a component mass balance equation.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Flux values on the boundary (Neumann type) are given by the overall darcy
            # flux and user-provided values for the fractional mobility
            fraction_boundary = self.create_boundary_operator(
                self.bc_data_fractional_mobility_key(component), domains
            )
            op = fraction_boundary * self.darcy_flux(domains)
            op.set_name(f"bc_component_flux_{component.name}")
            return op

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        discr = self.fractional_mobility_discretization(component, domains)
        weight = self.advective_weight_component_flux(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        weight_dirichlet_bc = partial(self.advective_weight_component_flux, component)
        weight_dirichlet_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            weight_dirichlet_bc,
        )
        fluid_flux_neumann_bc = partial(self.fluid_flux_for_component, component)
        fluid_flux_neumann_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            fluid_flux_neumann_bc,
        )
        interface_flux = partial(self.interface_flux_for_component, component)
        interface_flux = cast(Callable[[list[pp.MortarGrid]], pp.ad.Operator])

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=weight_dirichlet_bc,
            neumann_operator=fluid_flux_neumann_bc,
            bc_type=self.bc_type_darcy_flux,
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
        """Interface fluid flux using a component's fractional mobility and its
        discretization (see :class:`MobilityCF`).

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Operator representing the interface fluid flux in a component's mass
            balance.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_fractional_mobility_discretization(component, interfaces)
        weight = self.advective_weight_component_flux(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name(f"interface_component_flux_{component.name}")
        return flux

    def well_flux_for_component(
        self, component: ppc.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well fluid flux using a component's mobility and discretization for it.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Operator representing the well flux in a component's mass
            balance.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_fractional_mobility_discretization(component, interfaces)
        weight = self.advective_weight_component_flux(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, weight, discr)
        flux.set_name(f"well_component_flux_{component.name}")
        return flux

    def fluid_source_of_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Analogous to
        :meth:`~porepy.models.fluid_mass_balance.MassBalanceEquations.fluid_source`,
        but using the terms related to the component.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
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


class SoluteTransportEquations(ComponentMassBalanceEquations):
    """Simple transport equations for every solute in a fluid component, which is
    a compound.

    The only difference to the compound's mass balance is, that the accumulation term
    is additionally weighed with the solute fraction.

    """

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all independent components on all subdomains.

        """
        subdomains = self.mdg.subdomains()

        for component in self.fluid_mixture.components:
            if not isinstance(component, ppc.Compound):
                continue

            for solute in component.solutes:
                sd_eq = self.transport_equation_for_solute(
                    solute, component, subdomains
                )
                self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def transport_equation_for_solute(
        self,
        solute: ppc.ChemicalSpecies,
        component: ppc.Compound,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Mass balance equation for subdomains for a given component.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.mass_for_solute(solute, component, subdomains)
        flux = self.fluid_flux_for_component(component, subdomains)
        source = self.fluid_source_of_component(component, subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(f"transport_equation_{solute.name}_{component.name}")
        return eq

    def mass_for_solute(
        self,
        solute: ppc.ChemicalSpecies,
        component: ppc.Compound,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        r"""The accumulated mass of a solute, for a given solute in a fluid compound in
        the overall fraction formulation, i.e. cell-wise volume integral of

        .. math::

            \Phi \left(\sum_j \rho_j s_j\right) z_j c_i,

        which is essentially the accumulation of the compound mass scaled with the
        solute fraction.

        Parameters:
            component: A component in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Operator representing above expression.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid_mixture.density(subdomains)
            * component.fraction(subdomains)
            * component.solute_fraction_of[solute](subdomains)
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name(f"solute_mass_{solute.name}_{component.name}")
        return mass


class EquationsCompositionalFlow(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation_h,
    SoluteTransportEquations,
    ComponentMassBalanceEquations,
    ppc.SecondaryEquationsMixin,
    ppc.EquilibriumEquationsMixin,
):

    @property
    def secondary_equation_names(self) -> list[str]:
        """Returns a complete list of secondary equations introduced by the
        compositional framework.

        These include the equilibrium equations (if any), the density relations and
        custom secondary expressions (if any).

        """
        return ppc.EquilibriumEquationsMixin.get_equilibrium_equation_names(self) + [
            name for name in self._secondary_equation_names
        ]

    @property
    def primary_equation_names(self) -> list[str]:
        """Returns the list of primary equation names, defined as the complement of
        secondary equation names within the complete set of equations stored in
        :attr:`equation_system`."""
        all_equations = set(_ for _ in self.equation_system.equations.keys())
        secondary_equations = set(self.secondary_equation_names)
        return list(all_equations.difference(secondary_equations))

    def set_equations(self):
        """This method introduces:

        1. The total mass balance equation
        2. The total energy balance equation (with indepdnent enthalpy variable)
        3. Component mass balance equations
        4. Solute transport equations
        5. Local equilibrium equations (if :attr:`equilibrium_type` is not None)
        6. Secondary equations defined in respective mixin

        Important:
            Do not forget that the basic
            :class:`~porepy.composite.composite_mixins.SecondaryEquationsMixin`
            introduces the density relations into the system, which are required in case
            the model has independent phase molar fractions **and** saturations.

        """

        # PDEs
        TotalMassBalanceEquation.set_equations(self)
        TotalEnergyBalanceEquation_h.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)
        SoluteTransportEquations.set_equations(self)

        # LEEs if defined
        if self.equilibrium_type is not None:
            ppc.EquilibriumEquationsMixin.set_equations(self)

        # Other secondary equations such as the density relation equation
        self.set_secondary_equations()


class VariablesCompositionalFlow(
    mass_energy.VariablesFluidMassAndEnergy,
    ppc.CompositeVariables,
):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    enthalpy_variable: str
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    @property
    def primary_variable_names(self) -> list[str]:
        """Returns a list of primary variables, which in the basic set-up consist
        of pressure, fluid enthalpy and overall fractions."""
        return [
            self.pressure_variable,
            self.enthalpy_variable,
        ] + self.overall_fraction_variables

    @property
    def secondary_variables(self) -> list[str]:
        """Returns a list of secondary variables, which in the basic set-up consist of

        - temperature
        - saturations
        - phase molar fractions
        - extended fractions
        - solute fractions

        """
        return (
            [self.temperature_variable]
            + self.saturation_variables
            + self.phase_fraction_variables
            + self.extended_fraction_variables
            + self.solute_fraction_variables
        )

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


def _prolong_boundary_state(
    state: ppc.FluidState, nc: int, dir: np.ndarray
) -> ppc.FluidState:
    """Helper method to broadcast the fluid state computed on the Dirichlet-part of a
    boundary, to the whole boundary.

    Note:
        Values of derivatives are not prolonged as of now.

    Parameters:
        state: Flash results with values per Dirichlet cell on a grid.
        nc: Total number of cells.
        dir: ``dtype=bool``

            Boolean array indicating which elements of an array with shape ``(nc,)``
            correspond to the Dirichlet boundary.

    Returns:
        A fluid state where all values are prolongated to a vector with shape ``(nc,)``.

    """
    # base vector has only zeros
    vec = np.zeros(nc)
    out = ppc.FluidState()

    # prolonging intensive and extensive fluid state
    val = vec.copy()
    val[dir] = state.p
    out.p = val
    val = vec.copy()
    val[dir] = state.T
    out.T = val
    val = vec.copy()
    val[dir] = state.h
    out.h = val
    val = vec.copy()
    val[dir] = state.v
    out.v = val
    # feed fractions
    z = list()
    for z_ in state.z:
        val = vec.copy()
        val[dir] = z_
        z.append(z_)
    out.z = np.array(z)
    # saturations and molar fractions
    nphase = len(state.y)
    sat = list()
    y = list()
    phases: Sequence[ppc.PhaseState] = list()
    for j in range(nphase):
        val = vec.copy()
        val[dir] = state.sat[j]
        sat.append(val)
        val = vec.copy()
        val[dir] = state.y[j]
        y.append(val)
        x_j = list()
        phis = list()
        ncomp = len(state.phases[j].x)
        for i in range(ncomp):
            val = vec.copy()
            val[dir] = out.phases[j].x[i]
            x_j.append(val)

            val = vec.copy()
            val[dir] = out.phases[j].phis[i]
            phis.append(val)

        # phase properties. This is important, since densities and enthalpies
        # appear in the PDEs
        v = vec.copy()
        v[dir] = out.phases[j].v
        h = vec.copy()
        h[dir] = out.phases[j].h
        mu = vec.copy()
        mu[dir] = out.phases[j].mu
        kappa = vec.copy()
        kappa[dir] = out.phases[j].kappa

        # TODO parse derivatives if for some reason required on boundary
        phase_j = ppc.PhaseState(
            h=h,
            v=v,
            phasetype=out.phases[j].phasetype,
            x=np.array(x_j),
            phis=np.array(phis),
            mu=mu,
            kappa=kappa,
            dv=0,
            dh=0,
            dmu=0,
            dkappa=0,
            dphis=np.zeros(ncomp),
        )
        phases.append(phase_j)
    out.y = np.array(y)
    out.sat = np.array(sat)
    out.phases = phases

    return out


class BoundaryConditionsCompositionalFlow(
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
):
    """Mixin treating boundary conditions for the compositional flow.

    Atop of inheriting the treatment for single phase flow (which is exploited for the
    total mass balance) and the energy balance (total energy balance), this class has
    a treatment for BC for component mass balances, and secondary expressions.

    **If model has equilibrium conditions and a flash instance:**

    This class provides the :meth:`boundary_flash` method which computes the flash on
    the Dirichlet boundary for pressure, requiring also temperature values there.

    The result is used to populate various secondary expressions and to provide values
    for fractional variables on the boundary. The secondary expressions play a role
    within the mobility terms.

    The boundary flash uses the framework of secondary expressions to set boundary
    values directly.

    Note:
        This method performs the boundary flash in every time-step, only if the modeler
        has explicitely defined if the problem has time-dependent BC
        (see :attr:`has_time_dependent_boundary_equilibrium`).

        The first boundary equilibrium is performed anyways in the solution strategy, to
        cover the case of constant BC.

        This is for performance reasons.

    **If the model has no equilibrium conditions and flash defined:**

    The user must provide values for fractional variables and secondary expressions on
    the Dirichlet boundary, which appear in the weights for fluxes
    (only advective as of now). They are required for upwinding.

    For convenience, they can be stored in :attr:`boundary_fluid_state`.

    **Values which always need to be provided:**

    Overall fractions of components on the Dirichlet boundary (to compute the boundary
    flash), and fractional mobility values on the Neumann boundary.
    This somewhat unusual requirement for the mobility is due to the framework accessing
    the Neumann flux from single phase flow as BC, and then weighing it with the
    provided values in the flux terms in component mass and energy balance.

    Important:
        The Dirichlet boundary is solely defined by
        :meth:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow.
        bc_type_darcy_flux`.

        This class goes as far as overriding the parent method
        :meth:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow.
        bc_type_fluid_flux` and calling the ``bc_type_darcy_flux`` instead to provide
        a consistent representation of Dirichlet and Neumann boundary.

        This is also done for
        :meth:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance.
        bc_type_enthalpy_flux`.

        It is an inconsistency which probably should be removed in the parent classes.
        TODO

    Important:
        The boundary flash requires Dirichlet data for temperature as well, independent
        of how the boundary type for the Fourier flux is defined.
        This class implements a p-T flash on the Dirichlet boundary for the advective
        flux.

    Notes:

        1. BC values for the fluid enthalpy (which is an independent variable) are not
           required, since it only appears in the accumulation term of the energy
           balance.

        2. BC values for thermal conductivity are also not required, because of the
           nature of the Fourier flux.

        3. Derivative values of secondary expressions are not required on the boundary
           as of now, since they do not appear anywhere.

    """

    boundary_fluid_state: dict[pp.BoundaryGrid, ppc.FluidState] = dict()
    """Contains per boundary grid the fluid state, where the user can define boundary
    conditions on each boundary grid.

    This data is used if no boundary flash needs to be performed and the user has to
    provide custom data.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    flash: ppc.Flash
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""

    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    eliminate_reference_phase: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    has_extended_fractions: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    has_time_dependent_boundary_equilibrium: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by:class:`SolutionStrategyCompositionalFlow`."""

    _phase_fraction_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _extended_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""

    def bc_data_fractional_mobility_key(self, component: ppc.Component) -> str:
        """
        Parameters:
            component: A fluid component in the mixture with a mass balance equation.

        Returns:
            The key for storing (time-dependent) Neumann BC data for the advective
            weight (fractional mobility) in the advective flux in the components mass
            balance.

        """
        return f"{self.bc_data_fluid_flux_key}_fraction_{component.name}"

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of the darcy flux for consistency reasons."""
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of the darcy flux for consistency reasons."""
        return self.bc_type_darcy_flux(sd)

    def update_all_boundary_conditions(self) -> None:
        """Additionally to the parent method, this updates the values of all quantities
        which appear in the advective weights in component mass and energy balance.

        On the Neumann boundary, it updates the values of the weights of the advective
        flux.

        On the Dirichlet boundary, the boundary flash is performed if the BC are
        time-dependent.
        Otherwise it uses various ``bc_values_*`` functions to update

        - phase fractions
        - saturations
        - extended fractions
        - phase densities
        - phase enthalpies
        - phase viscosities

        While fractions are always updated the "conventional" way using the boundary
        mixin, phase properties (secondary expressions) are updated directly using
        the framework for secondary expressions if the flash was performed.

        """
        super().update_all_boundary_conditions()

        # values of fractional mobilities on Neumann boundaries.
        for component in self.fluid_mixture.components:
            # Skip if mass balance for reference component is eliminated.
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue

            bc_vals = partial(self.bc_values_fractional_mobility, component)
            bc_vals = cast(Callable[[pp.BoundaryGrid], np.ndarray], bc_vals)

            self.update_boundary_condition(
                name=self.bc_data_fractional_mobility_key(component),
                function=bc_vals,
            )

        # values for weight in enthalpy flux on Neumann boundaries
        self.update_boundary_condition(
            name=self.bc_data_enthalpy_flux_key,
            function=self.bc_values_enthalpy_flux_weight,
        )

        ### Secondary expressions
        # Update of BC values on the Dirichlet-boundary, in case of a time-dependent
        # BC. NOTE due to computational cost, this is only done if time-dependence is
        # indicated, otherwise it is done once in the solution strategy in the beginning
        if self.has_time_dependent_boundary_equilibrium:
            self.boundary_flash()
        # if no equilibrium conditions/flash defined, use user-provided methods
        elif (
            not self.has_time_dependent_boundary_equilibrium
            and self.equilibrium_type is None
        ):
            # BC data for advective weights in balance equations, and fractions
            for phase in self.fluid_mixture.phases:
                # phase properties which appear in mobilities
                rho_bc = partial(self.bc_values_phase_density, phase)
                rho_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], rho_bc)
                self.update_boundary_condition(phase.density.name, rho_bc)

                h_bc = partial(self.bc_values_phase_enthalpy, phase)
                h_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], h_bc)
                self.update_boundary_condition(phase.enthalpy.name, h_bc)

                mu_bc = partial(self.bc_values_phase_viscosity, phase)
                mu_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], mu_bc)
                self.update_boundary_condition(phase.viscosity.name, mu_bc)

                # NOTE no fugacity or conductivity on boundary
        # else we alert the user that the model might be incomplete
        else:
            raise NotImplementedError("Could not resolve strategy for BC update.")

        ### Updating fractions on boundary
        # BC data for fractions are updated using the boundary fluid state and the
        # conventional approach, since they are primary expressions (unknowns)
        for phase in self.fluid_mixture.phases:
            # phase fractions and saturations
            if self.eliminate_reference_phase:
                pass
            else:
                y_name = self._phase_fraction_variable(phase)
                y_bc = partial(self.bc_values_phase_fraction, phase)
                y_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], y_bc)
                self.update_boundary_condition(y_name, y_bc)

                s_name = self._saturation_variable(phase)
                s_bc = partial(self.bc_values_saturation, phase)
                s_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], s_bc)
                self.update_boundary_condition(s_name, s_bc)

            # extended fractions
            if self.has_extended_fractions:
                for comp in phase.components:
                    x_name = self._extended_fraction_variable(comp, phase)
                    x_bc = partial(self.bc_values_extended_fraction, component, phase)
                    x_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], x_bc)
                    self.update_boundary_condition(x_name, x_bc)

    def boundary_flash(self) -> None:
        """If a flash procedure is provided, this method performs
        the p-T flash on the Dirichlet boundary, where pressure and temperature are
        positive.

        The values are stored to represent the secondary expressions on the boundary for
        e.g., upwinding.

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        Important:
            If p or T are negative, the respective secondary expressions are stored as
            zero. Might have some implications for the simulation in weird cases.

        Raises:
            AssertionError: If no flash is provided.
            ValueError: If temperature or feed fractions are not positive where
                required.
            ValueError: If the flash did not succeed everywhere.

        """
        assert hasattr(self, "flash"), "Flash instance not defined."
        for sd in self.mdg.subdomains():
            bg = self.mdg.subdomain_to_boundary_grid(sd)
            if bg is None:  # skipping 0-D grids, or if grid not in mdg
                continue
            bg = cast(pp.BoundaryGrid, bg)

            # indexation on boundary grid
            # equilibrium is computable where pressure is given and positive
            dbc = self.bc_type_darcy_flux(sd).is_dir
            p = self.pressure([bg]).value(self.equation_system)
            p = self.bc_values_pressure(bg)
            idx = np.logical_and(dbc, p > 0)

            # Skip if not required anywhere
            if not np.any(idx):
                continue

            # BC consistency checks ensure that z, T are non-trivial where p is
            # non-trivial
            T = self.bc_values_temperature(bg)
            feed = [
                self.bc_values_overall_fraction(comp, bg)
                for comp in self.fluid_mixture.components
            ]

            boundary_state, success, _ = self.flash.flash(
                z=[z[idx] for z in feed],
                p=p[idx],
                T=T[idx],
                parameters=self.flash_params,
            )

            if not np.all(success):
                raise ValueError("Boundary flash did not succeed.")

            # Broadcast values into proper size for each boundary grid
            boundary_state = _prolong_boundary_state(boundary_state, bg.num_cells, idx)

            # storing state for fraction updates
            self.boundary_fluid_state[bg] = boundary_state

            # update the boundary values of secondary expressions using their methods
            for j, phase in enumerate(self.fluid_mixture.phases):
                phase.density.progress_value_in_time_on_grid(
                    boundary_state.phases[j].rho, bg
                )
                phase.volume.progress_value_in_time_on_grid(
                    boundary_state.phases[j].v, bg
                )
                phase.enthalpy.progress_value_in_time_on_grid(
                    boundary_state.phases[j].h, bg
                )
                phase.viscosity.progress_value_in_time_on_grid(
                    boundary_state.phases[j].mu, bg
                )
                # NOTE no conductivity or fugacity on boundary

    def check_bc_consistency(self) -> None:
        """Performs checks of the set boundary condition values and types to ensure
        consistency.

        This method implements certain requirements to how the model is configured,
        aiming for complete mathematical consistency.

        Checks:

        1. Dirichlet/Neumann/Robin faces must be consistently defined for all advective
           fluxes.
        2. Dirichlet faces for the advective flux (where temperature must also be
           defined on the boundary to compute the advective weights), must be contained
           in the Dirichlet faces for the Fourier flux (important!)
        3. Fractional mobilities on boundaries must add up to 1 or be zero.
        4. Overall fractions on boundaries must add up to 1 or be zero.
        5. Pressure, temperature and overall fraction values on Dirichlet boundary must
           be strictly positive or zero on the same faces (boundary flash).

        While most of the checks are straight forward, check 2 needs an explanation:

        On Dirichlet faces for the Darcy flux, pressure must be provided, and a way
        to compute the weights in the advective flux.
        Since in the non-isothermal setting the fluid properties depend also on T,
        the user must provide T values there as well.

        Also, the conductive flux in the energy equation on this part of the boundary
        must hence not be given by Neumann conditions, but be consistent since T is
        provided.

        To summarize, on the boundary with pressure defined, temperature must be defined
        as well to compute the propertie. And since temperature is given, the Fourier
        flux cannot have Neumann values on those faces.

        Note though, that temperature can be defined on faces, where there are no
        Dirichlet conditions for pressure. This gives flexibility to define heated
        boundaries with no mass flux, in terms of both temperature and heat flux.

        """

        # Checking consistency of BC type definitions of the advective flux
        # The parent classes have BC types for Darcy flux, fluid flux and enthalpy flux
        # in separate methods. But they must be all the same, since they are due to mass
        # entering the system.
        for sd in self.mdg.subdomains():
            bc_darcy = self.bc_type_darcy_flux(sd)
            bc_fluid = self.bc_type_fluid_flux(sd)
            bc_enthalpy = self.bc_type_fluid_flux(sd)

            # checking definition of Dirichlet boundary
            # performed by summing the boolean arrays, and asserting that only 0 or a
            # single number is in the resulting array, and not multiple non-zero numbers
            check_dir = bc_darcy.is_dir + bc_fluid.is_dir + bc_enthalpy.is_dir
            assert len(set(check_dir)) == 2, (
                "Inconsistent number of Dirichlet boundary faces defined for advective"
                + f" flux on subdomain {sd}."
            )

            # same must hold for Neumann and Robin BC
            check_neu = bc_darcy.is_neu + bc_fluid.is_neu + bc_enthalpy.is_neu
            assert len(set(check_neu)) == 2, (
                "Inconsistent number of Neumann boundary faces defined for advective"
                + f" flux on subdomain {sd}."
            )
            check_rob = bc_darcy.is_rob + bc_fluid.is_rob + bc_enthalpy.is_rob
            assert len(set(check_rob)) == 2, (
                "Inconsistent number of Neumann boundary faces defined for advective"
                + f" flux on subdomain {sd}."
            )

            # check if Dirichlet faces for Fourier flux are part of the Dirichlet
            # faces for the Fourier flux.
            # This is important because Temperature must be provided on the Dirichlet
            # boundary for the advective flux (to compute the influxing mass)
            # Temperature though, can be defined on faces where there is no advective
            # flux (hot boundary). This gives some flexibility for model set-ups.
            if np.any(bc_darcy.is_dir):
                bc_fourier = self.bc_type_fourier_flux(sd)
                Fourier_consistent_with_Darcy = bc_fourier.is_dir[bc_darcy.is_dir]
                assert np.all(Fourier_consistent_with_Darcy), (
                    "Darcy Dirichlet faces must be contained in Fourier Dirichlet faces"
                    + f" on subdomain {sd}."
                )

            ### checks involving boundary grids as arguments
            bg = self.mdg.subdomain_to_boundary_grid(sd)

            # check if fractional mobilities on Neumann faces add up to 1
            f = np.zeros(bg.num_cells)
            for comp in self.fluid_mixture.components:
                f_i = self.bc_values_fractional_mobility(comp, bg)
                # check positivitiy of fractional mobilities
                assert np.all(f_i >= 0.0), (
                    f"Fractional mobilities of component {comp} on boundary {bg} must"
                    + " be non-negative."
                )
                f += f_i
            # f must be either zero or 1
            assert np.allclose(
                f[f > 0], 1.0
            ), f"Sum of fractional mobilities must be either 1 or 0 on boundary {bg}."

            # check if overall fractions on boundaries add up to one
            z_sum = np.zeros(bg.num_cells)
            for comp in self.fluid_mixture.components:
                z_i = self.bc_values_overall_fraction(comp, bg)
                # check positivitiy of fractional mobilities
                assert np.all(z_i >= 0.0), (
                    f"Overall fraction of component {comp} on boundary {bg} must"
                    + " be non-negative."
                )
                z_sum += z_i
            # z_sum must be either 0 or 1
            assert np.allclose(
                z_sum[z_sum > 0], 1.0
            ), f"Sum of overall fractions must be either 1 or 0 on boundary {bg}."

            # Check if T and z are non-trivial where p is non-trivial, since this
            # is the part of the boundary where fluid properties are computed and should
            # be non-trivial
            p_bc = self.bc_values_pressure(bg)
            T_bc = self.bc_values_temperature(bg)

            assert np.all(T_bc[p_bc > 0.0] > 0), (
                "Temperature values must be positive where pressure is positive on"
                + f" boundary {bg}"
            )
            assert np.allclose(z_sum[p_bc > 0.0], 1), (
                "Overall fractions must be provided where pressure is positive on"
                + f" boundary {bg}."
            )

    ### BC which always need to be provided

    def bc_values_fractional_mobility(
        self, component: ppc.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC value of the fractional mobility on the Neumann boundary.

        Required to weigh the total mass influx for component mass balances, when
        Neumann conditions are defined (see how
        :meth:`ComponentMassBalanceEquations.advective_weight_enthalpy_flux` is defined).

        Parameters:
            component: A fluid component in the mixture with a mass balance equation.
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the fractional mobility on Neumann boundaries.

        """
        return np.zeros(bg.num_cells)

    def bc_values_enthalpy_flux_weight(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """BC value of the weight in the enthalpy flux on the Neumann boundary.

        Required to weigh the total mass influx for component mass balances, when
        Neumann conditions are defined (see how
        :meth:`TotalEnergyBalance_h.advective_weight_component_flux` is defined).

        Parameters:
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the enthalpy flux weight on Neumann boundaries.

        """
        return np.zeros(bg.num_cells)

    def bc_values_overall_fraction(
        self, component: ppc.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component.

        Required together with BC for pressure and temperature to perform the boundary
        flash on the Dirichlet boundary with positive p and T.

        Parameters:
            component: A component in the fluid mixture.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(bg.num_cells)

    ### BC which need to be provided in case no equilibrium calculations are included

    def bc_values_saturation(self, phase: ppc.Phase, bg: pp.BoundaryGrid) -> np.ndarray:
        """BC values for saturation on the Dirichlet boundary, for models which do not
        have equilibrium calculations.

        Parameters:
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            j = phases.index(phase)
            vals = self.boundary_fluid_state[bg].sat[j]
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required saturation values for phase {phase.name} on"
                + f" boundary {bg}."
            )
        return vals

    def bc_values_phase_fraction(
        self, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for molar phase fractions on the Dirichlet boundary, for models
        which do not have equilibrium calculations.

        Parameters:
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            j = phases.index(phase)
            vals = self.boundary_fluid_state[bg].y[j]
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required phase fraction values for phase {phase.name}"
                + f" on boundary {bg}."
            )
        return vals

    def bc_values_extended_fraction(
        self, component: ppc.Component, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for extended fractions of a component in a phase on the Dirichlet
        boundary, for models which do not have equilibrium calculations.

        Parameters:
            component: A component in the ``phase``.
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            comps = [c for c in phase.components]
            j = phases.index(phase)
            i = comps.index(component)
            vals = self.boundary_fluid_state[bg].phases[j].x[i]
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required ext. fraction values for component"
                + f" {component.name} in phase {phase.name} on boundary {bg}."
            )
        return vals

    def bc_values_phase_density(
        self, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the density of a phase on the Dirichlet boundary, for models
        which do not have equilibrium calculations.

        This value is required since in the general CF framework this is a secondary
        expression with values provided by the user.

        Parameters:
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            j = phases.index(phase)
            vals = self.boundary_fluid_state[bg].phases[j].rho
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required phase density values for phase {phase.name}"
                + f" on boundary {bg}."
            )
        return vals

    def bc_values_phase_enthalpy(
        self, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the enthalpy of a phase on the Dirichlet boundary, for models
        which do not have equilibrium calculations.

        This value is required since in the general CF framework this is a secondary
        expression with values provided by the user.

        Parameters:
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            j = phases.index(phase)
            vals = self.boundary_fluid_state[bg].phases[j].h
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required phase enthalpy values for phase {phase.name}"
                + f" on boundary {bg}."
            )
        return vals

    def bc_values_phase_viscosity(
        self, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the viscosity of a phase on the Dirichlet boundary, for models
        which do not have equilibrium calculations.

        This value is required since in the general CF framework this is a secondary
        expression with values provided by the user.

        Parameters:
            phase: A phase in the fluid mixture.
            bg: A boundary grid in the domain.

        Raises:
            AssertionError: If not exactly ``bg.num_cells`` values are stored in
            :attr:`boundary_fluid_state`.

        Returns:
            The values stored in :attr:`boundary_fluid_state, if available.
            Returns zeros otherwise.

        """
        if bg not in self.boundary_fluid_state:
            vals = np.zeros(bg.num_cells)
        else:
            phases = [p for p in self.fluid_mixture.phases]
            j = phases.index(phase)
            vals = self.boundary_fluid_state[bg].phases[j].mu
            assert vals.shape == (bg.num_cells,), (
                f"Mismatch in required phase viscosity values for phase {phase.name}"
                + f" on boundary {bg}."
            )
        return vals


class InitialConditionsCompositionalFlow:
    """Class for setting the initial values in a compositional flow model and computing
    the initial equilibrium.

    This mixin is introduced because of the complexity of the framework, guiding the
    user through what is required and dissalowing a "messing" with the order of methods
    executed in the model set-up.

    All method herein are part of the routine
    :meth:`SolutionStrategyCompositionalFlow.initial_conditions`.

    The basic initialization assumes that initial conditions are given in terms of

    - pressure,
    - temperature,
    - feed fractions,

    and that equilibrium calculations are included in the model.

    I.e., a flash is performed.
    More precisely, it assumes that initial conditions are given for feed fractions,
    pressure and temperature.

    Everything else is computed using the flash algorithm to provide values for
    fractions.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    flash: ppc.Flash
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""

    time_step_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    iterate_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`VariablesCompositionalFlow`."""

    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    eliminate_reference_phase: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    def set_initial_values(self) -> None:
        """Method to set initial values.

        The base method covers pressure, temperature and overall fractions.

        It stors values at the time step zero (current time).

        Override for initializing more variable values

        """

        for sd in self.mdg.subdomains():
            # setting pressure and temperature per domain
            p = self.intial_pressure(sd)
            T = self.initial_temperature(sd)

            self.equation_system.set_variable_values(
                p, self.pressure([sd]), time_step_index=0
            )
            self.equation_system.set_variable_values(
                T, self.temperature([sd]), time_step_index=0
            )

            # Setting overall fractions
            for comp in self.fluid_mixture.components:
                if (
                    comp == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue

                z_i = self.initial_overall_fraction(comp, sd)
                self.equation_system.set_variable_values(
                    z_i, comp.fraction([sd]), time_step_index=0
                )

    def initial_flash(self) -> None:
        """Method called by the solution strategy after the initial values are set and
        a local equilibrium is defined.

        It performs a p-T flash to obtain values for:

        - fluid enthalpy (see :meth:`VariablesCompositionalFlow.enthalpy`)
        - saturations
        - molar fraction for phases
        - extended fractions.

        Ergo, the user does not need to set them explicitely, if equilibrium conditions
        are modelled.

        Values are stored in the time stepth 0 (current time).

        """

        for sd in self.mdg.subdomains():
            # pressure, temperature and overall fractions
            p = self.intial_pressure(sd)
            T = self.intial_pressure(sd)
            z = [
                self.initial_overall_fraction(comp, sd)
                for comp in self.fluid_mixture.components
            ]

            # computing initial equilibrium
            state, success, _ = self.flash.flash(
                z, p=p, T=T, parameters=self.flash_params
            )

            if not np.all(success):
                raise ValueError(f"Initial equilibriam not successful on grid {sd}")

            # setting initial values for enthalpy
            # NOTE that in the initialization, h is dependent compared to p, T, z
            self.equation_system.set_variable_values(
                state.h, self.enthalpy([sd]), time_step_index=0
            )

            # setting initial values for all fractional variables and phase properties
            for j, phase in enumerate(self.fluid_mixture.phases):
                # phase fractions and saturations
                if (
                    phase == self.fluid_mixture.reference_phase
                    and self.eliminate_reference_phase
                ):
                    pass  # y and s of ref phase are dependent operators
                else:
                    self.equation_system.set_variable_values(
                        state.y[j], phase.fraction([sd]), time_step_index=0
                    )
                    self.equation_system.set_variable_values(
                        state.sat[j], phase.saturation([sd]), time_step_index=0
                    )
                # extended fractions
                for k, comp in enumerate(phase.components):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[k],
                        phase.fraction_of[comp]([sd]),
                        time_step_index=0,
                    )

                # phase properties and their derivatives
                phase.density.progress_value_in_time_on_grid(state.phases[j].rho, sd)
                phase.volume.progress_value_in_time_on_grid(state.phases[j].v, sd)
                phase.enthalpy.progress_value_in_time_on_grid(state.phases[j].h, sd)
                phase.viscosity.progress_value_in_time_on_grid(state.phases[j].mu, sd)
                phase.conductivity.progress_value_in_time_on_grid(
                    state.phases[j].kappa, sd
                )

                phase.density.progress_derivatives_in_time_on_grid(
                    state.phases[j].drho, sd
                )
                phase.volume.progress_derivatives_in_time_on_grid(
                    state.phases[j].dv, sd
                )
                phase.enthalpy.progress_derivatives_in_time_on_grid(
                    state.phases[j].dh, sd
                )
                phase.viscosity.progress_derivatives_in_time_on_grid(
                    state.phases[j].dmu, sd
                )
                phase.conductivity.progress_derivatives_in_time_on_grid(
                    state.phases[j].dkappa, sd
                )

                # fugacities
                for k, comp in enumerate(phase.components):
                    phase.fugacity_of[comp].progress_value_in_time_on_grid(
                        state.phases[j].phis[k], sd
                    )
                    phase.fugacity_of[comp].progress_derivatives_in_time_on_grid(
                        state.phases[j].dphis[k], sd
                    )

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """
        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain.

        """
        raise NotImplementedError("Initial pressure not provided")

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        """
        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain.

        """
        raise NotImplementedError("Initial temperature not provided.")

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            component: A component in the fluid mixture with an independent feed
                fraction.
            sd: A subdomain in the md-grid.

        Returns:
            The initial overall fraction values for a component on a subdomain.

        """
        raise NotImplementedError("Initial overall fractions not provided.")


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

    advected_enthalpy_discretization: Callable[[Sequence[pp.Grid]], pp.ad.UpwindAd]
    """Provided by :class:`MobilityCF`."""
    interface_advected_enthalpy_discretization: Callable[
        [Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    fourier_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    fractional_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`MobilityCF`."""
    interface_fractional_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`MobilityCF`."""

    overall_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    solute_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    phase_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    saturation_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    extended_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""

    equilibriate_fluid: Callable[
        [Optional[np.ndarray]], tuple[ppc.FluidState, np.ndarray]
    ]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    postprocess_failures: Callable[[ppc.FluidState, np.ndarray], ppc.FluidState]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    set_up_flasher: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FlashMixin`."""
    boundary_flash: Callable[[], None]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`"""
    create_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    assign_thermodynamic_properties_to_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    set_initial_values: Callable[[], None]
    """Provided by :class:`InitialConditionsCompositionalFlow`."""
    initial_flash: Callable[[], None]
    """Provided by :class:`InitialConditionsCompositionalFlow`."""
    check_bc_consistency: Callable[[], None]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`"""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations = list()

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid mixture."""

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

        self.has_time_dependent_boundary_equilibrium: bool = params.get(
            "has_time_dependent_boundary_equilibrium", False
        )
        """A bool indicating whether Dirichlet BC for pressure, temperature or
        feed fractions are time-dependent.

        If True, the boundary equilibrium will be re-computed at the beginning of every
        time step. This is required to provide e.g., values of the advective weights on
        the boundary for upwinding.

        Cannot be True if :attr:`equilibrium_type` is set to None (and hence no flash
        method was introduced).

        """

        # Input validation for set-up
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

        if (
            self.equilibrium_type is None
            and self.has_time_dependent_boundary_equilibrium
        ):
            raise ValueError(
                f"Conflicting model set-up: Time-dependent boundary equilibrium"
                + f" requested but no equilibrium type defined."
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

    @property
    def time_step_indices(self) -> np.ndarray:
        """Indices for storing time step solutions.

        The CF framework stores per default the current solution (0) and the previous
        solution (1)

        Returns:
            An array of the indices of which time step solutions will be stored.

        """
        return np.array([0, 1])

    @property
    def time_step_depth(self) -> int:
        """
        Returns:
            :meth:`time_step_indices` - 1, the number of additionally stored time step
            values, besides the current one.
        """
        return len(self.time_step_indices) - 1

    @property
    def iterate_depth(self) -> int:
        """
        Returns:
            :meth:`iterate_indices` - 1, the number of additionally stored iterate
            values, besides the current one.
        """
        return len(self.iterate_indices) - 1

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

        # If equilibrium defined, set the flash clsas and calculate initial equilibria
        # on subdomains and boundaries
        if self.equilibrium_type is not None:
            self.set_up_flasher()
            self.initial_flash()
            self.boundary_flash()
        self.initialize_timestep_and_iterate_indices()
        self.check_bc_consistency()

        self.reset_state_from_file()  # TODO check if this is in conflict with init vals
        self.set_equations()
        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()
        self.save_data_time_step()

    def initialize_timestep_and_iterate_indices(self) -> None:
        """Copies the initial values to all time step and iterate indices.

        This is done after the initialization, to populate the data dictionaries.
        It is performed for all variables, and all secondary expressions.

        This property assumes the the initial value is stored at the current time step
        index (0) for every quantity.

        Note:

            1. Boundary values are copied only on time indices.
            2. Derivative values are not copied to other time indices, since they are
               not accessed by this solution strategy.

        Can be customized by the user.

        """

        # updating variable values from current time step, to all previous and iterate
        val = self.equation_system.get_variable_values(time_step_index=0)
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )
        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        # copying the current value of secondary expressions to all indices
        for phase in self.fluid_mixture.phases:
            # phase properties and their derivatives on each subdomain
            rho_j = phase.density.subdomain_values
            v_j = phase.volume.subdomain_values
            h_j = phase.enthalpy.subdomain_values
            mu_j = phase.viscosity.subdomain_values
            kappa_j = phase.conductivity.subdomain_values

            d_rho_j = phase.density.subdomain_derivatives
            d_v_j = phase.volume.subdomain_derivatives
            d_h_j = phase.enthalpy.subdomain_derivatives
            d_mu_j = phase.viscosity.subdomain_derivatives
            d_kappa_j = phase.conductivity.subdomain_derivatives

            # all properties have iterate values, use framework from sec. expressions
            # to push back values
            for _ in self.iterate_indices:
                phase.density.subdomain_values = rho_j
                phase.volume.subdomain_values = v_j
                phase.enthalpy.subdomain_values = h_j
                phase.viscosity.subdomain_values = mu_j
                phase.conductivity.subdomain_values = kappa_j

                phase.density.subdomain_derivatives = d_rho_j
                phase.volume.subdomain_derivatives = d_v_j
                phase.enthalpy.subdomain_derivatives = d_h_j
                phase.viscosity.subdomain_derivatives = d_mu_j
                phase.conductivity.subdomain_derivatives = d_kappa_j

            # all properties have time step values, progress sec. exp. in time
            for _ in self.time_step_indices:
                phase.density.progress_value_in_time_on_subdomains(rho_j)
                phase.volume.progress_value_in_time_on_subdomains(v_j)
                phase.enthalpy.progress_value_in_time_on_subdomains(h_j)
                phase.viscosity.progress_value_in_time_on_subdomains(mu_j)
                phase.conductivity.progress_value_in_time_on_subdomains(kappa_j)

                # phase.density.progress_derivatives_in_time_on_subdomains(d_rho_j)
                # phase.volume.progress_derivatives_in_time_on_subdomains(d_v_j)
                # phase.enthalpy.progress_derivatives_in_time_on_subdomains(d_h_j)
                # phase.viscosity.progress_derivatives_in_time_on_subdomains(d_mu_j)
                # phase.conductivity.progress_derivatives_in_time_on_subdomains(d_kappa_j)

            # fugacity coeffs
            # NOTE as of now they have only iterate values, but it is here done for time
            # as well to cover everything
            for comp in phase.components:
                phi = phase.fugacity_of[comp].subdomain_values
                d_phi = phase.fugacity_of[comp].subdomain_derivatives

                for _ in self.iterate_indices:
                    phase.fugacity_of[comp].subdomain_values = phi
                    phase.fugacity_of[comp].subdomain_derivatives = d_phi
                for _ in self.time_step_indices:
                    phase.fugacity_of[comp].progress_value_in_time_on_subdomains(phi)
                    # phase.fugacity_of[comp].progress_derivatives_in_time_on_subdomains(d_phi)

            # properties have also (time-dependent) values on boundaries
            bc_rho_j = phase.density.boundary_values
            bc_v_j = phase.volume.boundary_values
            bc_h_j = phase.enthalpy.boundary_values
            bc_mu_j = phase.viscosity.boundary_values
            bc_kappa_j = phase.conductivity.boundary_values
            # NOTE the different usage of time progress for subdomain and boundary vals
            for _ in self.time_step_indices:
                phase.density.boundary_values = bc_rho_j
                phase.volume.boundary_values = bc_v_j
                phase.enthalpy.boundary_values = bc_h_j
                phase.viscosity.boundary_values = bc_mu_j
                phase.conductivity.boundary_values = bc_kappa_j

    def initial_condition(self) -> None:
        """Atop the parent methods, this method computes the initial equilibrium and the
        boundary equilibrium, if a local equilibrium is defined."""
        super().initial_condition()

        self.set_initial_values()

    def set_discretization_parameters(self) -> None:
        """Overrides the BC types for all advective fluxes and their weights to be
        consistent with the Darcy flux."""
        # For compatibility with inheritance
        super().set_discretization_parameters()

        # Use the same BC type for all advective fluxes
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_darcy_flux(sd),
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {
                    "bc": self.bc_type_darcy_flux(sd),
                },
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
            self.advected_enthalpy_discretization(subdomains).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_advected_enthalpy_discretization(interfaces).flux,
        )

        # Upwinding of mobilities in component balance equations
        # NOTE I think there should be only one discretization because of the fractional
        # flow formulation
        for component in self.fluid_mixture.components:
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue

            self.add_nonlinear_discretization(
                self.fractional_mobility_discretization(component, subdomains).upwind,
            )
            self.add_nonlinear_discretization(
                self.interface_fractional_mobility_discretization(
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
        vars_s = self.saturation_variables
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
            vars_x = self.extended_fraction_variables
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

    def update_thermodynamic_properties_iteratively(
        self, fluid_state: Optional[ppc.FluidState] = None
    ) -> None:
        """Method to update various thermodynamic properties of present phases, on all
        subdomains **in the iterative sense**.

        I.e., it stores the results in secondary properties in the current iterate value
        using the secondary expression class (which includes a shift of other values).

        If ``fluid_state`` is None, evaluates pressure, temperature and fractions,
        and computes the properties using using
        :meth:`~porepy.composite.base.Phase.compute_properties`

        If a ``fluid_state`` is given, uses respective values. This is the case if it
        was computed using the flash for example.

        """
        phase_states: list[ppc.PhaseState] = list()

        # This is the case if no flash was performed.
        # Evaluate the properties using what the user has provided
        if fluid_state is None:
            subdomains = self.mdg.subdomains()

            p = self.equation_system.get_variable_values([self.pressure_variable])
            T = self.equation_system.get_variable_values([self.temperature_variable])

            for phase in self.fluid_mixture.phases:
                xn = [
                    phase.partial_fraction_of[comp](subdomains).value(
                        self.equation_system
                    )
                    for comp in phase.components
                ]

                phase_states.append(phase.compute_properties(p, T, xn))
        # otherwise assume this is the flash result with a complete description
        else:
            phase_states = fluid_state.phases

        # updating the phase properties, values and derivatives in the iterative sense
        for phase, state in zip(self.fluid_mixture.phases, phase_states):
            phase.density.subdomain_values = state.rho
            phase.volume.subdomain_values = state.v
            phase.enthalpy.subdomain_values = state.h
            phase.viscosity.subdomain_values = state.mu
            phase.conductivity.subdomain_values = state.kappa

            # extend derivatives from partial to extended fractions, if any
            if self.has_extended_fractions:
                x = np.array(
                    [
                        self.equation_system.get_variable_values(
                            [phase.fraction_of[comp](subdomains)]
                        )
                        for comp in phase.components
                    ]
                )

                phase.density.subdomain_derivatives = _extend(state.drho, x)
                phase.volume.subdomain_derivatives = _extend(state.dv, x)
                phase.enthalpy.subdomain_derivatives = _extend(state.dh, x)
                phase.viscosity.subdomain_derivatives = _extend(state.dmu, x)
                phase.conductivity.subdomain_derivatives = _extend(state.dkappa, x)
            else:
                phase.density.subdomain_derivatives = state.drho
                phase.volume.subdomain_derivatives = state.dv
                phase.enthalpy.subdomain_derivatives = state.dh
                phase.viscosity.subdomain_derivatives = state.dmu
                phase.conductivity.subdomain_derivatives = state.dkappa

            for k, comp in enumerate(phase.components):
                phase.fugacity_of[comp].subdomain_values = state.phis[k]
                if self.has_extended_fractions:
                    phase.fugacity_of[comp].subdomain_derivatives = _extend(
                        state.dphis[k], x
                    )
                else:
                    phase.fugacity_of[comp].subdomain_derivatives = state.dphis[k]

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

    def update_discretization_parameters(self) -> None:
        """Method called before non-linear iterations to update discretization
        parameters.

        The base method assembles the conductivity tensor using the fluid mixture
        conductivity, and the diffusive tensor for the Darcy flux.

        It does the same for the pressure equation, where total mobility is evaluated
        and multiplied with absolute permeability.

        Might need more work when using DarcysLawAd and FouriersLawAd.

        """
        # NOTE The non-linear MPFA discretization for the Conductive flux in the heat
        # equation and the diffusive flux in the pressure equation are missing
        # derivatives w.r.t. their dependencies.. Jacobian is NOT exact.
        # NOTE this is critical if total mobility is formulated as an auxiliary variable
        for sd, data in self.mdg.subdomains(return_data=True):
            data[pp.PARAMETERS][self.fourier_keyword].update(
                {
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.thermal_conductivity([sd]),
                        self.fluid.thermal_conductivity(),
                    )
                }
            )
            data[pp.PARAMETERS][self.darcy_keyword].update(
                {
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd, self.total_mobility([sd]), self.solid.permeability()
                    )
                }
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

    def progress_thermodynamic_properties_in_time(self) -> None:
        """Updates the values of phase properties on subdomains in time, using the
        current iterate value, and the framework for secondary expressions.

        Note:
            The derivatives are not updated in time, since not required here.

        """

        for phase in self.fluid_mixture.phases:
            # progress values in time
            phase.density.progress_value_in_time_on_subdomains(
                phase.density.subdomain_values
            )
            phase.volume.progress_value_in_time_on_subdomains(
                phase.volume.subdomain_values
            )
            phase.enthalpy.progress_value_in_time_on_subdomains(
                phase.enthalpy.subdomain_values
            )
            phase.viscosity.progress_value_in_time_on_subdomains(
                phase.viscosity.subdomain_values
            )
            phase.conductivity.progress_value_in_time_on_subdomains(
                phase.conductivity.subdomain_values
            )

            for comp in phase.components:
                phase.fugacity_of[comp].progress_value_in_time_on_subdomains(
                    phase.fugacity_of[comp].subdomain_values
                )

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

        self.update_thermodynamic_properties_iteratively(fluid_state)

        # After updating the fluid properties, update discretizations
        self.update_discretizations()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Expands the Schur complement using ``solution_vector``, to include secondary
        variables."""

        global_solution_vector = ...  # TODO
        super().after_nonlinear_iteration(global_solution_vector)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Calls :meth:`update_thermodynamic_properties_in_time`."""
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.progress_thermodynamic_properties_in_time()

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
    InitialConditionsCompositionalFlow,
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
