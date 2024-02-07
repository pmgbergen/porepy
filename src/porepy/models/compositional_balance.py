"""Module defining basic equatios for fluid flow with multiple componens/species."""
from __future__ import annotations

import logging
import time
from functools import partial
from typing import Callable, Literal, Optional, Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc

from . import energy_balance as energy
from . import fluid_mass_balance as mass
from . import mass_and_energy_balance as mass_energy

logger = logging.getLogger(__name__)


class DiscretizationsCompositionalFlow:
    """Mixin class defining which discretization is to be used for the magnitude of
    terms in the compositional flow and transport model.

    The flexibility is required due to the varying mathematical nature of the pressure
    equation, energy balance and transport equations.

    They also need all separate instances of the discretization objects to avoid
    false storage access.

    Every discretization should be handled by this class, except the discretization for
    the Darcy flux and Fourier flux.
    Those are handled by the respective constitutive laws.

    """

    mobility_keyword: str
    """See :attr:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow.
    mobility_keyword`."""

    component_mobility_keyword: Callable[[ppc.Component], str]
    """See :meth:`SolutionStrategyCompositionalFlow.component_mobility_keyword`."""

    enthalpy_keyword: str
    """See :attr:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance.
    enthalpy_keyword`."""

    def total_mobility_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the total fluid mobility in the total mass balance on the
        subdomains.
        (non-linear weight in the Darcy flux in the pressure equation)

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

    def component_mobility_discretization(
        self, component: ppc.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the non-linear weight in the advective flux in the
        component mass balance equations.

        Parameters:
            component: A transportable fluid component in the mixture.
            subdomains: List of subdomains.

        Returns:
            Discretization of the mobility in the component's mass balance.

        """
        return pp.ad.UpwindAd(self.component_mobility_keyword(component), subdomains)

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
        return pp.ad.UpwindCouplingAd(
            self.component_mobility_keyword(component), interfaces
        )


class FouriersLawCF(pp.constitutive_laws.FouriersLaw):
    """Fourier's law in the compositional flow setting.

    Atop the parent class methods, it provides means to compute the conductivity of a
    fluid mixture.

    """

    fluid_mixture: ppc.Mixture
    """See :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """

    bc_data_conductivity_key: str
    """See :attr:`BoundaryConditionsCompositionalFlow.bc_data_conductivity_key`"""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """

    nd: int
    """Number of spatial dimensions."""

    def fluid_thermal_conductivity(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Assembles the fluid conductivity as a sum of phase conductivities weighed
        with saturations.

        Parameters:
            domains: List of subdomains or boundary grids.

        Returns:
            Thermal conductivity of fluid. The returned operator is a scalar defined in
            each cell.

            If ``domains`` is a sequence of boundary grids, it returns a boundary
            operator whose values must be updated.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            assert False, "Fluid thermal conductivity accessed on boundary"
            # return self.create_boundary_operator(  # type: ignore[call-arg]
            #     name=self.bc_data_conductivity_key, domains=domains
            # )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist of subdomains for the fluid conductivity."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)
        conductivity = pp.ad.sum_operator_list(
            [
                phase.conductivity * phase.saturation
                for phase in self.fluid_mixture.phases
            ],
            "fluid_thermal_conductivity",
        )
        return conductivity

    def solid_thermal_conductivity(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Solid thermal conductivity [W / (m K)].

        Parameters:
            domains: List of subdomains or boundary grids.

        Returns:
            Thermal conductivity of the solid skeleton. As of now, this is a constant
            scalar value.

        """
        return pp.ad.Scalar(
            self.solid.thermal_conductivity(), "solid_thermal_conductivity"
        )

    def thermal_conductivity(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        The thermal conductivity is computed as the porosity-weighted average of the
        fluid and solid thermal conductivities. In this implementation, both are
        considered constants, however, if the porosity changes with time, the weighting
        factor will also change.

        Parameters:
            domains: List of subdomains or boundary grids, where the thermal
                conductivity is defined for both fluid mixture and solid.

        Returns:
            Cell-wise conducivity operator.

        """
        phi = self.porosity(domains)
        if isinstance(phi, pp.ad.Scalar):
            size = sum([sd.num_cells for sd in domains])
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        conductivity = phi * self.fluid_thermal_conductivity(domains) - (
            phi - 1.0
        ) * self.solid_thermal_conductivity(domains)
        conductivity = cast(pp.ad.Operator, conductivity)
        conductivity.set_name("total_conductivity")
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
        # TODO model for normal thermal conductivity of the mixture.
        return pp.ad.Scalar(1.0, "normal_thermal_conductivity")

    def vector_source_fourier_flux(
        self, grids: list[pp.Grid] | list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Overriding to eliminate access to non-existent fluid constant mixin."""
        val = 0.0
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        source = pp.wrap_as_dense_ad_array(val, size=size, name="zero_vector_source")
        return source

    def interface_vector_source_fourier_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Overriding to eliminate access to non-existent fluid constant mixin."""
        val = 0.0
        size = int(np.sum([g.num_cells for g in interfaces]))
        source = pp.wrap_as_dense_ad_array(val, size=size, name="zero_vector_source")
        return source


class TotalMassBalanceEquation(mass.MassBalanceEquations):
    """Mixed-dimensional balance of total mass in a fluid mixture.

    Also referred to as *pressure equation*.

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
    """See :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """See :meth:`ConstitutiveLawsCompositionalFlow.relative_permeability`."""

    total_mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :attr:`DiscretizationsCompositionalFlow.total_mobility_discretization`"""

    interface_total_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See
    :attr:`DiscretizationsCompositionalFlow.interface_total_mobility_discretization`"""

    bc_data_total_mobility_key: str
    """See :attr:`BoundaryConditionsCompositionalFlow.bc_data_total_mobility_key`"""

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The inherited method gives another name to the equation, because there are
        multiple mass balances in the compositional setting."""
        eq = super().mass_balance_equation(subdomains)
        eq.set_name("total_mass_balance_equation")
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Analogous to
        :meth:`~porepy.models.fluid_mass_balance.MassBalanceEquations.fluid_mass`
        with the only difference that :attr:`~porepy.composite.base.Mixture.density`,
        which by procedure is defined using the pressure, temperature and fractions on
        all subdomains."""
        mass_density = self.fluid_mixture.density * self.porosity(subdomains)
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name("total_fluid_mass")
        return mass

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """A modified fluid flux, where the advected entity (total mobility) accounts
        for all phases in the mixture.

        See :meth:`total_mobility`.

        It also accounts for custom choices of mobility discretizations (see
        :class:`DiscretizationsCompositionalFlow`).

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_fluid_flux_key, domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist entirely of subdomains for the fluid flux."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        discr = self.total_mobility_discretization(domains)
        weight = self.total_mobility(domains)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.total_mobility,
            neumann_operator=self.fluid_flux,
            bc_type=self.bc_type_fluid_flux,
            name="bc_values_total_fluid_flux",
        )
        flux = self.advective_flux(
            domains, weight, discr, boundary_operator, self.interface_fluid_flux
        )
        flux.set_name("total_fluid_flux")
        return flux

    def total_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Returns the non-linear weight in the advective flux, assuming the mixture is
        defined on all subdomains.

        Parameters:
            subdomains: All subdomains in the md-grid or respective boundary grids.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j(p, T, x_j) k_r(s_j)}{\mu_j},

            which is the advected mass in the total mass balance equation.

            If boundary grids are passed, this returns a boundary operator whose values
            must be updated based on the boundary data for primary variables.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_total_mobility_key, domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist entirely of subdomains for the total mobility."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)
        weight = pp.ad.sum_operator_list(
            [
                phase.density
                / phase.viscosity
                * self.relative_permeability(phase.saturation)
                for phase in self.fluid_mixture.phases
            ],
            "total_mobility",
        )

        return weight

    def interface_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Analogous to :meth:`fluid_flux` this accounts for the modified weight
        :meth:`total_mobility` and a customizable discretization
        (see
        :meth:`DiscretizationsCompositionalFlow.interface_total_mobility_discretization`
        )."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_total_mobility_discretization(interfaces)
        weight = self.total_mobility(subdomains)
        flux = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name("interface_fluid_flux")
        return flux

    def well_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Modifications analogous to :meth:`interface_fluid_flux`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_total_mobility_discretization(interfaces)
        mob_rho = self.total_mobility(subdomains)
        # Call to constitutive law for advective fluxes.
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("well_fluid_flux")
        return flux


class TotalEnergyBalanceEquation(energy.EnergyBalanceEquations):
    """Mixed-dimensional balance of total energy in a fluid mixture, formulated with an
    independent enthalpy variable.

    Balance equation for all subdomains and advective and diffusive fluxes
    (Fourier flux) internally and on all interfaces of codimension one and advection on
    interfaces of codimension two (well-fracture intersections).

    """

    fluid_mixture: ppc.Mixture
    """See :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Enthalpy variable. Normally defined in a mixin instance of
    :class:`~VariablesCompositionalFlow.enthalpy`.

    """

    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """See :meth:`ConstitutiveLawsCompositionalFlow.relative_permeability`."""

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
    """Ad operator representing the advective flux. Normally provided by a mixin
    instance of :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """

    enthalpy_mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`DiscretizationsCompositionalFlow`."""
    interface_enthalpy_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`DiscretizationsCompositionalFlow`."""

    bc_type_enthalpy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See boundary conditions mixin for energy balance."""

    bc_data_enthalpy_flux_key: str
    """See boundary conditions mixin for energy balance."""

    bc_data_enthalpy_mobility_key: str
    """See :attr:`BoundaryConditionsCompositionalFlow.bc_data_enthalpy_mobility_key`."""

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites the parent method to use the fluix mixture density and the primary
        unknown enthalpy."""
        energy = (
            self.fluid_mixture.density * self.enthalpy(subdomains)
            - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_mixture_internal_energy")
        return energy

    def enthalpy_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Scalar weight for the enthalpy flux:

        .. math::

            \sum_j h_j(p, T, x_j) \rho_j(p, T, x_j) \dfrac{k_r(s_j)}{\mu_j(p, T, x_j)}

        Parameters:
            domains: A sequence of subdomains or boundary grids

        Returns:
            Above operator, if ``domains`` contains the actual subdomains.
            If boundary grids are passed, a boundary operator refering to the BC is
            returned.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_enthalpy_mobility_key, domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist entirely of subdomains for the total mobility."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)
        weight = pp.ad.sum_operator_list(
            [
                phase.enthalpy
                * phase.density
                / phase.viscosity
                * self.relative_permeability(phase.saturation)
                for phase in self.fluid_mixture.phases
            ],
            "enthalpy_flux_weight",
        )

        return weight

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
                dirichlet_operator=self.enthalpy_mobility,
                neumann_operator=self.enthalpy_flux,
                bc_type=self.bc_type_enthalpy_flux,
                name="bc_values_enthalpy",
            )
        )

        discr = self.enthalpy_mobility_discretization(subdomains)
        weight = self.enthalpy_mobility(subdomains)
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
        :class:`DiscretizationsCompositionalFlow`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_enthalpy_mobility_discretization(interfaces)
        weight = self.enthalpy_mobility(subdomains)
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
        :class:`DiscretizationsCompositionalFlow`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_enthalpy_mobility_discretization(interfaces)
        weight = self.enthalpy_mobility(subdomains)
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
    """See :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """See :meth:`ConstitutiveLawsCompositionalFlow.relative_permeability`."""

    component_mobility_discretization: Callable[
        [ppc.Component, list[pp.Grid]], pp.ad.UpwindAd
    ]
    """See :meth:`DiscretizationsCompositionalFlow.component_mobility_discretization`.
    """
    interface_component_mobility_discretization: Callable[
        [ppc.Component, list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :meth:`DiscretizationsCompositionalFlow.
    interface_component_mobility_discretization`.
    """

    eliminate_reference_component: bool
    """See :attr:`SolutionStrategyCompositionalFlow.eliminate_reference_component`."""

    bc_data_component_mobility_key: Callable[[ppc.Component], str]
    """See :meth:`BoundaryConditionsCompositionalFlow.bc_data_component_mobility_key`.
    """
    bc_data_component_flux_key: Callable[[ppc.Component], str]
    """See :meth:`BoundaryConditionsCompositionalFlow.bc_data_component_flux_key`.
    """
    bc_type_component_flux: Callable[[ppc.Component, pp.Grid], pp.BoundaryCondition]
    """See :meth:`BoundaryConditionsCompositionalFlow.bc_type_component_flux`."""

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
            self.fluid_mixture.density * self.porosity(subdomains) * component.fraction
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name(f"component_mass_{component.name}")
        return mass

    def mobility_of_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear weight in the advective flux of a component mass balance equation

        Parameters:
            subdomains: All subdomains in the md-grid or respective boundary grids.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j(p, T, x_j)}{\mu_j} x_{n, ij} k_r(s_j),

            which is the advected mass in the total mass balance equation.
            :math:`x_{n, ij}` denotes the normalized fraction of component i in phase j.

            If boundary grids are passed, this returns a boundary operator whose values
            must be updated based on the boundary data for primary variables.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_component_mobility_key(component), domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist of subdomains for the component mobility."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)
        mobility = pp.ad.sum_operator_list(
            [
                phase.density
                / phase.viscosity
                * phase.normalized_fraction_of[component]
                * self.relative_permeability(phase.saturation)
                for phase in self.fluid_mixture.phases
            ],
            f"mobility_{component.name}",
        )
        return mobility

    def fluid_flux_for_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A modified fluid flux, where the advected entity (component mobility)
        accounts for all phases in the mixture for a given component.

        See :meth:`mobility_of_component`.

        It also accounts for custom choices of mobility discretizations (see
        :class:`DiscretizationsCompositionalFlow`).

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
        weight = self.mobility_of_component(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        component_mobility = partial(self.mobility_of_component, component)
        component_mobility = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            component_mobility,
        )
        fluid_flux = partial(self.fluid_flux_for_component, component)
        fluid_flux = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            fluid_flux,
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
            dirichlet_operator=component_mobility,
            neumann_operator=fluid_flux,
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
        mobility = self.mobility_of_component(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(
            interfaces, mobility, discr
        )
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
        mobility = self.mobility_of_component(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, mobility, discr)
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
    TotalEnergyBalanceEquation,
    ComponentMassBalanceEquations,
    ppc.EquilibriumEquationsMixin,
):
    def set_equations(self):
        TotalMassBalanceEquation.set_equations(self)
        TotalEnergyBalanceEquation.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)
        ppc.EquilibriumEquationsMixin.set_equations(self)
        if "v" not in self.equilibrium_type:
            ppc.EquilibriumEquationsMixin.set_density_relations_for_phases(self)


class VariablesCompositionalFlow(mass_energy.VariablesFluidMassAndEnergy):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    enthalpy_variable: str
    """See :attr:`SolutionStrategyCompositionalFlow.enthalpy_variable`."""

    set_mixture: Callable
    """See :meth:`~porepy.composite.composite_mixins.MixtureMixin.set_mixture`."""

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        1. Sets up the pressure variables (domains, interfaces, wells)
        2. Sets up the energy related variables
        3. Creates the transported enthalpy variable
        4. Sets up the fluid mixture and creates all compositional variables

        """
        # pressure and temperature. This covers also the interface variables for
        # Fourier flux, Darcy flux and enthalpy flux.
        super().create_variables()

        # enthalpy variable
        self.equation_system.create_variables(
            self.enthalpy_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "J"},
        )
        # compositional variables
        self.set_mixture()

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
    DiscretizationsCompositionalFlow,
    FouriersLawCF,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantPermeability,
):
    """The central constitutive law is the equilibrium mixin, which models the local
    thermodynamic equilibrium using indirectly the EoS used by the mixture model.

    Other constitutive laws for the compositional flow treat (as of now) only
    Darcy's law, Fourier's law, and constant matrix parameters."""

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        This basic implementation implements the quadratic law :math:`s_j^2` for a phase
        :math:`j`. Overwrite for something else.

        Parameters:
            saturation: Operator representing the saturation of a phase.

        Returns:
            ``saturation ** 2``.

        """
        return saturation**2


class BoundaryConditionsCompositionalFlow(
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
):
    """Mixin treating boundary conditions for the compositional flow.

    Atop of inheriting the treatment for single phase flow (which is exploited for the
    total mass balance) and the energy balance (total energy balance), this class has
    a treatment for BC for component mass balances.

    By default, Dirichlet-type BC are implemented for pressure and temperature.

    """

    bc_data_total_mobility_key: str = "bc_data_total_mobility"
    """Key for the (time-dependent) Dirichlet BC data for the total mobility in the
    pressure equation.

    Note that this data must be computed in the solution strategy.

    """
    bc_data_enthalpy_mobility_key: str = "bc_data_enthalpy_mobility"
    """Key for the (time-dependent) Dirichlet BC data for the non-linear weight in the
    enthalpy flux in the total energy balance.

    Note that this data must be computed in the solution strategy.

    """
    bc_data_conductivity_key: str = "bc_data_fluid_conductivity"
    """Key for storing the (time-dependent) Dirichlet BC data for the fluid conductivity
    on the boundary.

    It is part of the total conductivity, the non-linear weight in the Fourier flux.

    Note that this data must be computed in the solution strategy.

    """

    def bc_data_component_mobility_key(self, component: ppc.Component) -> str:
        """
        Parameters:
            component: A fluid component in the mixture with a mass balance equation.

        Returns:
            The key for storing (time-dependent) Dirichlet BC data for the mobility in
            the component's mass balance.

            Note that this data must be computed in the solution strategy.

        """
        return f"bc_data_mobility_{component.name}"

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
        return f"bc_data_component_flux_{component.name}"

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
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(grid).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(grid, boundary_faces, "dir")


# TODO overwrite methods for iterations, initial conditions, bc update, ...
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

    """

    equilibrium_type: Literal["p-T", "p-h", "v-h"]
    """See :attr:`~porepy.composite.composite_mixins.EquilibriumEquationsMixin.
    equilibrium_type`."""

    fluid_mixture: ppc.Mixture
    """See :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    bc_type_component_flux: Callable[[ppc.Component, pp.Grid], pp.BoundaryCondition]
    """See :meth:`BoundaryConditionsCompositionalFlow.bc_type_component_flux`."""

    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :meth:`FouriersLawCF.thermal_conductivity`."""
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :meth:`~porepy.models.constitutive_laws.DarcysLaw.darcy_flux`."""
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """See :meth:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow.
    interface_darcy_flux`."""
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """See :meth:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow.
    well_flux`."""

    total_mobility_discretization: Callable[[Sequence[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`DiscretizationsCompositionalFlow`."""
    interface_total_mobility_discretization: Callable[
        [Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`DiscretizationsCompositionalFlow`."""
    enthalpy_mobility_discretization: Callable[[Sequence[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`DiscretizationsCompositionalFlow`."""
    interface_enthalpy_mobility_discretization: Callable[
        [Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`DiscretizationsCompositionalFlow`."""
    fourier_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """See :meth:`~porepy.models.constitutive_laws.FouriersLaw.
    fourier_flux_discretization`"""
    component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.Grid]], pp.ad.UpwindAd
    ]
    """See :class:`DiscretizationsCompositionalFlow`."""
    interface_component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`DiscretizationsCompositionalFlow`."""

    equilibriate_fluid: Callable[
        [Optional[np.ndarray]], tuple[ppc.FluidState, np.ndarray]
    ]
    """Defined by a FlashMixin instance."""
    postprocess_failures: Callable[[ppc.FluidState, np.ndarray], ppc.FluidState]
    """Defined by a FlashMixin instance."""

    temperature_variable: str
    """Defined in the solutions strategy for the energy balance."""

    pressure_variable: str
    """Defined in the solutions strategy for the single phase flow."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

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

    def component_mobility_keyword(self, component: ppc.Component) -> str:
        """
        Parameters:
            component: A transportable fluid component in the mixture.

        Returns:
            Keyword for storing the discretization parameters and matrices of the
            discretization for the mobility of a component in its mass balance.

        """
        return f"mobility_{component.name}"

    def initial_condition(self) -> None:
        """Initiates additionally zero fluxes for the component mobilities."""
        super().initial_condition()

        for component in self.fluid_mixture.components:
            for sd, data in self.mdg.subdomains(return_data=True):
                pp.initialize_data(
                    sd,
                    data,
                    self.component_mobility_keyword(component),
                    {"darcy_flux": np.zeros(sd.num_faces)},
                )
            for intf, data in self.mdg.interfaces(return_data=True):
                pp.initialize_data(
                    intf,
                    data,
                    self.component_mobility_keyword(component),
                    {"darcy_flux": np.zeros(intf.num_cells)},
                )

    def set_discretization_parameters(self) -> None:
        """Sets the discretization parameters, minding the custom Fourier law."""
        # First, discretizations in single phase flow, which is used for pressure equ.
        mass.SolutionStrategySinglePhaseFlow.set_discretization_parameters(self)

        subdomains = self.mdg.subdomains()
        conducivity = self.thermal_conductivity(subdomains).value(self.equation_system)
        projection = pp.ad.SubdomainProjections(subdomains)

        # Second, discretizations in the energy equation. Slightly different due to
        # FouriersLawCF and access to conductivity of fluid mixture.
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.fourier_keyword,
                {
                    "bc": self.bc_type_fourier_flux(sd),
                    "second_order_tensor": pp.SecondOrderTensor(
                        projection.cell_restriction([sd]) * conducivity
                    ),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_enthalpy_flux(sd),
                },
            )

        # Third, discretization parameters for mobility term in component mass balances
        for component in self.fluid_mixture.components:
            for sd, data in self.mdg.subdomains(return_data=True):
                pp.initialize_data(
                    sd,
                    data,
                    self.component_mobility_keyword(component),
                    {
                        "bc": self.bc_type_component_flux(component, sd),
                    },
                )

    def set_nonlinear_discretizations(self) -> None:
        """Overwrites parent methods to point to discretizations in
        :class:`DiscretizationsCompositionalFlow`.

        Adds additionally the MPF discretization of the Fourier flux to the set
        of non-linear discretizations which need to be re-discretized in every
        iteration.

        Note:
            Re-discretizing the MPFA for the conductive flux might be expensive. TODO

        """
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()

        # Upwind of total mobility in pressure equation
        self.add_nonlinear_discretization(
            self.total_mobility_discretization(subdomains).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_total_mobility_discretization(interfaces).flux,
        )

        # Upwind of enthalpy mobility in energy equation
        self.add_nonlinear_discretization(
            self.enthalpy_mobility_discretization(subdomains).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_enthalpy_mobility_discretization(interfaces).flux,
        )

        # MPFA of Fourier flux, which dpends on the conductivity tensor.
        self.add_nonlinear_discretization(
            self.fourier_flux_discretization(subdomains),  # TODO really everything?
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

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform the p-h flash as a predictor step.

        Subsequently it computes the fluxes for various Upwind discretiztions
        (without calling the parent methods of mass and energy though, to save time).

        Finally, it calles the base class' method to update discretization parameters
        and to re-discretize.

        """

        # Flashing the mixture as a predictor step
        state = self.postprocess_failures(*self.equilibriate_fluid(None))

        # Setting equilibrium values for fractional variables
        for j, phase in enumerate(self.fluid_mixture.phases):
            self.equation_system.set_variable_values(
                state.sat[j], [phase.saturation.name], iterate_index=0
            )
            self.equation_system.set_variable_values(
                state.y[j], [phase.fraction.name], iterate_index=0
            )

        # setting Temperature and pressure values, depending on equilibrium definition
        if "T" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                state.T, [self.temperature_variable], iterate_index=0
            )
        if "p" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                state.p, [self.pressure_variable], iterate_index=0
            )

        for sd, data in self.mdg.subdomains(return_data=True):
            # Computing Darcy flux and updating it in the mobility dicts for pressure
            # and energy equtaion
            vals = self.darcy_flux([sd]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
            # Updating the flux in the mobility dicts in each mass balance equation
            for component in self.fluid_mixture.components:
                if (
                    component == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue
                data[pp.PARAMETERS][self.component_mobility_keyword(component)].update(
                    {"darcy_flux": vals}
                )
        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            # Computing the darcy flux in fractures (given by variable)
            vals = self.interface_darcy_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
            # Updating the flux in the mobility dicts in each mass balance equation
            for component in self.fluid_mixture.components:
                if (
                    component == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue
                data[pp.PARAMETERS][self.component_mobility_keyword(component)].update(
                    {"darcy_flux": vals}
                )
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            # Computing the darcy flux in wells (given by variable)
            vals = self.well_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
            # Updating the flux in the mobility dicts in each mass balance equation
            for component in self.fluid_mixture.components:
                if (
                    component == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue
                data[pp.PARAMETERS][self.component_mobility_keyword(component)].update(
                    {"darcy_flux": vals}
                )

        # Call to base class method to update discr. parameters and re-discretize
        pp.SolutionStrategy.before_nonlinear_loop(self)

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
    ppc.MixtureMixin,
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
        and phases and derive a flow model from it.

        .. code:: python
            :linenos:

            class MyMixture(MixtureMixin):

                def get_components(self):
                    ...
                def get_phase_configuration(self, components):
                    ...

            class MyModel(MyMixture, CompositionalFlow):
                ...

    """

    pass
