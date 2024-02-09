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


class DiscretizationsCF:
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
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`."""
    component_mobility_keyword: Callable[[ppc.Component], str]
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    enthalpy_keyword: str
    """Provided by :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`.
    """

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


class ThermalConductivityCF(pp.constitutive_laws.ThermalConductivityLTE):
    """Fourier's law in the compositional flow setting.

    Atop the parent class methods, it provides means to compute the conductivity of a
    fluid mixture.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

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

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
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

    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
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
        phi = self.porosity(subdomains)
        if isinstance(phi, pp.ad.Scalar):
            size = sum([sd.num_cells for sd in subdomains])
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) - (
            phi - 1.0
        ) * self.solid_thermal_conductivity(subdomains)
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

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    phases_have_fractions: bool
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

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

    def total_mobility(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        r"""Returns the non-linear weight in the advective flux, assuming the mixture is
        defined on all subdomains.

        Parameters:
            subdomains: A list of subdomains.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j(p, T, x_j) k_r(s_j)}{\mu_j},

            which is the advected mass in the total mass balance equation.

        """
        weight = pp.ad.sum_operator_list(
            [
                phase.density(subdomains)
                / phase.viscosity(subdomains)
                * self.relative_permeability(phase.saturation(subdomains))
                for phase in self.fluid_mixture.phases
            ],
            "total_mobility",
        )

        return weight

    def mobility_of_component(
        self, component: ppc.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Non-linear weight in the advective flux of a component mass balance equation

        Parameters:
            subdomains: All subdomains in the md-grid.

        Raises:
            NotImplementedError: If the mixture mixin did not create fractions of
                components in phases.

        Returns:
            An operator representing

            .. math::

                \sum_j \dfrac{\rho_j(p, T, x_j)}{\mu_j} x_{n, ij} k_r(s_j),

            which is the advected mass in the total mass balance equation.
            :math:`x_{n, ij}` denotes the normalized fraction of component i in phase j.

        """

        if not self.phases_have_fractions:
            raise NotImplementedError(
                "Mobilities of a component not available:"
                + " No compositional fractions created."
            )
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(subdomains)
                / phase.viscosity(subdomains)
                * phase.normalized_fraction_of[component](subdomains)
                * self.relative_permeability(phase.saturation(subdomains))
                for phase in self.fluid_mixture.phases
            ],
            f"mobility_{component.name}",
        )
        return mobility

    def fractional_mobility(
        self, component: ppc.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the fractional mobility in a component's mass balance.

        The fractional mobility is the quotient of component mobility and total
        mobility, i.e. :meth:`mobility_of_component` / :meth:`total_mobility`.

        """

        fractional_mobility = self.mobility_of_component(
            component, subdomains
        ) / self.total_mobility(subdomains)
        fractional_mobility.set_name(f"fractional_mobility_{component.name}")
        return fractional_mobility


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
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

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
        mass_density = self.fluid_mixture.density * self.porosity(subdomains)
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


class TotalEnergyBalanceEquation(energy.EnergyBalanceEquations):
    """Mixed-dimensional balance of total energy in a fluid mixture, formulated with an
    independent enthalpy variable.

    Balance equation for all subdomains and advective and diffusive fluxes
    (Fourier flux) internally and on all interfaces of codimension one and advection on
    interfaces of codimension two (well-fracture intersections).

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~VariablesCompositionalFlow`."""

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

    enthalpy_mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Provided by :class:`DiscretizationsCF`."""
    interface_enthalpy_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`DiscretizationsCF`."""

    bc_type_enthalpy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by
    :class:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance`."""

    bc_data_enthalpy_flux_key: str
    """Provided by
    :class:`~porepy.models.energy_balance.BoundaryConditionsEnergyBalance`."""

    bc_data_enthalpy_mobility_key: str
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""

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
                phase.enthalpy(domains)
                * phase.density(domains)
                / phase.viscosity(domains)
                * self.relative_permeability(phase.saturation(domains))
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
        :class:`DiscretizationsCF`."""
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
        :class:`DiscretizationsCF`."""
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
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Provided by :class:`ConstitutiveLawsCompositionalFlow`."""

    fractional_mobility: Callable[
        [ppc.CompiledUnifiedFlash, Sequence[pp.Grid]], pp.ad.Operator
    ]
    """Provided by :class:`PermeabilityCF`."""

    component_mobility_discretization: Callable[
        [ppc.Component, list[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`DiscretizationsCF`."""
    interface_component_mobility_discretization: Callable[
        [ppc.Component, list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`DiscretizationsCF`."""

    total_mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`PermeabilityCF`."""

    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    bc_data_component_mobility_key: Callable[[ppc.Component], str]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""
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
            * self.fluid_mixture.density
            * component.fraction(subdomains)
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name(f"component_mass_{component.name}")
        return mass

    def advected_mass(
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
        return self.fractional_mobility(domains)

    def fluid_flux_for_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A modified fluid flux, where the advected entity (component mobility)
        accounts for all phases in the mixture for a given component.

        See :meth:`mobility_of_component`.

        It also accounts for custom choices of mobility discretizations (see
        :class:`DiscretizationsCF`).

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
        weight = self.advected_mass(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        advected_mass = partial(self.advected_mass, component)
        advected_mass = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            advected_mass,
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
            dirichlet_operator=advected_mass,
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
        advected_mass = self.advected_mass(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(
            interfaces, advected_mass, discr
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
        advected_mass = self.advected_mass(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(
            interfaces, advected_mass, discr
        )
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
    ppc.SecondaryExpressionsMixin,
    ppc.EquilibriumEquationsMixin,
):
    set_secondary_equations: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.SecondaryExpressionsMixin`
    or a childe class inheriting from it.
    """

    def get_all_secondary_equation_names(self) -> list[str]:
        """Returns a complete list of secondary equations introduced by the
        compositional framework.

        These include the equilibrium equations (if any), the density relations and
        custom secondary expressions (if any).

        """
        return ppc.EquilibriumEquationsMixin.get_equilibrium_equation_names(
            self
        ) + ppc.SecondaryExpressionsMixin.get_secondary_equation_names(self)

    def set_equations(self):
        TotalMassBalanceEquation.set_equations(self)
        TotalEnergyBalanceEquation.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)

        # If an equilibrium is defined, introduce the equations as a block of secondary
        # equations
        if self.equilibrium_type is not None and self.phases_have_fractions:
            ppc.EquilibriumEquationsMixin.set_equations(self)

        # density relations are always defined and the same, hence we use the base class
        ppc.SecondaryExpressionsMixin.set_density_relations_for_phases(self)

        # This might be custom by an inherited class, hence we use the mixin
        self.set_secondary_equations()


class VariablesCompositionalFlow(mass_energy.VariablesFluidMassAndEnergy):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by :class:`~porepy.composite.composite_mixins.EquilibriumEquationsMixin`
    ."""

    phases_have_fractions: bool
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    enthalpy_variable: str
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    create_fractional_variables: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    _feed_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    _solute_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    _saturation_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    _phase_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    _compositional_fraction_variables: list[str]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    @property
    def feed_fraction_variables(self) -> list[str]:
        """Names of feed fraction variables created by the mixture mixin."""
        if not (
            hasattr(self, "_feed_fraction_variables") and hasattr(self, "fluid_mixture")
        ):
            return list()
        else:
            if self.fluid_mixture.num_components == 1:
                return list()
            else:
                return [name for name in self._feed_fraction_variables]

    @property
    def solute_fraction_variables(self) -> list[str]:
        """Names of solute fraction variables created by the mixture mixin."""
        if not (
            hasattr(self, "_solute_fraction_variables")
            and hasattr(self, "fluid_mixture")
        ):
            return list()
        else:
            return [name for name in self._solute_fraction_variables]

    @property
    def phase_fraction_variables(self) -> list[str]:
        """Names of phase fraction variables created by the mixture mixin."""
        if not (
            hasattr(self, "_phase_fraction_variables")
            and hasattr(self, "fluid_mixture")
        ):
            return list()
        else:
            if self.fluid_mixture.num_phases == 1:
                return list()
            else:
                return [name for name in self._phase_fraction_variables]

    @property
    def saturation_variables(self) -> list[str]:
        """Names of phase saturation variables created by the mixture mixin."""
        if not (
            hasattr(self, "_saturation_variables") and hasattr(self, "fluid_mixture")
        ):
            return list()
        else:
            if self.fluid_mixture.num_phases == 1:
                return list()
            else:
                return [name for name in self._saturation_variables]

    @property
    def compositional_variables(self) -> list[str]:
        """Names of phase fraction variables created by the mixture mixin."""
        if not (
            hasattr(self, "_compositional_fraction_variables")
            and hasattr(self, "fluid_mixture")
            and self.phases_have_fractions
        ):
            return list()
        else:
            return [name for name in self._compositional_fraction_variables]

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
        # Performing sanity checks of set-up
        if self.equilibrium_type is not None and self.phases_have_fractions is False:
            raise ValueError(
                f"Conflicting model set-up: No fractional in phases requested but"
                + f" {self.equilibrium_type} equilibrium included."
            )
        elif self.equilibrium_type is None and self.phases_have_fractions is True:
            warnings.warn(
                "Unusual model set-up: Fractiona variables requested but no"
                + " equilibrium system requested. Check secondary equations."
            )
        self.create_fractional_variables()

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
    DiscretizationsCF,
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
    bc_data_enthalpy_mobility_key: str = "bc_values_enthalpy_mobility"
    """Key for the (time-dependent) Dirichlet BC data for the non-linear weight in the
    enthalpy flux in the total energy balance.

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
        return f"bc_values_{self.mobility_keyword}_{component.name}"

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
        # Define boundary faces.
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

    """

    equilibrium_type: Literal["p-T", "p-h", "v-h"]
    """Provided by :class:`~porepy.composite.composite_mixins.EquilibriumEquationsMixin`
    ."""

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    bc_type_component_flux: Callable[[ppc.Component, pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCompositionalFlow`."""

    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`ThermalConductivityCF`."""
    total_mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`PermeabilityCF`."""
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
    """Provided by :class:`DiscretizationsCF`."""
    interface_enthalpy_mobility_discretization: Callable[
        [Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`DiscretizationsCF`."""

    fourier_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """Provided by :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.Grid]], pp.ad.UpwindAd
    ]
    """Provided by :class:`DiscretizationsCF`."""
    interface_component_mobility_discretization: Callable[
        [ppc.Component, Sequence[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Provided by :class:`DiscretizationsCF`."""

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

    phases_have_fractions: bool
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    create_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    create_phase_properties: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""
    create_mixture_properties: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.MixtureMixin`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations = list()

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
        return f"{self.mobility_keyword}_{component.name}"

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
        self.create_phase_properties()
        self.create_mixture_properties()

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

        # Third, discretization parameters for mobility term in component mass balances
        for component in self.fluid_mixture.components:
            for sd, data in self.mdg.subdomains(return_data=True):
                pp.initialize_data(
                    sd,
                    data,
                    self.component_mobility_keyword(component),
                    {
                        "bc": self.bc_type_component_flux(component, sd),
                        # TODO copy boundary values from boundary operator
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
        :class:`DiscretizationsCF`.

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

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform the p-h flash as a predictor step.

        Subsequently it computes the fluxes for various Upwind discretiztions
        (without calling the parent methods of mass and energy though, to save time).

        Finally, it calles the base class' method to update discretization parameters
        and to re-discretize.

        """

        # Flashing the mixture as a predictor step, if equilibrium defined
        if self.equilibrium_type is not None:
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

            if self.phases_have_fractions:
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
                # TODO The resulting fluid enthalpy can change due to numerical
                # precision. Should it be updated as well?
            if "p" not in self.equilibrium_type:
                self.equation_system.set_variable_values(
                    state.p, [self.pressure_variable], iterate_index=0
                )

        # After updating the fluid properties, update discretization parameters and
        # re-discretize
        self.update_discretization_parameters()
        self.rediscretize_fluxes()

        # Evaluate new fluxes before re-discretizing Upwind
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

        # Re-discretize Upwinding
        self.rediscretize()

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
