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

# region CONSTITUTIVE LAWS taylored to pore.composite and its mixins


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
    """Provided by :class:`SolidSkeletonCF`."""

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


class SolidSkeletonCF(
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.SpecificHeatCapacities,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantSolidDensity,
):
    """Collection of constitutive laws for the solid skeleton in the compositional
    flow framework.

    It additionally provides constitutive laws defining the relative and normal
    permeability functions in the compositional framework, based on saturations and
    mobilities.

    It also provides an operator representing the pore volume, used to define
    the :meth:`volume` which is used for isochoric flash calculations.

    TODO Is this the right place to implement :meth:`volume`?

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`MobilityCF`."""

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`~porepy.models.VariableMixin`."""

    temperature_variable: str
    """Provided by :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`.
    """

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

    def solid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the solid.

        Note:
            This must override the definition of solid internal energy which is
            (for some reasons) defined in the basic energy balance equation.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid energy.

        """
        c_p = self.solid_specific_heat_capacity(subdomains)
        energy = (
            (pp.ad.Scalar(1) - self.porosity(subdomains))
            * self.solid_density(subdomains)
            * c_p
            * self.perturbation_from_reference(self.temperature_variable, subdomains)
        )
        energy.set_name("solid_internal_energy")
        return energy

    def pore_volume(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Returns an Ad representation of the pore volume, which is a multiplication
        of cell volumes and porosity on subdomains."""
        cell_volume = pp.wrap_as_dense_ad_array(
            np.hstack([g.cell_volumes for g in subdomains]), name="cell_volume"
        )
        op = cell_volume * self.porosity(subdomains)
        op.set_name("pore_volume")
        return op

    def volume(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Returns the target volume to be used in equilbrium calculations with
        isochoric constraints.

        The base implementation returns the :meth:`pore_volume`.

        """
        return self.pore_volume(subdomains)


# endregion
# region PDEs used in the (fractional) CF, taylored to pp.composite and its mixins


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

    @staticmethod
    def primary_equation_name() -> str:
        """Returns the string which is used to name the pressure equation on all
        subdomains, which is the primary PDE set by this class."""
        return "pressure_equation"

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
        eq.set_name(TotalMassBalanceEquation.primary_equation_name())
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

    Note:
        Since this class utilizes the basic energy balance, it introduces an
        interface enthalpy flux variable (advective energy flux) and respective
        equations on the interface.

        This is not necessarily required, and can be eliminated.

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

    @staticmethod
    def primary_equation_name():
        """Returns the name of the total energy balance equation introduced by this
        class, which is a primary PDE on all subdomains."""
        return "total_energy_balance"

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites the parent method to give the name assigned by
        :meth:`primary_equation_name`."""
        eq = super().energy_balance_equation(subdomains)
        eq.set_name(TotalEnergyBalanceEquation_h.primary_equation_name())
        return eq

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
                bc_type=self.bc_type_darcy_flux,
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

    def _mass_balance_equation_name(self, component: ppc.Component) -> str:
        """Method returning a name to be given to the mass balance equation of a
        component."""
        return f"mass_balance_equation_{component.name}"

    def mass_balance_equation_names(self) -> list[str]:
        """Returns the names of mass balance equations set by this class,
        which are primary PDEs on all subdomains for each independent fluid component.
        """
        names: list[str] = list()
        for component in self.fluid_mixture.components:
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue
            names.append(self._mass_balance_equation_name(component))
        return names

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
        eq.set_name(self._mass_balance_equation_name(component))
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
        interface_flux = cast(
            Callable[[list[pp.MortarGrid]], pp.ad.Operator],
            interface_flux,
        )

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

    def _solute_transport_equation_name(
        self,
        solute: ppc.ChemicalSpecies,
        component: ppc.Compound,
    ) -> str:
        """Method returning a name to be given to the transport equation of a
        solute in a compound."""
        return f"transport_equation_{solute.name}_{component.name}"

    def solute_transport_equation_names(self) -> list[str]:
        """Returns the names of transport equations set by this class,
        which are primary PDEs on all subdomains for each solute in each compound in the
        fluid mixture."""
        names: list[str] = list()
        for component in self.fluid_mixture.components:
            if not isinstance(component, ppc.Compound):
                continue
            for solute in component.solutes:
                names.append(self._solute_transport_equation_name(solute, component))
        return names

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
        eq.set_name(self._solute_transport_equation_name(solute, component))
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


# endregion
# region INTERMEDIATE CF MODEL MIXINS: collecting variables, equations, const. laws


class SecondaryEquationsMixin:
    """Base class for introducing secondary equations into the compositional flow
    formulation.

    Use the methods of this class to close the compositional flow model, in the case of
    some dangling secondary variables.

    Examples:
        1. If an equilibrium condition is defined, the framework introduces saturations
           and molar phase fractions as independent variables, for each independent
           phase.

           The system needs to be closed by introducing the density relations
           :meth:`set_density_relations_for_phases` as local equations.

        2. If no equilibrium condition is defined,
           :attr:`~porepy.composite.base.Phase.saturation` of independent phases
           as well as :attr:`~porepy.composite.base.Phase.partial_fraction_of` are
           dangling variables.

           The system can be closed by using :meth:`eliminate_by_constitutive_law`
           for example, to introduce local, secondary equations.

    """

    fluid_mixture: ppc.Mixture
    """Provided by: class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    normalize_state_constraints: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    eliminate_reference_phase: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""
    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    time_step_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""
    iterate_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""

    add_constitutive_expression: Callable[
        [
            ppc.SecondaryExpression,
            pp.GridLikeSequence,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
        ],
        None,
    ]
    """Provided by :class:`ConstitutiveLawsCompositionalFlow`"""

    def set_equations(self) -> None:
        """Inherit this method to set additional secondary expressions in equation form

        .. math::

            f(x) = 0

        by setting the left-hand side as an equation in the Ad framework.

        The parent method calls :meth:`set_density_relations_for_phases` for models with
        local equilibrium equations (:attr:`equilibrium_type`).

        """
        if self.equilibrium_type is not None:
            self.set_density_relations_for_phases()

    def set_density_relations_for_phases(self) -> None:
        """Introduced the mass relations for phases into the AD system.

        All equations are scalar, single, cell-wise equations on each subdomains.

        This method is separated, because it has another meaning when coupling the
        equilibrium problem with flow and transport.

        In multiphase flow in porous media, saturations must always be provided.
        Hense even if there are no isochoric specifications in the flash, the model
        necessarily introduced the saturations as unknowns.

        The mass relations per phase close the system, by relating molar phase fractions
        to saturations. Hence rendering the system solvable.

        Important:
            If there is only 1 phase, this method does nothing, since in that case
            the molar fraction and saturation of a phase is always 1.

            If the user wants it nevertheless for some reason,
            :meth:`density_relation_for_phase` must be called explicitly.

            That equation is not tracked by this class in this case.
            Hence :meth:`get_secondary_equation_names` will give an empty list.

        """
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()
        if self.fluid_mixture.num_phases > 1:
            for phase in self.fluid_mixture.phases:
                if phase == rphase and self.eliminate_reference_phase:
                    continue
                equ = self.density_relation_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def density_relation_for_phase(
        self, phase: ppc.Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs a local mass relation based on a relation between mixture
        density, saturated phase density and phase fractions.

        For a phase :math:`j` it holds:

        .. math::

            y_j \\rho - s_j \\rho_j = 0~,~
            y_j - s_j \\dfrac{\\rho_j}{rho} = 0

        with the mixture density :math:`\\rho = \\sum_k s_k \\rho_k`, assuming
        :math:`\\rho_k` is the density of a phase when saturated.

        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`s` : Phase :attr:`~porepy.composite.base.Phase.saturation`
        - :math:`\\rho` : Fluid mixture :attr:`~porepy.composite.base.Mixture.density`
        - :math:`\\rho_j` : Phase:attr:`~porepy.composite.base.Phase.density`

        Note:
            These equations can be used to close the model if molar phase fractions and
            saturations are independent variables.

            They also appear in the unified flash with isochoric specificitations.

        Parameters:
            phase: A phase for which the equation should be assembled.
            subdomains: A list of subdomains on which the equation is defined.

        Returns:
            The left-hand side of above equations.

            If normalization of state constraints is set in the solution strategy,
            it returns the normalized form.

        """
        if self.normalize_state_constraints:
            equ = phase.fraction(subdomains) - phase.saturation(
                subdomains
            ) * phase.density(subdomains) / self.fluid_mixture.density(subdomains)
        else:
            equ = phase.fraction(subdomains) * self.fluid_mixture.density(
                subdomains
            ) - phase.saturation(subdomains) * phase.density(subdomains)
        equ.set_name(f"density-relation-{phase.name}")
        return equ

    def eliminate_by_constitutive_law(
        self,
        independent_quantity: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable],
        dependencies: Sequence[
            Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
        ],
        func: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
        domains: pp.GridLikeSequence,
        dofs: dict = {"cells": 1},
    ) -> None:
        """Method to add a secondary equation eliminating a formally independent
        quantity by some constitutive law.

        For a formally independent quantity :math:`\\varphi`, this method introduces
        a secondary equation :math:`\\varphi - \\hat{\\varphi}(x) = 0`, with :math:`x`
        denoting the ``dependencies``.

        It uses :class:`~porepy.composite.composite_utils.SecondaryExpression` to
        provide AD representations of :math:`\\hat{\\varphi}` and to update its values
        and derivatives using ``func`` in the solutionstrategy.

        Note:
            Keep the limitations of
            :class:`~porepy.composite.composite_utils.SecondaryExpression` in mind,
            especially with regards to ``dofs``.

            Time step depth and iteration depth are assigned according to the numbers
            of indices in the solution strategy.

        Parameters:
            independent_quantity: AD representation :math:`\\varphi`, callable on some
                grids.
            dependencies: First order dependencies (variables) by which :math:`\\varphi`
                is expressed locally.
            func: A numerical function which computes the values of
                :math:`\\hat{\\varphi}(x)` and its derivatives.

                The return value must be a 2-tuple, containing a 1D value array and a
                2D derivative value array. The shape of the value array must be
                ``(N,)``, where ``N`` is consistent with ``dofs``, and the shape of
                the derivative value array must be ``(M,N)``, where ``M`` denotes the
                number of first order dependencies (``M == len(dependencies)``).

                The order of arguments for ``func`` must correspond with the order in
                ``dependencies``.
            domains: A Sequence of grids on which the quantity and its depdencies are
                defined and on which the equation should be introduces.
                Used to call ``independent_quantity`` and ``dependencies``.
            dofs: ``default={'cells':1}``

                Argument for when adding above equation to the equation system.

        """

        primary_expr = independent_quantity(domains)

        sec_expr = ppc.SecondaryExpression(
            name=f"secondary_expression_for_{primary_expr.name}",
            mdg=self.mdg,
            dependencies=dependencies,
            time_step_depth=len(self.time_step_indices),
            iterate_depth=len(self.iterate_indices),
        )

        local_equ = primary_expr - sec_expr(domains)
        local_equ.set_name(
            f"elimination_of_{primary_expr.name}_on_grids_{[g.id for g in domains]}"
        )
        self.equation_system.set_equation(local_equ, domains, dofs)

        self.add_constitutive_expression(sec_expr, domains, func)


class EquationsCompositionalFlow(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation_h,
    SoluteTransportEquations,
    ComponentMassBalanceEquations,
    SecondaryEquationsMixin,
    ppc.EquilibriumEquationsMixin,
):
    @property
    def primary_equation_names(self) -> list[str]:
        """Returns the list of primary equation, consisting of

        1. pressure equation,
        2. energy balance equation,
        3. mass balance equations per fluid component,
        4. transport equations per solute in compounds in the fluid.

        Note:
            Interface equations, which are non-local equations since they relate
            interface variables and respective subdomain variables on some subdomain
            cells, are not included.

            This might have an effect on the Schur complement in the solution strategy

        """

        return (
            [
                TotalMassBalanceEquation.primary_equation_name(),
                TotalEnergyBalanceEquation_h.primary_equation_name(),
            ]
            + self.mass_balance_equation_names()
            + self.solute_transport_equation_names()
        )

    @property
    def secondary_equation_names(self) -> list[str]:
        """Returns a list of secondary equations, which is defined as the complement
        of :meth:`primary_equation_names` and all equations found in the equation
        system."""
        all_equations = set(
            [name for name, equ in self.equation_system.equations.items()]
        )
        return list(all_equations.difference(set(self.primary_equation_names)))

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
        SecondaryEquationsMixin.set_equations(self)


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
        of

        1. pressure,
        2. fluid enthalpy,
        3. overall fractions,
        4. solute fractions.

        Primary variable names are used to define the primary block in the Schur
        elimination in the solution strategy.

        """
        return (
            [
                self.pressure_variable,
                self.enthalpy_variable,
            ]
            + self.overall_fraction_variables
            + self.solute_fraction_variables
        )

    @property
    def secondary_variables(self) -> list[str]:
        """Returns a list of secondary variables, which is defined as the complement
        of :meth:`primary_variable_names` and all variables found in the equation
        system."""
        all_vars = set([var.name for var in self.equation_system.get_variables()])
        primary_vars = set(self.primary_variable_names)
        return list(all_vars.difference(primary_vars))

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
    ppc.FluidMixtureMixin,
    MobilityCF,
    ThermalConductivityCF,
    SolidSkeletonCF,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.PeacemanWellFlux,
):
    """Constitutive laws for compositional flow, using the fluid mixture class
    and mobility and conductivity laws adapted to it.

    It also uses a separate class, which collects constitutive laws for the solid
    skeleton.

    All other constitutive laws are analogous to the underlying mass and energy
    transport.

    """

    _constitutive_expressions: dict[
        str,
        tuple[
            ppc.SecondaryExpression,
            pp.GridLikeSequence,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
        ],
    ]
    """Provided by :class:`SolutionStrategyCompositionalFlow`"""

    def add_constitutive_expression(
        self,
        expression: ppc.SecondaryExpression,
        domains: Sequence[pp.Grid | pp.MortarGrid],
        func: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """"""
        self._constitutive_expressions.update(
            {expression.name, (expression, domains, func)}
        )

    def update_all_constitutive_expressions(
        self, update_derivatives: bool = False
    ) -> None:
        """Method to update the values of all constitutive expressions in the iterative
        sense.

        Loops over all expressions stored using :meth:`add_constitutive_expression`,
        evaluates their dependencies on respective domains and calls the stored
        evaluation function to obtain derivatives and values.

        To be used before solving the system in a non-linear iteration.

        Parameters:
            update_derivatives: ``default=False``

                If True, it updates also the derivative values in the iterative
                sense.

        """
        for expr, domains, func in self._constitutive_expressions.values():
            for g in domains:
                X = [x([g]).value(self.equation_system) for x in expr._dependencies]

                vals, diffs = func(*X)

                expr.progress_iterate_values_on_grid(vals, g)
                if update_derivatives:
                    expr.progress_iterate_derivatives_on_grid(diffs, g)

    def progress_all_constitutive_expressions_in_time(
        self, progress_derivatives: bool = True
    ) -> None:
        """Method to progress the values of all added constitutive expressions in time.

        It takes the values at the most recent iterates, and stores them as the most
        recent previous time step.

        To be used after non-linear iterations converge in a time-dependent problem.

        Parameters:
            progress_derivatives: ``default=False``

                If True, it progresses also the derivative values in the time
                sense.

        """
        for expr, domains, _ in self._constitutive_expressions.values():
            expr.progress_values_in_time(domains)
            if progress_derivatives:
                expr.progress_derivatives_in_time(domains)


# endregion
# region SOLUTION STRATEGY, including separate treatment of BC and IC


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
    ncomp = np.array([len(ps.x) for ps in state.phases], dtype=int)
    nphase = len(state.y)
    phase_types = np.array([ps.phasetype for ps in state.phases], dtype=int)

    # No derivatives expected on boundary
    out = ppc.initialize_fluid_state(
        nc, ncomp, nphase, phase_types, with_derivatives=False
    )

    # prolonging intensive and extensive fluid state
    out.p[dir] = state.p
    out.T[dir] = state.T
    out.h[dir] = state.h
    out.v[dir] = state.v
    # feed fractions
    for i in range(len(state.z)):
        out.z[i, dir] = state.z[i]
    # saturations and molar fractions and properties per phase
    for j in range(nphase):
        out.y[j, dir] = state.y[j]
        out.sat[j, dir] = state.sat[j]
        for i in range(ncomp[j]):
            out.phases[j].x[i, dir] = state.phases[j].x[i]
            out.phases[j].phis[i, dir] = state.phases[j].phis[i]
        out.phases[j].mu[dir] = state.phases[j].mu
        out.phases[j].kappa[dir] = state.phases[j].kappa

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
    has_time_dependent_boundary_equilibrium: bool
    """Provided by :class:`SolutionStrategyCompositionalFlow`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by:class:`SolutionStrategyCompositionalFlow`."""

    _overall_fraction_variable: Callable[[ppc.Component], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _phase_fraction_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""

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
            assert len(set(check_dir)) <= 2, (
                "Inconsistent number of Dirichlet boundary faces defined for advective"
                + f" flux on subdomain {sd}."
            )

            # same must hold for Neumann and Robin BC
            check_neu = bc_darcy.is_neu + bc_fluid.is_neu + bc_enthalpy.is_neu
            assert len(set(check_neu)) <= 2, (
                "Inconsistent number of Neumann boundary faces defined for advective"
                + f" flux on subdomain {sd}."
            )
            check_rob = bc_darcy.is_rob + bc_fluid.is_rob + bc_enthalpy.is_rob
            assert len(set(check_rob)) <= 2, (
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
        """On top of the parent methods for mass and energy models, it updates
        values for fractional variables and thermodynamic properties of phases on the
        boundary.

        It updates BC values for overall fractions on the Dirichlet boundary
        (:meth:`bc_values_overall_fraction`), BC values for fractional mobilities on the
        Neumann boundary (:meth:`bc_values_fractional_mobility`),
        and BC values for the weight in the advective enthalpy flux on the Neumann
        boundary (:meth:`bc_values_enthalpy_flux_weight`)

        If the problem has time-dependent BC and an equilibrium problem
        (:attr:`has_time_dependent_boundary_equilibrium`),
        :meth:`boundary_flash` is performed to calculate the BC values on the
        Dirichlet boundary and to populate :attr:`boundary_fluid_state`.

        Note:
            The solution strategy performs the first setting of boundary values, if
            BC values not time dependent and equilibrium is defined.

            This is because of efficiency reasons to not invoke the flash all the time.

        It then proceedes to call the update of BC values for fractional variables
        (:meth:`update_boundar_conditions_for_fractions`).

        Finally, it updates the values of secondary expressions on the boundary
        (thermodynamic properties of phases, which appear in the non-linear term in
        various fluxes). These include

        - phase densities
        - phase volumes
        - phase enthalpies
        - phase viscosities
        - phase conductivities

        (see :meth:`update_boundar_conditions_for_secondary_expressions`).

        """
        super().update_all_boundary_conditions()

        # values of fractional mobilities on Neumann boundaries, and overall fractions
        # on Dirichlet boundaries
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

            bc_vals = partial(self.bc_values_overall_fraction, component)
            bc_vals = cast(Callable[[pp.BoundaryGrid], np.ndarray], bc_vals)
            self.update_boundary_condition(
                name=self._overall_fraction_variable(component),
                function=bc_vals,
            )

        # values for weight in enthalpy flux on Neumann boundaries
        self.update_boundary_condition(
            name=self.bc_data_enthalpy_flux_key,
            function=self.bc_values_enthalpy_flux_weight,
        )

        # Update of BC values on the Dirichlet-boundary, in case of a time-dependent
        # BC. NOTE due to computational cost, this is only done if time-dependence is
        # indicated, otherwise it is done once in the solution strategy in the beginning
        if (
            self.has_time_dependent_boundary_equilibrium
            and self.equilibrium_type is not None
        ):
            self.boundary_flash()

        self.update_boundar_conditions_for_fractions()
        self.update_boundar_conditions_for_secondary_expressions()

    def update_boundar_conditions_for_fractions(self) -> None:
        """Called by :meth:`update_all_boundary_conditions` to update the values of
        fractional variables.

        It updates the values of saturations in any case.

        In the case that a local equilibrium problem is included, BC values for
        molar phase fractions and extended fractions are updated.
        In the case of no local equilibrium formulation, BC values for partial fractions
        are updated (they are dependent in the equilibrium case).

        Uses partial evaluations of

        - :meth:`bc_values_saturation`
        - :meth:`bc_values_phase_fraction`
        - :meth:`bc_values_compositional_fraction`

        to update the grid dictionaries with the parent method
        :meth:`update_boundary_condition`.

        """

        for phase in self.fluid_mixture.phases:
            # phase fractions and saturations
            if (
                self.eliminate_reference_phase
                and phase == self.fluid_mixture.reference_phase
            ):
                pass
            else:
                s_name = self._saturation_variable(phase)
                s_bc = partial(self.bc_values_saturation, phase)
                s_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], s_bc)
                self.update_boundary_condition(s_name, s_bc)

                # phase fractions are only updated if equilibrium defined
                if self.equilibrium_type is not None:
                    y_name = self._phase_fraction_variable(phase)
                    y_bc = partial(self.bc_values_phase_fraction, phase)
                    y_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], y_bc)
                    self.update_boundary_condition(y_name, y_bc)

            # compositional fractions are always updated (partial and extended are both
            # named the same if independent)
            for comp in phase:
                x_name = self._relative_fraction_variable(comp, phase)
                x_bc = partial(self.bc_values_relative_fraction, comp, phase)
                x_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], x_bc)
                self.update_boundary_condition(x_name, x_bc)

    def update_boundar_conditions_for_secondary_expressions(self) -> None:
        """Updates the boundary conditions for secondary expressions appearing in
        the non-linear weights in various fluxes.

        It uses the framework of the class
        :class:`~porepy.composite.composite_utils.SecondaryExpression` to update the
        values directly (bypassing the parent method :meth:`update_boundary_condition`).

        The base updates includes updates for:

        - phase densities (:meth:`bc_values_phase_density`)
        - phase volumes (reciprocal of :meth:`bc_values_phase_density`)
        - phase enthalpies (:meth:`bc_values_phase_enthalpy`)
        - phase viscosities (:meth:`bc_values_phase_viscosity`)

        Note:
            In a consistent discretization, no BC values for conductivities are
            required. Hence they are not performed here.

            Also, no BC values for purely local quantities like fugacity coefficients
            are required.

        """
        for bg in self.mdg.boundaries():
            for phase in self.fluid_mixture.phases:
                # phase properties which appear in mobilities
                rho_bc = self.bc_values_phase_density(phase, bg)
                phase.density.update_boundary_value(rho_bc, bg)

                # volume as reciprocal of density, only where given
                v_bc = np.zeros_like(rho_bc)
                idx = rho_bc > 0
                v_bc[idx] = 1.0 / rho_bc[idx]
                phase.volume.update_boundary_value(v_bc, bg)

                phase.enthalpy.update_boundary_value(
                    self.bc_values_phase_enthalpy(phase, bg), bg
                )

                phase.viscosity.update_boundary_value(
                    self.bc_values_phase_viscosity(phase, bg), bg
                )

    def boundary_flash(self) -> None:
        """If a flash procedure is provided, this method performs
        the p-T flash on the Dirichlet boundary, where pressure and temperature are
        positive.

        The values are stored to represent the secondary expressions on the boundary for
        e.g., upwinding.

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        It stored the results in :attr:`boundary_fluid_state` to be accessed by various
        ``bc_values_*`` methods.

        Important:
            If p or T are non-positive, the respective secondary expressions are stored
            as zero. Might have some implications for the simulation in weird cases.

        Raises:
            AssertionError: If no flash is provided.
            ValueError: If temperature or feed fractions are not positive where
                required.
            ValueError: If the flash did not succeed everywhere.

        """
        if not hasattr(self, "flash"):
            raise ppc.CompositeModellingError(
                "Attempting to call the flash on the boundary when no flash is included"
                + " in the model."
            )

        nphase = self.fluid_mixture.num_components
        ncomp = np.array(
            [phase.num_components for phase in self.fluid_mixture.phases], dtype=int
        )
        ptype = np.array(
            [len(phase.type) for phase in self.fluid_mixture.phases], dtype=int
        )

        for sd in self.mdg.subdomains():
            bg = self.mdg.subdomain_to_boundary_grid(sd)
            # 0D grids are skipped
            if bg is None:
                continue
            # grids without cells (boundaries of lines) have empty states
            elif bg.num_cells == 0:
                boundary_state = ppc.FluidState()
            # if at least 1 cell, perform flash.
            else:
                bg = cast(pp.BoundaryGrid, bg)

                # indexation on boundary grid
                # equilibrium is computable where pressure is given and positive
                dbc = self.bc_type_darcy_flux(sd).is_dir
                # reduce vector with all faces to vector with boundary faces
                bf = self.domain_boundary_sides(sd).all_bf
                dbc = dbc[bf]
                p = self.bc_values_pressure(bg)
                dir_idx = dbc & (p > 0.0)

                # set zero values if not required anywhere (completeness)
                if not np.any(dir_idx):
                    boundary_state = ppc.initialize_fluid_state(
                        bg.num_cells, ncomp, nphase, ptype, with_derivatives=False
                    )
                else:
                    # BC consistency checks ensure that z, T are non-trivial where p is
                    # non-trivial
                    T = self.bc_values_temperature(bg)
                    feed = [
                        self.bc_values_overall_fraction(comp, bg)
                        for comp in self.fluid_mixture.components
                    ]

                    boundary_state, success, _ = self.flash.flash(
                        z=[z[dir_idx] for z in feed],
                        p=p[dir_idx],
                        T=T[dir_idx],
                        parameters=self.flash_params,
                    )

                    if not np.all(success == 0):
                        raise ValueError("Boundary flash did not succeed.")

                    # Broadcast values into proper size for each boundary grid
                    boundary_state = _prolong_boundary_state(
                        boundary_state, bg.num_cells, dir_idx
                    )

            # storing state on boundary for BC value updates
            self.boundary_fluid_state[bg] = boundary_state

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
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component.

        Required together with BC for pressure and temperature to perform the boundary
        flash on the Dirichlet boundary with positive p and T.

        Parameters:
            component: A component in the fluid mixture.
            boundary_grid: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(boundary_grid.num_cells)

    ### BC which need to be provided in case no equilibrium calculations are included.
    ### Default updates uses the data stored after the boundary flash
    ### If no data is provided, zero arrays are returned

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

    def bc_values_relative_fraction(
        self, component: ppc.Component, phase: ppc.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for fractions of a component in a phase on the Dirichlet
        boundary, for models which do not have equilibrium calculations.

        Note:
            For models with equilibrium calculations, this is used for BC values of
            extended fractions.
            For models without equilibrium calculations, this is used for BC values of
            partial fractions.

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
            comps = [c for c in phase]
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

        It stors values at the curent time (iterate index 0).

        Override for initializing more variable values

        """

        for sd in self.mdg.subdomains():
            # setting pressure and temperature per domain
            p = self.intial_pressure(sd)
            T = self.initial_temperature(sd)

            self.equation_system.set_variable_values(
                p, [self.pressure([sd])], iterate_index=0
            )
            self.equation_system.set_variable_values(
                T, [self.temperature([sd])], iterate_index=0
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
                    z_i, [comp.fraction([sd])], iterate_index=0
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

        Values are stored in the iterate index 0 (current time).

        This method also provides the first iterate values for secondary expressions
        In the basic set-up this includes the phase properties and fugacity
        coefficients.

        """

        for sd in self.mdg.subdomains():
            # pressure, temperature and overall fractions
            p = self.intial_pressure(sd)
            T = self.initial_temperature(sd)
            z = [
                self.initial_overall_fraction(comp, sd)
                for comp in self.fluid_mixture.components
            ]

            # computing initial equilibrium
            state, success, _ = self.flash.flash(
                z, p=p, T=T, parameters=self.flash_params
            )

            if not np.all(success == 0):
                raise ValueError(f"Initial equilibriam not successful on grid {sd}")

            # setting initial values for enthalpy
            # NOTE that in the initialization, h is dependent compared to p, T, z
            self.equation_system.set_variable_values(
                state.h, [self.enthalpy([sd])], iterate_index=0
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
                        state.y[j], [phase.fraction([sd])], iterate_index=0
                    )
                    self.equation_system.set_variable_values(
                        state.sat[j], [phase.saturation([sd])], iterate_index=0
                    )
                # extended fractions
                for k, comp in enumerate(phase.components):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[k],
                        [phase.fraction_of[comp]([sd])],
                        iterate_index=0,
                    )

                # phase properties and their derivatives
                phase.density.progress_iterate_values_on_grid(state.phases[j].rho, sd)
                phase.volume.progress_iterate_values_on_grid(state.phases[j].v, sd)
                phase.enthalpy.progress_iterate_values_on_grid(state.phases[j].h, sd)
                phase.viscosity.progress_iterate_values_on_grid(state.phases[j].mu, sd)
                phase.conductivity.progress_iterate_values_on_grid(
                    state.phases[j].kappa, sd
                )

                phase.density.progress_iterate_derivatives_on_grid(
                    state.phases[j].drho, sd
                )
                phase.volume.progress_iterate_derivatives_on_grid(
                    state.phases[j].dv, sd
                )
                phase.enthalpy.progress_iterate_derivatives_on_grid(
                    state.phases[j].dh, sd
                )
                phase.viscosity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dmu, sd
                )
                phase.conductivity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dkappa, sd
                )

                # fugacities
                for k, comp in enumerate(phase.components):
                    phase.fugacity_of[comp].progress_iterate_values_on_grid(
                        state.phases[j].phis[k], sd
                    )
                    phase.fugacity_of[comp].progress_iterate_derivatives_on_grid(
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

    The following parameters are required:

    - ``'equilibrium_type'``: See :attr:`equilibrium_type` and read its documentation.

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
    - ``'has_time_dependent_boundary_equilibrium'``: Defaults to False. If True, the
      solution strategy includes an update of boundary conditions using a p-T flash in
      every time step. Otherwise it assumes the boundary flash needs to be performed
      only once at the beginning. Meaningless if no equilibrium condition is defined.
      In that case the user must provide values
      (see :class:`BoundaryConditionsCompositionalFlow`).

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

    _phase_fraction_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
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
    progress_all_constitutive_expressions_in_time: Callable[[Optional[bool]], None]
    """Provided by :class:`ConstitutiveLawsCompositionalFlow`."""
    update_all_constitutive_expressions: Callable[[Optional[bool]], None]
    """Provided by :class:`ConstitutiveLawsCompositionalFlow`."""

    primary_equation_names: list[str]
    """Provided by :class:`EquationsCompositionalFlow`."""
    primary_variable_names: list[str]
    """Provided by :class:`VariablesCompositionalFlow`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations = list()

        self._constitutive_expressions: dict[
            str,
            tuple[
                ppc.SecondaryExpression,
                pp.GridLikeSequence,
                Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
            ],
        ] = dict()
        """Storage for secondary expressions which need to be covered by a separate
        update."""

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid mixture."""

        self.equilibrium_type: Literal["p-T", "p-h", "v-h"] | None = params[
            "equilibrium_type"
        ]
        """A string denoting the two state functions which are assumed constant in the
        local (phase) equilibrium problem.

        Important:
            The user **must** provide a value explictly in the input parameters.

        **If defined:**

            The strategy assumes there are equilibrium calculations and performes the
            flash before every non-linear iteration to update the values of secondary
            variables and (dependent) thermodynamic properties.

        **If set to ``None``**:

        The framework assumes there are no local equilibrium conditions, hence no flash.
        It uses then the framework of secondary equations and expressions to update
        the values of dependent thermodynamic properties before each non-linear
        iteration

        Examples:
            1. If set to ``'p-h'`` the basic framework is fully functional and includes
               the all molar fractions and local equilibrium equations.
            2. If set to ``None``, the basic framework has dangling variables
               (temperature, phase saturations and partial fractions), and the model
               needs to be closed using secondary equations.
            3. If set to ``'p-T'`` the basic framework has a dangling variable, the
               enthalpy of the fluid mixture, and needs to be closed with a constitutive
               law as a secondary equation.

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

        # Input validation for set-up
        if (
            self.equilibrium_type is None
            and self.has_time_dependent_boundary_equilibrium
        ):
            raise ppc.CompositeModellingError(
                f"Conflicting model set-up: Time-dependent boundary flash calculations"
                + f" requested but no equilibrium type defined."
            )
        if not self.eliminate_reference_component:
            warnings.warn(
                "Reference component (and its fraction) are not eliminated."
                + " The basic model needs to be closed (unity constraint)."
            )
        if not self.eliminate_reference_phase:
            warnings.warn(
                "Reference phase (and its saturation) are not eliminated."
                + " The basic model needs to be closed (unity constraint)."
            )

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
        # If equilibrium defined, set the flash class
        if self.equilibrium_type is not None:
            self.set_up_flasher()

        # initial_condition calls a BC update, and we must check its consistency first
        self.check_bc_consistency()

        self.initial_condition()
        self.reset_state_from_file()  # TODO check if this is in conflict with init vals
        self.set_equations()

        # If equilibrium defined, compute the initial and boundary equilibrium
        # to set values
        # NOTE This must be done after set_equations, so that the secondary expressions
        # know on which grid they are defined
        if self.equilibrium_type is not None:
            self.initial_flash()
            self.boundary_flash()
        # If no equilibrium type, update the constitutive expressions which were added
        # by the user
        else:
            self.update_all_constitutive_expressions(True)
        self.initialize_timestep_and_iterate_indices()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()
        self.save_data_time_step()

    def initialize_timestep_and_iterate_indices(self) -> None:
        """Copies the initial values to all time step and iterate indices on all grids.

        This is done after the initialization, to populate the data dictionaries.
        It is performed for all variables, and all secondary expressions.

        This method assumes that the initial values are stored at the the iterate index
        0 (current time step, most recent iterate).

        Note:
            Derivative values are not copied to other time indices or iterate indices
            since they are not accessed by this solution strategy.

        Note:
            This method progresses time step values for phase properties
            **on all subdomains**, including

            - phase densities
            - phase volumes
            - phase enthalpies

            Viscosities, conductivities and fugacities are not progressed in time,
            since they are not expected in the accumulation term.

            It also progresses boundary values of phase properties on
            **all boundary grids**, for which boundary values are required.
            That **excludes** the fugacity coefficients.

        Can be customized by the user.

        """

        subdomains = self.mdg.subdomains()
        # updating variable values from current time step, to all previous and iterate
        val = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_iterate_values()
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        # updateing constitutive expressions on domains

        # copying the current value of secondary expressions to all indices
        # NOTE Only values, not derivatives
        for phase in self.fluid_mixture.phases:
            # phase properties and their derivatives on each subdomain
            rho_j = phase.density.subdomain_values
            v_j = phase.volume.subdomain_values
            h_j = phase.enthalpy.subdomain_values
            mu_j = phase.viscosity.subdomain_values
            kappa_j = phase.conductivity.subdomain_values

            # all properties have iterate values, use framework from sec. expressions
            # to push back values
            for _ in self.iterate_indices:
                phase.density.subdomain_values = rho_j
                phase.volume.subdomain_values = v_j
                phase.enthalpy.subdomain_values = h_j
                phase.viscosity.subdomain_values = mu_j
                phase.conductivity.subdomain_values = kappa_j

            # all properties have time step values, progress sec. exp. in time
            for _ in self.time_step_indices:
                phase.density.progress_values_in_time(subdomains)
                phase.volume.progress_values_in_time(subdomains)
                phase.enthalpy.progress_values_in_time(subdomains)
            # NOTE viscosity and conductivity are not progressed in time

            # fugacity coeffs
            # NOTE their values are not progressed in time.
            for comp in phase:
                phi = phase.fugacity_of[comp].subdomain_values
                d_phi = phase.fugacity_of[comp].subdomain_derivatives

                for _ in self.iterate_indices:
                    phase.fugacity_of[comp].subdomain_values = phi
                    phase.fugacity_of[comp].subdomain_derivatives = d_phi

            # properties have also (time-dependent) values on boundaries
            # NOTE the different usage of progressing in time on boundaries
            # NOTE fugacities have no boundary values
            bc_rho_j = phase.density.boundary_values
            bc_v_j = phase.volume.boundary_values
            bc_h_j = phase.enthalpy.boundary_values
            bc_mu_j = phase.viscosity.boundary_values
            bc_kappa_j = phase.conductivity.boundary_values
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

    def solve_local_equilibrium_problem(self) -> ppc.FluidState:
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
        for j, phase in enumerate(self.fluid_mixture.phases):
            if (
                phase == self.fluid_mixture.reference_phase
                and self.eliminate_reference_phase
            ):
                continue
            self.equation_system.set_variable_values(
                state.sat[j], [self._phase_fraction_variable(phase)], iterate_index=0
            )
            self.equation_system.set_variable_values(
                state.y[j], [self._saturation_variable(phase)], iterate_index=0
            )

            for i, comp in enumerate(phase.components):
                self.equation_system.set_variable_values(
                    state.phases[j].x[i],
                    [self._relative_fraction_variable(comp, phase)],
                    iterate_index=0,
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

        # TODO The resulting fluid enthalpy can change due to numerical
        # precision. Should it be updated as well?

        return state

    def update_thermodynamic_properties(self, fluid_state: ppc.FluidState) -> None:
        """Method to update various thermodynamic properties of present phases, on all
        subdomains **in the iterative sense**.

        This is meant for thermodynamic properties which appear in the PDEs, and inclue

        - phase densities,
        - phase volumes,
        - phase enthalpies,
        - phase viscosities,
        - phase conductivities,

        as well as their derivatives w.r.t. their dependencies.

        ``fluid_state`` comes either from the flash (if equilibrium is defined),
        or from an separate evaluation of secondary quantities if no equilibrium
        conditions defined.

        Called before every non-linear iteration to update the values of the secondary
        expressions.

        Note:
            Fugacity coefficients are only updated if there is a local equilibrium
            formulation.

        """

        # updating the phase properties, values and derivatives in the iterative sense
        for phase, state in zip(self.fluid_mixture.phases, fluid_state.phases):
            phase.density.subdomain_values = state.rho
            phase.volume.subdomain_values = state.v
            phase.enthalpy.subdomain_values = state.h
            phase.viscosity.subdomain_values = state.mu
            phase.conductivity.subdomain_values = state.kappa

            # extend derivatives from partial to extended fractions.
            # NOTE This revers the hack performed by the composite mixins when creating
            # secondary expressions which depend on extended fractions (independent)
            # quantities, but should actually depend on partial fractions (dependent).
            if self.equilibrium_type is not None:
                x = np.array(
                    [
                        self.equation_system.get_variable_values(
                            [self._relative_fraction_variable(comp, phase)]
                        )
                        for comp in phase
                    ]
                )

                for k, comp in enumerate(phase.components):
                    phase.fugacity_of[comp].subdomain_values = state.phis[k]
                    phase.fugacity_of[comp].subdomain_derivatives = _extend(
                        state.dphis[k], x
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

    def evaluate_secondary_expressions(self) -> ppc.FluidState:
        """Method called instead of the flash to evaluate secondary expressions
        which appear in the flow and transport formulation, depending on their
        quantities.

        For phase properties it uses

        Uses the numerical functions added by
        :class:`ConstitutiveLawsCompositionalFlow` to evaluate the properties by
        """

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
        """Updates the values (and values only) of phase properties on all subdomains in
        time, using the current iterate value, and the framework for secondary
        expressions.

        Note:
            The derivatives are not updated in time, since not required here.

            It also progresses only properties in time, which are expected in the
            accumulation terms:

            - phase density
            - phase volume
            - phase enthalpy

        """
        subdomains = self.mdg.subdomains()
        for phase in self.fluid_mixture.phases:
            phase.density.progress_values_in_time(subdomains)
            phase.volume.progress_values_in_time(subdomains)
            phase.enthalpy.progress_values_in_time(subdomains)

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform the p-h flash as a predictor step.

        Subsequently it computes the fluxes for various Upwind discretiztions
        (without calling the parent methods of mass and energy though, to save time).

        Finally, it calles the base class' method to update discretization parameters
        and to re-discretize.

        """

        # Flashing the mixture as a predictor step, if equilibrium defined
        if self.equilibrium_type is not None:
            fluid_state = self.solve_local_equilibrium_problem()
            self.update_thermodynamic_properties(fluid_state)
        # Otherwise we evaluate the secondary expression according to their dependencies
        # and provided functions
        else:
            self.update_all_constitutive_expressions()

        # After updating the fluid properties, update discretizations
        self.update_discretizations()

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Calls :meth:`progress_thermodynamic_properties_in_time` at the end."""
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.progress_thermodynamic_properties_in_time()

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        This method performs a Schur complement elimination.

        Uses the primary equations defined in
        :meth:`EquationsCompositionalFlow.primary_equation_names` and the primary
        variables defined in
        :meth:`VariablesCompositionalFlow.primary_variable_names`.

        """
        t_0 = time.time()
        # TODO block diagonal inverter for secondary equations
        self.linear_system = self.equation_system.assemble_schur_complement_system(
            self.primary_equation_names, self.primary_variable_names
        )
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        sol = super().solve_linear_system()
        return self.equation_system.expand_schur_complement_solution(sol)


# endregion


class CompositionalFlow(  # type: ignore[misc]
    # const. laws on top to overwrite what is used in inherited mass and energy balance
    ConstitutiveLawsCompositionalFlow,
    ppc.FlashMixin,
    EquationsCompositionalFlow,
    VariablesCompositionalFlow,
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
    - solute fractions for pure transport without equilibrium (if any)

    The secondary, local variables are:

    - ``num_phases - 1`` saturations per independent phase
    - ``num_phases - 1`` molar phase fractions per independent phase
    - ``num_phases * num_comp`` extended fractions of components in phases
    - temperature

    The primary block of equations consists of:

    - pressure equation / transport of total mass
    - energy balance / transport of total energy
    - ``num_comp - 1`` transport equations for each independent component
    - solute transport equations

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

    Note:
        The model inherits the md-treatment of Darcy flux, advective enthalpy flux and
        Fourier flux. Some interface variables and interface equations are introduced
        there. They are treated as secondary equations and variables in the basic model.

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
