"""Model mixins for multi-phase, multi-component flow, also denoted as compositional
flow (CF).

Provides means to formulate pressure equation, mass balance equations per component
and a total energy equation, using a fractional flow formulation with molar quantities.

Primary variables are pressure, specific fluid enthalpy and overall fractions,
as well as solute fractions for purely transpored solutes.

By default, the model is not closed, since saturations and temperature are dangling
variables.
The user needs to close models by providing a constitutive expression
for them and eliminating the variables by a local, algebraic equation.

"""

from __future__ import annotations

import logging
import time
import warnings
from functools import partial
from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc
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
        implemented by the user in :class:`BoundaryConditionsCF`.
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
            In a fractional flow formulation with a single darcy flux (total flux), this
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

        # diffusive_tensor = self.isotropic_second_order_tensor(
        #     subdomains, self.total_mobility(subdomains)
        # ) @ self.permeability(subdomains)

        # TODO This does not cover anisotropic permeability tensors and tensors with
        # dependencies. In that case one needs to matrix multiply the total diffusive
        # tensor on cells, with (boundary) face normals for this to be projectable.

        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(
            self.solid.permeability(), size, name="permeability"
        )
        diffusive_tensor = self.total_mobility(subdomains) * permeability

        normal_permeability = projection.secondary_to_mortar_avg @ (diffusive_tensor)
        normal_permeability.set_name("normal_permeability")
        return normal_permeability

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        Parameters:
            saturation: Operator representing the saturation of a phase.

        Returns:
            The base class method implements the linear law ``saturation``.

        """
        return saturation

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
    """Provided by :class:`VariablesCF`."""

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

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCF`."""

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
            # # NOTE The advected enthalpy (Neumann-type flux) must be consistent with
            # # the total mass flux
            # sds = [g.parent for g in domains]
            # f = self.advective_weight_enthalpy_flux(domains)
            # m_b = self.darcy_flux(sds)
            # op = f * (
            #     pp.ad.BoundaryProjection(self.mdg, sds).subdomain_to_boundary @ m_b
            # )
            op = self.advective_weight_enthalpy_flux(domains) * self.darcy_flux(domains)
            return op

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                "Domains must consist entirely of subdomains for the enthalpy flux."
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        # NOTE Boundary conditions are different from the pressure equation
        # This is consistent with the usage of darcy_flux in advective_flux
        boundary_operator_enthalpy = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=domains,
                dirichlet_operator=self.advective_weight_enthalpy_flux,
                neumann_operator=self.enthalpy_flux,
                bc_type=self.bc_type_advective_flux,
                name="bc_values_enthalpy_flux",
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
    """Provided by :class:`SolutionStrategyCF`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCF`."""

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
            # NOTE consistent Neumann-type flux based on the total flux
            op = self.advective_weight_component_flux(
                component, domains
            ) * self.darcy_flux(domains)
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
        weight_inlet_bc = partial(self.advective_weight_component_flux, component)
        weight_inlet_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            weight_inlet_bc,
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

        # NOTE Boundary conditions are different from the pressure equation
        # This is consistent with the usage of darcy_flux in advective_flux
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=weight_inlet_bc,
            neumann_operator=fluid_flux_neumann_bc,
            bc_type=self.bc_type_advective_flux,
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
           as local equations.
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

    time_step_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""
    iterate_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""

    add_constitutive_expression: Callable[
        [
            pp.ad.MixedDimensionalVariable,
            ppc.SecondaryExpression,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
            pp.GridLikeSequence,
        ],
        None,
    ]
    """Provided by :class:`ConstitutiveLawsCF`."""

    def set_equations(self) -> None:
        """Inherit this method to set additional secondary expressions in equation form

        .. math::

            f(x) = 0

        by setting the left-hand side as an equation in the Ad framework.

        """

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
        """Method to add a secondary equation eliminating a secondary variable by some
        constitutive law dependeng on **primary variables**.

        Secondary variables are assumed to be formally independent quantities in the
        AD sense.

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
            dependencies: First order dependencies (primary variables) through which
                :math:`\\varphi` is expressed locally.
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

                Note that a quantity can be eliminated on the boundary as well!

            dofs: ``default={'cells':1}``

                Argument for when adding above equation to the equation system.

        """
        # separate these two because Boundary values for independent quantities are
        # stored differently
        non_boundaries = [g for g in domains if isinstance(g, (pp.Grid, pp.MortarGrid))]

        sec_var = independent_quantity(non_boundaries)
        g_ids = [d.id for d in non_boundaries]

        sec_expr = ppc.SecondaryExpression(
            name=f"secondary_expression_for_{sec_var.name}_on_grids_{g_ids}",
            mdg=self.mdg,
            dependencies=dependencies,
            time_step_depth=len(self.time_step_indices),
            iterate_depth=len(self.iterate_indices),
        )

        local_equ = sec_var - sec_expr(non_boundaries)
        local_equ.set_name(f"elimination_of_{sec_var.name}_on_grids_{g_ids}")
        self.equation_system.set_equation(local_equ, non_boundaries, dofs)

        self.add_constitutive_expression(sec_var, sec_expr, func, domains)


class PrimaryEquationsCF(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation_h,
    SoluteTransportEquations,
    ComponentMassBalanceEquations,
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
        TotalMassBalanceEquation.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)
        SoluteTransportEquations.set_equations(self)
        TotalEnergyBalanceEquation_h.set_equations(self)


class VariablesCF(
    mass_energy.VariablesFluidMassAndEnergy,
    ppc.CompositeVariables,
):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    enthalpy_variable: str
    """Provided by :class:`SolutionStrategyCF`."""

    @property
    def primary_variable_names(self) -> list[str]:
        """Returns a list of primary variables, which in the basic set-up consist
        of

        1. pressure,
        2. overall fractions,
        3. solute fractions,
        4. fluid enthalpy.

        Primary variable names are used to define the primary block in the Schur
        elimination in the solution strategy.

        """
        return (
            [
                self.pressure_variable,
            ]
            + self.overall_fraction_variables
            + self.solute_fraction_variables
            + [
                self.enthalpy_variable,
            ]
        )

    @property
    def secondary_variables_names(self) -> list[str]:
        """Returns a list of secondary variables, which is defined as the complement
        of :meth:`primary_variable_names` and all variables found in the equation
        system."""
        secondary_variables = [var.name for var in self.equation_system.get_variables()]
        [secondary_variables.remove(var) for var in self.primary_variable_names]
        return secondary_variables

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        1. Sets up the pressure and temperature variables from standard mass and energy
           transport models.
        3. Creates the transported enthalpy variable.
        4. Creates all compositional variables.

        """
        # pressure and temperature. This covers also the interface variables for
        # Fourier flux, Darcy flux and enthalpy flux.
        mass_energy.VariablesFluidMassAndEnergy.create_variables(self)

        # enthalpy variable
        self.equation_system.create_variables(
            self.enthalpy_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "J * mol ^ -1"},
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


class ConstitutiveLawsCF(
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

    It provides functionalities to register secondary expressions for secondary
    variables, which are locally eliminated by some constitituve law.

    All other constitutive laws are analogous to the underlying mass and energy
    transport.

    """

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            ppc.SecondaryExpression,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid | pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """Provided by :class:`SolutionStrategyCF`"""

    def add_constitutive_expression(
        self,
        primary: pp.ad.MixedDimensionalVariable,
        expression: ppc.SecondaryExpression,
        func: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
        grids: pp.GridLikeSequence,
    ) -> None:
        """Register a secondary expression with the model framework to have it's
        update automatized.

        Updates in the iterative sense are performed before every non-linear iteration.

        Updates in the time sense, are performed after the non-linear iterations
        converge.

        Parameters:
            primary: The formally independent Ad operator which was eliminated by the
                expression.
            expression: The secondary expression which eliminates ``primary``, with
                some dependencies on primary variables.
            func: A numerical function returning value and derivative values to be
                inserted into ``expression`` when updateing.

                The derivative values must be a 2D array with rows consistent with the
                number of dependencies in ``expression``.
            grids: A sequence of grids on which it was eliminated.

        """
        boundaries = [g for g in grids if isinstance(g, pp.BoundaryGrid)]
        domains = [g for g in grids if isinstance(g, (pp.Grid, pp.MortarGrid))]
        self._constitutive_eliminations.update(
            {expression.name: (primary, expression, func, domains, boundaries)}
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
        for _, expr, func, domains, _ in self._constitutive_eliminations.values():
            for g in domains:
                X = [x([g]).value(self.equation_system) for x in expr._dependencies]

                vals, diffs = func(*X)

                expr.progress_iterate_values_on_grid(vals, g)
                if update_derivatives:
                    expr.progress_iterate_derivatives_on_grid(diffs, g)

    def progress_all_constitutive_expressions_in_time(
        self, progress_derivatives: bool = False
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
        for _, expr, _, domains, _ in self._constitutive_eliminations.values():
            expr.progress_values_in_time(domains)
            if progress_derivatives:
                expr.progress_derivatives_in_time(domains)


# endregion
# region SOLUTION STRATEGY, including separate treatment of BC and IC


class BoundaryConditionsCF(
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
):
    """Mixin treating boundary conditions for the compositional flow.

    Atop of inheriting the treatment for single phase flow (which is exploited for the
    total mass balance) and the energy balance (total energy balance), this class has
    a treatment for BC for fractional unknowns and phase properties as secondary
    expressions.

    **Essential BC**:

    In any case, the user must provide BC values for primary variables on the Dirichlet
    boundary. They are also used to compute values of phase properties on the boundary.

    On the Neumann boundary, the user must provide values for non-linear weights
    in various advective fluxes (fractional mobilities f.e.).

    **Other BC**:

    Since the modelling framework always introduces saturations and some relative
    fractions as independent variables, the user must also provide values for them
    on the Dirichlet BC.

    Important:

        This class resolves some inconsistencies in the parent methods regarding the bc
        type definition. There can be only one definition of Dirichlet and Neumann
        boundary for the fluxes based to Darcy's Law (advective fluxes), and is to be
        set in :meth:`bc_type_darcy_flux`. Other bce type definition for advective
        fluxes are overriden here and point to the bc type for the darcy flux.

        The Fourier flux can have Dirichlet type BC as well, but it is required that the
        Dirichlet faces of the darcy flux are contained in the Dirichlet faces for the
        Fourier flux, for consistency reasons.
        I.e., where Dirichlet pressure is given, Dirichlet temperature must be given.

    """

    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    eliminate_reference_phase: bool
    """Provided by :class:`SolutionStrategyCF`."""
    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCF`."""
    enthalpy_variable: str
    """Provided by :class:`SolutionStrategyCF`."""

    _overall_fraction_variable: Callable[[ppc.Component], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _solute_fraction_variable: Callable[[ppc.ChemicalSpecies, ppc.Compound], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            ppc.SecondaryExpression,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid | pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """Provided by :class:`SolutionStrategyCF`"""

    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of hyperbolic boundary condition in the advective flux.

        Consider an advective flux :math:`f \\mathbf{m}`.
        While the BC type for :math:`\\mathbf{m}` is defined by
        :meth:`bc_type_darcy_flux`, the user can define Dirichlet-type faces where
        mass or energy in form of :math:`f` enters the system, independent of the values
        of :math:`\\mathbf{m}` (which can be zero).

        Note:
            Mass as well as energy are advected.

        Important:
            Due to how Upwinding is implemented, the boundaries here must all be flagged
            as `dir`, though the concept of Dirichlet and Neumann is not applicable
            here. This function should not be modified by the user, but it is left here
            to fix inconsistencies with parent methods used for advective fluxes.

        Base implementation sets all faces to Dirichlet-type.

        """
        return pp.BoundaryCondition(sd, self.domain_boundary_sides(sd).all_bf, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of the advective flux for consistency reasons."""
        return self.bc_type_advective_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of the advective flux for consistency reasons."""
        return self.bc_type_advective_flux(sd)

    def update_all_boundary_conditions(self) -> None:
        """On top of the parent methods for mass and energy models, it updates
        values for:

        1. primary variables on the Dirichlet boundary, and values for
           fractional weights in advective fluxes on the Neumann boundary.
           :meth:`update_essential_boundary_values`
        2. Consitutively eliminated quantities on the boundary
           :meth:`update_boundary_values_constitutive_eliminated`.
        3. phase properties which appear in the non-linear weights of various
           fluxes :meth:`update_boundary_values_phase_properties`.

        Important:
            Note that phase property values are computed using the underlying EoS.
            If their dependency includes f.e. enthalpy, the user must provide BC values
            for enthalpy even though it does not appear in the flux terms directly.

            It is either this (a consistent computation), or a bunch of other methods
            where users provide values for properties individually.

        Notes:
            1. Temperature is also a secondary variable for enthalpy-based formulations.
               Its update is taken care by the parent method for energy balance though.
            2. If the user provides a constitutive law for temperature, temperature
               values are also provided on the Dirichlet boundary, where the primary
               variables have values!
            3. If secondary variables are eliminated in terms of primary variables,
               the user has to ensure that the primary variables are defined on the
               boundary where the eliminated quantity is accessed.

        """
        # covers updates for pressure and temperature
        super().update_all_boundary_conditions()
        self.update_essential_boundary_values()
        self.update_boundary_values_constitutive_eliminated()
        self.update_boundary_values_phase_properties()

    def update_essential_boundary_values(self) -> None:
        """Method updating BC values of primary variables on Dirichlet boundary,
        and values of fractional weights in advective fluxes on Neumann boundary.

        This is separated, because the order matters: Must be first update.

        """
        for component in self.fluid_mixture.components:
            # Update of solute fractions on Dirichlet boundary
            if isinstance(component, ppc.Compound):
                for solute in component.solutes:
                    bc_vals = partial(self.bc_values_solute_fraction, solute, component)
                    bc_vals = cast(Callable[[pp.BoundaryGrid], np.ndarray], bc_vals)
                    self.update_boundary_condition(
                        self._solute_fraction_variable(solute, component),
                        function=bc_vals,
                    )
            # Skip if mass balance for reference component is eliminated.
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue

            # Update of overall fractions on Dirichlet boundary
            bc_vals = partial(self.bc_values_overall_fraction, component)
            bc_vals = cast(Callable[[pp.BoundaryGrid], np.ndarray], bc_vals)
            self.update_boundary_condition(
                name=self._overall_fraction_variable(component),
                function=bc_vals,
            )

        # Update of BC values for fluid enthalpy
        self.update_boundary_condition(
            name=self.enthalpy_variable, function=self.bc_values_enthalpy
        )

    def update_boundary_values_constitutive_eliminated(self) -> None:
        """Called by :meth:`update_all_boundary_conditions` to update the values of
        formerly independent quantities, which were eliminated on some boundaries.

        Uses the parent method :meth:`update_boundary_condition` assuming that the
        quantity is used correspondingly

        """

        for elim_var, expr, func, _, bgs in self._constitutive_eliminations.values():

            # skip if not eliminated on boundary
            if not bgs:
                continue

            def bc_values_prim(bg: pp.BoundaryGrid) -> np.ndarray:
                bc_vals: np.ndarray

                if bg in bgs:
                    X = [
                        x([bg]).value(self.equation_system) for x in expr._dependencies
                    ]
                    bc_vals, _ = func(*X)
                else:
                    bc_vals = np.zeros(bg.num_cells)

                return bc_vals

            self.update_boundary_condition(elim_var.name, bc_values_prim)

    def update_boundary_values_phase_properties(self) -> None:
        """Evaluates the phase properties using underlying EoS and progresses
        their values in time.

        This base method updates only properties which are expected in the non-linear
        weights of the avdective flux:

        - phase densities
        - phase volumes
        - phase enthalpies
        - phase viscosities

        Note that conductitivies are not updated, since the framework uses a consistent
        discretization of diffusive fluxes with non-linear tensors.

        Important:
            Due to the fractional flow formulation, values are required on all boundary
            faces, Neumann and Dirichlet.
            The fractional flow weights are multiplied on each face with the total flux.

            This implies, that the user must be aware that primary variables like
            pressure must be considered on all faces.

            Especially if they are zero, the underlying EoS
            (:meth:`~porepy.composite.base.Phase.compute_properties`) must be able
            to handle that input.

        """

        for bg in self.mdg.boundaries():
            for phase in self.fluid_mixture.phases:
                # some work is required for BGs with zero cells
                if bg.num_cells == 0:
                    rho_bc = np.zeros(0)
                    h_bc = np.zeros(0)
                    mu_bc = np.zeros(0)
                else:
                    dep_vals = [
                        d([bg]).value(self.equation_system)
                        for d in self.dependencies_of_phase_properties(phase)
                    ]
                    state = phase.compute_properties(*dep_vals)
                    rho_bc = state.rho
                    h_bc = state.h
                    mu_bc = state.mu

                # phase properties which appear in mobilities
                phase.density.update_boundary_value(rho_bc, bg)
                phase.enthalpy.update_boundary_value(h_bc, bg)
                phase.viscosity.update_boundary_value(mu_bc, bg)

                # volume as reciprocal of density, only where given
                v_bc = np.zeros_like(rho_bc)
                idx = rho_bc > 0
                v_bc[idx] = 1.0 / rho_bc[idx]
                phase.volume.update_boundary_value(v_bc, bg)

    ### BC values for primary variables which need to be given by the user in any case.

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """BC values for fluid enthalpy on the Dirichlet boundary.

        Important:
            Though strictly speaking not appearing in the flux terms, this method
            are required for completeness reasons.
            E.g., for cases where phase properties depend enthalpies.
            Phase properties **need** values on the Dirichlet boundary, to compute
            fractional weights in the advective fluxes.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing the value of
            the fluid enthalpy on the Dirichlet boundary.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            component: A component in the fluid mixture.
            boundary_grid: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_solute_fraction(
        self,
        solute: ppc.ChemicalSpecies,
        compound: ppc.Compound,
        boundary_grid: pp.BoundaryGrid,
    ) -> np.ndarray:
        """BC values for solute fractions (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            solute: A solute in the ``compound``.
            compound: A component in the fluid mixture.
            boundary_grid: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(boundary_grid.num_cells)


class InitialConditionsCF:
    """Class for setting the initial values in a compositional flow model.

    This mixin is introduced because of the complexity of the framework, guiding the
    user through what is required and dissalowing a "messing" with the order of methods
    executed in the model set-up.

    All method herein are part of the routine
    :meth:`SolutionStrategyCF.initial_conditions`.

    The basic initialization assumes that initial conditions are given for
    primary variables, and values for secondary variables are computed using
    registered constitutive expressions.

    As a final step of the initialization routine, initial values for
    secondary expressions representing phase properties are calculated using the
    provided EoS.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: ppc.Mixture
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

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
    """Provided by :class:`VariablesCF`."""

    eliminate_reference_component: bool
    """Provided by :class:`SolutionStrategyCF`."""
    eliminate_reference_phase: bool
    """Provided by :class:`SolutionStrategyCF`."""

    time_step_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""
    iterate_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""

    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            ppc.SecondaryExpression,
            Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid | pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """Provided by :class:`SolutionStrategyCF`"""

    def set_initial_values(self) -> None:
        """Collective method to initialize all values in the CF framework with
        secondary expressions and constitutive laws.

        1. Initializes primary variable values
           :meth:`set_initial_values_primary_variables`
        2. Initializes secondary variables and the secondary expressions by which
           they were eliminated
           :meth:`set_initial_values_constitutive_eliminated`
        3. Initializes phase properties by computing them using the underlying EoS
           :meth:`set_intial_values_phase_properties`
        4. Copies values of all independent variables to all time and iterate indices.

        """
        self.set_initial_values_primary_variables()
        self.set_initial_values_constitutive_eliminated()
        self.set_intial_values_phase_properties()

        # updating variable values from current time step, to all previous and iterate
        val = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_iterate_values()
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for primary variables at the current time
        (iterate index 0).

        The base method covers

        - pressure,
        - temperature,
        - fluid enthalpy,
        - independent overall fractions
        - solute fractions.

        """

        for sd in self.mdg.subdomains():
            # setting pressure and temperature per domain
            p = self.initial_pressure(sd)
            T = self.initial_temperature(sd)
            h = self.initial_enthalpy(sd)

            self.equation_system.set_variable_values(
                p, [self.pressure([sd])], iterate_index=0
            )
            self.equation_system.set_variable_values(
                T, [self.temperature([sd])], iterate_index=0
            )
            self.equation_system.set_variable_values(
                h, [self.enthalpy([sd])], iterate_index=0
            )

            # Setting overall fractions and solute fractions
            for comp in self.fluid_mixture.components:
                if isinstance(comp, ppc.Compound):
                    for solute in comp.solutes:
                        c = self.initial_solute_fraction(solute, comp, sd)
                        self.equation_system.set_variable_values(
                            c, [comp.solute_fraction_of[solute](sd)], iterate_index=0
                        )

                if (
                    comp == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue

                z_i = self.initial_overall_fraction(comp, sd)
                self.equation_system.set_variable_values(
                    z_i, [comp.fraction([sd])], iterate_index=0
                )

    def set_initial_values_constitutive_eliminated(self) -> None:
        """Sets the initial values of secondary variables which were eliminated by
        some constitutive expression.

        This assumes that the constitutive expressions depend on primary variables,
        not for mixed dependencies.

        Copies the value and derivative values of the secondary expression associated
        with the secondary variable to all time and iterate indices.
        The derivative values are only stored at the most recent iterate.

        """
        for secvar, expr, f, domains, _ in self._constitutive_eliminations.values():
            # store value for eliminated secondary variable globally
            dep_vals = [
                d(domains).value(self.equation_system) for d in expr._dependencies
            ]
            val, _ = f(*dep_vals)
            self.equation_system.set_variable_values(val, [secvar], iterate_index=0)

            # progressing values and derivatives iteratively on all grids
            # for the secondary expression
            for grid in domains:
                dep_vals_g = [
                    d([grid]).value(self.equation_system) for d in expr._dependencies
                ]
                val, diff = f(*dep_vals_g)
                for _ in self.iterate_indices:
                    expr.progress_iterate_values_on_grid(val, grid)
                # store the derivative value at the most recent iterate
                expr.progress_iterate_derivatives_on_grid(diff, grid)

            # progress values in time for all indices
            for _ in self.time_step_indices:
                expr.progress_values_in_time(domains)

    def set_intial_values_phase_properties(self) -> None:
        """Method to set the initial values and derivative values of phase
        properties, which are secondary expressions with some dependencies.

        This method also fills all time and iterate indices with the initial values.
        Derivative values are only stored for the current iterate.

        """
        subdomains = self.mdg.subdomains()

        for phase in self.fluid_mixture.phases:
            dep_vals = [
                d(subdomains).value(self.equation_system)
                for d in self.dependencies_of_phase_properties(phase)
            ]

            state = phase.compute_properties(*dep_vals)

            # propage values to all iterate indices
            for _ in self.iterate_indices:
                phase.density.subdomain_values = state.rho
                phase.volume.subdomain_values = state.v
                phase.enthalpy.subdomain_values = state.h
                phase.viscosity.subdomain_values = state.mu
                phase.conductivity.subdomain_values = state.kappa

            # set derivatives to current index.
            phase.density.subdomain_derivatives = state.drho
            phase.volume.subdomain_derivatives = state.dv
            phase.enthalpy.subdomain_derivatives = state.dh
            phase.viscosity.subdomain_derivatives = state.dmu
            phase.conductivity.subdomain_derivatives = state.dkappa

            # propagate values to all time step indices
            # Only for those with time depth
            for _ in self.time_step_indices:
                phase.density.progress_values_in_time(subdomains)
                phase.volume.progress_values_in_time(subdomains)
                phase.enthalpy.progress_values_in_time(subdomains)

    ### IC for primary variables which need to be given by the user in any case.

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """
        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain. Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        """
        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain. Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        """
        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial specific fluid enthalpy values on that subdomain.
            Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

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
            Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

    def initial_solute_fraction(
        self, solute: ppc.ChemicalSpecies, compound: ppc.Compound, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            component: A component in the fluid mixture with an independent feed
                fraction.
            sd: A subdomain in the md-grid.

        Returns:
            The initial solute fraction values for a solute in a compound on a
            subdomain.
            Defaults to zero array.

        """
        return np.zeros(sd.num_cells)


class SolutionStrategyCF(
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


    The base class checks if ``equilibrium_type`` is set as an attribute.
    If not, it sets it to None for consistency with the remaining framework.

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

    create_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    assign_thermodynamic_properties_to_mixture: Callable[[], None]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""
    set_initial_values: Callable[[], None]
    """Provided by :class:`InitialConditionsCF`."""
    progress_all_constitutive_expressions_in_time: Callable[[Optional[bool]], None]
    """Provided by :class:`ConstitutiveLawsCF`."""
    update_all_constitutive_expressions: Callable[[Optional[bool]], None]
    """Provided by :class:`ConstitutiveLawsCF`."""
    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """Provided by :class:`~porepy.composite.composite_mixins.FluidMixtureMixin`."""

    primary_equation_names: list[str]
    """Provided by :class:`EquationsCompositionalFlow`."""
    primary_variable_names: list[str]
    """Provided by :class:`VariablesCF`."""

    isotropic_second_order_tensor: Callable[
        [list[pp.Grid], pp.ad.Operator], pp.SecondOrderTensor
    ]
    """Provided by :class:`~porepy.models.constitutive_laws.SecondOrderTensorUtils`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Provided by :class:`BoundaryConditionsCF`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations = list()

        self._constitutive_eliminations: dict[
            str,
            tuple[
                pp.ad.MixedDimensionalVariable,
                ppc.SecondaryExpression,
                Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
                Sequence[pp.Grid, pp.MortarGrid],
                Sequence[pp.BoundaryGrid],
            ],
        ] = dict()
        """Storage of terms which were eliminated by some cosntitutive expression."""

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid mixture."""

        if not hasattr(self, "equilibrium_type"):
            self.equilibrium_type = None
        else:
            if self.equilibrium_type is None:
                raise ppc.CompositeModellingError(
                    "Cannot set the value of attribute `equilibrium_type` to None."
                    + " Use some valid description of equilibrium state, e.g. 'p-T'."
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

        # Input validation for set-up
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
        3. Then it creates the equations so that all secondary expressions are
           instantiated.

        """
        self.set_materials()
        self.set_geometry()
        self.initialize_data_saving()
        self.set_equation_system_manager()

        # This block is new and the order is critical
        self.create_mixture()
        self.create_variables()
        self.assign_thermodynamic_properties_to_mixture()

        self.set_equations()
        self.initial_condition()
        self.reset_state_from_file()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()
        self.save_data_time_step()

    def initial_condition(self) -> None:
        """Atop the parent methods, this method calles
        :meth:`InitialConditionsCF.set_initial_values` to initiate primary variables
        secondary variables, values of secondary expressions and phase properties
        properly."""
        super().initial_condition()
        self.set_initial_values()

    def add_nonlinear_flux_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of nonlinear flux discretizations.

        Important:
            The fluxes must be re-discretized before the upwinding is re-discretized,
            since the new flux values must be stored before upwinding is updated.

        Parameters:
            discretization: The nonlinear discretization to be added.

        """
        # This guardrail is very weak. However, the discretization list is uniquified
        # before discretization, so it should not be a problem.
        if discretization not in self._nonlinear_discretizations:
            self._nonlinear_flux_discretizations.append(discretization)

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
            self.advected_enthalpy_discretization(subdomains).upwind(),
        )
        self.add_nonlinear_discretization(
            self.interface_advected_enthalpy_discretization(interfaces).flux(),
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
                self.fractional_mobility_discretization(component, subdomains).upwind(),
            )
            self.add_nonlinear_discretization(
                self.interface_fractional_mobility_discretization(
                    component, interfaces
                ).flux(),
            )

        # TODO this is experimental and expensive
        self.add_nonlinear_flux_discretization(
            self.fourier_flux_discretization(subdomains).flux()
        )
        self.add_nonlinear_flux_discretization(
            self.darcy_flux_discretization(subdomains).flux()
        )

    def update_secondary_quantities(self) -> None:
        """Update of secondary quantities with evaluations performed outside the AD
        framework.

        This base method calls the update of all secondary expressions registered as
        constitutive laws
        (see :meth:`ConstitutiveLawsCF.add_constitutive_expression`
        and :meth:`ConstitutiveLawsCF.update_all_constitutive_expressions`),
        updating the the values and derivatives.

        Then it performs an update of thermodynmaic properties of phases by calling
        :meth:`update_thermodynamic_properties_of_phases`.

        """
        self.update_all_constitutive_expressions(True)
        self.update_thermodynamic_properties_of_phases()

    def update_thermodynamic_properties_of_phases(self) -> None:
        """This method uses for each phase the underlying EoS to calculate
        new values and derivative values of phase properties and to update them
        them in the iterative sense, on all subdomains."""

        subdomains = self.mdg.subdomains()

        for phase in self.fluid_mixture.phases:
            dep_vals = [
                d(subdomains).value(self.equation_system)
                for d in self.dependencies_of_phase_properties(phase)
            ]

            state = phase.compute_properties(*dep_vals)

            # Set current iterate indices of values and derivatives
            phase.density.subdomain_values = state.rho
            phase.volume.subdomain_values = state.v
            phase.enthalpy.subdomain_values = state.h
            phase.viscosity.subdomain_values = state.mu
            phase.conductivity.subdomain_values = state.kappa

            phase.density.subdomain_derivatives = state.drho
            phase.volume.subdomain_derivatives = state.dv
            phase.enthalpy.subdomain_derivatives = state.dh
            phase.viscosity.subdomain_derivatives = state.dmu
            phase.conductivity.subdomain_derivatives = state.dkappa

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
            mob = self.isotropic_second_order_tensor([sd], self.total_mobility([sd]))
            data[pp.PARAMETERS][self.darcy_keyword].update(
                {
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd, mob * self.permeability([sd]), self.solid.permeability()
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

    def progress_secondary_quantities_in_time(self) -> None:
        """Updates the values (and values only) of constitutive expressions in
        secondary equations, and phase properties on all subdomains in
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
        # progress only values in time
        self.progress_all_constitutive_expressions_in_time(False)

        subdomains = self.mdg.subdomains()
        for phase in self.fluid_mixture.phases:
            phase.density.progress_values_in_time(subdomains)
            phase.volume.progress_values_in_time(subdomains)
            phase.enthalpy.progress_values_in_time(subdomains)

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform an update of secondary quantities,
        and then performing customized updates of discretizations."""
        self.update_secondary_quantities()
        # After updating the fluid properties, update discretizations
        self.update_discretizations()

    def after_nonlinear_convergence(self) -> None:
        """Calls :meth:`progress_thermodynamic_properties_in_time` at the end."""
        super().after_nonlinear_convergence()
        self.progress_secondary_quantities_in_time()

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        This method performs a Schur complement elimination.

        Uses the primary equations defined in
        :meth:`EquationsCompositionalFlow.primary_equation_names` and the primary
        variables defined in
        :meth:`VariablesCF.primary_variable_names`.

        """
        t_0 = time.time()
        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)

        for name, eq in self.equation_system.equations.items():
            res = eq.value(self.equation_system)
            print(f"res {name}: ", np.linalg.norm(res))
        if reduce_linear_system_q:
            # TODO block diagonal inverter for secondary equations
            self.linear_system = self.equation_system.assemble_schur_complement_system(
                self.primary_equation_names, self.primary_variable_names
            )
        else:
            self.linear_system = self.equation_system.assemble()
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        sol = super().solve_linear_system()
        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            sol = self.equation_system.expand_schur_complement_solution(sol)
        return sol


# endregion


class CFModelMixin(  # type: ignore[misc]
    # const. laws on top to overwrite what is used in inherited mass and energy balance
    ConstitutiveLawsCF,
    PrimaryEquationsCF,
    VariablesCF,
    BoundaryConditionsCF,
    InitialConditionsCF,
    SolutionStrategyCF,
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
    - ``num_phases * num_comp`` relative fractions of components in phases
      (extended or partial)
    - temperature

    The primary block of equations consists of:

    - pressure equation / transport of total mass
    - energy balance / transport of total energy
    - ``num_comp - 1`` transport equations for each independent component
    - solute transport equations

    The secondary block of equations must be provided using constitutive relations
    or an equilibrium model for the fluid.

    Note:
        The model inherits the md-treatment of Darcy flux, advective enthalpy flux and
        Fourier flux. Some interface variables and interface equations are introduced
        there. They are treated as secondary equations and variables in the basic model.

    """
