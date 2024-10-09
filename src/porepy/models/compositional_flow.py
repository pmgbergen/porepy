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
import porepy.compositional as ppc

from . import energy_balance as energy
from . import mass_and_energy_balance as mass_energy

logger = logging.getLogger(__name__)


def update_phase_properties(
    grid: pp.Grid,
    phase: ppc.Phase,
    props: ppc.PhaseProperties,
    depth: int,
    update_derivatives: bool = True,
    use_extended_derivatives: bool = False,
) -> None:
    """Helper method to update the phase properties and its derivatives.

    This method is intended for a grid-local update of properties and their derivatives,
    using the methods of the
    :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`.

    Parameters:
        grid: A grid in the md-domain.
        phase: Any phase.
        props: A phase property structure containing the new values to be set.
        depth: An integer used to shift existing iterate values backwards.
        update_derivatives: ``default=True``

            If True, updates also the derivative values.
        use_extended_derivatives: ``default=False``

            If True, and if ``update_derivatives==True``, uses the extend
            derivatives of the ``state``.

            To be used in the the CFLE setting with the unified equilibrium formulation.

    """
    phase.density.progress_iterate_values_on_grid(props.rho, grid, depth=depth)
    phase.specific_enthalpy.progress_iterate_values_on_grid(props.h, grid, depth=depth)
    phase.viscosity.progress_iterate_values_on_grid(props.mu, grid, depth=depth)
    phase.conductivity.progress_iterate_values_on_grid(props.kappa, grid, depth=depth)

    if update_derivatives:
        if use_extended_derivatives:
            phase.density.set_derivatives_on_grid(props.drho_ext, grid)
            phase.specific_enthalpy.set_derivatives_on_grid(props.dh_ext, grid)
            phase.viscosity.set_derivatives_on_grid(props.dmu_ext, grid)
            phase.conductivity.set_derivatives_on_grid(props.dkappa_ext, grid)
        else:
            phase.density.set_derivatives_on_grid(props.drho, grid)
            phase.specific_enthalpy.set_derivatives_on_grid(props.dh, grid)
            phase.viscosity.set_derivatives_on_grid(props.dmu, grid)
            phase.conductivity.set_derivatives_on_grid(props.dkappa, grid)


# region CONSTITUTIVE LAWS taylored to pore.compositional and its mixins


class MobilityCF:
    """Mixin class defining mobilities for the balance equations in the CF setting, and
    which discretization to be used for the non-linear weights in advective fluxes.

    Flux discretizations are handled by respective constitutive laws.

    Provides various methods to assemble total, component and phase mobility, as well as
    fractional mobilities.

    Important:
        Mobility terms are designed to be representable also on boundary grids as user-
        given data.
        Values on the Neumann boundary (especially fractional mobilities) must be
        implemented by the user in :class:`BoundaryConditionsCF`.
        Those values are then consequently multiplied with boundary flux values in
        respective balance equations.
        TODO

    """

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """See :class:`SolidSkeletonCF`."""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """See :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.
    """

    mobility_keyword: str
    """Provided by
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`."""

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
            [
                self.phase_mobility(phase, domains)
                for phase in self.fluid_mixture.phases
            ],
            name,
        )
        return mobility

    def phase_mobility(
        self, phase: ppc.Phase, domains: pp.SubdomainsOrBoundaries
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
        mobility = (
            phase.density(domains)
            * self.relative_permeability(phase.saturation(domains))
            / phase.viscosity(domains)
        )
        mobility.set_name(name)
        return mobility

    def component_mobility(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
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
        mobility = pp.ad.sum_operator_list(
            [
                phase.partial_fraction_of[component](domains)
                * self.phase_mobility(phase, domains)
                for phase in self.fluid_mixture.phases
                if component in phase
            ],
            name,
        )
        return mobility

    def fractional_component_mobility(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
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
        self, phase: ppc.Phase, domains: pp.SubdomainsOrBoundaries
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

    def mobility_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.UpwindAd:
        """Discretization of mobility terms, to be based on the total mass flux.

        Hence the upwinding matrices are the same for all non-linear weights in any
        advective flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            AD represention of the upwinding discretization using the
            :attr:`mobility_keyword`.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_mobility_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of mobility terms based on the (total) interface Darcy flux.

        To be used to obtain a representation of all non-linear weights in any advective
        flux on the interfaces.

        See Also:
            :meth:`mobility_discretization`

        Parameters:
            interfaces: List of interface grids.

        Returns:
            AD-representation of the upwind-coupling discretization usint the
            :attr:`mobility_keyword`.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)


class ThermalConductivityCF(pp.constitutive_laws.ThermalConductivityLTE):
    """A constitutive law providing the fluid and normal thermal conductivity to be
    used with Fourier's Law in the compositional flow."""

    mdg: pp.MixedDimensionalGrid
    """See :class:`~porepy.models.geometry.ModelGeometry`."""

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

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
        """Normal thermal conductivity of the fluid.

        This is a constitutive law choosing the thermal conductivity of the fluid in the
        higher-dimensional domains as the normal conductivity.

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
    """See :class:`~porepy.models.geometry.ModelGeometry`."""

    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`MobilityCF`."""

    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """See :class:`~porepy.models.geometry.ModelGeometry`."""

    perturbation_from_reference: Callable[[str, list[pp.Grid]], pp.ad.Operator]
    """See :class:`~porepy.models.VariableMixin`."""

    temperature_variable: str
    """See :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`."""

    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference porosity.

        Parameters:
            subdomains: A list of subdomains.

        Returns:
            The constant solid porosity wrapped as an Ad scalar.

        """
        return pp.ad.Scalar(self.solid.porosity(), "reference_porosity")

    def diffusive_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Important:
            This implementation does not cover absolute permeabilities which are not
            constant.

        Parameters:
            subdomains: A list of subdomains

        Returns:
            The cell-wise, scalar, isotropic permeability, composed of the total
            mobility and the absolut permeability of the underlying solid.
            Used for the diffusive tensor in the fractional flow formulation.

        """
        abs_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability(),
            size=sum(sd.num_cells for sd in subdomains),
            name="absolute_permeability",
        )
        op = self.total_mobility(subdomains) * abs_perm
        op.set_name("isotropic_permeability")
        return op

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: A list of subdomains

        Returns:
            The value of :meth:`diffusive_permeability` wrapped into an isotropic,
            second-order tensor.

        """
        op = self.isotropic_second_order_tensor(
            subdomains, self.diffusive_permeability(subdomains)
        )
        op.set_name("diffusive_tensor_darcy")
        return op

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """A constitutive law returning the normal permeability as the product of
        total mobility and the permeability on the lower-dimensional subdomain.

        Parameters:
            interfaces: A list of mortar grids.

        Returns:
            The product of total mobility and permeability of the lower-dimensional.

        """

        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        normal_permeability = (
            projection.secondary_to_mortar_avg @ self.diffusive_permeability(subdomains)
        )
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
            * self.temperature(subdomains)
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
# region PDEs used in the (fractional) CF, taylored to pp.compositional and its mixins


class TotalMassBalanceEquation(pp.BalanceEquation):
    """Mixed-dimensional balance of total mass in a fluid mixture.

    Also referred to as *pressure equation*.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    Note:
        This balance equation assumes that the total mobility is part of the
        diffusive, second-order tensor in the non-linear (MPFA) discretization of the
        Darcy flux.

    """

    mdg: pp.MixedDimensionalGrid
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`SolidSkeletonCF`."""

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    interface_darcy_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""
    well_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.PiecmannWellFlux`."""

    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """See :class:`~porepy.models.geometry.ModelGeometry`."""

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
        """Pressure equation (or total mass balance) for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.volume_integral(
            self.fluid_mass(subdomains), subdomains, dim=1
        )
        flux = self.fluid_flux(subdomains)
        source = self.fluid_source(subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(TotalMassBalanceEquation.primary_equation_name())
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        r"""Returns the accumulation term in the pressure equation
        :math:`\Phi\rho`, using the :attr:`~SolidSkeletonCF.permeability` and the
        :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid mixture,
        in AD operator form on a given set of ``subdomains``."""
        mass_density = self.fluid_mixture.density(subdomains) * self.porosity(
            subdomains
        )
        mass_density.set_name("total_fluid_mass")
        return mass_density

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """The fluid flux is given by the :attr:`darcy_flux`, assuming the total
        mobility is part of the (non-linear) diffuse tensor."""
        return self.darcy_flux(domains)

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Source terms in the pressure equation, accounting for the interface fluxes
        from higher-dimensional neighbouring subdomains, as well as from wells.

        The interface flux variable and the well flux variable are used to account for
        mass passing between subdomains and wells.

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
    independent (specific) fluid enthalpy variable.

    Balance equation for all subdomains and advective and diffusive fluxes
    (Fourier flux) internally and on all interfaces of codimension one and advection on
    interfaces of codimension two (well-fracture intersections).

    Defines an advective weight to be used in the advective flux, assuming the total
    mobility is part of the diffusive tensor in the pressure equation.

    Notes:
        1. Since enthalpy is an independent variable, models using this balance need
           a local equation relating temperature to the fluid enthalpy.
        2. (Developers) Room for unification with :mod:`~porepy.models.energy_balance`.
        3. This class relies on an interface enthalpy flux variable, which is put into
           relation with the interface darcy. It can be eliminated by using the
           interface darcy flux, weighed with the transported enthalpy. TODO

    """

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """See :class:`VariablesCF`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcyFlux`."""
    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`MobilityCF`"""
    phase_mobility: Callable[[ppc.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`MobilityCF`"""
    fractional_phase_mobility: Callable[
        [ppc.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`MobilityCF`"""

    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`MobilityCF`."""
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`MobilityCF`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`BoundaryConditionsCF`."""

    bc_data_fractional_flow_energy_key: str
    """See :class:`BoundaryConditionsCF`."""
    uses_fractional_flow_bc: bool
    """See :class:`BoundaryConditionsCF`."""

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
        r"""Returns the internal energy of the fluid using the independent
        :meth:`~VariablesCF.enthalpy` variable and the
        :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid mixture

        .. math::

                \Phi\left(\rho h - p\right),

        in AD operator form on the ``subdomains``.

        """
        energy = self.porosity(subdomains) * (
            self.fluid_mixture.density(subdomains) * self.enthalpy(subdomains)
            - self.pressure(subdomains)
        )
        energy.set_name("fluid_mixture_internal_energy")
        return energy

    def advective_weight_enthalpy_flux(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the (advective) enthalpy flux.

        It is computed by summing :meth:`~MobilityCF.phase_mobility` weighed with
        :attr:`~porepy.compositional.base.Phase.specific_enthalpy` for each phase,
        and diviging by :meth:`~MobilityCF.total_mobility`.

        This is consistent with the fractional flow formulation, assuming the total
        mobility is part of the diffusive tensor in :class:`TotalMassBalanceEquation`.

        Creates a boundary operator, in case explicit values for fractional flow BC are
        used.

        """

        op: pp.ad.Operator | pp.ad.TimeDependentDenseArray

        if self.uses_fractional_flow_bc and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_energy_key,
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        else:
            # TODO is it worth reducing the operator tree size, by pulling the division
            # by total mobility out of the sum?
            op = pp.ad.sum_operator_list(
                [
                    phase.specific_enthalpy(domains)
                    * self.fractional_phase_mobility(phase, domains)
                    # * self.phase_mobility(phase, domains)
                    for phase in self.fluid_mixture.phases
                ],
                name="advected_enthalpy",
            )  # / self.total_mobility(domains)

        op.set_name("bc_advected_enthalpy")
        return op

    def enthalpy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Returns the advective enthalpy flux, using the Darcy flux and the non-linear
        weight given by :meth:`advective_weight_enthalpy_flux`.

        Can be called on boundaries to obtain a representation of user-given Neumann
        data on inlet faces.

        """
        # BC representation on the Neumann boundary
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # # NOTE The advected enthalpy (Neumann-type flux) must be consistent with
            # # the total mass flux
            op = self.advective_weight_enthalpy_flux(domains) * self.darcy_flux(domains)
            return op

        # Check that the domains are grids, not interfaces
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                "Domains must consist entirely of subdomains for the enthalpy flux."
            )
        domains = cast(list[pp.Grid], domains)

        # NOTE Boundary conditions are different from the pressure equation
        # This is consistent with the usage of darcy_flux in advective_flux
        boundary_operator_enthalpy = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=domains,
                dirichlet_operator=self.advective_weight_enthalpy_flux,
                neumann_operator=self.enthalpy_flux,
                robin_operator=None,
                bc_type=self.bc_type_advective_flux,
                name="bc_values_enthalpy_flux",
            )
        )

        discr = self.mobility_discretization(domains)
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
        discr = self.interface_mobility_discretization(interfaces)
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
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advective_weight_enthalpy_flux(subdomains)
        flux = self.well_advective_flux(
            interfaces,
            weight,
            discr,
        )

        eq = self.well_enthalpy_flux(interfaces) - flux
        eq.set_name("well_enthalpy_flux_equation")
        return eq


class ComponentMassBalanceEquations(pp.BalanceEquation):
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

    mdg: pp.MixedDimensionalGrid
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`SolidSkeletonCF`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcyFlux`."""
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
    """See :class:`~porepy.models.constitutive_laws.AdvectiveFlux`."""
    interface_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """See :class:`~porepy.models.constitutive_laws.AdvectiveFlux`."""
    well_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """See :class:`~porepy.models.constitutive_laws.AdvectiveFlux`."""

    fractional_component_mobility: Callable[
        [ppc.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`MobilityCF`."""

    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`MobilityCF`."""
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`MobilityCF`."""

    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """See :class:`~porepy.models.geometry.ModelGeometry`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`BoundaryConditionsCF`."""

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Optional[Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator]],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]
    """See :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`"""
    uses_fractional_flow_bc: bool
    """See :class:`BoundaryConditionsCF`."""

    bc_data_fractional_flow_component_key: Callable[[ppc.Component], str]
    """See :class:`BoundaryConditionsCF`"""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """See :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`."""

    has_independent_fraction: Callable[[ppc.Component | ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def _mass_balance_equation_name(self, component: ppc.Component) -> str:
        """Method returning a name to be given to the mass balance equation of a
        component."""
        return f"mass_balance_equation_{component.name}"

    def mass_balance_equation_names(self) -> list[str]:
        """Returns the names of mass balance equations set by this class,
        which are primary PDEs on all subdomains for each independent fluid component.
        """
        return [
            self._mass_balance_equation_name(component)
            for component in self.fluid_mixture.components
            if self.has_independent_fraction(component)
        ]

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all independent components on all subdomains.

        """
        subdomains = self.mdg.subdomains()

        for component in self.fluid_mixture.components:
            if self.has_independent_fraction(component):
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
        accumulation = self.volume_integral(
            self.fluid_mass_for_component(component, subdomains), subdomains, dim=1
        )
        flux = self.fluid_flux_for_component(component, subdomains)
        source = self.fluid_source_of_component(component, subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(self._mass_balance_equation_name(component))
        return eq

    def fluid_mass_for_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Returns the accumulation term in a ``component``'s mass balance equation
        using the :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid
        mixture and the component's :attr:`~porepy.compositional.base.Component.fraction`

        .. math::

            \Phi \rho \z_{\eta},

        in AD operator form on the given ``subdomains``.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid_mixture.density(subdomains)
            * component.fraction(subdomains)
        )
        mass_density.set_name(f"component_mass_{component.name}")
        return mass_density

    def advective_weight_component_flux(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the advective component flux.

        It uses the ``component``'s :meth:`~MobilityCF.fractional_component_mobility`,
        assuming the flux contains the total mobility in the diffusive tensor.

        This is consistent with the fractional flow formulation, based on overall
        fractions.

        Creates a boundary operator, in case explicit values for fractional flow BC are
        used.

        """

        op: pp.ad.Operator | pp.ad.TimeDependentDenseArray

        if self.uses_fractional_flow_bc and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_component_key(component),
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        else:
            op = self.fractional_component_mobility(component, domains)

        op.set_name(f"advected_mass_{component.name}")
        return op

    def fluid_flux_for_component(
        self, component: ppc.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A fractional component mass flux, where the total flux consists of the Darcy
        flux multiplied with a non-linear weight.

        See Also:
            :meth:`advective_weight_component_flux`

        Can be called on the boundary to obtain a representation of user-given Neumann
        data.

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
        domains = cast(list[pp.Grid], domains)

        discr = self.mobility_discretization(domains)
        weight = self.advective_weight_component_flux(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        weight_inlet_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            partial(self.advective_weight_component_flux, component),
        )
        fluid_flux_neumann_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            partial(self.fluid_flux_for_component, component),
        )
        interface_flux = cast(
            Callable[[list[pp.MortarGrid]], pp.ad.Operator],
            partial(self.interface_flux_for_component, component),
        )

        # NOTE Boundary conditions are different from the pressure equation
        # This is consistent with the usage of darcy_flux in advective_flux
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=weight_inlet_bc,
            neumann_operator=fluid_flux_neumann_bc,
            robin_operator=None,
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
        """Interface component flux using a the interface darcy flux and
        :meth:`advective_weight_component_flux`.

        See Also:
            :attr:`interface_advective_flux`

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advective_weight_component_flux(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name(f"interface_component_flux_{component.name}")
        return flux

    def well_flux_for_component(
        self, component: ppc.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well component flux using a the well flux and
        :meth:`advective_weight_component_flux`.

        See Also:
            :attr:`well_advective_flux`

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advective_weight_component_flux(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, weight, discr)
        flux.set_name(f"well_component_flux_{component.name}")
        return flux

    def fluid_source_of_component(
        self, component: ppc.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Analogous to
        :meth:`~TotalMassBalanceEquation.fluid_source`,
        but using :meth:`interface_flux_for_component` and
        :meth:`well_flux_for_component` to obtain the correct, fractional flow accross
        interfaces.

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


class TracerTransportEquations(ComponentMassBalanceEquations):
    """Simple transport equations for every tracer in a fluid component, which is
    a compound.

    The only difference to the compound's mass balance
    (given by :class:`ComponentMassBalanceEquations`) is, that the accumulation term
    is additionally weighed with the relative tracer fraction fraction.

    """

    has_independent_tracer_fraction: Callable[[ppc.ChemicalSpecies, ppc.Compound], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def _tracer_transport_equation_name(
        self,
        tracer: ppc.ChemicalSpecies,
        component: ppc.Compound,
    ) -> str:
        """Method returning a name to be given to the transport equation of an active
        tracer in a compound."""
        return f"tracer_transport_equation_{tracer.name}_{component.name}"

    def tracer_transport_equation_names(self) -> list[str]:
        """Returns the names of transport equations set by this class,
        which are primary PDEs on all subdomains for each tracer in each compound in the
        fluid mixture."""
        return [
            self._tracer_transport_equation_name(tracer, component)
            for component in self.fluid_mixture.components
            if isinstance(component, ppc.Compound)
            for tracer in component.active_tracers
            if self.has_independent_tracer_fraction(tracer, component)
        ]

    def set_equations(self):
        """Transport equations are set for all active tracers in each compound in the
        fluid mixture."""
        subdomains = self.mdg.subdomains()

        for component in self.fluid_mixture.components:
            if isinstance(component, ppc.Compound):

                for tracer in component.active_tracers:
                    sd_eq = self.transport_equation_for_tracer(
                        tracer, component, subdomains
                    )
                    self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def transport_equation_for_tracer(
        self,
        tracer: ppc.ChemicalSpecies,
        compound: ppc.Compound,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Mass balance equation for subdomains for a given compound.

        Parameters:
            tracer: An active tracer in the ``compound``.
            compound: A transportable fluid compound in the mixture.
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.volume_integral(
            self.mass_for_tracer(tracer, compound, subdomains), subdomains, dim=1
        )
        flux = self.fluid_flux_for_component(compound, subdomains)
        source = self.fluid_source_of_component(compound, subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(self._tracer_transport_equation_name(tracer, compound))
        return eq

    def mass_for_tracer(
        self,
        tracer: ppc.ChemicalSpecies,
        compound: ppc.Compound,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        r"""The accumulated mass of a tracer, for a given tracer in a fluid compound in
        the overall fraction formulation, i.e. cell-wise volume integral of

        .. math::

            \Phi \left(\sum_j \rho_j s_j\right) z_j c_i,

        which is essentially the accumulation of the compound mass scaled with the
        tracer fraction.

        Parameters:
            tracer: An active tracer in the ``compound``.
            component: A compound in the fluid mixture with a mass balance equation.
            subdomains: List of subdomains.

        Returns:
            Operator representing above expression.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid_mixture.density(subdomains)
            * compound.fraction(subdomains)
            * compound.tracer_fraction_of[tracer](subdomains)
        )
        mass_density.set_name(f"solute_mass_{tracer.name}_{compound.name}")
        return mass_density


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
           :attr:`~porepy.compositional.base.Phase.saturation` of independent phases
           as well as :attr:`~porepy.compositional.base.Phase.partial_fraction_of` are
           dangling variables.

           The system can be closed by using :meth:`eliminate_by_constitutive_law`
           for example, to introduce local, secondary equations.

    """

    fluid_mixture: ppc.FluidMixture
    """Provided by: class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

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
            pp.ad.SurrogateFactory,
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
        independent_quantity: Callable[
            [pp.GridLikeSequence], pp.ad.MixedDimensionalVariable
        ],
        dependencies: Sequence[
            Callable[[pp.GridLikeSequence], pp.ad.MixedDimensionalVariable]
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

        It uses :class:`~porepy.compositional.utils.SecondaryExpression` to
        provide AD representations of :math:`\\hat{\\varphi}` and to update its values
        and derivatives using ``func`` in the solutionstrategy.

        Note:
            Keep the limitations of
            :class:`~porepy.compositional.utils.SecondaryExpression` in mind,
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
        non_boundaries = cast(
            pp.GridLikeSequence,
            [g for g in domains if isinstance(g, (pp.Grid, pp.MortarGrid))],
        )

        sec_var = independent_quantity(non_boundaries)
        g_ids = [d.id for d in non_boundaries]

        sec_expr = pp.ad.SurrogateFactory(
            name=f"secondary_expression_for_{sec_var.name}_on_grids_{g_ids}",
            mdg=self.mdg,
            dependencies=dependencies,
        )

        local_equ = sec_var - sec_expr(non_boundaries)
        local_equ.set_name(f"elimination_of_{sec_var.name}_on_grids_{g_ids}")
        self.equation_system.set_equation(
            local_equ, cast(list[pp.Grid] | list[pp.MortarGrid], non_boundaries), dofs
        )

        self.add_constitutive_expression(sec_var, sec_expr, func, domains)


class PrimaryEquationsCF(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation_h,
    TracerTransportEquations,
    ComponentMassBalanceEquations,
):
    """A collection of primary equatons in the CF setting.

    They are PDEs consisting of

    - 1 pressure equation
    - 1 energy balance
    - mass balance equations per component
    - transport equation for each tracer in every compound.

    """

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
            + self.tracer_transport_equation_names()
        )

    @property
    def secondary_equation_names(self) -> list[str]:
        """Returns a list of secondary equations, which is defined as the complement
        of :meth:`primary_equation_names` and all equations found in the equation
        system.

        Note:
            Due to usage of Python's ``set``- operations, the resulting list may or may
            not be in the order the equations were added to the model.

        """
        all_equations = set(
            [name for name, _ in self.equation_system.equations.items()]
        )
        return list(all_equations.difference(set(self.primary_equation_names)))

    def set_equations(self):
        """Sets the PDE's of this model in the following order:

        1. Pressure equation :class:`TotalMassBalanceEquation`
        2. Mass balance equation per independent component
           :class:`ComponentMassBalanceEquations`
        3. Tracer transport equations per active tracer in compounds
           :class:`TracerTransportEquations`
        4. Total energy balance equation using independent temperature and fluid
           enthalpy variables :class:`TotalEnergyBalanceEquation_h`

        This order was chosen to cluster the mass-related rows into a row block.
        Effects on the sparsity pattern f.e. were not considered. TODO

        """
        TotalMassBalanceEquation.set_equations(self)
        ComponentMassBalanceEquations.set_equations(self)
        TracerTransportEquations.set_equations(self)
        TotalEnergyBalanceEquation_h.set_equations(self)


class VariablesCF(
    mass_energy.VariablesFluidMassAndEnergy,
    ppc.CompositionalVariables,
):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the specific fluid enthalpy."""

    enthalpy_variable: str
    """See :class:`SolutionStrategyCF`."""

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
            + self.tracer_fraction_variables
            + [
                self.enthalpy_variable,
            ]
        )

    @property
    def secondary_variables_names(self) -> list[str]:
        """Returns a list of secondary variables, which is defined as the complement
        of :meth:`primary_variable_names` and all variables found in the equation
        system.

        Note:
            Due to usage of Python's ``set``- operations, the resulting list may or may
            not be in the order the variables were created in the final model.

        """
        all_variables = set([var.name for var in self.equation_system.get_variables()])
        return list(all_variables.difference(set(self.primary_variable_names)))

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
        ppc.CompositionalVariables.create_variables(self)

    def enthalpy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Representation of the fluid enthalpy as an AD-Operator, more precisely as an
        independent variable on subdomains.

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
                """Argument domains a mixture of subdomain and boundary grids."""
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

    time_step_indices: np.ndarray
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`"""
    iterate_indices: np.ndarray
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`"""

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            pp.ad.SurrogateFactory,
            Callable[..., tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid] | Sequence[pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """See :class:`SolutionStrategyCF`"""

    def add_constitutive_expression(
        self,
        primary: pp.ad.MixedDimensionalVariable,
        expression: pp.ad.SurrogateFactory,
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
        domains = cast(
            list[pp.Grid] | list[pp.MortarGrid],
            [g for g in grids if isinstance(g, (pp.Grid, pp.MortarGrid))],
        )
        self._constitutive_eliminations.update(
            {expression.name: (primary, expression, func, domains, boundaries)}
        )

    def update_all_constitutive_expressions(self) -> None:
        """Method to update the values of all constitutive expressions in the iterative
        sense.

        Loops over all expressions stored using :meth:`add_constitutive_expression`,
        evaluates their dependencies on respective domains and calls the stored
        evaluation function to obtain derivatives and values.

        To be used before solving the system in a non-linear iteration.

        """
        ni = self.iterate_indices.size
        for _, expr, func, domains, _ in self._constitutive_eliminations.values():
            for g in domains:
                X = [
                    x(cast(list[pp.Grid] | list[pp.MortarGrid], [g])).value(
                        self.equation_system
                    )
                    for x in expr._dependencies
                ]

                vals, diffs = func(*X)

                expr.progress_iterate_values_on_grid(vals, g, depth=ni)
                expr.set_derivatives_on_grid(diffs, g)

    def progress_all_constitutive_expressions_in_time(self) -> None:
        """Method to progress the values of all added constitutive expressions in time.

        It takes the values at the most recent iterates, and stores them as the most
        recent previous time step.

        To be used after non-linear iterations converge in a time-dependent problem.

        """
        nt = self.time_step_indices.size
        for _, expr, _, domains, _ in self._constitutive_eliminations.values():
            expr.progress_values_in_time(domains, depth=nt)


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

    Essential BC denote the values of primary variables on the Dirichlet boundary,
    and flux values (Darcy + Fourier) on the Neumann boundary.

    They must be provided for all set-ups.

    **BC values for constitutively eliminated variables**:

    If a variable was eliminated using
    :meth:`SecondaryEquationsMixin.eliminate_by_constitutive_law` on a boundary grid,
    the passed function is used to evaluate and store its value on the boundary.
    This assumes the BC values for its dependencies are also provided.

    **BC Values for thermodynamic properties of phases**:

    If the user choce to pass ``'use_fractional_flow_bc' == False`` as a model
    parameter, the boundary values of properties are evaluated and stored.
    This happens for properties which are appearing in the non-linear weights of
    advective fluxes in component mass and energy balance.

    **BC Values in the fractional flow setting**:

    If the user passed ``'use_fractional_flow_bc' == True`` as a model parameter,
    the non-linear weights in the advecitve fluxes are treated as *closed* expressions
    on the boundary. The user has to provide values for them explicitly and they are
    not computed using values of e.g. viscosity, density, enthalpies, saturations and
    partial fractions on the boundary.

    Important:

        This class resolves some inconsistencies in the parent methods regarding the BC
        type definition. There can be only one definition of Dirichlet and Neumann
        boundary for the fluxes based on Darcy's Law (advective fluxes), and is to be
        set in :meth:`bc_type_darcy_flux`. Other BC type definition for advective
        fluxes are overriden here and point to the BC type for the darcy flux.

        Due to how the Upwinding is implemented, it has its own BC type definition,
        **which must be flagged as Dirichlet everywhere**. Do not mess with
        :meth:`bc_type_advective_flux`. This inconsistency will be removed in the near
        future.

    """

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    enthalpy_variable: str
    """See :class:`SolutionStrategyCF`."""
    params: dict
    """See :class:`SolutionStrategyCF`."""

    has_independent_fraction: Callable[[ppc.Component | ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    _overall_fraction_variable: Callable[[ppc.Component], str]
    """See :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """See :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""
    _partial_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """See :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""
    _tracer_fraction_variable: Callable[[ppc.ChemicalSpecies, ppc.Compound], str]
    """See :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            pp.ad.SurrogateFactory,
            Callable[..., tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid] | Sequence[pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """See :class:`SolutionStrategyCF`"""

    bc_data_fractional_flow_energy_key: str = "bc_data_fractional_flow_energy"
    """Key to store the BC values for the non-linear weight in the advective flux in the
    energy balance equation, for the case where explicit values are provided."""

    @property
    def uses_fractional_flow_bc(self) -> bool:
        """Flag passed to the model set-up, indicating whether values for non-linear
        weights in advectie fluxes are used explicitly on the boundary, or calculated
        inderectly from properties and variables involved.

        Defaults to False.

        """
        return bool(self.params.get("use_fractional_flow_bc", False))

    def bc_data_fractional_flow_component_key(self, component: ppc.Component) -> str:
        """Key to store the BC values of the non-linear weight in the advective flux
        of a component's mass balance equation"""
        return f"bc_data_fractional_flow_{component.name}"

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
        # TODO This needs more documentation in tutorials or similar, such that the
        # user is aware of which data is computed and stored, and which not.
        if self.uses_fractional_flow_bc:
            self.update_fractional_boundary_values()
        else:
            self.update_boundary_values_phase_properties()

    def update_essential_boundary_values(self) -> None:
        """Method updating BC values of primary variables on Dirichlet boundary,
        and values of fractional weights in advective fluxes on Neumann boundary.

        This is separated, because the order matters: Must be first update.

        """
        for component in self.fluid_mixture.components:
            # Update of tracer fractions on Dirichlet boundary
            if isinstance(component, ppc.Compound):
                for tracer in component.active_tracers:
                    bc_vals = cast(
                        Callable[[pp.BoundaryGrid], np.ndarray],
                        partial(self.bc_values_tracer_fraction, tracer, component),
                    )
                    self.update_boundary_condition(
                        self._tracer_fraction_variable(tracer, component),
                        function=bc_vals,
                    )

            # Update of independent overall fractions on Dirichlet boundary
            if self.has_independent_fraction(component):
                bc_vals = cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_overall_fraction, component),
                )
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
            (:meth:`~porepy.compositional.base.Phase.compute_properties`) must be able
            to handle that input.

        """

        nt = self.time_step_indices.size
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
                    state = phase.compute_properties(*cast(list[np.ndarray], dep_vals))
                    rho_bc = state.rho
                    h_bc = state.h
                    mu_bc = state.mu

                # phase properties which appear in mobilities
                phase.density.update_boundary_values(rho_bc, bg, depth=nt)
                phase.specific_enthalpy.update_boundary_values(h_bc, bg, depth=nt)
                phase.viscosity.update_boundary_values(mu_bc, bg, depth=nt)

    def update_fractional_boundary_values(self) -> None:
        """If the user instructs the model to use explicit values for the non-linear
        weights in advective fluxes, they are updated here on the boundary."""

        # Updating BC values of non-linear weights in component mass balance equations
        # Dependent components are skipped.
        for component in self.fluid_mixture.components:
            # NOTE the independency of overall fractions is used to characterize the
            # dependency of fractional flow, since the fractional weights also fulfill
            # the unity constraint.
            # In practice, the fractional flow weight for the dependent component is
            # never used in any equation, since it's mass balance is not part of the
            # model equations
            if self.has_independent_fraction(component):
                bc_func = cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_fractional_flow_component, component),
                )

                self.update_boundary_condition(
                    name=self.bc_data_fractional_flow_component_key(component),
                    function=bc_func,
                )

        # Updaing BC values of the non-linear weight in the energy balance
        # (advected enthalpy)
        self.update_boundary_condition(
            name=self.bc_data_fractional_flow_energy_key,
            function=self.bc_values_fractional_flow_energy,
        )

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

    def bc_values_tracer_fraction(
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

    def bc_values_fractional_flow_component(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the non-linear weight in the advective flux in
        :class:`ComponentMassBalanceEquations`, determining how much mass for respecitve
        ``component`` is entering the system on some inlet faces.

        Parameters:
            component: A component in the fluid mixture.
            boundary_grid: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_fractional_flow_energy(
        self, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the non-linear weight in the advective flux in
        :class:`TotalEnergyBalanceEquation_h`, determining how much energy/enthalpy is
        entering the system on some inlet faces.

        Parameters:
            boundary_grid: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

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
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    time_step_indices: np.ndarray
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    iterate_indices: np.ndarray
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]
    """See :class:`VariablesCF`."""

    has_independent_fraction: Callable[[ppc.Component | ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_tracer_fraction: Callable[[ppc.ChemicalSpecies, ppc.Compound], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    _constitutive_eliminations: dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            pp.ad.SurrogateFactory,
            Callable[..., tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid] | Sequence[pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]
    """See :class:`SolutionStrategyCF`"""

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
        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(
                val,
                iterate_index=iterate_index,
            )
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
            self.equation_system.set_variable_values(
                self.initial_pressure(sd), [self.pressure([sd])], iterate_index=0
            )
            self.equation_system.set_variable_values(
                self.initial_temperature(sd), [self.temperature([sd])], iterate_index=0
            )
            self.equation_system.set_variable_values(
                self.initial_enthalpy(sd), [self.enthalpy([sd])], iterate_index=0
            )

            # Setting overall fractions and tracer fractions
            for component in self.fluid_mixture.components:
                # independent overall fractions must have an initial value
                if self.has_independent_fraction(component):
                    self.equation_system.set_variable_values(
                        self.initial_overall_fraction(component, sd),
                        [
                            cast(
                                pp.ad.MixedDimensionalVariable, component.fraction([sd])
                            )
                        ],
                        iterate_index=0,
                    )

                # All tracer fractions must have an initial value
                if isinstance(component, ppc.Compound):
                    for tracer in component.active_tracers:
                        if self.has_independent_tracer_fraction(tracer, component):
                            self.equation_system.set_variable_values(
                                self.initial_tracer_fraction(tracer, component, sd),
                                [
                                    cast(
                                        pp.ad.MixedDimensionalVariable,
                                        component.tracer_fraction_of[tracer]([sd]),
                                    )
                                ],
                                iterate_index=0,
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
        ni = self.iterate_indices.size
        nt = self.time_step_indices.size
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
                    d(cast(list[pp.Grid] | list[pp.MortarGrid], [grid])).value(
                        self.equation_system
                    )
                    for d in expr._dependencies
                ]
                val, diff = f(*dep_vals_g)
                # values for each iterate index
                for _ in self.iterate_indices:
                    expr.progress_iterate_values_on_grid(val, grid, depth=ni)
                # derivative values for the current iterate
                expr.set_derivatives_on_grid(diff, grid)

            # progress values in time for all indices
            for _ in self.time_step_indices:
                expr.progress_values_in_time(domains, depth=nt)

    def set_intial_values_phase_properties(self) -> None:
        """Method to set the initial values and derivative values of phase
        properties, which are secondary expressions with some dependencies.

        This method also fills all time and iterate indices with the initial values.
        Derivative values are only stored for the current iterate.

        """
        subdomains = self.mdg.subdomains()
        ni = self.iterate_indices.size
        nt = self.time_step_indices.size

        # Set the initial values on individual grids for the iterate indices
        for grid in subdomains:
            for phase in self.fluid_mixture.phases:
                dep_vals = [
                    d([grid]).value(self.equation_system)
                    for d in self.dependencies_of_phase_properties(phase)
                ]

                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals)
                )

                # Set values and derivative values for current current index
                update_phase_properties(grid, phase, phase_props, ni)

                # progress iterate values to all iterate indices
                for _ in self.iterate_indices:
                    phase.density.progress_iterate_values_on_grid(
                        phase_props.rho, grid, depth=ni
                    )
                    phase.specific_enthalpy.progress_iterate_values_on_grid(
                        phase_props.h, grid, depth=ni
                    )
                    phase.viscosity.progress_iterate_values_on_grid(
                        phase_props.mu, grid, depth=ni
                    )
                    phase.conductivity.progress_iterate_values_on_grid(
                        phase_props.kappa, grid, depth=ni
                    )
                # Copy values to all time step indices
                for _ in self.time_step_indices:
                    phase.density.progress_values_in_time([grid], depth=nt)
                    phase.specific_enthalpy.progress_values_in_time([grid], depth=nt)

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
            component: A component in the fluid mixture with an independent overall
                fraction.
            sd: A subdomain in the md-grid.

        Returns:
            The initial overall fraction values for a component on a subdomain.
            Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

    def initial_tracer_fraction(
        self, tracer: ppc.ChemicalSpecies, compound: ppc.Compound, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            tracer: An active tracer in the ``compound``.
            component: A compound in the fluid mixture.
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
    - ``'use_fractional_flow_bc'``: Defaults to False. If True, the model treats the
      non-linear weights in the advective fluxes in mass and energy balances as a closed
      term on the boundary. The user must then provide values for the non-linear weights
      directly. Otherwise, their values are calculated based on the values of inidivual
      terms they are composed of (e.g. density depending on pressure and temperature on
      the boundary).
    - ``'equilibrium_type'``: Defaults to None. If the model contains an equilibrium
      part, it should be a string indicating the fixed state of the local phase
      equilibrium problem e.g., ``'p-T'``,``'p-h'``. The string can also contain other
      qualifiers providing information about the equilibrium model, for example
      ``'unified-p-h'``.

    """

    fluid_mixture: ppc.FluidMixture
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    total_mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`MobilityCF`."""
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """

    fourier_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[Sequence[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    create_mixture: Callable[[], None]
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""
    assign_thermodynamic_properties_to_mixture: Callable[[], None]
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""
    set_initial_values: Callable[[], None]
    """See :class:`InitialConditionsCF`."""
    progress_all_constitutive_expressions_in_time: Callable[[], None]
    """See :class:`ConstitutiveLawsCF`."""
    update_all_constitutive_expressions: Callable[[], None]
    """See :class:`ConstitutiveLawsCF`."""
    dependencies_of_phase_properties: Callable[
        [ppc.Phase], Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]
    ]
    """See :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`."""

    primary_equation_names: list[str]
    """See :class:`EquationsCompositionalFlow`."""
    primary_variable_names: list[str]
    """See :class:`VariablesCF`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`BoundaryConditionsCF`."""

    _is_ref_phase_eliminated: bool
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`"""
    _is_ref_comp_eliminated: bool
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`"""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations: list[pp.ad._ad_utils.MergedOperator] = []

        self._constitutive_eliminations: dict[
            str,
            tuple[
                pp.ad.MixedDimensionalVariable,
                pp.ad.SurrogateFactory,
                Callable[..., tuple[np.ndarray, np.ndarray]],
                Sequence[pp.Grid] | Sequence[pp.MortarGrid],
                Sequence[pp.BoundaryGrid],
            ],
        ] = dict()
        """Storage of terms which were eliminated by some cosntitutive expression."""

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid mixture."""

        # Input validation for set-up
        if not self._is_ref_comp_eliminated:
            warnings.warn(
                "Reference component (and its fraction) are not eliminated."
                + " The basic model needs to be closed (unity constraint)."
            )
        if not self._is_ref_phase_eliminated:
            warnings.warn(
                "Reference phase (and its saturation) are not eliminated."
                + " The basic model needs to be closed (unity constraint)."
            )

    @property
    def equilibrium_type(self) -> Optional[str]:
        """The equilibrium type of the model can be set by passing the
        ``'equilibrium_type'`` argument to the model ``params`` at instantiation.

        By default, ``None`` is the type for flow models without equilibrium
        calculations.

        For defining an equilibrium in a model, use the two quantities which are fixed
        at equilibrium, e.g. ``'p-T'``, ``'p-h'``, and some qualifiers for the
        equilibrium model, e.g. ``'unified-p-h'``.

        """
        return self.params.get("equilibrium_type", None)

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

        # Discretization for non-linear weights in advective parts (Upwinding)
        self.add_nonlinear_discretization(
            self.mobility_discretization(subdomains).upwind(),
        )
        self.add_nonlinear_discretization(
            self.interface_mobility_discretization(interfaces).flux(),
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
        self.update_all_constitutive_expressions()
        self.update_thermodynamic_properties_of_phases()

    def update_thermodynamic_properties_of_phases(self) -> None:
        """This method uses for each phase the underlying EoS to calculate
        new values and derivative values of phase properties and to update them
        them in the iterative sense, on all subdomains."""

        subdomains = self.mdg.subdomains()
        ni = self.iterate_indices.size

        for grid in subdomains:
            for phase in self.fluid_mixture.phases:
                dep_vals = [
                    d([grid]).value(self.equation_system)
                    for d in self.dependencies_of_phase_properties(phase)
                ]

                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals)
                )

                # Set current iterate indices of values and derivatives
                update_phase_properties(grid, phase, phase_props, ni)

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
        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            # Computing the darcy flux in fractures (given by variable)
            vals = self.interface_darcy_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            # Computing the darcy flux in wells (given by variable)
            vals = self.well_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

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
                        sd, self.permeability([sd]), self.solid.permeability()
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
        self.progress_all_constitutive_expressions_in_time()
        subdomains = self.mdg.subdomains()
        nt = self.time_step_indices.size
        for phase in self.fluid_mixture.phases:
            phase.density.progress_values_in_time(subdomains, depth=nt)
            phase.specific_enthalpy.progress_values_in_time(subdomains, depth=nt)

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

        if reduce_linear_system_q:
            # TODO block diagonal inverter for secondary equations
            self.linear_system = self.equation_system.assemble_schur_complement_system(
                self.primary_equation_names, self.primary_variable_names
            )
        else:
            self.linear_system = self.equation_system.assemble()
        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        sol = super().solve_linear_system()
        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            sol = self.equation_system.expand_schur_complement_solution(sol)
        return sol

    # def _log_res(self, loc: str = ""):
    #     all_res = list()
    #     msg = "Residuals per equation:\n"
    #     eq_names = list(self.equation_system.equations.keys())
    #     for i, name in enumerate(eq_names):
    #         eq = self.equation_system.equations[name]
    #         res = eq.value(self.equation_system)
    #         all_res.append(res)
    #         if i == (len(eq_names) - 1):  # last equation without line break
    #             msg += f"{name}: {np.linalg.norm(res)}"
    #         else:
    #             msg += f"{name}: {np.linalg.norm(res)}\n"
    #     all_res = np.hstack(all_res)
    #     logger.info(
    #         f"\nResidual {loc}: {np.linalg.norm(all_res) / np.sqrt(all_res.size)}"
    #     )
    #     logger.debug(msg)


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
    - tracer fractions for pure transport without equilibrium (if any)

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
    - tracer transport equations

    The secondary block of equations must be provided using constitutive relations
    or an equilibrium model for the fluid.

    Note:
        The model inherits the md-treatment of Darcy flux, advective enthalpy flux and
        Fourier flux. Some interface variables and interface equations are introduced
        there. They are treated as secondary equations and variables in the basic model.

    """
