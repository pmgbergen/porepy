"""Model mixins for compositional flow (CF) and fractional flow (CFF).

The CF settings is general in terms of number of phases, number of components and
thermal conditions. It therefore does not contain a runnable model, but building blocks
for models concretized in terms of phases, components and thermal setting.

The following equations are available:

- :class:`ComponentMassBalanceEquations`: While the :class:`~porepy.models.
  fluid_mass_balance.MassBalanceEquations` represent the balance of total mass
  (pressure equation), this equation represent the balance equation of individual
  components in the fluid mixture. Interface fluxes are handled using the overall
  Darcy interface flux implemented in the pressure equations and an adapted non-linear
  term. It introduces a single equation, all interface equations are introduced by
  :class:`~porepy.models.fluid_mass_balance.MassBalanceEquations`.
- :class:`DiffusiveMassBalanceEquations`: A diffusive variant of the total mass balance,
  where the total mobility is an isotropic contribution to the permeability tensor.
  To be used in a fractional flow model. Expensive, since it requires a
  re-discretization of the MPFA, but numerically consistent.
- :class:`TwoVariableEnergyBalanceEquations`: A specialized total energy balance using
  an independent (specific fluid) enthalpy variable in the accumulation term.
  Otherwise completely analogous to its base
  :class:`~porepy.models.energy_balance.EnergyBalanceEquations`.
- :class:`TracerTransportEquations`: A special set of transport equations for
  :class:`~porepy.compositional.base.Compound` and tracer contained therein.
  Analogous to :class:`ComponentMassBalanceEquations`, but with a modified accumulation
  term to account for the relative tracer fractions.

Primary equations include a total mass balance, component balance and the some energy
balance equation. A collection for non-isothermal, non-diffusive flow is given in
:class:`PrimaryEquationsCF`.

Primary variables are assumed to be a single pressure (no capillarity), an enthalpy or
temperature variable, and overall fraction variables. They are collected in
:class:`VariablesCF`.

In a general setting, where phase properties are given by :class:`~porepy.numerics.ad.
surrogate_operator.SurrogateFactory`, special IC and BC classes handling the consistent
update are given by :class:`InitialConditionsPhaseProperties` and
:class:`BoundaryConditionsPhaseProperties` respectively. They handle the automatized
computation of respective values during the simulation.

A collective

Following IC mixins are available:

- :class:`InitialConditionsFractions`: Provides an interface to set initial conditions
  for overall fractions and active tracer fractions, if any. Other fractions are for now
  not covered since they are considered secondary and their initialization can be done
  based on the primary fractions.
- :class:`InitialConditionsPhaseProperties`: An initialization routine for models where
  phase properties are represented by :class:`~porepy.numerics.ad.surrogate_operator.
  SurrogateFactory`. Their initialization is dependent on variable values, and is
  automatized to happen after the variables are initialized.
- :class:`InitialConditionsCF`: A collection of above initialization routines, and
  the IC mixins for mass & energy, including an independent enthalpy variable.

Following BC mixins are available:

- :class:`BoundaryConditionsFractions`: The analogy to
  :class:`InitialConditionsFractions` but for BC.
- :class:'BoundaryConditionsPhaseProperties': The analogy to
  :class:`InitialConditionsPhaseProperties` but for BC. Their update can be automatized
  to happen after values for variables are set on the boundary.
- :class:`BoundaryConditionsFF`: An alternative to
  :class:`BoundaryConditionsPhaseProperties` for the fractional flow setting, where
  various non-linear terms in fluxes can be given explicitly on the boundary,
  instead of providing variable values which in return compute phase properties
  appearing in those expressions.
- :class:`BoundaryConditionsCF`: A collection of BC update routines for primary
  variables and phase properties as surrogate factories, including those from
  mass & energy, enthalpy and fractions.
- :class:`BoundaryConditionsCFF`: The alternative to :class:`BoundaryConditionsCF` for
  the fractional flow setting with explicit values for non-linear weights in fluxes.

The :class:`SolutionStrategyCF` handles the general Cf model, with or without fractional
flow simulation, and provides means to re-discretize the MPFA in the diffusive setting,
and to eliminate local, secondary equations via Schur-complement. Those equations are
required in any case to close the general multi-phase setting.

The two setups, :class:`ModelSetupCF` and :class:`ModelSetupCFF` are in principle
complete set-ups for non-isothermal, compositional flow. The steps required by users to
close the setups are:

1. Define a fluid with all its phases and components.
2. Close the system with local equations for dangling, fractional variables. These can
   either be :class:`~porepy.models.abstract_equations.LocalElimination` using some
   map or third-party correlations, or a closure in form of a local equilibrium system.

"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
import porepy.compositional as ppc

from . import energy_balance as energy
from . import fluid_mass_balance as mass
from . import mass_and_energy_balance as mass_energy
from .boundary_condition import BoundaryConditionsPrimaryVariables
from .constitutive_laws import ThermalConductivityCF
from .protocol import CompositionalFlowModelProtocol, PorePyModel

logger = logging.getLogger(__name__)


def update_phase_properties(
    grid: pp.Grid,
    phase: pp.Phase,
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
    if isinstance(phase.density, pp.ad.SurrogateFactory):
        phase.density.progress_iterate_values_on_grid(props.rho, grid, depth=depth)
        if update_derivatives:
            phase.density.set_derivatives_on_grid(
                props.drho_ext if use_extended_derivatives else props.drho, grid
            )
    if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
        phase.specific_enthalpy.progress_iterate_values_on_grid(
            props.h, grid, depth=depth
        )
        if update_derivatives:
            phase.specific_enthalpy.set_derivatives_on_grid(
                props.dh_ext if use_extended_derivatives else props.dh, grid
            )
    if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
        phase.viscosity.progress_iterate_values_on_grid(props.mu, grid, depth=depth)
        if update_derivatives:
            phase.viscosity.set_derivatives_on_grid(
                props.dmu_ext if use_extended_derivatives else props.dmu, grid
            )
    if isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory):
        phase.thermal_conductivity.progress_iterate_values_on_grid(
            props.kappa, grid, depth=depth
        )
        if update_derivatives:
            phase.thermal_conductivity.set_derivatives_on_grid(
                props.dkappa_ext if use_extended_derivatives else props.dkappa, grid
            )


# region general PDEs used in the (fractional) CF


class DiffusiveMassBalanceEquations(pp.BalanceEquation, CompositionalFlowModelProtocol):
    """Mixed-dimensional balance of total mass in a fluid mixture.

    Also referred to as *pressure equation*.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    Note:
        This balance equation assumes that the total mobility is part of the
        diffusive, second-order tensor in the non-linear (MPFA) discretization of the
        Darcy flux.

    See also:

        For an equation using upwinding for the total mass, see
        :class:`~porepy.models.fluid_mass_balance.MassBalanceEquations`.

    """

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`ConstitutiveLawsSolidSkeletonCF`."""

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

    def set_equations(self) -> None:
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        super().set_equations()

        assert pp.is_fractional_flow(
            self
        ), "fractional flow flag must be True in model params"
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
        # NOTE use same name to raise an error if one attempts to set both, diffusive
        # and regular mass balance equations
        eq.set_name(mass.MassBalanceEquations.primary_equation_name())
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        r"""Returns the accumulation term in the pressure equation
        :math:`\Phi\rho`, using the
        :attr:`~ConstitutiveLawsSolidSkeletonCF.permeability` and the
        :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid mixture, in
        AD operator form on a given set of ``subdomains``."""
        mass_density = self.fluid.density(subdomains) * self.porosity(subdomains)
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


class TwoVariableEnergyBalanceEquations(energy.EnergyBalanceEquations):
    """Mixed-dimensional balance of total energy in a fluid mixture, formulated with an
    independent (specific fluid) enthalpy variable in the accumulation term *and* a
    temperature variable in the Fourier flux.

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

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`EnthalpyVariable`."""

    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    fractional_phase_mobility: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""

    bc_data_fractional_flow_energy_key: str
    """See :class:`BoundaryConditionsCF`."""

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        r"""Returns the internal energy of the fluid using the independent
        :meth:`~porepy.models.energy_balance.EnthalpyVariable.enthalpy` variable and the
        :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid mixture

        .. math::

                \Phi\left(\rho h - p\right),

        in AD operator form on the ``subdomains``.

        """
        energy = self.porosity(subdomains) * (
            self.fluid.density(subdomains) * self.enthalpy(subdomains)
            - self.pressure(subdomains)
        )
        energy.set_name("fluid_internal_energy")
        return energy

    def advection_weight_energy_balance(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the (advective) enthalpy flux.

        In the fractional flow setting, this returns a time-dependent dense array on the
        boundary. On the internal domain it returns :math:`\\sum_j h_j f_j`, with
        :math:`j` denoting a phase and :math:`f_j` the fractional phase mobility.

        In the regular setting it performs a super-call to the parent class
        implementation.

        """

        op: pp.ad.Operator | pp.ad.TimeDependentDenseArray

        if pp.is_fractional_flow(self) and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_energy_key,
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        elif pp.is_fractional_flow(self):
            # TODO is it worth reducing the operator tree size, by pulling the division
            # by total mobility out of the sum?
            op = pp.ad.sum_operator_list(
                [
                    phase.specific_enthalpy(domains)
                    * self.fractional_phase_mobility(phase, domains)
                    # * phase.density(domains)
                    # * self.phase_mobility(phase, domains)
                    for phase in self.fluid.phases
                ],
            )  # / self.total_mobility(domains)
        else:
            op = super().advection_weight_energy_balance(domains)

        op.set_name("advected_enthalpy")
        return op


class ComponentMassBalanceEquations(pp.BalanceEquation, CompositionalFlowModelProtocol):
    """Mixed-dimensional balance of mass in a fluid mixture for present components.

    The total mass balance is the sum of all component mass balances.

    Since feed fractions per independent component are unknowns, the model requires
    additional transport equations to close the system.

    This equation is defined on all subdomains. Due to a single pressure and interface
    flux variable, there is no need for additional equations as is the case in the
    pressure equation.

    """

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`ConstitutiveLawsSolidSkeletonCF`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcyFlux`."""
    advective_flux: Callable[
        [
            list[pp.Grid],
            pp.ad.Operator,
            pp.ad.UpwindAd,
            pp.ad.Operator,
            Optional[Callable[[list[pp.MortarGrid]], pp.ad.Operator]],
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
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    component_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""

    bc_type_advective_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`BoundaryConditionsCF`."""

    bc_data_fractional_flow_component_key: Callable[[pp.Component], str]
    """See :class:`BoundaryConditionsCF`"""

    def _mass_balance_equation_name(self, component: pp.Component) -> str:
        """Method returning a name to be given to the mass balance equation of a
        component."""
        return f"component_mass_balance_equation_{component.name}"

    def component_mass_balance_equation_names(self) -> list[str]:
        """Returns the names of mass balance equations set by this class,
        which are primary PDEs on all subdomains for each independent fluid component.
        """
        return [
            self._mass_balance_equation_name(component)
            for component in self.fluid.components
            if self.has_independent_fraction(component)
        ]

    def set_equations(self) -> None:
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all independent components on all subdomains.

        """
        super().set_equations()

        subdomains = self.mdg.subdomains()

        for component in self.fluid.components:
            if self.has_independent_fraction(component):
                sd_eq = self.mass_balance_equation_for_component(component, subdomains)
                self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def mass_balance_equation_for_component(
        self, component: pp.Component, subdomains: list[pp.Grid]
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
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Returns the accumulation term in a ``component``'s mass balance equation
        using the :attr:`~porepy.compositional.base.FluidMixture.density` of the fluid
        mixture and the component's
        :attr:`~porepy.compositional.base.Component.fraction`

        .. math::

            \Phi \rho \z_{\eta},

        in AD operator form on the given ``subdomains``.

        """
        mass_density = (
            self.porosity(subdomains)
            * self.fluid.density(subdomains)
            * component.fraction(subdomains)
        )
        mass_density.set_name(f"component_mass_{component.name}")
        return mass_density

    def advection_weight_component_mass_balance(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The non-linear weight in the advective component flux.

        It uses the ``component``'s :meth:`~porepy.models.fluid_property_library.
        FluidMobility.fractional_component_mobility`, assuming the flux contains the
        total mobility in the diffusive tensor.

        This is consistent with the fractional flow formulation, based on overall
        fractions.

        Creates a boundary operator, in case explicit values for fractional flow BC are
        used.

        """

        op: pp.ad.Operator | pp.ad.TimeDependentDenseArray

        if pp.is_fractional_flow(self) and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_component_key(component),
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        elif pp.is_fractional_flow(self):
            op = self.fractional_component_mobility(component, domains)
        else:
            op = self.component_mobility(component, domains)

        op.set_name(f"advected_mass_{component.name}")
        return op

    def fluid_flux_for_component(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """A fractional component mass flux, where the total flux consists of the Darcy
        flux multiplied with a non-linear weight.

        See Also:
            :meth:`advection_weight_component_mass_balance`

        Can be called on the boundary to obtain a representation of user-given Neumann
        data.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # NOTE consistent Neumann-type flux based on the total flux
            op = self.advection_weight_component_mass_balance(
                component, domains
            ) * self.darcy_flux(domains)
            return op

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        discr = self.mobility_discretization(domains)
        weight = self.advection_weight_component_mass_balance(component, domains)

        # Use a partially evaluated function call to functions to mimic
        # functions solely depend on a sequence of grids
        weight_inlet_bc = cast(
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            partial(self.advection_weight_component_mass_balance, component),
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
        self, component: pp.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface component flux using a the interface darcy flux and
        :meth:`advection_weight_component_mass_balance`.

        See Also:
            :attr:`interface_advective_flux`

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advection_weight_component_mass_balance(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name(f"interface_component_flux_{component.name}")
        return flux

    def well_flux_for_component(
        self, component: pp.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well component flux using a the well flux and
        :meth:`advection_weight_component_mass_balance`.

        See Also:
            :attr:`well_advective_flux`

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advection_weight_component_mass_balance(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, weight, discr)
        flux.set_name(f"well_component_flux_{component.name}")
        return flux

    def fluid_source_of_component(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Analogous to
        :meth:`~porepy.models.fluid_mass_balance.MassBalanceEquations.fluid_source`,
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

    def _tracer_transport_equation_name(
        self,
        tracer: pp.Component,
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
            for component in self.fluid.components
            if isinstance(component, ppc.Compound)
            for tracer in component.active_tracers
            if self.has_independent_tracer_fraction(tracer, component)
        ]

    def set_equations(self) -> None:
        """Transport equations are set for all active tracers in each compound in the
        fluid mixture."""
        super().set_equations()

        subdomains = self.mdg.subdomains()

        for component in self.fluid.components:
            if isinstance(component, ppc.Compound):

                for tracer in component.active_tracers:
                    if self.has_independent_tracer_fraction(tracer, component):
                        sd_eq = self.transport_equation_for_tracer(
                            tracer, component, subdomains
                        )
                        self.equation_system.set_equation(
                            sd_eq, subdomains, {"cells": 1}
                        )

    def transport_equation_for_tracer(
        self,
        tracer: pp.Component,
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
        tracer: pp.Component,
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
            * self.fluid.density(subdomains)
            * compound.fraction(subdomains)
            * compound.tracer_fraction_of[tracer](subdomains)
        )
        mass_density.set_name(f"solute_mass_{tracer.name}_{compound.name}")
        return mass_density


# endregion
# region INTERMEDIATE CF MODEL MIXINS: collecting variables, equations, const. laws


class PrimaryEquationsCF(
    mass.MassBalanceEquations,
    TracerTransportEquations,
    ComponentMassBalanceEquations,
    TwoVariableEnergyBalanceEquations,
):
    """A collection of primary equations in the CF setting.

    They are PDEs consisting of

    - 1 pressure equation
    - 1 energy balance
    - mass balance equations per component
    - transport equation for each tracer in every compound.

    """


class VariablesCF(
    mass_energy.VariablesFluidMassAndEnergy,
    energy.EnthalpyVariable,
    ppc.CompositionalVariables,
):
    """Bundles standard variables for non-isothermal flow (pressure and temperature)
    with fractional variables and an independent enthalpy variable."""


class ConstitutiveLawsSolidSkeletonCF(
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.EnthalpyFromTemperature,
):
    """Collection of constitutive laws for the solid skeleton in the compositional
    flow framework.

    It additionally provides constitutive laws defining the relative and normal
    permeability functions in the compositional framework, based on saturations and
    mobilities.

    It also provides an operator representing the pore volume, used to define
    the :meth:`volume` which is used for isochoric flash calculations.

    TODO Is this the right place to implement :meth:`volume`?
    TODO Omar noted that relative permeabilities depend in general on all saturation
    variables, not only on the saturation for which phase it is meant. This requires
    re-evaluationg the signature.

    """

    total_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""

    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference porosity.

        Parameters:
            subdomains: A list of subdomains.

        Returns:
            The constant solid porosity wrapped as an Ad scalar.

        """
        return pp.ad.Scalar(self.solid.porosity, "reference_porosity")

    def diffusive_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Required by constitutive laws implementing differentiable MPFA/TPFA.

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
            self.solid.permeability,
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

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        Parameters:
            phase: A phase in the fluid.
            domains: A list of subdomains or boundaries.

        Returns:
            The base class method implements the linear law.

        """
        return phase.saturation(domains)

    def solid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the solid.

        Note:
            This must override the definition of solid internal energy which is
            (for some reasons) defined in the basic energy balance equation. TODO

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


class ConstitutiveLawsCF(
    # NOTE must be on top to overwrite phase properties as general surrogate factories
    ppc.FluidMixin,
    # must be on top to overwrite mobility and thermal conductivity from base class
    # constitutive laws
    ThermalConductivityCF,
    ConstitutiveLawsSolidSkeletonCF,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
):
    """Constitutive laws for compositional flow with mobility and thermal conductivity
    laws adapted to general fluid mixtures.

    It also uses a separate class, which collects constitutive laws for the solid
    skeleton, and puts the FluidMixin on top to overwrite the base class treatment of
    thermodynamic phase properties with general surrogate factories provided by the
    fluid mixin.

    All other constitutive laws are analogous to the underlying mass and energy
    transport.

    """


# endregion
# region BC, IC, Solution strategy


class _BoundaryConditionsAdvection(PorePyModel):
    """Temporary class fixing some inconsistencies for hyperbolic-type BC.

    FIXME throughout the package.

    """

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


class BoundaryConditionsFractions(
    BoundaryConditionsPrimaryVariables, CompositionalFlowModelProtocol
):
    """Mixin providing boundary values for overall fractions of components and tracer
    fractions in compounds (primary fractions).

    Note:
        As for the enthalpy, these variables are only in specific cases accessed on the
        boundary. But they are required in case phase properties depend on them, and
        subsequently a consistent evaluation of non-linear terms in advective fluxes is
        required.

        A case where overall fractions appear explicitely on the boundary, is the
        single-phase, multi-component model. In this case the partial fractions of the
        single phase are equal to the overall fractions.

    """

    def update_boundary_values_primary_variables(self) -> None:
        """Calls the user-provided data for overall fractions and tracer fractions after
        a super-call.

        See also:

            - :meth:`bc_values_overall_fraction`
            - :meth:`bc_values_tracer_fraction`

        """
        super().update_boundary_values_primary_variables()

        for component in self.fluid.components:
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

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            component: A component in the fluid mixture.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(bg.num_cells)

    def bc_values_tracer_fraction(
        self,
        tracer: pp.Component,
        compound: ppc.Compound,
        bg: pp.BoundaryGrid,
    ) -> np.ndarray:
        """BC values for active tracer fractions (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            tracer: A tracer in the ``compound``.
            compound: A component in the fluid mixture.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of
            the overall fraction.

        """
        return np.zeros(bg.num_cells)


class BoundaryConditionsPhaseProperties(
    pp.BoundaryConditionMixin, CompositionalFlowModelProtocol
):
    """Intermediate mixin layer to provide an interface for calculating values of phase
    properties on the boundary, which are represented by surrogate factories.

    Important:
        The computation of phase properties is performed on all boundary cells,
        Neumann and Dirichlet. This may require non-trivial values for primary
        variables like pressure on the Neumann boundary as well.

        This is due to phase properties being part of the non-linear advection term,
        i.e. on the Neumann boundary they are multiplied with the flux.

        The users themselves must implement zero values on non-inlet/outlet
        boundaries if specified.

        For a more direct approach, see :class:`BoundaryConditionsFF`,
        where the value of the non-linear terms in the advection must be given
        directly.

    """

    def update_all_boundary_conditions(self) -> None:
        """Calls :meth:`update_boundary_values_phase_properties` after the super-call."""
        super().update_all_boundary_conditions()
        self.update_boundary_values_phase_properties()

    def update_boundary_values_phase_properties(self) -> None:
        """Evaluates the phase properties using underlying EoS and progresses
        their values in time on the boundary.

        This base method updates only properties which are expected in the non-linear
        weights of the advective and diffusive flux:

        - phase densities
        - phase enthalpies
        - phase viscosities
        - phase thermal conductivities

        Phase volumes are not updated, as their assumed to be the reciprocals of
        density.

        """

        nt = self.time_step_indices.size
        for bg in self.mdg.boundaries():
            for phase in self.fluid.phases:
                # some work is required for BGs with zero cells
                if bg.num_cells == 0:
                    rho_bc = np.zeros(0)
                    h_bc = np.zeros(0)
                    mu_bc = np.zeros(0)
                    kappa_bc = np.zeros(0)
                else:
                    dep_vals = [
                        d([bg]).value(self.equation_system)
                        for d in self.dependencies_of_phase_properties(phase)
                    ]
                    state = phase.compute_properties(*cast(list[np.ndarray], dep_vals))
                    rho_bc = state.rho
                    h_bc = state.h
                    mu_bc = state.mu
                    kappa_bc = state.kappa

                if isinstance(phase.density, pp.ad.SurrogateFactory):
                    phase.density.update_boundary_values(rho_bc, bg, depth=nt)
                if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                    phase.specific_enthalpy.update_boundary_values(h_bc, bg, depth=nt)
                if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
                    phase.viscosity.update_boundary_values(mu_bc, bg, depth=nt)
                if isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory):
                    phase.thermal_conductivity.update_boundary_values(
                        kappa_bc, bg, depth=nt
                    )


class BoundaryConditionsFF(pp.BoundaryConditionMixin, CompositionalFlowModelProtocol):
    """Analogous to :class:`BoundaryConditionsPhaseProperties`, but providing means to
    define values of the non-linear terms in fluxes directly without using their full
    expression.

    This mixin requires a fractional flow model.

    """

    bc_data_fractional_flow_energy_key: str = "bc_data_fractional_flow_energy"
    """Key to store the BC values for the non-linear weight in the advective flux in the
    energy balance equation, for the case where explicit values are provided."""

    def bc_data_fractional_flow_component_key(self, component: pp.Component) -> str:
        """Key to store the BC values of the non-linear weight in the advective flux
        of a component's mass balance equation"""
        return f"bc_data_fractional_flow_{component.name}"

    def update_all_boundary_conditions(self) -> None:
        """Calls :meth:`update_boundary_values_fractional_flow` after the super-call.

        Raises:
            CompositionalModellingError: If this mixin is used in a non-fractional flow
                setting (see :class:`SolutionStrategyCF`).

        """
        super().update_all_boundary_conditions()
        if pp.is_fractional_flow(self):
            self.update_boundary_values_fractional_flow()
        else:
            raise pp.compositional.CompositionalModellingError(
                "Computing boundary values of fractional weights without flagging a"
                + " fractional flow setting in model parameters."
            )

    def update_boundary_values_fractional_flow(self) -> None:
        """Evaluates user provided data for non-linear terms in advective fluxes on the
        boundary and stores them.

        Values fetched and stored include:

        - :meth:`bc_values_fractional_flow_component` (advection in component mass
          balance)
        - :meth:`bc_values_fractional_flow_energy` (advection in total energy mass
          balance)


        """

        # Updating BC values of non-linear weights in component mass balance equations
        # Dependent components are skipped.
        for component in self.fluid.components:
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

        # Updating BC values of the non-linear weight in the energy balance
        # (advected enthalpy)
        self.update_boundary_condition(
            name=self.bc_data_fractional_flow_energy_key,
            function=self.bc_values_fractional_flow_energy,
        )

    def bc_values_fractional_flow_component(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for the non-linear weight in the advective flux in
        :class:`ComponentMassBalanceEquations`, determining how much mass for respective
        ``component`` is entering the system on some inlet faces in relative terms.

        Parameters:
            component: A component in the fluid mixture.
            bg: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

        """
        return np.zeros(bg.num_cells)

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """BC values for the non-linear weight in the advective flux in
        the energy balance equation, determining how much energy/enthalpy is
        entering the system on some inlet faces in terms relative to the mass.

        Parameters:
            bg: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

        """
        return np.zeros(bg.num_cells)


class BoundaryConditionsCFF(
    _BoundaryConditionsAdvection,
    # put on top for override of update_all_boundary_values, which includes sub-routine
    # for fractional flow.
    BoundaryConditionsFF,
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
    energy.BoundaryConditionsEnthalpy,
    BoundaryConditionsFractions,
):
    """Collection of BC value routines required for CF in the fractional flow
    formulation."""


class BoundaryConditionsCF(
    _BoundaryConditionsAdvection,
    # put on top for override of update_all_boundary_values, which includes sub-routine
    # for updating phase properties on boundaries.
    BoundaryConditionsPhaseProperties,
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
    energy.BoundaryConditionsEnthalpy,
    BoundaryConditionsFractions,
):
    """Collection of BC values update routines required for CF, where phase properties
    are represented by surrogate factories and values need to be computed on the
    boundary, depending on primary variables."""


class InitialConditionsPhaseProperties(pp.InitialConditionMixin):
    """Extension of the initial condition mixing to provide a method which initializes
    values and derivative values for phase properties.

    This class assumes that phase properties are given as surrogate factories, which
    can get values assigned after initial values for their dependencies are set.

    """

    def initial_condition(self) -> None:
        """Calls :meth:`set_initial_values_phase_properties` after the super-call."""
        super().initial_condition()
        self.set_initial_values_phase_properties()

    def set_initial_values_phase_properties(self) -> None:
        """Method to set the initial values and derivative values of phase
        properties, which are surrogate factories with some dependencies.

        This method also fills all time and iterate indices with the initial values.
        Derivative values are only stored for the current iterate.

        """
        subdomains = self.mdg.subdomains()
        ni = self.iterate_indices.size
        nt = self.time_step_indices.size

        # Set the initial values on individual grids for the iterate indices
        for grid in subdomains:
            for phase in self.fluid.phases:
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
                # NOTE need the if-checks to satisfy mypy, since the properties are
                # type aliases containing some other type as well.
                for _ in self.iterate_indices:
                    if isinstance(phase.density, pp.ad.SurrogateFactory):
                        phase.density.progress_iterate_values_on_grid(
                            phase_props.rho, grid, depth=ni
                        )
                    if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                        phase.specific_enthalpy.progress_iterate_values_on_grid(
                            phase_props.h, grid, depth=ni
                        )
                    if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
                        phase.viscosity.progress_iterate_values_on_grid(
                            phase_props.mu, grid, depth=ni
                        )
                    if isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory):
                        phase.thermal_conductivity.progress_iterate_values_on_grid(
                            phase_props.kappa, grid, depth=ni
                        )
                # Copy values to all time step indices
                for _ in self.time_step_indices:
                    if isinstance(phase.density, pp.ad.SurrogateFactory):
                        phase.density.progress_values_in_time([grid], depth=nt)
                    if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                        phase.specific_enthalpy.progress_values_in_time(
                            [grid], depth=nt
                        )


class InitialConditionsFractions(
    pp.InitialConditionMixin, CompositionalFlowModelProtocol
):
    """Class providing interfaces to set initial values for various fractions in a
    general multi-component mixture.

    This base class provides only initialization routines for primary fractional
    variables, namely overall fractions per components and tracer fractions in
    compounds. Other fractions are assumed secondary.

    """

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for fractions at iterate index 0.

        See also:

            - :meth:`initial_overall_fraction`
            - :meth:`initial_tracer_fraction`

        """
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():

            # Setting overall fractions and tracer fractions
            for component in self.fluid.components:
                # independent overall fractions must have an initial value
                if self.has_independent_fraction(component):
                    self.equation_system.set_variable_values(
                        self.initial_overall_fraction(component, sd),
                        [cast(pp.ad.Variable, component.fraction([sd]))],
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
                                        pp.ad.Variable,
                                        component.tracer_fraction_of[tracer]([sd]),
                                    )
                                ],
                                iterate_index=0,
                            )

    def initial_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            component: A component in the fluid mixture with an independent overall
                fraction.
            sd: A subdomain in the md-grid.

        Returns:
            The initial overall fraction values for a component on a subdomain. Defaults
            to zero array.

        """
        return np.zeros(sd.num_cells)

    def initial_tracer_fraction(
        self, tracer: pp.Component, compound: ppc.Compound, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            tracer: An active tracer in the ``compound``.
            component: A compound in the fluid mixture.
            sd: A subdomain in the md-grid.

        Returns:
            The initial solute fraction values for a solute in a compound on a
            subdomain. Defaults to zero array.

        """
        return np.zeros(sd.num_cells)


class InitialConditionsCF(
    # put this on top because it overrides initial_condition
    InitialConditionsPhaseProperties,
    mass_energy.InitialConditionsMassAndEnergy,
    energy.InitialConditionsEnthalpy,
    InitialConditionsFractions,
):
    """Collection of initialization procedures for the general CF model."""


class SolutionStrategyCF(
    mass_energy.SolutionStrategyFluidMassAndEnergy,
    CompositionalFlowModelProtocol,
):
    """Solution strategy for general compositional flow.

    It provides means to enable a re-discretization of MPFA matrices for consistent
    discretizations of the diffusive parts of pressure and energy equation without
    upwinding.

    It provides a sub-routine to update thermodynamic phase properties which are
    assumed to be given by surrogate factories. The values and derivative values must
    respectively be updated before re-discretization. This update is performed
    before every nonlinear iteration. An update in time is performed after convergence.

    It also provides utilities for defining primary equations and variables
    to eliminate local equations via Schur complement.

    The initialization parameters can contain the following entries:

    - ``'eliminate_reference_phase'``: Defaults to True. If True, the molar fraction
      and saturation of the reference phase are eliminated by unity, reducing the size
      of the system. If False, more work is required by the modeller.
    - ``'eliminate_reference_component'``: Defaults to True. If True, the overall
      fraction of the reference component is eliminated by unity, reducing the number
      of unknowns. Also, the mass balance equation for the reference component is
      removed as an equation. If False, the modeller must close the system.
    - ``'fractional_flow'``: Defaults to False. If True, the model treats the
      non-linear weights in the advective fluxes in mass and energy balances as closed
      terms on the boundary. The user must then provide values for the non-linear
      weights explicitly. It also uses fractional mobilities, instead of regular ones.
      To be used with consistently discretized diffusive parts or balance equations
      (see also ``'rediscretize_mpfa'``).
    - ``'equilibrium_type'``: Defaults to None. If the model contains an equilibrium
      part, it should be a string indicating the fixed state of the local phase
      equilibrium problem e.g., ``'p-T'``,``'p-h'``. The string can also contain other
      qualifiers providing information about the equilibrium model, for example
      ``'unified-p-h'``.
    - ``'rediscretize_mpfa'``: Defaults to False. If True, the diffusive parts of the
      pressure and energy equation will be discretized consistently without using
      upwinding. This is computationally very expensive and assumes that the non-linear
      terms in the diffusive fluxes are isotropic contributions to the second-order
      tensors.
    - ``'reduce_linear_system'``: Defaults to False, If True, the solution strategy
      performs a Schur-complement elimination using defined primary variables and
      equations.

    """

    fourier_flux_discretization: Callable[[list[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[list[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self._nonlinear_flux_discretizations: list[pp.ad._ad_utils.MergedOperator] = []
        """Separate container for fluxes which need to be re-discretized. The separation
        is necessary due to the re-discretization being performed at different stages
        of the algorithm."""

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid."""

        # TODO consider purging this completely from porepy, similar to the advective
        # type BC. There is only 1 total mass flux and 1 inlet/outlet boundary.
        # No need to duplicate it for every equation with an advective part.
        self.enthalpy_keyword = self.mobility_keyword
        """Overwrites the enthalpy keyword for storing upwinding matrices for the
        advected enthalpy to be consistent with the general notion of mobility (both
        are based on the Darcy flux)."""

    @property
    def _rediscretize_mpfa(self) -> bool:
        """Property returning a flag from the model params, indicating whether the
        MPFA should be consistently re-discretized or upwinding is used.

        Can be passed as ``params['rediscretize_mpfa'] = True``.

        Defaults to False.

        """
        return bool(self.params.get("rediscretize_mpfa", False))

    @property
    def _reduce_linear_system(self) -> bool:
        """Property returning a flag from the model params, indicating whether the
        global linear system should be reduced via Schur complement using
        :meth:`primary_equation_names` and :meth:`primary_variable_names`.

        Can be passed as ``params['reduce_linear_system'] = True``.

        Defaults to False.

        """
        return bool(self.params.get("reduce_linear_system", False))

    def prepare_simulation(self) -> None:
        """Logs some information about the system."""
        start = time.time()
        super().prepare_simulation()
        duration = time.time() - start

        p_elim = pp.is_reference_phase_eliminated(self)
        c_elim = pp.is_reference_component_eliminated(self)
        is_ff = pp.is_fractional_flow(self)
        et = pp.get_equilibrium_type(self)
        var_names = set([v.name for v in self.equation_system.variables])
        dofs = self.equation_system.num_dofs()
        dofs_loc = dofs / len(var_names)
        var_msg = "\n\t\t".join(var_names)
        logger.info(
            f"Initialized CF model in {duration} seconds:\n"
            + f"\tEquilibrium type: {et}\n"
            + f"\tFractional flow: {is_ff}"
            + f"\tLocal equations eliminated (Schur): {self._reduce_linear_system}\n"
            + f"\tRe-discretize MPFA: {self._rediscretize_mpfa}\n"
            + f"\tNumber of phases: {self.fluid.num_phases}\n"
            + f"\tNumber of components: {self.fluid.num_components}\n"
            + f"\tReference phase eliminated: {p_elim}\n"
            + f"\tReference component eliminated: {c_elim}\n"
            + f"\tDOFs (locally): {dofs} ({dofs_loc})\n"
            + f"\tVariables: \n\t\t{var_msg}\n"
        )

    def primary_variable_names(self) -> list[str]:
        """Returns a list of primary variables, which in the basic set-up consist of

        1. pressure,
        2. overall fractions,
        3. tracer fractions,
        4. specific fluid enthalpy.

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

    def secondary_variables_names(self) -> list[str]:
        """Returns a list of secondary variables, which is defined as the complement
        of :meth:`primary_variable_names` and all variables found in the equation
        system.

        Note:
            Due to usage of Python's ``set``- operations, the resulting list may or may
            not be in the order the variables were created in the final model.

        """
        all_variables = set([var.name for var in self.equation_system.get_variables()])
        return list(all_variables.difference(set(self.primary_variable_names())))

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
                mass.MassBalanceEquations.primary_equation_name(),
                energy.EnergyBalanceEquations.primary_equation_name(),
            ]
            + self.component_mass_balance_equation_names()
            + self.tracer_transport_equation_names()
        )

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
        return list(all_equations.difference(set(self.primary_equation_names())))

    def add_nonlinear_flux_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of non-linear flux discretizations.

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
        """Adds additionally the non-linear MPFA discretizations to a separate list,
        since the updates are performed at different steps in the algorithm."""
        super().set_nonlinear_discretizations()

        if self._rediscretize_mpfa:
            subdomains = self.mdg.subdomains()
            self.add_nonlinear_flux_discretization(
                self.fourier_flux_discretization(subdomains).flux()
            )
            self.add_nonlinear_flux_discretization(
                self.darcy_flux_discretization(subdomains).flux()
            )

    def update_thermodynamic_properties_of_phases(self) -> None:
        """This method uses for each phase the underlying EoS to calculate
        new values and derivative values of phase properties and to update them
        them in the iterative sense, on all subdomains."""

        subdomains = self.mdg.subdomains()
        ni = self.iterate_indices.size

        for grid in subdomains:
            for phase in self.fluid.phases:
                dep_vals = [
                    d([grid]).value(self.equation_system)
                    for d in self.dependencies_of_phase_properties(phase)
                ]

                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals)
                )

                # Set current iterate indices of values and derivatives
                update_phase_properties(grid, phase, phase_props, ni)

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

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform an update of phase properties before
        calling the parent method to update discretizations.

        Performs also a re-discretization of the MPFA, if requested in a setting with
        consistent discretizations of the diffusive parts.

        See also:

            - :meth:`update_thermodynamic_properties_of_phases`
            - :meth:`rediscretize_fluxes`

        """
        self.update_thermodynamic_properties_of_phases()

        if self._rediscretize_mpfa:
            self.rediscretize_fluxes()
        # After updating the fluid properties and fluxes, call super to update
        # discretization parameters and non-linear discretizations like upwinding
        super().before_nonlinear_iteration()

    def after_nonlinear_convergence(self) -> None:
        """Progresses phase properties in time, if they are surrogate factories.

        Phase properties expected in the accumulation term (time-derivative) include
        density and specific enthalpy.

        """
        super().after_nonlinear_convergence()

        subdomains = self.mdg.subdomains()
        nt = self.time_step_indices.size
        for phase in self.fluid.phases:
            if isinstance(phase.density, pp.ad.SurrogateFactory):
                phase.density.progress_values_in_time(subdomains, depth=nt)
            if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                phase.specific_enthalpy.progress_values_in_time(subdomains, depth=nt)

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        This method performs a Schur complement elimination.

        Uses the primary equations defined in
        :meth:`EquationsCompositionalFlow.primary_equation_names` and the primary
        variables defined in
        :meth:`VariablesCF.primary_variable_names`.

        """
        t_0 = time.time()

        if self._reduce_linear_system:
            # TODO block diagonal inverter for secondary equations
            self.linear_system = self.equation_system.assemble_schur_complement_system(
                self.primary_equation_names(), self.primary_variable_names()
            )
        else:
            self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        sol = super().solve_linear_system()

        if self._reduce_linear_system:
            sol = self.equation_system.expand_schur_complement_solution(sol)
        return sol


# endregion


class ModelSetupCF(  # type: ignore[misc]
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
    """General class for setting up a multi-phase multi-component flow model, with
    thermodynamic properties of phases being represented as surrogate factories.

    The setup can be used as a starting point to add various thermodynamic models and
    correlations.

    The primary, transportable variables are:

    - pressure
    - specific enthalpy of the mixture
    - overall fractions per independent component
    - tracer fractions for pure transport without equilibrium (if any)

    The secondary, local variables are:

    - saturations per independent phase
    - phase fractions per independent phase (if any, related to equilibrium formulation)
    - fractions of components in phases (extended or partial)
    - temperature

    The primary block of equations consists of:

    - pressure equation / transport of total mass
    - energy balance / transport of total energy
    - transport equations for each independent component
    - tracer transport equations

    The secondary block of equations must be provided using constitutive relations
    or an equilibrium model for the fluid.

    Note:
        The model inherits the md-treatment of Darcy flux, advective enthalpy flux and
        Fourier flux. Some interface variables and interface equations are introduced
        there. They are treated as secondary equations and variables.

    """


class ModelSetupCFF(  # type: ignore[misc]
    # const. laws on top to overwrite what is used in inherited mass and energy balance
    ConstitutiveLawsCF,
    PrimaryEquationsCF,
    VariablesCF,
    BoundaryConditionsCFF,
    InitialConditionsCF,
    SolutionStrategyCF,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Similar to :class:`ModelSetupCF`, with the difference being the mixed-in
    BC values class.

    Fractional flow offer the possibility to provide non-linear terms in advective
    fluxes explicitely, without evaluating phase properties. This functionality is given
    by :class:`BoundaryConditionsFF`.

    """
