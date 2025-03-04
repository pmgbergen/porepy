"""Model mixins for compositional flow (CF) and fractional flow (CFF).

Also serves as a basis for compositional flow with local equilibrium formulations
(CFLE).

The CF settings is general in terms of number of phases, number of components and
thermal conditions. It therefore does not contain a runnable model, but building blocks
for models concretized in terms of phases, components and thermal setting.

The following equations are available:

- :class:`ComponentMassBalanceEquations`: While the :class:`~porepy.models.
  fluid_mass_balance.FluidMassBalanceEquations` represent the balance of total mass
  (pressure equation), this equation represent the balance equation of individual
  components in the fluid mixture. Interface fluxes are handled using the overall
  Darcy interface flux implemented in the pressure equations and an adapted non-linear
  term. It introduces a single equation, all interface equations are introduced by
  :class:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations`. Interface fluxes
  for individual components are computed using the total interface flux and a
  corresponding weight. As of now, the component mass balance equations are purely
  advective.
- :class:`MassicPressureEquations`: A diffusive variant of the total mass
  balance, where the total mobility is an isotropic contribution to the permeability
  tensor. To be used in a fractional flow model. This model is computationally
  expensive, since it requires a re-discretization of the MPFA, but also numerically
  consistent.
- :class:`EnthalpyBasedEnergyBalanceEquations`: A specialized total energy balance
  using an independent (specific fluid) enthalpy variable in the accumulation term.
  Otherwise completely analogous to its base
  :class:`~porepy.models.energy_balance.TotalEnergyBalanceEquations`.

The primary equations include a total mass balance, component balance and the energy
balance equation. A collection for non-isothermal, non-diffusive flow is given in
:class:`PrimaryEquationsCF`.

The primary variables are assumed to be a single pressure (no capillarity), an enthalpy
or temperature variable, and overall fraction variables. They are collected in
:class:`VariablesCF`.

Considering solely the flow formulation without constitutive modelling, where phase
properties are given by :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`
, special IC and BC classes handling the consistent update are given by
:class:`InitialConditionsPhaseProperties` and :class:`BoundaryConditionsPhaseProperties`
respectively. They handle the automatized computation of respective values during the
simulation.

The following IC mixins are available:

- :class:`InitialConditionsFractions`: Provides an interface to set initial conditions
  for overall fractions and active tracer fractions, if any. Other fractions are for now
  not covered since they are considered secondary and their initialization can be done
  based on the primary fractions.
- :class:`InitialConditionsPhaseProperties`: An initialization routine for models where
  phase properties are represented by :class:`~porepy.numerics.ad.surrogate_operator.
  SurrogateFactory`. Their initialization is dependent on variable values, and is
  automatized to happen after the variables are initialized.
- :class:`InitialConditionsCF`: A collection of above initialization
  routines, and the IC mixins for mass & energy, including an independent enthalpy
  variable.

The following BC mixins are available:

- :class:`BoundaryConditionsMulticomponent`: The analogy to
  :class:`InitialConditionsFractions` but for BC.
- :class:'BoundaryConditionsPhaseProperties': The analogy to
  :class:`InitialConditionsPhaseProperties` but for BC. Their update can be automatized
  to happen after values for variables are set on the boundary.
- :class:`BoundaryConditionsFractionalFlow`: An alternative to
  :class:`BoundaryConditionsPhaseProperties` for the fractional flow setting, where
  various non-linear terms in fluxes can be given explicitly on the boundary,
  instead of providing variable values which in return compute phase properties
  appearing in those expressions.
- :class:`BoundaryConditionsCF`: A collection of BC update routines for
  primary variables and phase properties as surrogate factories, including those from
  mass & energy, enthalpy and fractions.
- :class:`BoundaryConditionsCFF`: The alternative to
  :class:`BoundaryConditionsCF` for the fractional flow setting with
  explicit values for non-linear weights in fluxes.

The following solution strategy mixins are available:

- :class:`SolutionStrategyPhaseProperties`: A mixed in strategy for evaluating and
  storing fluid phase properties based on the underlying EoS and using the surrogate
  operator framework. This is a proper mixin, meaning it is not inheriting from any
  solution strategy and must be used together with some solution strategy for equations.
- :class:`SolutionStrategyNonlinearMPFA`: A mixed in strategy for re-discretizing MPFA
  discretizations of the Darcy and Fourier flux. It is also a proper mixin, to be used
  in combination with some other, fully functional solution strategy.
- :class:`SolutionStrategyCF`: Combining the solution strategy for updating phase
  properties, with the strategy for fluid mass and energy balance.
- :class:`SolutionStrategyCFF`: Like the previous strategy but with
  :class:`SolutionStrategyNonlinearMPFA` as an additional base.

The two template models, :class:`CompositionalFlowTemplate` and
:class:`CompositionalFractionalFlowTemplate` are starting points for non-isothermal,
multiphase, multicomponent flow & transport models. They do not contain the constitutive
modelling of fluid properties though and are hence not runable.
The steps required by users to close the models and obtain runable models are:

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
import porepy.compositional as compositional

logger = logging.getLogger(__name__)


def update_phase_properties(
    sd: pp.Grid,
    phase: pp.Phase,
    props: compositional.PhaseProperties,
    depth: int,
    update_derivatives: bool = True,
    use_extended_derivatives: bool = False,
) -> None:
    """Helper method to update the phase properties and its derivatives.

    This method is intended for a grid-local update of properties and their derivatives,
    using the methods of the
    :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`.

    Parameters:
        sd: A subdomain grid in the md-domain.
        phase: Any phase.
        props: A phase property structure containing the new values to be set.
        depth: An integer used to shift existing iterate values backwards.
        update_derivatives: ``default=True``

            If True, updates also the derivative values.
        use_extended_derivatives: ``default=False``

            If True, and if ``update_derivatives==True``, uses the extended derivatives
            of the ``state``.

            To be used in the the CFLE setting with the unified equilibrium formulation,
            where partial fractions are obtained by normalization of extended fractions.

    """
    if isinstance(phase.density, pp.ad.SurrogateFactory):
        phase.density.progress_iterate_values_on_grid(props.rho, sd, depth=depth)
        if update_derivatives:
            phase.density.set_derivatives_on_grid(
                props.drho_ext if use_extended_derivatives else props.drho, sd
            )
    if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
        phase.specific_enthalpy.progress_iterate_values_on_grid(
            props.h, sd, depth=depth
        )
        if update_derivatives:
            phase.specific_enthalpy.set_derivatives_on_grid(
                props.dh_ext if use_extended_derivatives else props.dh, sd
            )
    if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
        phase.viscosity.progress_iterate_values_on_grid(props.mu, sd, depth=depth)
        if update_derivatives:
            phase.viscosity.set_derivatives_on_grid(
                props.dmu_ext if use_extended_derivatives else props.dmu, sd
            )
    if isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory):
        phase.thermal_conductivity.progress_iterate_values_on_grid(
            props.kappa, sd, depth=depth
        )
        if update_derivatives:
            phase.thermal_conductivity.set_derivatives_on_grid(
                props.dkappa_ext if use_extended_derivatives else props.dkappa, sd
            )


def is_fractional_flow(model: pp.PorePyModel) -> bool:
    """Checking the model parameters for the ``'fractional_flow'`` flag.

    This check is separated from the models and their mixins for compatibility reasons
    since it is mainly used in the CF framework.

    Parameters:
        model: A PorePy model.

    Returns:
        True if ``model.params['fractional_flow'] == True`. Defaults to False.

    """
    return bool(model.params.get("fractional_flow", False))


def log_cf_model_configuration(model: pp.PorePyModel) -> None:
    """Performs a log of some model parameters and properties relevant for the CF
    framework."""

    p_elim = model._is_reference_phase_eliminated()
    c_elim = model._is_reference_component_eliminated()
    is_ff = is_fractional_flow(model)
    et = compositional.get_equilibrium_type(model)
    schur = model.params.get("reduce_linear_system", False)
    darcy = model.params.get("rediscretize_darcy_flux", False)
    fourier = model.params.get("rediscretize_fourier_flux", False)
    var_names = set([v.name for v in model.equation_system.variables])
    dofs = model.equation_system.num_dofs()
    dofs_loc = dofs / len(var_names)
    var_msg = "\n\t\t".join(var_names)

    logger.info(
        f"Configuration of model {model}:\n"
        + f"\tEquilibrium type: {et}\n"
        + f"\tFractional flow: {is_ff}"
        + f"\tEliminating secondary block via Schur complement: {schur}"
        + f"\tRe-discretizing Darcy flux: {darcy}"
        + f"\tRe-discretizing Fourier flux: {fourier}"
        + f"\tNumber of phases: {model.fluid.num_phases}\n"
        + f"\tNumber of components: {model.fluid.num_components}\n"
        + f"\tReference phase eliminated: {p_elim}\n"
        + f"\tReference component eliminated: {c_elim}\n"
        + f"\tDOFs (locally): {dofs} ({dofs_loc})\n"
        + f"\tVariables: \n\t\t{var_msg}\n"
    )


# region general PDEs.


class MassicPressureEquations(pp.fluid_mass_balance.FluidMassBalanceEquations):
    """A version of the pressure equation (total mass balance) which does not rely on
    upwinding of non-linear weights in the fluid flux on both subdomains and interfaces.

    This balance equation assumes that the transported mass and mobility terms are part
    of the diffusive, second-order tensor in the non-linear (MPFA) discretization of the
    Darcy flux. Correspondingly, the fluxes are not volumetric anymore, but massic.

    It **requires** a re-discretization of the flux (MPFA) for a correct computation
    of the flux at every iteration due to a non-linear tensor.

    """

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""

    def set_equations(self) -> None:
        """After the super-call, this method asserts additionally that the model is
        using the fractional flow framework."""
        super().set_equations()

        assert self.params["rediscretize_darcy_flux"], (
            "Model params['rediscretize_darcy_flux'] must be flagged as True."
        )

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """The fluid flux is given solely by the :attr:`darcy_flux`, assuming the total
        mobility is part of the (non-linear) diffuse tensor.

        Parameters:
            domains: List of subdomains or boundary grids.

        Returns:
            Whatever :attr:`darcy_flux` returns.

        """
        return self.darcy_flux(domains)

    def interface_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """The interface fluid flux is given solely by the :attr:`interface_darcy_flux`,
        assuming it is a massic flux.

        Parameters:
            interfaces: List of mortar grids.

        Returns:
            Whatever :attr:`interface_darcy_flux` returns.

        """
        return self.interface_darcy_flux(interfaces)

    def well_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """The well fluid flux is given solely by the :attr:`well_flux`,
        assuming it is a massic flux.

        Parameters:
            interfaces: List of mortar grids.

        Returns:
            Whatever :attr:`well_flux` returns.

        """
        return self.well_flux(interfaces)


class EnthalpyBasedEnergyBalanceEquations(
    pp.energy_balance.TotalEnergyBalanceEquations
):
    """Mixed-dimensional balance of total energy in a fluid mixture, formulated with an
    independent (specific fluid) enthalpy variable in the accumulation term *and* a
    temperature variable in the Fourier flux.

    Balance equation for all subdomains and advective and diffusive fluxes (Fourier
    flux) internally and on all interfaces of codimension one and advection on
    interfaces of codimension two (well-fracture intersections).

    The :meth:`advection_weight_energy_balance` is implemented such that this class is
    compatible with the fractional flow formulation. I.e., in the case where the total
    mass mobility is part of the permability tensor, the weight is equal to the
    transported enthalpy divided by total mass mobility. In the standard formulation it
    is only the transported enthalpy.

    Notes:
        1. Since enthalpy is an independent variable, models using this balance need
           a local equation relating temperature to the fluid enthalpy.
        2. This class relies on an interface enthalpy flux variable, which is put into
           relation with the interface darcy flux. It can be eliminated by using the
           interface darcy flux, weighed with the transported enthalpy. Room for
           optimization.

    """

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`EnthalpyVariable`."""

    total_mass_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    fractional_phase_mass_mobility: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""

    bc_data_fractional_flow_energy_key: str
    """See :class:`BoundaryConditionsMulticomponent`."""

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        r"""Returns the internal energy of the fluid using the independent
        :meth:`~porepy.models.energy_balance.EnthalpyVariable.enthalpy` variable and the
        :attr:`~porepy.compositional.base.Fluid.density` of the fluid mixture

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

        if is_fractional_flow(self) and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_energy_key,
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        elif is_fractional_flow(self):
            # Implementation NOTE: The operator tree size can be reduced, by pulling the
            # division by total mobility out of the sum. Consider at some point.
            op = pp.ad.sum_operator_list(
                [
                    phase.specific_enthalpy(domains)
                    * self.fractional_phase_mass_mobility(phase, domains)
                    # * phase.density(domains)
                    # * self.phase_mobility(phase, domains)
                    for phase in self.fluid.phases
                ],
            )  # / self.total_mobility(domains)
        else:
            # If the fractional-flow framework is not used, the weight corresponds to
            # the advected enthalpy and a super call is performed (where the respective
            # term is implemented).
            op = super().advection_weight_energy_balance(domains)

        op.set_name("advected_enthalpy")
        return op


class ComponentMassBalanceEquations(pp.BalanceEquation):
    """Mixed-dimensional balance of mass in a fluid mixture for present components.

    Since feed fractions per independent component are unknowns, the model requires
    additional transport equations to close the system.

    This equation is defined on all subdomains. Due to a single pressure and interface
    flux variable, there is no need for additional equations as is the case in the
    pressure equation. The interface flux for individual components is computed using
    the total interface flux provided by the pressure equation, a respective weight and
    upwind-coupling discretization.

    Important:
        The component mass balance equations expect the total mass balance (pressure
        equation) to be part of the system. That equation defines interface fluxes
        between subdomains, which are used by the present class to advect components.

        Also, this class relies on the Upwind discretization implemented there,
        especially on the definition of the boundary faces as either Neumann-type or
        Dirichlet-type. This is for memory and sanity reasons. The alternative would be
        to give every component mass balance the opportunity to define the advective
        flux on the boundary (not supported).

        In any case, the total fluid flux on the boundary is the sum of each component
        flux on the boundary (advection). For this reason, this class overrides the
        representation of the total mass flux on the boundary given by
        :class:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations.
        boundary_fluid_flux`.

    """

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`ConstitutiveLawsSolidSkeletonCF`."""

    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

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

    fractional_component_mass_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    component_mass_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See :class:`~porepy.models.fluid_property_library.FluidMobility`."""

    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.
    """

    bc_data_fractional_flow_component_key: Callable[[pp.Component], str]
    """See :class:`BoundaryConditionsFractionalFlow`."""
    bc_data_component_flux_key: Callable[[pp.Component], str]
    """See :class:`BoundaryConditionsMulticomponent`."""

    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

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
                sd_eq = self.component_mass_balance_equation(component, subdomains)
                self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def component_mass_balance_equation(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Mass balance equation for subdomains for a given component.

        Parameters:
            component: A component in the :attr:`fluid`.
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.volume_integral(
            self.component_mass(component, subdomains), subdomains, dim=1
        )
        flux = self.component_flux(component, subdomains)
        source = self.component_source(component, subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name(self._mass_balance_equation_name(component))
        return eq

    def component_mass(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Returns the accumulation term in a ``component``'s mass balance equation
        using the :attr:`~porepy.compositional.base.Fluid.density` of the fluid
        mixture and the component's
        :attr:`~porepy.compositional.base.Component.fraction`

        .. math::

            \Phi \rho \z_{\eta},

        in AD operator form on the given ``subdomains``.

        Parameters:
            component: A component in the :attr:`fluid`.
            subdomains: A list of subdomains on which above operator is called.

        Returns:
            Above expression in AD operator form.

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

        In the standard formulation, this term represents the mass of the component,
        advected by the Darcy flux. The advected quantity then has the physical
        dimensions [kg * m^(-3) * Pa^(-1) * s^(-1)].

        In the fractional flow formulation, it uses the ``component``'s
        :meth:`~porepy.models.fluid_property_library.FluidMobility.
        fractional_component_mass_mobility`, assuming the flux contains the total
        mobility in the diffusive tensor. Creates a boundary operator, in case explicit
        values for fractional flow BC are used.
        The advected quantity is dimensionless in this case.

        Parameters:
            component: A component in the :attr:`fluid`.
            domains: A list of subdomains or boundaries on which the operator is called.

        Returns:
            The non-linear weight in AD operator form.

        """

        op: pp.ad.Operator | pp.ad.TimeDependentDenseArray

        if is_fractional_flow(self) and all(
            [isinstance(g, pp.BoundaryGrid) for g in domains]
        ):
            op = self.create_boundary_operator(
                self.bc_data_fractional_flow_component_key(component),
                cast(Sequence[pp.BoundaryGrid], domains),
            )
        elif is_fractional_flow(self):
            op = self.fractional_component_mass_mobility(component, domains)
        else:
            op = self.component_mass_mobility(component, domains)

        return op

    def component_flux(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """The advective flux for the component mass balance equation.

        Can be called on the boundary to obtain a representation of user-given Neumann
        data.

        See Also:
            :meth:`advection_weight_component_mass_balance`

        Parameters:
            component: A component in the :attr:`fluid`.
            domains: A list of subdomains or boundaries on which the operator is called.

        Returns:
            If called with boundary grids, the Neumann data for the mass flux are
            returned wrapped in a dense array.

            If called with subdomains, the complete flux (including BC operators) is
            returned as an AD operator.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            if is_fractional_flow(self):
                return self.advection_weight_component_mass_balance(
                    component, domains
                ) * self.darcy_flux(domains)
            else:
                return self.create_boundary_operator(
                    self.bc_data_component_flux_key(component),
                    cast(Sequence[pp.BoundaryGrid], domains),
                )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("Domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        flux = self.advective_flux(
            domains,
            self.advection_weight_component_mass_balance(component, domains),
            self.mobility_discretization(domains),
            self.boundary_component_flux(component, domains),
            cast(
                Callable[[list[pp.MortarGrid]], pp.ad.Operator],
                partial(self.interface_component_flux, component),
            ),
        )
        flux.set_name(f"component_flux_{component.name}")
        return flux

    def boundary_component_flux(
        self, component: pp.Component, domains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """
        Parameters:
            component: A component in the :attr:`fluid`.

        Returns:
            An operator representing the combined BC data for the advective flux on the
            boundary
            (see :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`).

        """
        return self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=cast(
                Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
                partial(self.advection_weight_component_mass_balance, component),
            ),
            neumann_operator=cast(
                Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
                partial(self.component_flux, component),
            ),
            robin_operator=None,
            bc_type=self.bc_type_fluid_flux,
            name=f"bc_values_component_flux_{component.name}",
        )

    def boundary_fluid_flux(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Overrides the total fluid flux on the boundary in the pressure equation for
        consistency.

        In the multi-component case, an in/out-flux of mass can be provided for
        individual components. In this case though, the total flux accross the boundary
        is not a quantity which can be given independently. The total flux is given
        consistently by summing over individual component mass fluxes crossing the
        boundary. The respective term on the boundary in the total mass balance equation
        is hence not an explicit term, but implicitly computed. For this reason,
        :class:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations.
        boundary_fluid_flux` needs to be overridden.

        Important:
            When combining equation classes into a collective mixin, this class must be
            above the total mass balance for the override to hold.

        Parameters:
            subdomains: A list of subdomains.

        Returns:
            The total fluid flux accross the boundaries corresponding to ``subdomains``,
            given as a sum of fluxes accross boundaries of individual components.

        """
        return pp.ad.sum_operator_list(
            [
                self.boundary_component_flux(component, subdomains)
                for component in self.fluid.components
            ],
            "bc_values_total_fluid_flux",
        )

    def interface_component_flux(
        self, component: pp.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface component flux using a the interface darcy flux and
        :meth:`advection_weight_component_mass_balance`.

        See Also:
            :attr:`interface_advective_flux`

        Parameters:
            component: A component in the :attr:`fluid`.
            interfaces: A list of mortar grids in the :attr:`mdg` grid.

        Returns:
            The interface flux of mass corresponding to the given component.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advection_weight_component_mass_balance(component, subdomains)
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name(f"interface_component_flux_{component.name}")
        return flux

    def well_component_flux(
        self, component: pp.Component, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well component flux using a the well flux and
        :meth:`advection_weight_component_mass_balance`.

        See Also:
            :attr:`well_advective_flux`

        Parameters:
            component: A component in the :attr:`fluid`.
            interfaces: A list of mortar grids in the :attr:`mdg` grid.

        Returns:
            The well flux of mass corresponding to the given component.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        weight = self.advection_weight_component_mass_balance(component, subdomains)
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, weight, discr)
        flux.set_name(f"well_component_flux_{component.name}")
        return flux

    def component_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Analogous to
        :meth:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations.fluid_source`
        , but using :meth:`interface_component_flux` and :meth:`well_component_flux` to
        obtain the correct component flux accross
        interfaces.

        Parameters:
            component: A component in the :attr:`fluid`.
            subdomains: A list of subdomains in the :attr:`mdg`.

        Returns:
            The base method returns the sources corresponding to the fluxes in the
            mixed-dimensional setting.

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
        source = projection.mortar_to_secondary_int() @ self.interface_component_flux(
            component, interfaces
        )
        source.set_name(f"interface_component_flux_source_{component.name}")
        well_fluxes = (
            well_projection.mortar_to_secondary_int()
            @ self.well_component_flux(component, well_interfaces)
            - well_projection.mortar_to_primary_int()
            @ self.well_component_flux(component, well_interfaces)
        )
        well_fluxes.set_name(f"well_component_flux_source_{component.name}")
        source += subdomain_projection.cell_restriction(subdomains) @ (
            subdomain_projection.cell_prolongation(well_subdomains) @ well_fluxes
        )
        return source


# endregion
# region Intermediate mixins collecting variables, equations and constitutive laws.


class PrimaryEquationsCF(
    EnthalpyBasedEnergyBalanceEquations,
    ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
):
    """A collection of primary equations in the CF setting.

    They are PDEs consisting of

    - 1 fluid mass balance equation,
    - mass balance equations per component,
    - 1 energy balance,

    in this order (reverse order to the base classes).

    """


class PrimaryEquationsCFF(
    EnthalpyBasedEnergyBalanceEquations,
    ComponentMassBalanceEquations,
    MassicPressureEquations,
):
    """A collection of primary equations in the CFF setting.

    They are PDEs consisting of

    - 1 (massic) pressure equation,
    - mass balance equations per component,
    - 1 energy balance,

    and relies on re-discretization of the Darcy flux.

    """


class VariablesCF(
    pp.mass_and_energy_balance.VariablesFluidMassAndEnergy,
    pp.energy_balance.EnthalpyVariable,
    compositional.CompositionalVariables,
):
    """Bundles standard variables for non-isothermal flow (pressure and temperature)
    with fractional variables and an independent enthalpy variable."""


class ConstitutiveLawsSolidSkeletonCF(
    pp.constitutive_laws.MassWeightedPermeability,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.EnthalpyFromTemperature,
):
    """Collection of constitutive laws for the solid skeleton in the compositional
    flow framework.

    Note:
        It additionally provides a mixed-in method for relative permability which can be
        overridden until a proper framework for relative permabilities is developed.

    """

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


class ConstitutiveLawsCF(
    # NOTE must be on top to overwrite phase properties as general surrogate factories
    compositional.FluidMixin,
    ConstitutiveLawsSolidSkeletonCF,
    pp.constitutive_laws.ThermalConductivityCF,
    pp.constitutive_laws.FluidMobility,
    # Contains the Upwind for the enthalpy flux, otherwise not required.
    # TODO Consider putting discretizations strictly outside of classes providing
    # heuristics for thermodynamic properties.
    pp.constitutive_laws.EnthalpyFromTemperature,
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
# region Boundary condition mixins.


class BoundaryConditionsMulticomponent(pp.BoundaryConditionMixin):
    """Mixin providing boundary values for primary variables concering multi-component
    flow, and component flux values.

    Note:
        As for the enthalpy, these variables are only in specific cases accessed on the
        boundary. But they are required in case phase properties depend on them, and
        subsequently a consistent evaluation of non-linear terms in advective fluxes is
        required.

        A case where overall fractions appear explicitely on the boundary, is the
        single-phase, multi-component model. In this case the partial fractions of the
        single phase are equal to the overall fractions.

        The component flux values on the other hand, appear in every equation where
        Neumann-type data is given for the advective flux in the component mass balance
        equations.

    Important:
        In theory, each component flux can be given either as Neumann-type (mass flux)
        or Dirichlet-type (advection weight times volumetric flux) on the boundary for
        the Upwind discretization.

        With current states of implementations, this requires individual Upwind classes
        and respective storage of matrices, which can easily explode.

        For this reason, the discretization-type of the advected mass on the
        boundary is only given by ``bc_type_fluid_flux`` for *all* advective mass fluxes
        (component mass balance). Its implementation is mixed in by :class:`~porepy.
        models.mass_and_energy_balance.BoundaryConditionsMassAndEnergy`.

    """

    _overall_fraction_variable: Callable[[pp.Component], str]
    """Provided by mixin for compositional variables."""
    _tracer_fraction_variable: Callable[[pp.Component, compositional.Compound], str]
    """Provided by mixin for compositional variables."""
    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

    def update_all_boundary_conditions(self) -> None:
        """After the super-call, an update of component flux values on the boundary
        is performed for **all** components."""
        super().update_all_boundary_conditions()

        for component in self.fluid.components:
            # NOTE: We need the massic boundary flux for all components, also the
            # dependent one, since the total flux on the boundary is computed using the
            # user-provided values in bc_values_component_flux.
            self.update_boundary_condition(
                name=self.bc_data_component_flux_key(component),
                function=cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_component_flux, component),
                ),
            )

    def update_boundary_values_primary_variables(self) -> None:
        """Calls the user-provided data for overall fractions and tracer fractions after
        a super-call.

        See also:

            - :meth:`bc_values_overall_fraction`
            - :meth:`bc_values_tracer_fraction`

        """
        super().update_boundary_values_primary_variables()

        for component in self.fluid.components:
            # Update of tracer fractions on Dirichlet boundary.
            if isinstance(component, compositional.Compound):
                for tracer in component.active_tracers:
                    bc_vals = cast(
                        Callable[[pp.BoundaryGrid], np.ndarray],
                        partial(self.bc_values_tracer_fraction, tracer, component),
                    )
                    self.update_boundary_condition(
                        self._tracer_fraction_variable(tracer, component),
                        function=bc_vals,
                    )

            # Update of independent overall fractions on Dirichlet boundary.
            if self.has_independent_fraction(component):
                bc_vals = cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_overall_fraction, component),
                )
                self.update_boundary_condition(
                    name=self._overall_fraction_variable(component),
                    function=bc_vals,
                )

    def bc_data_component_flux_key(self, component: pp.Component) -> str:
        r"""
        Parameters:
            component: A component in the :attr:`fluid`.

        Returns:
            The name under which BC data in the form of
            :math:`\mathbf{f}\cdot\mathbf{n}` for the advective flux :math:`\mathbf{f}`
            in :class:`ComponentMassBalance` are stored in the dictionaries in the
            :attr:`mdg`.

        """
        return f"component_flux_{component.name}"

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            component: A component in the :attr:`fluid`.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of the overall
            fraction.

        """
        return np.zeros(bg.num_cells)

    def bc_values_tracer_fraction(
        self,
        tracer: pp.Component,
        compound: compositional.Compound,
        bg: pp.BoundaryGrid,
    ) -> np.ndarray:
        """BC values for active tracer fractions (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            tracer: A tracer in the ``compound``.
            compound: A component in the fluid mixture.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of the overall
            fraction.

        """
        return np.zeros(bg.num_cells)

    def bc_values_component_flux(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        r"""**Massic** component flux values on the boundary flagged as ``'neu'`` by
        :meth:`bc_type_fluid_flux`.

        The value of the component flux is given by :math:`\mathbf{f}\cdot\mathbf{n}`,
        where :math:`\mathbf{f} = a\mathbf{d}`. I.e. the massic component flux is
        given by the Darcy flux and an additional advection weight.

        Important:
            The component flux must be given for **each** component on the boundary,
            also for the (dependent) reference component. Otherwise the total advective
            flux on the boundary cannot be consistently computed.

        See also:
            :class:`ComponentMassBalance`

        Parameters:
            component: A component in the :attr:`fluid`.
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the mass
            component flux values on the provided boundary grid.
            Defaults to a zero array.

        """
        return np.zeros(bg.num_cells)


class BoundaryConditionsPhaseProperties(pp.BoundaryConditionMixin):
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

        For a more direct approach, see :class:`BoundaryConditionsFractionalFlow`,
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
                # Some work is required for BGs with zero cells.
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


class BoundaryConditionsFractionalFlow(pp.BoundaryConditionMixin):
    """Analogous to :class:`BoundaryConditionsPhaseProperties`, but providing means to
    define values of the non-linear terms in fluxes directly without using their full
    expression.

    This mixin requires a fractional flow model.

    """

    bc_data_fractional_flow_energy_key: str = "bc_data_fractional_flow_energy"
    """Key to store the BC values for the non-linear weight in the advective flux in the
    energy balance equation, for the case where explicit values are provided."""

    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

    def update_all_boundary_conditions(self) -> None:
        """Calls :meth:`update_boundary_values_fractional_flow` after the super-call.

        Raises:
            CompositionalModellingError: If this mixin is used in a non-fractional flow
                setting (see :class:`SolutionStrategyCFF`).

        """
        super().update_all_boundary_conditions()
        if is_fractional_flow(self):
            self.update_boundary_values_fractional_flow()
        else:
            raise pp.compositional.CompositionalModellingError(
                "Computing boundary values of fractional weights without flagging a"
                + " fractional flow setting in model parameters."
            )

    def bc_data_fractional_flow_component_key(self, component: pp.Component) -> str:
        """Key to store the BC values of the non-linear weight in the advective flux
        of a component's mass balance equation"""
        return f"bc_data_fractional_flow_{component.name}"

    def update_boundary_values_fractional_flow(self) -> None:
        """Evaluates user provided data for non-linear terms in advective fluxes on the
        boundary and stores them.

        Values fetched and stored include:

        - :meth:`bc_values_fractional_flow_component` (advection in component mass
          balance)
        - :meth:`bc_values_fractional_flow_energy` (advection in total energy mass
          balance)


        """

        # Updating BC values of non-linear weights in component mass balance equations.
        # Dependent components are skipped.
        for component in self.fluid.components:
            # NOTE: The independency of overall fractions is used to characterize the
            # dependency of fractional flow, since the fractional weights also fulfill
            # the unity constraint.
            # In practice, the fractional flow weight for the dependent component is
            # never used in any equation, since it's mass balance is not part of the
            # model equations.
            if self.has_independent_fraction(component):
                bc_func = cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_fractional_flow_component, component),
                )

                self.update_boundary_condition(
                    name=self.bc_data_fractional_flow_component_key(component),
                    function=bc_func,
                )

        # Updating BC values of the non-linear weight in the energy balance (advected
        # enthalpy).
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
            component: A component in the :attr:`fluid`.
            bg: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

        """
        return np.zeros(bg.num_cells)

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """BC values for the non-linear weight in the advective flux in the energy
        balance equation, determining how much energy/enthalpy is entering the system on
        some inlet faces in terms relative to the mass.

        Parameters:
            bg: A boundary grid in the mixed-dimensional grid.

        Returns:
            By default a zero array with shape ``(boundary_grid.num_cells,)``.

        """
        return np.zeros(bg.num_cells)


class BoundaryConditionsCF(
    # Put on top for override of update_all_boundary_values, which includes sub-routine
    # for updating phase properties on boundaries.
    BoundaryConditionsPhaseProperties,
    pp.energy_balance.BoundaryConditionsEnthalpy,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
    BoundaryConditionsMulticomponent,
):
    """Collection of BC values update routines required for CF, where phase properties
    are represented by surrogate factories and values need to be computed on the
    boundary, depending on primary variables."""


class BoundaryConditionsCFF(
    # put on top for override of update_all_boundary_values, which includes sub-routine
    # for fractional flow.
    BoundaryConditionsFractionalFlow,
    pp.energy_balance.BoundaryConditionsEnthalpy,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
    BoundaryConditionsMulticomponent,
):
    """Collection of BC value routines required for CF in the fractional flow
    formulation."""


# endregion
# region Initial condition mixins.


class InitialConditionsFractions(pp.InitialConditionMixin):
    """Class providing interfaces to set initial values for various fractions in a
    general multi-component mixture.

    This base class provides only initialization routines for primary fractional
    variables, namely overall fractions per components and tracer fractions in
    compounds. Other fractions are assumed secondary.

    """

    has_independent_tracer_fraction: Callable[
        [pp.Component, compositional.Compound], bool
    ]
    """Provided by mixin for compositional variables."""
    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for fractions at iterate index 0.

        See also:

            - :meth:`ic_values_overall_fraction`
            - :meth:`ic_values_tracer_fraction`

        """
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():
            # Setting overall fractions and tracer fractions.
            for component in self.fluid.components:
                # independent overall fractions must have an initial value.
                if self.has_independent_fraction(component):
                    self.equation_system.set_variable_values(
                        self.ic_values_overall_fraction(component, sd),
                        [cast(pp.ad.Variable, component.fraction([sd]))],
                        iterate_index=0,
                    )

                # All tracer fractions must have an initial value.
                if isinstance(component, compositional.Compound):
                    for tracer in component.active_tracers:
                        if self.has_independent_tracer_fraction(tracer, component):
                            self.equation_system.set_variable_values(
                                self.ic_values_tracer_fraction(tracer, component, sd),
                                [
                                    cast(
                                        pp.ad.Variable,
                                        component.tracer_fraction_of[tracer]([sd]),
                                    )
                                ],
                                iterate_index=0,
                            )

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            component: A component in the :attr:`fluid` with an independent overall
                fraction.
            sd: A subdomain in the md-grid.

        Returns:
            The initial overall fraction values for a component on a subdomain. Defaults
            to zero array.

        """
        return np.zeros(sd.num_cells)

    def ic_values_tracer_fraction(
        self, tracer: pp.Component, compound: compositional.Compound, sd: pp.Grid
    ) -> np.ndarray:
        """
        Parameters:
            tracer: An active tracer in the ``compound``.
            component: A compound in the :attr:`fluid`.
            sd: A subdomain in the md-grid.

        Returns:
            The initial solute fraction values for a solute in a compound on a
            subdomain. Defaults to zero array.

        """
        return np.zeros(sd.num_cells)


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

        # Set the initial values on individual grids for the iterate indices.
        for grid in subdomains:
            for phase in self.fluid.phases:
                dep_vals = [
                    d([grid]).value(self.equation_system)
                    for d in self.dependencies_of_phase_properties(phase)
                ]

                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals)
                )

                # Set values and derivative values for current current index.
                update_phase_properties(grid, phase, phase_props, ni)

                # Progress iterate values to all iterate indices.
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
                # Copy values to all time step indices.
                for _ in self.time_step_indices:
                    if isinstance(phase.density, pp.ad.SurrogateFactory):
                        phase.density.progress_values_in_time([grid], depth=nt)
                    if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                        phase.specific_enthalpy.progress_values_in_time(
                            [grid], depth=nt
                        )


class InitialConditionsCF(
    # Put this on top because it overrides initial_condition.
    InitialConditionsPhaseProperties,
    # Put this above mass and energy, in case enthalpy is evaluated depending on
    # p, T and fractions.
    pp.energy_balance.InitialConditionsEnthalpy,
    pp.mass_and_energy_balance.InitialConditionsMassAndEnergy,
    InitialConditionsFractions,
):
    """Collection of initialization procedures for the general CF model."""


# endregion
# region Solution strategies.


class SolutionStrategyPhaseProperties(pp.PorePyModel):
    """A mixin solution strategy for CF models which use surrogate operators for phase
    properties (as is the default in the fluid mixin).

    In this case, the phase properties must be evaluated and respective values and
    derivative values stored. The EoS of each phase is used to perform respective
    evaluation.

    Intended use is for models which implement custom EoS or correlations as EoS objects
    and use them as part of the constitutive modelling of fluid phase properties.

    This is a proper mixin providing only overloads of some methods. It is to be used
    in a model on top of a fully functional solution strategy.

    An understanding of constitutive modelling using
    :attr:`~porepy.compositional.base.Phase.eos` and :attr:`~porepy.compositional.
    compositional_mixins.FluidMixin.dependencies_of_phase_properties` is required.

    Note:
        When using this solution strategy mixin, make sure it is **above** all other
        solution strategies in order to work property. This is due to the assumed order
        of execution implemented here (property update before any super-call).

    """

    def update_thermodynamic_properties_of_phases(self) -> None:
        """This method uses for each phase the underlying EoS to calculate
        new values and derivative values of phase properties and to update them
        them in the iterative sense, on all subdomains.

        It is called in :meth:`before_nonlinear_iteration`.

        """

        subdomains = self.mdg.subdomains()
        ni = self.iterate_indices.size

        for grid in subdomains:
            for phase in self.fluid.phases:
                # Compute the values of variables/state functions on which the phase
                # properties depend.
                dep_vals = [
                    d([grid]).value(self.equation_system)
                    for d in self.dependencies_of_phase_properties(phase)
                ]
                # Compute phase properties using the phase EoS.
                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals)
                )

                # Set current iterate indices of values and derivatives.
                update_phase_properties(grid, phase, phase_props, ni)

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform an update of phase properties before
        performing a super-call.

        This overload assumes that re-discretizations are performed in the super-call,
        especially of upwinding and potentially the flux discretization.

        Fluid properties (surrogate operators) and their values must be updated before
        any re-discretization due to discretizations depending on these values. They
        appear in the non-linear part of various fluxes.

        The update is scoped in :meth:`update_thermodynamic_properties_of_phases`, which
        can be customized.

        """
        self.update_thermodynamic_properties_of_phases()
        # NOTE: Mypy complaints about trivial body of protocol.
        # But this is a mixin. We assert it is indeed a solution strategy and proceed.
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().before_nonlinear_iteration()  # type:ignore[safe-super]

    def after_nonlinear_convergence(self) -> None:
        """Progresses phase properties in time, if they are surrogate factories.

        Phase properties expected in the accumulation term (time-derivative) include
        density and specific enthalpy.

        The progression is performed after the super-call.

        """
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().after_nonlinear_convergence()  # type:ignore[safe-super]

        subdomains = self.mdg.subdomains()
        nt = self.time_step_indices.size
        for phase in self.fluid.phases:
            if isinstance(phase.density, pp.ad.SurrogateFactory):
                phase.density.progress_values_in_time(subdomains, depth=nt)
            if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                phase.specific_enthalpy.progress_values_in_time(subdomains, depth=nt)


class SolutionStrategyNonlinearMPFA(pp.PorePyModel):
    """Solution strategy mixin for models using a non-linear MPFA flux discretization
    which requires a re-discretization in each iteration.

    An example use case is compositional flow in the fractional formulation, where the
    total mass mobility is a non-linear, isotropic contribution of the second-order
    tensor in the Darcy flux.

    This solution strategy searches for flags ``params['rediscretize_darcy_flux']`` and
    ``params['rediscretize_fourier_flux']`` in the model parameters, and performs the
    re-discretization if flagged True. By default, they are assumed to be False.

    The flux discretizations are performed in :meth:`before_nonlinear_iteration`
    **before** the super-call. Note that this is critical since upwinding (which is
    expected to be part of the super-call) must be re-discretized **after** the fluxes
    are re-discretized.

    Notes:

        1. Re-discretizing the MPFA is expensive and will slow down the simulation
           noticeably.
        2. When mixing in this class, it must be above other, fully functional solution
           strategies in order for the super-call to work as intended.

    """

    fourier_flux_discretization: Callable[[list[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.FouriersLaw`."""
    darcy_flux_discretization: Callable[[list[pp.Grid]], pp.ad.MpfaAd]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    def __init__(self, params: Optional[dict] = None) -> None:
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().__init__(params)  # type:ignore[safe-super]

        self._nonlinear_flux_discretizations: list[pp.ad._ad_utils.MergedOperator] = []
        """Separate container for fluxes which need to be re-discretized. The separation
        is necessary due to the re-discretization being performed at different stages
        of the algorithm."""

    def add_nonlinear_flux_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of non-linear flux discretizations.

        Parameters:
            discretization: The nonlinear discretization to be added.

        """
        if discretization not in self._nonlinear_flux_discretizations:
            self._nonlinear_flux_discretizations.append(discretization)

    def set_nonlinear_discretizations(self) -> None:
        """After the super-call, this method adds the
        :meth:`fourier_flux_discretization` and the :meth:`darcy_flux_discretization`
        to the update framework using :meth:`add_nonlinear_flux_discretization`."""
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().set_nonlinear_discretizations()  # type:ignore[safe-super]

        subdomains = self.mdg.subdomains()

        if self.params.get("rediscretize_fourier_flux", False):
            self.add_nonlinear_flux_discretization(
                self.fourier_flux_discretization(subdomains).flux()
            )
        if self.params.get("rediscretize_darcy_flux", False):
            self.add_nonlinear_flux_discretization(
                self.darcy_flux_discretization(subdomains).flux()
            )

    def rediscretize_fluxes(self) -> None:
        """Discretizes added, nonlinear fluxes after ensuring uniqueness of
        discretizations for efficiency reasons."""
        # If the list is empty, fluxes are not re-discretized.
        # The list is empty if nothing was added during set_nonlinear_discretizations.
        if self._nonlinear_flux_discretizations:
            tic = time.time()
            # Get unique discretizations to save computational time, then discretize.
            unique_discretizations = pp.ad._ad_utils.uniquify_discretization_list(
                self._nonlinear_flux_discretizations
            )
            pp.ad._ad_utils.discretize_from_list(unique_discretizations, self.mdg)
            logger.info(
                "Re-discretized nonlinear fluxes in {} seconds".format(
                    time.time() - tic
                )
            )

    def before_nonlinear_iteration(self) -> None:
        """Overloads the parent method to call :meth:`rediscretize_fluxes` before the
        super-call.

        This order is crucial since the re-discretization of upwinding is expected in
        the super-call and the fluxes must be re-discretized before that.

        """
        self.rediscretize_fluxes()
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().before_nonlinear_iteration()  # type:ignore[safe-super]


class SolutionStrategySchurComplement(pp.PorePyModel):
    """Solution strategy mixing allowing the definition of primary variables and
    equations in order to perform a Schur-complement elimination and expansion during
    the solution of the linear system.

    Intended use is for large models with local, algebraic equations which are secondary
    in some sense.

    Example use cases are any CF models with multiple phases and components, which
    require either a closure in the form of constitutive equations or local equilibrium
    conditions.

    The Schur complement is defined by setting primary equations and variables. They
    define the rows and columns respectively of the primary diagonal block, which is
    *not* inverted for the Schur complement.

    In order for the Schur complement reduction to be performed, the user must
    provide a flag ``params['reduce_linear_system']`` in the model parameters. By
    default this flag is False.

    """

    pressure_variable: str
    enthalpy_variable: str

    overall_fraction_variables: list[str]
    tracer_fraction_variables: list[str]

    component_mass_balance_equation_names: Callable[[], list[str]]

    @property
    def primary_equations(self) -> list[str]:
        """Names of the primary equations.

        They define the row-block which does not contain the sub-matrix which is to be
        inverted for the Schur complement.

        Parameters:
            names: List of equation names to be set as primary equations.

        Raises:
            ValueError: If any name is not known to the model's equation system or the
                given names are not unique.

        Returns:
            The names of the equations (currently) defined as primary equations.

        """
        return self._primary_equations

    @primary_equations.setter
    def primary_equations(self, names: list[str]) -> None:
        known_equations = list(self.equation_system.equations.keys())
        for n in names:
            if n not in known_equations:
                raise ValueError(f"Equation {n} unknown to the equation system.")
        if len(set(names)) != len(names):
            raise ValueError("Primary equation names must be unique.")
        # Shallow copy for safety
        self._primary_equations = [n for n in names]

    @property
    def primary_variables(self) -> list[str]:
        """Names of the primary variables.

        They define the column-block which does not contain the sub-matrix which is to
        be inverted for the Schur complement.

        Parameters:
            names: List of variable names to be set as primary variables.

        Raises:
            ValueError: If any name is not known to the model's equation system or the
                given names are not unique.

        Returns:
            The names of the variables (currently) defined as primary variables.

        """
        return self._primary_variables

    @primary_variables.setter
    def primary_variables(self, names: list[str]) -> None:
        known_variables = list(set(v.name for v in self.equation_system.variables))
        for n in names:
            if n not in known_variables:
                raise ValueError(f"Variable {n} unknown to the equation system.")
        if len(set(names)) != len(names):
            raise ValueError("Primary variables names must be unique.")
        # Shallow copy for safety
        self._primary_variables = [n for n in names]

    @property
    def secondary_equations(self) -> list[str]:
        """The list of equation names indirectly defined as secondary.

        They are given as the complement of :meth:`primary_equations` within all
        equations found in the equation system.

        Note:
            Due to usage of Python's ``set``- operations, the resulting list may or may
            not be in the order the equations were added to the model.

        Returns:
            A list of equation names defining the rows of the sub-matrix to be
            inverted for the Schur complement.

        """
        all_equations = set([n for n in self.equation_system.equations.keys()])
        return list(all_equations.difference(set(self.primary_equations)))

    @property
    def secondary_variables(self) -> list[str]:
        """The list of variable (names) indirectly defined as secondary.

        They are given as the complement of :meth:`primary_variables` within all
        variables found in the equation system.

        Note:
            Due to usage of Python's ``set``- operations, the resulting list may or may
            not be in the order the variables were created in the final model.

        Returns:
            A list of variable names defining the columns of the sub-matrix to be
            inverted for the Schur complement.

        """
        all_variables = set([var.name for var in self.equation_system.variables])
        return list(all_variables.difference(set(self.primary_variables)))

    def get_primary_equations_cf(self) -> list[str]:
        """Returns a list of primary equations assumed to be the default in the CF
        setting.

        The list includes:

        1. The total mass balance equation.
        2. Component mass balance equations for each independent component.
        3. The total energy balance equation.

        """
        return [
            pp.fluid_mass_balance.FluidMassBalanceEquations.primary_equation_name(),
            pp.energy_balance.TotalEnergyBalanceEquations.primary_equation_name(),
        ] + self.component_mass_balance_equation_names()

    def get_primary_variables_cf(self) -> list[str]:
        """Returns a list of primary variables assumed to be the default in the CF
        setting.

        The list includes:

        1. The pressure variable.
        2. The overall fraction variables for each independent component.
        3. The tracer fraction variables for tracers in compounds (if any).
        4. The (specific fluid) enthalpy variable.

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

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        This method performs a Schur complement elimination.

        Uses the :meth:`primary_equations` and :meth:`primary_variables` to define the
        Schur complement.

        """
        t_0 = time.time()

        if self.params.get("reduce_linear_system", False):
            import scipy.sparse as sps
            from pypardiso import spsolve

            self.linear_system = self.equation_system.assemble_schur_complement_system(
                self.primary_equations,
                self.primary_variables,
                inverter=lambda x: sps.csr_matrix(spsolve(x, np.eye(x.shape[0]))),
            )
        else:
            self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""

        # NOTE mypy complaints about trivial body of protocol.
        # But this is a mixin. We assert it is indeed a solution strategy and proceed
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        sol = super().solve_linear_system()  # type:ignore[safe-super]

        if self.params.get("reduce_linear_system", False):
            sol = self.equation_system.expand_schur_complement_solution(sol)
        return sol


class SolutionStrategyExtendedFluidMassAndEnergy(
    pp.mass_and_energy_balance.SolutionStrategyFluidMassAndEnergy
):
    """Extended solution strategy for fluid mass and energy balance including an
    independent enthalpy variable.

    This solutionstrategy also equates the storage keyword for the hyperbolic
    discretization in the energy equation to the keyword of the hyperbolic
    discretization in the mass balance equations.

    This is a simple optimization step to reduce the number of upwind discretizations
    and stored matrices.

    In theory, the hyperbolic discretization for the advective flux in the energy
    equation can have its own class of discretization. But in practice it is the same as
    for the advective fluxes in various mass balances (upwinding). All are based on the
    total mass flux (Darcy).

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self.enthalpy_variable: str = "enthalpy"
        """Primary variable in the compositional flow model, denoting the total,
        transported (specific molar) enthalpy of the fluid."""

        self.enthalpy_keyword = self.mobility_keyword
        """Overwrites the enthalpy keyword for storing upwinding matrices for the
        advected enthalpy to be equal to the upwinding discretization for component
        mass balances."""


class SolutionStrategyCF(
    # NOTE: The MRO order here is critical for the execution of update routines before
    # the linear system is solved.
    SolutionStrategyPhaseProperties,
    SolutionStrategySchurComplement,
    SolutionStrategyExtendedFluidMassAndEnergy,
):
    """Solution strategy for general compositional flow.

    The generality refers to the fluid phase properties being surrogate operators. I.e,
    they are given by some underlying EoS and their values must be computed and stored
    explicitly at several steps in the algorithm.

    It uses a mixed-in solution strategy for phase property updates and is based on the
    fully functional solution strategy for fluid mass and energy balance equations,
    including an independent fluid enthalpy variable.

    Supports the following model parameters:

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
      (see also :class:`SolutionStrategyNonlinearMPFA`).

    """


class SolutionStrategyCFF(
    # NOTE: The MRO order here is critical for the execution of update routines before
    # the linear system is solved.
    SolutionStrategyPhaseProperties,
    SolutionStrategyNonlinearMPFA,
    SolutionStrategySchurComplement,
    SolutionStrategyExtendedFluidMassAndEnergy,
):
    """Solution strategy for compositional flow including a re-discretization of MPFA
    for Fourier and/or Darcy flux.

    It is analogous to :class:`SolutionStrategyCF`, with the addition of
    :class:`SolutionStrategyNonlinearMPFA`.

    The order of base classes is critical for the functionality to work as intended.
    I.e., first the phase properties are updated, then the flux is re-discretized, and
    finally other discretizations like upwinding are re-discretized via super-call to
    the solution strategy for fluid mass and energy balance.

    """


# endregion


class CompositionalFlowTemplate(  # type: ignore[misc]
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

    The model can be used as a starting point to add various thermodynamic models and
    correlations (constitutive modelling).

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

    The secondary block of equations must be provided using constitutive relations or an
    equilibrium model for the fluid.

    Important:
        This model is not runable. It is a skeleton for non-isothermal compositional
        flow. To close it, constitutive modelling is required.

    Note:
        The model inherits the md-treatment of Darcy flux, advective enthalpy flux and
        Fourier flux. Some interface variables and interface equations are introduced
        there. They are treated as secondary equations and variables.

    """


class CompositionalFractionalFlowTemplate(  # type: ignore[misc]
    ConstitutiveLawsCF,
    PrimaryEquationsCFF,
    VariablesCF,
    BoundaryConditionsCFF,
    InitialConditionsCF,
    SolutionStrategyCFF,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Similar to :class:`CompositionalFlowTemplate`, with the difference being the
    mixed-in BC values class.

    Fractional flow offer the possibility to provide non-linear terms in advective
    fluxes explicitely, without evaluating phase properties. This functionality is given
    by :class:`BoundaryConditionsFractionalFlow`.

    Correspondingly, this is a skeleton model. It is not runable and requires
    constitutive modelling from the user's side.

    """
