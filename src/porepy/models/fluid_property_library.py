"""Module containing some constant heuristic fluid property implementations and
the mixin :class:`FluidMobility`, which is required in all flow & transport problems.

Most of the laws implemented here are meant for 1-phase, 1-component mixtures, using
some fluid component stored in the fluid's reference component and have an analytical
expression which can be handled by PorePy's AD framework.

Note:
    In order to override default implementations of fluid properties in
    :class:`~porepy.compositional.compositional_mixins.FluidMixin`, the classes hererin
    must be mixed into the model *before* the fluid mixin.

    .. code::python

        class MyModel(
            # ...
            ConstitutiveLawFromFluidLibrary,
            # ...
            FluidMixin,
            # ...
        )

    E.g., Python must find :meth:`FluidDensityFromPressure.density_of_phase` before
    it finds the default
    :meth:`porepy.compositional.compositional_mixins.FluidMixin.density_of_phase`.

"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, List, Literal, Sequence, Union, cast

import numpy as np

import porepy as pp

__all__ = [
    "FluidDensityFromPressure",
    "FluidDensityFromTemperature",
    "FluidDensityFromPressureAndTemperature",
    "FluidMobility",
    "FluidBuoyancy",
    "ConstantViscosity",
    "ConstantFluidThermalConductivity",
    "FluidEnthalpyFromTemperature",
]

Scalar = pp.ad.Scalar
ExtendedDomainFunctionType = pp.ExtendedDomainFunctionType


class FluidDensityFromPressure(pp.PorePyModel):
    """Fluid density as a function of pressure for a single-phase, single-component
    fluid."""

    def fluid_compressibility(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant compressibility [Pa^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The fluid constant wrapped as an AD Scalar.

        """
        return Scalar(
            self.fluid.reference_component.compressibility, "fluid_compressibility"
        )

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a density exponential law for the fluid's phase.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        The reference density and the compressibility are taken from the material
        constants of the reference component, while the reference pressure is accessible
        by mixin; a typical implementation will provide this in a variable class.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

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

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Wrap compressibility from fluid class as matrix (left multiplication with dp).
        c = self.fluid_compressibility(subdomains)
        return exp(c * dp)


class FluidDensityFromTemperature(pp.PorePyModel):
    """Fluid density as a function of temperature for a single-phase, single-component
    fluid."""

    def fluid_thermal_expansion(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant thermal expansion [K^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            The constant wrapped in as an AD scalar.

        """
        val = self.fluid.reference_component.thermal_expansion
        return Scalar(val, "fluid_thermal_expansion")

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`FluidDensityFromPressure.density_of_phase`, but using
        temperature and the thermal expansion of the reference component.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ - c_T \\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

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

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        c = self.fluid_thermal_expansion(subdomains)
        return exp(Scalar(-1) * c * dtemp)


class FluidDensityFromPressureAndTemperature(
    FluidDensityFromPressure, FluidDensityFromTemperature
):
    """Fluid density which is a function of pressure and temperature, for a single-phase
    single-component fluid."""

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Returns a combination of the laws in the parent class methods:

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right)
            - c_T\\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

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


class FluidMobility(pp.PorePyModel):
    """Class for fluid mobility and its discretization in flow & transport equations."""

    relative_permeability: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """Provided by some mixin dealing with the porous medium (work in progress).

    Only relevant in the multi-phase case.

    """

    mobility_keyword: str
    """Keyword for the discretization of the mobility. Normally provided by a mixin of
    instance :class:`~porepy.models.SolutionStrategy`.

    """

    def mobility_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.UpwindAd:
        r"""Discretization of the fluid mobility.

        This includes any non-linear, scalar expression :math:`a` in front of the
        advective flux :math:`q`.

        .. math::

            -\nabla \cdot \left(a q\right).

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

        As for :meth:`mobility_discretization`, this involves any advection weight.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)

    def total_mass_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Total mass mobility of the fluid mixture is given by

        .. math::

                \sum_j \frac{\rho_j k_r(s_j)}{\mu_j}.

        Used as a non-linear part of the diffusive tensor in the (total) mass balance
        equation.

        Note:
            In the single-phase, single-component case, this is reduced to
            :math:`\frac{\rho}{\mu}`.

        Parameters:
            domains: A list of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        name = "total_mass_mobility"
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains) * self.phase_mobility(phase, domains)
                for phase in self.fluid.phases
            ],
            name,
        )
        return mobility

    def phase_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the mobility of a phase :math:`j`

        .. math::

            \frac{k_r(s_j)}{\mu_j}.

        Notes:
            For the single-phase case it returns simply :math:`\frac{1}{\mu}`.

        Important:
            Contrary to all other mobility methods implemented here, this one does not
            contain any mass term, it is a volumetric term. This is the term commonly
            denoted 'mobility in the literature.

        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        # Distinguish between single-phase case and multi-phase case: Usage of rel-perm
        # makes this class compatible with single-phase models, without requiring some
        # rel-perm mixin.
        if self.fluid.num_phases > 1:
            mobility = self.relative_permeability(phase, domains) / phase.viscosity(
                domains
            )
        else:
            assert phase == self.fluid.reference_phase
            mobility = phase.viscosity(domains) ** pp.ad.Scalar(-1.0)
        mobility.set_name(f"phase_mobility_{phase.name}")
        return mobility

    def component_mass_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear term in the advective flux in a component mass balance equation.

        It is obtained by summing :meth:`phase_mobility` weighed with
        :attr:`~porepy.compositional.base.Phase.partial_fraction_of` the component,
        and the phase :attr:`~porepy.compositional.base.Phase.density`,
        if the component is present in the phase.

        .. math::

                \sum_j x_{n, ij} \rho_j \frac{k_r(s_j)}{\mu_j},

        Note:
            In the single-phase, single-component case, this is reduced to
            :math:`\frac{\rho}{\mu}`.

        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        if self.fluid.num_phases > 1 or self.fluid.num_components > 1:
            # NOTE: This method is kept as general as possible when typing the
            # signature. But the default fluid of the PorePyModel consists of
            # FluidComponent, not Component. Adding type:ignore for this reason.
            mobility = pp.ad.sum_operator_list(
                [
                    phase.partial_fraction_of[component](domains)
                    * phase.density(domains)
                    * self.phase_mobility(phase, domains)
                    for phase in self.fluid.phases
                    if component in phase  # type:ignore[operator]
                ],
            )
        # This branch is for compatibility with single-phase or single component
        # models, which do not have the complete notion of fractions.
        else:
            assert component == self.fluid.reference_component
            mobility = self.fluid.reference_phase.density(
                domains
            ) * self.phase_mobility(self.fluid.reference_phase, domains)

        mobility.set_name(f"component_mass_mobility_{component.name}")
        return mobility

    def fractional_component_mass_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the :meth:`component_mass_mobility` divided by the
        :meth:`total_mass_mobility` for a component :math:`\eta`.

        To be used in component mass balance equations in a fractional flow model, where
        the total mobility is part of the non-linear diffusive tensor in the Darcy flux.

        .. math::

            - \nabla \cdot \left(f_{\eta} D(x) \nabla p\right),

        where the tensor :math:`D(x)` contains the total mobility.

        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\eta}` in above expession in operator form.

        """
        frac_mob = self.component_mass_mobility(
            component, domains
        ) / self.total_mass_mobility(domains)
        frac_mob.set_name(f"fractional_component_mass_mobility_{component.name}")
        return frac_mob

    def fractional_phase_mass_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the product of the ``phase`` density and :meth:`phase_mobility`
        divided by the :meth:`total_mass_mobility`.

        To be used in balance equations in a fractional flow model, where the total
        mobility is part of the non-linear diffusive tensor in the Darcy flux.

        I.e. for a phase :math:`\gamma`

        .. math::

            - \nabla \cdot \left(f_{\gamma} D(x) \nabla p\right),

        assuming the tensor :math:`D(x)` contains the total mobility.

        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\gamma}` in above expession in operator form.

        """
        frac_mob = (
            phase.density(domains)
            * self.phase_mobility(phase, domains)
            / self.total_mass_mobility(domains)
        )
        frac_mob.set_name(f"fractional_phase_mass_mobility_{phase.name}")
        return frac_mob


class FluidBuoyancy(pp.PorePyModel):
    """
    Buoyancy terms and discretizations for multiphase, multicomponent flow.

    This class is based on the fixed-dimensional hybrid upwinding scheme presented in
    Bosma et al. (2022), "Smooth implicit hybrid upwinding for compositional multiphase
    flow in porous media" (CMA, 388, 114288). Here, we implement an alternate version of
    the scheme using a total mass formulation.

    The main difference in this implementation is the consistent treatment of the
    gravity term, following Starnoni et al. (2019),
    "Consistent MPFA Discretization for Flow in the Presence of Gravity"
    (WRR, 55(12), 10105–10118).

    This implementation is novel in three main aspects: non-isothermal compositional
    multiphase flow, mixed-dimensional formulation,
    and the fractional flow form of the equations.

    Mass, component mass, and energy conservation are tested in `test_buoyancy_flow.py`,
    and a benchmark against an analytical buoyancy solution is provided in
    `test_buoyancy_flow_benchmark.py`.

    The class uses operator caching to reduce the size of the AD operator tree by
    reusing phase-dependent operators across different component/enthalpy buoyancy
    calculations.
    """

    # Storage for common operators to reduce operator tree size.
    _common_operators: dict

    component_mass_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`FluidMobility`."""

    fractional_phase_mass_mobility: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`FluidMobility`."""

    phase_mobility: Callable[[pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`FluidMobility`."""

    darcy_flux_discretization: Callable[
        [list[pp.Grid]], pp.ad.MpfaAd
    ]  # because it contains the div(w(rho)) term
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.ConstantPermeability`."""

    def buoyancy_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for subdomain buoyancy between phases gamma and delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the buoyancy term between the two phases. Can be used on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyancy_intf_key`

        """
        return "buoyancy_" + gamma.name + "_" + delta.name

    def buoyant_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on subdomains.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the array of buoyancy flux between the two phases on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyant_intf_flux_array_key`

        """
        return "buoyant_flux_" + gamma.name + "_" + delta.name

    def buoyancy_intf_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for interface buoyancy between phases gamma and delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the buoyancy term between the two phases. Can be used on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyancy_key`

        """
        return "buoyancy_intf_" + gamma.name + "_" + delta.name

    def buoyant_intf_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on interfaces.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the array of buoyancy flux between the two phases on
                interfaces.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyant_flux_array_key`

        """
        return "buoyant_intf_flux_" + gamma.name + "_" + delta.name

    def buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Return upwind discretization for subdomain buoyancy term gamma↔delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            subdomains: The subdomains to consider for the discretization.

        Returns:
            An Upwind discretization for the buoyancy term between the two phases.

        """
        storage = self._get_common_operators_storage()
        key = f"discr_{gamma.name}_{delta.name}"

        if key not in storage:
            discr = pp.ad.UpwindAd(self.buoyancy_key(gamma, delta), subdomains)
            assert isinstance(discr._discretization, pp.Upwind)
            discr._discretization.flux_array_key = self.buoyant_flux_array_key(gamma, delta)
            storage[key] = discr

        return storage[key]

    def _get_common_bound_transport_neu(
        self, gamma: pp.Phase, delta: pp.Phase, domains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Get common bound_transport_neu operator for the phase pair.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.

        Returns:
            The bound_transport_neu operator.

        """
        storage = self._get_common_operators_storage()
        key = f"bound_neu_{gamma.name}_{delta.name}"

        if key not in storage:
            discr = self.buoyancy_discretization(gamma, delta, domains)
            storage[key] = discr.bound_transport_neu()

        return storage[key]

    def _get_common_upwind_op(
        self, gamma: pp.Phase, delta: pp.Phase, domains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Get cached upwind operator for a phase pair.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.

        Returns:
            The upwind operator.

        """
        storage = self._get_common_operators_storage()
        key = f"upwind_op_{gamma.name}_{delta.name}"

        if key not in storage:
            discr = self.buoyancy_discretization(gamma, delta, domains)
            storage[key] = discr.upwind()

        return storage[key]

    def _get_common_intf_upwind_primary_chain(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached interface upwind_primary @ mortar_avg @ trace chain.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            primary_trace: Trace operator.

        Returns:
            The combined upwind_primary @ mortar_avg @ trace operator chain.

        """
        storage = self._get_common_operators_storage()
        key = f"intf_up_primary_chain_{gamma.name}_{delta.name}"

        if key not in storage:
            intf_discr = self.interface_buoyancy_discretization(gamma, delta, interfaces)
            storage[key] = intf_discr.upwind_primary() @ mortar_avg @ primary_trace

        return storage[key]

    def _get_common_intf_upwind_secondary_chain(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        interfaces: list[pp.MortarGrid],
        secondary_to_mortar: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached interface upwind_secondary @ secondary_to_mortar chain.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            interfaces: The interfaces.
            secondary_to_mortar: Projection from secondary to mortar.

        Returns:
            The combined upwind_secondary @ secondary_to_mortar operator chain.

        """
        storage = self._get_common_operators_storage()
        key = f"intf_up_secondary_chain_{gamma.name}_{delta.name}"

        if key not in storage:
            intf_discr = self.interface_buoyancy_discretization(gamma, delta, interfaces)
            storage[key] = intf_discr.upwind_secondary() @ secondary_to_mortar

        return storage[key]

    def _get_common_bound_neu_mortar_chain(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        mortar_to_primary: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached bound_transport_neu @ mortar_to_primary chain.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            mortar_to_primary: Projection from mortar to primary.

        Returns:
            The combined bound_neu @ mortar_to_primary operator chain.

        """
        storage = self._get_common_operators_storage()
        key = f"bound_neu_mortar_chain_{gamma.name}_{delta.name}"

        if key not in storage:
            bound_neu = self._get_common_bound_transport_neu(gamma, delta, domains)
            storage[key] = bound_neu @ mortar_to_primary

        return storage[key]

    def interface_buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Return upwind discretization for interface buoyancy term gamma-delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            interfaces: The interfaces to consider for the discretization.

        Returns:
            An Upwind discretization for the buoyancy term between the two phases.

        """
        storage = self._get_common_operators_storage()
        key = f"intf_discr_{gamma.name}_{delta.name}"

        if key not in storage:
            discr = pp.ad.UpwindCouplingAd(self.buoyancy_intf_key(gamma, delta), interfaces)
            assert isinstance(discr._discretization, pp.UpwindCoupling)
            discr._discretization.flux_array_key = self.buoyant_intf_flux_array_key(
                gamma, delta
            )
            storage[key] = discr

        return storage[key]

    def _get_common_operators_storage(self) -> dict:
        """Get or initialize storage for common buoyancy operators.

        Returns:
            The dictionary for storing common operators and related objects.

        """
        if not hasattr(self, "_common_operators"):
            self._common_operators: dict = {}
        return self._common_operators

    def _phase_pair_key(self, gamma: pp.Phase, delta: pp.Phase) -> tuple[str, str, int]:
        """Get ordered phase pair key and sign for density difference.

        Uses alphabetical ordering of phase names to ensure consistent caching.
        Returns the ordered key and a sign (+1 or -1) to apply to get the correct
        density difference direction (rho_gamma - rho_delta).

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            Tuple of (first_name, second_name, sign) where sign is +1 if gamma comes
            first alphabetically, -1 otherwise.

        """
        if gamma.name <= delta.name:
            return (gamma.name, delta.name, 1)
        else:
            return (delta.name, gamma.name, -1)

    def _get_common_density_flux(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        flux_type: str,
    ) -> pp.ad.Operator:
        """Get common density-driven flux operator for a phase pair.

        Uses reciprocity: w_flux(gamma, delta) = -w_flux(delta, gamma).

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            flux_type: Either "subdomain" or "interface".

        Returns:
            The density-driven flux operator with correct sign.

        """
        storage = self._get_common_operators_storage()
        first, second, sign = self._phase_pair_key(gamma, delta)

        # Key for the specific direction (gamma, delta)
        key = f"w_flux_{flux_type}_{gamma.name}_{delta.name}"

        if key not in storage:
            # First check if we have the canonical (first, second) version
            canonical_key = f"w_flux_{flux_type}_{first}_{second}"
            negated_key = f"w_flux_{flux_type}_{second}_{first}"

            if canonical_key not in storage:
                # Compute canonical version (first - second)
                phases_by_name = {p.name: p for p in self.fluid.phases}
                phase_first = phases_by_name[first]
                phase_second = phases_by_name[second]

                # Use cached density operators
                rho_first = self._get_common_density(phase_first, domains)
                rho_second = self._get_common_density(phase_second, domains)
                rho_diff = rho_first - rho_second

                if flux_type == "subdomain":
                    storage[canonical_key] = self.density_driven_flux(domains, rho_diff)
                else:  # interface
                    interfaces = self.subdomains_to_interfaces(domains, [1])
                    storage[canonical_key] = self.interface_density_driven_flux(interfaces, rho_diff)

                # Also cache the negated version to avoid creating new operator
                storage[negated_key] = pp.ad.Scalar(-1) * storage[canonical_key]

            # Now set the requested direction
            storage[key] = storage[key] if key in storage else (
                storage[canonical_key] if sign == 1 else storage[negated_key]
            )

        return storage[key]

    def _get_common_density(
        self, phase: pp.Phase, domains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Get cached density operator for a phase.

        Parameters:
            phase: The phase.
            domains: The subdomains.

        Returns:
            The density operator.

        """
        storage = self._get_common_operators_storage()
        key = f"density_{phase.name}"

        if key not in storage:
            storage[key] = phase.density(domains)

        return storage[key]

    def _get_common_fractional_mobility(
        self, phase: pp.Phase, domains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Get common fractional phase mass mobility operator.

        Parameters:
            phase: The phase.
            domains: The subdomains.

        Returns:
            The fractional mobility operator.

        """
        storage = self._get_common_operators_storage()
        key = f"f_mob_{phase.name}"

        if key not in storage:
            storage[key] = self.fractional_phase_mass_mobility(phase, domains)

        return storage[key]

    def clear_common_operators(self) -> None:
        """Clear the common buoyancy operators storage.

        Call this when domains change or at the start of a new operator tree
        construction if needed.

        """
        if hasattr(self, "_common_operators"):
            self._common_operators.clear()
        if hasattr(self, "_common_projections"):
            self._common_projections = None

    def fractionally_weighted_density(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Compute the fractional-flow-weighted density.

        The method computes the sum_j f_j rho_j where f_j is the fractional flow
        function for phase j.

        Parameters:
            domains: The domains to consider for the density computation.

        Returns:
            An operator representing the fractional-flow-weighted density.

        """
        storage = self._get_common_operators_storage()
        key = "fractionally_weighted_density"

        if key not in storage:
            domains_list = cast(list[pp.Grid], domains)
            overall_rho = pp.ad.sum_operator_list(
                [
                    self._get_common_fractional_mobility(phase, domains_list)
                    * self._get_common_density(phase, domains_list)
                    for phase in self.fluid.phases
                ]
            )
            overall_rho.set_name("fractionally_weighted_density")
            storage[key] = overall_rho

        return storage[key]

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Return gravity magnitude.

        Parameters:
            subdomains: The subdomains to consider for the gravity field computation.
                Not used, but included for consistency.

        Returns:
            An operator representing the gravity field.

        """
        storage = self._get_common_operators_storage()
        key = "gravity_field"

        if key not in storage:
            g_constant = pp.GRAVITY_ACCELERATION
            val = self.units.convert_units(g_constant, "m*s^-2")
            gravity_field = pp.ad.Scalar(val)
            gravity_field.set_name("gravity_field")
            storage[key] = gravity_field

        return storage[key]

    def gravity_force(
        self,
        subdomains: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: Literal["fluid", "solid", "bulk"],
    ) -> pp.ad.Operator:
        """Return gravity force term (fluid only if buoyancy enabled).

        Depending on the material type, the gravity force term is computed differently.
        If `material` is "fluid" and gravitational effects are not explicitly disabled
        (self.params["enable_buoyancy_effects"] is set to False), the method calculates
        the product of the fractionally weighted density with the gravity field, as a
        vector pointing in the negative third direction. If material is "solid" or
        "bulk", or gravitational effects are disabled, the method passes the calculation
        to a equally named super class.

        Parameters:
            subdomains: The domains to consider for the gravity force computation.
            material: The material for which to compute the gravity force.

        Raises:
            TypeError: If subdomains are instances of pp.MortarGrid.

        Returns:
            An Ad operator representing the gravitational force.

        """
        if material == "fluid" and self.params.get("enable_buoyancy_effects", True):
            # Narrow to list[pp.Grid] for calls needing subdomain grids, which is the
            # intended usage of this method. The listing of list[pp.MortarGrid] in the
            # method annotation is used for compatibility with equally named methods in
            # other parts of the code.
            if not all(isinstance(g, pp.Grid) for g in subdomains):
                raise TypeError(
                    "gravity_force expects only subdomain grids for "
                    "buoyancy computation."
                )
            subdomains_list = cast(list[pp.Grid], list(subdomains))

            # Cache the overall gravity flux
            storage = self._get_common_operators_storage()
            key = "overall_gravity_flux"

            if key not in storage:
                fractionally_weighted_rho = self.fractionally_weighted_density(
                    subdomains_list
                )
                e_n = self._get_common_e_n(subdomains_list)
                g = self.gravity_field(subdomains_list)
                neg_one = self._get_common_neg_one()
                overall_gravity_flux = (
                    neg_one
                    * e_n
                    @ (fractionally_weighted_rho * g)
                )
                overall_gravity_flux.set_name("overall gravity flux")
                storage[key] = overall_gravity_flux

            return storage[key]
        else:
            return super().gravity_force(subdomains, material)  # type:ignore

    def _get_common_neg_one(self) -> pp.ad.Operator:
        """Get cached Scalar(-1) operator.

        Returns:
            The Scalar(-1) operator.

        """
        storage = self._get_common_operators_storage()
        key = "neg_one"

        if key not in storage:
            storage[key] = pp.ad.Scalar(-1)

        return storage[key]

    def _get_common_e_n(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Get cached e_n unit vector operator.

        Parameters:
            subdomains: The subdomains.

        Returns:
            The e_n unit vector operator.

        """
        storage = self._get_common_operators_storage()
        key = "e_n"

        if key not in storage:
            storage[key] = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)

        return storage[key]

    def _get_common_jump_zero(self, domains: list[pp.Grid]) -> pp.ad.Operator:
        """Get cached zero array for buoyancy jump terms.

        Parameters:
            domains: The subdomains.

        Returns:
            Zero operator for buoyancy jumps.

        """
        storage = self._get_common_operators_storage()
        key = "jump_zero"

        if key not in storage:
            size = sum(g.num_cells for g in domains)
            storage[key] = pp.wrap_as_dense_ad_array(
                np.zeros(size), name="buoyancy_jump_zero"
            )

        return storage[key]


    def density_driven_flux(
        self, subdomains: pp.SubdomainsOrBoundaries, density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        """Compute flux induced by density_metric * g along gravity direction.

        The density metric will be defined elsewhere, it can for instance be a measure
        of density difference between two phases.

        The gravitational flux is discretized by the vector source scheme of the Darcy
        flux discretization.

        Parameters:
            subdomains: The subdomains to consider for the density-driven flux
                computation.
            density_metric: The density metric to use in the flux computation.

        Raises:
            TypeError: If subdomains are not instances of pp.Grid.

        Returns:
            An Ad operator representing the density-driven flux.

        """
        if not all(isinstance(g, pp.Grid) for g in subdomains):
            raise TypeError("density_driven_flux expects only subdomain grids.")
        subdomains_list = cast(list[pp.Grid], list(subdomains))

        # Use common intermediate operators.
        storage = self._get_common_operators_storage()
        key = "subdomain_flux_ops"

        if key not in storage:
            e_n = self._get_common_e_n(subdomains_list)
            g = self.gravity_field(subdomains_list)
            discr = self.darcy_flux_discretization(subdomains_list)
            vec_src = discr.vector_source()
            neg_one = self._get_common_neg_one()
            storage[key] = (e_n, g, vec_src, neg_one)

        e_n, g, vec_src, neg_one = storage[key]

        gravity_flux = neg_one * e_n @ (density_metric * g)
        w_flux = vec_src @ gravity_flux
        w_flux.set_name("density_driven_flux_" + density_metric.name)
        return w_flux

    def interface_density_driven_flux(
        self, interfaces: list[pp.MortarGrid], density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        """Compute interface flux induced by density_metric * g along gravity
        direction.

        The density metric is computed elsewhere and can for instance be a measure of
        density difference between two phases.

        Parameters:
            interfaces: Interfaces where the density metric is applied.
            density_metric: The density metric to use in the flux computation.

        Returns:
            An Ad operator representing the interface density-driven flux.

        """
        # Use common intermediate operators.
        storage = self._get_common_operators_storage()
        key = "interface_flux_ops"

        if key not in storage:
            normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
            subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
            projection = pp.ad.MortarProjections(
                self.mdg, subdomain_neighbors, interfaces, dim=self.nd
            )
            e_n = self.e_i(subdomain_neighbors, i=self.nd - 1, dim=self.nd)
            g = self.gravity_field(subdomain_neighbors)
            sec_to_mortar = projection.secondary_to_mortar_avg()
            nd_to_scalar_sum = pp.ad.sum_projection_list(
                [e.T for e in self.basis(interfaces, dim=self.nd)]
            )
            normal_perm = self.normal_permeability(interfaces)
            neg_one = self._get_common_neg_one()
            storage[key] = (normals, e_n, g, sec_to_mortar, nd_to_scalar_sum,
                               normal_perm, interfaces, neg_one)

        normals, e_n, g, sec_to_mortar, nd_to_scalar_sum, normal_perm, intf, neg_one = storage[key]

        gravity_flux = neg_one * e_n @ (density_metric * g)
        intf_vector_source = sec_to_mortar @ gravity_flux
        normals_times_source = normals * intf_vector_source

        w_flux = self.volume_integral(
            normal_perm * (nd_to_scalar_sum @ normals_times_source),
            intf,
            1,
        )
        w_flux.set_name("interface_density_driven_flux_" + density_metric.name)
        return w_flux

    def _interface_upwinded_quantity(
        self,
        quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        interfaces: list[pp.MortarGrid],
        domains: list[pp.Grid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Upwind a quantity to interfaces using both primary and secondary sides.

        Parameters:
            quantity: The quantity to upwind (already defined on domains).
            gamma: The phase for which the discretization is set up.
            delta: The other phase in the pair.
            interfaces: The interfaces where the upwinding is performed.
            domains: The subdomains.
            mortar_avg: Projection from primary to mortar (pre-computed).
            secondary_to_mortar: Projection from secondary to mortar (pre-computed).
            primary_trace: Trace operator (pre-computed).

        Returns:
            Upwinded quantity on interfaces.

        """
        # Use cached projection chains
        primary_chain = self._get_common_intf_upwind_primary_chain(
            gamma, delta, interfaces, mortar_avg, primary_trace
        )
        secondary_chain = self._get_common_intf_upwind_secondary_chain(
            gamma, delta, interfaces, secondary_to_mortar
        )
        return primary_chain @ quantity + secondary_chain @ quantity


    def phase_pairs_for(self, phase: pp.Phase) -> list[tuple[pp.Phase, pp.Phase]]:
        """Get all phase pairs involving a specific phase.

        The pair is ordered so that the given phase is first.

        Parameters:
            phase: The phase for which to get the pairs.

        Returns:
            A list of ordered phase pairs (gamma, delta).

        """
        combination_by_pairs = [
            pair for pair in list(combinations(self.fluid.phases, 2)) if phase in pair
        ]
        selected_pairs = []
        for pair in combination_by_pairs:
            idx = pair.index(phase)
            if idx == 0:
                phase_gamma, phase_delta = pair
            elif idx == 1:
                phase_delta, phase_gamma = pair
            else:
                continue
            selected_pairs.append((phase_gamma, phase_delta))
        return selected_pairs

    def _unique_phase_pairs(self) -> list[tuple[pp.Phase, pp.Phase]]:
        """Get unique unordered phase pairs.

        Returns:
            List of unique phase pairs (gamma, delta) where gamma.name < delta.name.

        """
        return list(combinations(self.fluid.phases, 2))

    def _get_common_upwinded_mobility(
        self,
        phase: pp.Phase,
        other_phase: pp.Phase,
        domains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Get common upwinded fractional mobility operator for a phase.

        Parameters:
            phase: The phase for which to get the upwinded mobility.
            other_phase: The other phase in the pair (used for discretization key).
            domains: The subdomains.

        Returns:
            Upwinded fractional mobility operator.

        """
        storage = self._get_common_operators_storage()
        key = f"f_upwind_{phase.name}_{other_phase.name}"

        if key not in storage:
            f = self._get_common_fractional_mobility(phase, domains)
            upwind_op = self._get_common_upwind_op(phase, other_phase, domains)
            storage[key] = upwind_op @ f

        return storage[key]

    def _get_common_interface_upwinded_mobility(
        self,
        phase: pp.Phase,
        other_phase: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get common interface-upwinded fractional mobility operator for a phase.

        Parameters:
            phase: The phase for which to get the upwinded mobility.
            other_phase: The other phase in the pair.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            Interface-upwinded fractional mobility operator.

        """
        storage = self._get_common_operators_storage()
        key = f"f_intf_upwind_{phase.name}_{other_phase.name}"

        if key not in storage:
            f = self._get_common_fractional_mobility(phase, domains)
            storage[key] = self._interface_upwinded_quantity(
                f, phase, other_phase, interfaces, domains,
                mortar_avg, secondary_to_mortar, primary_trace,
            )

        return storage[key]

    def _get_common_mobility_product(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Get common mobility product (f_gamma_up * f_delta_up) for a phase pair.

        This product is the same for all component and enthalpy buoyancy calculations,
        so caching it significantly reduces the operator tree size.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.

        Returns:
            The product of upwinded mobilities for the phase pair.

        """
        storage = self._get_common_operators_storage()
        # Use ordered key to cache both (gamma, delta) and (delta, gamma) products
        key = f"f_product_{gamma.name}_{delta.name}"

        if key not in storage:
            f_gamma_up = self._get_common_upwinded_mobility(gamma, delta, domains)
            f_delta_up = self._get_common_upwinded_mobility(delta, gamma, domains)
            storage[key] = f_gamma_up * f_delta_up

        return storage[key]

    def _get_common_interface_mobility_product(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get common interface mobility product for a phase pair.

        This product is the same for all component and enthalpy buoyancy calculations,
        so caching it significantly reduces the operator tree size.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The product of interface-upwinded mobilities for the phase pair.

        """
        storage = self._get_common_operators_storage()
        key = f"f_intf_product_{gamma.name}_{delta.name}"

        if key not in storage:
            f_gamma_intf = self._get_common_interface_upwinded_mobility(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            f_delta_intf = self._get_common_interface_upwinded_mobility(
                delta, gamma, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            storage[key] = f_gamma_intf * f_delta_intf

        return storage[key]

    def _get_common_weighted_mobility_flux(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Get common (f_gamma_up * f_delta_up) * w_flux for a phase pair.

        This combined operator is used in all buoyancy flux calculations.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.

        Returns:
            The mobility product times density-driven flux.

        """
        storage = self._get_common_operators_storage()
        key = f"weighted_flux_{gamma.name}_{delta.name}"

        if key not in storage:
            f_product = self._get_common_mobility_product(gamma, delta, domains)
            w_flux = self._get_common_density_flux(gamma, delta, domains, "subdomain")
            storage[key] = f_product * w_flux

        return storage[key]

    def _get_common_interface_weighted_mobility_flux(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get common interface (f_gamma * f_delta) * w_flux for a phase pair.

        This combined operator is used in all interface buoyancy flux calculations.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The interface mobility product times density-driven flux.

        """
        storage = self._get_common_operators_storage()
        key = f"weighted_intf_flux_{gamma.name}_{delta.name}"

        if key not in storage:
            f_product = self._get_common_interface_mobility_product(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            intf_w_flux = self._get_common_density_flux(
                gamma, delta, domains, "interface"
            )
            storage[key] = f_product * intf_w_flux

        return storage[key]


    def _get_common_chi_diff_interface(
        self,
        component: pp.Component,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached (chi_gamma_intf - chi_delta_intf) for interface buoyancy.

        Parameters:
            component: The component.
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The chi difference operator on interfaces.

        """
        storage = self._get_common_operators_storage()
        key = f"chi_diff_intf_{component.name}_{gamma.name}_{delta.name}"

        if key not in storage:
            chi_gamma_intf = self._get_common_interface_upwinded_chi(
                component, gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            chi_delta_intf = self._get_common_interface_upwinded_chi(
                component, delta, gamma, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            storage[key] = chi_gamma_intf - chi_delta_intf


        return storage[key]

    def _get_common_enthalpy_diff_interface(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached (h_gamma_intf - h_delta_intf) for interface buoyancy.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The enthalpy difference operator on interfaces.

        """
        storage = self._get_common_operators_storage()
        key = f"h_diff_intf_{gamma.name}_{delta.name}"

        if key not in storage:
            h_gamma_intf = self._get_common_interface_upwinded_enthalpy(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            h_delta_intf = self._get_common_interface_upwinded_enthalpy(
                delta, gamma, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            storage[key] = h_gamma_intf - h_delta_intf

        return storage[key]

    def _get_common_intf_coupling_chi(
        self,
        component: pp.Component,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached interface coupling for component buoyancy.

        This is (chi_gamma - chi_delta) * weighted_intf_flux, shared between
        component_buoyancy and component_buoyancy_jump.

        Parameters:
            component: The component.
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The interface coupling operator for component buoyancy.

        """
        storage = self._get_common_operators_storage()
        key = f"intf_coupling_chi_{component.name}_{gamma.name}_{delta.name}"

        if key not in storage:
            chi_diff = self._get_common_chi_diff_interface(
                component, gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            weighted_intf_flux = self._get_common_interface_weighted_mobility_flux(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            storage[key] = chi_diff * weighted_intf_flux

        return storage[key]

    def _get_common_intf_coupling_enthalpy(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get cached interface coupling for enthalpy buoyancy.

        This is (h_gamma - h_delta) * weighted_intf_flux, shared between
        enthalpy_buoyancy and enthalpy_buoyancy_jump.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            The interface coupling operator for enthalpy buoyancy.

        """
        storage = self._get_common_operators_storage()
        key = f"intf_coupling_h_{gamma.name}_{delta.name}"

        if key not in storage:
            h_diff = self._get_common_enthalpy_diff_interface(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            weighted_intf_flux = self._get_common_interface_weighted_mobility_flux(
                gamma, delta, domains, interfaces,
                mortar_avg, secondary_to_mortar, primary_trace,
            )
            storage[key] = h_diff * weighted_intf_flux

        return storage[key]

    def _get_common_chi(
        self,
        component: pp.Component,
        phase: pp.Phase,
        domains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Get cached chi (partial fraction) operator for a component/phase.

        Parameters:
            component: The component.
            phase: The phase.
            domains: The subdomains.

        Returns:
            Chi operator.

        """
        storage = self._get_common_operators_storage()
        key = f"chi_{component.name}_{phase.name}"

        if key not in storage:
            storage[key] = phase.partial_fraction_of[component](domains)

        return storage[key]

    def _get_common_enthalpy(
        self,
        phase: pp.Phase,
        domains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Get cached enthalpy operator for a phase.

        Parameters:
            phase: The phase.
            domains: The subdomains.

        Returns:
            Enthalpy operator.

        """
        storage = self._get_common_operators_storage()
        key = f"enthalpy_{phase.name}"

        if key not in storage:
            storage[key] = phase.specific_enthalpy(domains)

        return storage[key]

    def _get_common_interface_upwinded_chi(
        self,
        component: pp.Component,
        phase: pp.Phase,
        other_phase: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get common interface-upwinded chi operator for a component/phase.

        Parameters:
            component: The component.
            phase: The phase for which to get the upwinded chi.
            other_phase: The other phase in the pair.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            Interface-upwinded chi operator.

        """
        storage = self._get_common_operators_storage()
        key = f"chi_intf_{component.name}_{phase.name}_{other_phase.name}"

        if key not in storage:
            chi = self._get_common_chi(component, phase, domains)
            storage[key] = self._interface_upwinded_quantity(
                chi, phase, other_phase, interfaces, domains,
                mortar_avg, secondary_to_mortar, primary_trace,
            )

        return storage[key]


    def _get_common_interface_upwinded_enthalpy(
        self,
        phase: pp.Phase,
        other_phase: pp.Phase,
        domains: list[pp.Grid],
        interfaces: list[pp.MortarGrid],
        mortar_avg: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Get common interface-upwinded enthalpy operator for a phase.

        Parameters:
            phase: The phase for which to get the upwinded enthalpy.
            other_phase: The other phase in the pair.
            domains: The subdomains.
            interfaces: The interfaces.
            mortar_avg: Projection from primary to mortar.
            secondary_to_mortar: Projection from secondary to mortar.
            primary_trace: Trace operator.

        Returns:
            Interface-upwinded enthalpy operator.

        """
        storage = self._get_common_operators_storage()
        key = f"h_intf_{phase.name}_{other_phase.name}"

        if key not in storage:
            h = self._get_common_enthalpy(phase, domains)
            storage[key] = self._interface_upwinded_quantity(
                h, phase, other_phase, interfaces, domains,
                mortar_avg, secondary_to_mortar, primary_trace,
            )

        return storage[key]

    def _get_common_projections(
        self, domains: list[pp.Grid], interfaces: list[pp.MortarGrid]
    ) -> tuple:
        """Get common projection operators for interface computations.

        Parameters:
            domains: The subdomains.
            interfaces: The interfaces.

        Returns:
            Tuple of (mortar_avg, secondary_to_mortar, primary_trace,
                     mortar_to_primary, mortar_to_secondary).

        """
        if not hasattr(self, "_common_projections"):
            self._common_projections = None

        if self._common_projections is None:
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, domains, interfaces, dim=1
            )
            trace = pp.ad.Trace(domains)
            self._common_projections = (
                mortar_projection.primary_to_mortar_avg(),
                mortar_projection.secondary_to_mortar_avg(),
                trace.trace,
                mortar_projection.mortar_to_primary_int(),
                mortar_projection.mortar_to_secondary_int(),
            )

        return self._common_projections

    def _compute_all_buoyancy_operators(
        self, domains: list[pp.Grid]
    ) -> None:
        """Pre-compute all buoyancy operators for all components and enthalpy.

        This method computes and caches all buoyancy-related operators in one pass,
        maximizing operator reuse and minimizing the AD graph size.

        Parameters:
            domains: The subdomains.
        """
        storage = self._get_common_operators_storage()

        # Check if already computed
        if "all_buoyancy_computed" in storage:
            return

        # Get all components that need buoyancy computation
        components = list(self.fluid.components)

        # Pre-compute shared operators for all phase pairs
        for gamma, delta in self._unique_phase_pairs():
            # Compute and cache weighted mobility flux (shared across all)
            _ = self._get_common_weighted_mobility_flux(gamma, delta, domains)

        # Pre-compute chi and enthalpy base values (leaf operators)
        for phase in self.fluid.phases:
            _ = self._get_common_enthalpy(phase, domains)
            for comp in components:
                _ = self._get_common_chi(comp, phase, domains)

        # Handle interfaces if present
        interfaces = self.subdomains_to_interfaces(domains, [1])
        if len(interfaces) != 0:
            mortar_avg, secondary_to_mortar, primary_trace, mortar_to_primary, _ = \
                self._get_common_projections(domains, interfaces)

            for gamma, delta in self._unique_phase_pairs():
                # Pre-compute interface weighted flux and bound_neu chain (shared)
                _ = self._get_common_interface_weighted_mobility_flux(
                    gamma, delta, domains, interfaces,
                    mortar_avg, secondary_to_mortar, primary_trace,
                )
                _ = self._get_common_bound_neu_mortar_chain(
                    gamma, delta, domains, mortar_to_primary
                )

        storage["all_buoyancy_computed"] = True

    def component_buoyancy(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the buoyancy flux for a given component."""
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        storage = self._get_common_operators_storage()
        key = f"component_buoyancy_final_{component_xi.name}"

        if key not in storage:
            self._compute_all_buoyancy_operators(domains)

            b_fluxes: List[pp.ad.Operator] = []

            for gamma, delta in self._unique_phase_pairs():
                weighted_flux = self._get_common_weighted_mobility_flux(gamma, delta, domains)
                upwind_gamma = self._get_common_upwind_op(gamma, delta, domains)
                upwind_delta = self._get_common_upwind_op(delta, gamma, domains)
                chi_gamma = self._get_common_chi(component_xi, gamma, domains)
                chi_delta = self._get_common_chi(component_xi, delta, domains)

                chi_diff = upwind_gamma @ chi_gamma - upwind_delta @ chi_delta
                b_fluxes.append(chi_diff * weighted_flux)

            interfaces = self.subdomains_to_interfaces(domains, [1])
            if len(interfaces) != 0:
                mortar_avg, secondary_to_mortar, primary_trace, mortar_to_primary, _ = \
                    self._get_common_projections(domains, interfaces)

                for gamma, delta in self._unique_phase_pairs():
                    intf_coupling = self._get_common_intf_coupling_chi(
                        component_xi, gamma, delta, domains, interfaces,
                        mortar_avg, secondary_to_mortar, primary_trace,
                    )
                    bound_neu_chain = self._get_common_bound_neu_mortar_chain(
                        gamma, delta, domains, mortar_to_primary
                    )
                    b_fluxes.append(bound_neu_chain @ intf_coupling)

            if not b_fluxes:
                b_fluxes.append(self._get_common_jump_zero(domains))

            b_flux = pp.ad.sum_operator_list(b_fluxes)
            b_flux.set_name("component_buoyancy_" + component_xi.name)
            storage[key] = b_flux

        return storage[key]

    def enthalpy_buoyancy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Get the buoyancy flux for specific enthalpy."""
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        storage = self._get_common_operators_storage()
        key = "enthalpy_buoyancy_final"

        if key not in storage:
            self._compute_all_buoyancy_operators(domains)

            b_fluxes: List[pp.ad.Operator] = []

            for gamma, delta in self._unique_phase_pairs():
                weighted_flux = self._get_common_weighted_mobility_flux(gamma, delta, domains)
                upwind_gamma = self._get_common_upwind_op(gamma, delta, domains)
                upwind_delta = self._get_common_upwind_op(delta, gamma, domains)
                h_gamma = self._get_common_enthalpy(gamma, domains)
                h_delta = self._get_common_enthalpy(delta, domains)

                h_diff = upwind_gamma @ h_gamma - upwind_delta @ h_delta
                b_fluxes.append(h_diff * weighted_flux)

            interfaces = self.subdomains_to_interfaces(domains, [1])
            if len(interfaces) != 0:
                mortar_avg, secondary_to_mortar, primary_trace, mortar_to_primary, _ = \
                    self._get_common_projections(domains, interfaces)

                for gamma, delta in self._unique_phase_pairs():
                    intf_coupling = self._get_common_intf_coupling_enthalpy(
                        gamma, delta, domains, interfaces,
                        mortar_avg, secondary_to_mortar, primary_trace,
                    )
                    bound_neu_chain = self._get_common_bound_neu_mortar_chain(
                        gamma, delta, domains, mortar_to_primary
                    )
                    b_fluxes.append(bound_neu_chain @ intf_coupling)

            if not b_fluxes:
                b_fluxes.append(self._get_common_jump_zero(domains))

            b_flux = pp.ad.sum_operator_list(b_fluxes)
            b_flux.set_name("enthalpy_buoyancy")
            storage[key] = b_flux

        return storage[key]

    def component_buoyancy_jump(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for component buoyancy."""
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        storage = self._get_common_operators_storage()
        key = f"component_buoyancy_jump_final_{component_xi.name}"

        if key not in storage:
            self._compute_all_buoyancy_operators(domains)

            zero = self._get_common_jump_zero(domains)
            b_flux_jumps: List[pp.ad.Operator] = [zero]

            interfaces = self.subdomains_to_interfaces(domains, [1])
            if len(interfaces) != 0:
                mortar_avg, secondary_to_mortar, primary_trace, _, mortar_to_secondary = \
                    self._get_common_projections(domains, interfaces)

                for gamma, delta in self._unique_phase_pairs():
                    # Reuse the same intf_coupling as component_buoyancy
                    intf_coupling = self._get_common_intf_coupling_chi(
                        component_xi, gamma, delta, domains, interfaces,
                        mortar_avg, secondary_to_mortar, primary_trace,
                    )
                    b_flux_jumps.append(mortar_to_secondary @ intf_coupling)

            b_flux = pp.ad.sum_operator_list(b_flux_jumps)
            b_flux.set_name("component_buoyancy_jump_" + component_xi.name)
            storage[key] = b_flux

        return storage[key]

    def enthalpy_buoyancy_jump(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for enthalpy buoyancy."""
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        storage = self._get_common_operators_storage()
        key = "enthalpy_buoyancy_jump_final"

        if key not in storage:
            self._compute_all_buoyancy_operators(domains)

            zero = self._get_common_jump_zero(domains)
            b_flux_jumps: List[pp.ad.Operator] = [zero]

            interfaces = self.subdomains_to_interfaces(domains, [1])
            if len(interfaces) != 0:
                mortar_avg, secondary_to_mortar, primary_trace, _, mortar_to_secondary = \
                    self._get_common_projections(domains, interfaces)

                for gamma, delta in self._unique_phase_pairs():
                    # Reuse the same intf_coupling as enthalpy_buoyancy
                    intf_coupling = self._get_common_intf_coupling_enthalpy(
                        gamma, delta, domains, interfaces,
                        mortar_avg, secondary_to_mortar, primary_trace,
                    )
                    b_flux_jumps.append(mortar_to_secondary @ intf_coupling)

            b_flux = pp.ad.sum_operator_list(b_flux_jumps)
            b_flux.set_name("enthalpy_buoyancy_jump")
            storage[key] = b_flux

        return storage[key]

    def set_buoyancy_discretization_parameters(self):
        """Initialize parameter containers and zero flux arrays for buoyancy."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                for sd, data in self.mdg.subdomains(return_data=True):
                    pp.initialize_data(data, self.buoyancy_key(gamma, delta))
                    pp.initialize_data(data, self.buoyancy_key(delta, gamma))
                    null_vals = np.zeros(sd.num_faces)
                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {self.buoyant_flux_array_key(gamma, delta): +null_vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {self.buoyant_flux_array_key(delta, gamma): -null_vals}
                    )
                for intf, data in self.mdg.interfaces(return_data=True):
                    null_vals = np.zeros(intf.num_cells)
                    pp.initialize_data(data, self.buoyancy_intf_key(gamma, delta))
                    pp.initialize_data(data, self.buoyancy_intf_key(delta, gamma))
                    data[pp.PARAMETERS][self.buoyancy_intf_key(gamma, delta)].update(
                        {self.buoyant_intf_flux_array_key(gamma, delta): +null_vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(delta, gamma)].update(
                        {self.buoyant_intf_flux_array_key(delta, gamma): -null_vals}
                    )

    def set_nonlinear_buoyancy_discretization(self):
        """Register nonlinear upwind discretizations for buoyancy terms."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                self.add_nonlinear_discretization(
                    self.buoyancy_discretization(
                        gamma, delta, self.mdg.subdomains()
                    ).upwind(),
                )
                self.add_nonlinear_discretization(
                    self.buoyancy_discretization(
                        delta, gamma, self.mdg.subdomains()
                    ).upwind(),
                )
                # Coupling discretizations are separated components from the subdomain
                # ones.
                self.add_nonlinear_discretization(
                    self.interface_buoyancy_discretization(
                        gamma, delta, self.mdg.interfaces(codim=1)
                    ).upwind_primary(),
                )
                self.add_nonlinear_discretization(
                    self.interface_buoyancy_discretization(
                        gamma, delta, self.mdg.interfaces(codim=1)
                    ).upwind_secondary(),
                )
                self.add_nonlinear_discretization(
                    self.interface_buoyancy_discretization(
                        delta, gamma, self.mdg.interfaces(codim=1)
                    ).upwind_primary(),
                )
                self.add_nonlinear_discretization(
                    self.interface_buoyancy_discretization(
                        delta, gamma, self.mdg.interfaces(codim=1)
                    ).upwind_secondary(),
                )

    def update_buoyancy_driven_fluxes(self):
        """Update stored buoyancy flux arrays (subdomains and interfaces)."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs

                # Compute the values for all subdomains jointly, then distribute in a
                # for-loop. This is faster evaluation inside a loop over subdomains.
                subdomains = self.mdg.subdomains()

                # Use cached density operators
                rho_gamma_full = self._get_common_density(gamma, list(subdomains))
                rho_delta_full = self._get_common_density(delta, list(subdomains))
                subdomain_vals = self.equation_system.evaluate(
                    self.density_driven_flux(
                        subdomains, rho_gamma_full - rho_delta_full
                    )
                )
                # Offsets for the indices of individual subdomains.
                subdomain_offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])

                for id, (sd, data) in enumerate(self.mdg.subdomains(return_data=True)):
                    sd_offset = subdomain_offsets[id]
                    vals_loc = subdomain_vals[sd_offset : sd_offset + sd.num_faces]

                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {self.buoyant_flux_array_key(gamma, delta): +vals_loc}
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {self.buoyant_flux_array_key(delta, gamma): -vals_loc}
                    )

                # Same procedure for interfaces.
                interfaces = self.subdomains_to_interfaces(subdomains, [1])

                if len(interfaces) < 1:
                    # Shortcut for fracture-less domains.
                    continue

                interface_values = self.equation_system.evaluate(
                    self.interface_density_driven_flux(
                        interfaces, rho_gamma_full - rho_delta_full
                    )
                )
                interface_offsets = np.cumsum(
                    [0] + [intf.num_cells for intf in interfaces]
                )

                for id, (intf, data) in enumerate(
                    self.mdg.interfaces(return_data=True, codim=1)
                ):
                    intf_offset = interface_offsets[id]
                    vals_loc = interface_values[
                        intf_offset : intf_offset + intf.num_cells
                    ]

                    data[pp.PARAMETERS][self.buoyancy_intf_key(gamma, delta)].update(
                        {self.buoyant_intf_flux_array_key(gamma, delta): +vals_loc}
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(delta, gamma)].update(
                        {self.buoyant_intf_flux_array_key(delta, gamma): -vals_loc}
                    )


class ConstantViscosity(pp.PorePyModel):
    """Constant viscosity for a single-phase fluid."""

    def viscosity_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant viscosity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing representing the constant phase viscosity on some
            domains.

        """

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(self.fluid.reference_component.viscosity, "viscosity")

        return mu


class ConstantFluidThermalConductivity(pp.PorePyModel):
    """Ïmplementation of a constant thermal conductivity for a single-phase fluid."""

    def thermal_conductivity_of_phase(
        self, phase: pp.Phase
    ) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant thermal conductivity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing the constant phase conductivity on some domains.

        """

        def kappa(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(
                self.fluid.reference_component.thermal_conductivity,
                "fluid_thermal_conductivity",
            )

        return kappa

    def normal_thermal_conductivity(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Constant normal thermal conductivity of the fluid given by the fluid
        constants stored in the fluid's reference component.

        Using the fluid value corresponds to assuming a fluid-filled fracture.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        # NOTE this is not really a fluid-related const. law, it is more related to
        # mixed-dimensional problems.
        val = self.fluid.reference_component.normal_thermal_conductivity
        return Scalar(val, "normal_thermal_conductivity")


class FluidEnthalpyFromTemperature(pp.PorePyModel):
    """Implementation of a linearized fluid enthalpy :math:`c(T - T_{ref})` for a
    single-phase, single-component fluid.

    It uses the specific heat capacity of the fluid's reference component as :math:`c`,
    which is constant.

    """

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity  [J/kg/K]. The value
            is picked from the constants of the reference component.

        """
        return Scalar(
            self.fluid.reference_component.specific_heat_capacity,
            "fluid_specific_heat_capacity",
        )

    def specific_enthalpy_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a linear specific enthalpy for the fluid's phase.

        .. math::

            h = c \\left(T - T_0\\right)

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def h(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            c = self.fluid_specific_heat_capacity(cast(list[pp.Grid], domains))
            enthalpy = c * self.perturbation_from_reference(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h
