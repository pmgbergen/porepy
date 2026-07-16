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
from typing import TYPE_CHECKING, Callable, List, Literal, Sequence, Union, cast

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

if TYPE_CHECKING:
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

        The perturbation is measured relative to the thermodynamic state values.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dp = self.perturbation_from_thermodynamic_state("pressure", subdomains)

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

        The perturbation is measured relative to the thermodynamic state values.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dtemp = self.perturbation_from_thermodynamic_state("temperature", subdomains)
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

    @pp.ad.cached_method
    def _phase_mass_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Un-normalised phase MASS mobility :math:`m_j = \rho_j\,k_r(s_j)/\mu_j` (phase
        density times :meth:`phase_mobility`).

        The same product is the per-phase summand of :meth:`total_mass_mobility`, the
        numerator of :meth:`fractional_phase_mass_mobility`, and the buoyancy's simplicial
        ``l_gamma`` / ``l_delta``. Sharing this single cached object across all three lets the
        identity-keyed AD parser evaluate the common subtree once instead of rebuilding it in
        each. Bit-exact: same operands and operation.
        """
        m = phase.density(domains) * self.phase_mobility(phase, domains)
        m.set_name(f"phase_mass_mobility_{phase.name}")
        return m

    @pp.ad.cached_method
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
                self._phase_mass_mobility(phase, domains)
                for phase in self.fluid.phases
            ],
            name,
        )
        return mobility

    @pp.ad.cached_method
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

    @pp.ad.cached_method
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

    @pp.ad.cached_method
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

    @pp.ad.cached_method
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
        frac_mob = self._phase_mass_mobility(
            phase, domains
        ) / self.total_mass_mobility(domains)
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
    """

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

    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`FluidMobility`."""

    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.
    """

    darcy_flux_discretization: Callable[
        [list[pp.Grid]], pp.ad.MpfaAd
    ]  # because it contains the div(w(rho)) term
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.ConstantPermeability`."""

    interface_darcy_flux: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    _nonlinear_discretizations: list[pp.ad.MergedOperator]
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    def hybrid_upwind_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Discretization/parameter keyword for the pair's hybrid upwind."""
        return "hybrid_upwind_" + gamma.name + "_" + delta.name

    @pp.ad.cached_method
    def hybrid_upwind_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.HUpwindAd:
        """Return the two-direction hybrid-upwind discretization for the pair.

        A single :class:`HUpwind` whose two stored directions ``"hybrid_gamma_flux"`` /
        ``"hybrid_delta_flux"`` yield ``upwind_gamma`` and ``upwind_delta`` respectively.
        The model fills the two directions per scheme (inter-phase gravity flux ``+/-`` for
        hybrid; the per-phase potential fluxes for phase-potential upwinding).

        Parameters:
            gamma: The first (reference) phase.
            delta: The second phase.
            subdomains: The subdomains to consider.

        Returns:
            The hybrid-upwind AD discretization.

        """
        return pp.ad.HUpwindAd(self.hybrid_upwind_key(gamma, delta), subdomains)

    def hybrid_interface_upwind_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.HUpwindCouplingAd:
        """Return the two-direction hybrid interface (mortar) upwind for the pair.

        A single :class:`HUpwindCoupling` whose two stored interface directions
        (``hybrid_gamma_flux`` / ``hybrid_delta_flux``) yield the two phases' mortar upwind
        matrices.
        """
        return pp.ad.HUpwindCouplingAd(self.hybrid_upwind_key(gamma, delta), interfaces)

    def interface_potential_driven_flux(
        self, interfaces: list[pp.MortarGrid], phase: pp.Phase
    ) -> pp.ad.Operator:
        """Per-phase interface potential flux (mortar) for the PPU upwind direction.

        Mirrors :meth:`potential_driven_flux` on the interface: the interface (mortar)
        Darcy flux plus the phase's interface density-driven flux, so its sign upwinds
        that phase across the interface.
        """
        subdomains = self.mdg.subdomains()
        rho_mixture = self.fractionally_weighted_density(subdomains)
        rho_phase = phase.density(subdomains)
        flux = self.interface_darcy_flux(
            interfaces
        ) + self.interface_density_driven_flux(interfaces, rho_phase - rho_mixture)
        flux.set_name("interface_potential_driven_flux_" + phase.name)
        return flux

    @pp.ad.cached_method
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
        overall_rho = pp.ad.sum_operator_list(
            [
                self.fractional_phase_mass_mobility(phase, domains)
                * phase.density(domains)
                for phase in self.fluid.phases
            ]
        )
        overall_rho.set_name("fractionally_weighted_density")
        return overall_rho

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Return gravity magnitude.

        Parameters:
            subdomains: The subdomains to consider for the gravity field computation.
                Not used, but included for consistency.

        Returns:
            An operator representing the gravity field.

        """
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        gravity_field = pp.ad.Scalar(val)
        gravity_field.set_name("gravity_field")
        return gravity_field

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

            fractionally_weighted_rho = self.fractionally_weighted_density(
                subdomains_list
            )
            e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
            overall_gravity_flux = (
                pp.ad.Scalar(-1)
                * e_n
                @ (fractionally_weighted_rho * self.gravity_field(subdomains_list))
            )
            overall_gravity_flux.set_name("overall gravity flux")
            return overall_gravity_flux
        else:
            return super().gravity_force(subdomains, material)  # type:ignore

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

        e_n = self.e_i(subdomains_list, i=self.nd - 1, dim=self.nd)
        gravity_flux = (
            pp.ad.Scalar(-1)
            * e_n
            @ (density_metric * self.gravity_field(subdomains_list))
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains_list
        )

        w_flux = discr.vector_source() @ gravity_flux
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
        normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )

        e_n = self.e_i(subdomain_neighbors, i=self.nd - 1, dim=self.nd)
        gravity_flux = (
            pp.ad.Scalar(-1)
            * e_n
            @ (density_metric * self.gravity_field(subdomain_neighbors))
        )

        intf_vector_source = projection.secondary_to_mortar_avg() @ gravity_flux

        normals_times_source = normals * intf_vector_source
        nd_to_scalar_sum = pp.ad.sum_projection_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]
        )
        w_flux = self.volume_integral(
            self.normal_permeability(interfaces)
            * (nd_to_scalar_sum @ normals_times_source),
            interfaces,
            1,
        )
        w_flux.set_name("interface_density_driven_flux_" + density_metric.name)
        return w_flux

    #: Valid buoyancy upwinding schemes.
    #: ``"phase_potential"`` selects phase-potential upwinding (PPU): the full phase
    #: flux (viscous and gravitational) is upwinded by that phase's own potential.
    #: ``"hybrid"`` selects hybrid upwinding (HU): the viscous part is upwinded by the
    #: total Darcy flux and the buoyancy part by the inter-phase gravity flux.
    _valid_buoyancy_upwinding_schemes: tuple[str, str] = ("phase_potential", "hybrid")

    def buoyancy_upwinding_scheme(self) -> str:
        """Return the selected buoyancy upwinding scheme.

        Controlled by the model parameter ``"buoyancy_upwinding"``. Defaults to
        ``"phase_potential"`` (PPU).

        Raises:
            ValueError: If the parameter holds an unsupported scheme name.

        Returns:
            Either ``"phase_potential"`` or ``"hybrid"``.

        """
        scheme = self.params.get("buoyancy_upwinding", "hybrid")
        if scheme not in self._valid_buoyancy_upwinding_schemes:
            raise ValueError(
                f"Unknown buoyancy_upwinding scheme {scheme!r}. "
                f"Choose one of {self._valid_buoyancy_upwinding_schemes}."
            )
        return scheme

    def is_phase_potential_upwinding(self) -> bool:
        """Whether the phase-potential (PPU) buoyancy scheme is active.

        Returns:
            True for PPU, False for hybrid upwinding (HU).

        """
        return self.buoyancy_upwinding_scheme() == "phase_potential"

    def potential_driven_flux(
        self, subdomains: pp.SubdomainsOrBoundaries, phase: pp.Phase
    ) -> pp.ad.Operator:

        rho_mixture = self.fractionally_weighted_density(subdomains)
        rho_phase = phase.density(subdomains)
        # Proper PPU direction: sign of the phase-potential flux
        # -K (grad p - rho_phase g). Built from the total (mixture) potential flux as
        # darcy_flux + density_driven_flux(rho_phase - rho_mixture). No explicit total
        # mobility factor: darcy_flux and density_driven_flux share the same Darcy
        # tensor, so lambda_T is handled consistently (inside the tensor when
        # mass-mobility-weighted, absent otherwise) and cannot change the upwind sign.
        m_star = self.darcy_flux(subdomains) + self.density_driven_flux(
            subdomains, rho_phase - rho_mixture
        )
        m_star.set_name("potential_driven_flux_" + phase.name)
        return m_star

    def passive_phase_interference_factor(self) -> pp.ad.Scalar:
        r"""Spatial localization factor :math:`\chi \in [0, 1]` for passive-phase
        interference in the simplicial hybrid-upwind buoyancy term.

        For a phase pair (:math:`\gamma`, :math:`\delta`), the *background* mass mobility

        .. math::

            \lambda_{bg} = \lambda_{tot} - \lambda_\gamma - \lambda_\delta

        gathers the contribution of every phase other than the active pair (the passive
        phases). :math:`\chi` distributes that background mobility between the two
        inter-phase hybrid-upwind directions: a fraction :math:`\chi` is upwinded along
        the :math:`\gamma` direction and the complement :math:`1 - \chi` along the
        :math:`\delta` direction. The symmetric default :math:`\chi = 0.5` weights the
        two passive directions equally.

        Configurable through ``params["passive_phase_interference_factor"]`` (default
        ``0.5``).

        Returns:
            The localization factor wrapped as an Ad scalar.

        Raises:
            ValueError: If the configured value lies outside :math:`[0, 1]`.

        """
        chi = self.params.get("passive_phase_interference_factor", 0.5)
        if not 0.0 <= chi <= 1.0:
            raise ValueError(
                "passive_phase_interference_factor (chi) must lie in [0, 1]; "
                f"got {chi}."
            )
        return pp.ad.Scalar(chi)

    def __interface_lambda_upwind(
        self,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        intf_discr: pp.ad.HUpwindCouplingAd,
        mortar_avg: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
    ) -> pp.ad.Operator:
        r"""Total mass mobility upwinded onto the interface for the (gamma, delta) pair.

        Mirrors the simplicial cell-wise treatment in :meth:`__entity_buoyancy_flux`: the
        active-pair mobilities :math:`\lambda_\gamma`, :math:`\lambda_\delta` are upwinded
        along their own inter-phase directions, and the passive-phase background mobility
        :math:`\lambda_{bg} = \lambda_{tot} - \lambda_\gamma - \lambda_\delta` is split
        between the two directions by the localization factor :math:`\chi` (see
        :meth:`passive_phase_interference_factor`).

        Both the interface buoyancy *flux* (projected to the primary) and the interface
        buoyancy *jump* (projected to the secondary) build the mortar coupling from this
        single expression, so the discretization stays locally conservative across the
        interface.

        Parameters:
            gamma: The first (reference) phase of the pair.
            delta: The second phase of the pair.
            domains: The subdomains on which the cell-wise mobilities are evaluated.
            intf_discr: The pair's hybrid interface (mortar) upwind discretization.
            mortar_avg: Primary-to-mortar average projection.
            primary_trace: Trace operator onto the primary boundary.
            secondary_to_mortar: Secondary-to-mortar average projection.

        Returns:
            The interface-upwinded total mass mobility :math:`\lambda` on mortar cells.

        """

        def to_interface(
            primary_dir: Callable[[], pp.ad.MergedOperator],
            secondary_dir: Callable[[], pp.ad.MergedOperator],
            quantity: pp.ad.Operator,
        ) -> pp.ad.Operator:
            # Upwind a cell quantity onto the mortar from both the primary side (via the
            # trace) and the secondary side, using the given inter-phase direction.
            return (
                primary_dir() @ mortar_avg @ primary_trace @ quantity
                + secondary_dir() @ secondary_to_mortar @ quantity
            )

        l_gamma = gamma.density(domains) * self.phase_mobility(gamma, domains)
        l_delta = delta.density(domains) * self.phase_mobility(delta, domains)
        l_background = self.total_mass_mobility(domains) - l_gamma - l_delta

        l_gamma_interface = to_interface(
            intf_discr.upwind_primary_gamma, intf_discr.upwind_secondary_gamma, l_gamma
        )
        l_delta_interface = to_interface(
            intf_discr.upwind_primary_delta, intf_discr.upwind_secondary_delta, l_delta
        )
        # Passive-phase background mobility, localized between the two inter-phase
        # directions by the factor chi (gamma) / 1 - chi (delta).
        l_background_gamma_interface = to_interface(
            intf_discr.upwind_primary_gamma,
            intf_discr.upwind_secondary_gamma,
            l_background,
        )
        l_background_delta_interface = to_interface(
            intf_discr.upwind_primary_delta,
            intf_discr.upwind_secondary_delta,
            l_background,
        )
        xi = self.passive_phase_interference_factor()
        xi_complement = pp.ad.Scalar(1.0) - xi
        l_background_interface = (
            xi * l_background_gamma_interface
            + xi_complement * l_background_delta_interface
        )
        return l_gamma_interface + l_delta_interface + l_background_interface + pp.ad.Scalar(1.0e-15)

    def __interface_mp_coupling(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: list[pp.Grid],
        intf_discr: pp.ad.HUpwindCouplingAd,
        mortar_avg: pp.ad.Operator,
        primary_trace: pp.ad.Operator,
        secondary_to_mortar: pp.ad.Operator,
        intf_w_flux_gamma_delta: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """HU-BM(mp) interface buoyancy coupling for the pair ``(gamma, delta)``.

        Mobility-product form ``lambda_gamma lambda_delta / lambda_T`` on the mortar: the
        advected gamma quantity is upwinded with the FULL gamma mass mobility, the delta full
        mass mobility the other way, and both are normalised by the simplicial interface total
        mobility (:meth:`__interface_lambda_upwind`). The SAME expression feeds the
        primary-projected flux (:meth:`__entity_buoyancy_flux`) and the secondary-projected
        jump (:meth:`__entity_buoyancy_jump`), so the mortar coupling is locally conservative.
        """
        l_gamma = self._phase_mass_mobility(gamma, domains)
        l_delta = self._phase_mass_mobility(delta, domains)
        gamma_interface = (
            intf_discr.upwind_primary_gamma()
            @ mortar_avg
            @ primary_trace
            @ (advected_gamma_quantity * l_gamma)
            + intf_discr.upwind_secondary_gamma()
            @ secondary_to_mortar
            @ (advected_gamma_quantity * l_gamma)
        )
        delta_interface = (
            intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ l_delta
            + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ l_delta
        )
        lambda_interface_upwind = self.__interface_lambda_upwind(
            gamma,
            delta,
            domains,
            intf_discr,
            mortar_avg,
            primary_trace,
            secondary_to_mortar,
        )
        return (
            gamma_interface * delta_interface / lambda_interface_upwind
        ) * intf_w_flux_gamma_delta

    @pp.ad.cached_method
    def _entity_buoyancy_pair_common(
        self, gamma: pp.Phase, delta: pp.Phase, domains: list[pp.Grid]
    ) -> tuple:
        """Advected-INDEPENDENT pieces of the subdomain buoyancy flux for the ordered pair
        ``(gamma, delta)``: the density-metric Darcy flux ``w_flux_gamma_delta``, the
        delta-side upwinded fractional mobility ``f_delta_upwind`` and -- in the
        non-fractional-flow (mobility-product) branch -- the simplicial inter-phase total
        mobility ``lambda_upwind`` and the delta-side upwinded FULL mass mobility
        ``l_delta_upwind`` (both ``None`` in the fractional-flow branch).

        Cached per ``(gamma, delta, domains)`` so these are built ONCE per ordered pair
        instead of once per advected quantity (each component + enthalpy). Since the AD parser
        keys on object identity, sharing these objects removes the duplicate subtrees it would
        otherwise re-evaluate. Bit-exact: same operations and operands, only fewer distinct
        objects.
        """
        from porepy.models.compositional_flow import (
            is_fractional_flow,
        )

        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)
        density_metric = rho_gamma - rho_delta
        w_flux_gamma_delta = self.density_driven_flux(domains, density_metric)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)
        discr = self.hybrid_upwind_discretization(gamma, delta, domains)
        f_delta_upwind = discr.upwind_delta() @ f_delta

        if is_fractional_flow(self):
            lambda_upwind = None
            l_delta_upwind = None
        else:
            l_gamma = self._phase_mass_mobility(gamma, domains)
            l_delta = self._phase_mass_mobility(delta, domains)
            l_background = self.total_mass_mobility(domains) - l_gamma - l_delta
            xi = self.passive_phase_interference_factor()
            xi_complement = pp.ad.Scalar(1.0) - xi
            l_background_upwind = xi * discr.upwind_gamma() @ (
                l_background
            ) + xi_complement * discr.upwind_delta() @ (l_background)
            l_gamma_upwind = discr.upwind_gamma() @ (l_gamma)
            l_delta_upwind = discr.upwind_delta() @ (l_delta)
            lambda_upwind = l_gamma_upwind + l_delta_upwind + l_background_upwind + pp.ad.Scalar(1.0e-15)
        return w_flux_gamma_delta, f_delta_upwind, lambda_upwind, l_delta_upwind

    @pp.ad.cached_method
    def __entity_buoyancy_flux(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Helper function to compute the buoyancy flux on subdomains induced by gamma
        and delta, advecting a quantity associated to phase gamma.

        Parameters:
            advected_gamma_quantity: The quantity advected in phase gamma.
            gamma: The phase gamma.
            delta: The phase delta.
            domains: The domains over which the flux is computed.

        Raises:
            ValueError: If the domains are not subdomains.

        Returns:
            A list of Ad operators representing the buoyancy flux is phase gamma.

        """
        from porepy.models.compositional_flow import (
            is_fractional_flow,
        )

        b_fluxes: List[pp.ad.Operator] = []

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)
        # One hybrid-upwind discretization for the pair: the two phases upwind along the
        # single inter-phase gravity direction, gamma one way (upwind_gamma) and delta the
        # other (upwind_delta), instead of two separate Upwind objects.
        discr = self.hybrid_upwind_discretization(gamma, delta, domains)

        # Advected-INDEPENDENT pieces (density-metric Darcy flux, delta-side upwinded
        # mobility, simplicial inter-phase mobility): built ONCE per ordered pair via the
        # cached helper and shared across this pair's advected quantities (each component +
        # enthalpy), instead of rebuilt on every call.
        w_flux_gamma_delta, f_delta_upwind, lambda_upwind, l_delta_upwind = (
            self._entity_buoyancy_pair_common(gamma, delta, domains)
        )

        f_gamma_upwind: pp.ad.Operator = discr.upwind_gamma() @ (
            advected_gamma_quantity * f_gamma
        )  # well-defined fractional flow on facets.

        if is_fractional_flow(self):
            b_flux_gamma_delta = (f_gamma_upwind * f_delta_upwind) * w_flux_gamma_delta
        else:
            # HU-BM(mp): mobility-product buoyancy  lambda_gamma lambda_delta / lambda_T
            # (classical Lee/Hamon U^HU) instead of the fractional-flow  f_gamma f_delta
            # lambda_T. The advected gamma quantity rides the gamma phase flux, so it is
            # upwinded together with the FULL gamma mass mobility; the delta mass mobility is
            # upwinded the other way; both are normalised by the simplicial face total mobility
            # lambda_upwind (active pair + chi-split passive background).
            l_gamma = self._phase_mass_mobility(gamma, domains)
            l_gamma_advected_upwind = discr.upwind_gamma() @ (
                advected_gamma_quantity * l_gamma
            )
            b_flux_gamma_delta = (
                l_gamma_advected_upwind * l_delta_upwind / lambda_upwind
            ) * w_flux_gamma_delta

        b_fluxes.append(b_flux_gamma_delta)

        interfaces = self.subdomains_to_interfaces(domains, [1])

        if len(interfaces) != 0:
            # Get interface flux contribution.
            rho_gamma = gamma.density(domains)
            rho_delta = delta.density(domains)
            intf_density_metric = rho_gamma - rho_delta
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, intf_density_metric
            )

            # One hybrid interface discretization: gamma/delta from one gravity direction.
            intf_discr = self.hybrid_interface_upwind_discretization(
                gamma, delta, interfaces
            )

            # Projections and trace operators
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, domains, interfaces, dim=1
            )
            trace = pp.ad.Trace(domains)

            # Modified coupling that properly handles dimensional transition
            primary_trace = trace.trace
            mortar_avg = mortar_projection.primary_to_mortar_avg()
            secondary_to_mortar = mortar_projection.secondary_to_mortar_avg()

            # Project quantities to interface with proper upwinding for both
            # primary and secondary sides
            gamma_interface = (
                intf_discr.upwind_primary_gamma()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr.upwind_secondary_gamma()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to primary grid
            if is_fractional_flow(self):
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * intf_w_flux_gamma_delta
            else:
                # HU-BM(mp): mobility-product interface coupling (full mobilities normalised by
                # the simplicial interface total mobility). The SAME helper feeds the
                # secondary-projected jump, so the mortar coupling is locally conservative.
                interface_coupling_intf = self.__interface_mp_coupling(
                    advected_gamma_quantity,
                    gamma,
                    delta,
                    domains,
                    intf_discr,
                    mortar_avg,
                    primary_trace,
                    secondary_to_mortar,
                    intf_w_flux_gamma_delta,
                )
            # The gamma/delta boundary-transport matrices are equal; use the gamma one.
            b_intf_flux_gamma_delta = (
                discr.bound_transport_neu_gamma()
                @ mortar_projection.mortar_to_primary_int()
                @ interface_coupling_intf
            )
            b_fluxes.append(b_intf_flux_gamma_delta)
        return b_fluxes

    @pp.ad.cached_method
    def __entity_buoyancy_jump(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Helper function to compute the buoyancy flux jump on interfaces induced by
        gamma and delta, advecting a quantity associated to phase gamma.

        Parameters:
            advected_gamma_quantity: The quantity advected in phase gamma.
            gamma: The phase gamma.
            delta: The phase delta.
            domains: The domains over which the flux is computed.

        Raises:
            ValueError: If the domains are not subdomains.

        Returns:
            A list of Ad operators representing the buoyancy flux is phase gamma.

        """

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        b_flux_jumps: List[pp.ad.Operator] = []
        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="component_buoyancy_jump_zero"
        )
        b_flux_jumps.append(zero)

        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)

        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)
        interfaces = self.subdomains_to_interfaces(domains, [1])
        if len(interfaces) != 0:
            intf_density_metric = rho_gamma - rho_delta
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, intf_density_metric
            )

            # One hybrid interface discretization: gamma/delta from one gravity direction.
            intf_discr = self.hybrid_interface_upwind_discretization(
                gamma, delta, interfaces
            )

            # Projections and trace operators
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, domains, interfaces, dim=1
            )
            trace = pp.ad.Trace(domains)

            # Modified coupling that properly handles dimensional transition
            primary_trace = trace.trace
            mortar_avg = mortar_projection.primary_to_mortar_avg()
            secondary_to_mortar = mortar_projection.secondary_to_mortar_avg()

            # Project quantities to interface with proper upwinding for both
            # primary and secondary sides
            gamma_interface = (
                intf_discr.upwind_primary_gamma()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr.upwind_secondary_gamma()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to secondary grid
            from porepy.models.compositional_flow import (
                is_fractional_flow,
            )

            if is_fractional_flow(self):
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * intf_w_flux_gamma_delta
            else:
                # HU-BM(mp): mobility-product interface coupling (full mobilities normalised by
                # the simplicial interface total mobility). The SAME helper feeds the
                # primary-projected flux, so the mortar coupling is locally conservative.
                interface_coupling_intf = self.__interface_mp_coupling(
                    advected_gamma_quantity,
                    gamma,
                    delta,
                    domains,
                    intf_discr,
                    mortar_avg,
                    primary_trace,
                    secondary_to_mortar,
                    intf_w_flux_gamma_delta,
                )
            b_flux_jump_gamma_delta = (
                mortar_projection.mortar_to_secondary_int() @ interface_coupling_intf
            )
            b_flux_jumps.append(b_flux_jump_gamma_delta)
        return b_flux_jumps

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

    def component_buoyancy(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the buoyancy flux for a given component.

        Parameters:
            component_xi: The component for which to get the buoyancy flux.
            domains: The domains to consider for the buoyancy flux calculation.

        Returns:
            Ad operator representing the buoyancy flux for the component.

        """
        b_fluxes: List[pp.ad.Operator] = []
        b_fluxes.append(self.density_driven_flux(domains, pp.ad.Scalar(0.0)))
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                chi_xi_gamma = gamma.partial_fraction_of[component_xi](domains)
                b_fluxes += self.__entity_buoyancy_flux(
                    chi_xi_gamma, gamma, delta, domains
                )
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("component_buoyancy_" + component_xi.name)
        return b_flux

    def enthalpy_buoyancy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Get the buoyancy flux for specific enthalpy.

        Both schemes use the same simplicial inter-phase buoyancy term
        (:meth:`__entity_buoyancy_flux`); they differ only in the two upwind directions
        stored for the :class:`HUpwind` discretization (hybrid: inter-phase gravity flux
        with opposite signs; phase-potential: each phase's own potential flux).

        Parameters:
            domains: The domains to consider for the buoyancy flux calculation.

        Returns:
            Ad operator representing the buoyancy flux for the specific enthalpy.

        """
        b_fluxes: List[pp.ad.Operator] = []
        b_fluxes.append(self.density_driven_flux(domains, pp.ad.Scalar(0.0)))
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_flux(h_gamma, gamma, delta, domains)
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy")
        return b_flux

    def component_buoyancy_jump(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for component buoyancy.

        Parameters:
            component_xi: The component for which to get the interface jump term.
            domains: The domains to consider for the interface jump term calculation.

        Returns:
            Ad operator representing the interface jump term for the component.

        """
        b_fluxes: List[pp.ad.Operator] = []

        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="component_buoyancy_jump_zero"
        )
        b_fluxes.append(zero)
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                chi_xi_gamma = gamma.partial_fraction_of[component_xi](domains)
                b_fluxes += self.__entity_buoyancy_jump(
                    chi_xi_gamma, gamma, delta, domains
                )
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("component_buoyancy_jump_" + component_xi.name)
        return b_flux

    def enthalpy_buoyancy_jump(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for enthalpy buoyancy.

        Parameters:
            domains: The domains to consider for the interface jump term calculation.

        Returns:
            Ad operator representing the interface jump term for the enthalpy.

        """
        b_fluxes: List[pp.ad.Operator] = []
        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="enthalpy_buoyancy_jump_zero"
        )
        b_fluxes.append(zero)
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_jump(h_gamma, gamma, delta, domains)
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy_jump")
        return b_flux

    def set_buoyancy_discretization_parameters(self):
        """Initialize parameter containers and zero direction arrays for buoyancy.

        Each pair's hybrid upwind keyword holds the *two* direction arrays
        ``hybrid_gamma_flux`` / ``hybrid_delta_flux`` (filled per scheme in
        :meth:`update_buoyancy_driven_fluxes`).
        """
        for phase_gamma in self.fluid.phases:
            for gamma, delta in self.phase_pairs_for(phase_gamma):
                key = self.hybrid_upwind_key(gamma, delta)
                for sd, data in self.mdg.subdomains(return_data=True):
                    null_vals = np.zeros(sd.num_faces)
                    # Same advective-flux boundary classification as the mobility
                    # discretization, so the upwind/bound_transport matrices partition the
                    # boundary faces consistently with the advective flux.
                    pp.initialize_data(data, key)
                    data[pp.PARAMETERS][key].update(
                        {
                            "hybrid_gamma_flux": +null_vals,
                            "hybrid_delta_flux": +null_vals,
                            "bc": self.bc_type_fluid_flux(sd),
                        }
                    )
                for intf, data in self.mdg.interfaces(return_data=True):
                    null_vals = np.zeros(intf.num_cells)
                    pp.initialize_data(data, key)
                    data[pp.PARAMETERS][key].update(
                        {
                            "hybrid_gamma_flux": +null_vals,
                            "hybrid_delta_flux": +null_vals,
                        }
                    )

    def _register_hybrid_nonlinear_discretization(
        self, op: pp.ad.MergedOperator
    ) -> None:
        """Register a hybrid upwind operator for per-iteration re-discretization.

        Appends directly to the nonlinear-discretization list, bypassing
        :meth:`add_nonlinear_discretization`'s exact-type guardrail. That guardrail admits
        only the base ``Upwind`` / ``UpwindCoupling`` types; ``HUpwind`` /
        ``HUpwindCoupling`` are functionally valid subclasses, so this keeps the change
        contained to this module instead of relaxing the core check.
        """
        if op not in self._nonlinear_discretizations:
            self._nonlinear_discretizations.append(op)

    def set_nonlinear_buoyancy_discretization(self):
        """Register nonlinear upwind discretizations for buoyancy terms."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                # Hybrid upwind: one discretization per pair, two matrices
                # (re-discretized each iteration as the gravity direction is refreshed).
                hybrid_discr = self.hybrid_upwind_discretization(
                    gamma, delta, self.mdg.subdomains()
                )
                self._register_hybrid_nonlinear_discretization(
                    hybrid_discr.upwind_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    hybrid_discr.upwind_delta()
                )
                # One hybrid interface (mortar) discretization per pair, four matrices.
                intf_discr = self.hybrid_interface_upwind_discretization(
                    gamma, delta, self.mdg.interfaces(codim=1)
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_primary_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_secondary_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_primary_delta()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_secondary_delta()
                )

    def update_buoyancy_driven_fluxes(self):
        """Evaluate and store the two upwind directions of the buoyancy discretizations.

        For each ordered phase pair ``(gamma, delta)`` the :class:`HUpwind` /
        :class:`HUpwindCoupling` discretizations read two direction arrays:

        - hybrid upwinding (HU): the inter-phase gravity flux with opposite signs
          (``+ddf(rho_gamma - rho_delta)`` for gamma, ``-ddf(...)`` for delta);
        - phase-potential upwinding (PPU), guarded by
          :meth:`is_phase_potential_upwinding`: each phase's own potential flux
          (``Psi_gamma`` for gamma, ``Psi_delta`` for delta).

        When this is called (every Newton iteration, or only once per time step) is a
        *model* concern: see ``FlowModelBase.refresh_buoyancy_direction`` /
        ``before_time_step`` and the ``"lag_buoyancy_direction"`` parameter.
        """
        phase_potential = self.is_phase_potential_upwinding()
        subdomains = self.mdg.subdomains()
        subdomain_offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        interface_offsets = np.cumsum([0] + [intf.num_cells for intf in interfaces])

        for phase_gamma in self.fluid.phases:
            for gamma, delta in self.phase_pairs_for(phase_gamma):
                key = self.hybrid_upwind_key(gamma, delta)

                # Two subdomain directions per pair (evaluated jointly, then distributed).
                if phase_potential:
                    gamma_vals = self.equation_system.evaluate(
                        self.potential_driven_flux(subdomains, gamma)
                    )
                    delta_vals = self.equation_system.evaluate(
                        self.potential_driven_flux(subdomains, delta)
                    )
                else:
                    gravity_vals = self.equation_system.evaluate(
                        self.density_driven_flux(
                            subdomains,
                            gamma.density(subdomains) - delta.density(subdomains),
                        )
                    )
                    gamma_vals = +gravity_vals
                    delta_vals = -gravity_vals

                for k, (sd, data) in enumerate(self.mdg.subdomains(return_data=True)):
                    sl = slice(
                        subdomain_offsets[k], subdomain_offsets[k] + sd.num_faces
                    )
                    data[pp.PARAMETERS][key].update(
                        {
                            "hybrid_gamma_flux": gamma_vals[sl],
                            "hybrid_delta_flux": delta_vals[sl],
                        }
                    )

                if len(interfaces) < 1:
                    # Shortcut for fracture-less domains.
                    continue

                # Two interface directions per pair.
                if phase_potential:
                    gamma_intf = self.equation_system.evaluate(
                        self.interface_potential_driven_flux(interfaces, gamma)
                    )
                    delta_intf = self.equation_system.evaluate(
                        self.interface_potential_driven_flux(interfaces, delta)
                    )
                else:
                    grav_intf = self.equation_system.evaluate(
                        self.interface_density_driven_flux(
                            interfaces,
                            gamma.density(subdomains) - delta.density(subdomains),
                        )
                    )
                    gamma_intf = +grav_intf
                    delta_intf = -grav_intf

                for k, (intf, data) in enumerate(
                    self.mdg.interfaces(return_data=True, codim=1)
                ):
                    sl = slice(
                        interface_offsets[k], interface_offsets[k] + intf.num_cells
                    )
                    data[pp.PARAMETERS][key].update(
                        {
                            "hybrid_gamma_flux": gamma_intf[sl],
                            "hybrid_delta_flux": delta_intf[sl],
                        }
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
            enthalpy = c * self.perturbation_from_thermodynamic_state(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h
