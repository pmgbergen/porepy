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

from typing import Callable, Sequence, cast, Union, Literal, List
import numpy as np
from itertools import combinations

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
    """Buoyancy terms and discretizations for multi-phase  multi-component flow."""

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
        """Key for subdomain buoyancy between phases gamma and delta."""
        return "buoyancy_" + gamma.name + "_" + delta.name

    def buoyant_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on subdomains."""
        return "buoyant_flux_" + gamma.name + "_" + delta.name

    def buoyancy_intf_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for interface buoyancy between phases gamma and delta."""
        return "buoyancy_intf_" + gamma.name + "_" + delta.name

    def buoyant_intf_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on interfaces."""
        return "buoyant_intf_flux_" + gamma.name + "_" + delta.name

    def buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Return upwind discretization for subdomain buoyancy term gamma↔delta."""
        discr = pp.ad.UpwindAd(self.buoyancy_key(gamma, delta), subdomains)
        assert isinstance(discr._discretization, pp.Upwind)
        discr._discretization.flux_array_key = self.buoyant_flux_array_key(gamma, delta)
        return discr

    def interface_buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Return upwind discretization for interface buoyancy term gamma-delta."""
        discr = pp.ad.UpwindCouplingAd(self.buoyancy_intf_key(gamma, delta), interfaces)
        assert isinstance(discr._discretization, pp.UpwindCoupling)
        discr._discretization.flux_array_key = self.buoyant_intf_flux_array_key(
            gamma, delta
        )
        return discr

    def fractionally_weighted_density(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Return sum f_j rho_j (fractional-flow-weighted density)."""
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
        """Return gravity magnitude field (pointing in negative last coord)."""
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def gravity_force(
        self,
        subdomains: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: Literal["fluid", "solid", "bulk"],
    ) -> pp.ad.Operator:
        """Return gravity force term (fluid only if buoyancy enabled)."""
        if material == "fluid" and self.params.get("enable_buoyancy_effects", True):
            # Narrow to list[pp.Grid] for calls needing subdomain grids
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
        """Compute flux induced by density_metric * g along gravity direction."""
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
        direction."""
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

    def __entity_buoyancy_flux(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Internal: Buoyancy flux induced by gamma and delta, advecting a quantity
        associated to phase gamma."""
        b_fluxes: List[pp.ad.Operator] = []
        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)
        w_flux_gamma_delta = self.density_driven_flux(domains, rho_gamma - rho_delta)
        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        discr_gamma = self.buoyancy_discretization(gamma, delta, domains)
        discr_delta = self.buoyancy_discretization(delta, gamma, domains)

        f_gamma_upwind: pp.ad.Operator = discr_gamma.upwind() @ (
            advected_gamma_quantity * f_gamma
        )  # well-defined fractional flow on facets
        f_delta_upwind: pp.ad.Operator = (
            discr_delta.upwind() @ f_delta
        )  # well-defined fractional flow on facets

        b_flux_gamma_delta = (f_gamma_upwind * f_delta_upwind) * w_flux_gamma_delta
        b_fluxes.append(b_flux_gamma_delta)

        interfaces = self.subdomains_to_interfaces(domains, [1])

        if len(interfaces) != 0:
            # Get interface flux contribution
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, rho_gamma - rho_delta
            )

            # Setup discretizations for interface coupling
            intf_discr_gamma = self.interface_buoyancy_discretization(
                gamma, delta, interfaces
            )
            intf_discr_delta = self.interface_buoyancy_discretization(
                delta, gamma, interfaces
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
                intf_discr_gamma.upwind_primary()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr_gamma.upwind_secondary()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr_delta.upwind_primary() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr_delta.upwind_secondary() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to primary grid
            interface_coupling_intf = (
                gamma_interface * delta_interface
            ) * intf_w_flux_gamma_delta
            # discr_gamma.bound_transport_neu() and discr_delta.bound_transport_neu()
            # are equal, discr_gamma.bound_transport_neu() is selected to operate
            b_intf_flux_gamma_delta = (
                discr_gamma.bound_transport_neu()
                @ mortar_projection.mortar_to_primary_int()
                @ interface_coupling_intf
            )
            b_fluxes.append(b_intf_flux_gamma_delta)
        return b_fluxes

    def __entity_buoyancy_jump(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Internal: Buoyancy flux jump induced by gamma and delta, advecting a quantity
        associated to phase gamma."""

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
            # Get interface flux contribution
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, rho_gamma - rho_delta
            )

            # Setup discretizations for interface coupling
            intf_discr_gamma = self.interface_buoyancy_discretization(
                gamma, delta, interfaces
            )
            intf_discr_delta = self.interface_buoyancy_discretization(
                delta, gamma, interfaces
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
                intf_discr_gamma.upwind_primary()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr_gamma.upwind_secondary()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr_delta.upwind_primary() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr_delta.upwind_secondary() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to secondary grid
            interface_coupling_intf = (
                gamma_interface * delta_interface
            ) * intf_w_flux_gamma_delta
            b_flux_jump_gamma_delta = (
                mortar_projection.mortar_to_secondary_int() @ interface_coupling_intf
            )
            b_flux_jumps.append(b_flux_jump_gamma_delta)
        return b_flux_jumps

    def phase_pairs_for(self, phase: pp.Phase) -> list[tuple[pp.Phase, pp.Phase]]:
        """Return ordered phase pairs (gamma, delta) including given phase first."""
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
        """Return buoyancy flux for a component."""
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
        """Return buoyancy flux for specific enthalpy."""
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
        """Return interface jump term for component buoyancy."""
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
        """Return interface jump term for enthalpy buoyancy."""
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
        """Initialize parameter containers and zero flux arrays for buoyancy."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                for sd, data in self.mdg.subdomains(return_data=True):
                    pp.initialize_data(sd, data, self.buoyancy_key(gamma, delta))
                    pp.initialize_data(sd, data, self.buoyancy_key(delta, gamma))
                    null_vals = np.zeros(sd.num_faces)
                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {self.buoyant_flux_array_key(gamma, delta): +null_vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {self.buoyant_flux_array_key(delta, gamma): -null_vals}
                    )
                for intf, data in self.mdg.interfaces(return_data=True):
                    null_vals = np.zeros(intf.num_cells)
                    pp.initialize_data(intf, data, self.buoyancy_intf_key(gamma, delta))
                    pp.initialize_data(intf, data, self.buoyancy_intf_key(delta, gamma))
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
                # coupling discretizations are separated components
                # from the subdomain ones
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
                for sd, data in self.mdg.subdomains(return_data=True):
                    rho_gamma = gamma.density([sd])
                    rho_delta = delta.density([sd])
                    vals = self.equation_system.evaluate(
                        self.density_driven_flux([sd], rho_gamma - rho_delta)
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {self.buoyant_flux_array_key(gamma, delta): +vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {self.buoyant_flux_array_key(delta, gamma): -vals}
                    )
                for intf, data in self.mdg.interfaces(return_data=True, codim=1):
                    subdomain_neighbors = self.interfaces_to_subdomains([intf])
                    rho_gamma = gamma.density(subdomain_neighbors)
                    rho_delta = delta.density(subdomain_neighbors)
                    vals = self.equation_system.evaluate(
                        self.interface_density_driven_flux(
                            [intf], rho_gamma - rho_delta
                        )
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(gamma, delta)].update(
                        {self.buoyant_intf_flux_array_key(gamma, delta): +vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(delta, gamma)].update(
                        {self.buoyant_intf_flux_array_key(delta, gamma): -vals}
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
