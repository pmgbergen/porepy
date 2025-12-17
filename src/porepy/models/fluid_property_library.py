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

    Refactored to optimize the AD graph size while preserving the exact
    stabilization logic of the original code and ensuring safety for
    state-dependent discretizations.
    """

    component_mass_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    fractional_phase_mass_mobility: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    phase_mobility: Callable[[pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator]
    darcy_flux_discretization: Callable[
        [list[pp.Grid]], pp.ad.MpfaAd
    ]
    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]

    def buoyancy_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        return "buoyancy_" + gamma.name + "_" + delta.name

    def buoyant_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        return "buoyant_flux_" + gamma.name + "_" + delta.name

    def buoyancy_intf_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        return "buoyancy_intf_" + gamma.name + "_" + delta.name

    def buoyant_intf_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        return "buoyant_intf_flux_" + gamma.name + "_" + delta.name

    # -------------------------------------------------------------------------
    # Cached Static Geometry (SAFE to cache, high optimization value)
    # -------------------------------------------------------------------------

    @pp.ad.cached_method
    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Cached gravity scalar."""
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        gravity_field = pp.ad.Scalar(val)
        gravity_field.set_name("gravity_field")
        return gravity_field

    @pp.ad.cached_method
    def _subdomain_gravity_geometry(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Optimized Fusion: div_source @ (-1 * e_n)
        Maps (Scalar Density * g) -> (Face Flux) independent of state.
        """
        e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
        discr = self.darcy_flux_discretization(subdomains)

        # Fuse the negation and the unit vector
        neg_e_n = (pp.ad.Scalar(-1) * e_n).simplify(self.mdg)

        # Fuse the discretization source map with the geometry
        return (discr.vector_source() @ neg_e_n).simplify(self.mdg)

    @pp.ad.cached_method
    def _interface_gravity_geometry(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """
        Optimized Fusion: Proj_Sec_to_Mortar @ (-1 * e_n)
        """
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )
        e_n = self.e_i(subdomain_neighbors, i=self.nd - 1, dim=self.nd)
        neg_e_n = (pp.ad.Scalar(-1) * e_n).simplify(self.mdg)
        return (projection.secondary_to_mortar_avg() @ neg_e_n).simplify(self.mdg)

    @pp.ad.cached_method
    def _interface_projections(self, domains: list[pp.Grid]):
        """
        Helper to retrieve projection operators ONLY if interfaces exist.
        Returns None if no fractures, preventing crashes.
        """
        interfaces = self.subdomains_to_interfaces(domains, [1])
        if not interfaces:
            return None

        mortar_projection = pp.ad.MortarProjections(
            self.mdg, domains, interfaces, dim=1
        )
        trace = pp.ad.Trace(domains)

        return (
            interfaces,
            mortar_projection,
            trace,
            mortar_projection.primary_to_mortar_avg(),
            mortar_projection.secondary_to_mortar_avg(),
            trace.trace,
            mortar_projection.mortar_to_primary_int(),
            mortar_projection.mortar_to_secondary_int()
        )

    # -------------------------------------------------------------------------
    # Discretizations (NOT CACHED due to state dependency)
    # -------------------------------------------------------------------------

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
        if not interfaces:
            raise ValueError("Interfaces list cannot be empty.")
        discr = pp.ad.UpwindCouplingAd(self.buoyancy_intf_key(gamma, delta), interfaces)
        assert isinstance(discr._discretization, pp.UpwindCoupling)
        discr._discretization.flux_array_key = self.buoyant_intf_flux_array_key(
            gamma, delta
        )
        return discr

    # -------------------------------------------------------------------------
    # Helper Logic (Dynamic)
    # -------------------------------------------------------------------------

    def fractionally_weighted_density(
            self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Compute sum_j f_j rho_j."""
        overall_rho = pp.ad.sum_operator_list(
            [
                self.fractional_phase_mass_mobility(phase, domains)
                * phase.density(domains)
                for phase in self.fluid.phases
            ]
        )
        overall_rho.set_name("fractionally_weighted_density")
        return overall_rho.simplify(self.mdg)

    def gravity_force(
            self,
            subdomains: Union[list[pp.Grid], list[pp.MortarGrid]],
            material: Literal["fluid", "solid", "bulk"],
    ) -> pp.ad.Operator:
        if material == "fluid" and self.params.get("enable_buoyancy_effects", True):
            if not all(isinstance(g, pp.Grid) for g in subdomains):
                raise TypeError("gravity_force expects only subdomain grids.")
            subdomains_list = cast(list[pp.Grid], list(subdomains))

            fractionally_weighted_rho = self.fractionally_weighted_density(
                subdomains_list
            )
            # Use cached geometric fusion
            e_n = self.e_i(subdomains_list, i=self.nd - 1, dim=self.nd)
            neg_e_n = (pp.ad.Scalar(-1) * e_n).simplify(self.mdg)

            overall_gravity_flux = neg_e_n @ (
                    fractionally_weighted_rho * self.gravity_field(subdomains_list)
            )
            overall_gravity_flux.set_name("overall gravity flux")
            return overall_gravity_flux.simplify(self.mdg)
        else:
            return super().gravity_force(subdomains, material)  # type:ignore

    def density_driven_flux(
            self, subdomains: pp.SubdomainsOrBoundaries, density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        if not all(isinstance(g, pp.Grid) for g in subdomains):
            raise TypeError("density_driven_flux expects only subdomain grids.")
        subdomains_list = cast(list[pp.Grid], list(subdomains))

        # Retrieve cached geometric map
        geometry_map = self._subdomain_gravity_geometry(subdomains_list)

        w_flux = geometry_map @ (
                density_metric * self.gravity_field(subdomains_list)
        )
        w_flux.set_name("density_driven_flux_" + density_metric.name)
        return w_flux.simplify(self.mdg)

    def interface_density_driven_flux(
            self, interfaces: list[pp.MortarGrid], density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        # Retrieve cached geometric map
        proj_map = self._interface_gravity_geometry(interfaces)
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)

        # Apply to state
        gravity_flux_mortar = proj_map @ (
                density_metric * self.gravity_field(subdomain_neighbors)
        )

        normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
        normals_times_source = normals * gravity_flux_mortar

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
        return w_flux.simplify(self.mdg)

    def __entity_buoyancy_flux(
            self,
            advected_gamma_quantity: pp.ad.Operator,
            gamma: pp.Phase,
            delta: pp.Phase,
            domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """
        Exact logic from old code: Product formulation (f_gamma * f_delta * w_flux).
        """
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        b_fluxes: List[pp.ad.Operator] = []

        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)
        w_flux_gamma_delta = self.density_driven_flux(domains, rho_delta - rho_gamma)

        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)

        discr_gamma = self.buoyancy_discretization(gamma, delta, domains)
        discr_delta = self.buoyancy_discretization(delta, gamma, domains)

        f_gamma_upwind = discr_gamma.upwind() @ (advected_gamma_quantity * f_gamma)
        f_delta_upwind = discr_delta.upwind() @ f_delta

        b_flux_gamma_delta = (f_gamma_upwind * f_delta_upwind) * w_flux_gamma_delta
        b_fluxes.append(b_flux_gamma_delta.simplify(self.mdg))

        # Check for interfaces using cached helper
        projections = self._interface_projections(domains)

        if projections is not None:
            (
                interfaces, mortar_proj, trace,
                mortar_avg, sec_to_mortar, prim_trace,
                mortar_to_prim, _
            ) = projections

            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, rho_delta - rho_gamma
            )

            intf_discr_gamma = self.interface_buoyancy_discretization(gamma, delta, interfaces)
            intf_discr_delta = self.interface_buoyancy_discretization(delta, gamma, interfaces)

            term_gamma = (advected_gamma_quantity * f_gamma)
            gamma_interface = (
                    intf_discr_gamma.upwind_primary() @ mortar_avg @ prim_trace @ term_gamma
                    + intf_discr_gamma.upwind_secondary() @ sec_to_mortar @ term_gamma
            )

            term_delta = f_delta
            delta_interface = (
                    intf_discr_delta.upwind_primary() @ mortar_avg @ prim_trace @ term_delta
                    + intf_discr_delta.upwind_secondary() @ sec_to_mortar @ term_delta
            )

            # EXACT OLD LOGIC: Product coupling
            interface_coupling_intf = (gamma_interface * delta_interface) * intf_w_flux_gamma_delta

            bound_transport = discr_gamma.bound_transport_neu()
            b_intf_flux_gamma_delta = (
                    bound_transport @ mortar_to_prim @ interface_coupling_intf
            )

            b_fluxes.append(b_intf_flux_gamma_delta.simplify(self.mdg))

        return b_fluxes

    def __entity_buoyancy_jump(
            self,
            advected_gamma_quantity: pp.ad.Operator,
            gamma: pp.Phase,
            delta: pp.Phase,
            domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:

        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="component_buoyancy_jump_zero"
        )
        b_flux_jumps: List[pp.ad.Operator] = [zero]

        projections = self._interface_projections(domains)

        if projections is not None:
            (
                interfaces, mortar_proj, trace,
                mortar_avg, sec_to_mortar, prim_trace,
                _, mortar_to_sec
            ) = projections

            rho_gamma = gamma.density(domains)
            rho_delta = delta.density(domains)
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces,  rho_delta - rho_gamma
            )

            f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
            f_delta = self.fractional_phase_mass_mobility(delta, domains)

            intf_discr_gamma = self.interface_buoyancy_discretization(gamma, delta, interfaces)
            intf_discr_delta = self.interface_buoyancy_discretization(delta, gamma, interfaces)

            term_gamma = (advected_gamma_quantity * f_gamma)
            gamma_interface = (
                    intf_discr_gamma.upwind_primary() @ mortar_avg @ prim_trace @ term_gamma
                    + intf_discr_gamma.upwind_secondary() @ sec_to_mortar @ term_gamma
            )

            term_delta = f_delta
            delta_interface = (
                    intf_discr_delta.upwind_primary() @ mortar_avg @ prim_trace @ term_delta
                    + intf_discr_delta.upwind_secondary() @ sec_to_mortar @ term_delta
            )

            interface_coupling_intf = (gamma_interface * delta_interface) * intf_w_flux_gamma_delta

            b_flux_jump_gamma_delta = (
                    mortar_to_sec @ interface_coupling_intf
            )

            b_flux_jumps.append(b_flux_jump_gamma_delta.simplify(self.mdg))

        return b_flux_jumps

    def phase_pairs_for(self, phase: pp.Phase) -> list[tuple[pp.Phase, pp.Phase]]:
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
        b_fluxes: List[pp.ad.Operator] = []
        # Baseline term from old code
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
        return b_flux.simplify(self.mdg)

    def enthalpy_buoyancy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        b_fluxes: List[pp.ad.Operator] = []
        b_fluxes.append(self.density_driven_flux(domains, pp.ad.Scalar(0.0)))

        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_flux(h_gamma, gamma, delta, domains)

        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy")
        return b_flux.simplify(self.mdg)

    def component_buoyancy_jump(
            self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        b_fluxes: List[pp.ad.Operator] = []

        # Consistent zero baseline
        if all(isinstance(d, pp.Grid) for d in domains):
            size = sum(g.num_cells for g in domains)
            b_fluxes.append(pp.wrap_as_dense_ad_array(np.zeros(size)))

        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                chi_xi_gamma = gamma.partial_fraction_of[component_xi](domains)
                b_fluxes += self.__entity_buoyancy_jump(
                    chi_xi_gamma, gamma, delta, domains
                )
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("component_buoyancy_jump_" + component_xi.name)
        return b_flux.simplify(self.mdg)

    def enthalpy_buoyancy_jump(
            self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        b_fluxes: List[pp.ad.Operator] = []

        if all(isinstance(d, pp.Grid) for d in domains):
            size = sum(g.num_cells for g in domains)
            b_fluxes.append(pp.wrap_as_dense_ad_array(np.zeros(size)))

        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_jump(h_gamma, gamma, delta, domains)

        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy_jump")
        return b_flux.simplify(self.mdg)

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

                # Iterate safe: if no interfaces, this loop does nothing
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

                # Strict Guard for interface discretizations
                interfaces = list(self.mdg.interfaces(codim=1))
                if interfaces:
                    self.add_nonlinear_discretization(
                        self.interface_buoyancy_discretization(
                            gamma, delta, interfaces
                        ).upwind_primary(),
                    )
                    self.add_nonlinear_discretization(
                        self.interface_buoyancy_discretization(
                            gamma, delta, interfaces
                        ).upwind_secondary(),
                    )
                    self.add_nonlinear_discretization(
                        self.interface_buoyancy_discretization(
                            delta, gamma, interfaces
                        ).upwind_primary(),
                    )
                    self.add_nonlinear_discretization(
                        self.interface_buoyancy_discretization(
                            delta, gamma, interfaces
                        ).upwind_secondary(),
                    )

    def update_buoyancy_driven_fluxes(self):
        """Update stored buoyancy flux arrays (subdomains and interfaces)."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs

                # Subdomains
                subdomains = self.mdg.subdomains()
                rho_gamma_full = gamma.density(subdomains)
                rho_delta_full = delta.density(subdomains)

                subdomain_vals = self.equation_system.evaluate(
                    self.density_driven_flux(
                        subdomains,  rho_delta_full - rho_gamma_full
                    )
                )
                subdomain_offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])

                for id, (sd, data) in enumerate(self.mdg.subdomains(return_data=True)):
                    sd_offset = subdomain_offsets[id]
                    vals_loc = subdomain_vals[sd_offset: sd_offset + sd.num_faces]
                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {self.buoyant_flux_array_key(gamma, delta): +vals_loc}
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {self.buoyant_flux_array_key(delta, gamma): -vals_loc}
                    )

                # Interfaces (Strict Guard)
                interfaces = self.subdomains_to_interfaces(subdomains, [1])
                if not interfaces:
                    continue

                interface_values = self.equation_system.evaluate(
                    self.interface_density_driven_flux(
                        interfaces,   rho_delta_full - rho_gamma_full
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
                               intf_offset: intf_offset + intf.num_cells
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
