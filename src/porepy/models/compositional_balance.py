"""Module defining basic equatios for fluid flow with multiple componens/species."""
from __future__ import annotations

from typing import Callable, Optional, cast

import porepy as pp
import porepy.composite as ppc

from . import fluid_mass_balance as mass
from . import mass_and_energy_balance as mass_energy
from .fluid_mixture_equilibrium import EquilibriumMixin, MixtureMixin


class DiscretizationsCompositionalFlow:
    """Mixin class defining which discretization is to be used for the magnitude of
    terms in the compositional flow and transport model.

    The flexibility is required due to the varying mathematical nature of the pressure
    equation, energy balance and transport equations.

    They also need all separate instances of the discretization objects to avoid
    false storage access.

    """

    total_mobility_keyword: str
    """See :attr:`SolutionStrategyCompositionalFlow.total_mobility_keyword`"""

    def total_mobility_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Discretization of the total fluid mobility in the total mass balance on the
        subdomains.
        (non-linear weight in the Darcy flux in the pressure equation)

        Uses Upwinding. Overwrite for to use a different discretization.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """
        return pp.ad.UpwindAd(self.total_mobility_keyword, subdomains)

    def interface_total_mobility_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the total fluid mobility in the total mass balance on the
        interfaces.

        Uses upwinding, i.e. it choses the upstream value based on the interface flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.total_mobility_keyword, interfaces)


class TotalMassBalanceEquation(mass.MassBalanceEquations):
    """Mixed-dimensional balance of total mass in a fluid mixture.

    Also referred to as *pressure equation*

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
    """A mixture containing all modelled phases and components, and required fluid
    properties as a combination of phase properties. Usually defined in a mixin class
    defining the mixture."""

    relative_permeability: Callable[[pp.ad.Operator], pp.ad.Operator]
    """See :meth:`ConstitutiveLawsCompositionalFlow.relative_permeability`."""

    total_mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :attr:`DiscretizationsCompositionalFlow.total_mobility_discretization`"""

    interface_total_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """See
    :attr:`DiscretizationsCompositionalFlow.interface_total_mobility_discretization`"""

    bc_data_total_mobility_key: str
    """See :attr:`BoundaryConditionsCompositionalFlow.bc_data_total_mobility_key`"""

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The inherited method gives another name to the equation, because there are
        multiple mass balances in the compositional setting."""
        eq = super().mass_balance_equation(subdomains)
        eq.set_name("total_mass_balance_equation")
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
        """A modified fluid flux, where the advected entity (total mobility) accounts
        for all phases in the mixture.

        See :meth:`total_mobility`.

        It also accounts for custom choices of mobility discretizations (see
        :class:`DiscretizationsCompositionalFlow`).

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_fluid_flux_key, domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError(
                "Domains must consist entirely of subdomains for the fluid flux."
            )
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        discr = self.total_mobility_discretization(domains)
        weight = self.total_mobility(domains)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.total_mobility,
            neumann_operator=self.fluid_flux,
            bc_type=self.bc_type_fluid_flux,
            name="bc_values_total_fluid_flux",
        )
        flux = self.advective_flux(
            domains, weight, discr, boundary_operator, self.interface_fluid_flux
        )
        flux.set_name("total_fluid_flux")
        return flux

    def total_mobility(
        self, domains: list[pp.SubdomainsOrBoundaries]
    ) -> pp.ad.Operator:
        r"""Returns the non-linear weight in the advective flux, assuming the mixture is
        defined on all subdomains.

        Parameters:
            subdomains: All subdomains in the md-grid or respective boundary grids.

        Returns:
            An operator representing

            .. math::

                \sum_j  / \mu_j \dfrac{\rho(p, T, x_j) k_r(s_j)}{\mu_j},

            which is the advected mass in the total mass balance equation.

            If boundary grids are passed, this returns a boundary operator whose values
            must be updated based on the boundary data for primary variables.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_total_mobility_key, domains=domains
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
                phase.density
                / phase.viscosity
                * self.relative_permeability(phase.saturation)
                for phase in self.fluid_mixture.phases
            ],
            "total_mobility",
        )

        return weight

    def interface_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Analogous to :meth:`fluid_flux` this accounts for the modified weight
        :meth:`total_mobility` and a customizable discretization
        (see
        :meth:`DiscretizationsCompositionalFlow.interface_total_mobility_discretization`
        )."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_total_mobility_discretization(interfaces)
        weight = self.total_mobility(subdomains)
        flux = self.interface_advective_flux(interfaces, weight, discr)
        flux.set_name("interface_fluid_flux")
        return flux

    def well_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Modifications analogous to :meth:`interface_fluid_flux`."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_total_mobility_discretization(interfaces)
        mob_rho = self.total_mobility(subdomains)
        # Call to constitutive law for advective fluxes.
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("well_fluid_flux")
        return flux


class EquationsCompositionalFlow(
    TotalMassBalanceEquation,
    TotalEnergyBalanceEquation,
    ComponentMassBalanceEquations,
    EquilibriumMixin,
):
    def set_equations(self):
        TotalMassBalanceEquation.set_equations(self)
        EquilibriumMixin.set_equations(self)
        if "v" not in self.equilibrium_type:
            EquilibriumMixin.set_density_relations_for_phases(self)


class VariablesCompositionalFlow(mass_energy.VariablesFluidMassAndEnergy):
    """Extension of the standard variables pressure and temperature by an additional
    variable, the transported enthalpy."""

    enthalpy_variable: str
    """See :attr:`SolutionStrategyCompositionalFlow.enthalpy_variable`."""

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        Call both parent classes' set_variables methods.

        """
        # pressure and temperature. This covers also the interface variables for
        # Fourier flux, Darcy flux and enthalpy flux.
        mass_energy.VariablesFluidMassAndEnergy.create_variables(self)

        self.equation_system.create_variables(
            self.enthalpy_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "J"},
        )

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
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantPermeability,
):
    """The central constitutive law is the equilibrium mixin, which models the local
    thermodynamic equilibrium using indirectly the EoS used by the mixture model.

    Other constitutive laws for the compositional flow treat (as of now) only
    Darcy's law, Fourier's law, and constant matrix parameters."""

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        This basic implementation implements the quadratic law :math:`s_j^2` for a phase
        :math:`j`. Overwrite for something else.

        Parameters:
            saturation: Operator representing the saturation of a phase.

        Returns:
            ``saturation ** 2``.

        """
        return saturation**2


class BoundaryConditionsCompositionalFlow(
    mass_energy.BoundaryConditionsFluidMassAndEnergy,
):
    """Mixin treating boundary conditions for the compositional flow.

    Atop of inheriting the treatment for single phase flow (which is exploited for the
    total mass balance) and the energy balance (total energy balance), this class has
    a treatment for BC for component mass balances."""

    bc_data_total_mobility_key: str = "bc_data_total_mobility"
    """Key for the (time-dependent) BC data for the total mobility in the pressure
    equation.

    Note that this data is not set, but computed by the algorithm based on BC data for
    primary and secondary variables.

    """


# TODO overwrite methods for iterations, initial conditions, bc update, ...
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

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

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

        self.total_mobility_keyword: str = "total_mobility"
        """Keyword for storing the discretization parameters and matrices of the
        discretization for the total mobility in the pressure equation."""


class CompositionalFlow(  # type: ignore[misc]
    MixtureMixin,
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
