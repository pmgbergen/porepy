"""Energy balance with advection and diffusion.

Local thermal equilibrium is assumed, i.e., the solid and fluid temperatures are assumed
to be constant within each cell. This leads to a single equation with "effective" or
"total" quantities and parameters.

Since the current implementation assumes a flow field provided by a separate model, the
energy balance equation is not stand-alone. Thus, no class `EnergyBalance` is provided,
as would be consistent with the other models. However, the class is included in coupled
models, notably :class:`~porepy.models.mass_and_energy_balance.MassAndEnergyBalance`.

"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union, cast

import numpy as np

import porepy as pp


class EnergyBalanceEquations(pp.BalanceEquation):
    """Mixed-dimensional energy balance equation.

    Balance equation for all subdomains and advective and diffusive fluxes internally
    and on all interfaces of codimension one and advection on interfaces of codimension
    two (well-fracture intersections).

    The class is not meant to be used stand-alone, but as a mixin in a coupled model.

    """

    # Expected attributes for this mixin
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    interface_fourier_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Fourier flux variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    fluid_density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid density. Defined in a mixin class with a suitable constitutive relation.
    """
    fluid_enthalpy: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Fluid enthalpy. Defined in a mixin class with a suitable constitutive relation.
    """
    solid_enthalpy: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Solid enthalpy. Defined in a mixin class with a suitable constitutive relation.
    """
    solid_density: Callable[[list[pp.Grid]], pp.ad.Scalar]
    """Solid density. Defined in a mixin class with a suitable constitutive relation.
    """
    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    fourier_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Fourier flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FouriersLaw`.

    """
    interface_enthalpy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Variable for interface enthalpy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    enthalpy_keyword: str
    """Keyword used to identify the enthalpy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
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
    """Ad operator representing the advective flux. Normally provided by a mixin
    instance of :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    bc_values_enthalpy_flux: Callable[[list[pp.Grid]], pp.ad.DenseArray]
    """Boundary condition for enthalpy flux. Normally defined in a mixin instance
    of :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """
    interface_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """Ad operator representing the advective flux on internal boundaries. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    well_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """Ad operator representing the advective flux on well interfaces. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    well_enthalpy_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Variable for well enthalpy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.

    """
    enthalpy_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Discretization of the enthalpy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.EnthalpyFromTemperature`.

    """
    interface_enthalpy_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the enthalpy flux on internal boundaries. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.EnthalpyFromTemperature`.

    """
    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
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

    bc_type_enthalpy_flux: Callable[[pp.Grid], pp.BoundaryCondition]

    bc_data_enthalpy_flux_key: str

    def set_equations(self):
        """Set the equations for the energy balance problem.

        A energy balance equation is set for each subdomain, and advective and diffusive
        fluxes are set for each interface of codimension one.

        """
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        codim_2_interfaces = self.mdg.interfaces(codim=2)
        # Define the equations
        sd_eq = self.energy_balance_equation(subdomains)
        intf_cond = self.interface_fourier_flux_equation(codim_1_interfaces)
        intf_adv = self.interface_enthalpy_flux_equation(codim_1_interfaces)
        well_eq = self.well_enthalpy_flux_equation(codim_2_interfaces)

        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_cond, codim_1_interfaces, {"cells": 1})
        self.equation_system.set_equation(intf_adv, codim_1_interfaces, {"cells": 1})
        self.equation_system.set_equation(well_eq, codim_2_interfaces, {"cells": 1})

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy balance equation.

        """
        accumulation = self.volume_integral(
            self.total_internal_energy(subdomains), subdomains, dim=1
        )
        flux = self.energy_flux(subdomains)
        source = self.energy_source(subdomains)
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("energy_balance_equation")
        return eq

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the fluid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid energy.

        """
        energy = (
            self.fluid_density(subdomains) * self.fluid_enthalpy(subdomains)
            - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_internal_energy")
        return energy

    def solid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the solid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid energy.

        """
        energy = (
            self.solid_density(subdomains)
            * self.solid_enthalpy(subdomains)
            * (pp.ad.Scalar(1) - self.porosity(subdomains))
        )
        energy.set_name("solid_internal_energy")
        return energy

    def total_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Total energy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the total energy, i.e. the sum of the fluid and solid
            energy.

        """
        energy = self.fluid_internal_energy(subdomains) + self.solid_internal_energy(
            subdomains
        )
        energy.set_name("total_energy")
        return energy

    def energy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy flux.

        Energy flux is the sum of the advective and diffusive fluxes.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy flux.

        """
        flux = self.fourier_flux(subdomains) + self.enthalpy_flux(subdomains)
        flux.set_name("energy_flux")
        return flux

    def interface_energy_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface fluid flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux.

        """
        flux: pp.ad.Operator = self.interface_fourier_flux(
            interfaces
        ) + self.interface_enthalpy_flux(interfaces)
        flux.set_name("interface_energy_flux")
        return flux

    def enthalpy_flux(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Enthalpy flux.

        Note:
            The advected entity in the enthalpy flux is a product of density,
            enthalpy and mobility of the fluid.
            When using upwinding, Dirichlet-type data for pressure and temperature
            must also be provided on the Neumann-boundary when there is
            an in-flux into the domain.
            The advected entity must provide values on the boundary in this case, since
            the upstream value of it is on the boundary.

        Parameters:
            subdomains: List of subdomains or boundary grids.

        Raises:
            ValueError: If the domains are not all grids or all boundary grids.

        Returns:
            Operator representing the enthalpy flux.

        """

        if len(subdomains) == 0 or all(
            [isinstance(g, pp.BoundaryGrid) for g in subdomains]
        ):
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_enthalpy_flux_key,
                domains=subdomains,
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in subdomains]):
            raise ValueError(
                """Argument domains a mixture of grids and
                                boundary grids"""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        subdomains = cast(list[pp.Grid], subdomains)

        def enthalpy_dirichlet(boundary_grids):
            result = self.fluid_enthalpy(boundary_grids)
            result *= self.mobility_rho(boundary_grids)
            return result

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=enthalpy_dirichlet,
            neumann_operator=self.enthalpy_flux,
            # Robin operator is not relevant for advective fluxes
            robin_operator=None,
            bc_type=self.bc_type_enthalpy_flux,
            name="bc_values_enthalpy",
        )

        discr = self.enthalpy_discretization(subdomains)
        flux = self.advective_flux(
            subdomains,
            self.fluid_enthalpy(subdomains)
            * self.mobility(subdomains)
            * self.fluid_density(subdomains),
            discr,
            boundary_operator,
            self.interface_enthalpy_flux,
        )
        flux.set_name("enthalpy_flux")
        return flux

    def interface_enthalpy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface enthalpy flux.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_enthalpy_discretization(interfaces)
        flux = self.interface_advective_flux(
            interfaces,
            self.fluid_enthalpy(subdomains)
            * self.mobility(subdomains)
            * self.fluid_density(subdomains),
            discr,
        )

        eq = self.interface_enthalpy_flux(interfaces) - flux
        eq.set_name("interface_enthalpy_flux_equation")
        return eq

    def well_enthalpy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Well interface enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface enthalpy flux.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = pp.ad.UpwindCouplingAd(self.enthalpy_keyword, interfaces)
        flux = self.well_advective_flux(
            interfaces,
            self.fluid_enthalpy(subdomains)
            * self.mobility(subdomains)
            * self.fluid_density(subdomains),
            discr,
        )

        eq = self.well_enthalpy_flux(interfaces) - flux
        eq.set_name("well_enthalpy_flux_equation")
        return eq

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy source term.

        Includes

            - external sources
            - interface flow from neighboring subdomains of higher dimension.
            - well flow from neighboring subdomains of lower and higher dimension

        .. note::
            When overriding this method to assign internal energy sources, one is
            advised to call the base class method and add the new contribution, thus
            ensuring that the source term includes the contribution from the interface
            fluxes.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term.

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Interfaces relating to wells, and the associated subdomains.
        well_interfaces = self.subdomains_to_interfaces(subdomains, [2])
        well_subdomains = self.interfaces_to_subdomains(well_interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        well_projection = pp.ad.MortarProjections(
            self.mdg, well_subdomains, well_interfaces
        )
        subdomain_projection = pp.ad.SubdomainProjections(self.mdg.subdomains())
        flux = self.interface_enthalpy_flux(interfaces) + self.interface_fourier_flux(
            interfaces
        )
        # Matrix-vector product, use @
        source = projection.mortar_to_secondary_int @ flux
        # Add contribution from well interfaces
        source.set_name("interface_energy_source")
        well_fluxes = well_projection.mortar_to_secondary_int @ self.well_enthalpy_flux(
            well_interfaces
        ) - well_projection.mortar_to_primary_int @ self.well_enthalpy_flux(
            well_interfaces
        )
        well_fluxes.set_name("well_enthalpy_flux_source")
        source += subdomain_projection.cell_restriction(subdomains) @ (
            subdomain_projection.cell_prolongation(well_subdomains) @ well_fluxes
        )
        return source


class VariablesEnergyBalance:
    """
    Creates necessary variables (temperature, advective and diffusive interface flux)
    and provides getter methods for these and their reference values. Getters construct
    mixed-dimensional variables on the fly, and can be called on any subset of the grids
    where the variable is defined. Setter method (assign_variables), however, must
    create on all grids where the variable is to be used.

    Note:
        Wrapping in class methods and not calling equation_system directly allows for
        easier changes of primary variables. As long as all calls to enthalpy_flux()
        accept Operators as return values, we can in theory add it as a primary variable
        and solved mixed form. Similarly for different formulations of enthalpy instead
        of temperature.

    """

    # Expected attributes for this mixin
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    temperature_variable: str
    """See :attr:`SolutionStrategyEnergyBalance.temperature_variable`."""
    interface_fourier_flux_variable: str
    """Name of the primary variable representing the Fourier flux across an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
    interface_enthalpy_flux_variable: str
    """Name of the primary variable representing the enthalpy flux across an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
    well_enthalpy_flux_variable: str
    """Name of the primary variable representing the enthalpy flux across a well
    interface. Normally defined in a mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.

    """
    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    nd: int
    """Number of spatial dimensions. Normally defined in a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def create_variables(self) -> None:
        """Assign primary variables to subdomains and interfaces of the
        mixed-dimensional grid.

        """
        self.equation_system.create_variables(
            self.temperature_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "K"},
        )
        # Flux variables are extensive (surface integrated) and thus have units of W.
        self.equation_system.create_variables(
            self.interface_fourier_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"W * m^{self.nd - 3}"},
        )
        self.equation_system.create_variables(
            self.interface_enthalpy_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"W * m^{self.nd - 3}"},
        )
        self.equation_system.create_variables(
            self.well_enthalpy_flux_variable,
            interfaces=self.mdg.interfaces(codim=2),
            tags={"si_units": f"W * m^{self.nd - 3}"},
        )

    def temperature(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Representation of the temperature as an AD-Operator.

        Parameters:
            domains: List of subdomains or list of boundary grids.

        Raises:
            ValueError: If the passed sequence of domains does not consist entirely
                of instances of boundary grid.

        Returns:
            A mixed-dimensional variable representing the temperature, if called with a
            list of subdomains.

            If called with a list of boundary grids, returns an operator representing
            boundary values.

        """
        if len(domains) > 0 and all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return self.create_boundary_operator(
                name=self.temperature_variable, domains=domains  # type: ignore[call-arg]
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument domains a mixture of subdomain and boundary grids"""
            )

        domains = cast(list[pp.Grid], domains)

        return self.equation_system.md_variable(self.temperature_variable, domains)

    def interface_fourier_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface Fourier flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Fourier flux.

        """
        flux = self.equation_system.md_variable(
            self.interface_fourier_flux_variable, interfaces
        )
        return flux

    def interface_enthalpy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface enthalpy flux.
        """
        flux = self.equation_system.md_variable(
            self.interface_enthalpy_flux_variable, interfaces
        )
        return flux

    def well_enthalpy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Well enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the well enthalpy flux.

        """
        flux = self.equation_system.md_variable(
            self.well_enthalpy_flux_variable, interfaces
        )
        return flux

    def reference_temperature(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference temperature.

        For now, we assume that the reference temperature is the same for solid and
        fluid. More sophisticated models may require different reference temperatures.

        Parameters:
            subdomains: List of subdomains.

            Returns:
                Operator representing the reference temperature.

        """
        t_ref = self.fluid.temperature()
        assert t_ref == self.solid.temperature()
        size = sum([sd.num_cells for sd in subdomains])
        return pp.wrap_as_dense_ad_array(t_ref, size, name="reference_temperature")


class ConstitutiveLawsEnergyBalance(
    pp.constitutive_laws.SpecificHeatCapacities,
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.ThermalConductivityLTE,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    pp.constitutive_laws.ConstantSolidDensity,
):
    """Collect constitutive laws for the energy balance."""


class BoundaryConditionsEnergyBalance(pp.BoundaryConditionMixin):
    """Boundary conditions for the energy balance.

    Boundary type and value for both diffusive Fourier flux and advective enthalpy flux.

    """

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    bc_data_fourier_flux_key: str = "fourier_flux"
    """Keyword for the storage of Neumann-type boundary conditions for the Fourier
    flux."""
    bc_data_enthalpy_flux_key: str = "enthalpy_flux"
    """Keyword for the storage of Neumann-type boundary conditions for the advective
    enthalpy flux."""
    temperature_variable: str
    """See :attr:`SolutionStrategyEnergyBalance.temperature_variable`."""
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :meth:`VariablesEnergyBalance.temperature`."""
    fluid_enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid enthalpy. Defined in a mixin class with a suitable constitutive relation.
    """
    fluid_density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid density. Defined in a mixin class with a suitable constitutive relation.
    """
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :meth:`~porepy.models.constitutive_laws.DarcysLaw.darcy_flux`."""
    fourier_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :meth:`~porepy.models.constitutive_laws.FouriersLaw.fourier_flux`."""
    enthalpy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :meth:`EnergyBalanceEquations.enthalpy_flux`."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries for the conductive flux
        in the energy equation.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring temperature values on the bonudary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries for the advective flux in the
        energy equation.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring pressure and some energy-related values on the bonudary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for temperature
        on the boundary, such as the Fourier flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for
            temperature, per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing temperature
            values on the provided boundary grid.

        """
        return self.fluid.temperature() * np.ones(boundary_grid.num_cells)

    def bc_values_fourier_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """**Heat** flux values on the Neumann boundary to be used with Fourier's law.

        The values are used on the boundary for :math:`c \\nabla T` where Neumann data
        is required for the whole expression
        (``c`` being the conductivity on the boundary).

        Important:
            Override this method to provide custom Neumann boundary data for
            the flux, per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grids: Boundary grid to provide values for.

        Returns:
            Numeric Fourier flux values for a Neumann-type BC.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_enthalpy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        r"""**Energy** flux values on the Neumann boundary.

        These values are used on the boundary for
        :math:`\frac{\rho h}{\mu} \mathbf{K} \nabla p` where Neumann data is required
        for the whole expression.

        Important:
            Override this method to provide custom Neumann boundary data for
            the flux, per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grids: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing values for the
            flux on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)

    def update_all_boundary_conditions(self) -> None:
        """Set values for the temperature and the Fourier flux on boundaries.

        Note:
            This assumes as of now that Dirichlet-type BC are provided only for
            temperature.
            Work must be done if other energy-related quantities are defined as
            primary variables.

        """
        super().update_all_boundary_conditions()

        # Update Neumann conditions
        self.update_boundary_condition(
            name=self.bc_data_fourier_flux_key, function=self.bc_values_fourier_flux
        )
        # Update Dirichlet conditions
        self.update_boundary_condition(
            name=self.temperature_variable, function=self.bc_values_temperature
        )
        #
        self.update_boundary_condition(
            name=self.bc_data_enthalpy_flux_key, function=self.bc_values_enthalpy_flux
        )


class SolutionStrategyEnergyBalance(pp.SolutionStrategy):
    """Solution strategy for the energy balance.

    Parameters:
        params: Parameters for the solution strategy.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    """Function that returns the specific volume of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Thermal conductivity. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a subclass.

    """
    bc_type_fourier_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the Fourier flux. Normally
    defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """
    bc_type_enthalpy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the enthalpy flux.
    Normally defined in a mixin instance
    of :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """
    enthalpy_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Discretization of the enthalpy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.EnthalpyFromTemperature`.

    """
    interface_enthalpy_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the enthalpy flux on internal boundaries. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.EnthalpyFromTemperature`.

    """
    operator_to_SecondOrderTensor: Callable[
        [pp.Grid, pp.ad.Operator, pp.number], pp.SecondOrderTensor
    ]
    """Function that returns a SecondOrderTensor provided a method returning
    permeability as a Operator. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.SecondOrderTensorUtils`.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        # Generic solution strategy initialization in pp.SolutionStrategy and specific
        # initialization for the fluid mass balance (variables, discretizations...)
        super().__init__(params)

        # Define the energy balance
        # Variables
        self.temperature_variable: str = "temperature"
        """Name of the temperature variable."""

        self.interface_fourier_flux_variable: str = "interface_fourier_flux"
        """Name of the primary variable representing the Fourier flux on interfaces of
        codimension one."""

        self.interface_enthalpy_flux_variable: str = "interface_enthalpy_flux"
        """Name of the primary variable representing the enthalpy flux on interfaces of
        codimension one."""

        self.well_enthalpy_flux_variable: str = "well_enthalpy_flux"
        """Name of the primary variable representing the well enthalpy flux on
        interfaces of codimension two."""

        # Discretization
        self.fourier_keyword: str = "fourier_discretization"
        """Keyword for Fourier flux term.

        Used to access discretization parameters and store discretization matrices.

        """
        self.enthalpy_keyword: str = "enthalpy_flux_discretization"
        """Keyword for enthalpy flux term.

        Used to access discretization parameters and store discretization matrices.

        """

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the energy problem.

        The parameter fields of the data dictionaries are updated for all subdomains and
        interfaces (of codimension 1).
        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.fourier_keyword,
                {
                    "bc": self.bc_type_fourier_flux(sd),
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.thermal_conductivity([sd]),
                        self.fluid.thermal_conductivity(),
                    ),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_enthalpy_flux(sd),
                },
            )

    def initial_condition(self) -> None:
        """Add darcy flux to discretization parameter dictionaries."""
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.enthalpy_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
            )

    def before_nonlinear_iteration(self):
        """Evaluate Darcy flux (super) and copy to the enthalpy flux keyword, to be used
        in upstream weighting.

        """
        # Update parameters *before* the discretization matrices are re-computed.
        equation_system = self.equation_system
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self.darcy_flux([sd]).value(equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).value(equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.well_flux([intf]).value(equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        super().set_nonlinear_discretizations()
        self.add_nonlinear_discretization(
            self.enthalpy_discretization(self.mdg.subdomains()).upwind(),
        )
        self.add_nonlinear_discretization(
            self.interface_enthalpy_discretization(self.mdg.interfaces()).flux(),
        )
