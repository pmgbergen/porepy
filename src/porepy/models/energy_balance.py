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

from typing import Callable, Optional

import numpy as np

import porepy as pp


class EnergyBalanceEquations(pp.BalanceEquation):
    """Mixed-dimensional energy balance equation.

    Balance equation for all subdomains and advective and diffusive fluxes internally
    and on all interfaces of codimension one.

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
    fluid_density: Callable[[list[pp.Grid]], pp.ad.Operator]
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
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
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
    bc_values_enthalpy_flux: Callable[[list[pp.Grid]], pp.ad.Array]
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

    def set_equations(self):
        """Set the equations for the energy balance problem.

        A energy balance equation is set for each subdomain, and advective and diffusive
        fluxes are set for each interface of codimension one.

        """
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()
        sd_eq = self.energy_balance_equation(subdomains)
        intf_cond = self.interface_fourier_flux_equation(interfaces)
        intf_adv = self.interface_enthalpy_flux_equation(interfaces)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_cond, interfaces, {"cells": 1})
        self.equation_system.set_equation(intf_adv, interfaces, {"cells": 1})

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy balance equation.

        """
        accumulation = self.total_internal_energy(subdomains)
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
        energy_density = (
            self.fluid_density(subdomains) * self.fluid_enthalpy(subdomains)
            - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy = self.volume_integral(energy_density, subdomains, dim=1)
        energy.set_name("fluid_internal_energy")
        return energy

    def solid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the solid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid energy.

        """
        energy_density = (
            self.solid_density(subdomains)
            * self.solid_enthalpy(subdomains)
            * (1 - self.porosity(subdomains))
        )
        energy = self.volume_integral(energy_density, subdomains, dim=1)
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

    def enthalpy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Enthalpy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the enthalpy flux.

        """
        # Only one option for discretization currently. Refactor (method
        # enthalpy_flux_discretization) if more options are added.
        discr = pp.ad.UpwindAd(self.enthalpy_keyword, subdomains)
        flux = self.advective_flux(
            subdomains,
            self.fluid_enthalpy(subdomains)
            * self.mobility(subdomains)
            * self.fluid_density(subdomains),
            discr,
            self.bc_values_enthalpy_flux(subdomains),
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
        discr = pp.ad.UpwindCouplingAd(self.enthalpy_keyword, interfaces)
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

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy source term.

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
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        flux = self.interface_enthalpy_flux(interfaces) + self.interface_fourier_flux(
            interfaces
        )
        source = projection.mortar_to_secondary_int * flux
        source.set_name("interface_energy_source")
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
    """Name of the primary variable representing the temperature. Normally defined in a
    mixin of instance
    :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`.

    """
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

    def create_variables(self) -> None:
        """Assign primary variables to subdomains and interfaces of the
        mixed-dimensional grid.

        """
        self.equation_system.create_variables(
            self.temperature_variable,
            subdomains=self.mdg.subdomains(),
        )
        self.equation_system.create_variables(
            self.interface_fourier_flux_variable,
            interfaces=self.mdg.interfaces(),
        )
        self.equation_system.create_variables(
            self.interface_enthalpy_flux_variable,
            interfaces=self.mdg.interfaces(),
        )

    def temperature(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        """Temperature variable.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Mixed-dimensional variable representing the temperature.

        """
        t = self.equation_system.md_variable(self.temperature_variable, subdomains)
        return t

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
        return pp.wrap_as_ad_array(t_ref, size, name="reference_temperature")


class ConstitutiveLawsEnergyBalance(
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.ThermalConductivityLTE,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidDensityFromTemperature,
    pp.constitutive_laws.ConstantSolidDensity,
):
    """Collect constitutive laws for the energy balance."""

    pass


class BoundaryConditionsEnergyBalance:
    """Boundary conditions for the energy balance.

    Boundary type and value for both diffusive Fourier flux and advective enthalpy flux.

    """

    domain_boundary_sides: Callable[
        [pp.Grid],
        pp.bounding_box.DomainSides,
    ]
    """Boundary sides of the domain. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_fourier(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the Fourier flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Ad array representing the boundary condition values for the Fourier flux.

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.wrap_as_ad_array(0, num_faces, "bc_values_fourier")

    def bc_values_enthalpy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the enthalpy.

        SI units for Dirichlet: [J/m^3]
        SI units for Neumann: TODO

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Array with boundary values for the enthalpy.

        """
        # List for all subdomains
        bc_values = []

        # Loop over subdomains to collect boundary values
        for sd in subdomains:
            vals = np.zeros(sd.num_faces)
            # If you know the boundary temperature, do something like:
            # boundary_faces = self.domain_boundary_sides(sd).all_bf
            # vals[boundary_faces] = self.fluid.specific_heat_capacity() * dirichlet_values
            # Append to list of boundary values
            bc_values.append(vals)

        # Concatenate to single array and wrap as ad.Array
        bc_values_ad = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_enthalpy"
        )
        return bc_values_ad


class SolutionStrategyEnergyBalance(pp.SolutionStrategy):
    """Solution strategy for the energy balance.

    Parameters:
        params: Parameters for the solution strategy.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the specific volume of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Thermal conductivity. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a subclass.

    """
    bc_type_fourier: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the Fourier flux. Normally
    defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

    """
    bc_type_enthalpy: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the enthalpy flux.
    Normally defined in a mixin instance
    of :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`.

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
        """Name of the primary variable representing the Fourier flux on the interface."""

        self.interface_enthalpy_flux_variable: str = "interface_enthalpy_flux"
        """Name of the primary variable representing the enthalpy flux on the interface."""

        # Discrretization
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
                    "bc": self.bc_type_fourier(sd),
                    "second_order_tensor": self.thermal_conductivity_tensor(sd),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_enthalpy(sd),
                },
            )

    def thermal_conductivity_tensor(self, sd: pp.Grid) -> pp.SecondOrderTensor:
        """Convert ad conductivity to :class:`~pp.params.tensor.SecondOrderTensor`.

        Override this method if the conductivity is anisotropic.

        Parameters:
            sd: Subdomain for which the conductivity is requested.

        Returns:
            Thermal conductivity tensor.

        """
        conductivity_ad = self.specific_volume([sd]) * self.thermal_conductivity([sd])
        conductivity = conductivity_ad.evaluate(self.equation_system)
        # The result may be an Ad_array, in which case we need to extract the
        # underlying array.
        if isinstance(conductivity, pp.ad.Ad_array):
            conductivity = conductivity.val
        return pp.SecondOrderTensor(conductivity)

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
        super().before_nonlinear_iteration()
        for _, data in self.mdg.subdomains(return_data=True) + self.mdg.interfaces(
            return_data=True
        ):
            vals = data[pp.PARAMETERS][self.mobility_keyword]["darcy_flux"]
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        # TODO: Targeted rediscretization.
        self.discretize()
