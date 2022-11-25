"""Energy balance with advection and diffusion.

Local thermal equilibrium is assumed, i.e., the solid and fluid temperatures are assumed
to be constant within each cell. This leads to a single equation with "effective" or
"total" quantities and parameters."""
from typing import Optional

import numpy as np

import porepy as pp


class EnergyBalanceEquations(pp.fluid_mass_balance.MassBalanceEquations):
    """Mixed-dimensional energy balance equation.

    Balance equation for all subdomains and advective and diffusive fluxes internally
    and on all interfaces of codimension one.

    """

    # Expected attributes for this mixin
    mdg: pp.MixedDimensionalGrid
    equation_system: pp.ad.EquationSystem
    fluid_internal_energy: pp.ad.Operator
    solid_internal_energy: pp.ad.Operator
    fluid_enthalpy: pp.ad.Operator
    solid_enthalpy: pp.ad.Operator
    interface_fourier_flux: pp.ad.MixedDimensionalVariable
    interface_enthalpy_flux: pp.ad.MixedDimensionalVariable

    def set_equations(self):
        """Set the equations for the energy balance problem.

        A energy balance equation is set for each subdomain, and advective and diffusive
        fluxes are set for each interface of codimension one.

        """
        # Set flow equations
        super().set_equations()
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
        energy_density = self.fluid_density(subdomains) * self.fluid_enthalpy(
            subdomains
        ) - self.pressure(subdomains)
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
        energy_density = self.solid_enthalpy(subdomains) * (
            1 - self.porosity(subdomains)
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
            self.fluid_enthalpy(subdomains),
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
            self.fluid_enthalpy(subdomains),
            discr,
        )

        eq = self.interface_enthalpy_flux(interfaces) - flux
        eq.set_name("interface_enthalpy_flux_equation")
        return eq

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy source term.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells)
        source = pp.ad.Array(vals, "energy_source")
        return source


class VariablesEnergyBalance(pp.fluid_mass_balance.VariablesSinglePhaseFlow):
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
    fluid: pp.FluidConstants

    def create_variables(self) -> None:
        """
        Assign primary variables to subdomains and interfaces of the mixed-dimensional
        grid. Old implementation awaiting SystemManager

        """
        # Mass balance variables
        super().create_variables()

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

        Parameters:
            subdomains: List of subdomains.

            Returns:
                Operator representing the reference temperature.

        """
        t_ref = self.fluid.convert_units(0, "K")
        size = sum([sd.num_cells for sd in subdomains])
        return pp.constitutive_laws.ad_wrapper(
            t_ref, True, size, name="reference_temperature"
        )


class ConstitutiveLawsEnergyBalance(
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.FouriersLawFV,
    pp.constitutive_laws.ThermalConductivityLTE,
    # Reuses advection and dimension reduction as well as the specific mass balance
    # constitutive laws.
    pp.fluid_mass_balance.ConstitutiveLawsSinglePhaseFlow,
):
    """Collect constitutive laws for the energy balance."""

    pass


class BoundaryConditionsEnergyBalance(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Boundary conditions for the energy balance.

    Boundary type and value for both diffusive Fourier flux and advective enthalpy flux.

    TODO: Unify method names. With or without flux? See also other model classes.
    """

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, all_bf, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, all_bf, "dir")

    def bc_values_fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the Fourier flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Ad array representing the boundary condition values for the Fourier flux.

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.constitutive_laws.ad_wrapper(0, True, num_faces, "bc_values_fourier")

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
            # Get enthalpy values on boundary faces applying trace to interior values.
            all_bf, *_ = self.domain_boundary_sides(sd)
            vals = np.zeros(sd.num_faces)
            vals[all_bf] = self.fluid.specific_heat_capacity()
            # Append to list of boundary values
            bc_values.append(vals)

        # Concatenate to single array and wrap as ad.Array
        bc_values = pp.constitutive_laws.ad_wrapper(
            np.hstack(bc_values), True, name="bc_values_enthalpy"
        )
        return bc_values


class SolutionStrategyEnergyBalance(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Solution strategy for the energy balance."""

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
        # Set parameters for the mass balance
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):

            specific_volume_mat = self.specific_volume([sd]).evaluate(
                self.equation_system
            )
            # Extract diagonal of the specific volume matrix.
            specific_volume = specific_volume_mat * np.ones(sd.num_cells)
            # Check that the matrix is actually diagonal.
            assert np.all(np.isclose(specific_volume, specific_volume_mat.data))

            kappa = self.thermal_conductivity([sd])
            diffusivity = pp.SecondOrderTensor(kappa * specific_volume)

            pp.initialize_data(
                sd,
                data,
                self.fourier_keyword,
                {
                    "bc": self.bc_type_fourier(sd),
                    "second_order_tensor": diffusivity,
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
            # Assign diffusivity in the normal direction of the fractures.
        # for intf, intf_data in self.mdg.interfaces(return_data=True):
        #     pp.initialize_data(
        #         intf,
        #         intf_data,
        #         self.fourier_keyword,
        #         {
        #             "ambient_dimension": self.nd,
        #         },
        #     )

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
        """
        Evaluate Darcy flux (super) and copy to the enthalpy flux keyword, to be used in
        upstream weighting.

        """
        super().before_nonlinear_iteration()
        for _, data in self.mdg.subdomains(return_data=True) + self.mdg.interfaces(
            return_data=True
        ):
            vals = data[pp.PARAMETERS][self.mobility_keyword]["darcy_flux"]
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        # TODO: Rediscretize
        self.discretize()
