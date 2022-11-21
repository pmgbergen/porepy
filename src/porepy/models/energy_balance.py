"""Energy balance with advection and diffusion."""
import numpy as np
import porepy as pp


class EnergyBalanceEquations(pp.ScalarBalanceEquation):
    """Mixed-dimensional energy balance equation.

    Balance equation for all subdomains and advective and diffusive fluxes internally
    and on all interfaces of codimension one.

    """

    def set_equations(self):
        """Set the equations for the energy balance problem.

        A energy balance equation is set for each subdomain, and advective and diffusive
        fluxes are set for each interface of codimension one.
        """
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()
        sd_eq = self.energy_balance_equation(subdomains)
        intf_eq = self.interface_darcy_flux_equation(interfaces)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_eq, interfaces, {"cells": 1})

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy balance equation.
        """
        accumulation = self.total_energy(subdomains)
        flux = self.energy_flux(subdomains)
        source = self.energy_source(subdomains)
        eq = self.balance_equation(subdomains, accumulation, flux, source)
        eq.set_name("energy_balance_equation")
        return eq

    def fluid_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid energy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid energy.

        """
        energy_density = self.fluid_density(subdomains) * self.porosity(subdomains)
        energy = self.volume_integral(energy_density, subdomains)
        energy.set_name("fluid_energy")
        return energy

    def solid_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid energy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid energy.

        """
        energy_density = self.solid_density(subdomains) * (
            1 - self.porosity(subdomains)
        )
        energy = self.volume_integral(energy_density, subdomains)
        energy.set_name("solid_energy")
        return energy

    def total_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Total energy.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the total energy, i.e. the sum of the fluid and solid
                energy.

        """
        energy = self.fluid_energy(subdomains) + self.solid_energy(subdomains)
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

    def enthalpy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Enthalpy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the enthalpy flux.

        """
        discr = self.enthalpy_discretization(subdomains)
        enth = self.fluid_enthalpy(subdomains)
        bc_values = self.bc_values_enthalpy(subdomains)
        flux = self.advective_flux(
            subdomains, enth, discr, bc_values, self.interface_enthalpy_flux
        )
        flux.set_name("enthalpy_flux")
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

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source term.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term.
        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells)
        source = pp.ad.Array(vals, "fluid_source")
        return source
