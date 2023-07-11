import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- copied from test_hu_model.py
"""

class PrimaryVariables(pp.VariableMixin):
    """ """

    def create_variables(self) -> None:
        """ """
        self.equation_system.create_variables(
            self.pressure_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "Pa"},
        )

        self.equation_system.create_variables(  # saturation is an attribute of the model, but please use mixture to take the saturation # TODO: improve it (as everything)
            self.saturation_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": ""},
        )

        self.equation_system.create_variables(
            self.interface_mortar_flux_phase_0_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

        self.equation_system.create_variables(
            self.interface_mortar_flux_phase_1_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

    def pressure(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        p = self.equation_system.md_variable(self.pressure_variable, subdomains)
        return p

    # saturation is inside mixture

    def interface_mortar_flux_phase_0(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        mortar_flux = self.equation_system.md_variable(
            self.interface_mortar_flux_phase_0_variable, interfaces
        )
        return mortar_flux

    def interface_mortar_flux_phase_1(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        mortar_flux = self.equation_system.md_variable(
            self.interface_mortar_flux_phase_1_variable, interfaces
        )
        return mortar_flux


class EquationsPPU(test_hu_model.Equations):
    """I dont see the point of splitting this class more than this, I prefer commented titles instead of mixin classes"""

    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def eq_fcn_pressure(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(0).saturation_operator(subdomains)

        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(1).saturation_operator(subdomains)

        mass_density = self.porosity(subdomains) * (
            mass_density_phase_0 + mass_density_phase_1
        ) 

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_p_eq")

        # subdomains flux: -----------------------------------------------------------
        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        mob_rho = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure) * pp.tobedefined.mobility.mobility_operator(subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)

        darcy_flux_phase_0 = self.darcy_flux_phase_0(subdomains, self.mixture.get_phase(0))
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_0: pp.ad.Operator = (
            darcy_flux_phase_0 * (discr.upwind @ mob_rho)
            # - discr.bound_transport_neu @ bc_values # original from fluid_mass_balance.py
            + self.bc_values(subdomains) # bc might be wrong, but they are zero, so i let the next one to fix it...
        )
    
        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        mob_rho = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure) * pp.tobedefined.mobility.mobility_operator(
            subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)

        darcy_flux_phase_1 = self.darcy_flux_phase_1(subdomains, self.mixture.get_phase(1))
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_1: pp.ad.Operator = (
            darcy_flux_phase_1 * (discr.upwind @ mob_rho)
            + self.bc_values(subdomains) 
        )

        flux_tot = flux_phase_0 + flux_phase_1

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        flux = flux_tot - flux_intf_phase_0 - flux_intf_phase_1

        # sources: --------------------------------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )
        source_phase_0.set_name("interface_fluid_mass_flux_source_phase_0")

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )
        source_phase_1.set_name("interface_fluid_mass_flux_source_phase_1")

        source = source_phase_0 + source_phase_1

        eq = self.balance_equation(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
        eq.set_name("pressure_equation")

        return (
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0,
            flux_intf_phase_1,
            flux,
            source_phase_0,
            source_phase_1,
            source,
        )

    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_pressure(subdomains)
        return eq

    # MASS BALANCE: ----------------------------------------------------------------------------------------------------------------

    def eq_fcn_mass(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------------------------
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_mass_eq")

        # subdomains flux contribution: -------------------------------------
        if self.ell == 0:
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
            mob_rho = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure) * pp.tobedefined.mobility.mobility_operator(
                subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)

            darcy_flux_phase_0 = self.darcy_flux_phase_0(subdomains, self.mixture.get_phase(0))
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_0: pp.ad.Operator = (
                darcy_flux_phase_0 * (discr.upwind @ mob_rho)
                # - discr.bound_transport_neu @ bc_values # original from fluid_mass_balance.py
                + self.bc_values(subdomains) # bc might be wrong, but they are zero, so i let the next one to fix it...
            )

            flux_no_intf = flux_phase_0 # just to keep the structure of hu code

        else: # self.ell == 1
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
            mob_rho = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure) * pp.tobedefined.mobility.mobility_operator(
                subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)

            darcy_flux_phase_1 = self.darcy_flux_phase_1(subdomains, self.mixture.get_phase(1))
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_1: pp.ad.Operator = (
                darcy_flux_phase_1 * (discr.upwind @ mob_rho)
                # - discr.bound_transport_neu @ bc_values # original from fluid_mass_balance.py
                + self.bc_values(subdomains) # bc might be wrong, but they are zero, so i let the next one to fix it...
            )

            flux_no_intf = flux_phase_1

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu  # -1,0,1 matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, I need to test it

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        if (
            self.ell == 0
        ):  # TODO: move the flux computation inside if (which btw could be removed) after all the bugs are fixed
            flux = flux_phase_0 - flux_intf_phase_0

        else:  # self.ell == 1
            flux = flux_phase_1 - flux_intf_phase_1

        flux.set_name("ppu_flux")

        # sources: ---------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, there is a bug and I need to output everything for the tests

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )  # sorry, there is a bug and I need to output everything for the tests

        if self.ell == 0:
            source = source_phase_0  # I need this for the tests
            source.set_name("interface_fluid_mass_flux_source_phase_0")

        else:  # self.ell == 1:
            source = source_phase_1
            source.set_name("interface_fluid_mass_flux_source_phase_1")

        eq = self.balance_equation(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
        eq.set_name("mass_balance_equation")

        return (
            eq,
            accumulation,
            None,
            None,
            flux_no_intf,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_mass(subdomains)

        return eq


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    EquationsPPU,
    test_hu_model. ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    test_hu_model.SolutionStrategyPressureMass,
    test_hu_model.MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":

    class FinalModel(PartialFinalModel):  # I'm sorry...
        def __init__(self, mixture, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0 # 0 = wetting, 1 = non-wetting
            self.gravity_value = 1  # pp.GRAVITY_ACCELERATION
            self.dynamic_viscosity = 1  # TODO: it is hardoced everywhere, you know...

    fluid_constants = pp.FluidConstants({})
    solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "permeability": 1,
            "normal_permeability": 1e1,
            "residual_aperture": 0.1,
        }
    )
    # material_constants = {"solid": solid_constants}
    material_constants = {"fluid": fluid_constants, "solid": solid_constants}

    time_manager = pp.TimeManager(
        schedule=[0, 10],
        dt_init=1e-2,
        constant_dt=True,
        iter_max=10,
        print_info=True,
    )

    params = {
        "material_constants": material_constants,
        "max_iterations": 100,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "time_manager": time_manager,
    }

    wetting_phase = pp.composite.phase.Phase(rho0=1)
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    model = FinalModel(mixture, params) # why... add it directly as attribute: model.mixture = mixture

    pp.run_time_dependent_model(model, params)
