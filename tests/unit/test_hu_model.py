import numpy as np
import scipy as sp
import scipy.sparse as sps
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays


import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- is it ok to consider mu (dynamic viscosity, mu = nu/rho) constant? 
- 
"""

class FunctionTotalFlux(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        # equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False  # TODO: 100% sure?
        # self.equation_system = equation_system
        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """
        pressure = data["for_hu"]["pressure_AdArray"]
        gravity_value = data["for_hu"][
            "gravity_value"
        ]  # just saying, you can add it as attribute during initialization

        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        expansion_matrix = data["for_hu"]["expansion_matrix"]
        transmissibility_internal_tpfa = data["for_hu"][
            "transmissibility_internal_tpfa"
        ]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry
        dim_max = data["for_hu"]["dim_max"]

        return [
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            expansion_matrix,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
            dim_max,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """
        remember that *args_not_used are result[1] in _parse_operator
        - I rely on the fact the _parse first calld get_values and THEN get_jacobian
        """
        self.result_list = []  # otherwise it will remembers old values...

        known_term_list = (
            []
        )  # TODO: find a better name, which is not rhs because it is not yet


        # subdomains = []
        # datas = []
        # for subdomain, data in self.mdg.subdomains(return_data=True): #######################################
        #     subdomains.append(subdomain)
        #     datas.append(data)


        for subdomain, data in self.mdg.subdomains(return_data=True):
        # for i in [1,0]: #reverse the order
        #     subdomain = subdomains[i] ############################################################ see picture for the result ad.val
        #     data = datas[i] 


            args = [subdomain]

            args.append(
                # self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
                self.mixture.mixture_for_subdomain(subdomain)
            )  # sorry

            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))    


            # print('\n\n\n before result')
            # print(subdomain)
            # pdb.set_trace()

            result = self.func(*args)
            self.result_list.append(result)

            known_term_list.append(
                result.val
            )  # equation_system.assemble_subsystem will add the "-", so it will actually become a rhs

            # print('\n\n\nafter result')
            # pdb.set_trace()

        # return np.hstack(known_term_list)  # TODO: how much are you sure about this hstack? idem for jacobian
        return np.concatenate(known_term_list) # same as before, I want to use exacly the same commands int equation_system.assemble_subsystem

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """
        - I rely on the fact the _parse first calls get_values and THEN get_jacobian
        """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        return sp.sparse.vstack(jac_list)  # TODO: are you 100% sure?


class FunctionRhoV(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        # equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        # self.equation_system = equation_system
        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """

        ell = data["for_hu"]["ell"]
        pressure = data["for_hu"]["pressure_AdArray"]
        total_flux_internal = data["for_hu"]["total_flux_internal"]
        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry

        return [
            ell,
            pressure,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        self.result_list = []

        known_term_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                # self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
                self.mixture.mixture_for_subdomain(subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            results = self.func(*args)
            self.result_list.append(results)
            known_term_list.append(results.val)

        # return np.hstack(known_term_list)
        return np.concatenate(known_term_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        # return sp.sparse.block_diag(jac_list)
        return sp.sparse.vstack(jac_list)


class FunctionRhoG(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        # equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        # self.equation_system = equation_system
        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """

        ell = data["for_hu"]["ell"]
        pressure = data["for_hu"]["pressure_AdArray"]
        gravity_value = data["for_hu"]["gravity_value"]
        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        transmissibility_internal_tpfa = data["for_hu"][
            "transmissibility_internal_tpfa"
        ]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry
        dim_max = data["for_hu"]["dim_max"]

        return [
            ell,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
            dim_max,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        self.result_list = []

        known_term_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                # self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
                self.mixture.mixture_for_subdomain(subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            result = self.func(*args)
            self.result_list.append(result)

            known_term_list.append(result.val)
        # return np.hstack(known_term_list)
        return np.concatenate(known_term_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        # return sp.sparse.block_diag(jac_list)
        return sp.sparse.vstack(jac_list)


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


class Equations(pp.BalanceEquation):
    """I dont see the point of splitting this class more than this, I prefer commented titles instead of mixin classes"""

    def set_equations(self):
        # pressure equation:
        subdomains = self.mdg.subdomains()
        subdomain_p_eq = self.pressure_equation(subdomains)
        self.equation_system.set_equation(subdomain_p_eq, subdomains, {"cells": 1})

        # mass balance:
        subdomains = self.mdg.subdomains()
        subdomain_mass_bal = self.mass_balance_equation(subdomains)
        self.equation_system.set_equation(subdomain_mass_bal, subdomains, {"cells": 1})

        # mortar fluxes pahse 0:
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        interface_mortar_eq_phase_0 = self.interface_mortar_flux_equation_phase_0(
            codim_1_interfaces
        )
        self.equation_system.set_equation(
            interface_mortar_eq_phase_0, codim_1_interfaces, {"cells": 1}
        )
        
        # mortar fluxes phase 1:
        interface_mortar_eq_phase_1 = self.interface_mortar_flux_equation_phase_1(
            codim_1_interfaces
        )
        self.equation_system.set_equation(
            interface_mortar_eq_phase_1, codim_1_interfaces, {"cells": 1}
        )

    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def eq_fcn_pressure(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(0).saturation_operator(subdomains)

        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(1).saturation_operator(subdomains)

        # lala = mass_density_phase_0.evaluate(self.equation_system)
        # head = mass_density_phase_1.evaluate(self.equation_system)
        # horsefly = self.mixture.get_phase(0).saturation_operator(subdomains).evaluate(self.equation_system)
        # boy = self.mixture.get_phase(1).saturation_operator(subdomains).evaluate(self.equation_system)
        # pdb.set_trace()

        mass_density = self.porosity(subdomains) * (
            mass_density_phase_0 + mass_density_phase_1
        ) # pp.ad.Scalar(0)*

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_p_eq")

        # subdoamins flux: ---------------------------------------------------------
        rho_total_flux_operator = FunctionTotalFlux(
            pp.rho_total_flux,
            # self.equation_system,
            self.mixture,
            self.mdg,
            name="rho qt",
        )

        fake_input = self.pressure(subdomains)
        flux_tot = rho_total_flux_operator(fake_input) # + self.bc_values(subdomains)  # this is wrong, but bc val are 0 so I dont care...

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
        rho_V_operator = FunctionRhoV(
            pp.rho_flux_V,
            # self.equation_system,
            self.mixture,
            self.mdg,
            name="rho V",
        )
        rho_G_operator = FunctionRhoG(
            pp.rho_flux_G,
            # self.equation_system,
            self.mixture,
            self.mdg,
            name="rho G",
        )

        fake_input = self.pressure(subdomains)
        flux_V_G = (
            rho_V_operator(fake_input)
            + rho_G_operator(fake_input)
            # + self.bc_values(subdomains)
        )  # TODO: bc wrong, but bc val are 0 so I dont care...

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
            flux = flux_V_G - flux_intf_phase_0

        else:  # self.ell == 1
            flux = flux_V_G - flux_intf_phase_1

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
            rho_V_operator(fake_input),
            rho_G_operator(fake_input),
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_mass(subdomains)

        return eq

    # DARCY LAWS (to be imporeved): ----------------------------------------------------------------------------------------------------------

    def interface_mortar_flux_equation_phase_0(self, interfaces: list[pp.MortarGrid]):
        """definition of mortar flux
        - mortar flux is integrated mortar flux
        - mobility, Kr(s)/mu, is not included into this definition of mortar flux
        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        trace = pp.ad.Trace(subdomains, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace_phase_0(
            subdomains
        )
        specific_volume_intf = (
            projection.primary_to_mortar_avg
            @ trace.trace
            @ self.specific_volume(subdomains)
        )
        specific_volume_intf.set_name("specific_volume_at_interfaces")

        discr = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_0"
        )

        density_upwinded = self.var_upwinded_interfaces(
            interfaces,
            self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure),
            discr,
        )

        eq = self.interface_mortar_flux_phase_0(interfaces) - (
            cell_volumes
            * (
                pp.ad.Scalar(self.solid.normal_permeability())
                * normal_gradient  # this is 2/eps
                * specific_volume_intf
                * (
                    pressure_h
                    - pressure_l
                    - density_upwinded
                    / normal_gradient  # Why is the density in fluid_mass_balance not here?
                    * self.interface_vector_source(
                        interfaces, material="fluid"
                    )  # gravity
                )
            )
        )

        eq.set_name("interface_darcy_flux_equation_phase_0")
        return eq

    def interface_mortar_flux_equation_phase_1(self, interfaces: list[pp.MortarGrid]):
        """ """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        trace = pp.ad.Trace(subdomains, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace_phase_1(
            subdomains
        )
        specific_volume_intf = (
            projection.primary_to_mortar_avg
            @ trace.trace
            @ self.specific_volume(subdomains)
        )
        specific_volume_intf.set_name("specific_volume_at_interfaces")

        discr = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_1"
        )
        density_upwinded = self.var_upwinded_interfaces(
            interfaces,
            self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure),
            discr,
        )

        eq = self.interface_mortar_flux_phase_1(interfaces) - (
            cell_volumes
            * (
                pp.ad.Scalar(self.solid.normal_permeability())
                * normal_gradient  # this is 2/eps
                * specific_volume_intf
                * (
                    pressure_h
                    - pressure_l
                    - density_upwinded
                    / normal_gradient  # Why is here not the density in fluid_mass_balance? bcs someone wants to discretize it differently
                    * self.interface_vector_source(
                        interfaces, material="fluid"
                    )  # gravity
                )
            )
        )

        eq.set_name("interface_mortar_flux_equation_phase_1")
        return eq

    # MISCELLANEA FOR EQUATIONS: -------------------------------------------------------------------------------------------------

    def interface_fluid_mass_flux_phase_0(
        self, mortar, interfaces: list[pp.MortarGrid], phase, flux_array_key
    ) -> pp.ad.Operator:
        """TODO: redundant input
        - it's stupid to have mortar as input, I need more flexibility for the tests
        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_ppu_discretization(interfaces, flux_array_key)
        rho_mob = phase.mass_density_operator(
            subdomains, self.pressure
        ) * pp.tobedefined.mobility.mobility_operator(
            subdomains, phase.saturation_operator, self.dynamic_viscosity
        )

        rho_mob_upwinded = self.var_upwinded_interfaces(
            interfaces,
            rho_mob,
            discr,
        )
        flux = mortar * rho_mob_upwinded

        flux.set_name("interface_fluid_mass_flux_phase_0")
        return flux

    def interface_fluid_mass_flux_phase_1(
        self, mortar, interfaces: list[pp.MortarGrid], phase, flux_array_key
    ) -> pp.ad.Operator:
        """TODO: redundant input"""
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_ppu_discretization(interfaces, flux_array_key)
        rho_mob = phase.mass_density_operator(
            subdomains, self.pressure
        ) * pp.tobedefined.mobility.mobility_operator(
            subdomains, phase.saturation_operator, self.dynamic_viscosity
        )

        rho_mob_upwinded = self.var_upwinded_interfaces(
            interfaces,
            rho_mob,
            discr,
        )
        flux = mortar * rho_mob_upwinded

        flux.set_name("interface_fluid_mass_flux_phase_1")
        return flux


    def ppu_discretization(
        self, subdomains: list[pp.Grid], flux_array_key
    ) -> pp.ad.UpwindAd:
        """
        flux_array_key =  either darcy_flux_phase_0 or darcy_flux_phase_1
        """
        return pp.ad.UpwindAd(self.ppu_keyword, subdomains, flux_array_key)

    def interface_ppu_discretization(
        self, interfaces: list[pp.MortarGrid], flux_array_key
    ) -> pp.ad.UpwindCouplingAd:
        """
        flux_array_key = either interface_mortar_flux_phase_0 or interface_mortar_flux_phase_1
        """
        return pp.ad.UpwindCouplingAd(self.ppu_keyword, interfaces, flux_array_key)

    def var_upwinded_interfaces(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """same of constitutive_law.interface_advective_flux without interface_darcy_flux"""
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        trace = pp.ad.Trace(subdomains)

        var_upwinded: pp.ad.Operator = (
            discr.upwind_primary
            @ mortar_projection.primary_to_mortar_avg
            @ trace.trace
            @ advected_entity
            + discr.upwind_secondary
            @ mortar_projection.secondary_to_mortar_avg
            @ advected_entity
        )
        return var_upwinded

    def pressure_trace_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        pressure_trace = (
            discr.bound_pressure_cell @ self.pressure(subdomains)
            + discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_0(interfaces)
            )
            + discr.bound_pressure_face @ self.bc_values_darcy(subdomains)
            + discr.vector_source @ self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace

    def pressure_trace_phase_1(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        pressure_trace = (
            discr.bound_pressure_cell @ self.pressure(subdomains)
            + discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_1(interfaces)
            )
            + discr.bound_pressure_face @ self.bc_values_darcy(subdomains)
            + discr.vector_source @ self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace

    def darcy_flux_phase_0(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """
        - need only darcy flux at internal boundaries but i compute it everywhere. sorry.
        - need only the direction, I never use its norm
        - TODO: redundant input
        - pay attention to its definition: mob rho is not included

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        mass_density = phase.mass_density_operator(subdomains, self.pressure)

        # OLD:
        # expansion_matrix = sp.sparse.eye(
        #     subdomains[0].num_cells
        # )  # hope there is only one grid
        # mat_list = [expansion_matrix] * self.nd
        # expansion_matrix = pp.ad.SparseArray(sp.sparse.vstack(mat_list))
        # rho = expansion_matrix @ mass_density

        # NEW:
        rho = self.vector_expansion(subdomains) @ mass_density # there are a lot of extra computation in vector_expansion
        
        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(subdomains)
            + discr.bound_flux
            @ (
                self.bc_values_darcy(subdomains)
                + projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_0(interfaces)
            )
            + discr.vector_source
            @ (
                rho  ### VECTOR source, not scalar source
                * self.vector_source(subdomains, material="fluid")
            )  # NOTE: mass_density not upwinded
        )

        flux.set_name("darcy_flux_phase_0")


        # pdb.set_trace()
        # flux.evaluate(self.equation_system).val

        return flux

    def darcy_flux_phase_1(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )
        
        mass_density = phase.mass_density_operator(subdomains, self.pressure)
        rho = self.vector_expansion(subdomains) @ mass_density

        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(subdomains)
            + discr.bound_flux
            @ (
                self.bc_values_darcy(subdomains)
                + projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_1(interfaces)
            )
            + discr.vector_source
            @ (
                rho * self.vector_source(subdomains, material="fluid")
            )  # NOTE: mass_density not upwinded
        )
        flux.set_name("darcy_flux_phase_1")
        return flux

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """ """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source(
        self,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: str,  # phase,
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects"""

        # val = self.fluid.convert_units(0, "m*s^-2")
        val = self.gravity_value
        size = int(np.sum([g.num_cells for g in grids]) * self.nd)

        # rho = phase.mass_density() ### just in case...
        source = pp.wrap_as_ad_array(val, size=size, name="gravity_vector_source")

        return source

    def interface_vector_source(
        self, interfaces: list[pp.MortarGrid], material: str
    ) -> pp.ad.Operator:
        """ """
        normals = self.outwards_internal_boundary_normals(
            interfaces, unitary=True  # type: ignore[call-arg]
        )
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )
        vector_source = projection.secondary_to_mortar_avg @ self.vector_source(
            subdomain_neighbors,
            material,
            # self.mixture.get_phase(0),
        )
        normals_times_source = normals * vector_source
        nd_to_scalar_sum = pp.ad.sum_operator_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]  # type: ignore[call-arg]
        )
        dot_product = nd_to_scalar_sum @ normals_times_source
        dot_product.set_name("Interface vector source term")
        return dot_product
    
    def vector_expansion(self, subdomains):
        """used only for exanding the densisty in darcy fluxes"""

        # I have to make the mass density a vector, from [rho_h, rho_l] to [rho_h, rho_h, rho_h, rho_l, rho_l, rho_l]
        num_cells_tot = 0
        for sd in subdomains:
            num_cells_tot += sd.num_cells

        i = 0
        offset = 0
        expansion_list = [None] * self.mdg.num_subdomains()
        for sd in subdomains:
            shape = (sd.num_cells, num_cells_tot)
            sd_expansion = sps.diags([1]*sd.num_cells, offsets=offset, shape=shape)
            sd_expansion_vect  = [ sd_expansion ] * self.nd
            expansion_list[i] = sps.vstack(sd_expansion_vect)
            i += 1 # TODO: I don't like this solution, improve it
            offset += sd.num_cells # TODO: I don't like this solution, improve it

        expansion = sps.vstack(expansion_list) 
        expansion_operator = pp.ad.SparseArray(expansion)
        
        self.vector_expansion_operator = expansion_operator

        # np.set_printoptions(linewidth=700, threshold=sys.maxsize)
        # print(expansion.todense())
        # pdb.set_trace()
        return expansion_operator


class ConstitutiveLawPressureMass(
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.ConstantPorosity,
    # pp.constitutive_laws.ConstantPermeability, # TODO: there is a lot to change...
    # pp.constitutive_laws.ConstantViscosity, # TODO: there is a lot to change...
):
    pass


class BoundaryConditionsPressureMass:
    def bc_values(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """
        it's quite useless now, but i like how it is written
        """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            boundary_faces = self.domain_boundary_sides(sd).all_bf

            vals = np.zeros(sd.num_faces)
            vals[
                boundary_faces
            ] = 0  # change this if you want different bc val # pay attention, there is a mistake in calance eq, take a look...
            bc_values.append(vals)

        bc_values_array: pp.ad.DenseArray = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_flux"
        )

        return bc_values_array


class SolutionStrategyPressureMass(pp.SolutionStrategy):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self.pressure_variable: str = "pressure"
        self.saturation_variable = "saturation"
        self.interface_mortar_flux_phase_0_variable: str = (
            "interface_mortar_flux_phase_0"
        )
        self.interface_mortar_flux_phase_1_variable: str = (
            "interface_mortar_flux_phase_1"
        )
        self.darcy_keyword: str = "flow"
        self.ppu_keyword: str = "ppu"

    def prepare_simulation(self) -> None:
        """ """

        self.set_geometry()

        # pp.plot_grid(self.mdg, info='c', alpha=0)
        # pdb.set_trace()

        self.initialize_data_saving()

        self.set_materials()
        self.set_equation_system_manager()

        self.add_equation_system_to_phases()
        self.mixture.apply_constraint(self.ell)

        self.create_variables()

        self.initial_condition()

        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()

        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        self.save_data_time_step()  # it is in pp.viz.data_saving_model_mixin, not copied here

        # added:
        self.computations_for_hu()

    # methods called in prepare_simulation: ------------------------------------------------------

    def set_equation_system_manager(self) -> None:
        """ """
        if not hasattr(self, "equation_system"):
            self.equation_system = pp.ad.EquationSystem(self.mdg)

    def add_equation_system_to_phases(self):
        """this is probably the wrong place"""
        for phase in self.mixture.phases:
            phase.equation_system = self.equation_system

    def set_materials(self):
        """ """
        constants = {}

        constants.update(self.params.get("material_constants", {}))

        for name, const in constants.items():
            assert isinstance(const, pp.models.material_constants.MaterialConstants)
            const.set_units(self.units)
            setattr(self, name, const)

    def reset_state_from_file(self) -> None:
        """ """
        if self.restart_options.get("restart", False):
            if self.restart_options.get("pvd_file", None) is not None:
                pvd_file = self.restart_options["pvd_file"]
                is_mdg_pvd = self.restart_options.get("is_mdg_pvd", False)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_pvd(
                    pvd_file,
                    is_mdg_pvd,
                    times_file,
                )
            else:
                vtu_files = self.restart_options["vtu_files"]
                time_index = self.restart_options.get("time_index", -1)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_vtu(
                    vtu_files,
                    time_index,
                    times_file,
                )
            vals = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(
                vals, iterate_index=0, time_step_index=0
            )

    def initial_condition(self) -> None:
        """ """
        val = np.zeros(self.equation_system.num_dofs())
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        for sd in self.mdg.subdomains():
            # if sd.dim == self.mdg.dim_max():  # thers is a is_frac attribute somewhere
            saturation_variable = (
                # self.mixture.mixture_for_subdomain(self.equation_system, sd)
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(self.ell)
                .saturation_operator([sd])
            )

            # saturation_variable = self.equation_system.md_variable(
            #     "saturation", [sd]
            # )

            saturation_values = 0*np.ones(sd.num_cells)
            y_max = self.size  # TODO: not 100% sure size = y_max
            saturation_values[
                np.where(sd.cell_centers[1] >= y_max / 2)
            ] = 1.0  # TODO: HARDCODED for 2D

            if sd.dim == 1:
                saturation_values = 0.5*np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,  ### not totally clear, aqp you have to change both timestep values and iterate
            )  # TODO: i don't understand, this stores the value in data[pp.TIME_STEP_SOLUTION]["saturation"] without modifying the value of saturation

            pressure_variable = self.pressure([sd])

            pressure_values = (
                2 - 1 * sd.cell_centers[1] / y_max
            )  # TODO: hardcoded for 2D

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )

        for sd, data in self.mdg.subdomains(return_data=True):
            # aperture = np.ones(sd.num_cells) * np.power(
            #     7e-2, self.mdg.dim_max() - sd.dim
            # )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword,
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
                    "darcy_flux_phase_1": np.zeros(sd.num_faces),
                    # "aperture": aperture,
                },
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword,
                {
                    "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                    "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )

    def set_discretization_parameters(self) -> None:
        """ """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy(
                        sd
                    ),  ### I don't understand this... let's fix it from errors # tpfa search for "bc" in the data discionary, .....
                    "second_order_tensor": self.permeability_tensor(sd),
                    "ambient_dimension": self.nd,
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword,
                {
                    "bc": self.bc_type_darcy(sd),
                },
            )

        for intf, intf_data in self.mdg.interfaces(return_data=True, codim=1):
            pp.initialize_data(
                intf,
                intf_data,
                self.darcy_keyword,
                {
                    "ambient_dimension": self.nd,
                },
            )

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """required by tpfa, upwind
        TODO: bc_type_darcy is a wrong name... change it
        """

        boundary_faces = self.domain_boundary_sides(sd).all_bf

        return pp.BoundaryCondition(
            sd, boundary_faces, "neu"
        )  # definig "neu" should be useless, improve it...

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.wrap_as_ad_array(0, num_faces, "bc_values_darcy")

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """non centra un cazzo qui, copio da constitutive_laws..."""
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_ad_array(
            self.solid.permeability(), size, name="permeability"
        )
        return permeability

    def permeability_tensor(self, sd: pp.Grid) -> pp.SecondOrderTensor:
        """non centra un cazzo qui, copio da fluid_mass_balance..."""
        permeability_ad = self.specific_volume([sd]) * self.permeability([sd])
        try:
            permeability = permeability_ad.evaluate(self.equation_system)
        except KeyError:
            volume = self.specific_volume([sd]).evaluate(self.equation_system)
            permeability = self.solid.permeability() * np.ones(sd.num_cells) * volume
        if isinstance(permeability, pp.ad.AdArray):
            permeability = permeability.val
        return pp.SecondOrderTensor(permeability)

    def discretize(self) -> None:
        """ """
        self.equation_system.discretize()

    def _initialize_linear_solver(self) -> None:
        """ """
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in ["scipy_sparse", "pypardiso", "umfpack"]:
            raise ValueError(f"Unknown linear solver {solver}")

    def computations_for_hu(self):
        """
        this will change a lot so it is useless to split this function
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            data[
                    "for_hu"
                ] = {}  # this is the first call, so i need to create the dictionary first

            data["for_hu"]["gravity_value"] = self.gravity_value
            data["for_hu"]["ell"] = self.ell

            if sd.dim == 0:
                data["for_hu"]["left_restriction"] = None
                data["for_hu"]["right_restriction"] = None
                data["for_hu"]["expansion_matrix"] = None
                data["for_hu"]["transmissibility_internal_tpfa"] = None
                data["for_hu"]["dim_max"] = None

            else:
                (
                    left_restriction,
                    right_restriction,
                ) = pp.numerics.fv.hybrid_upwind_utils.restriction_matrices_left_right(sd)
                data["for_hu"]["left_restriction"] = left_restriction
                data["for_hu"]["right_restriction"] = right_restriction
                data["for_hu"][
                    "expansion_matrix"
                ] = pp.numerics.fv.hybrid_upwind_utils.expansion_matrix(sd)

                pp.numerics.fv.hybrid_upwind_utils.compute_transmissibility_tpfa(
                    sd, data, keyword="flow"
                )

                (
                    _,
                    transmissibility_internal,
                ) = pp.numerics.fv.hybrid_upwind_utils.get_transmissibility_tpfa(
                    sd, data, keyword="flow"
                )
                data["for_hu"]["transmissibility_internal_tpfa"] = transmissibility_internal
                data["for_hu"]["dim_max"] = self.mdg.dim_max()

    # other methods not called in prepared_simulation: -------------------------------------------------

    def set_nonlinear_discretizations(self) -> None:
        """ """
        super().set_nonlinear_discretizations()  # TODO: EH? It does literally nothing...!

        # self.add_nonlinear_discretization(
        #     self.interface_ppu_discretization(self.mdg.interfaces()).flux,
        # )

    def before_nonlinear_iteration(self):
        """ """
        self.mixture.apply_constraint(self.ell)

        for sd, data in self.mdg.subdomains(return_data=True):
            pressure_adarray = self.pressure([sd]).evaluate(self.equation_system)
            left_restriction = data["for_hu"]["left_restriction"]
            right_restriction = data["for_hu"][
                "right_restriction"
            ]  # created in prepare_simulation
            transmissibility_internal_tpfa = data["for_hu"][
                "transmissibility_internal_tpfa"
            ]  # idem...
            ad = True
            dynamic_viscosity = 1  # Sorry
            dim_max = data["for_hu"]["dim_max"]
            total_flux_internal = (
                pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
                    sd,
                    # self.mixture.mixture_for_subdomain(self.equation_system, sd),
                    self.mixture.mixture_for_subdomain(sd),
                    pressure_adarray,
                    self.gravity_value,
                    left_restriction,
                    right_restriction,
                    transmissibility_internal_tpfa,
                    ad,
                    dynamic_viscosity,
                    dim_max,
                )
            )
            data["for_hu"]["total_flux_internal"] = (
                total_flux_internal[0] + total_flux_internal[1]
            )
            
            data["for_hu"]["pressure_AdArray"] = self.pressure([sd]).evaluate(
                self.equation_system
            )

            vals = (
                self.darcy_flux_phase_0([sd], self.mixture.get_phase(0))
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword].update({"darcy_flux_phase_0": vals})

            vals = (
                self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword].update({"darcy_flux_phase_1": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword].update(
                {"interface_mortar_flux_phase_0": vals}
            )

            vals = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword].update(
                {"interface_mortar_flux_phase_1": vals}
            )

        # self.assemble_linear_system()
        # A, b = self.linear_system
        # # A = A.todense()

        # np.set_printoptions(precision=5, threshold=sys.maxsize, linewidth=300)
        # print("\n A = ", A)
        # print("\n b = ", b)
        # pdb.set_trace()

        # for sd in self.mdg.subdomains():
        #     pp.plot_grid(sd, self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation.val, info='c', alpha=0.5)

        # super().before_nonlinear_iteration() # attento... se ineriti questa classe crei un loop di merda. Ho copiato le funzioni qui sotto
        self.set_discretization_parameters()
        self.rediscretize()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """ """
        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()

        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )
      
        # # added:
        # print("self.time_manager.time = ", self.time_manager.time)
        # if np.isclose(np.mod(self.time_manager.time, 0.002), 0, rtol=0, atol=1e-5):
        #     for sd, _ in self.mdg.subdomains(return_data=True):
        #         gigi = (
        #             self.mixture.mixture_for_subdomain(sd)
        #             .get_phase(0)
        #             .saturation
        #         )

        #         mario = (
        #             self.mixture.mixture_for_subdomain(sd)
        #             .get_phase(1)
        #             .saturation
        #         )

        #         # sss = self.equation_system.md_variable(
        #         #     "saturation", self.mdg.subdomains()
        #         # ).evaluate(self.equation_system)

        #         ppp = self.equation_system.md_variable(
        #             "pressure", self.mdg.subdomains()
        #         ).evaluate(self.equation_system)

        #         print("saturation = ", mario.val)
        #         print("pressure = ", ppp.val)

        #         pp.plot_grid(
        #             sd,
        #             gigi.val,
        #             alpha=0.5,
        #             info="c",
        #             title="saturation " + str(self.ell),
        #         )
        #         # pp.plot_grid(
        #         #     sd,
        #         #     sss.val,
        #         #     alpha=0.5,
        #         #     info="c",
        #         #     title="saturation " + str(self.ell),
        #         # )
        #         pp.plot_grid(sd, ppp.val, alpha=0.5, info="c", title="pressure")

    def eb_after_timestep(self):
        """ """
        print("self.time_manager.time = ", self.time_manager.time)
        if np.isclose(np.mod(self.time_manager.time, 0.01), 0, rtol=0, atol=1e-5) or self.time_manager.time == self.time_manager.dt:
            for sd, _ in self.mdg.subdomains(return_data=True):
                gigi = (
                    # self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    self.mixture.mixture_for_subdomain(sd)
                    .get_phase(0)
                    .saturation
                )

                ppp = self.equation_system.md_variable("pressure", [sd]).evaluate(
                    self.equation_system
                )

                print("saturation = ", gigi.val)
                print("pressure = ", ppp.val)

                if sd.dim > 0: # otherwise plot_grid gets mad
                    pp.plot_grid(
                        sd,
                        gigi.val,
                        alpha=0.5,
                        info="",
                        title="saturation " + str(self.ell),
                    )
                    pp.plot_grid(sd, ppp.val, alpha=0.5, info="", title="pressure")


class MyModelGeometry(pp.ModelGeometry):
    def set_geometry(self) -> None:
        """ """

        self.set_domain()
        self.set_fractures()

        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        self.mdg = pp.create_mdg(
            "simplex",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, pp.FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)

    def set_domain(self) -> None:
        """ """
        self.size = 1 / self.units.m

        # structired square:
        # self._domain = pp.applications.md_grids.domains.nd_cube_domain(2, self.size)

        # unstructred unit square:
        bounding_box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1} #, "zmin": 0, "zmax": 1}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[0.2, 0.7], [0.5, 0.5]]))
        # frac2 = pp.LineFracture(np.array([[0.2, 0.7], [0.2, 0.2]]))
        frac3 = pp.LineFracture(np.array([[0.5, 0.5], [0.2, 0.7]]))

        # frac1 = pp.PlaneFracture(np.array([[0.2, 0.7, 0.7, 0.2],[0.2, 0.2, 0.8, 0.8],[0.5, 0.5, 0.5, 0.5]]))
        self._fractures: list = [frac1, frac3]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.1 / self.units.m,
            "cell_size_fracture": 0.05 / self.units.m,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    PrimaryVariables,
    Equations,
    ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMass,
    SolutionStrategyPressureMass,
    MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":

    class FinalModel(PartialFinalModel):  # I'm sorry...
        def __init__(self, mixture, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0  # 0 = wetting, 1 = non-wetting
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
        dt_init=1e-3,
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
