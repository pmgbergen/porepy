import numpy as np
import scipy as sp
import scipy.sparse as sps
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import copy

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
this was test_hu_model.py, renamed as /models/two_phase.py on 13/11/23


- is it ok to consider mu (dynamic viscosity, mu = nu/rho) constant? 
- inside var_upwinded_interfaces there should be ..._int not avg. Doesn't matter as long as the grids are conforming.
"""




class ZeroOperator(pp.ad.operator_functions.AbstractFunction):
    """ 
    I need it as trick to kill the accumulation of only the fracture
    """

    def __init__(
        self,
        func: Callable,
        subdomains,
        name: str = "vzevo opevatov",
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False  

        self.subdomains = subdomains

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        return np.concatenate([
           (sd.dim-1)*np.ones(sd.num_cells) for sd in self.subdomains
             ])
            
    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """

        dof_tot = 0
        for sd in self.subdomains:
            dof_tot += 2*sd.num_cells # HARDCODED for p and S

        dof_tot += 4 # HARCODED... interfaces dofs

        jac_list = []
        for sd in self.subdomains:
            jac_list.append( sp.sparse.csr_matrix( (np.zeros(sd.num_cells), (np.arange(sd.num_cells),np.arange(sd.num_cells))), shape=[sd.num_cells, dof_tot] ) )

        return sp.sparse.vstack(jac_list)



#############################################################################################################################################################################



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
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        dim_max = data["for_hu"]["dim_max"]
        mobility = data["for_hu"]["mobility"] # sorry
        relative_permeability = data["for_hu"]["relative_permeability"] # sorry

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
            mobility,
            relative_permeability,
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
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        mobility = data["for_hu"]["mobility"]

        return [
            ell,
            pressure,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
            dynamic_viscosity,
            mobility,
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
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        dim_max = data["for_hu"]["dim_max"]
        mobility = data["for_hu"]["mobility"]

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
            mobility,
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

        if self.mdg.num_subdomains() > 1: # pp... why do try to set morart eq even though there arent fracts...
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

        mass_density = self.porosity(subdomains) * (
            mass_density_phase_0 + mass_density_phase_1
        ) # pp.ad.Scalar(0)*

        fake_input = self.pressure(subdomains)

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

        flux_tot = rho_total_flux_operator(fake_input)

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        neu_boundary_matrix = discr.bound_transport_neu

        flux_intf_phase_0 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        # discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        # discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        # discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")

         
        flux = ( flux_tot - flux_intf_phase_0 - flux_intf_phase_1
                - neu_boundary_matrix @ self.bc_neu_phase_0(subdomains)
                - neu_boundary_matrix @ self.bc_neu_phase_1(subdomains) ) 

        # flux = ( flux_tot - flux_intf_phase_0 - flux_intf_phase_1
        #         - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains)
        #         - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains) ) 


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

        # zevo = ZeroOperator(pp.rho_flux_V, subdomains)
        fake_input = self.pressure(subdomains)

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

        flux_V_G = (
            rho_V_operator(fake_input)
            + rho_G_operator(fake_input)
        )

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        neu_boundary_matrix = discr.bound_transport_neu
        flux_intf_phase_0 = (
            neu_boundary_matrix  # -1,0,1 matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, I need to test it

        # discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        # discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        # discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        
        if (
            self.ell == 0
        ):  # TODO: move the flux computation inside if (which btw could be removed) after all the bugs are fixed
            flux = flux_V_G - flux_intf_phase_0 - neu_boundary_matrix @ self.bc_neu_phase_0(subdomains) 
            # flux = self.specific_volume(subdomains) * flux_V_G - flux_intf_phase_0 - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains) 

        else:  # self.ell == 1
            flux = flux_V_G - flux_intf_phase_1 - neu_boundary_matrix @ self.bc_neu_phase_1(subdomains)
            # flux = self.specific_volume(subdomains) * flux_V_G - flux_intf_phase_1 - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains)


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

    def eq_fcn_mortar_phase_0(self, interfaces: list[pp.MortarGrid]):
        """definition of mortar flux
        - mortar flux is integrated mortar flux
        - mobility, Kr(s)/mu, is not included into this definition of mortar flux
        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1  # type: ignore[call-arg]
        ) # surface in 2D, length in 1D, the thickness isn't included

        trace = pp.ad.Trace(subdomains, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(
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

        # anna = specific_volume_intf.evaluate(self.equation_system)
        # pdb.set_trace()

        density_upwinded = self.var_upwinded_interfaces(
            interfaces,
            self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure),
            discr,
        )

        kn = pp.ad.Scalar(self.solid.normal_permeability()) * normal_gradient  # this is 2/eps

        # print("self.solid.normal_permeability() = ", self.solid.normal_permeability())
        # lara = kn.evaluate(self.equation_system)
        # pdb.set_trace()

        delta_p = pressure_h - pressure_l
        g_term = self.interface_vector_source(
                        interfaces, material="fluid"
                    )  # gravity


        eq = self.interface_mortar_flux_phase_0(interfaces) - (
            cell_volumes
            * (
                kn
                * specific_volume_intf
                * (delta_p
                    - density_upwinded
                    / normal_gradient  # Why is the density in fluid_mass_balance not here?
                    * g_term
                )
            )
        )

        eq.set_name("interface_darcy_flux_equation_phase_0")
        return (eq, kn, pressure_h, pressure_l, density_upwinded, g_term)

    def interface_mortar_flux_equation_phase_0(self, interfaces: list[pp.MortarGrid]):
        eq, _, _, _, _, _ = self.eq_fcn_mortar_phase_0(interfaces)
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
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(
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
        ) * self.mobility_operator(
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
        ) * self.mobility_operator(
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
        # print('\n inside interface_ppu_discretization: ------------------------')
        # print("flux_array_key = ", flux_array_key)
        # print("---------------------------\n")
        aaa = pp.ad.UpwindCouplingAd(self.ppu_keyword + "_" + flux_array_key, interfaces, flux_array_key)
        # pdb.set_trace()
        return aaa

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
            @ mortar_projection.primary_to_mortar_avg # should't it be int?
            @ trace.trace
            @ advected_entity
            + discr.upwind_secondary
            @ mortar_projection.secondary_to_mortar_avg # should't it be int?
            @ advected_entity
        )
        
        return var_upwinded

    def eq_fcn_pressure_trace(self, subdomains: list[pp.Grid]):
        """ 
        ref F17 f3 ->->
        """
        # old version:
        # interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        # projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        # discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
        #     subdomains
        # )

        # mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure)
        # mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure)

        # rho_mob_phase_0 = mass_density_phase_0 * self.mobility_operator(subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)
        # rho_mob_phase_1 = mass_density_phase_1 * self.mobility_operator(subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)
        # # without upwind:
        # # TODO

        # # with upwind:
        # epsilon = 1e-10 # avoid division by zero # TODO: NO... they shouldnt be zero
        # discr_up_0 = self.interface_ppu_discretization(interfaces, "interface_mortar_flux_phase_0")        
        # rho_mob_upwinded_phase_0 = self.var_upwinded_interfaces(interfaces, rho_mob_phase_0, discr_up_0)

        # discr_up_1 = self.interface_ppu_discretization(interfaces, "interface_mortar_flux_phase_1")
        # rho_mob_upwinded_phase_1 = self.var_upwinded_interfaces(interfaces, rho_mob_phase_1, discr_up_1)

        # pressure_cell = ( discr.bound_pressure_cell # 0, 1 matrix ### I dont like it, Id rahter see mortart to primary here, not outside this funcion, for ex. the non zero els of pressure cell are too many and they will be kelled after
        #     @ self.pressure(subdomains) )

        # correction = (
        #     discr.bound_pressure_face # not 0, 1 matrix, this is -1/half_transmissibilities
        #     @ (
        #          projection.mortar_to_primary_int # 0, 1 matrix 
        #         @ ((
        #         self.interface_fluid_mass_flux_phase_0(self.interface_mortar_flux_phase_0(interfaces), interfaces, self.mixture.get_phase(0), "interface_mortar_flux_phase_0") +
        #         self.interface_fluid_mass_flux_phase_1(self.interface_mortar_flux_phase_1(interfaces), interfaces, self.mixture.get_phase(1), "interface_mortar_flux_phase_1")
        #         ) / (rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1 + pp.ad.Scalar(epsilon)))
                
        #         + self.bc_values_darcy(subdomains) # for me they are always zero 
            
        #         + ( (projection.mortar_to_primary_int @ rho_mob_upwinded_phase_0) 
        #         * ( discr.vector_source # Starnoni included
        #         @ ((self.vector_expansion(subdomains) @ mass_density_phase_0)
        #            * self.vector_source(subdomains, material="fluid")) ) ) / (projection.mortar_to_primary_int @ rho_mob_upwinded_phase_0 + projection.mortar_to_primary_int @ rho_mob_upwinded_phase_1 + pp.ad.Scalar(epsilon))

        #         + ( (projection.mortar_to_primary_int @ rho_mob_upwinded_phase_1)
        #         * ( discr.vector_source # Starnoni included
        #         @ ((self.vector_expansion(subdomains) @ mass_density_phase_1)
        #            * self.vector_source(subdomains, material="fluid")) ) ) / (projection.mortar_to_primary_int @ rho_mob_upwinded_phase_0 + projection.mortar_to_primary_int @ rho_mob_upwinded_phase_1 + pp.ad.Scalar(epsilon))
        #     )
        # )



        # new version:
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(subdomains) # THIS IS SUSPICIOUS...

        pressure_cell = ( discr.bound_pressure_cell # 0, 1 matrix 
            @ self.pressure(subdomains) )
        
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure)
        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure)

        rho_mob_phase_0 = mass_density_phase_0 * self.mobility_operator(subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)
        rho_mob_rho_phase_0 = mass_density_phase_0 * self.mobility_operator(subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity) * mass_density_phase_0
        rho_mob_phase_1 = mass_density_phase_1 * self.mobility_operator(subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)
        rho_mob_rho_phase_1 = mass_density_phase_1 * self.mobility_operator(subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity) * mass_density_phase_1


        discr_up_0 = self.interface_ppu_discretization(interfaces, "interface_mortar_flux_phase_0")   
        rho_mob_upwinded_phase_0 = self.var_upwinded_interfaces(interfaces, rho_mob_phase_0, discr_up_0)
        rho_mob_rho_upwinded_phase_0 = self.var_upwinded_interfaces(interfaces, rho_mob_rho_phase_0, discr_up_0)

        discr_up_1 = self.interface_ppu_discretization(interfaces, "interface_mortar_flux_phase_1")
        rho_mob_upwinded_phase_1 = self.var_upwinded_interfaces(interfaces, rho_mob_phase_1, discr_up_1)
        rho_mob_rho_upwinded_phase_1 = self.var_upwinded_interfaces(interfaces, rho_mob_rho_phase_1, discr_up_1)

        f_0 = self.interface_fluid_mass_flux_phase_0(self.interface_mortar_flux_phase_0(interfaces), interfaces, self.mixture.get_phase(0), "interface_mortar_flux_phase_0") 
        f_1 = self.interface_fluid_mass_flux_phase_1(self.interface_mortar_flux_phase_1(interfaces), interfaces, self.mixture.get_phase(1), "interface_mortar_flux_phase_1") 

        correction =  (
           discr.bound_pressure_face @ ( projection.mortar_to_primary_int @ ( ( f_0 + f_1 ) / ( rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1 ) ) ) 
            - discr.bound_pressure_face @ ( projection.mortar_to_primary_int @ ( ( ( rho_mob_rho_upwinded_phase_0 + rho_mob_rho_upwinded_phase_1 ) / ( rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1 ) ) ) * ( discr.vector_source @ self.vector_source(subdomains) ) )  
            # - projection.mortar_to_primary_int @ ( ( ( rho_mob_rho_upwinded_phase_0 + rho_mob_rho_upwinded_phase_1 ) / ( rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1 ) ) ) * ( XXX @ self.vector_source(subdomains_vector_source) )  # third version, ref F19f2, subdomains_vector_source are fractures domains, XXX 
            )        

        pressure_trace = pressure_cell + correction 
        return pressure_trace, pressure_cell, correction

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        pressure_trace, _, _ = self.eq_fcn_pressure_trace(subdomains)
        return pressure_trace


    def darcy_flux_phase_0(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """
        useless, need it only to make pp happy. It is used only in ppu_discretization from which I need only the Neumann boundary matrix, which doesnt depend on any flow direction...
        """

        # # actual Darcy flux computation:
        # interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        # projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        # discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
        #     subdomains
        # )

        # mass_density = phase.mass_density_operator(subdomains, self.pressure)

        # rho = self.vector_expansion(subdomains) @ mass_density # there are a lot of extra computations in vector_expansion
        
        # flux: pp.ad.Operator = (
        #     discr.flux @ self.pressure(subdomains)
        #     + discr.bound_flux
        #     @ (
        #         self.bc_values_darcy(subdomains)
        #         + projection.mortar_to_primary_int
        #         @ self.interface_mortar_flux_phase_0(interfaces)
        #     )
        #     + discr.vector_source
        #     @ (
        #         rho  ### VECTOR source, not scalar source
        #         * self.vector_source(subdomains, material="fluid")
        #     )  # NOTE: mass_density not upwinded
        # )

        # fake Darcy flux, only to make pp happy
        num_faces_tot = 0
        for sd in subdomains: num_faces_tot += sd.num_faces 

        flux = pp.ad.DenseArray(np.nan*np.ones(num_faces_tot))
        flux.set_name("darcy_flux_phase_0")

        return flux

    def darcy_flux_phase_1(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """
        see darcy_flux_phase_0 and comments therein
        """

        # interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        # projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        # discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
        #     subdomains
        # )
        
        # mass_density = phase.mass_density_operator(subdomains, self.pressure)
        # rho = self.vector_expansion(subdomains) @ mass_density

        # flux: pp.ad.Operator = (
        #     discr.flux @ self.pressure(subdomains)
        #     + discr.bound_flux
        #     @ (
        #         self.bc_values_darcy(subdomains)
        #         + projection.mortar_to_primary_int
        #         @ self.interface_mortar_flux_phase_1(interfaces)
        #     )
        #     + discr.vector_source
        #     @ (
        #         rho * self.vector_source(subdomains, material="fluid")
        #     )  # NOTE: mass_density not upwinded
        # )

        # fake Darcy flux, only to make pp happy
        num_faces_tot = 0
        for sd in subdomains: num_faces_tot += sd.num_faces 

        flux = pp.ad.DenseArray(np.nan*np.ones(num_faces_tot))
        flux.set_name("darcy_flux_phase_1")

        return flux

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """ """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source(
        self,
        subdomains: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: str = "useless",  # phase,
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects"""

        # val = self.gravity_value
        # size = int(np.sum([g.num_cells for g in grids]) * self.nd)
        # source = pp.wrap_as_ad_array(val, size=size, name="gravity_vector_source")

        source = []
 
        # # order 1. [sd_1_[x,y,z], sd_2_[x,y,z], ...] # WRONG
        # for sd in subdomains:
        #     source_sd = [np.zeros(sd.num_cells)] * self.nd
        #     source_sd[self.nd-1] = self.gravity_value * np.ones(sd.num_cells) # gravity acts on the last coordinate. This is an implicit notation in HU and pp (not 100% sure about the latter)
        #     source.append(np.concatenate(source_sd))


        # # order 2. [all x, all y, all z] # WRONG
        # num_cells_tot = 0
        # for sd in subdomains:
        #     num_cells_tot += sd.num_cells # sorry...
        
        # for i in np.arange(self.nd-1):
        #     source.append(np.zeros(num_cells_tot))
        # source.append(self.gravity_value*np.ones(num_cells_tot))

        # # order 3. x y z cell1, x y z cell2, .... 
        source_cell = np.zeros(self.nd)
        source_cell[self.nd-1] = self.gravity_value 

        num_cells_tot = 0
        for sd in subdomains:
            num_cells_tot += sd.num_cells # sorry..

        source = [source_cell] * num_cells_tot

        try:
            source = pp.ad.DenseArray(np.concatenate(source))
        except:
            source = pp.ad.DenseArray(np.array(source)) # if frac = [], interface_vector_source calls this fcn with source = [], so np.concatenate gets mad.
            # I don't like this except because source = [] for a bug you dont see it... find a better solution
            print('FIX THIS...') # in pp code sometimes you see a "if len(grids) == 0: ..."
        source.set_name("vector source (gravity)")
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

        vector_source_all = self.vector_source(
            subdomain_neighbors,
            material,
        )
        vector_source = projection.secondary_to_mortar_avg @ vector_source_all
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

        return expansion_operator


class ConstitutiveLawPressureMass(
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.ConstantPorosity,
    # pp.constitutive_laws.ConstantPermeability, # TODO: there is a lot to change...
    # pp.constitutive_laws.ConstantViscosity, # TODO: there is a lot to change...
):
    pass


class BoundaryConditionsPressureMass:
    def bc_neu_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf
                
                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0]<(0+eps))[0] # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0]>(self.xmax-eps))[0] # add xmax as in the tests pls
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0.0 * sd.face_areas[left_boundary] # inlet
                vals[right_boundary] = 0*2.0 * sd.face_areas[right_boundary] # outlet
                bc_values.append(vals)
            
            else:
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
    
    def bc_neu_phase_1(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """
        """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf
                
                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0]<(0+eps))[0] # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0]>(self.xmax-eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0*1.0 * sd.face_areas[left_boundary] # inlet
                vals[right_boundary] = 0.0 * sd.face_areas[right_boundary] # outlet
                bc_values.append(vals)
            
            else:
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
        # pdb.

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

            saturation_values = 0.0*np.ones(sd.num_cells)
            # y_max = self.size  # TODO: not 100% sure size = y_max
            saturation_values[
                np.where(sd.cell_centers[1] >= self.ymax / 2)
            ] = 1.0  # TODO: HARDCODED for 2D

            if sd.dim == 1:
                saturation_values = 0.8*np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,  ### not totally clear, aqp you have to change both timestep values and iterate
            )  # TODO: i don't understand, this stores the value in data[pp.TIME_STEP_SOLUTION]["saturation"] without modifying the value of saturation

            pressure_variable = self.pressure([sd])

            pressure_values = (
                2 - 1 * sd.cell_centers[1] / self.ymax
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

            # pp.initialize_data(
            #     sd,
            #     data,
            #     self.ppu_keyword + "_" + "interface_mortar_flux_phase_1",
            #     {
            #         # "darcy_flux_phase_0": np.zeros(sd.num_faces),
            #         "darcy_flux_phase_1": np.zeros(sd.num_faces),
            #         # "aperture": aperture,
            #     },
            # )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_0",
                {
                    "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                    # "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )

            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_1",
                {
                    # "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                    "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )


    def set_discretization_parameters(self) -> None:
        """ """
        
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data( # initiale or update, bad name
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy(
                        sd
                    ),  ### I don't understand this... let's fix it from errors # tpfa search for "bc" in the data discionary, .....
                    "second_order_tensor": self.intrinsic_permeability_tensor(sd),
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

    def intrinsic_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """non centra un cazzo qui, copio da constitutive_laws..."""
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_ad_array(
            self.solid.permeability(), size, name="intrinsic_permeability"
        )
        return permeability

    def intrinsic_permeability_tensor(self, sd: pp.Grid) -> pp.SecondOrderTensor:
        """non centra un cazzo qui, copio da fluid_mass_balance..."""
        permeability_ad = self.specific_volume([sd]) * self.intrinsic_permeability([sd])
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

    def rediscretize(self) -> None:
        """TODO: Discretize only nonlinear terms"""
        # tic = time.time()
        # # Uniquify to save computational time, then discretize.
        # unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
        #     self.nonlinear_discretizations
        # ) # empty dictionary
        # pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)
        # logger.info(
        #     "Re-discretized nonlinear terms in {} seconds".format(time.time() - tic)
        # )
        self.discretize()

    def _initialize_linear_solver(self) -> None:
        """ """
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in ["scipy_sparse", "pypardiso", "umfpack"]:
            raise ValueError(f"Unknown linear solver {solver}")

    def computations_for_hu(self):
        """
        this will change a lot so it is useless to improve it right now
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            data[
                    "for_hu"
                ] = {}  # this is the first call, so i need to create the dictionary first

            data["for_hu"]["gravity_value"] = self.gravity_value
            data["for_hu"]["ell"] = self.ell
            data["for_hu"]["dynamic_viscosity"] = self.dynamic_viscosity

            if sd.dim == 0:
                data["for_hu"]["left_restriction"] = None
                data["for_hu"]["right_restriction"] = None
                data["for_hu"]["expansion_matrix"] = None
                data["for_hu"]["transmissibility_internal_tpfa"] = None
                data["for_hu"]["dim_max"] = None
                data["for_hu"]["mobility"] = None
                data["for_hu"]["relative_permeability"] = None

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
                data["for_hu"]["mobility"] = self.mobility
                data["for_hu"]["relative_permeability"] = self.relative_permeability

    # other methods not called in prepared_simulation: -------------------------------------------------

    # def set_nonlinear_discretizations(self) -> None:
    #     """ """
    #     super().set_nonlinear_discretizations()  # TODO: EH? It does literally nothing...!

    #     # self.add_nonlinear_discretization(
    #     #     self.interface_ppu_discretization(self.mdg.interfaces()).flux,
    #     # )

    def before_nonlinear_iteration(self):
        """ """

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
            dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
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
                    self.mobility,
                    self.relative_permeability,
                )
            )
            data["for_hu"]["total_flux_internal"] = (
                total_flux_internal[0] + total_flux_internal[1]
            )
            
            data["for_hu"]["pressure_AdArray"] = self.pressure([sd]).evaluate(self.equation_system) 


        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword + "_" + "interface_mortar_flux_phase_0"].update(
                {"interface_mortar_flux_phase_0": vals}
            )

            vals = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword + "_" + "interface_mortar_flux_phase_1"].update(
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
        
        # subdomains = self.mdg.subdomains()
        # interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])

        # discr = self.darcy_flux_discretization(self.mdg.subdomains())
        # projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
                                             
        # gigi = discr.bound_pressure_cell.evaluate(self.equation_system)
        # mario = discr.bound_pressure_face.evaluate(self.equation_system)
        # giulia = projection.mortar_to_primary_int.evaluate(self.equation_system)
        
        # pressure_trace, pressure_cell, correction = self.eq_fcn_pressure_trace(subdomains)
        # pressure_trace = pressure_trace.evaluate(self.equation_system)
        # pressure_cell = pressure_cell.evaluate(self.equation_system)
        # correction = correction.evaluate(self.equation_system)
        # actual_ph = (projection.primary_to_mortar_avg @ pressure_trace).evaluate(self.equation_system)

        # (eq, kn, pressure_h, pressure_l, g_term) = self.eq_fcn_mortar_phase_0(interfaces)
        # p_h = pressure_h.evaluate(self.equation_system).val
        
        # discr = self.darcy_flux_discretization(subdomains)
        # strega = discr.vector_source.evaluate(self.equation_system)
        # avvoltoio = self.vector_source(subdomains, material="fluid").evaluate(self.equation_system)
    
        # discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        # lara = discr_phase_0.bound_transport_neu.evaluate(self.equation_system)
        # pdb.set_trace()

        self.set_discretization_parameters()
        self.rediscretize() 

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """ """

        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()

        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )
      
        # added:
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
        
<


    def eb_after_timestep(self):
        """ """

        print("self.time_manager.time = ", self.time_manager.time)
        # if np.isclose(np.mod(self.time_manager.time, 0.5), 0, rtol=0, atol=1e-4) or self.time_manager.time == self.time_manager.dt:
            
            #  # open the fracture:
            # for sd in self.mdg.subdomains():
            #     if sd.dim == 2:
            #         internal_nodes_id = sd.get_internal_nodes()
            #         # only for this grid: "simplex";  "cell_size": 0.2 / self.units.m;      "cell_size_fracture": 0.1 / self.units.m;        [frac1]
            #         # print(internal_nodes_id)
            #         # sd.nodes[1][26] += 0.04
            #         # sd.nodes[1][27] -= 0.04
            #         # sd.nodes[1][28] += 0.04
            #         # sd.nodes[1][29] -= 0.04

            #         tmp_0 = np.zeros(sd.num_faces)
            #         tmp_0[sd.frac_pairs[0]] = 1                    

            #         tmp_1 = np.zeros(sd.num_faces)
            #         tmp_1[sd.frac_pairs[1]] = 1

            #         side_0_nodes = np.where((sd.face_nodes @ tmp_0)==2)[0]
            #         side_1_nodes = np.where((sd.face_nodes @ tmp_1)==2)[0]

            #         sd.nodes[1][side_0_nodes] += 0.01
            #         sd.nodes[1][side_1_nodes] -= 0.01

            # self.mdg.compute_geometry()
            # pp.plot_grid(self.mdg, info="", alpha=0.1)
            
            # for sd, _ in self.mdg.subdomains(return_data=True):
            #     gigi = (
            #         # self.mixture.mixture_for_subdomain(self.equation_system, sd)
            #         self.mixture.mixture_for_subdomain(sd)
            #         .get_phase(0)
            #         .saturation
            #     )
            #     gigi_1 = (self.mixture.mixture_for_subdomain(sd)
            #         .get_phase(1)
            #         .saturation)
        

            #     ppp = self.equation_system.md_variable("pressure", [sd]).evaluate(
            #         self.equation_system
            #     )

            #     print("saturation = ", gigi.val)
            #     print("pressure = ", ppp.val)
                
            #     interfaces = self.subdomains_to_interfaces(self.mdg.subdomains(), [1])
            #     (eq, kn, pressure_h, pressure_l, g_term) = self.eq_fcn_mortar_phase_0(interfaces)
            #     pressure_h = pressure_h.evaluate(self.equation_system).val
            #     pressure_l = pressure_l.evaluate(self.equation_system).val
            #     dp = pressure_h - pressure_l
            #     g_term = g_term.evaluate(self.equation_system)

                # if sd.dim == 2:
                #     mortar_projection = pp.ad.MortarProjections(self.mdg, [sd], interfaces, dim=1)
                #     M = mortar_projection.mortar_to_primary_avg.evaluate(self.equation_system)
                #     g_term_all = M @ g_term
                #     normals = self.outwards_internal_boundary_normals(interfaces, unitary=True).evaluate(self.equation_system) 
                #     normals = normals.reshape((int(np.shape(normals)[0]/2), 2)).T
                #     normals_all = sd.face_normals
                #     g_term_vect = normals_all
                #     g_term_vect[0] = 0
                #     g_term_vect[1] = (M @ normals[1]) * g_term_all # MAYBE YOU DON'T NEED M HERE, check the function outwards_internal...

                #     print("g_term = ", g_term)
                #     pp.plot_grid(sd, vector_value=0.1*g_term_vect, alpha=0)

                # if sd.dim > 0: # otherwise plot_grid gets mad
                #     pp.plot_grid(
                #         sd,
                #         gigi.val,
                #         alpha=0.5,
                #         info="",
                #         title="saturation " + str(self.ell),
                #     )
                #     pp.plot_grid(
                #         sd,
                #         gigi_1.val,
                #         alpha=0.5,
                #         info="",
                #         title="saturation 1",
                #     )
                #     pp.plot_grid(sd, ppp.val, alpha=0.5, info="", title="pressure")
            
                

            # for sd, data in self.mdg.subdomains(return_data=True):
            #     if sd.dim == 2:

            #         interfaces = self.subdomains_to_interfaces(self.mdg.subdomains(), [1])

            #         # mortar_projection = pp.ad.MortarProjections(self.mdg, [sd], interfaces, dim=1)
            #         # M = mortar_projection.mortar_to_primary_int.evaluate(self.equation_system)

            #         normals_all = sd.face_normals
            #         # normals = self.outwards_internal_boundary_normals(interfaces, unitary=True).evaluate(self.equation_system) # they are not unitary... ???
            #         # normals = normals.reshape( (int(normals.shape[0]/2), 2) ).T # Hope it is right

            #         # ff0 = self.interface_fluid_mass_flux_phase_0(self.interface_mortar_flux_phase_0(interfaces), interfaces, self.mixture.get_phase(0), "interface_mortar_flux_phase_0").evaluate(self.equation_system).val
            #         # ff0_vect = normals
            #         # ff0_vect[1] = 1000*ff0
            #         # ff0_vect[1, sd.tags["fracture_faces"]] = 1000*ff0
            #         # ff0_vect_all = 0*normals_all
            #         # ff0_vect_all[1, sd.tags["fracture_faces"]] = ff0_vect[1]

            #         (
            #         eq,
            #         accumulation,
            #         flux_tot,
            #         flux_intf_phase_0,
            #         flux_intf_phase_1,
            #         flux,
            #         source_phase_0,
            #         source_phase_1,
            #         source,
            #         ) = self.eq_fcn_pressure([sd])

            #         ff0_from_p = -flux_intf_phase_0.evaluate(self.equation_system).val # "-" bcs weird pp notation, you know
            #         ff1_from_p = -flux_intf_phase_1.evaluate(self.equation_system).val

            #         ff0_vect_from_p = copy.deepcopy(normals_all)
            #         ff0_vect_from_p[0] = 0
            #         ff0_vect_from_p[1] *= ff0_from_p

            #         ff1_vect_from_p = copy.deepcopy(normals_all)
            #         ff1_vect_from_p[0] = 0
            #         ff1_vect_from_p[1] *= ff1_from_p

            #         # pp.plot_grid(sd, vector_value=0.5 * normals, alpha=0)
            #         # print("ff0 = ", ff0)
            #         print("ff0_vect_from_p = ", ff0_vect_from_p)
            #         # pp.plot_grid(sd, vector_value=0.1 * ff0_vect_all, alpha=0, title="kind of ff0")
            #         pp.plot_grid(sd, vector_value=200 * ff0_vect_from_p, alpha=0, title="ff0 from p")
            #         #  print("ff1 = ", ff1)
            #         print("ff1_vect_from_p = ", ff1_vect_from_p)
            #         pp.plot_grid(sd, vector_value=1000 * ff1_vect_from_p, alpha=0, title="ff1 from p")
            #         pdb.set_trace()
                    
                
        # # close the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         internal_nodes_id = sd.get_internal_nodes()
                
        #         sd.nodes[1][26] -= 0.04
        #         sd.nodes[1][27] += 0.04
        #         sd.nodes[1][28] -= 0.04
        #         sd.nodes[1][29] += 0.04

        # self.mdg.compute_geometry()



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
        bounding_box = {"xmin": 0, "xmax": self.xmax, "ymin": 0, "ymax": self.ymax} #, "zmin": 0, "zmax": 1}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture( 1 * np.array([[0.2, 0.8], [0.6, 0.6]]) )
        frac2 = pp.LineFracture( np.array([[0.2, 0.8], [0.2, 0.2]]) )
        frac3 = pp.LineFracture( np.array([[0.5, 0.5], [0.2, 0.8]]) )

        frac4 = pp.LineFracture( np.array([[0.2, 0.8], [0.2, 0.8]]) )

        # frac1 = pp.PlaneFracture(np.array([[0.2, 0.7, 0.7, 0.2],[0.2, 0.2, 0.8, 0.8],[0.5, 0.5, 0.5, 0.5]]))
        self._fractures: list = [frac1, frac4]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 1 * 0.2, # 0.05
            "cell_size_fracture": 1 * 0.05,
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

fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": 1,
            "normal_permeability": 1, # this does NOT include the aperture
            "residual_aperture": 0.1,
        }
    )

material_constants = {"fluid": fluid_constants, "solid": solid_constants}

time_manager = pp.TimeManager(
    schedule=[0, 1],
    dt_init= 1 * 1e-3 ,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 20,
    "nl_convergence_tol": 5e-3,
    "nl_divergence_tol": 1e5,
    "time_manager": time_manager,
}

wetting_phase = pp.composite.phase.Phase(rho0=1, beta=1e-10)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5, beta=1e-10)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

if __name__ == "__main__":

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0 
            self.gravity_value = 1  # pp.GRAVITY_ACCELERATION
            self.dynamic_viscosity = 1  # TODO: wrong place...
            
            self.xmax = 1 * 1
            self.ymax = 1 * 1 

            self.relative_permeability = pp.tobedefined.relative_permeability.rel_perm_quadratic
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability) 
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(self.mobility)

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
