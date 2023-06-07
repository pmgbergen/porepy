import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import os
import sys
import pdb

os.system("clear")


class FunctionTotalFlux(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False  # TODO: 100% sure?

        self.equation_system = equation_system  # I'm sorry...
        self.mixture = mixture
        self.mdg = mdg

    def get_args_from_model(self):
        """to be overriden"""

        return

    def get_args_from_sd_data_dictionary(self, data):
        """ """
        pressure = data["for_hu"]["pressure_AdArray"]  # SOMETHING SIMILAR FOR SATURAION
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

        return [
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            expansion_matrix,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """
        remember that *args_not_used are result[1] in _parse_operator
        """
        rhs_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]

            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )  # sorry

            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            rhs_list.append(self.func(*args).val)

        return np.hstack(rhs_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """
        TODO: find a smart way to not repeat the hu call for val and jac
        """
        jac_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class FunctionRhoV(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        self.equation_system = equation_system  # I'm sorry...
        self.mixture = mixture
        self.mdg = mdg

    def get_args_from_model(self):
        """to be overriden"""

        return

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
        rhs_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            rhs_list.append(self.func(*args).val)
            # TODO: not your they are actually rhs, check the sign
        return np.hstack(rhs_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class FunctionRhoG(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        equation_system,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        self.equation_system = equation_system  # I'm sorry...
        self.mixture = mixture
        self.mdg = mdg

    def get_args_from_model(self):
        """to be overriden"""

        return

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
        return [
            ell,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        rhs_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            rhs_list.append(self.func(*args).val)
        return np.hstack(rhs_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(
                self.mixture.mixture_for_subdomain(self.equation_system, subdomain)
            )  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class PressureEquation(pp.BalanceEquation):
    """ """

    def set_equations(self):
        subdomains = self.mdg.subdomains()
        # codim_1_interfaces = self.mdg.interfaces(codim=1)
        subdomain_eq = self.pressure_equation(subdomains)
        # interface_eq = self.interface_darcy_flux_equation(codim_1_interfaces)
        self.equation_system.set_equation(subdomain_eq, subdomains, {"cells": 1})
        # self.equation_system.set_equation(
        #     interface_eq, codim_1_interfaces, {"cells": 1}
        # )

    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        m = np.logical_not(self.ell) * 1  # sorry

        mass_density_ell = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        mass_density_m = (
            self.porosity(subdomains)
            * self.mixture.get_phase(m).mass_density_operator(subdomains, self.pressure)
            * self.mixture.get_phase(m).saturation_operator(subdomains)
        )
        mass_density = mass_density_ell + mass_density_m

        accumulation = pp.ad.Scalar(0, "zero") * self.volume_integral(
            mass_density, subdomains, dim=1
        )
        accumulation.set_name("fluid_mass")

        rho_total_flux_operator = FunctionTotalFlux(
            pp.rho_total_flux,
            self.equation_system,
            self.mixture,
            self.mdg,
            name="rho qt",
        )

        fake_input = self.pressure(subdomains)
        flux = rho_total_flux_operator(fake_input) + self.bc_values(
            subdomains
        )  # this is wrong, but bc val are 0 so I dont care...

        source = pp.ad.Scalar(0)  # HARDCODED

        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("pressure_equation")
        return eq


class MassBalance(pp.BalanceEquation):
    """ """

    def set_equations(self):
        subdomains = self.mdg.subdomains()
        # codim_1_interfaces = self.mdg.interfaces(codim=1)
        subdomain_eq = self.mass_balance_equation(subdomains)
        # interface_eq = self.interface_darcy_flux_equation(codim_1_interfaces)
        self.equation_system.set_equation(subdomain_eq, subdomains, {"cells": 1})
        # self.equation_system.set_equation(
        #     interface_eq, codim_1_interfaces, {"cells": 1}
        # )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        accumulation = pp.ad.Scalar(0, "zero") * self.volume_integral(
            mass_density, subdomains, dim=1
        )
        accumulation.set_name("fluid_mass")

        rho_V_operator = FunctionRhoV(
            pp.rho_flux_V, self.equation_system, self.mixture, self.mdg, name="rho V"
        )
        rho_G_operator = FunctionRhoG(
            pp.rho_flux_G, self.equation_system, self.mixture, self.mdg, name="rho G"
        )

        fake_input = self.pressure(subdomains)
        flux = (
            pp.ad.Scalar(0, "zero") * rho_V_operator(fake_input)
            + rho_G_operator(fake_input)
            + pp.ad.Scalar(0, "zero") * self.bc_values(subdomains)
        )  # this is wrong, but bc val are 0 so I dont care...  # TODO: not sure I have to add an argument, we'll see...

        source = pp.ad.Scalar(0)  # HARDCODED

        # eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        # eq.set_name("mass_balance_equation")

        eq = flux  # I need to test only this one...

        return eq


class EquationsPressureMass(
    PressureEquation,
    MassBalance,
):
    def set_equations(self):
        PressureEquation.set_equations(self)
        MassBalance.set_equations(self)


class VariablesPressureMass(pp.VariableMixin):
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
            self.interface_darcy_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

    def pressure(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        p = self.equation_system.md_variable(self.pressure_variable, subdomains)
        return p

    # saturation is inside mixture

    def interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        flux = self.equation_system.md_variable(
            self.interface_darcy_flux_variable, interfaces
        )
        return flux

    # def reference_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
    #     """ """
    #     p_ref = self.fluid.pressure()
    #     size = sum([sd.num_cells for sd in subdomains])
    #     return pp.wrap_as_ad_array(p_ref, size, name="reference_pressure")


class ConstitutiveLawPressureMass(
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.ConstantPorosity,
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
            vals[boundary_faces] = 0  # change this if you want different bc val
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
        self.interface_darcy_flux_variable: str = "interface_darcy_flux"
        self.darcy_keyword: str = "flow"
        self.mobility_keyword: str = "mobility"  # used for darcy flux (? boh...)

    def prepare_simulation(self) -> None:
        """ """
        self.set_geometry()

        self.initialize_data_saving()

        self.set_materials()
        self.set_equation_system_manager()

        self.add_equation_system_to_phases()

        self.create_variables()
        self.mixture.apply_constraint(
            self.ell, self.equation_system, self.mdg.subdomains()
        )
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

        # from email:
        for sd in self.mdg.subdomains():
            if sd.dim == self.mdg.dim_max():  # thers is a is_frac attribute somewhere
                # saturation_variable = self.equation_system.md_variable(
                #     name="saturation", grids=[sd]
                # )
                saturation_variable = (
                    self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    .get_phase(0)
                    .saturation_operator([sd])
                )

                saturation_values = 0 * np.ones(sd.num_cells)
                y_max = 3  # sorry, see domain...
                saturation_values[np.where(sd.cell_centers[1] >= y_max / 2)] = 1.0

                self.equation_system.set_variable_values(
                    saturation_values,
                    variables=[saturation_variable],
                    time_step_index=0,
                    iterate_index=0,
                )

                pressure_variable = self.pressure([sd])

                pressure_values = np.array(
                    [1.0, 1, 1, 1, 1, 1, 1, 1, 1]
                )  # np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0])
                self.equation_system.set_variable_values(
                    pressure_values,
                    variables=[pressure_variable],
                    time_step_index=0,
                    iterate_index=0,
                )

        # just copied, fix it:
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
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
            # I souldn't need this neither:
            # pp.initialize_data(
            #     sd,
            #     data,
            #     self.mobility_keyword,
            #     {
            #         "bc": self.bc_type_mobrho(sd),
            #     },
            # )

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
        """rewuired by tpfa (only?)"""

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

            # expansion matrices? è un errore che non ci siano anche loro qui?
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

    # other methods not called in prepared_simulation: -------------------------------------------------

    def set_nonlinear_discretizations(self) -> None:
        """ """
        super().set_nonlinear_discretizations()  # TODO: EH? It does literally nothing...!

        # self.add_nonlinear_discretization(
        #     self.interface_mobility_discretization(self.mdg.interfaces()).flux,
        # )

    def before_nonlinear_iteration(self):
        """ """

        self.mixture.apply_constraint(
            self.ell, self.equation_system, self.mdg.subdomains()
        )  ### TODO: redo...

        for sd, data in self.mdg.subdomains(return_data=True):
            gigi = (
                self.mixture.mixture_for_subdomain(self.equation_system, [sd])
                .get_phase(0)
                .saturation
            )

            mario = (
                self.mixture.mixture_for_subdomain(self.equation_system, [sd])
                .get_phase(1)
                .saturation
            )

            sss = self.equation_system.md_variable(
                "saturation", self.mdg.subdomains()
            ).evaluate(self.equation_system)

            ppp = self.equation_system.md_variable(
                "pressure", self.mdg.subdomains()
            ).evaluate(self.equation_system)

            print("saturation = ", mario.val)
            print("pressure = ", ppp.val)
            # pdb.set_trace()

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
            total_flux_internal = (
                pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
                    sd,
                    self.mixture.mixture_for_subdomain(self.equation_system, sd),
                    pressure_adarray,
                    self.gravity_value,
                    left_restriction,
                    right_restriction,
                    transmissibility_internal_tpfa,
                    ad,
                    dynamic_viscosity,
                )
            )
            data["for_hu"]["total_flux_internal"] = np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            )
            data["for_hu"]["pressure_AdArray"] = self.pressure([sd]).evaluate(
                self.equation_system
            )

            # from original before_nonlinear_iteration:
            vals = (
                self.darcy_flux([sd]).evaluate(self.equation_system).val
            )  # TODO: i think i need only interface_darcy_flux, i dont' want to bother about htis now...
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

            # some plot: ---------------
            pp.plot_grid(sd, sss.val, alpha=0.5, info="c")
            pp.plot_grid(sd, gigi.val, alpha=0.5, info="c")
            pp.plot_grid(sd, ppp.val, alpha=0.5, info="c")

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        # TEST SECTION: -----------------------
        # sorry, I dont have time to create different tests... modify this one looking at test_hu.py

        subdomain = self.mdg.subdomains()

        # rho_V = self.mass_balance_equation(subdomain)
        # rho_V = rho_V.evaluate(self.equation_system).val
        # rho_V = rho_V[sd.get_internal_faces()]
        # assert np.all(rho_V == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]))  # PASSED

        rho_G = self.mass_balance_equation(subdomain)
        rho_G = rho_G.evaluate(self.equation_system).val
        rho_G = rho_G[sd.get_internal_faces()]

        assert np.all(
            np.isclose(
                rho_G,
                np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0]),
                rtol=0,
                atol=1e-9,
            )
        )

        print("\n\n TEST PASSED ------------------- ")
        pdb.set_trace()

        super().before_nonlinear_iteration()


class MyModelGeometry(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        self.size = 1.0 / self.units.m
        box = {"xmin": 0, "xmax": 3 * self.size, "ymin": 0, "ymax": 3 * self.size}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """ """
        self._fractures: list = []

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {"cell_size": 0.9}
        return self.params.get("meshing_arguments", default_meshing_args)


class ConstantDensityPhase(pp.Phase):
    """ """

    def mass_density(self, p):
        """ """
        if isinstance(p, pp.ad.AdArray):
            rho = self._rho0 * pp.ad.AdArray(
                np.ones(p.val.shape), 0 * p.jac
            )  # TODO: is it right?
        else:
            rho = self._rho0 * np.ones(p.shape)
        return rho

    def mass_density_operator(self, subdomains, pressure):
        """ """
        p = pressure(subdomains)
        mass_rho_operator = pp.ad.Function(self.mass_density, "mass_density_operator")
        rho = mass_rho_operator(p)

        return rho


wetting_phase = ConstantDensityPhase(rho0=1)
non_wetting_phase = ConstantDensityPhase(rho0=2)


mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])


class PartialFinalModel(
    EquationsPressureMass,
    VariablesPressureMass,
    ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMass,
    SolutionStrategyPressureMass,
    MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


class FinalModel(PartialFinalModel):  # I'm sorry...
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = -1


fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants({"porosity": 0.25, "permeability": 1})

material_constants = {"fluid": fluid_constants, "solid": solid_constants}
params = {"material_constants": material_constants}


model = FinalModel(mixture, params)


pp.run_time_dependent_model(model, params)
