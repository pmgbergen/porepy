import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model
from test_hu_model import FunctionRhoG, FunctionRhoV, FunctionTotalFlux


import os
import sys
import pdb

os.system("clear")


"""
NOTE: 
- test copied from test_hu, so see comments there
- accumulation and sources = 0
- g = 1
- p = [1,0]
- constant density
(- come negli altri test, ci potrebbero esser problemi con la mixture che non ho proprio capito)

"""


class EquationsTest(test_hu_model.Equations):
    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        accumulation = pp.ad.Scalar(0, "zero")
        accumulation.set_name("fluid_mass_p_eq")

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

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux -= (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux -= (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        # sources: ---------------------------------------
        source = pp.ad.Scalar(0, "zero")
        source.set_name("zero source")

        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("pressure_equation")

        return eq

    # MASS BALANCE: ----------------------------------------------------------------------------------------------------------------

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # accumulation term: ------------------------------------------------
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        accumulation = pp.ad.Scalar(0, "zero")
        accumulation.set_name("fluid_mass_mass_eq")

        # subdomains flux contribution: -------------------------------------
        rho_V_operator = FunctionRhoV(
            pp.rho_flux_V, self.equation_system, self.mixture, self.mdg, name="rho V"
        )
        rho_G_operator = FunctionRhoG(
            pp.rho_flux_G, self.equation_system, self.mixture, self.mdg, name="rho G"
        )

        fake_input = self.pressure(subdomains)
        flux = (
            rho_V_operator(fake_input)
            + rho_G_operator(fake_input)
            + self.bc_values(subdomains)
        )  # TODO: this is wrong, but bc val are 0 so I dont care...

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        if self.ell == 0:  # TODO: you can avoid if condition
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")

            flux -= (
                discr.bound_transport_neu  # -1,0,1 matrix
                @ mortar_projection.mortar_to_primary_int
                @ self.interface_fluid_mass_flux_phase_0(
                    interfaces,
                    self.mixture.get_phase(0),
                    "interface_mortar_flux_phase_0",
                )
            )

        else:  # self.ell == 1
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
            flux -= (
                discr.bound_transport_neu
                @ mortar_projection.mortar_to_primary_int
                @ self.interface_fluid_mass_flux_phase_1(
                    interfaces,
                    self.mixture.get_phase(1),
                    "interface_mortar_flux_phase_1",
                )
            )

        # sources: ---------------------------------------
        source = pp.ad.Scalar(0, "zero")
        source.set_name("zero source")

        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("mass_balance_equation")
        return eq


class SolutionStrategyPressureMassTest(test_hu_model.SolutionStrategyPressureMass):
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

                saturation_values = np.array([1.0, 1.0])

                self.equation_system.set_variable_values(
                    saturation_values,
                    variables=[saturation_variable],
                    time_step_index=0,
                    iterate_index=0,
                )

                pressure_variable = self.pressure([sd])

                pressure_values = np.array([1.0, 0.0])  # np.array([1.0, 0.0])
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

    def before_nonlinear_iteration(self):
        """ """

        self.mixture.apply_constraint(
            self.ell, self.equation_system, self.mdg.subdomains()
        )  ### TODO: redo...

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
                    self.mixture.mixture_for_subdomain(self.equation_system, sd),
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

        # TEST SECTION: -----------------------
        # sorry, I dont have time to create different tests... modify this one looking at test_hu.py

        # subdomain = self.mdg.subdomains()
        # rho_G = self.mass_balance_equation(subdomain)
        # rho_G = rho_G.evaluate(self.equation_system)

        # pdb.set_trace()

        self.assemble_linear_system()
        A_ad = self.linear_system[0].todense()
        np.set_printoptions(precision=3)

        A_exact = np.array([[1, -1, 0, 0], [-1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0]])
        print("A_ad = ", A_ad)
        print("A_exact = ", A_exact)

        pdb.set_trace()

        assert np.all(np.isclose(A_ad, A_exact, rtol=0, atol=1e-10))  # PASSED

        # # only mass balace jac: -----

        # rho_V = pp.rho_flux_V(
        #     sd,
        #     mixture.mixture_for_subdomain(self.equation_system, sd),
        #     self.ell,
        #     pressure_adarray,
        #     data["for_hu"]["total_flux_internal"],
        #     left_restriction,
        #     right_restriction,
        #     ad,
        #     dynamic_viscosity,
        # )

        # rho_G = pp.rho_flux_G(
        #     sd,
        #     mixture.mixture_for_subdomain(self.equation_system, sd),
        #     self.ell,
        #     pressure_adarray,
        #     self.gravity_value,
        #     left_restriction,
        #     right_restriction,
        #     transmissibility_internal_tpfa,
        #     ad,
        #     dynamic_viscosity,
        # )

        # pp_div = pp.fvutils.scalar_divergence(sd)

        # rho_V_cell_no_bc = pp_div @ rho_V
        # rho_G_cell_no_bc = pp_div @ rho_G
        # flux_cell_no_bc = rho_V_cell_no_bc + rho_G_cell_no_bc

        # mass_bal_jac = flux_cell_no_bc.jac.A
        # print("mass_val_jac = ", mass_bal_jac)

        pdb.set_trace()

        print("\n\n TEST PASSED ------------------- ")

        super().before_nonlinear_iteration()


class MyModelGeometryTest(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        self.size = 1.0 / self.units.m
        box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2 * self.size}
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


class PartialFinalModelTest(
    test_hu_model.PrimaryVariables,
    EquationsTest,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassTest,
    MyModelGeometryTest,
    pp.DataSavingMixin,
):
    """ """


class FinalModel(PartialFinalModelTest):  # I'm sorry...
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
        "normal_permeability": 1e3,
        "residual_aperture": 0.1,
    }
)

material_constants = {"fluid": fluid_constants, "solid": solid_constants}
time_manager = pp.TimeManager(
    schedule=[0, 10],
    dt_init=5e-4,
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

wetting_phase = ConstantDensityPhase(rho0=1)
non_wetting_phase = ConstantDensityPhase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

# mixture = test_hu_model.mixture

model = FinalModel(mixture, params)  # eh... non capisco il problema


pp.run_time_dependent_model(model, params)
