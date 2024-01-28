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
OLD TEST

"""


class EquationsTest(test_hu_model.Equations):
    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        (
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0,
            flux_intf_phase_1,
            flux,
            source_phase_0,
            source_phase_1,
            source,
        ) = self.eq_fcn_pressure(subdomains)

        accumulation = pp.ad.Scalar(0)
        flux = flux
        source = pp.ad.Scalar(0)
        eq_mod = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq_mod.set_name("pressure_equation_test")
        return eq_mod

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        (
            eq,
            accumulation,
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(subdomains)

        accumulation = pp.ad.Scalar(0)
        flux = flux_V_G - flux_intf_phase_0  # self.ell = 0
        source = pp.ad.Scalar(0)
        eq_mod = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq_mod.set_name("mass_balance_equation_test")
        return eq_mod


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

        for sd in self.mdg.subdomains():
            saturation_variable = (
                # self.mixture.mixture_for_subdomain(self.equation_system, sd)
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(0)
                .saturation_operator([sd])
            )

            if sd.dim == 2:
                saturation_values = self.saturation_values_2d
            else:
                saturation_values = self.saturation_values_1d

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            if sd.dim == 2:
                pressure_values = self.pressure_values_2d
            else:  # sd.dim == 1
                pressure_values = self.pressure_values_1d

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )

        for sd, data in self.mdg.subdomains(return_data=True):
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
            self.ell
        )  # , self.equation_system, self.mdg.subdomains())

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
            total_flux_internal = pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
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

        # TEST SECTION: --------------------------------------------------------------

        pp.plot_grid(self.mdg, info="c", alpha=0)

        sss = self.mixture.get_phase(0).saturation
        ppp = self.pressure(self.mdg.subdomains()).evaluate(self.equation_system)

        ### self.equation_system.md_variable("saturation", self.mdg.subdomains()).evaluate(self.equation_system).val
        ### self.mixture.get_phase(0).saturation_operator(self.mdg.subdomains()).evaluate(self.equation_system).val

        self.assemble_linear_system()
        A_ad = self.linear_system[0].todense()
        np.set_printoptions(precision=2, linewidth=700, threshold=sys.maxsize)
        A_exact = np.array([])
        print("A_ad = ", A_ad)
        print("A_exact = ", A_exact)

        pdb.set_trace()

        assert np.all(np.isclose(A_ad, A_exact, rtol=0, atol=1e-10))

        print("\n\n TEST PASSED ----------------------------------------- ")
        pdb.set_trace()

        super().before_nonlinear_iteration()


class MyModelGeometryTest(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        self.size = 1.0 / self.units.m
        box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2 * self.size}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[0, 1], [1, 1]]))
        self._fractures: list = [frac1]

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

        self.case = None
        self.saturation_values_2d = None
        self.saturation_values_1d = None
        self.pressure_values_2d = None
        self.pressure_values_1d = None


fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
    {
        "porosity": 0.25,
        "permeability": 1,
        "normal_permeability": 1,
        "residual_aperture": 0.1,
    }
)

material_constants = {"fluid": fluid_constants, "solid": solid_constants}
time_manager = pp.TimeManager(
    schedule=[0, 10],
    dt_init=1,
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

model = FinalModel(mixture, params)

model.saturation_values_2d = np.array([1.0, 56161])
model.saturation_values_1d = np.array([1451.0])
model.pressure_values_2d = np.array([-541.0, 550])
model.pressure_values_1d = np.array([-60.0])


pp.run_time_dependent_model(model, params)
