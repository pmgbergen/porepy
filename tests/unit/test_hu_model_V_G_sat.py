import numpy as np
import scipy as sp
import porepy as pp
import abc

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
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(self.ell)
                .saturation_operator([sd])
            )

            saturation_values = self.saturation_values_2d

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = self.pressure_values_2d

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

        for sd in self.mdg.subdomains():
            gigi = self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation
            print("saturation = ", gigi.val)
            pp.plot_grid(
                sd,
                gigi.val,
                alpha=0.5,
                info="c",
                title="saturation " + str(self.ell),
            )

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

        # TEST: --------------------------------------------------------------------------------------------
        # pp.plot_grid(self.mdg, info="cf", alpha=0)
        # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

        (
            eq,
            accumulation,
            rho_V_operator,
            rho_G_operator,
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(self.mdg.subdomains())

        rho_V = rho_V_operator.evaluate(self.equation_system).val
        rho_G = rho_G_operator.evaluate(self.equation_system).val

        print(
            "\nself.mixture.get_phase(0).saturation.val = ",
            self.mixture.get_phase(0).saturation.val,
        )
        print("\nrho_V = ", rho_V)
        print("\nrho_G = ", rho_G)

        pdb.set_trace()

        if self.case == 1:  # p = [1,0], g = 0
            assert np.all(rho_V == np.array([0, 0, 0, 0, 0, 1, 0]))

        if self.case == 2:  # p = [1,0], g = 1
            np.all(
                np.isclose(rho_V, np.array([0, 0, 0, 0, 0, 0, 0]))
            )  # IF COMPRESSIBLE I GET A SMALL VALUE OF THE SAME ORDER OF MAGNITUDE OF COMPRESSIBIITY

        if self.case == 3:  # s = [1, 1] => G = 0
            assert np.all(
                rho_G == np.array([0, 0, 0, 0, 0, 0, 0])
            )  # PASSED. IF COMPRESSIBLE I GET A SMALL VALUE OF THE SAME ORDER OF MAGNITUDE OF COMPRESSIBIITY

        if self.case == 4:  # s = [0, 1]
            assert np.all(
                np.isclose(rho_G, np.array([0, 0, 0, 0, 0, -0.5, 0]), rtol=0, atol=1e-3)
            )  # PASSED. if compressible not exaclty -0.5

        # same with larger cell:
        if self.case == 5:  # p = [1,0], g = 0
            assert np.all(rho_V == 2 * np.array([0, 0, 0, 0, 0, 1, 0]))

        if self.case == 6:  # p = [1,0], g = 1
            np.all(np.isclose(rho_V, 2 * np.array([0, 0, 0, 0, 0, 0, 0])))  # PASSED...

        if self.case == 7:  # s = [1, 1] => G = 0
            assert np.all(rho_G == 2 * np.array([0, 0, 0, 0, 0, 0, 0]))  # PASSED...

        if self.case == 8:  # s = [0, 1]
            assert np.all(
                np.isclose(
                    rho_G, 2 * np.array([0, 0, 0, 0, 0, -0.5, 0]), rtol=0, atol=1e-3
                )
            )  # PASSED

        print("\n\n TEST PASSED ------------------- ")
        pdb.set_trace()

        self.set_discretization_parameters()
        self.rediscretize()


class MyModelGeometryTest(pp.ModelGeometry):
    def set_geometry(self) -> None:
        """ """
        nxny = np.array([1, 2])
        physical_dim = {"xmin": 0, "xmax": self.xmax, "ymin": 0, "ymax": 2}
        sd = pp.CartGrid(nxny, physical_dim)

        self.mdg = pp.MixedDimensionalGrid()
        self.mdg._subdomain_data = {sd: {}}
        self.mdg.compute_geometry()

        self.nd: int = self.mdg.dim_max()

        # fake:
        self._domain = pp.Domain(bounding_box=physical_dim)


class PartialFinalModelTest(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
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
        self.gravity_value = None  # pp.GRAVITY_ACCELERATION
        self.dynamic_viscosity = 1  # TODO: it is hardoced everywhere, you know...

        self.case = None
        self.xmax = None

        self.pressure_values_2d = None
        self.pressure_values_1d = None

        self.saturation_values_2d = None
        self.saturation_values_1d = None


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
    schedule=[0, 100],
    dt_init=1e0,
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


class ConstantDensityPhase(pp.composite.phase.Phase):
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


wetting_phase = ConstantDensityPhase(rho0=1)
non_wetting_phase = ConstantDensityPhase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

model = FinalModel(mixture, params)

case = 100
model.case = case

if case == 1:
    model.xmax = 1
    model.gravity_value = 0
    model.pressure_values_2d = np.array([1.0, 0])
    model.saturation_values_2d = np.array([1.0, 1])

if case == 100:
    model.xmax = 1
    model.gravity_value = 0
    model.pressure_values_2d = np.array([1.0, 0])
    model.saturation_values_2d = np.array([-2.0, -2])


if case == 2:
    model.xmax = 1
    model.gravity_value = 1
    model.pressure_values_2d = np.array([1.0, 0])
    model.saturation_values_2d = np.array([1.0, 1])

if case == 3:
    model.xmax = 1
    model.gravity_value = 1
    model.pressure_values_2d = np.array([0.0, 0.0])
    model.saturation_values_2d = np.array([1.0, 1.0])

if case == 4:
    model.xmax = 1
    model.gravity_value = 1
    model.pressure_values_2d = np.array([0.0, 0.0])
    model.saturation_values_2d = np.array([0.0, 1])

if case == 5:
    model.xmax = 2
    model.gravity_value = 0
    model.pressure_values_2d = np.array([1.0, 0])
    model.saturation_values_2d = np.array([1.0, 1])

if case == 6:
    model.xmax = 2
    model.gravity_value = 1
    model.pressure_values_2d = np.array([1.0, 0])
    model.saturation_values_2d = np.array([1.0, 1])

if case == 7:
    model.xmax = 2
    model.gravity_value = 1
    model.pressure_values_2d = np.array([0.0, 0.0])
    model.saturation_values_2d = np.array([1.0, 1.0])

if case == 8:
    model.xmax = 2
    model.gravity_value = 1
    model.pressure_values_2d = np.array([0.0, 0.0])
    model.saturation_values_2d = np.array([0.0, 1])


pp.run_time_dependent_model(model, params)
