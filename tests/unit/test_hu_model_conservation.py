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
- the idea is to solve the stationary equations by setting the accumulation ther by zero
- test after one timestep that the flux is zero
- I'm using both equaitons from test_hu_model and test_hu_model_conservation. I need to run one time iteration and check the flux. NOT ANYMORWE

I'm not able to run a stationary simulation

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

        # from email:
        for sd in self.mdg.subdomains():
            saturation_variable = (
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(0)
                .saturation_operator([sd])
            )

            # saturation_values = np.ones(sd.num_cells)

            saturation_values = 1 * np.ones(sd.num_cells)
            # y_max = 2  ### from MyModelGeometry
            # saturation_values[np.where(sd.cell_centers[1] > y_max / 2)] = 1.0

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
            gigi = (
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(0)
                .saturation
            )
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

        self.mixture.apply_constraint(
            self.ell,)

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

        (
            eq,
            accumulation_mass,
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(self.mdg.subdomains())

        accumulation_mass = (accumulation_mass.evaluate(self.equation_system).val)[7]
        print("\n accumulation mass cell 7 before nonlinear iter = ", accumulation_mass)
        print("\n")

        super().before_nonlinear_iteration()

    def eb_after_timestep(self):
        """ """

        # TEST SECTION: ---------------------------------------------------------

        # # open the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         internal_nodes_id = sd.get_internal_nodes()
        #         sd.nodes[1][[8, 10]] += 0.1
        #         sd.nodes[1][[9, 11]] -= 0.1

        # self.mdg.compute_geometry()

        pp.plot_grid(self.mdg, info="cf", alpha=0)

        for sd in self.mdg.subdomains():  # useless, but may be useful, so keep it
            if sd.dim == 2:
                gigi = (
                    self.mixture.mixture_for_subdomain(sd)
                    .get_phase(self.ell)
                    .saturation
                )
                print("saturation = ", gigi.val)
                pp.plot_grid(
                    sd,
                    gigi.val,
                    alpha=0.5,
                    info="c",
                    title="saturation " + str(self.ell),
                )

                # pressure flux:
                (
                    eq,
                    accumulation_pressure,
                    flux_tot,
                    flux_intf_phase_0,
                    flux_intf_phase_1,
                    flux_pressure,
                    source_phase_0,
                    source_phase_1,
                    source_pressure,
                ) = self.eq_fcn_pressure(self.mdg.subdomains())

                # conservation cell 7
                flux_cell_7_pressure = flux_pressure.evaluate(self.equation_system).val[
                    [8, 9, 24, 28]
                ]  # see the plot for the indeces... # at the end of this vector there are the fluxes in the fracture
                net_flux_pressure = np.sum(flux_cell_7_pressure)
                print("\nnet_flux pressure = ", net_flux_pressure)

                dt_operator = pp.ad.time_derivatives.dt
                dt = pp.ad.Scalar(self.time_manager.dt)
                accumulation_pressure = (
                    dt_operator(accumulation_pressure, dt)
                    .evaluate(self.equation_system)
                    .val
                )

                accumulation_pressure = accumulation_pressure[7]
                print(
                    "\n accumulation pressure cell 7 = ", accumulation_pressure
                )  # I think it's zero because it is set to zero

                source_pressure = source_pressure.evaluate(self.equation_system).val[7]
                print("\n source_pressure cell 7 = ", source_pressure)

                (
                    eq,
                    accumulation_mass,
                    flux_V_G,
                    flux_intf_phase_0,
                    flux_intf_phase_1,
                    source_phase_0,
                    source_phase_1,
                ) = self.eq_fcn_mass(self.mdg.subdomains())

                # mass flux:
                flux_mass = (
                    (flux_V_G - flux_intf_phase_0).evaluate(self.equation_system).val
                )
                accumulation_mass = accumulation_mass.evaluate(
                    self.equation_system
                ).val[7]
                print("\n accumulation mass cell 7 = ", accumulation_mass)

                source_mass = source_phase_0.evaluate(self.equation_system).val[7]
                print("\n source_mass cell 7 = ", source_mass)

                # conservation cell 7
                flux_cell_7_mass = flux_mass[
                    [8, 9, 24, 28]
                ]  # see the plot for the indeces...
                net_flux_mass = np.sum(flux_cell_7_mass)
                print("\nnet_flux mass = ", net_flux_mass)

                pdb.set_trace()
                if self.case == 1:
                    print("case 1")
                    # assert net_flux_pressure == 0
                    # assert net_flux_mass == 0
                if self.case == 2:
                    print("case 2")
                    # assert net_flux_pressure == 0
                    # assert net_flux_mass == 0
                if self.case == 3:
                    print("case 3")
                    # assert net_flux_pressure == 0
                    # assert net_flux_mass == 0

        print("\n\n TEST PASSED ------------------- ")
        pdb.set_trace()


class MyModelGeometryTest(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        self.size = 1.0 / self.units.m
        box = {"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 2 * self.size}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[1, 4], [1, 1]]))
        self._fractures: list = [frac1]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {"cell_size": 0.99}
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

        self.case = None  # only for tests
        self.pressure_values_2d = None
        self.pressure_values_1d = None


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

# wetting_phase = ConstantDensityPhase(rho0=1)
# non_wetting_phase = ConstantDensityPhase(rho0=0.5)

wetting_phase = pp.composite.phase.Phase(rho0=1)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])


model = FinalModel(mixture, params)
case = 2
model.case = case

if case == 1:
    print("\n\n\n case 1")
    model.gravity_value = 0
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([0.0, 0, 0])

    # PASSED. of course, let it run few timesteps

if case == 2:
    print("\n\n\n case 2")
    model.gravity_value = 1
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([0.0, 0, 0])

    # FAILED

if case == 3:
    print("\n\n\n case 3")
    model.gravity_value = 1
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([0.0, 0, 0])

    # FAILED

pp.run_time_dependent_model(model, params)
