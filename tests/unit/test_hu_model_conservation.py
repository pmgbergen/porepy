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


NOTE: 
- the idea is to solve the stationary equations by setting the accumulation ther by zero
- test after one timestep that the flux is zero
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
                .get_phase(self.ell)
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
        print("\n\n --------- inside before_nonlinear_iteration: -----------")

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

        # check the conservation equation, the sum of the three terms must be zero. If not, see what happens.
        # this is done before Nweton iter, so values at previoes iter are taken.

        dt_operator = pp.ad.time_derivatives.dt
        dt = pp.ad.Scalar(self.time_manager.dt)

        # PRESSURE EQ: ----------------
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
        ) = self.eq_fcn_pressure(self.mdg.subdomains())

        print("2D:")
        accumulation_pressure = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[7]
        )
        net_flux_pressure = flux.evaluate(self.equation_system).val[[8, 9, 19, 24]]
        net_flux_pressure = (
            net_flux_pressure[0]
            - net_flux_pressure[1]
            + net_flux_pressure[2]
            - net_flux_pressure[3]
        )
        source_pressure = source.evaluate(self.equation_system).val[7]

        print("accumulation pressure cell 7 = ", accumulation_pressure)
        print("net_flux_pressure cell 7     = ", net_flux_pressure)
        print("source pressure cell 7       = ", source_pressure)

        print("1D:")  # cell 1 frac = 11 global ref
        accumulation_pressure = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[11]
        )
        net_flux_pressure = flux.evaluate(self.equation_system).val[[31, 32]]
        net_flux_pressure = net_flux_pressure[0] - net_flux_pressure[1]
        source_pressure = source.evaluate(self.equation_system).val[11]

        print("accumulation pressure cell 1 = ", accumulation_pressure)
        print("net_flux_pressure cell 1     = ", net_flux_pressure)
        print("source pressure cell 1       = ", source_pressure)

        # MASS BALANCE: ---------------
        (
            eq,
            accumulation,
            rho_V,
            rho_G,
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(self.mdg.subdomains())

        print("2D:")
        accumulation_mass = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[7]
        )
        net_flux_mass = flux_V_G - flux_intf_phase_0  # ell = 0
        net_flux_mass = net_flux_mass.evaluate(self.equation_system).val[[8, 9, 19, 24]]
        net_flux_mass = (
            net_flux_mass[0] - net_flux_mass[1] + net_flux_mass[2] - net_flux_mass[3]
        )
        source_mass = source_phase_0.evaluate(self.equation_system).val[7]

        print("accumulation mass cell 7 = ", accumulation_mass)
        print("net_flux mass cell 7     = ", net_flux_mass)
        print("source mass cell 7       = ", source_mass)

        print("1D:")
        accumulation_mass = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[11]
        )
        net_flux_mass = flux_V_G - flux_intf_phase_0  # ell = 0
        net_flux_mass = net_flux_mass.evaluate(self.equation_system).val[[31, 32]]
        net_flux_mass = net_flux_mass[0] - net_flux_mass[1]
        source_mass = source_phase_0.evaluate(self.equation_system).val[11]

        print("accumulation mass cell 1 = ", accumulation_mass)
        print("net_flux mass cell 1     = ", net_flux_mass)
        print("source mass cell 1       = ", source_mass)

        self.set_discretization_parameters()
        self.rediscretize()

    def eb_after_timestep(self):
        """ """
        print("\n\n -------------- inside eb_after_timestep: -------------------")

        # TEST SECTION: ---------------------------------------------------------

        # # open the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         internal_nodes_id = sd.get_internal_nodes()
        #         sd.nodes[1][[8, 10]] -= 0.1
        #         sd.nodes[1][[9, 11]] += 0.1

        # self.mdg.compute_geometry()

        # pp.plot_grid(self.mdg, info="cf", alpha=0.1)

        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         normals = sd.face_normals
        #         pp.plot_grid(
        #             sd, vector_value=0.5 * normals, info='f', alpha=0
        #         )

        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         gigi = (
        #             self.mixture.mixture_for_subdomain(sd)
        #             .get_phase(self.ell)
        #             .saturation
        #         )
        #         print("saturation = ", gigi.val)
        #         pp.plot_grid(
        #             sd,
        #             gigi.val,
        #             alpha=0.5,
        #             info="c",
        #             title="saturation " + str(self.ell),
        #         )

        dt_operator = pp.ad.time_derivatives.dt
        dt = pp.ad.Scalar(self.time_manager.dt)
        div = pp.ad.Divergence(self.mdg.subdomains(), dim=1)

        # check confervation of cell 7
        # PRESSURE EQ: -----------------------
        (
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0,
            flux_intf_phase_1,
            flux_pressure,
            source_phase_0,
            source_phase_1,
            source,
        ) = self.eq_fcn_pressure(self.mdg.subdomains())

        print("2D:")
        accumulation_pressure = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val
        )[7]

        flux_cell_7_pressure = flux_pressure.evaluate(self.equation_system).val[
            [8, 9, 19, 24]
        ]  # see the plot for the indeces... # at the end of this vector there are the fluxes in the fracture
        net_flux_pressure = (
            flux_cell_7_pressure[0]
            - flux_cell_7_pressure[1]
            + flux_cell_7_pressure[2]
            - flux_cell_7_pressure[3]
        )

        # print(div.evaluate(self.equation_system)) # 2D: 8-1, 9+1, 19-1, 24+1. # 1D: 31-1, 32+1
        assert (
            net_flux_pressure
            == -(div @ flux_pressure).evaluate(self.equation_system).val[7]
        )
        print("minus? I'm confused...")

        source_pressure = source.evaluate(self.equation_system).val[7]
        print(
            "no time derivative acc cell 7 = ",
            accumulation.evaluate(self.equation_system).val[7],
        )
        print(
            "accumulation pressure cell 7  = ", accumulation_pressure
        )  # I think it's zero because it is set to zero. DON'T THINK, CHECK.
        print("net_flux pressure             = ", net_flux_pressure)
        print("source_pressure cell 7        = ", source_pressure)

        print("1D:")
        accumulation_pressure = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val
        )[11]

        flux_cell_7_pressure = flux_pressure.evaluate(self.equation_system).val[
            [31, 32]
        ]
        net_flux_pressure = flux_cell_7_pressure[0] - flux_cell_7_pressure[1]

        assert (
            net_flux_pressure
            == -(div @ flux_pressure).evaluate(self.equation_system).val[11]
        )

        source_pressure = source.evaluate(self.equation_system).val[11]

        print(
            "no time derivative acc cell 11 = ",
            accumulation.evaluate(self.equation_system).val[11],
        )
        print("accumulation pressure cell 11  = ", accumulation_pressure)
        print("net_flux pressure cell 11      = ", net_flux_pressure)
        print("source_pressure cell 7         = ", source_pressure)

        # MASS BALANCE: -----------------------
        (
            eq,
            accumulation,
            rho_V,
            rho_G,
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(self.mdg.subdomains())

        print("2D:")
        accumulation_mass = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[7]
        )

        flux_mass_op = flux_V_G - flux_intf_phase_0
        flux_mass = flux_mass_op.evaluate(self.equation_system).val
        # conservation cell 7
        flux_cell_7_mass = flux_mass[[8, 9, 19, 24]]  # see the plot for the indeces...
        net_flux_mass = (
            flux_cell_7_mass[0]
            - flux_cell_7_mass[1]
            + flux_cell_7_mass[2]
            - flux_cell_7_mass[3]
        )
        assert (
            net_flux_mass == -(div @ flux_mass_op).evaluate(self.equation_system).val[7]
        )

        source_mass = source_phase_0.evaluate(self.equation_system).val[7]

        print(
            "no time derivative acc cell 7 = ",
            accumulation.evaluate(self.equation_system).val[7],
        )
        print("accumulation mass cell 7      = ", accumulation_mass)
        print("net_flux mass cell 11         = ", net_flux_mass)
        print("source_mass cell 7            = ", source_mass)

        print("1D:")
        accumulation_mass = (
            dt_operator(accumulation, dt).evaluate(self.equation_system).val[11]
        )

        flux_mass_op = flux_V_G - flux_intf_phase_0
        flux_mass = flux_mass_op.evaluate(self.equation_system).val
        flux_cell_7_mass = flux_mass[[31, 32]]
        net_flux_mass = flux_cell_7_mass[0] - flux_cell_7_mass[1]

        assert (
            net_flux_mass
            == -(div @ flux_mass_op).evaluate(self.equation_system).val[11]
        )

        source_mass = source_phase_0.evaluate(self.equation_system).val[11]

        print(
            "no time derivative acc cell 11 = ",
            accumulation.evaluate(self.equation_system).val[11],
        )
        print("accumulation mass cell 11      = ", accumulation_mass)
        print("net_flux mass cell 11          = ", net_flux_mass)
        print("source_mass cell 11            = ", source_mass)

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

if case == 2:
    print("\n\n\n case 2")
    model.gravity_value = 1
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([0.0, 0, 0])

if case == 3:
    print("\n\n\n case 3")
    model.gravity_value = 1
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([0.0, 0, 0])

pp.run_time_dependent_model(model, params)
