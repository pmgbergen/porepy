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
OLD TEST

TODO: todo... but it is useless now bcs the error is at the boundaries and darcy is zero at boundaries...


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

    # other methods not called in prepared_simulation: -------------------------------------------------

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

            print("darcy_flux_phase_0 = ", vals)

            if self.case == 100:
                if sd.dim == 2:
                    print(
                        "darcy_flux_phase_0 case 100 = ",
                        self.darcy_flux_phase_0_values_2d,
                    )
                    data[pp.PARAMETERS][self.ppu_keyword].update(
                        {"darcy_flux_phase_0": self.darcy_flux_phase_0_values_2d}
                    )
            else:
                data[pp.PARAMETERS][self.ppu_keyword].update(
                    {"darcy_flux_phase_0": vals}
                )

            vals = (
                self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                .evaluate(self.equation_system)
                .val
            )

            print("darcy_flux_phase_1 = ", vals)

            if self.case == 100:
                if sd.dim == 2:
                    print(
                        "darcy_flux_phase_1 case 100 = ",
                        self.darcy_flux_phase_1_values_2d,
                    )
                    data[pp.PARAMETERS][self.ppu_keyword].update(
                        {"darcy_flux_phase_1": self.darcy_flux_phase_1_values_2d}
                    )
            else:
                data[pp.PARAMETERS][self.ppu_keyword].update(
                    {"darcy_flux_phase_1": vals}
                )

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

        self.set_discretization_parameters()
        self.rediscretize()

        # TEST: ---------------------------------------------------------------------------

        # open the fracture:
        for sd in self.mdg.subdomains():
            if sd.dim == 2:
                for i in dir(sd):
                    print(i)

                internal_nodes_id = sd.get_internal_nodes()
                sd.nodes[1][internal_nodes_id[1]] += 0.1
                sd.nodes[1][internal_nodes_id[0]] -= 0.1

        self.mdg.compute_geometry()

        pp.plot_grid(self.mdg, info="cf", alpha=0)

        # for intf, data in self.mdg.interfaces(return_data=True, codim=1):
        #     print("intf.mortar_to_primary_avg() = ", intf.mortar_to_primary_avg())

        # cells of mortar grid:
        #  1    0
        # _________
        # |   |   |
        # ---------
        #  3     2

        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         normals = sd.face_normals
        #         pp.plot_grid(
        #             sd, vector_value=0.5 * normals, alpha=0
        #         )  # REMARK: mortar are positive from higer to lower, they do NOT respect the face normals
        #         pdb.set_trace()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == 2:
                discr = self.darcy_flux_discretization([sd])
                T_starnoni = discr.vector_source.evaluate(self.equation_system)
                print(T_starnoni.todense())

                normals = sd.face_normals
                darcy_phase_0 = (
                    data[pp.PARAMETERS][self.ppu_keyword]["darcy_flux_phase_0"]
                    * normals
                )  # DO NOT FORGET that this it integrated normal darcy flux, not the vector darcy velocity
                darcy_phase_1 = (
                    data[pp.PARAMETERS][self.ppu_keyword]["darcy_flux_phase_1"]
                    * normals
                )
                pp.plot_grid(
                    sd, vector_value=0.5 * darcy_phase_0, alpha=0, title="darcy_phase_0"
                )
                pp.plot_grid(
                    sd, vector_value=0.5 * darcy_phase_1, alpha=0, title="darcy_phase_1"
                )

        print("\n\n")
        print("\ndarcy_phase_0", darcy_phase_0)
        print("\ndarcy_phase_1", darcy_phase_1)
        pdb.set_trace()

        if self.case == 1:  # delta p = 0, g = 1, s0=0
            assert np.all(
                np.isclose(darcy_phase_0, 0 * darcy_phase_0, rtol=0, atol=1e-10)
            )
            assert np.all(
                np.isclose(
                    darcy_phase_1,
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 1]),
                    rtol=0,
                    atol=1e-10,
                )
            )

        print("\n\n\n TEST PASSED ----------------------------------------")
        pdb.set_trace()


class MyModelGeometryTest(test_hu_model.MyModelGeometry):
    def set_geometry(self) -> None:
        """ """

        self.set_domain()
        self.set_fractures()

        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        self.mdg = pp.create_mdg(
            "cartesian",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

    def set_domain(self) -> None:
        """ """
        self.size = 1 / self.units.m
        bounding_box = {"xmin": 0, "xmax": self.xmax, "ymin": 0, "ymax": 2}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[0, self.xmax], [1, 1]]))
        self._fractures: list = [frac1]  # , frac2]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size_x": self.xmax / 2 - 0.1,
            "cell_size_y": 1.0,
            "cell_size_fracture": 1.0,
        }
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


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassTest,
    MyModelGeometryTest,
    pp.DataSavingMixin,
):
    """ """


class FinalModelTest(PartialFinalModel):  # I'm sorry...
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = None  # pp.GRAVITY_ACCELERATION
        self.dynamic_viscosity = 1  # TODO: it is hardoced everywhere, you know...

        self.case = None

        self.xmax = None

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
# material_constants = {"solid": solid_constants}
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


model = FinalModelTest(mixture, params)


case = 1
model.case = case

if case == 1:  # delta p = 0, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])  # whatever...
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])

if case == 2:  # delta p = 0, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1, 1.0, 1, 1])
    model.pressure_values_1d = 1 * np.array([1, 1.0])

if case == 3:  # delta p = 1, g = 0
    model.xmax = 2
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 4:  # delta p = 1, g = 0
    model.xmax = 2
    model.gravity_value = 0
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 5:  # delta p = -1, g = 0
    model.xmax = 2
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])
    # delta p negative

if case == 6:  # delta p = 1, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 7:  # delta p = 1, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 8:  # delta p = -1, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])


# TODO:
if case == 9:  # delta p = 0, g = 1
    model.xmax = 4
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])
    # test both mortar and interface fluid flux of phase 1 and 2

if case == 10:  # delta p = 0, g = 1
    model.xmax = 4
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1, 1.0, 1, 1])
    model.pressure_values_1d = 1 * np.array([1, 1.0])

if case == 11:  # delta p = 1, g = 0
    model.xmax = 4
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 12:  # delta p = 1, g = 0
    model.xmax = 4
    model.gravity_value = 0
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 13:  # delta p = -1, g = 0
    model.xmax = 4
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])
    # delta p negative

if case == 14:  # delta p = 1, g = 1
    model.xmax = 4
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 15:  # delta p = 1, g = 1
    model.xmax = 4
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 16:  # delta p = -1, g = 1
    model.xmax = 4
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])


if case == 100:  # delta p = 0, g = 1
    model.xmax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0.5 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])

    model.darcy_flux_phase_0_values_2d = np.array(
        [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    # model.darcy_flux_phase_1_values_2d = np.array([0.,0,0,0,0, 0,0,0,-1,-1, 0,0,-1,-1])
    model.darcy_flux_phase_1_values_2d = np.ones(14)


pp.run_time_dependent_model(model, params)
