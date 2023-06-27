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
            # if sd.dim == self.mdg.dim_max():  # thers is a is_frac attribute somewhere
            if True:  ### ...
                saturation_variable = (
                    self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    .get_phase(0)
                    .saturation_operator([sd])
                )

                saturation_values = np.ones(sd.num_cells)

                self.equation_system.set_variable_values(
                    saturation_values,
                    variables=[saturation_variable],
                    time_step_index=0,
                    iterate_index=0,
                )

                pressure_variable = self.pressure([sd])

                if sd.dim == 2:
                    print("setting IC for 2D grid")
                    pressure_values = np.ones(sd.num_cells)
                else:  # sd.dim == 1
                    print("setting IC for 1D grid")
                    pressure_values = np.zeros(sd.num_cells)
                    # pressure_values = np.ones(sd.num_cells)

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

        super().before_nonlinear_iteration()

        # TEST: ---------------------------------------------------------------------------

        # open the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         for i in dir(sd):
        #             print(i)

        #         internal_nodes_id = sd.get_internal_nodes()
        #         sd.nodes[1][internal_nodes_id[1]] += 0.1
        #         sd.nodes[1][internal_nodes_id[0]] -= 0.1

        # self.mdg.compute_geometry()

        # pp.plot_grid(self.mdg, info="cf", alpha=0)

        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         normals = sd.face_normals
        #         pp.plot_grid(
        #             sd, vector_value=0.5 * normals, alpha=0
        #         )  # REMARK: mortar are positive from higer to lower, they do NOT respect the face normals
        #         pdb.set_trace()

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            mortar_initialized = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            assert np.all(mortar_initialized == np.array([0, 0, 0, 0]))

            mortar_initialized = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            assert np.all(mortar_initialized == np.array([0, 0, 0, 0]))

            # pay attention, you are goning to evaluate the equation "lamnda - rhs = ...". Since lambda = 0 ny initialization the evaluation of thr equations is -mortar
            # that's why i'll write -np.array([...

            mortar = (
                self.interface_mortar_flux_equation_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            # print("intf.mortar_to_primary_avg = ", intf.mortar_to_primary_avg())
            # print("intf.mortar_to_primary_int = ", intf.mortar_to_primary_int())

        # assert np.all(
        #     mortar == -np.array([1, 1, -1, -1])
        # )  # phase 0, delta p = 0, g = 1 # PASSED
        # assert np.all(
        #     mortar == -np.array([0.5, 0.5, -0.5, -0.5])
        # )  # phase 1, delta p = 0, g = 1 # PASSED # remember that this mortar flux is NOT the mass flux
        # assert np.all(
        #     mortar == -np.array([0, 0, 0, 0])
        # )  # phase 0 and 1,  delta p = 0, g = 0 # PASSED

        # assert np.all(
        #     mortar == -np.array([20, 20, 20, 20])
        # )  # phase 0 and 1, delta p = 1, g = 0 # PASSED

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            mass_flux = (
                self.interface_fluid_mass_flux_phase_0(
                    [intf], self.mixture.get_phase(0), "interface_mortar_flux_phase_0"
                )
                .evaluate(self.equation_system)
                .val
            )

        print(mass_flux)
        pdb.set_trace()

        # assert np.all(mortar == np.array([1, 1, -1, -1])) # phase 0, delta p = 0, g = 1 # TODO
        # assert np.all(
        #     mass_flux == -np.array([0, 0, 0, 0])
        # )  # phase 1, delta p = 0, g = 1 # PASSED
        # assert np.all(
        #     mortar == -np.array([0, 0, 0, 0])
        # )  # phase 0 and 1,  delta p = 0, g = 0 # TODO
        # assert np.all(
        #     mortar == -np.array([-20, -20, -20, -20])
        # )  # phase 0 and 1, delta p = 1, g = 0 # TODO. You have yo fix the mortar flux to unitary value

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
        self.size = 2 / self.units.m
        self._domain = pp.applications.md_grids.domains.nd_cube_domain(2, self.size)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[0.0, 2.0], [1.0, 1.0]]))
        self._fractures: list = [frac1]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 1 / self.units.m,
            "cell_size_fracture": 1 / self.units.m,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


# wetting_phase = pp.composite.phase.Phase(rho0=1)
# non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

# mixture = pp.Mixture()
# mixture.add([wetting_phase, non_wetting_phase])

mixture = test_hu_model.mixture  # ??? I had to copy the mixture to make it work


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassTest,
    # test_hu_model.SolutionStrategyPressureMass,
    MyModelGeometryTest,
    # test_hu_model.MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


class FinalModelTest(PartialFinalModel):
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = 0
        self.dynamic_viscosity = 1


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


model = FinalModelTest(mixture, params)


pp.run_time_dependent_model(model, params)
