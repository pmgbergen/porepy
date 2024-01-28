import numpy as np
import scipy as sp
import scipy.sparse as sps
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model

import copy

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
OLD TEST

"""


class BoundaryConditionsPressureMassTest:
    def bc_neu_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0] < (0 + eps))[
                    0
                ]  # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0] > (self.xmax - eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = (
                    self.neu_val_left_phase_0 * sd.face_areas[left_boundary]
                )  # inlet
                vals[right_boundary] = (
                    self.neu_val_right_phase_0 * sd.face_areas[right_boundary]
                )  # outlet
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
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0] < (0 + eps))[
                    0
                ]  # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0] > (self.xmax - eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = (
                    self.neu_val_left_phase_1 * sd.face_areas[left_boundary]
                )  # inlet
                vals[right_boundary] = (
                    self.neu_val_right_phase_1 * sd.face_areas[right_boundary]
                )  # outlet
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

            if sd.dim == 1:
                saturation_values = None

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,  ### not totally clear, aqp you have to change both timestep values and iterate
            )  # TODO: i don't understand, this stores the value in data[pp.TIME_STEP_SOLUTION]["saturation"] without modifying the value of saturation

            pressure_variable = self.pressure([sd])

            # pressure_values = (
            #     2 - 1 * sd.cell_centers[1] / self.ymax
            # )  # TODO: hardcoded for 2D

            pressure_values = self.pressure_values_2d

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

    def eb_after_timestep(self):
        """ """
        print("self.time_manager.time = ", self.time_manager.time)
        if (
            np.isclose(np.mod(self.time_manager.time, 0.05), 0, rtol=0, atol=1e-4)
            or self.time_manager.time == self.time_manager.dt
        ):
            for sd, _ in self.mdg.subdomains(return_data=True):
                gigi = (
                    # self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    self.mixture.mixture_for_subdomain(sd)
                    .get_phase(0)
                    .saturation
                )
                gigi_1 = self.mixture.mixture_for_subdomain(sd).get_phase(1).saturation

                ppp = self.equation_system.md_variable("pressure", [sd]).evaluate(
                    self.equation_system
                )

                print("saturation = ", gigi.val)
                # print("pressure = ", ppp.val)

                if sd.dim > 0:  # otherwise plot_grid gets mad
                    pp.plot_grid(
                        sd,
                        gigi.val,
                        alpha=0.5,
                        info="",
                        title="saturation " + str(self.ell),
                    )
                    # pp.plot_grid(
                    #     sd,
                    #     gigi_1.val,
                    #     alpha=0.5,
                    #     info="",
                    #     title="saturation 1",
                    # )
                    # pp.plot_grid(sd, ppp.val, alpha=0.5, info="", title="pressure")

        # TEST: ---------------------------------------------------------------------------

        if self.case == 4:
            for subdomain in self.mdg.subdomains():  # again, where is "get subdomain"?
                if subdomain.dim == 2:
                    sd = subdomain

            ff0 = -self.neu_val_left_phase_0  # inlet boundary condition
            delta_m_ex_phase_0 = (
                ff0 * self.time_manager.dt_init
            )  # I assume dt is correct

            m_old_phase_0 = 2 * (0.25 * 1 * 1 * 0.5)
            m_new_phase_0 = np.sum(
                sd.cell_volumes
                * self.solid.porosity()
                * self.mixture.get_phase(0).mass_density(
                    self.pressure([sd]).evaluate(self.equation_system).val
                )
                * self.mixture.get_phase(0).saturation.val
            )
            delta_m_phase_0 = m_new_phase_0 - m_old_phase_0

            print("\ndelta_m_ex_phase_0 = ", delta_m_ex_phase_0)
            print("delta_m_phase_0 = ", delta_m_phase_0)

            assert np.isclose(delta_m_phase_0, delta_m_ex_phase_0, rtol=0, atol=1e-10)

        if self.case == 5:
            for subdomain in self.mdg.subdomains():  # again, where is "get subdomain"?
                if subdomain.dim == 2:
                    sd = subdomain

            ff0 = -self.neu_val_left_phase_0  # inlet boundary condition
            delta_m_ex_phase_0 = (
                ff0 * self.time_manager.dt_init
            )  # I assume dt is correct

            m_old_phase_0 = 2 * (0.25 * 1 * 1 * 0.5)
            m_new_phase_0 = np.sum(
                sd.cell_volumes
                * self.solid.porosity()
                * self.mixture.get_phase(0).mass_density(
                    self.pressure([sd]).evaluate(self.equation_system).val
                )
                * self.mixture.get_phase(0).saturation.val
            )
            delta_m_phase_0 = m_new_phase_0 - m_old_phase_0

            print("\ndelta_m_ex_phase_0 = ", delta_m_ex_phase_0)
            print("delta_m_phase_0 = ", delta_m_phase_0)

            assert np.isclose(delta_m_phase_0, delta_m_ex_phase_0, rtol=0, atol=1e-10)

        if self.case == 6:
            for subdomain in self.mdg.subdomains():  # again, where is "get subdomain"?
                if subdomain.dim == 2:
                    sd = subdomain

            ff0 = -2 * self.neu_val_left_phase_0  # inlet boundary condition
            delta_m_ex_phase_0 = (
                ff0 * self.time_manager.dt_init
            )  # I assume dt is correct

            m_old_phase_0 = 2 * (0.25 * 1 * 1 * 0.5)
            m_new_phase_0 = np.sum(
                sd.cell_volumes
                * self.solid.porosity()
                * self.mixture.get_phase(0).mass_density(
                    self.pressure([sd]).evaluate(self.equation_system).val
                )
                * self.mixture.get_phase(0).saturation.val
            )
            delta_m_phase_0 = m_new_phase_0 - m_old_phase_0

            print("\ndelta_m_ex_phase_0 = ", delta_m_ex_phase_0)
            print("delta_m_phase_0 = ", delta_m_phase_0)

            assert np.isclose(delta_m_phase_0, delta_m_ex_phase_0, rtol=0, atol=1e-10)

        if self.case == 7:
            for subdomain in self.mdg.subdomains():  # again, where is "get subdomain"?
                if subdomain.dim == 2:
                    sd = subdomain

            ff0 = -2 * self.neu_val_left_phase_0  # inlet boundary condition
            delta_m_ex_phase_0 = (
                ff0 * self.time_manager.dt_init
            )  # I assume dt is correct

            m_old_phase_0 = 2 * (0.25 * 1 * 1 * 0.5)
            m_new_phase_0 = np.sum(
                sd.cell_volumes
                * self.solid.porosity()
                * self.mixture.get_phase(0).mass_density(
                    self.pressure([sd]).evaluate(self.equation_system).val
                )
                * self.mixture.get_phase(0).saturation.val
            )
            delta_m_phase_0 = m_new_phase_0 - m_old_phase_0

            print("\ndelta_m_ex_phase_0 = ", delta_m_ex_phase_0)
            print("delta_m_phase_0 = ", delta_m_phase_0)

            assert np.isclose(delta_m_phase_0, delta_m_ex_phase_0, rtol=0, atol=1e-10)

        if self.case == 8:
            for subdomain in self.mdg.subdomains():  # again, where is "get subdomain"?
                if subdomain.dim == 2:
                    sd = subdomain

            ff0 = -self.neu_val_left_phase_0  # inlet boundary condition
            delta_m_ex_phase_0 = (
                ff0 * self.time_manager.dt_init
            )  # I assume dt is correct

            m_old_phase_0 = 2 * (0.25 * 1 * 1 * 0.5)
            m_new_phase_0 = np.sum(
                sd.cell_volumes
                * self.solid.porosity()
                * self.mixture.get_phase(0).mass_density(
                    self.pressure([sd]).evaluate(self.equation_system).val
                )
                * self.mixture.get_phase(0).saturation.val
            )
            delta_m_phase_0 = m_new_phase_0 - m_old_phase_0

            print("\ndelta_m_ex_phase_0 = ", delta_m_ex_phase_0)
            print("delta_m_phase_0 = ", delta_m_phase_0)

            assert np.isclose(delta_m_phase_0, delta_m_ex_phase_0, rtol=0, atol=1e-10)

        print("\n\n\n TEST PASSED ----------------------------------------")

        pdb.set_trace()


class MyModelGeometry(pp.ModelGeometry):
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
        self.size = 1 / self.units.m

        # structired square:
        # self._domain = pp.applications.md_grids.domains.nd_cube_domain(2, self.size)

        # unstructred unit square:
        bounding_box = {
            "xmin": 0,
            "xmax": self.xmax,
            "ymin": 0,
            "ymax": self.ymax,
        }  # , "zmin": 0, "zmax": 1}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        self._fractures: list = []  # [frac1]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 1 / self.units.m,  # 0.05
            "cell_size_fracture": 0.1 / self.units.m,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMassTest,
    SolutionStrategyPressureMassTest,
    MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
    {
        "porosity": 0.25,
        "permeability": 1,
        "normal_permeability": 1,  # this does NOT include the aperture
        "residual_aperture": 0.1,
    }
)
# material_constants = {"solid": solid_constants}
material_constants = {"fluid": fluid_constants, "solid": solid_constants}

time_manager = pp.TimeManager(
    schedule=[0, 10],
    dt_init=1e-1,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 100,
    "nl_convergence_tol": 1e-11,
    "nl_divergence_tol": 1e5,
    "time_manager": time_manager,
}


class FinalModelNeu(PartialFinalModel):  # I'm sorry...
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0
        self.gravity_value = 1
        self.dynamic_viscosity = 1

        self.case = None
        self.xmax = None
        self.ymax = None

        self.saturation_values_2d = None

        self.neu_val_left_phase_0 = None
        self.neu_val_right_phase_0 = None
        self.neu_val_left_phase_1 = None
        self.neu_val_right_phase_1 = None


wetting_phase = pp.composite.phase.Phase(rho0=1)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

model = FinalModelNeu(
    mixture, params
)  # why... add it directly as attribute: model.mixture = mixture


case = 4  # cases 1,2,3 only graphical. cases 4,5,... realt tests
model.case = case

# inlet < 1
# outlet > 1


"""
NOTE: 
- newton converges to a non-physical solution for high dt. IT CONVERGES. Sometimes.
- missing tests at successive timesteps


"""


if case == 1:
    model.xmax = 2
    model.ymax = 1
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.ones(
        model.xmax * model.ymax
    )  # dont hate me, please, I have my issues...
    # model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = 0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 0

    # model.time_manager.dt_init = 1e-2 # YOU WISH! Why can't I modify the timestep after the creation of the time manager?

if case == 2:
    model.xmax = 2
    model.ymax = 1
    model.gravity_value = 0
    model.saturation_values_2d = 1 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = 0
    model.neu_val_right_phase_0 = 1
    model.neu_val_left_phase_1 = -1
    model.neu_val_right_phase_1 = 0

if case == 3:
    model.xmax = 3
    model.ymax = 1
    model.gravity_value = 0
    model.saturation_values_2d = 1 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1  # immetto fase presente
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 1  # tolgo quella assente
    model.neu_val_right_phase_1 = 0

    # ottengo valori non fisici, esattamente come ci si aspetta, E NON SI PROPAGANO


if case == 4:  # horizontal
    model.xmax = 2
    model.ymax = 1
    model.gravity_value = 1
    model.saturation_values_2d = 0.5 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1.0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 1.0


if case == 5:  # horizontal
    model.xmax = 2
    model.ymax = 1
    model.gravity_value = 0
    model.saturation_values_2d = 0.5 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1.0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 1.0


if case == 6:  # vertical
    model.xmax = 1
    model.ymax = 2
    model.gravity_value = 1
    model.saturation_values_2d = 0.5 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1.0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 1


if case == 7:  # vertical
    model.xmax = 1
    model.ymax = 2
    model.gravity_value = 0
    model.saturation_values_2d = 0.5 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1.0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 1


if case == 8:  # not physical
    model.xmax = 2
    model.ymax = 1
    model.gravity_value = 0
    model.saturation_values_2d = 0.5 * np.ones(model.xmax * model.ymax)
    model.pressure_values_2d = 1 * np.ones(model.xmax * model.ymax)

    model.neu_val_left_phase_0 = -1.0
    model.neu_val_right_phase_0 = 0
    model.neu_val_left_phase_1 = 0
    model.neu_val_right_phase_1 = 0


pp.run_time_dependent_model(model, params)
