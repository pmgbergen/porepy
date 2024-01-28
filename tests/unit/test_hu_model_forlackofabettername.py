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

- from test_hu_model_mortar_fluid_flux.py

- I'd like to test each term of the equation but:
 idk how to test d/dt
 hu flux alredy tested
 interface fluxes tested
 there is only the source left that in 2D is 0 and in 1D has to be equal to the delta flux. 
 This is what I added in this test 

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

        self.set_discretization_parameters()
        self.rediscretize()

        # TEST: ---------------------------------------------------------------------------

        # # open the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         for i in dir(sd):
        #             print(i)

        #         internal_nodes_id = sd.get_internal_nodes()
        #         sd.nodes[1][internal_nodes_id[1]] -= 0.1
        #         sd.nodes[1][internal_nodes_id[0]] += 0.1

        # self.mdg.compute_geometry()

        # pp.plot_grid(self.mdg, info="cf", alpha=0)

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

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            mortar_initialized = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            assert np.all(mortar_initialized == np.array([0, 0, 0, 0]))

            mortar_initialized = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            assert np.all(mortar_initialized == np.array([0, 0, 0, 0]))

            # pay attention, you are goning to evaluate the equation "lamnda - rhs = ...". Since lambda = 0 ny initialization the evaluation of thr equations is -mortar
            # that's why i'll write -np.array([...
            # NO, please, put a minus in mortar

            # notations:
            # (see upwind.py) boundary flux >0 => outwards, so boundary flux <0 corrensponds to accumulation
            # mortar >0 if from higher to lower
            # so mortar > 0 => fluid flux >0 => exiting boundary flux => a minus in the conservation law: div(F-boundary_flux)

            mortar_phase_0 = -(
                self.interface_mortar_flux_equation_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )

            mortar_phase_1 = -(
                self.interface_mortar_flux_equation_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )

            fluid_flux_phase_0 = (
                self.interface_fluid_mass_flux_phase_0(
                    pp.ad.DenseArray(mortar_phase_0),
                    [intf],
                    self.mixture.get_phase(0),
                    "interface_mortar_flux_phase_0",
                )
                .evaluate(self.equation_system)
                .val
            )

            fluid_flux_phase_1 = (
                self.interface_fluid_mass_flux_phase_1(
                    pp.ad.DenseArray(mortar_phase_1),
                    [intf],
                    self.mixture.get_phase(1),
                    "interface_mortar_flux_phase_1",
                )
                .evaluate(self.equation_system)
                .val
            )



            # mi sono rotto:
            self.equation_system.set_variable_values(
                mortar_phase_0,
                variables=[self.interface_mortar_flux_phase_0_variable],
                time_step_index=0,
                iterate_index=0,
            )

            self.equation_system.set_variable_values(
                mortar_phase_1,
                variables=[self.interface_mortar_flux_phase_1_variable],
                time_step_index=0,
                iterate_index=0,
            )


            div = pp.ad.Divergence(self.mdg.subdomains(), dim=1)

            # PRESSURE EQ: -------------------------
            ( 
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0_p,
            flux_intf_phase_1_p,
            flux,
            source_phase_0_p,
            source_phase_1_p,
            source, 
            ) = self.eq_fcn_pressure(self.mdg.subdomains())

            flux_intf_phase_0_p = ( div @ flux_intf_phase_0_p ).evaluate(self.equation_system).val[[0, 1, 2, 3]] 
            flux_intf_phase_1_p = ( div @ flux_intf_phase_1_p ).evaluate(self.equation_system).val[[0, 1, 2, 3]]

            source_phase_0_p_2d = source_phase_0_p.evaluate(self.equation_system).val[[0, 1, 2, 3]]
            source_phase_1_p_2d = source_phase_1_p.evaluate(self.equation_system).val[[0, 1, 2, 3]]
            
            source_phase_0_p_1d = source_phase_0_p.evaluate(self.equation_system).val[[4, 5]]
            source_phase_1_p_1d = source_phase_1_p.evaluate(self.equation_system).val[[4, 5]]



            # MASS BAL: ----------------------------
            (
            eq,
            accumulation,
            rho_V, 
            rho_G,
            flux_V_G,
            flux_intf_phase_0_m,
            flux_intf_phase_1_m,
            source_phase_0_m,
            source_phase_1_m,
            ) = self.eq_fcn_mass(self.mdg.subdomains())

            flux_intf_phase_0_m = ( div @ flux_intf_phase_0_m ).evaluate(self.equation_system).val[[0, 1, 2, 3]]
            flux_intf_phase_1_m = ( div @ flux_intf_phase_1_m ).evaluate(self.equation_system).val[[0, 1, 2, 3]]

            source_phase_0_m_2d = source_phase_0_m.evaluate(self.equation_system).val[[0, 1, 2, 3]]
            source_phase_1_m_2d = source_phase_1_m.evaluate(self.equation_system).val[[0, 1, 2, 3]]
            
            source_phase_0_m_1d = source_phase_0_m.evaluate(self.equation_system).val[[4, 5]]
            source_phase_1_m_1d = source_phase_1_m.evaluate(self.equation_system).val[[4, 5]]


            # the following has to be always true, and each term depends on fluid flux, so then only the fluid flux in the cases
            assert np.any( source_phase_0_p_2d ==  np.array([0, 0, 0, 0]) )
            assert np.any( source_phase_1_p_2d ==  np.array([0, 0, 0, 0]) )
            assert np.any( source_phase_0_m_2d ==  np.array([0, 0, 0, 0]) )
            assert np.any( source_phase_1_m_2d ==  np.array([0, 0, 0, 0]) )

            assert np.any( np.isclose( source_phase_0_p_1d[0], (flux_intf_phase_0_p[0] + flux_intf_phase_0_p[2]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_0_p_1d[1], (flux_intf_phase_0_p[1] + flux_intf_phase_0_p[3]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_1_p_1d[0], (flux_intf_phase_1_p[0] + flux_intf_phase_1_p[2]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_1_p_1d[1], (flux_intf_phase_1_p[1] + flux_intf_phase_1_p[3]), rtol=0, atol=1e-10) )

            assert np.any( np.isclose( source_phase_0_m_1d[0], (flux_intf_phase_0_m[0] + flux_intf_phase_0_m[2]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_0_m_1d[1], (flux_intf_phase_0_m[1] + flux_intf_phase_0_m[3]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_1_m_1d[0], (flux_intf_phase_1_m[0] + flux_intf_phase_1_m[2]), rtol=0, atol=1e-10) )
            assert np.any( np.isclose( source_phase_1_m_1d[1], (flux_intf_phase_1_m[1] + flux_intf_phase_1_m[3]), rtol=0, atol=1e-10) )

            assert np.any(flux_intf_phase_0_p == fluid_flux_phase_0)
            assert np.any(flux_intf_phase_1_p == fluid_flux_phase_1)
            assert np.any(flux_intf_phase_0_m == fluid_flux_phase_0)
            assert np.any(flux_intf_phase_1_m == fluid_flux_phase_1)


            print("\n\n\n")
            print("mortar_phase_0 = ", mortar_phase_0)
            print("mortar_phase_1 = ", mortar_phase_1)
            print("fluid_flux_phase_0 = ", fluid_flux_phase_0)
            print("fluid_flux_phase_1 = ", fluid_flux_phase_1)

            pdb.set_trace()

            if self.case == 1:  # delta p = 0, g = 1, s0=0
                rho_mob_phase_0 = 0
                rho_mob_phase_1 = 0.5

                assert np.all(
                    np.isclose(
                        mortar_phase_0, np.array([1, 1, -1, -1]), rtol=0, atol=1e-10
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([1, 1, -1, -1]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([0.5, 0.5, -0.5, -0.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([0.5, 0.5, -0.5, -0.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 2:  # delta p = 0, g = 1, s0=1
                rho_mob_phase_0 = 1
                rho_mob_phase_1 = 0

                assert np.all(
                    np.isclose(
                        mortar_phase_0, np.array([1, 1, -1, -1]), rtol=0, atol=1e-10
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([1, 1, -1, -1]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([0.5, 0.5, -0.5, -0.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([0.5, 0.5, -0.5, -0.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 3:  # delta p = 1, g = 0, s0=0
                rho_mob_phase_0 = 0
                rho_mob_phase_1 = 0.5

                assert np.all(
                    np.isclose(
                        mortar_phase_0, np.array([20, 20, 20, 20]), rtol=0, atol=1e-10
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 4:  # delta p = 1, g = 0, s0=1
                rho_mob_phase_0 = 1
                rho_mob_phase_1 = 0

                assert np.all(
                    np.isclose(
                        mortar_phase_0, np.array([20, 20, 20, 20]), rtol=0, atol=1e-10
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([20, 20, 20, 20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 5:  # delta p = -1, g = 0, s0=0
                rho_mob_phase_0 = 0
                rho_mob_phase_1 = 0.5

                assert np.all(
                    np.isclose(
                        mortar_phase_0,
                        np.array([-20, -20, -20, -20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([-20, -20, -20, -20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([-20, -20, -20, -20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([-20, -20, -20, -20]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 6:  # delta p = 1, g = 1, s0=0
                rho_mob_phase_0 = 0
                rho_mob_phase_1 = 0.5

                assert np.all(
                    np.isclose(
                        mortar_phase_0,
                        np.array([21, 21, 19, 19]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([21, 21, 19, 19]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([20.5, 20.5, 19.5, 19.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([20.5, 20.5, 19.5, 19.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 6:  # delta p = 1, g = 1, s0=0
                rho_mob_phase_0 = 1
                rho_mob_phase_1 = 0

                assert np.all(
                    np.isclose(
                        mortar_phase_0,
                        np.array([21, 21, 19, 19]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([21, 21, 19, 19]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([20.5, 20.5, 19.5, 19.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([20.5, 20.5, 19.5, 19.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

            if self.case == 8:  # delta p = -1, g = 1, s0=1
                rho_mob_phase_0 = 1
                rho_mob_phase_1 = 0

                assert np.all(
                    np.isclose(
                        mortar_phase_0,
                        np.array([-19, -19, -21, -21]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_0,
                        rho_mob_phase_0 * np.array([-19, -19, -21, -21]),
                        rtol=0,
                        atol=1e-10,
                    )
                )

                assert np.all(
                    np.isclose(
                        mortar_phase_1,
                        np.array([-19.5, -19.5, -20.5, -20.5]),
                        rtol=0,
                        atol=1e-10,
                    )
                )
                assert np.all(
                    np.isclose(
                        fluid_flux_phase_1,
                        rho_mob_phase_1 * np.array([-19.5, -19.5, -20.5, -20.5]),
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
        self.gravity_value = None  
        self.dynamic_viscosity = 1  

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


case = 2
model.case = case

if case == 1:  # delta p = 0, g = 1
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])


if case == 2:  # delta p = 0, g = 1
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1, 1.0, 1, 1])
    model.pressure_values_1d = 1 * np.array([1, 1.0])

if case == 3:  # delta p = 1, g = 0
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 4:  # delta p = 1, g = 0
    model.gravity_value = 0
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 5:  # delta p = -1, g = 0
    model.gravity_value = 0
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])
    # delta p negative

if case == 6:  # delta p = 1, g = 1
    model.gravity_value = 1
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 7:  # delta p = 1, g = 1
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 0 * np.array([1.0, 1])

if case == 8:  # delta p = -1, g = 1
    model.gravity_value = 1
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1])
    model.pressure_values_2d = 0 * np.array([1.0, 1, 1, 1])
    model.pressure_values_1d = 1 * np.array([1.0, 1])


pp.run_time_dependent_model(model, params)
