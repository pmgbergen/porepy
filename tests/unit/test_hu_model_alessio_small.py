import numpy as np
import scipy as sp
import scipy.sparse as sps
import porepy as pp
import test_hu_model

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import copy

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""

"""

class BoundaryConditionsPressureMassTest:
    def bc_neu_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf
                
                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0]<(0+eps))[0] # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0]>(self.xmax-eps))[0] # add xmax as in the tests pls
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0.0 * sd.face_areas[left_boundary] # inlet
                vals[right_boundary] = 0.0 * sd.face_areas[right_boundary] # outlet
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
        """
        """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf
                
                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0]<(0+eps))[0] # add xmin as in the tests pls
                right_boundary = np.where(sd.face_centers[0]>(self.xmax-eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0.0 * sd.face_areas[left_boundary] # inlet
                vals[right_boundary] = 0.0 * sd.face_areas[right_boundary] # outlet
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
    """ """
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

                saturation_values = 0.0*np.ones(sd.num_cells)
                saturation_values[
                    np.where(sd.cell_centers[1] >= self.ymax / 2)
                ] = 1.0 

                if sd.dim == 1:
                    saturation_values = 0.8*np.ones(sd.num_cells)

                self.equation_system.set_variable_values(
                    saturation_values,
                    variables=[saturation_variable],
                    time_step_index=0,
                    iterate_index=0,  
                ) 

                pressure_variable = self.pressure([sd])
                pressure_values = ( 2 - 1 * sd.cell_centers[1] / self.ymax ) 

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



    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """ """

        # for sd in self.mdg.subdomains():
        #     pp.plot_grid(sd, info="fc", alpha=0)
        #     for i in dir(sd):
        #         print(i)
        #     pdb.set_trace()
        # # face 5 cell 1
        # # face 7 cell 0 
        # # => 5 is upper face, 7 is lower face

        # for intf in self.mdg.interfaces():
        #     for i in dir(intf):
        #         print(i)
        #     pdb.set_trace()

        # # face 5 mortar cell 0
        # # face 7 mortar cell 1
        # # => mortar cell 0 is the top one, 1 the bottom


        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()

        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )

        # TEST HERE:

        div = pp.ad.Divergence(self.mdg.subdomains(), dim=1)

        (
            _,
            _,
            flux_tot,
            flux_intf_phase_0_p,
            flux_intf_phase_1_p,
            _,
            _,
            _,
            _,
        ) = self.eq_fcn_pressure(self.mdg.subdomains())

        # flux_tot_cell_1 = flux_tot.evaluate(self.equation_system).val[[1,2,9,17]] # faces of cell 1
        # flux_tot_cell_4 = flux_tot.evaluate(self.equation_system).val[[5,6,12,15]] # faces of cell 4
        flux_intf_phase_0_p_val = -flux_intf_phase_0_p.evaluate(self.equation_system).val # minus because of notation for mortars
        flux_intf_phase_1_p_val = -flux_intf_phase_1_p.evaluate(self.equation_system).val # [upper face, lower face]

        (
            _,
            _,
            rho_V_operator,
            rho_G_operator,
            _,
            flux_intf_phase_0_m,
            flux_intf_phase_1_m,
            _,
            _,
        ) = self.eq_fcn_mass(self.mdg.subdomains())


        # rho_V_operator_cell_1 = rho_V_operator.evaluate(self.equation_system).val[[1,2,9,17]]
        # rho_V_operator_cell_4 = rho_V_operator.evaluate(self.equation_system).val[[5,6,12,15]]
        # rho_G_operator_cell_1 = rho_G_operator.evaluate(self.equation_system).val[[1,2,9,17]]
        # rho_G_operator_cell_4 = rho_G_operator.evaluate(self.equation_system).val[[5,6,12,15]]
        # flux_intf_phase_0_m_val = -flux_intf_phase_0_m.evaluate(self.equation_system).val[[12,17]] # ok, fluxes from p eq and mass bal are the same
        # flux_intf_phase_1_m_val = -flux_intf_phase_1_m.evaluate(self.equation_system).val[[12,17]]


        # print("\nflux_tot_cell_1 = ", flux_tot_cell_1)
        # print("flux_tot_cell_4 = ", flux_tot_cell_4)
        # print("\nrho_V_operator_cell_1 = ", rho_V_operator_cell_1)
        # print("rho_V_operator_cell_4 = ", rho_V_operator_cell_4)
        # print("\nrho_G_operator_cell_1 = ", rho_G_operator_cell_1)
        # print("rho_G_operator_cell_4 = ", rho_G_operator_cell_4)


        print("\nflux_intf_phase_0_p_val = ", flux_intf_phase_0_p_val)
        print("flux_intf_phase_1_p_val = ", flux_intf_phase_1_p_val)

        # print("\nflux_intf_phase_0_m_val = ", flux_intf_phase_0_m_val)
        # print("flux_intf_phase_1_m_val = ", flux_intf_phase_1_m_val)


        # div_flux_0_cell_1_4 = -( div @ flux_intf_phase_0_p ).evaluate(self.equation_system).val[[1,4]] # you know why there is a minus # contribution to cell 1 and 4. positive if exiting the cell
        # div_flux_1_cell_1_4 = -( div @ flux_intf_phase_1_p ).evaluate(self.equation_system).val[[1,4]]

        # print("\ndiv_flux_0_cell_1_4 = ", div_flux_0_cell_1_4)
        # print("div_flux_1_cell_1_4 = ", div_flux_1_cell_1_4)


        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         saturation_0 = self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation.val
        #         saturation_1 = self.mixture.mixture_for_subdomain(sd).get_phase(1).saturation.val
        #     if sd.dim == 1:
        #         saturation_0_frac = self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation.val
        #         saturation_1_frac = self.mixture.mixture_for_subdomain(sd).get_phase(1).saturation.val

        # print("\nsaturaton_0 = ", saturation_0)
        # print("saturaton_1 = ", saturation_1)
        # print("saturation_0 + saturation_1 = ", saturation_0 + saturation_1)

        # print("\nsaturaton_0_frac = ", saturation_0_frac)
        # print("saturaton_1_frac = ", saturation_1_frac)
        # print("saturation_0_frac + saturation_1_frac = ", saturation_0_frac + saturation_1_frac)


        # print("\n==============================")
        # (eq, kn, pressure_h, pressure_l, density_upwinded, g_term) = self.eq_fcn_mortar_phase_0(self.mdg.interfaces())
        # print("\n density_upwinded = ", density_upwinded.evaluate(self.equation_system).val)
        # print("\n g_term = ", g_term.evaluate(self.equation_system)) 


        np.set_printoptions(precision=3, linewidth=sys.maxsize)
        A, b = self.linear_system
        A = A.todense()
        print("A = ", A)
        print("b = ", b)
        print("increment = ", np.linalg.solve(A, b))

        # self.save_data_time_step()
        
        # pdb.set_trace()





    def eb_after_timestep(self):
        """ """
        print("self.time_manager.time = ", self.time_manager.time)

        # TEST HERE:

        """
        div = pp.ad.Divergence(self.mdg.subdomains(), dim=1)

        (
            _,
            _,
            flux_tot,
            flux_intf_phase_0_p,
            flux_intf_phase_1_p,
            _,
            _,
            _,
            _,
        ) = self.eq_fcn_pressure(self.mdg.subdomains())

        # flux_tot_cell_1 = flux_tot.evaluate(self.equation_system).val[[1,2,9,17]] # faces of cell 1
        # flux_tot_cell_4 = flux_tot.evaluate(self.equation_system).val[[5,6,12,15]] # faces of cell 4
        # flux_intf_phase_0_p_val = -flux_intf_phase_0_p.evaluate(self.equation_system).val[[12,17]] # minus because of notation for mortars
        # flux_intf_phase_1_p_val = -flux_intf_phase_1_p.evaluate(self.equation_system).val[[12,17]] # [upper face, lower face]

        (
            _,
            _,
            rho_V_operator,
            rho_G_operator,
            _,
            flux_intf_phase_0_m,
            flux_intf_phase_1_m,
            _,
            _,
        ) = self.eq_fcn_mass(self.mdg.subdomains())


        # rho_V_operator_cell_1 = rho_V_operator.evaluate(self.equation_system).val[[1,2,9,17]]
        # rho_V_operator_cell_4 = rho_V_operator.evaluate(self.equation_system).val[[5,6,12,15]]
        # rho_G_operator_cell_1 = rho_G_operator.evaluate(self.equation_system).val[[1,2,9,17]]
        # rho_G_operator_cell_4 = rho_G_operator.evaluate(self.equation_system).val[[5,6,12,15]]
        # flux_intf_phase_0_m_val = -flux_intf_phase_0_m.evaluate(self.equation_system).val[[12,17]] # ok, fluxes from p eq and mass bal are the same
        # flux_intf_phase_1_m_val = -flux_intf_phase_1_m.evaluate(self.equation_system).val[[12,17]]


        # print("\nflux_tot_cell_1 = ", flux_tot_cell_1)
        # print("flux_tot_cell_4 = ", flux_tot_cell_4)
        # print("\nrho_V_operator_cell_1 = ", rho_V_operator_cell_1)
        # print("rho_V_operator_cell_4 = ", rho_V_operator_cell_4)
        # print("\nrho_G_operator_cell_1 = ", rho_G_operator_cell_1)
        # print("rho_G_operator_cell_4 = ", rho_G_operator_cell_4)


        # print("\nflux_intf_phase_0_p_val = ", flux_intf_phase_0_p_val)
        # print("flux_intf_phase_1_p_val = ", flux_intf_phase_1_p_val)

        # print("\nflux_intf_phase_0_m_val = ", flux_intf_phase_0_m_val)
        # print("flux_intf_phase_1_m_val = ", flux_intf_phase_1_m_val)


        # div_flux_0_cell_1_4 = -( div @ flux_intf_phase_0_p ).evaluate(self.equation_system).val[[1,4]] # you know why there is a minus # contribution to cell 1 and 4. positive if exiting the cell
        # div_flux_1_cell_1_4 = -( div @ flux_intf_phase_1_p ).evaluate(self.equation_system).val[[1,4]]

        # print("\ndiv_flux_0_cell_1_4 = ", div_flux_0_cell_1_4)
        # print("div_flux_1_cell_1_4 = ", div_flux_1_cell_1_4)


        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         saturation_0 = self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation.val
        #         saturation_1 = self.mixture.mixture_for_subdomain(sd).get_phase(1).saturation.val
        #     if sd.dim == 1:
        #         saturation_0_frac = self.mixture.mixture_for_subdomain(sd).get_phase(0).saturation.val
        #         saturation_1_frac = self.mixture.mixture_for_subdomain(sd).get_phase(1).saturation.val

        # print("\nsaturaton_0 = ", saturation_0)
        # print("saturaton_1 = ", saturation_1)
        # print("saturation_0 + saturation_1 = ", saturation_0 + saturation_1)

        # print("\nsaturaton_0_frac = ", saturation_0_frac)
        # print("saturaton_1_frac = ", saturation_1_frac)
        # print("saturation_0_frac + saturation_1_frac = ", saturation_0_frac + saturation_1_frac)


        # print("\n==============================")
        # (eq, kn, pressure_h, pressure_l, density_upwinded, g_term) = self.eq_fcn_mortar_phase_0(self.mdg.interfaces())
        # print("\n density_upwinded = ", density_upwinded.evaluate(self.equation_system).val)
        # print("\n g_term = ", g_term.evaluate(self.equation_system)) 
        # # pdb.set_trace()

        """ 
        

class MyModelGeometryTest(pp.ModelGeometry):
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
        self.size = 1.

        bounding_box = {"xmin": 0, "xmax": self.xmax, "ymin": 0, "ymax": self.ymax} 
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture( np.array([[0., 1], [1, 1]]) )
        self._fractures: list = [frac1]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 1.,
            "cell_size_fracture": 1.,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMassTest,
    SolutionStrategyPressureMassTest,
    MyModelGeometryTest,
    pp.DataSavingMixin,
):
    """ """

fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": 1,
            "normal_permeability": 1, # this does NOT include the aperture
            "residual_aperture": 1e9, ### tests for different aperture??    ###################################################################################################################
        }
    )

material_constants = {"fluid": fluid_constants, "solid": solid_constants}

time_manager = pp.TimeManager(
    schedule=[0, 30],
    dt_init=1e-1,  
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 15,
    "nl_convergence_tol": 1e-6,
    "nl_divergence_tol": 1e5,
    "time_manager": time_manager,
}



class FinalModel(PartialFinalModel): 
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = 1 
        self.dynamic_viscosity = 1  
        
        self.xmax = 1
        self.ymax = 2 

        self.relative_permeability = pp.tobedefined.relative_permeability.rel_perm_quadratic
        self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability) 
        self.mobility_operator = pp.tobedefined.mobility.mobility_operator(self.mobility)

wetting_phase = pp.composite.phase.Phase(rho0=1, beta=1e-10)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5, beta=1e-10)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

model = FinalModel(mixture, params)

pp.run_time_dependent_model(model, params)




