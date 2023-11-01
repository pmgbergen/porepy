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
this shows an issue

REMEMBER TO SCALE p0 IN THE DENSITY EQUATION. one dimless group is p0/L0 => scale p0 => it is also in compressibility formula
the compressibility is the culprit of the issue!!!

"""




class SolutionStrategyPressureMass_1(test_hu_model.SolutionStrategyPressureMass):
    
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
            saturation_variable = (
                # self.mixture.mixture_for_subdomain(self.equation_system, sd)
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(self.ell)
                .saturation_operator([sd])
            )

            # saturation_variable = self.equation_system.md_variable(
            #     "saturation", [sd]
            # )

            saturation_values = 0.0*np.ones(sd.num_cells)
            # y_max = self.size  # TODO: not 100% sure size = y_max
            saturation_values[
                np.where(sd.cell_centers[0] >= self.xmax / 2)
            ] = 1.0  # TODO: HARDCODED for 2D

            if sd.dim == 1:
                saturation_values = 0.8*np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,  ### not totally clear, aqp you have to change both timestep values and iterate
            )  # TODO: i don't understand, this stores the value in data[pp.TIME_STEP_SOLUTION]["saturation"] without modifying the value of saturation

            pressure_variable = self.pressure([sd])

            pressure_values = 1./50 * ( 2 - 1 * sd.cell_centers[1] / self.ymax  )  

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

    def intrinsic_permeability(self, subdomains):
        """ """
        darcy_to_m2 = 1

        for sd in subdomains:
            permeability = 1./50 * 1e-4 * np.ones(sd.num_cells) * darcy_to_m2 # bottom # 100 mD
            # permeability[np.where(sd.cell_centers[1] >= self.ymax / 2)] = 50 * 1e-3 * darcy_to_m2 # top
            # permeability = 1 * np.ones(sd.num_cells)

        return pp.ad.DenseArray(permeability)


    def eb_after_timestep(self):
        """ """
        return
    
    

class BoundaryConditionsPressureMass_1:
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
                vals[boundary_faces] = 0  
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
                vals[boundary_faces] = 0 
                bc_values.append(vals)


        bc_values_array: pp.ad.DenseArray = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_flux"
        )

        return bc_values_array




class ModelGeometry_1(pp.ModelGeometry):
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
        bounding_box = {"xmin": 0, "xmax": self.xmax, "ymin": 0, "ymax": self.ymax} #, "zmin": 0, "zmax": 1}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        self._fractures: list = []

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 1. / 50 / self.units.m, 
            # "cell_size": 0.1 / self.units.m, 
            "cell_size_fracture": 0.05 / self.units.m,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMass_1,
    SolutionStrategyPressureMass_1,
    ModelGeometry_1,
    pp.DataSavingMixin,
):
    """ """

fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": None,
            "normal_permeability": 1, # this does NOT include the aperture
            "residual_aperture": 0.1,
        }
    )
# material_constants = {"solid": solid_constants}
material_constants = {"fluid": fluid_constants, "solid": solid_constants}

time_manager = pp.TimeManager(
    schedule=[0, 10],
    dt_init=1e-3,
    # schedule=[0, 1],
    # dt_init=1e-2,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 100,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e5,
    "time_manager": time_manager,
}


class FinalModel(PartialFinalModel):  
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = 10  # 
        self.dynamic_viscosity = 1e-1 # 1e-3 [kg / (m*s)] water        
        self.xmax = 50. / 50
        self.ymax = 50. / 50

        # TODO: add second model for region omega_beta where rho_beta=2.5
        self.relative_permeability = pp.tobedefined.relative_permeability.rel_perm_quadratic
        self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
        self.mobility_operator = pp.tobedefined.mobility.mobility_operator(self.mobility) 



wetting_phase = pp.composite.phase.Phase(rho0=1000) 
non_wetting_phase = pp.composite.phase.Phase(rho0=100) 


mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

model = FinalModel(mixture, params) 

pp.run_time_dependent_model(model, params)




