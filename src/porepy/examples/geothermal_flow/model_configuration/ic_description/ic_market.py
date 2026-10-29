import numpy as np

import porepy as pp
from ...vtk_sampler import VTKSampler


class ICSinglePhaseHighPressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for liquid fluid flow
        """
        p_inlet = 14.0e6
        p_outlet = 10.0e6

#       old pressure values for testing
#        p_inlet = 35.0e6
#        p_outlet = 5.0e6

        domain_length = self.domain.bounding_box["xmax"]  # in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init

    def ic_values_salt(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial salt concentration distribution """
        z_NaCl_inlet = 0.3  # [mass fraction]
        z_NaCl_outlet = 0.3  # [mass fraction]
        domain_length = self.domain.bounding_box["xmax"]  # in m
        cell_centers_x = sd.cell_centers[0]
        salt_gradient = (z_NaCl_outlet - z_NaCl_inlet) / domain_length
        z_NaCl_init = z_NaCl_inlet + salt_gradient * cell_centers_x
        return z_NaCl_init    

    
    # def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
    #     p = self.ic_values_pressure(sd)
    #     t = self.ic_values_temperature(sd)
    #     z_NaCl = np.zeros_like(p)
    #     assert len(p) == len(t) == len(z_NaCl)
    #     par_points = np.array((z_NaCl, t, p)).T
    #     self.vtk_sampler_ptz.sample_at(par_points)
    #     h_init = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
    #     return h_init

    # def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
    #     t_init = 423.15  # [K]
    #     return np.ones(sd.num_cells) * t_init

    def ic_values_overall_fraction(self, component: pp.Component, sd: pp.Grid) -> np.ndarray:
        """Defining a constant fraction for values on all initial conditions"""
        z_NaCl_init = self.ic_values_salt(sd)
        if component.name == "NaCl":
            return z_NaCl_init
        return 1.0 - z_NaCl_init
    
