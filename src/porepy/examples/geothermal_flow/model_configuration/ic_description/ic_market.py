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
        p_inlet = 50.0e6
        p_outlet = 25.0e6
        domain_length = 2000.0  # in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:

        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h_init

    def ic_values_enthalpy_const(self, sd: pp.Grid) -> np.ndarray:
        """Constant enthalpy for isothermal system."""
        # For constant properties at T = 423.15 K:
        # h = cp * T (approximately)
        cp = 4186.0  # J/(kg·K) - specific heat of water
        T = 423.15   # K (150°C)
        h_init = cp * T * np.ones(sd.num_cells)
        
        return h_init
    
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init
    