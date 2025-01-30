import numpy as np

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import InitialConditionsCF


class ICSinglePhaseHighPressure(InitialConditionsCF):

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for liquid fluid flow
        """
        p_inlet = 50.0e6
        p_outlet = 25.0e6
        domain_length = 2000.0 #in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:

        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15 #[K]
        return np.ones(sd.num_cells) * t_init
    
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)
        

class ICSinglePhaseModeratePressure(InitialConditionsCF):
    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for supercritical fluid flow
        """

        p_inlet = 40.0e6
        p_outlet = 20.0e6
        domain_length = 2000.0 #in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:

        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
 
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 573.15 #[K]
        return np.ones(sd.num_cells) * t_init

   
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class ICSinglePhaseLowPressure(InitialConditionsCF):
    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for vapor phase flow
        """
        p_inlet = 15.0e6
        p_outlet = 1.0e6
        domain_length = 2000.0 #in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = 0.0*np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']

        return h_init
    
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 623.15 #[K]
        return np.ones(sd.num_cells) * t_init
    
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)
class ICTwoPhaseHighPressure(InitialConditionsCF):
    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.
           
           Initial condition setup for short two-phase liquid+vapor phase flow 
        """
        p_inlet = 20.0e6
        p_outlet = 1.0e6
        domain_length = 2000.0  # in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = 0.0 * np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init
    
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class ICTwoPhaseLowPressure(InitialConditionsCF):
    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for long two-phase liquid+vapor phase flow 
        """
        p_inlet = 4.0e6
        p_outlet = 1.0e6
        domain_length = 2000.0 #in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init

    def initial_enthalpy(self, sd: 'pp.Grid') -> np.ndarray:
        """Compute the initial specific enthalpy for each cell in the grid using CoolProp."""
        
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init
    
    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)
