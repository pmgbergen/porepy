import numpy as np

from typing import Tuple

import porepy as pp

from ...vtk_sampler import VTKSampler

from scipy.optimize import root_scalar


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
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init
    
    # def ic_values_overall_fraction(
    #     self, component: pp.Component, sd: pp.Grid
    # ) -> np.ndarray:
    #     z = 0.0
    #     if component.name == "H2O":
    #         return (1 - z) * np.ones(sd.num_cells)
    #     else:
    #         return z * np.ones(sd.num_cells)
        

class ICSinglePhaseModeratePressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for supercritical fluid flow
        """

        p_inlet = 40.0e6
        p_outlet = 20.0e6
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
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
 
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 573.15  # [K]
        return np.ones(sd.num_cells) * t_init
 
    # def ic_values_overall_fraction(
    #     self, component: pp.Component, sd: pp.Grid
    # ) -> np.ndarray:
    #     z = 0.0
    #     if component.name == "H2O":
    #         return (1 - z) * np.ones(sd.num_cells)
    #     else:
    #         return z * np.ones(sd.num_cells)

        
class ICSinglePhaseLowPressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for vapor phase flow
        """
        p_inlet = 15.0e6
        p_outlet = 1.0e6
        domain_length = 2000.0  # in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init
    
    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = 0.0*np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']

        return h_init
    
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 623.15  # [K]
        return np.ones(sd.num_cells) * t_init
    
    # def ic_values_overall_fraction(
    #     self, component: pp.Component, sd: pp.Grid
    # ) -> np.ndarray:
    #     z = 0.0
    #     if component.name == "H2O":
    #         return (1 - z) * np.ones(sd.num_cells)
    #     else:
    #         return z * np.ones(sd.num_cells)
    

class ICTwoPhaseHighPressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
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
   
    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = 0.0 * np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init
    

class ICTwoPhaseLowPressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.

           Initial condition setup for long two-phase liquid+vapor phase flow 
        """
        p_inlet = 4.0e6
        p_outlet = 1.0e6
        domain_length = 2000.0  # in m
        cell_centers_x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length
        p_init = p_inlet + pressure_gradient * cell_centers_x
        return p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial specific enthalpy for each cell in the grid using CoolProp."""
        
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init
    
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init


def find_z_for_target_sh(
    p_values: np.ndarray,
    T0: float,
    sh_target: float,
    sampler: "VTKSampler",
    z_bounds: Tuple[float, float] = (0.0, 0.2)
) -> np.ndarray:
    """
    For each pressure value, find the NaCl mass fraction z such that the halite saturation S_h equals a target value.

    Parameters:
        p_values: Array of pressure values for each grid cell.
        T0: Fixed temperature (K) used for all cells.
        sh_target: Target halite saturation value to match.
        sampler: VTKSampler
            Sampler instance that provides S_h and its gradient.
        z_bounds: Bounds within which to search for z (default: (0.0, 0.2)).

    Returns:
        np.ndarray of shape (N,)
            Array of z values where S_h ≈ sh_target, or np.nan if no root is found.
    """
    N: int = len(p_values)
    z_solutions: np.ndarray = np.full(N, np.nan)

    for i, p_i in enumerate(p_values):
        def f(z: float) -> float:
            par_point: np.ndarray = np.array([[z, T0, p_i]])
            sampler.sample_at(par_point)
            return sampler.sampled_could.point_data["S_h"][0] - sh_target

        try:
            z_vals = np.linspace(z_bounds[0], z_bounds[1], 2)
            s_vals = np.array([
                f(z) for z in z_vals
            ])
            idx = np.where(np.diff(np.sign(s_vals)))[0]
            z_low = z_vals[idx[0]]
            z_high = z_vals[idx[0] + 1]
            sol = root_scalar(
                f,
                bracket=[z_low, z_high],
                method="brentq",
                xtol=1.0e-6
            )
            if sol.converged:
                z_solutions[i] = sol.root
        except ValueError:
            # No root in bracket: leave as np.nan
            continue

    return z_solutions


class ICThreePhaseLowPressure(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.
        """
        p_inlet = 4.0e6
        p_outlet = 1.0e6
        # domain_length = 2000.0  # in m
        # cell_centers_x = sd.cell_centers[0]
        # pressure_gradient = (p_outlet - p_inlet) / domain_length
        # p_init = p_inlet + pressure_gradient * cell_centers_x
        p_init = np.linspace(p_inlet, p_outlet, sd.num_cells)
        return p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_low, z_high = 0.3, 0.42
        z_NaCl = find_z_for_target_sh(
            T0=t[0],
            sh_target=0.1,
            p_values=p,
            sampler=self.vtk_sampler_ptz,
            z_bounds=(z_low, z_high)
        )
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']

        return h_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15  # [K]
        return np.ones(sd.num_cells) * t_init

    def ic_values_overall_fraction(
        self,
        component: pp.Component,
        sd: pp.Grid
    ) -> np.ndarray:
         
        T = self.ic_values_temperature(sd)
        p = self.ic_values_pressure(sd)
        if component.name != "H2O":
            z_low, z_high = 0.3, 0.42
            z_values = find_z_for_target_sh(
                T0=T[0],
                sh_target=0.1,
                p_values=p,
                sampler=self.vtk_sampler_ptz,
                z_bounds=(z_low, z_high)
            )
            return z_values
        else:
            # Assuming NaCl is the only non-water component
            z: np.ndarray = self.ic_values_overall_fraction(
                next(c for c in self.fluid.components if c.name == "NaCl"),
                sd
            )
            return 1.0 - z
