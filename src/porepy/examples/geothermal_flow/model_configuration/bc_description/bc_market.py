from typing import Callable
import numpy as np

import porepy as pp
from ...vtk_sampler import VTKSampler


class BCSinglePhaseHighPressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Set up for Liquid phase flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
 
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 50.0e6
        p_outlet = 25.0e6
        p = p_outlet * np.ones(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p
    
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.15  # [K]
        t_outlet = 423.15  # [K]
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T
    
    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h


class BCSinglePhaseModeratePressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet.
        BC Set up for Supercritical fluid flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 40.0e6
        p_outlet = 20.0e6
        p = p_outlet * np.ones(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 723.15  # [K]
        t_outlet = 573.15  # [K]
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:

        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h


class BCSinglePhaseLowPressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC setup for vapour phase flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 15.0e6
        p_outlet = 1.0e6
        p = p_outlet * np.zeros(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 773.15  # [K]
        t_outlet = 623.15  # [K]
        T = t_outlet * np.zeros(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h

    # def bc_values_overall_fraction(
    #     self, 
    #     component: pp.Component, 
    #     boundary_grid: pp.BoundaryGrid
    # ) -> np.ndarray:
    #     inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
    #     z_init = 0.0
    #     z_inlet = 0.0
    #     if component.name == "H2O":
    #         z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
    #         z_H2O[inlet_idx] = 1 - z_inlet
    #         return z_H2O
    #     else:
    #         z_NaCl = z_init * np.ones(boundary_grid.num_cells)
    #         z_NaCl[inlet_idx] = z_inlet
    #         return z_NaCl
    
    # Testing
    # def bc_values_fractional_flow_energy(
    #     self, bg: pp.BoundaryGrid
    # ) -> np.ndarray:
    #     """BC values for the non-linear weight in the advective flux in the energy
    #     balance equation, determining how much energy/enthalpy is entering the system on
    #     some inlet faces in terms relative to the mass.

    #     Parameters:
    #         bg: A boundary grid in the mixed-dimensional grid.

    #     Returns:
    #         By default a zero array with shape ``(boundary_grid.num_cells,)``.

    #     """
    #     return np.zeros(bg.num_cells)


class BCTwoPhaseLowPressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Setup for long two phase liquid+vapour flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 4.0e6  # in Pa
        p_outlet = 1.0e6
        p = p_outlet * np.zeros(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 573.15  # [K]
        t_outlet = 423.15  # [K]
        T = t_outlet * np.zeros(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h


class BCTwoPhaseHighPressure(pp.PorePyModel): 
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Setup for short two phase liquid+vapour flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 20.0e6
        p_outlet = 1.0e6
        p = p_outlet * np.zeros(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 673.15  # [K]
        t_outlet = 423.15  # [K]
        T = t_outlet * np.zeros(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        h[2:] = 0.0
        return h


class BCThreePhaseLowPressure(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Set up for Liquid phase flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
    
    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")
  
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 4.0e6
        p_outlet = 1.0e6
        p = p_outlet * np.zeros(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
     
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 573.15  # [K]
        t_outlet = 423.15  # [K]
        T = t_outlet * np.zeros(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = self.bc_values_overall_fraction(
            self.get_components()[1],
            boundary_grid
        )
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        # h = np.zeros(boundary_grid.num_cells)
        # Compute specific enthalpy for each cell using CoolProp
        # for i in range(boundary_grid.num_cells):
        #     # if i == 0:
        #     #     t_k = 573.15  # [K]
        #     # else:
        #     #     t_k = t[i]
        #     if p[i] == 0:
        #         continue
        #     h[i] = CP.PropsSI('H', 'P', p[i], 'T', t[i], 'Water')
        return h

    def bc_values_overall_fraction(
        self,
        component: pp.Component,
        boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, outlet_index = self.get_inlet_outlet_sides(boundary_grid)
        z_inlet = 0.000
        z_outlet = 0.41699537203872905
        if component.name == "H2O":
            z_H2O = np.zeros(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            z_H2O[outlet_index] = 1 - z_outlet
            return z_H2O
        else:
            z_NaCl = np.zeros(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            z_NaCl[outlet_index] = z_outlet
            return z_NaCl


class BCThreePhaseLowPressure2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""
    """BC Set up for Liquid phase flow"""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = sd.get_boundary_faces()
        return pp.BoundaryCondition(sd, facet_idx, "neu")
    
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = sd.get_boundary_faces()
        return pp.BoundaryCondition(sd, facet_idx, "neu")
    
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = sd.get_boundary_faces()
        return pp.BoundaryCondition(sd, facet_idx, "neu")
    
    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = sd.get_boundary_faces()
        return pp.BoundaryCondition(sd, facet_idx, "neu")
  
    # def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     # no Dirichlet applied ⇒ not used
    #     return np.zeros(boundary_grid.num_cells)

    # def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     # no Dirichlet applied ⇒ not used
    #     return np.zeros(boundary_grid.num_cells)

    # def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     # no Dirichlet applied ⇒ not used
    #     return np.zeros(boundary_grid.num_cells)

    # def bc_values_overall_fraction(
    #     self,
    #     component: pp.Component,
    #     boundary_grid: pp.BoundaryGrid
    # ) -> np.ndarray:
    #     # no Dirichlet applied ⇒ not used
    #     return np.zeros(boundary_grid.num_cells)


class BCLiquidPhaseLowPressure_no_wells_2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        _, outlet_facet_indx = self.get_inlet_outlet_sides(sd) 
        return pp.BoundaryCondition(sd, outlet_facet_indx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        _, outlet_facet_indx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_facet_indx, "dir")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, outlet_facet_indx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_facet_indx, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _, outlet_facet_indx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_facet_indx, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        _, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_outlet = 0.101e6
        pressure_values = np.zeros(boundary_grid.num_cells)
        pressure_values[outlet_idx] = p_outlet
        return pressure_values

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        _, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_outlet = 300.15  # [K]  # # Note: this wont work if the minimum temperature is far above 283.15 K in the VTK file.
        temperature_values = np.zeros(boundary_grid.num_cells)
        temperature_values[outlet_idx] = t_outlet
        return temperature_values

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h

    def bc_values_overall_fraction(
        self,
        component: pp.Component,
        boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 1e-4
        z_inlet = 1e-4
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl


class BCLiquidPhaseLowPressure_wells_fracture_2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")


class BCLiquidPhaseLowPressure_Pointwells_Fracture_2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        if sd.dim == 0 and "production_well" in sd.tags:
            return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        if sd.dim == 0 and "production_well" in sd.tags:
            return np.array([self._p_PRODUCTION[0]])
        return np.zeros(boundary_grid.num_cells)


class BCLiquidPhaseLowPressure_Well_Flux(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    
    # def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     return np.zeros(boundary_grid.num_cells)
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(boundary_grid.num_cells)


class BCLiquidPhaseLowPressure_Pointwells_Fracture_Salt_2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        # if sd.dim == 0 and "production_well" in sd.tags:
        #     return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # sd = boundary_grid.parent
        # if sd.dim == 0 and "production_well" in sd.tags:
        #     return np.array([self._p_PRODUCTION[0]])
        return np.zeros(boundary_grid.num_cells)


class BCBrineSystem2D(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for temperature
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        if sd.dim == 0 and "production_well" in sd.tags:
            return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        if sd.dim == 0 and "production_well" in sd.tags:
            return np.array([self._p_PRODUCTION[0]])
        return np.zeros(boundary_grid.num_cells)


class BCLiquidPhaseLowPressure_SmallBox(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]
    vtk_sampler_ptz: VTKSampler

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        inlet, _ = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, inlet, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # We have dirichlet BC only on the top for pressure
        inlet, _ = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, inlet, "dir")
   
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # inlet, _ = self.get_inlet_outlet_sides(sd)
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        inlet, _ = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, inlet, "dir")
    
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        if boundary_grid.parent.dim != 2:
            return np.ones(boundary_grid.num_cells) * self._p_OUT
            # return np.zeros(boundary_grid.num_cells)
        inlet, _ = self.get_inlet_outlet_sides(boundary_grid)
        p = np.zeros(boundary_grid.num_cells)
        p[inlet] = self._p_INJECTION
        return p
    
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        if boundary_grid.parent.dim != 2:
            return np.ones(boundary_grid.num_cells) * self._T_PRODUNTION[0]
            # return np.zeros(boundary_grid.num_cells)
        inlet, _ = self.get_inlet_outlet_sides(boundary_grid)
        T = np.zeros(boundary_grid.num_cells)
        T[inlet] = self._T_INJ
        return T
    
    def bc_values_enthalpy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # if boundary_grid.parent.dim != 2:
        #     return np.zeros(boundary_grid.num_cells)
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = self._z_INJ["NaCl"]*np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_cloud.point_data['H']
        return h
    
    def bc_values_overall_fraction(
        self, component: pp.Component,
        bg: pp.BoundaryGrid
    ) -> np.ndarray:
        if bg.parent.dim != 2:
            return self._z_INIT.get(component.name, 0.0) * np.ones(bg.num_cells)
        return self._z_INJ.get(component.name, 0.0) * np.ones(bg.num_cells)

