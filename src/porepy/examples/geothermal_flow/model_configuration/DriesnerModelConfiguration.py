from typing import Callable, Literal, Union, cast

import numpy as np

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import CompositionalFlowTemplate

from ..vtk_sampler import VTKSampler
from .constitutive_description.BrineConstitutiveDescription import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry


class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler
    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

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
        p_inlet = 50.0
        p_outlet = 25.0
        xc = boundary_grid.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.15
        t_outlet = 423.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.0
        z_inlet = 0.0
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl


class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = sd.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h_init

    def ic_values_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class DriesnerBrineFlowModel(  # type:ignore[misc]
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    CompositionalFlowTemplate,
):
    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler

    @property
    def vtk_sampler_ptz(self):
        return self._vtk_sampler_ptz

    @vtk_sampler_ptz.setter
    def vtk_sampler_ptz(self, vtk_sampler):
        self._vtk_sampler_ptz = vtk_sampler

    def gravity_force(
        self,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: Literal["fluid", "solid"],
    ) -> pp.ad.Operator:
        if material == "fluid":
            rho = self.fluid.density(cast(pp.SubdomainsOrBoundaries, grids))
        elif material == "solid":
            rho = self.solid_density(grids)  # type:ignore[arg-type]
        else:
            raise ValueError(f"Unsupported gravity force for material '{material}'.")

        # Keeping the following line for quantitative verification purposes
        # rho_avg = np.sum(overall_rho.value(self.equation_system) * subdomains[0].cell_volumes) / np.sum(subdomains[0].cell_volumes)

        scaling = 1.0e-6
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * scaling
        size = np.sum([g.num_cells for g in grids]).astype(int)

        # Gravity acts along the last coordinate direction (z in 3d, y in 2d)
        e_n = self.e_i(grids, i=self.nd - 1, dim=self.nd)
        overall_gravity_flux = (
            pp.ad.Scalar(-1) * e_n @ (rho * pp.wrap_as_dense_ad_array(val, size=size))
        )
        overall_gravity_flux.set_name("overall gravity flux")
        return overall_gravity_flux
