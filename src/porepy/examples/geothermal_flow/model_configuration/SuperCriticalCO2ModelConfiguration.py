from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp
from porepy.models.compositional_flow import CompositionalFractionalFlowTemplate as FlowTemplate
from tests.compositional.test_materials import subdomains

from .constitutive_description.SuperCriticalCO2ConstitutiveDescription import (
    gas_saturation_func,
    CO2_gas_func,
    CO2_liq_func,
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometryHayekVertical as ModelGeometry

class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _ , outlet_idx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        _ , outlet_idx = self.get_inlet_outlet_sides(sd)
        return pp.BoundaryCondition(sd, outlet_idx, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_top = 10.0
        p = p_top * np.ones(boundary_grid.num_cells)
        p[outlet_idx] = p_top
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        _, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        h_inlet = 2.0
        h = h_inlet * np.ones(boundary_grid.num_cells)
        h[outlet_idx] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_inlet = 0.0
        z_CO2 = np.zeros(boundary_grid.num_cells)
        z_CO2[inlet_idx] = z_inlet
        return z_CO2



class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        subdomains = self.mdg.subdomains()
        CO2 = self.fluid.components[1]
        z_v = self.ic_values_overall_fraction(CO2,subdomains[0])
        x_CO2_liq_v = np.clip(np.zeros_like(z_v), 1.0e-16, 1.0-1.0e-16)
        x_CO2_gas_v = np.clip(np.ones_like(z_v), 1.0e-16, 1.0-1.0e-16)

        liq, gas = self.fluid.phases
        s_gas = gas.saturation(subdomains)
        x_CO2_liq = liq.partial_fraction_of[CO2](subdomains)
        x_CO2_gas = gas.partial_fraction_of[CO2](subdomains)

        self.equation_system.set_variable_values(z_v, [s_gas], 0, 0)
        self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
        self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

        # values: np.ndarray,
        # variables: Optional[VariableList] = None,

        # time_step_index: Optional[int] = None, -> 0
        # iterate_index: Optional[int] = None, -> 0
        # additive: bool = False,


    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 2.0
        return np.ones(sd.num_cells) * h

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        z = np.where((xc[:,1] >= 2.0) & (xc[:,1] <= 4.0), 0.7, 0.0)
        z = np.where((xc[:, 1] >= 5.5) & (xc[:, 1] <= 6.5), 1.0, 0.0)
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class SuperCriticalCO2FlowModel(
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    FlowTemplate,
):

    # def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
    #     # # See equations (14) and (15) in https://doi.org/10.1016/j.advwatres.2008.12.009
    #     # sr_g = pp.ad.Scalar(0.0)
    #     # sr_l = pp.ad.Scalar(0.0)
    #     # nu_exp = 2.0
    #     # if saturation.name == "reference_phase_saturation_by_unity":
    #     #     s_red = (saturation - sr_l) / (pp.ad.Scalar(1.0) - sr_g - sr_l)
    #     #     kr = s_red**((2.0+3.0*nu_exp)/nu_exp)
    #     # else:
    #     #     s_red = (saturation - sr_g) / (pp.ad.Scalar(1.0) - sr_g - sr_l)
    #     #     kr = (s_red**2.0) * (1.0 - (pp.ad.Scalar(1.0) - s_red)**( (2.0+nu_exp) / nu_exp))
    #     # return kr
    #     return saturation

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)