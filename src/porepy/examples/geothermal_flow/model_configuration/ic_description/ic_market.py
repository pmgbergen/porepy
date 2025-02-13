import numpy as np

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import InitialConditionsCF


class IC_single_phase_high_pressure(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h_init

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class IC_single_phase_moderate_pressure(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 40.0
        p_outlet = 20.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 573.15
        return np.ones(sd.num_cells) * t_init

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h_init

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class IC_single_phase_low_pressure(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 15.0
        p_outlet = 1.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 623.15
        return np.ones(sd.num_cells) * t_init

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.initial_pressure(sd)
        t = self.initial_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h_init

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)
