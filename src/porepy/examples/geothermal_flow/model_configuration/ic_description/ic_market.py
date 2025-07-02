import numpy as np

import porepy as pp

from ...vtk_sampler import VTKSampler


class IC_Base(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_gas_saturation(sd)
            x_CO2_liq_v, x_CO2_gas_v = self.ic_values_partial_fractions(sd)

            x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
            self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

    def ic_values_partial_fractions(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        x_CO2_liq = np.clip(self.vtk_sampler_ptz.sampled_could.point_data["Xl"], 0, 1.0)
        x_CO2_gas = np.clip(self.vtk_sampler_ptz.sampled_could.point_data["Xv"], 0, 1.0)
        return x_CO2_liq, x_CO2_gas

    def ic_values_gas_saturation(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        s_init = np.clip(self.vtk_sampler_ptz.sampled_could.point_data["S_v"], 0, 1.0)
        return s_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data["H"] * 1.0e-6
        return h_init


class IC_single_phase_high_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        return z * np.ones(sd.num_cells)


class IC_single_phase_moderate_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 40.0
        p_outlet = 20.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 573.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        return z * np.ones(sd.num_cells)


class IC_single_phase_low_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 15.0
        p_outlet = 1.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 623.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        return z * np.ones(sd.num_cells)


class IC_two_phase_moderate_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 20.0
        p_outlet = 1.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

class IC_two_phase_low_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 5.0
        p_outlet = 1.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 353.15
        return np.ones(sd.num_cells) * t_init

