import numpy as np

import porepy as pp

from ...obl_sampler import VTKSampler


class IC_Base(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions (phase order is
        # [non-gas..., gas-like], so look phases up by NAME, never by position)
        liq = next(ph for ph in self.fluid.phases if ph.name == "liq")
        gas = next(ph for ph in self.fluid.phases if ph.name == "gas")
        hal = [ph for ph in self.fluid.phases if ph.name == "halite"]
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_gas_saturation(sd)
            x_CO2_liq_v, x_CO2_gas_v = self.ic_values_partial_fractions(sd)

            x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
            self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)
            if hal:
                s_hal = hal[0].saturation([sd])
                x_hal = hal[0].partial_fraction_of[self.fluid.components[1]]([sd])
                self.equation_system.set_variable_values(
                    self.ic_values_halite_saturation(sd), [s_hal], 0, 0)
                self.equation_system.set_variable_values(
                    np.ones(sd.num_cells), [x_hal], 0, 0)  # halite is pure NaCl

    def ic_values_partial_fractions(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        x_CO2_liq = np.clip(self.obl_sampler_ptz.sampled_could.point_data["Xl"], 0, 1.0)
        x_CO2_gas = np.clip(self.obl_sampler_ptz.sampled_could.point_data["Xv"], 0, 1.0)
        return x_CO2_liq, x_CO2_gas

    def ic_values_gas_saturation(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        s_init = np.clip(self.obl_sampler_ptz.sampled_could.point_data["S_v"], 0, 1.0)
        return s_init

    def ic_values_halite_saturation(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        return np.clip(self.obl_sampler_ptz.sampled_could.point_data["S_h"], 0, 1.0)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        h_init = self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h_init


class IC_single_phase_high_pressure(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler

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

    obl_sampler_ptz: VTKSampler

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

    obl_sampler_ptz: VTKSampler

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

    obl_sampler_ptz: VTKSampler

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

    obl_sampler_ptz: VTKSampler

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

class IC_two_phase_Figure_8_left_panel(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 5.0
        p_outlet = 1.0 #1  atm = 0.101325 MPa
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return np.ones(sd.num_cells) * p_outlet

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 283.15
        return np.ones(sd.num_cells) * t_init

class IC_two_phase_steady_state(IC_Base):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 15.0
        p_outlet = 5.0
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        p_linear = (
            lambda x: (x[dir_idx] * p_outlet + (2000.0 - x[dir_idx]) * p_inlet) / 2000.0
        )
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_inlet = 723.15
        t_outlet = 473.15
        xc = sd.cell_centers.T
        dir_idx = np.argmax(np.max(xc, axis=0))
        T_linear = (
            lambda x: (x[dir_idx] * t_outlet + (2000.0 - x[dir_idx]) * t_inlet) / 2000.0
        )
        T = np.array(list(map(T_linear, xc)))
        return T


class IC_three_phase_segregation(pp.PorePyModel):
    """Initial condition for the immiscible 3-phase gravity-segregation-through-barriers
    test (Bosma et al. 2022, Ex. 6.3 / Fig. 5).

    Top 10% of the 100 m box = heavy 'water', bottom 10% = light 'gas', the rest =
    intermediate 'oil'. Immiscible: each component lives entirely in one phase. Self-
    contained (does NOT use the VTK table base). Components are ordered [H2O, C5H12, CH4]
    with H2O the reference; phases are [water, oil, gas] with water the reference phase.
    """

    _height = 100.0          # m, vertical (y) extent of the box
    _p_ref = 10.0            # reference pressure [MPa] (closed incompressible -> arbitrary)

    def _bands(self, sd: pp.Grid):
        y = sd.cell_centers[1]
        height = self.units.convert_units(self._height, "m")
        top = y > 0.9 * height          # heavy water
        bottom = y < 0.1 * height       # light gas
        middle = ~(top | bottom)        # intermediate oil
        return top, middle, bottom

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, self._p_ref)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # isothermal/decoupled energy -> uniform placeholder (mirrors buoyancy_flow_model)
        return np.ones(sd.num_cells)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        top, middle, bottom = self._bands(sd)
        z = np.zeros(sd.num_cells)
        if component == self.fluid.components[1]:       # C5H12 (oil) -> middle band
            z[middle] = 1.0
        elif component == self.fluid.components[2]:     # CH4 (gas) -> bottom band
            z[bottom] = 1.0
        return z                                        # H2O (reference) = 1 - sum (top band)

    def ic_values_saturation(self, sd: pp.Grid):
        top, middle, bottom = self._bands(sd)
        s_oil = np.where(middle, 1.0, 0.0)
        s_gas = np.where(bottom, 1.0, 0.0)
        return s_oil, s_gas                              # s_water = 1 - s_oil - s_gas

    def initial_condition(self) -> None:
        super().initial_condition()
        water, oil, gas = self.fluid.phases
        c5h12, ch4 = self.fluid.components[1], self.fluid.components[2]
        for sd in self.mdg.subdomains():
            s_oil, s_gas = self.ic_values_saturation(sd)
            # Seed initial values only for quantities that are still independent
            # variables. When a quantity is substituted as a function (SurrogateFactory)
            # its value is computed from z, so there is no variable to seed (and
            # set_variable_values would reject the surrogate operator).
            if self.has_independent_saturation(oil):
                self.equation_system.set_variable_values(s_oil, [oil.saturation([sd])], 0, 0)
            if self.has_independent_saturation(gas):
                self.equation_system.set_variable_values(s_gas, [gas.saturation([sd])], 0, 0)

            one = np.ones(sd.num_cells)
            zero = np.zeros(sd.num_cells)
            # immiscible partial fractions: C5H12 only in oil, CH4 only in gas
            for value, phase, comp in (
                (zero, water, c5h12), (one, oil, c5h12), (zero, gas, c5h12),
                (zero, water, ch4), (zero, oil, ch4), (one, gas, ch4),
            ):
                if self.has_independent_partial_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        value, [phase.partial_fraction_of[comp]([sd])], 0, 0)