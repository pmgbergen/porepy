"""
Simulation: Salt dissolution and precipitation in a horizontal 2D geothermal reservoir 
with point injection and production wells.

Description:
------------
This experiment injects low-salinity water (z_NaCl ≈ 1e-4) into a halite-saturated 
geological formation, with initial NaCl mass fraction z_NaCl = 0.40. The setup uses 
a mixed-dimensional grid with fractures and point grid wells at x ≈ 15 (injector) and 
x ≈ 85 (producer), simulating three phases: liquid, vapor, and solid halite.

Boundary conditions:
--------------------
- Left boundary: **Dirichlet pressure** fixed at the initial reservoir pressure, 
  representing connection to an infinite reservoir (outflow allowed).
- All other outer boundaries: **no-flow** (impermeable).
- Wells: injection well imposes enthalpy/composition of the injected fluid; 
  production well is controlled by a fixed bottom-hole pressure.


Observations from the halite saturation profile:
------------------------------------------------
- A sharp **drop in halite saturation** is observed near the injection well due 
  to halite **dissolution** from undersaturated fluid.
- A local peak in halite saturation appears near the production well, consistent with salt
  precipitation as boiling occurs, water vaporizes, and the residual brine becomes more concentrated.
- The rest of the domain maintains a **stable saturation baseline**, indicating 
  physically consistent behavior away from the wells.

Remarks:
--------
- The pattern is physically plausible and consistent with expected dissolution/
  precipitation dynamics in geothermal saline systems.
- The sharpness of the features suggests either limited dispersion or a stiff 
  phase transition model.
- Saturation values remain stable across most of the domain, but monitoring 
  porosity reduction and timestep stability is advised.
"""


import porepy as pp
import numpy as np

from porepy.models.compositional_flow import (
    CompositionalFlowTemplate
    # CompositionalFractionalFlowTemplate
)
from porepy.models.constitutive_laws import PeacemanWellFlux
# from porepy.models.constitutive_laws import CubicLawPermeability
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

from .constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem,
    PhaseMode,
)

# import scipy.sparse as sps
from typing import Sequence, ClassVar
from dataclasses import dataclass


class VTKSamplerMixin:
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


class LiquidSecondaryEquation2D(SecondaryEquations):
    component_system = ComponentSystem.WATER_SALT
    phase_mode = PhaseMode.THREE_PHASE


@dataclass(kw_only=True)
class FractureSolidConstants(pp.SolidConstants):
    """Solid constants tailored to the current model."""
    SI_units: ClassVar[dict[str, str]] = dict(**pp.SolidConstants.SI_units)
    SI_units.update({"fracture_permeability": "m^2"})
    fracture_permeability: pp.number = 1.0


def clamped_halite_saturation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
    """Clamp halite saturation between [0, 0.5]."""
    halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
    if len(halite_phase) != 1:
        raise ValueError("Expected exactly one halite phase.")
    
    s_h_raw = halite_phase[0].saturation(subdomains)
    return s_h_raw
    # s_h_raw = self.equation_system.evaluate(s_h_raw_op)
    # s_h_raw = pp.wrap_as_dense_ad_array(s_h_raw, name="halite_saturation_raw")

    max_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

    def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
        return -max_fn(-a, -b)
    
    min_val = min_fn(pp.ad.Scalar(0.8), max_fn(s_h_raw, pp.ad.Scalar(0.0))) # 0.7(329), 0.98 (328), 0.5(370)
    return min_val


# =============================================================================
# WELL FLOW DATA WITH RAMPING
# =============================================================================
class WellFlowData(pp.PorePyModel):
    """Helper class to bundle configuration of pressure, temperature, and injected mass
    for a multiphase and salt-water geothermal problem with one injector and one producer."""

    vtk_sampler_ptz: VTKSampler

    # Initial reservoir conditions (representative of ~2 km depth).
    _p_INIT: float = 10.5e6         # Pa       # 15.0e6 maybe put 20.0e6 for matrix cell
    _T_INIT: float = 586.651        # K       # 588.451,  Note, at 10.5MPa, 587.6 wont work! 
    _z_INIT: dict[str, float] = {"H2O": 0.6, "NaCl": 0.4}

    # In- and outflow values.
    _T_INJ: float = 300.651    # 300.15   # K (injection temperature)
    _z_INJ: dict[str, float] = {"H2O": 0.9999, "NaCl": 1.0e-4}

    # Value obtained from a p-T flash with values defined above.
    # Divide by 3600 to obtain an injection of unit per hour
    # Multiplied by some number for how many units per hour
    _INJECTION_FRACTION: float = 1.3   # kg/kg
    
    _p_OUT: float = 7.0e6          # Pa (fixed production pressure) NOTE: 7.0e6 for matrix cell!
    _well_radius: float = 0.1      # m (for well index calculation) 0.08 works if i kick out fractures

    # Initial and injected fluid composition.
    _fracture_aperture = 1.0e-3  # m (initial fracture aperture)

    # Injection schedule (can extend later with time-dependent keys)
    _T_INJECTION: dict[int, float] = {0: _T_INJ}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
    _p_INJECTION: dict[int, float] = {0: _p_INIT}

    # =========================================================================
    # RAMPING PARAMETERS - Adjust these to control startup behavior
    # =========================================================================
    _t_RAMP_PERIOD: float = 10.0 * pp.DAY      # days (duration of ramp-up period for injection/production)
    _cached_alpha: float = 0.0  # Cache for ramp factor to avoid redundant calculations within the same timestep 
    _cached_time_for_alpha: float = -1.0  # Time at which alpha was last calculated

    def ramp_factor_old(self) -> float:
        """Returns a factor from 0 to 1 over the ramp period.
        
        - At t=0: returns 0 (use initial reservoir conditions)
        - At t=t_ramp: returns 1 (use target well conditions)
        - Between: linear interpolation
        """
        # return 1.0  # DEBUG
        t_start = self.time_manager.time
        dt = self.time_manager.dt
        t_current = t_start - dt
        if t_current != self._cached_time_for_alpha:
            alpha = min(t_current / self._t_RAMP_PERIOD, 1.0)
            self._cached_alpha = alpha
            self._cached_time_for_alpha = t_current
            print(f"  Cached alpha updated: t={t_current/pp.DAY:.3f} days, alpha={alpha:.6f}")
        else:
            print(f"  Using cached alpha: t={t_current/pp.DAY:.3f} days, alpha={self._cached_alpha:.6f}")
        return self._cached_alpha
    
    _alpha_prev: float = 0.0

    def ramp_factor(self) -> float:
        return 1.0  # DEBUG: No ramping
        # Use monotone time (current simulation time)
        t = float(self.time_manager.time)

        # Clamp and make monotone
        alpha = max(0.0, min(t / self._t_RAMP_PERIOD, 1.0))
        alpha = max(self._alpha_prev, alpha)  # prevent going backwards if dt changes
        self._alpha_prev = alpha

        return alpha
 
    def get_ramped_production_bhp(self) -> float:
        """Get ramped production BHP.

        Starts at reservoir pressure (no drawdown), gradually drops to target BHP.
        """
        # return self._p_OUT  # DEBUG: No ramping for production BHP
        alpha = self.ramp_factor()
        p_bhp = self._p_INIT + 1.0 * (self._p_OUT - self._p_INIT)
        return p_bhp

    def get_ramped_injection_temperature(self) -> float:
        """Get ramped injection temperature.
    
        Starts at reservoir temperature, gradually drops to injection temperature.
        """

        alpha = self.ramp_factor()
        T_inj = self._T_INIT + alpha * (self._T_INJ - self._T_INIT)
        return T_inj

    def get_ramped_injection_composition(self, component_name: str) -> float:
        """Get ramped injection composition for a given component.
    
        Starts at reservoir composition, gradually shifts to injection composition.
        """

        alpha = self.ramp_factor()
        z_init = self._z_INIT.get(component_name, 0.0)
        z_final = self._z_INJ.get(component_name, 0.0)
        z_ramped = z_init + 1.0 * (z_final - z_init)
        return z_ramped
    
    def _get_fluid_density(
        self,
        temperature: float,
        pressure: float,
        z_NaCl: float
    ) -> float:
        """Sample bulk fluid density from the VTK table."""
        par_point = np.array([[z_NaCl, temperature, pressure]])
        self.vtk_sampler_ptz.sample_at(par_point)
        data = self.vtk_sampler_ptz.sampled_cloud.point_data
        rho = data["Rho"][0]
        return rho

    def _get_total_injected_mass_rate(
        self
    ) -> float:
        """Calculate total injected mass (in kg/s).
        NOTE: Uses ramped temperature and composition for density calculation.
        """

        T_inj = self.get_ramped_injection_temperature()
        z_NaCl = self.get_ramped_injection_composition("NaCl")
        rho = self._get_fluid_density(
            temperature=T_inj,
            pressure=self._p_INIT,
            z_NaCl=z_NaCl
        )

        base_rate = self._INJECTION_FRACTION * rho / 3600.0  # kg/m3/s
        return base_rate

    def _injected_component_mass(
        self,
        component: pp.Component,
        subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Return injected mass source density for a given component [kg/m³/s].
        NOTE: Uses ramped composition.
        """

        return pp.ad.TimeDependentDenseArray(
            name=f"injection_mass_{component.name}",
            domains=subdomains,
        )
        # total injected mass rate [kg/m3/s]
        _total_injected_mass = self._get_total_injected_mass_rate()

        # USE RAMPED COMPOSITION
        # z_H2O_ramped = self.get_ramped_injection_composition("H2O")
        # z_NaCl_ramped = self.get_ramped_injection_composition("NaCl")
        z_H2O = self._z_INJ["H2O"]  # z_H2O_ramped
        z_NaCl = self._z_INJ["NaCl"]  # z_NaCl_ramped

        # total injected mass rate [kg/m3/s]
        _injected_mass_rate: dict[str, dict[int, float]] = { 
            "H2O": {0: _total_injected_mass * z_H2O},
            "NaCl": {0: _total_injected_mass * z_NaCl},
        }

        injected_mass: list[np.ndarray] = []
        for sd in subdomains:
            assert "injection_well" in sd.tags, (
                f"Grid {sd.id} not tagged as injection well."
            )
            injected_mass.append(
                np.ones(sd.num_cells)
                * _injected_mass_rate[component.name][sd.tags["injection_well"]]
            )
        
        if injected_mass:
            source = np.hstack(injected_mass)
        else:
            source = np.zeros((0,))

        return pp.ad.DenseArray(source, f"injected_mass_density_{component.name}")
    
    # def calculate_reasonable_injection_rate(self):
    #     # Get domain size
    #     # matrix_grids = [sd for sd in self.mdg.subdomains() if sd.dim == self.nd]
    #     bbox = self.domain.bounding_box
    #     x_max, y_max = bbox['xmax'], bbox['ymax']
    #     area = x_max * y_max  # m²
    #     pore_volume = area * self.solid.porosity  # m³ (per meter thickness)

    #     # Inject 10% of pore volume per year
    #     yearly_injection = 0.5 * pore_volume  # m³/year
    #     hourly_rate = yearly_injection / (365 * 24)  # m³/h

    #     return hourly_rate  # max(hourly_rate, 0.01)


# =============================================================================
# COMPONENT SOURCE MIXIN
# =============================================================================
class ModifyComponentSourceMixin:
    """
    Adjusts the component source terms for the mass balance equations.

    This mixin adds injected component mass at injection wells and removes any explicit
    component source values at production wells. While component mass can still leave 
    the domain implicitly through fluxes (driven by pressure gradients and constraints),
    this ensures that no artificial source terms are present at the production locations.
    """
    def component_source(
        self,
        component: pp.Component,
        subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """
        Constructs the adjusted source term for the mass balance of a given fluid component.

        Adds injected component mass to the injection wells and explicitly zeros out any
        component source contributions at the production wells. Actual outflow at production
        wells occurs implicitly through fluxes induced by pressure constraints, not through
        explicit source terms.

        Parameters:
            component: The component for which the source term is defined (e.g., H2O, NaCl).
            subdomains: All active subdomain grids, including matrix, fractures, and wells.

        Returns:
            AD operator representing the component source term with proper well adjustments.
        """
       
        source: pp.ad.Operator = super().component_source(component, subdomains)  # type:ignore[misc]

        injection_wells_grid, _ = self._filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        # injected mass rate [kg/s]
        injected_mass = self.volume_integral(
            self._injected_component_mass(component, injection_wells_grid),
            injection_wells_grid,
            1,
        )

        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells_grid) @ injected_mass
        )

        production_wells_grid, _ = self._filter_wells(subdomains, "production")
        source -= subdomain_projections.cell_prolongation(production_wells_grid) @ (
            subdomain_projections.cell_restriction(production_wells_grid) @ source
        )

        return source


# =============================================================================
# INITIALIZATION MIXIN FOR TIME-DEPENDENT Constraints and Well Variables
# =============================================================================
class InitializedBHPScheduleMixin:
    def set_equations(self):
        """Set up equations and initialize time-dependent BHP storage."""
        # Initialize BHP in data dictionary BEFORE calling super
        self._initialize_production_bhp()
        self._initialize_injection_temperature()
        self._initialize_injection_composition()
        self._initialize_injection_sources()
        super().set_equations()

    def _initialize_production_bhp(self):
        """Store initial BHP in data dictionaries for production wells."""
        prod_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "production")

        p_bhp_value = self._p_INIT  # Start at reservoir pressure

        for sd in prod_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="production_bhp",
                values=np.ones(sd.num_cells) * p_bhp_value,
                data=data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name="production_bhp",
                values=np.ones(sd.num_cells) * p_bhp_value,
                data=data,
                time_step_index=0,
            )
    
    def _initialize_injection_temperature(self):
        """Initialize injection T at reservoir temperature."""
        inj_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "injection")
        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="injection_temperature",
                values=np.ones(sd.num_cells) * self._T_INIT,
                data=data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name="injection_temperature",
                values=np.ones(sd.num_cells) * self._T_INIT,
                data=data,
                time_step_index=0,
            )
    
    def _initialize_injection_composition(self):
        """Initialize injection Compositions at reservoir conditions."""
        inj_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "injection")

        # At t=0, we want the injection composition to match the 
        # initial reservoir composition to avoid shocks.
        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            for comp_name in ["H2O", "NaCl"]:
                pp.set_solution_values(
                    name=f"injection_fraction_{comp_name}",
                    values=np.ones(sd.num_cells) * self.get_ramped_injection_composition(comp_name),            #self._z_INIT[comp_name],
                    data=data,
                    iterate_index=0,
                )
                pp.set_solution_values(
                    name=f"injection_fraction_{comp_name}",
                    values=np.ones(sd.num_cells) * self.get_ramped_injection_composition(comp_name),
                    data=data,
                    time_step_index=0,
                )
    
    def _initialize_injection_sources(self):
        inj_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "injection")
        # At t=0: full rate, but with reservoir composition (z_INIT)
        T_init = self._T_INIT
        z_NaCl_init = self.get_ramped_injection_composition("NaCl")  # self._z_INIT["NaCl"]
        rho = self._get_fluid_density(
            temperature=T_init,
            pressure=self._p_INIT,
            z_NaCl=z_NaCl_init
        )
        base_rate = self._INJECTION_FRACTION * rho / 3600.0

        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            for comp_name in ["H2O", "NaCl"]:
                pp.set_solution_values(
                    name=f"injection_mass_{comp_name}",
                    values=np.ones(sd.num_cells) * base_rate * self.get_ramped_injection_composition(comp_name),
                    data=data,
                    iterate_index=0,
                )
                pp.set_solution_values(
                    name=f"injection_mass_{comp_name}",
                    values=np.ones(sd.num_cells) * base_rate * self.get_ramped_injection_composition(comp_name),
                    data=data,
                    time_step_index=0,
                )


# =============================================================================
# FLUID SOURCE MIXIN WITH RAMPED PRODUCTION BHP
# =============================================================================
class ModifyFluidSourceMixin(WellFlowData):
    """ Modify the fluid source term for the pressure equations at the injection and production wells .
    """

    def before_nonlinear_iteration(self):
        """Update production BHP before each time step"""
        super().before_nonlinear_iteration()
        # alpha = self.ramp_factor()
        # === Production BHP ===
        # Get ramped BHP for this time step
        p_bhp_value = self.get_ramped_production_bhp()
        # Update in data dictionaries
        prod_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "production")
        inj_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "injection")
        for sd in prod_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="production_bhp",
                values=np.ones(sd.num_cells) * p_bhp_value,
                data=data,
                iterate_index=0,
            )

        # ----DEBUG: Read back to verify----
        # curr_bhp = pp.get_solution_values(
        #     name="production_bhp",
        #     data=data,
        #     iterate_index=0,
        # )
        # print(f">>> BHP: alpha={alpha:.4f}, set={p_bhp_value/1e6:.4f} MPa, stored={curr_bhp[0]/1e6:.4f} MPa")
        
        # === Injection temperature ===
        temperature_value = self.get_ramped_injection_temperature()
        inj_wells_grid, _ = self._filter_wells(self.mdg.subdomains(), "injection")
        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="injection_temperature",
                values=np.ones(sd.num_cells) * temperature_value,
                data=data,
                iterate_index=0,
            )
        
        # === Injection mass rates ===
        total_rate = self._get_total_injected_mass_rate()
        z_H2O = self.get_ramped_injection_composition("H2O")
        z_NaCl = self.get_ramped_injection_composition("NaCl")
        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="injection_mass_H2O",
                values=np.ones(sd.num_cells) * total_rate * z_H2O,
                data=data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name="injection_mass_NaCl",
                values=np.ones(sd.num_cells) * total_rate * z_NaCl,
                data=data,
                iterate_index=0,
            )

        # ----DEBUG: Read back to verify----
        # curr_temperature = pp.get_solution_values(
        #     name="injection_temperature",
        #     data=data,
        #     iterate_index=0,
        # # )
        # print(f">>> Injection Temp: alpha={alpha:.4f}, set={temperature_value:.2f} K, stored={curr_temperature[0]:.2f} K")
        # curr_NaCl_mass = pp.get_solution_values(
        #     name="injection_mass_NaCl",
        #     data=data,
        #     iterate_index=0,
        # )
        # print(f">>> Injection NaCl mass rate: alpha={alpha:.4f}, set={total_rate * z_NaCl:.6e} kg/m3/s, stored={curr_NaCl_mass[0]:.6e} kg/m3/s")
        # curr_H2O_mass = pp.get_solution_values(
        #     name="injection_mass_H2O",
        #     data=data,
        #     iterate_index=0,
        # )
        # print(f">>> Injection H2O mass rate: alpha={alpha:.4f}, set={total_rate * z_H2O:.6e} kg/m3/s, stored={curr_H2O_mass[0]:.6e} kg/m3/s")

        # === Optional: Update injection composition fractions if needed (not shown here) ===
        z_H2O = self.get_ramped_injection_composition("H2O")
        z_NaCl = self.get_ramped_injection_composition("NaCl")
        for sd in inj_wells_grid:
            data = self.mdg.subdomain_data(sd)
            pp.set_solution_values(
                name="injection_fraction_H2O",
                values=np.ones(sd.num_cells) * z_H2O,
                data=data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name="injection_fraction_NaCl",
                values=np.ones(sd.num_cells) * z_NaCl,
                data=data,
                iterate_index=0,
            )
            # ----DEBUG: Read back to verify----
            # curr_NaCl_composition = pp.get_solution_values(
            #     name="injection_fraction_NaCl",
            #     data=data,
            #     iterate_index=0,
            # )
            # print(f">>> Injection NaCl composition: alpha={alpha:.4f}, set={z_NaCl:.6e}, stored={curr_NaCl_composition[0]:.6e}")
            # curr_H20_composition = pp.get_solution_values(
            #     name="injection_fraction_H2O",
            #     data=data,
            #     iterate_index=0,
            # )
            # print(f">>> Injection H2O composition: alpha={alpha:.4f}, set={z_H2O:.6e}, stored={curr_H20_composition[0]:.6e}")

    def WI(self) -> pp.ad.Operator:
        # Pick the hosting matrix grid (first 2D subdomain)
        # Reservoir properties
        prod_intf = [
            intf for intf in self.mdg.interfaces()
            if "production_well" in self.mdg.interface_to_subdomain_pair(intf)[1].tags
        ]
        subdomains = self.interfaces_to_subdomains(prod_intf)
        h = pp.ad.Scalar(1.0)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, prod_intf)
        r_w = pp.ad.Scalar(self._well_radius)
        f_log = pp.ad.Function(pp.ad.functions.log, "log_function_Peaceman")
        r_e = PeacemanWellFlux.equivalent_well_radius(self, subdomains)
        skin_factor = pp.ad.Scalar(0.0)

        # We assume isotropic permeability and extract xx component.
        e_i = self.e_i(subdomains, i=0, dim=9).T
        isotropic_permeability = e_i @ self.permeability(subdomains)
        well_geo = pp.ad.Scalar(2 * np.pi) * h * projection.primary_to_mortar_avg() @ (
            isotropic_permeability / (f_log(r_e / r_w) + skin_factor))

        well_index = self.volume_integral(well_geo, prod_intf, 1)
        return well_index
                
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Source term in the mass balance.
        - Injection wells: positive source (mass in).
        - Production wells: negative source (flux to well via WI).
        """
        source: pp.ad.Operator = super().fluid_source(subdomains)

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        # -------------------
        # Injection wells
        # -------------------
        inj_wells_grid, _ = self._filter_wells(subdomains, "injection")

        # injected mass rate [kg/s]
        injected_mass: pp.ad.Operator = pp.ad.sum_operator_list(
            [
                self.volume_integral(
                    self._injected_component_mass(comp, inj_wells_grid),
                    inj_wells_grid,
                    1,
                )
                for comp in self.fluid.components
            ],
            "total_injected_fluid_mass",
        )
        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(inj_wells_grid) @ injected_mass
        )

        # -------------------
        # Production wells (WITH RAMPING)
        # -------------------
        prod_wells_grid, _ = self._filter_wells(subdomains, "production")

        # Well index in 
        WI = self.WI()

        # lamda_f = sum_i sum_j x_ij rho_j krj/mu_j
        lambda_f = self.total_mass_mobility(prod_wells_grid)

        # =========================================================
        # USE RAMPED BHP instead of fixed _p_PRODUCTION[0]
        # =========================================================
        # Sink term: -WI * (p_cell - p_bhp)
        
        # ------TIME-DEPENDENT BHP VALUE------
        p_bhp = pp.ad.TimeDependentDenseArray(
            name="production_bhp",
            domains=prod_wells_grid,
        )
        q_prod = - lambda_f * WI*(self.pressure(prod_wells_grid) - p_bhp)  # / pp.ad.Scalar(cell_volumes)  # kg/(m^3.s)
        q_prod.set_name("produced_fluid_mass")

        # Add directly to source operator: (No need to compute volumes integral because it is in kg/s here)
        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(prod_wells_grid) @ q_prod
        )

        source.set_name("fluid_source")
        return source


# =============================================================================
# ENERGY SOURCE MIXIN
# =============================================================================
class ModifyEnergySourceMixin:

    vtk_sampler_ptz: VTKSampler
    
    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

        """
        Energy source term with zero values enforced at production wells.

        This method returns the energy source operator used in the energy balance equation.
        By default, the base class may provide a uniform or user-defined energy source across 
        all subdomains. This override removes any contributions at the production wells to 
        ensure no explicit energy is added or removed there. Energy can still leave the system 
        implicitly via fluxes due to pressure constraint applied at the production well.

        Parameters:
            subdomains: List of all subdomain grids.

        Returns:
            An AD operator representing the energy source term, with enforced zeros at 
            production wells.
        """

        source = super().energy_source(subdomains)  # type:ignore[misc]

        production_wells, _ = self._filter_wells(subdomains, "production")
        _, no_injection_wells = self._filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(no_injection_wells)
        source -= subdomain_projections.cell_prolongation(production_wells) @ (
            subdomain_projections.cell_restriction(production_wells) @ source
        )
        return source


# =============================================================================
# PRIMARY EQUATIONS MIXIN WITH RAMPED TEMPERATURE CONSTRAINT
# =============================================================================
class ModifiedPrimaryEquationsMixin:
    # Adjusting PDEs
    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""

        _, no_injection_well_grids = self._filter_wells(subdomains, "injection")
        eq: pp.ad.Operator = super().energy_balance_equation(no_injection_well_grids)  # type:ignore[misc]
        # name = eq.name
        return eq

    def set_equations(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively, and removes the balance equation each one replaces."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        injection_well_grid, no_injection = self._filter_wells(subdomains, "injection")

        t_constraint = self.temperature_constraint_at_injection_wells(injection_well_grid)
        self.equation_system.set_equation(
            t_constraint, injection_well_grid, {"cells": 1}
        )

        # The base class registered these on ALL subdomains. Re-register them on
        # the reduced domains, so the constraint above supplies the row for the
        # well cell instead of duplicating it.
        self.equation_system.update_equation(
            "energy_balance_equation",
            super().energy_balance_equation(no_injection),
            no_injection,
        )

    def set_equations_old(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        injection_well_grid, _ = self._filter_wells(subdomains, "injection")
        t_constraint = self.temperature_constraint_at_injection_wells(injection_well_grid)
        self.equation_system.set_equation(
            t_constraint,
            injection_well_grid,
            {"cells": 1}
        )

    def pressure_constraint_at_production_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns an constraint of form :math:`p - p_p=0` which replaces the
        pressure equation in production wells.

        Parameters:
            subdomains: A list of grids (tagged as production wells).

        Returns:
            The left-hand side of above equation.

        """

        p_production = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells)
                    * self._p_PRODUCTION[0]
                    for sd in subdomains
                ]
            ),
            name="production_pressure",
        )
        pressure_constraint_production = (self.pressure(subdomains) - p_production)
        pressure_constraint_production.set_name("production_pressure_constraint")
        return pressure_constraint_production
    
    def enthalpy_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        enthalpy at production wells."""
        T_injection = self._T_INJECTION[0]  # K
        # T_injection = self.get_ramped_injection_temperature()
        z_NaCl_injection = self._z_INJ["NaCl"]
        # self.get_ramped_injection_composition("NaCl")  # self._z_INJ['NaCl']  # 0.01% NaCl

        # Pressure at injection wells is assumed to be previously calculated pressure
        p_prev_val = self.pressure(subdomains).previous_timestep().value(self.equation_system)
        # Compute the enthalpy at the injection wells
        par_points = np.array([[z_NaCl_injection, T_injection, p_prev_val[0]]])
        self.vtk_sampler_ptz.sample_at(par_points)
        constant_h = self.vtk_sampler_ptz.sampled_cloud.point_data['H'][0]
        h_injection = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells) * constant_h
                    for sd in subdomains
                ]
            ),
            name="injection_enthalpy",
        )
      
        enthalpy_constraint_injection = self.enthalpy(subdomains) - h_injection
        enthalpy_constraint_injection.set_name("injection_enthalpy_constraint")
        return enthalpy_constraint_injection

    def temperature_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        enthalpy at production wells."""

        # =========================================================
        # USE RAMPED TEMPERATURE instead of fixed _T_INJECTION[0]
        # =========================================================
        # T_injection_value = self._T_INJECTION[0]  # self.get_ramped_injection_temperature()
        T_injection = pp.ad.TimeDependentDenseArray(
            name="injection_temperature", 
            domains=subdomains,
        )
        # T_injection = pp.wrap_as_dense_ad_array(
        #     np.hstack(
        #         [
        #             np.ones(sd.num_cells) * T_injection_value
        #             for sd in subdomains
        #         ]
        #     ),
        #     name="injection_temperature",
        # )
        
        temperature_constraint_injection = self.temperature(subdomains) - T_injection
        temperature_constraint_injection.set_name("injection_temperature_constraint")
        return temperature_constraint_injection


# =============================================================================
# VARIABLE CLAMPING MIXIN
# =============================================================================
class VariableClampingMixin:
    """Clamp variables to physical bounds after each Newton iteration.

    This prevents Newton overshoot from creating unphysical negative values
    for saturations and compositions.
    """

    def after_nonlinear_iteration(self, solution_increment):
        """Called after each Newton iteration. Clamps variables to valid ranges."""
        super().after_nonlinear_iteration(solution_increment)

        # List of variables to clamp to [0, 1]
        variables_to_clamp = ["s_gas", "s_halite", "z_NaCl"]

        for var_name in variables_to_clamp:
            try:
                vals = self.equation_system.get_variable_values([var_name], iterate_index=0)
                clamped = np.clip(vals, 0.0, 1.0)
                if not np.allclose(vals, clamped, atol=1e-10):
                    print(f"  Clamped {var_name}: [{vals.min():.2e}, {vals.max():.2e}] → [0, 1]")
                    self.equation_system.set_variable_values(clamped, [var_name], iterate_index=0)
            except (KeyError, ValueError) as e:
                # Variable might not exist or have different name
                print(f"  Warning: Could not clamp {var_name}: {e}")


class HaliteDependentApertureMixin:
    """
    Update fracture aperture as halite precipitates.

    Effective aperture:
        b_eff = b0 * (1 - alpha * S_halite)
    """
    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0
        well_tags = {"injection_well", "production_well"}
        fractures_and_intersections = [
            sd for sd in subdomains if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]
        wells_grids = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        for i, sd in enumerate(subdomains):
            if sd.dim == self.nd:
                a_loc = super().aperture([sd])
            elif sd in fractures_and_intersections:
                a_loc = self.fracture_or_intersection_aperture([sd])
            elif sd in wells_grids:
                a_loc = np.ones(sd.num_cells)*self._well_radius
            else:
                a_loc = super().apperture([sd])
            a_glob = projection.cell_prolongation([sd]) @ a_loc
            if i == 0:
                aperture = a_glob
            else:
                aperture += a_glob
        aperture.set_name("aperture")
        return aperture
    
    def fracture_or_intersection_aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        b_min = 1.0e-4  # m, minimum aperture to avoid zero permeability
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0
        # Clamp s_h to [0, 0.5]
        s_halite_array = clamped_halite_saturation(self, subdomains)
        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")
        # Project halite saturation into the global AD vector
        b0 = pp.ad.Scalar(self._fracture_aperture)

        # TODO: Idea from Eirik: use previous timestep saturation to relax non-linearity
        # and prevent stiff Jacobian.
        # Use previous Newton iterate instead of current
        # if hasattr(self, "_s_halite_prev"):
        #     s_h_prev = self._s_halite_prev
        #     s_halite_array = pp.wrap_as_dense_ad_array(s_h_prev, name="halite_s_prev")
        # else:
        #     # Fallback: current saturation if no history yet
        #     s_halite_array = clamped_halite_saturation(self, subdomains)

        effective_b = b0 * (pp.ad.Scalar(1.0) - s_halite_array)**pp.ad.Scalar(1.0)
        effective_b = max_fn(effective_b, pp.ad.Scalar(b_min))
        effective_b.set_name("fracture_intersection_aperture")

        return effective_b


# =============================================================================
# POROSITY MIXIN
# =============================================================================
class PorosityWithHaliteMixin2D(pp.PorePyModel):
    """
    Porosity model that reduces effective porosity based on halite saturation.

    Assumes that the presence of halite reduces the pore volume available to fluid phases.
    """

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity.

        Pressure and displacement dependent porosity in the matrix. Unitary in fractures
        and intersections.

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Porosity operator.

        """
        well_tags = {"injection_well", "production_well"}
        subdomains_nd = [sd for sd in subdomains if sd.dim == self.nd]
        subdomains_lower = [sd for sd in subdomains if sd.dim < self.nd and not well_tags.intersection(sd.tags)]
        subdomains_wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # Constant unitary porosity in fractures and intersections
        phi_nd = projection.cell_prolongation(subdomains_nd) @ self.porosity_matrix(
            subdomains_nd
        )
        
        phi_lower = projection.cell_prolongation(subdomains_lower) @ self.porosity_fracture_and_intersection(
            subdomains_lower
        )
        phi_wells = projection.cell_prolongation(subdomains_wells) @ self.porosity_wells(
            subdomains_wells
        )
        phi = phi_nd + phi_lower + phi_wells
        phi.set_name("porosity")
        
        return phi
    
    def porosity_fracture_and_intersection(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Sanity check
        """Porosity specifically for fractures and intersections."""
        return self.porosity_matrix(subdomains)
        # aperture = self.aperture(subdomains)
        # cell_sizes = np.hstack([np.sqrt(sd.cell_volumes) for sd in subdomains])
        return one   # aperture / (pp.wrap_as_dense_ad_array(cell_sizes))
    
    def porosity_wells(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Sanity check
        """Porosity specifically for OD grid"""
        return self.porosity_matrix(subdomains)
        well_poro = pp.wrap_as_dense_ad_array(0.1, size=sum(sd.num_cells for sd in subdomains), name="well_poro")   
        return well_poro  # self.porosity_matrix(subdomains)

    def porosity_matrix(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity in the nd-dimensional matrix [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].

        """

        # Sanity check
        # if not all([sd.dim == self.nd for sd in subdomains]):
        #     raise ValueError("Subdomains must be of dimension nd.")
        
        phi_0 = pp.ad.Scalar(self.solid.porosity, name="porosity")
        s_h_clamped = clamped_halite_saturation(self, subdomains)

        # Effective porosity: phi = phi_0 * (1 - s_halite)
        phi_matrix = phi_0 * (1.0 - s_h_clamped)
        phi_matrix.set_name("halite_updated_matrix_porosity")
        return phi_matrix


# =============================================================================
# THERMAL CONDUCTIVITY MIXIN
# =============================================================================
class ThermalConductivityMixinWithClampedHalite(pp.PorePyModel):
    """
    Mixin to compute thermal conductivity with halite saturation clamped to [0, 1].

    This avoids divergence or non-physical behavior caused by Newton updates that
    temporarily drive saturation outside [0, 1].
    """
    def fluid_thermal_conductivity(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """
        Computes effective thermal conductivity as:
            κ_eff = sum_j S_j * κ_j

        where S_j is clamped to [0, 1] for the halite phase only.
        """
        max_fn = pp.ad.Function(pp.ad.maximum, name="maximum_function")

        def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -max_fn(-a, -b)

        ops = []
        if self.fluid.num_phases > 1:
            ref_sat = self.fluid.reference_phase.saturation(domains)
            # halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
            # s_h = halite_phase[0].saturation(domains)

            # Clamp only once
            s_ref_clamped = min_fn(pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0)))
            s_h_clamped = clamped_halite_saturation(self, domains)

            for phase in self.fluid.phases:
                if phase.name.lower() == "halite":
                    saturation = s_h_clamped
                elif phase == self.fluid.reference_phase:
                    saturation = s_ref_clamped
                else:
                    total_sat = min_fn(pp.ad.Scalar(1.0), s_ref_clamped + s_h_clamped)
                    inferred = pp.ad.Scalar(1.0) - total_sat
                    # inferred = pp.ad.Scalar(1.0) - (s_ref_clamped + s_h_clamped)
                    saturation = min_fn(pp.ad.Scalar(1.0), max_fn(inferred, pp.ad.Scalar(0.0)))

                kappa = phase.thermal_conductivity(domains)
                ops.append(saturation * kappa)

            op = pp.ad.sum_operator_list(ops, name="fluid_thermal_conductivity")
        else:
            op = self.fluid.reference_phase.thermal_conductivity(domains)
            op.set_name("fluid_thermal_conductivity")

        return op


# =============================================================================
# PERMEABILITY MIXIN
# =============================================================================
class PermeabilityWithHaliteMixin2D(pp.PorePyModel):
    
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        This function combines the permeability of the matrix, fractures and
        intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        well_tags = {"injection_well", "production_well"}
        projection = pp.ad.SubdomainProjections(subdomains, dim=9)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        intersections = [
            sd for sd in subdomains if sd.dim == self.nd - 2 and not well_tags.intersection(sd.tags)
        ]
        wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        permeability = (
            projection.cell_prolongation(matrix)
            @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(wells)
            @ self.well_permeability(wells)
            + projection.cell_prolongation(fractures)
            @ self.fracture_permeability(fractures)
            + projection.cell_prolongation(intersections)
            @ self.intersection_permeability(intersections)
        )
        permeability.set_name("permeability")
        return permeability

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size,
            name="permeability"
        )

        s_h_clamped = clamped_halite_saturation(self, subdomains)
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm*reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    
    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.matrix_permeability(subdomains)
        size = sum(sd.num_cells for sd in subdomains)
        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, 
            size,
            name="permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, base_perm)
    
    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)
    
    def cubic_law_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic law permeability for fractures or intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        if len(subdomains) == 0:
            size = 0
            return pp.wrap_as_dense_ad_array(
                np.zeros((size,)), name="cubic_law_permeability"
            )
        aperture = self.aperture(subdomains)
        permeability = (aperture ** pp.ad.Scalar(2)) / pp.ad.Scalar(12)
        
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def cubic_law_permeability_old(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic law permeability for fractures or intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        if len(subdomains) == 0:
            size = 0
            return pp.wrap_as_dense_ad_array(
                np.zeros((size,)), name="cubic_law_permeability"
            )
        size = sum(sd.num_cells for sd in subdomains)
        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.fracture_permeability, 
            size,
            name="permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, base_perm)


# =============================================================================
# FINAL MODEL CONFIGURATION - WITH VariableClampingMixin
# =============================================================================
class LiquidPhaseFlowModelConfiguration2D(
    # VariableClampingMixin,
    InitializedBHPScheduleMixin,
    HaliteDependentApertureMixin,
    PorosityWithHaliteMixin2D,
    PermeabilityWithHaliteMixin2D,
    ThermalConductivityMixinWithClampedHalite,
    FluidMixture,
    ModifyComponentSourceMixin,
    ModifyFluidSourceMixin,
    ModifyEnergySourceMixin,
    ModifiedPrimaryEquationsMixin,
    LiquidSecondaryEquation2D,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):

    def relative_permeability_old(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(0.0)
        # halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        # if len(halite_phase) != 1:
        #     raise ValueError("Expected exactly one halite phase.")
    
        max = pp.ad.Function(pp.ad.maximum, "maximum_function")

        # name = phase.name
        s = phase.saturation(domains)

        # s_halite = halite_phase[0].saturation(domains)
        # Total mobile pore volume
        mobile_pore_volume = pp.ad.Scalar(1.0)  # * (1-s_halite)

        # Define residual saturations
        r_l = mobile_pore_volume * pp.ad.Scalar(0.3)
        r_v = pp.ad.Scalar(0.0)

        # Choose appropriate residual saturation
        if phase.name == "halite":
            return pp.ad.Scalar(0.0) * s
 
        if phase == self.fluid.reference_phase:
            s_eff = (s - r_l) / (1.0 - r_l - r_v)
            return max(s_eff, epsilon)
        else:
            s_eff = (s - r_v) / (1.0 - r_l - r_v)
            return max(s_eff, epsilon)
        
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(1e-12)  # small floor
        #halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        # if len(halite_phase) != 1:
        #     raise ValueError("Expected exactly one halite phase.")

        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")

        s = phase.saturation(domains)
        # s_halite = halite_phase[0].saturation(domains)
        s_halite = clamped_halite_saturation(self, domains)

        # Effective pore volume available for fluids
        mobile_pore_volume = 1.0 - s_halite

        # Residual saturations (scalable by mobile volume)
        s_l_res = pp.ad.Scalar(0.3) * mobile_pore_volume
        s_v_res = pp.ad.Scalar(0.00) * mobile_pore_volume

        if phase.name == "halite":
            return pp.ad.Scalar(0.0) * s

        if phase == self.fluid.reference_phase:  # say liquid
            s_eff = (s - s_l_res) / (mobile_pore_volume - s_l_res - s_v_res)
            s_eff = max_fn(s_eff, epsilon)
            return s_eff ** pp.ad.Scalar(1.5)  # Corey-type curve
        else:  # vapor
            s_eff = (s - s_v_res) / (mobile_pore_volume - s_l_res - s_v_res)
            return s_eff  # ** pp.ad.Scalar(2.0)