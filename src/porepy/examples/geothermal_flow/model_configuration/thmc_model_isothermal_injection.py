"""
Simulation: Salt dissolution and precipitation in a horizontal 2D geothermal reservoir 
with point injection and production wells, coupled with Biot poromechanics.

Description:
------------
This experiment injects low-salinity water (z_NaCl ≈ 1e-4) into a halite-saturated 
geological formation, with initial NaCl mass fraction z_NaCl = 0.40. The setup uses 
a mixed-dimensional grid with fractures and point grid wells at x ≈ 15 (injector) and 
x ≈ 85 (producer), simulating three phases: liquid, vapor, and solid halite.

The model now includes Biot thermo-poromechanics coupling:
- Effective stress: sigma = sigma_mechanical - alpha·p·I - β·K·(T-T_ref)·I
- Porosity: phi = phi_ref + (alpha-phi_ref)(1-alpha)/K·Δp + alpha·∇·u + thermal_contribution
- Halite precipitation effects on porosity and permeability

Boundary conditions:
--------------------
- Left boundary: **Dirichlet pressure** fixed at the initial reservoir pressure, 
  representing connection to an infinite reservoir (outflow allowed).
- All other outer boundaries: **no-flow** (impermeable).
- Mechanical: roller BCs on sides and bottom, free surface on top (or overburden stress).
- Wells: injection well imposes enthalpy/composition of the injected fluid; 
  production well is controlled by a fixed bottom-hole pressure.

Author: Michael Oguntola
"""

from typing import Callable
from xml.parsers.expat import model
import porepy as pp
import numpy as np
import scipy.sparse as sps

# TODO: cff or cf, both will be merged into the same modules, for now it is experimenter
from porepy.models.derived_models.poromechanics_compositional_cff import (
    PoromechanicsCompositionalTemplate
)
import porepy.models.compositional_flow as cf
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


class MultiphaseSecondaryEquation(SecondaryEquations):
    component_system = ComponentSystem.WATER_SALT
    phase_mode = PhaseMode.THREE_PHASE

# class MultiphaseSecondaryEquation(SecondaryEquations):
#     component_system = ComponentSystem.WATER
#     phase_mode = PhaseMode.TWO_PHASE # pp.compositional.PhysicalState.liquid


# =============================================================================
# Shared vertical-section geometry / reference constants
# =============================================================================
class VerticalSectionConstants:
    """Depth anchoring and gradients for the vertical section.

    Override these on your model (or in the driver) to match your site. Defaults
    assume the section top sits at 2 km and reuse the existing reservoir state as
    the TOP-of-section reference.
    """
    _H_SECTION: float = 30.0        # m, vertical extent of the box
    _D_TOP: float = 2000.0          # m, true depth of the section TOP
    _G_T: float = 0.03              # K/m, geothermal gradient (~30 K/km)

    # Top-of-section reference state (anchors the hydrostatic / geothermal profiles).
    # These reuse your existing reservoir values as the reference at the section top.
    # _p_TOP: float = 15.5e6          # Pa, pressure at y = H (section top)
    # _T_TOP: float = 586.15          # K, temperature at y = H (section top)

    # Representative constant fluid density for the hydrostatic profile [kg/m^3].
    # For a 30 m span the density variation is negligible; if you want the exact
    # implicit profile, iterate _hydrostatic_pressure once using sampled rho.
    # _RHO_F_REF: float = 1000.0

    def _reference_fluid_density(self) -> float:
        """Sample EOS density at the section-top reference state."""
        par = np.array([[self._z_INIT["NaCl"], self._T_INIT, self._p_INIT]])
        self.vtk_sampler_ptz.sample_at(par)
        return float(self.vtk_sampler_ptz.sampled_cloud.point_data["Rho"][0])

    def _depth_from_y(self, y: np.ndarray) -> np.ndarray:
        """True depth at cell-centre height y."""
        return self._D_TOP + (self._H_SECTION - y)

    def _hydrostatic_pressure(self, y: np.ndarray) -> np.ndarray:
        """p(y) = p_TOP + rho_f g (H - y)."""
        if not self.params.get("enable_buoyancy_effects", False):
            return np.full_like(y, self._p_INIT)
        g = pp.GRAVITY_ACCELERATION
        self._RHO_F_REF = self._reference_fluid_density()
        return self._p_INIT + self._RHO_F_REF * g * (self._H_SECTION - y)

    def _geothermal_temperature(self, y: np.ndarray) -> np.ndarray:
        """T(y) = T_TOP + G_T (H - y)."""
        if not self.params.get("enable_buoyancy_effects", False):
            return np.full_like(y, self._T_INIT)
        return self._T_INIT + self._G_T * (self._H_SECTION - y)


# =============================================================================
# SOLID CONSTANTS WITH MECHANICAL PROPERTIES
# =============================================================================
@dataclass(kw_only=True)
class PoromechanicsSolidConstants(pp.SolidConstants):
    """Solid constants extended for poromechanics (basalt).
    
    Includes standard mechanical properties plus fracture permeability.
    """
    SI_units: ClassVar[dict[str, str]] = dict(**pp.SolidConstants.SI_units)
    SI_units.update({"fracture_permeability": "m^2"})
    
    # Fracture properties
    fracture_permeability: pp.number = 1.0
    
    # Mechanical properties (typical values for sandstone/limestone)
    # These should be overridden with site-specific values
    lame_lambda: pp.number = 31.2e9       # Pa (first Lamé parameter)
    shear_modulus: pp.number = 31.2e9     # Pa (second Lamé parameter, μ)
    biot_coefficient: pp.number = 0.6    # [-] (0 ≤ α ≤ 1)
    specific_storage: pp.number = 2.5e-10
    friction_coefficient:pp.number = 0.7

    # Thermal properties
    thermal_expansion: pp.number = 5.4e-6  # [1/Kelvin] # was: 5.0e-6 


def clamped_halite_saturation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
    """Clamp halite saturation between [0, 0.5]."""
    halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
    if len(halite_phase) == 0:
        return pp.ad.Scalar(0.0)
    elif len(halite_phase) > 1:
        raise ValueError("Expected exactly one halite phase.")
    
    s_h_raw = halite_phase[0].saturation(subdomains)
    # return s_h_raw
    # s_h_raw = self.equation_system.evaluate(s_h_raw_op)
    # s_h_raw = pp.wrap_as_dense_ad_array(s_h_raw, name="halite_saturation_raw")

    max_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

    def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
        return -max_fn(-a, -b)
    
    min_val = min_fn(pp.ad.Scalar(0.8), max_fn(s_h_raw, pp.ad.Scalar(0.0)))
    return min_val


# =============================================================================
# WELL FLOW DATA
# =============================================================================
class WellFlowData(pp.PorePyModel):
    """Helper class to bundle configuration of pressure, temperature, and injected mass
    for a multiphase and salt-water geothermal problem with one injector and one producer."""

    vtk_sampler_ptz: VTKSampler

    # Initial reservoir conditions (representative of ~2 km depth). I am bad at french. List going to.
    _p_INIT: float = 11.0e6         # Pa       # 15.0e6 maybe put 20.0e6 for matrix cell
    _T_INIT: float = 586.651      # K       # 586.451,  Note, at 10.5MPa, 587.6 wont work! 
    _z_INIT: dict[str, float] = {"H2O": 0.618, "NaCl": 0.382}

    # In- and outflow values.
    _T_INJ: float = 587.76     # K (injection temperature computed from _geothermal_temperature())
    _z_INJ: dict[str, float] = {"H2O": 0.999, "NaCl": 0.001}

    _INJECTION_FRACTION: float = 1.0   # kg/kg
    
    _p_OUT: float = 7.0e6          # Pa (fixed production pressure) NOTE: 7.0e6 for matrix cell!
    _well_radius: float = 0.1      # m (for well index calculation)

    # Initial and injected fluid composition.
    _fracture_aperture = 1.0e-3  # m (initial fracture aperture)

    # Injection schedule (can extend later with time-dependent keys)
    _T_INJECTION: dict[int, float] = {0: _T_INJ}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
    _p_INJECTION: dict[int, float] = {0: _p_INIT}
    
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
        """Calculate total injected mass (in kg/m3/s).
        """

        T_inj = self._T_INJ
        z_NaCl = self._z_INJ["NaCl"]
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
        """Return injected mass source density for a given component [kg/m3/s].
        """
        
        # Total mass injection rate in [kg / s]
        total_mass_injection_kg_per_s = self._get_total_injected_mass_rate()  

        # Injected mass fraction per components
        injected_mass_per_component = {
            "H2O": total_mass_injection_kg_per_s * self._z_INJ["H2O"],
            "NaCl": total_mass_injection_kg_per_s * self._z_INJ["NaCl"],
        }

        injected_mass: list[np.ndarray] = []
        for sd in subdomains:
            assert "injection_well" in sd.tags, (
                f"Grid {sd.id} not tagged as injection well."
            )
            # Source term is constant for now (could generalize if multiple injection wells)
            injected_mass.append(
                np.ones(sd.num_cells)
                * injected_mass_per_component[component.name]
            )

        if injected_mass:
            source = np.hstack(injected_mass)
        else:
            source = np.zeros((0,))

        return pp.ad.DenseArray(source, f"injected_mass_density_{component.name}")


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
# FLUID SOURCE MIXIN
# =============================================================================
class ModifyFluidSourceMixin(WellFlowData):
    """ Modify the fluid source term for the pressure equations at the injection wells.
    """
    
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Augments the source term in the pressure equation to account for the mass
        injected through injection wells."""
        source: pp.ad.Operator = super().fluid_source(subdomains)  # type:ignore[misc]

        injection_wells_grid, _ = self._filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        src_inj = self._get_total_injected_mass_rate()  # self.units.convert_units(0.1, "kg * m^-3 * s^-1")

        # Unit: kg/s
        injected_mass: pp.ad.Operator = self.volume_integral(
            src_inj,
            injection_wells_grid,
            1,
        )
        injected_mass.set_name("injected_fluid_mass")
        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells_grid) @ injected_mass
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
# PRIMARY EQUATIONS MIXIN
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

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""
        _, no_production_wells = self._filter_wells(subdomains, "production")
        eq: pp.ad.Operator = super().mass_balance_equation(no_production_wells)  # type:ignore[misc]
        # name = eq.name
        return eq

    def set_equations(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively, and removes the balance equation each one replaces."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        injection_well_grid, no_injection = self._filter_wells(subdomains, "injection")
        production_well_grid, no_production = self._filter_wells(subdomains, "production")

        t_constraint = self.temperature_constraint_at_injection_wells(injection_well_grid)
        self.equation_system.set_equation(
            t_constraint, injection_well_grid, {"cells": 1}
        )

        p_constraint = self.pressure_constraint_at_production_wells(production_well_grid)
        self.equation_system.set_equation(
            p_constraint, production_well_grid, {"cells": 1}
        )

        # The base class registered these on ALL subdomains. Re-register them on
        # the reduced domains, so the constraint above supplies the row for the
        # well cell instead of duplicating it.
        self.equation_system.update_equation(
            "mass_balance_equation",
            super().mass_balance_equation(no_production),
            no_production,
        )
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
        production_well_grid, _ = self._filter_wells(subdomains, "production")

        t_constraint = self.temperature_constraint_at_injection_wells(injection_well_grid)
        self.equation_system.set_equation(
            t_constraint,
            injection_well_grid,
            {"cells": 1}
        )
        
        p_constraint = self.pressure_constraint_at_production_wells(production_well_grid)
        self.equation_system.set_equation(
            p_constraint,
            production_well_grid,
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
    

    def temperature_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        enthalpy at production wells."""

        T_inj = self._T_INJECTION[0]
        temperature_constraint_injection = self.temperature(subdomains) - T_inj
        temperature_constraint_injection.set_name("injection_temperature_constraint")
        return temperature_constraint_injection
    
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

        effective_b = b0 * (pp.ad.Scalar(1.0) - s_halite_array)**pp.ad.Scalar(0.1)
        effective_b = max_fn(effective_b, pp.ad.Scalar(b_min))
        effective_b.set_name("fracture_intersection_aperture")

        return effective_b


# =============================================================================
# POROSITY MIXIN WITH BIOT + HALITE COUPLING
# =============================================================================
class PorosityWithBiotAndHaliteMixin(pp.PorePyModel):
    """
    Porosity model combining:
    1. Biot poromechanics: phi = φ_ref + (alpha-phi_ref)(1-alpha)/K·Δp + alpha·∇·u
    2. Thermal effects: - (alpha - phi_ref)·β·ΔT
    3. Halite saturation: phi_eff = phi * (1 - S_halite)
    """

    # Type hints for methods from other mixins
    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    bulk_modulus: Callable[[list[pp.Grid]], pp.ad.Operator]
    displacement_divergence: Callable[[list[pp.Grid]], pp.ad.Operator]
    solid_thermal_expansion_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Total porosity including Biot, thermal, and halite effects."""
        
        well_tags = {"injection_well", "production_well"}
        subdomains_nd = [sd for sd in subdomains if sd.dim == self.nd]
        subdomains_lower = [
            sd for sd in subdomains 
            if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]
        subdomains_wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # Matrix porosity with full Biot coupling
        phi_nd = projection.cell_prolongation(subdomains_nd) @ self.matrix_porosity(
            subdomains_nd
        )
        
        # Fractures: unit porosity reduced by halite
        phi_lower = projection.cell_prolongation(subdomains_lower) @ self.fracture_porosity(
            subdomains_lower
        )
        
        # Wells: same as matrix for simplicity
        phi_wells = projection.cell_prolongation(subdomains_wells) @ self.well_porosity(
            subdomains_wells
        )
        
        phi = phi_nd + phi_lower + phi_wells
        phi.set_name("porosity")
        return phi
    
    def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Matrix porosity with Biot poromechanics + halite reduction.
        
        φ = [φ_ref + pressure_contribution + displacement_contribution + thermal_contribution] * (1 - S_h)
        """
        if len(subdomains) == 0 or not all(sd.dim == self.nd for sd in subdomains):
            # Return reference porosity for empty or non-matrix subdomains
            # return pp.wrap_as_dense_ad_array(0.0, size=0, name="matrix_porosity_empty")
            return pp.ad.Scalar(self.solid.porosity, name="reference_porosity")
        
        # Reference porosity
        phi_ref = self.reference_porosity(subdomains)
        
        # Biot contributions (from ThermoPoroMechanicsPorosity)
        phi_biot = (
            phi_ref
            + self.porosity_change_from_pressure(subdomains)
            + self.porosity_change_from_displacement(subdomains)
            + self.porosity_change_from_temperature(subdomains)
        )
        
        # Add MPSA consistency term if not using TPSA
        if not isinstance(self.stress_discretization(subdomains), pp.ad.TpsaAd):
            phi_biot += self._mpsa_consistency(
                subdomains, self.darcy_keyword, self.pressure_variable
            )
        
        # Halite reduction
        s_h = clamped_halite_saturation(self, subdomains)
        phi_matrix = phi_biot  * (pp.ad.Scalar(1.0) - s_h)
        phi_matrix.set_name("matrix_porosity_biot_halite")
        
        return phi_matrix
    
    def fracture_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture porosity: unit porosity reduced by halite."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(1.0, size=0, name="fracture_porosity")
        
        size = sum(sd.num_cells for sd in subdomains)
        one = pp.wrap_as_dense_ad_array(1.0, size=size, name="one")
        
        s_h = clamped_halite_saturation(self, subdomains)
        phi_fracture = one * (pp.ad.Scalar(1.0) - s_h)
        phi_fracture.set_name("fracture_porosity")
        
        return phi_fracture
    
    def well_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Well porosity: same as matrix for consistency."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0.0, size=0, name="well_porosity_empty")
        # return self.matrix_porosity(subdomains)
        size = sum(sd.num_cells for sd in subdomains)
        return pp.wrap_as_dense_ad_array(
            self.solid.porosity, size, name="well_porosity_constant"
        )

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
class CaprockPermeabilityMixin2D(pp.PorePyModel):
    """Permeability with a low-permeability caprock band at the top of the section.

    Inherit this INSTEAD OF (or above) your existing ``PermeabilityWithHaliteMixin2D``
    so its ``matrix_permeability`` override wins. It reuses the halite reduction and
    the CFF total-mass-mobility factor, and multiplies in a depth-dependent caprock
    factor that is ~1 in the reservoir and ``_CAP_PERM_FACTOR`` (<<1) in the cap band.
    """

    # ---- caprock geometry / strength (override per case) ----
    _CAP_THICKNESS: float = 5.0       # m, vertical extent of the seal at the top
    _CAP_PERM_FACTOR: float = 1.0e-4  # permeability multiplier inside the cap (1e-4 = 10^4 tighter)
    _H_SECTION: float = 30.0          # m, total vertical extent (top at y = H)

    def _caprock_multiplier(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Cell-wise multiplier: 1.0 in the reservoir, _CAP_PERM_FACTOR in the cap.

        The cap is the band y > H - CAP_THICKNESS (top of the section). Returns a
        length-(sum num_cells) array aligned with the subdomain cell ordering.
        """
        parts = []
        for sd in subdomains:
            y = sd.cell_centers[1]
            in_cap = y >= (self._H_SECTION - self._CAP_THICKNESS)
            parts.append(np.where(in_cap, self._CAP_PERM_FACTOR, 1.0))
        if not parts:
            return np.zeros(0)
        return np.hstack(parts)

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Matrix permeability = K_abs * (1-S_h)^2 * caprock_factor * [total_mobility].

        Same structure as the halite mixin, with the caprock factor folded into the
        scalar permeability before the tensor expansion.
        """
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size, name="permeability"
        )
        
        porosity_eff = self.porosity(subdomains)
        ref_porosity = self.reference_porosity(subdomains)

        corrected_perm = base_perm * (porosity_eff / ref_porosity) ** 2.0

        # Caprock: depth-dependent low-perm band at the top.
        cap = pp.wrap_as_dense_ad_array(
            self._caprock_multiplier(subdomains), name="caprock_factor"
        )

        corrected_perm = corrected_perm * cap

        # CFF: fold total mass mobility into the diffusive tensor.
        if cf.is_fractional_flow(self):
            corrected_perm = corrected_perm * self.total_mass_mobility(subdomains)

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    

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

    def matrix_permeability_no_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size,
            name="permeability"
        )

        s_h_clamped = clamped_halite_saturation(self, subdomains)
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm *reduction
        
        if cf.is_fractional_flow(self):
            corrected_perm = corrected_perm * self.total_mass_mobility(subdomains)

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    
    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        size = sum(sd.num_cells for sd in subdomains)
        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size,
            name="permeability"
        )
        porosity_eff = self.porosity(subdomains)

        ref_porosity = self.reference_porosity(subdomains)
        corrected_perm = base_perm * (porosity_eff / ref_porosity) ** 2.0
        
        if cf.is_fractional_flow(self):
            corrected_perm = corrected_perm * self.total_mass_mobility(subdomains)

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    
    
    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # return self.matrix_permeability(subdomains)
        size = sum(sd.num_cells for sd in subdomains)
        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, 
            size,
            name="permeability"
        )
        
        if cf.is_fractional_flow(self):
            base_perm = base_perm * self.total_mass_mobility(subdomains)
            
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
class CFPoromechanicalModelConfiguration2D(
    # PorosityWithHaliteMixin2D, 
    VerticalSectionConstants,   
    PorosityWithBiotAndHaliteMixin,  #<--- Possible to switch off if I dont want Halite impact on Permeability?
    CaprockPermeabilityMixin2D,      #<----Caprock configuration
    PermeabilityWithHaliteMixin2D,   #<--- Possible to switch off if I dont want Halite impact on permeability?
    ThermalConductivityMixinWithClampedHalite,
    FluidMixture,
    ModifyComponentSourceMixin,
    ModifyFluidSourceMixin,
    ModifyEnergySourceMixin,
    ModifiedPrimaryEquationsMixin,
    MultiphaseSecondaryEquation,
    PoromechanicsCompositionalTemplate,
    VTKSamplerMixin
):
    """
        Biot poromechanics coupled with compositional H2O-NaCl flow.
        
        This model includes:
        - Momentum balance with thermo-poroelastic effective stress
        - Mass balance for H2O and NaCl components
        - Energy balance (non-isothermal)
        - Halite precipitation/dissolution effects on porosity/permeability
        - Optional fracture contact mechanics (inactive without fractures)
    """
        
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(1e-12)  # small floor
    
        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")

        s = phase.saturation(domains)
        # s_halite = halite_phase[0].saturation(domains)
        s_halite = clamped_halite_saturation(self, domains)

        # Effective pore volume available for fluids
        mobile_pore_volume = 1.0
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