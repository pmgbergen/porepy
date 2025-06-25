"""
Simulation: Salt dissolution and precipitation in a horizontal 2D geothermal reservoir 
with point injection and production wells.

Description:
------------
This experiment injects low-salinity water (z_NaCl ≈ 1e-4) into a halite-saturated 
geological formation, with initial NaCl mass fraction z_NaCl = 0.40. The setup uses 
a mixed-dimensional grid with fractures and point grid wells at x ≈ 15 (injector) and 
x ≈ 85 (producer), simulating three phases: liquid, vapor, and solid halite.

Observations from the halite saturation profile:
------------------------------------------------
- A sharp **drop in halite saturation** is observed near the injection well due 
  to halite **dissolution** from undersaturated fluid.
- A **local spike in saturation** occurs near the production well, consistent with 
  **salt precipitation** as water is extracted and brine concentrates.
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
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

from .constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem,
    PhaseMode,
)

from typing import Sequence


class VTKSamplerMixin:
    """Mixin to handle VTK sampling for pressure-temperature/enthalpy-composition data."""
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


class BrineSecondaryEquationMixin(SecondaryEquations):
    component_system = ComponentSystem.WATER_SALT
    phase_mode = PhaseMode.THREE_PHASE


class WellFlowData(pp.PorePyModel):
    """Helper class to bundle configuration of pressure, temperature, and injected mass
    for a single-phase pure-water geothermal problem with one injector and one producer."""

    vtk_sampler_ptz: VTKSampler

    # Initial reservoir conditions (representative of ~2 km depth).
    _p_INIT: float = 4.0e6          # Pa
    _T_INIT: float = 423.15         # K

    # Injection temperature
    _T_INJ: float = 300.15          # K

    # Production pressure
    _p_OUT: float = 3.0e6           # Pa

    # Initial and injected fluid composition.
    _z_INIT: dict[str, float] = {"H2O": 0.60, "NaCl": 0.40}
    _z_INJ: dict[str, float] = {"H2O": 0.9999, "NaCl": 1.0e-4}

    # Injection rate (m³/h)
    _INJECTION_RATE: float = 0.1

    # Injection schedule (can extend later with time-dependent keys)
    _T_INJECTION: dict[int, float] = {0: _T_INJ}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}

    def _get_fluid_density(
        self,
        temperature: float,
        pressure: float,
        z_NaCl: float
    ) -> tuple[float, float, float]:
        """Sample bulk fluid density from the VTK table."""
        par_point = np.array([[z_NaCl, temperature, pressure]])
        self.vtk_sampler_ptz.sample_at(par_point)
        data = self.vtk_sampler_ptz.sampled_could.point_data
        rho = data["Rho"][0]
        return rho
    
    def _get_total_injected_mass_rate(
        self
    ) -> float:
        """Calculate total injected mass (kg/s)."""
        rho = self._get_fluid_density(
            temperature=self._T_INJECTION[0],
            pressure=self._p_INIT,
            z_NaCl=self._z_INJ["NaCl"]
        )
        return self._INJECTION_RATE * rho / 3600.0  # kg/m3/s
  
    def _injected_component_mass(  # Not correct! TODO: i need to set the accurate amount of injected mass for NaCL
        self,
        component: pp.Component,
        subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the injected mass of a fluid component in [kg / s].

        Parameters:
            component: A fluid component.
            subdomains: A list of injection wells (dim=0 grids)

        Returns:
            AD dense array for source term [kg / s].
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
            injected_mass.append(
                np.ones(sd.num_cells)
                * injected_mass_per_component[component.name]
            )

        if injected_mass:
            source = np.hstack(injected_mass)
        else:
            source = np.zeros((0,))

        return pp.ad.DenseArray(source, f"injected_mass_density_{component.name}")


class ModifiedComponentSourceMixin:
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

        injection_wells_grid, _ = self.filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        injected_mass = self.volume_integral(
            self._injected_component_mass(component, injection_wells_grid),
            injection_wells_grid,
            1,
        )

        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells_grid) @ injected_mass
        )

        production_wells_grid, _ = self.filter_wells(subdomains, "production")
        source -= subdomain_projections.cell_prolongation(production_wells_grid) @ (
            subdomain_projections.cell_restriction(production_wells_grid) @ source
        )

        return source


class ModifiedFluidSourceMixin(WellFlowData):
    """ Modify the fluid source term for the pressure equations at the injection wells.
    """
    
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Augments the source term in the pressure equation to account for the mass
        injected through injection wells."""
        source: pp.ad.Operator = super().fluid_source(subdomains)  # type:ignore[misc]

        injection_wells_grid, _ = self.filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        src_inj = self._get_total_injected_mass_rate() #self.units.convert_units(0.1, "kg * m^-2 * s^-1")

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


class ModifiedEnergySourceMixin:

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

        production_wells, _ = self.filter_wells(subdomains, "production")
        _, no_injection_wells = self.filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(no_injection_wells)
        source -= subdomain_projections.cell_prolongation(production_wells) @ (
            subdomain_projections.cell_restriction(production_wells) @ source
        )
        return source


class ModifiedPrimaryEquationsMixin:
    # Adjusting PDEs
    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""

        _, no_production_well_grids = self.filter_wells(subdomains, "production")
        eq: pp.ad.Operator = super().mass_balance_equation(no_production_well_grids)  # type:ignore[misc]
        return eq
    
    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""

        _, no_injection_well_grids = self.filter_wells(subdomains, "injection")
        eq: pp.ad.Operator = super().energy_balance_equation(no_injection_well_grids)  # type:ignore[misc]
        # name = eq.name
        return eq

    def set_equations(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        production_well_grid, _ = self.filter_wells(subdomains, "production")
        injection_well_grid, _ = self.filter_wells(subdomains, "injection")

        p_constraint = self.pressure_constraint_at_production_wells(production_well_grid)
        self.equation_system.set_equation(
            p_constraint,
            production_well_grid,
            {"cells": 1}
        )

        h_constraint = self.enthalpy_constraint_at_injection_wells(injection_well_grid)
        self.equation_system.set_equation(
            h_constraint,
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
        pressure_constraint_production = self.pressure(subdomains) - p_production
        pressure_constraint_production.set_name("production_pressure_constraint")
        return pressure_constraint_production
    
    def enthalpy_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        enthalpy at production wells."""
        T_injection = self._T_INJECTION[0]  # K
        z_NaCl_injection = self._z_INJ['NaCl']  # 0.01% NaCl

        # Pressure at injection wells is assumed to be previously calculated pressure
        # p_prev_val = pp.ad.TimeDependentDenseArray(
        #     name="pressure", domains=subdomains
        # ).previous_timestep().value(self.equation_system)
        p_prev_val = self.pressure(subdomains).value(self.equation_system)

        # Compute the enthalpy at the injection wells
        par_points = np.array([[z_NaCl_injection, T_injection, p_prev_val[0]]])
        self.vtk_sampler_ptz.sample_at(par_points)
        h_injection = self.vtk_sampler_ptz.sampled_could.point_data['H'][0]

        h_injection = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells) * h_injection
                    for sd in subdomains
                ]
            ),
            name="injection_enthalpy",
        )
        
        enthalpy_constraint_injection = self.enthalpy(subdomains) - h_injection
        enthalpy_constraint_injection.set_name("injection_enthalpy_constraint")
        return enthalpy_constraint_injection


class PorosityWithHaliteMixin2D(pp.PorePyModel):
    """
    Porosity model that reduces effective porosity based on halite saturation.

    Assumes that the presence of halite reduces the pore volume available to fluid phases.
    """

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Base porosity from solid
        phi_0 = pp.ad.Scalar(self.solid.porosity, name="porosity")

        # Retrieve halite phase (must be present in self.fluid.phases)
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required for porosity correction.")

        s_h_raw = halite_phase[0].saturation(subdomains)

        # Clamp s_h to [0, 0.5]   
        maximum_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

        def minimum_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum_fn(-a, -b)

        # similar to a minmod limiter
        s_h_clamped = minimum_fn(
            pp.ad.Scalar(0.5),
            maximum_fn(s_h_raw, pp.ad.Scalar(0.0))
        )

        # Effective porosity: phi = phi_0 * (1 - s_halite)
        return phi_0 * (1.0 - s_h_clamped)


class PermeabilityWithHaliteMixin2D(pp.PorePyModel):

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size, 
            name="permeability"
        )

        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required for permeability correction.")

        s_h_raw = halite_phase[0].saturation(subdomains)

        # Clamp s_h to [0, 0.5]   
        maximum_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

        def minimum_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum_fn(-a, -b)
        
        # similar to a minmod limiter
        s_h_clamped = minimum_fn(
            pp.ad.Scalar(0.5),
            maximum_fn(s_h_raw, pp.ad.Scalar(0.0))
        )

        # Example reduction: perm_eff = perm_0 * (1 - s_halite)^2
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm*reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)


class BrineFlowModelConfiguration2D(
    PorosityWithHaliteMixin2D,
    PermeabilityWithHaliteMixin2D,
    FluidMixture,
    ModifiedComponentSourceMixin,
    ModifiedFluidSourceMixin,
    ModifiedEnergySourceMixin,
    ModifiedPrimaryEquationsMixin,
    BrineSecondaryEquationMixin,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):

    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(0.0)
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        if len(halite_phase) != 1:
            raise ValueError("Expected exactly one halite phase.")
    
        max = pp.ad.Function(pp.ad.maximum, "maximum_function")

        # name = phase.name
        s = phase.saturation(domains)

        # Total mobile pore volume
        mobile_pore_volume = pp.ad.Scalar(1.0)  # (1-s_halite)

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
            return (s - r_v) / (1.0 - r_l - r_v)
