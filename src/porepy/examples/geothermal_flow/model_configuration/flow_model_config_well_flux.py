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
from porepy.models.constitutive_laws import CubicLawPermeability
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


# def adaptive_injection_rate(self):
#     """Boost injection to push front to producer"""
#     time_years = self.time_manager.time / (365 * 86400)

#     if time_years < 0.12:
#         return self._INJECTION_RATE  # Double injection
#     else:
#         return 5.0


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
    # if hasattr(self, "_s_halite_prev"):
    #     s_h_prev = self._s_halite_prev
    #     s_h_raw = pp.wrap_as_dense_ad_array(s_h_prev, name="halite_s_prev")
    # else:
        # Fallback: current saturation if no history yet
    halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
    if len(halite_phase) != 1:
        raise ValueError("Expected exactly one halite phase.")
    s_h_raw = halite_phase[0].saturation(subdomains)

    # s_h_raw = self.equation_system.evaluate(s_h_raw_op)
    # s_h_raw = pp.wrap_as_dense_ad_array(s_h_raw, name="halite_saturation_raw")

    max_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

    def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
        return -max_fn(-a, -b)
    
    min_val = min_fn(pp.ad.Scalar(0.5), max_fn(s_h_raw, pp.ad.Scalar(0.0)))
    return min_val


class WellFlowData(pp.PorePyModel):
    """Helper class to bundle configuration of pressure, temperature, and injected mass
    for a single-phase pure-water geothermal problem with one injector and one producer."""

    vtk_sampler_ptz: VTKSampler

    # Initial reservoir conditions (representative of ~2 km depth).
    _p_INIT: float = 10.5e6         # Pa       # 15.0e6 maybe put 20.0e6 for matrix cell
    _T_INIT: float = 586.651         # K       # 588.451
    # _H_INIT: float = 2.35e6         # J/kg (enthalpy at initial conditions)

    # In- and outflow values.
    _T_INJ: float = 300.15  # 300.15, 343.15          # K (injection temperature)
    _p_OUT: float = 6.5e6          # Pa (fixed production pressure) NOTE: 7.0e6 for matrix cell!
    _well_radius: float = 0.1      # m (for well index calculation)

    # Initial and injected fluid composition.
    _z_INIT: dict[str, float] = {"H2O": 0.6, "NaCl": 0.4}
    _z_INJ: dict[str, float] = {"H2O": 0.9999, "NaCl": 1.0e-4}
    _fracture_aperture = 1.0e-3  # m (initial fracture aperture)

    # Injection rate of low-salinity water per well.
    _INJECTION_RATE: float = 2.0  # m³/h # use 5.0 for a single fracture with cartessian grid

    # Injection schedule (can extend later with time-dependent keys)
    _T_INJECTION: dict[int, float] = {0: _T_INJ}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
    _p_INJECTION: dict[int, float] = {0: _p_INIT}  #TODO: +7.0e6

    def _get_fluid_density(
        self,
        temperature: float,
        pressure: float, 
        z_NaCl: float
    ) -> tuple[float, float, float]:
        """Sample bulk fluid density from the VTK table."""
        par_point = np.array([[z_NaCl, temperature, pressure]])
        self.vtk_sampler_ptz.sample_at(par_point)
        data = self.vtk_sampler_ptz.sampled_cloud.point_data
        rho = data["Rho"][0]
        return rho
    
    def _get_fluid_viscosity(
        self,
        temperature: float,
        pressure: float,
        z_NaCl: float
    ) -> tuple[float, float, float]:
        """Sample bulk fluid density from the VTK table."""
        par_point = np.array([[z_NaCl, temperature, pressure]])
        self.vtk_sampler_ptz.sample_at(par_point)
        data = self.vtk_sampler_ptz.sampled_cloud.point_data
        mu = data["mu_l"][0]
        return mu
 
    def adaptive_production_pressure(self):
        """Reduce drawdown during critical phase"""
        time_years = self.time_manager.time / (365 * 86400)

        if time_years < 20.6:         # First ~2 days
            self._p_OUT = 6.5e6
            return 6.5e6             # Gentle drawdow
        else:
            self._p_OUT = 6.5e6
            return 6.5e6

    def _get_total_injected_mass_rate(
        self,
        subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Calculate total injected mass (in kg/s)."""
   
        rho = self.fluid.density(subdomains)
        return pp.ad.Scalar(self._INJECTION_RATE) * rho / pp.ad.Scalar(3600.0)  # kg/s

    def _injected_component_mass_rate(
        self,
        component: pp.Component,
        subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Return injected mass source density for a given component [kg/m³/s].

        Parameters
        ----------
        component : pp.Component
            The fluid component (e.g. H2O, NaCl).
        subdomains : list of pp.Grid
            Grids representing injection wells (0D).

        Returns
        -------
        pp.ad.Operator
            AD operator for the volumetric source density [kg/m³/s].
        """

        if not subdomains:
            return pp.ad.DenseArray(np.zeros((0,)), f"no_injection_{component.name}")

        # --- total injected mass rate [kg/s]
        total_mass_injection = self._get_total_injected_mass_rate(subdomains)

        # --- component-specific mass rate [kg/s]
        comp_mass_rate = pp.ad.Scalar(self._z_INJ[component.name]) * total_mass_injection

        # --- convert to source density [kg/m³/s]
        cell_volumes = np.hstack([sd.cell_volumes for sd in subdomains])
        src_density = comp_mass_rate / pp.ad.Scalar(cell_volumes)

        # --- integrate to AD operator
        volumetric_mass_rate = self.volume_integral(src_density, subdomains, 1)
        volumetric_mass_rate.set_name(f"injected_mass_density_{component.name}")

        return volumetric_mass_rate


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

        injection_wells_grid, _ = self.filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(self.mdg.subdomains())

        # volumetric source density [kg/m³/s]
        injected_mass = self._injected_component_mass_rate(component, injection_wells_grid)

        source += subdomain_projections.cell_restriction(subdomains) @ (
            subdomain_projections.cell_prolongation(injection_wells_grid) @ injected_mass
        )

        production_wells_grid, _ = self.filter_wells(subdomains, "production")
        source -= subdomain_projections.cell_prolongation(production_wells_grid) @ (
            subdomain_projections.cell_restriction(production_wells_grid) @ source
        )

        return source


class ModifyFluidSourceMixin(WellFlowData):
    """ Modify the fluid source term for the pressure equations at the injection wells.
    """
    def WI(self) -> pp.ad.Operator:
        # Pick the hosting matrix grid (first 2D subdomain)
        # Reservoir properties
        prod_intf = [
            intf for intf in self.mdg.interfaces()
            if "production_well" in self.mdg.interface_to_subdomain_pair(intf)[1].tags
        ]
        subdomains = self.interfaces_to_subdomains(prod_intf)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, prod_intf)
        r_w = pp.ad.Scalar(self._well_radius)
        f_log = pp.ad.Function(pp.ad.functions.log, "log_function_Piecmann")
        r_e = PeacemanWellFlux.equivalent_well_radius(self, subdomains)
        skin_factor = pp.ad.Scalar(0.0)
        # We assume isotropic permeability and extract xx component.
        e_i = self.e_i(subdomains, i=0, dim=9).T
        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability,
            size=sum(sd.num_cells for sd in subdomains), 
            name="base_perm_for_WI"
        )
        isotropic_permeability = e_i@ self.isotropic_second_order_tensor(subdomains, base_perm)  
        # isotropic_permeability = e_i @ self.permeability(subdomains)
        well_index = self.volume_integral(
            pp.ad.Scalar(2 * np.pi) * projection.primary_to_mortar_avg()
            @ (isotropic_permeability / (f_log(r_e / r_w) + skin_factor)),
            prod_intf,
            1,
        )

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
        injection_wells_grid, _ = self.filter_wells(subdomains, "injection")
        if injection_wells_grid:
            cell_volumes = injection_wells_grid[0].cell_volumes
            src_density_rate = self._get_total_injected_mass_rate(injection_wells_grid) / pp.ad.Scalar(cell_volumes)  # kg/m3/s
            injected_mass_rate: pp.ad.Operator = self.volume_integral(
                src_density_rate, injection_wells_grid, 1
            )

            source += subdomain_projections.cell_restriction(subdomains) @ (
                subdomain_projections.cell_prolongation(injection_wells_grid) @ injected_mass_rate
            )

        # -------------------
        # Production wells
        # -------------------
        prod_wells_grid, _ = self.filter_wells(subdomains, "production")
        if prod_wells_grid:
            # Assume single producer for now

            # Well index (to be tuned / computed)
            WI = self.WI()
            lamda_f = self.total_mass_mobility(prod_wells_grid)
            # Sink term: -WI * (p_cell - p_bhp)
            p_bhp = pp.wrap_as_dense_ad_array(
                np.hstack(
                    [
                        np.ones(sd.num_cells)
                        * self.adaptive_production_pressure() # self._p_PRODUCTION[0]
                        for sd in prod_wells_grid
                    ]
                ),
                name="production_pressure",
            )
            # WI * (p_cell - p_bhp) is already a flux-like expression, not a volumetric source density.
            q_prod = -lamda_f*WI * (self.pressure(prod_wells_grid) - p_bhp)  # kg/s
            q_prod.set_name("produced_fluid_mass")

            # Add directly to source operator: (No need to compute volumes integral because it is in kg/s here)
            source += subdomain_projections.cell_restriction(subdomains) @ (
                subdomain_projections.cell_prolongation(prod_wells_grid) @ q_prod
            )

        source.set_name("fluid_source")
        return source


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

        production_wells, _ = self.filter_wells(subdomains, "production")
        _, no_injection_wells = self.filter_wells(subdomains, "injection")

        subdomain_projections = pp.ad.SubdomainProjections(no_injection_wells)
        source -= subdomain_projections.cell_prolongation(production_wells) @ (
            subdomain_projections.cell_restriction(production_wells) @ source
        )
        return source


class ModifiedPrimaryEquationsMixin:
    # Adjusting PDEs
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

        # p_constraint = self.pressure_constraint_at_production_wells(production_well_grid)
        # self.equation_system.set_equation(
        #     p_constraint,
        #     production_well_grid,
        #     {"cells": 1}
        # )

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
        pressure_constraint_production = (self.pressure(subdomains) - p_production)
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
        beta = pp.ad.Scalar(0.7)  # Penalty parameter to enforce constraint   
        enthalpy_constraint_injection = beta * (self.enthalpy(subdomains) - h_injection)
        enthalpy_constraint_injection.set_name("injection_enthalpy_constraint")
        return enthalpy_constraint_injection


class HaliteDependentApertureMixin:
    """
    Update fracture aperture as halite precipitates.

    Effective aperture:
        b_eff = b0 * (1 - alpha * S_halite)
    """
    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

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
        b_min = 9.0e-4  # m, minimum aperture to avoid zero permeability
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0
        # Clamp s_h to [0, 0.5]
        s_halite_array = clamped_halite_saturation(self, subdomains)
        # max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")
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

        effective_b = b0 * (pp.ad.Scalar(1.0) - s_halite_array)**pp.ad.Scalar(0.5)
        # effective_b = max_fn(effective_b, pp.ad.Scalar(b_min))
        effective_b.set_name("fracture_intersection_aperture")

        return effective_b


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
        # size = sum(sd.num_cells for sd in subdomains_lower)
        # one = pp.wrap_as_dense_ad_array(0.1, size=size, name="one")
        return self.porosity_matrix(subdomains)
        aperture = self.aperture(subdomains)
        cell_sizes = np.hstack([np.sqrt(sd.cell_volumes) for sd in subdomains])
        return aperture / (pp.wrap_as_dense_ad_array(cell_sizes))
    
    def porosity_wells(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Sanity check
        """Porosity specifically for OD grid"""
        # return self.porosity_matrix(subdomains)
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
            halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
            s_h = halite_phase[0].saturation(domains)

            # Clamp only once
            s_ref_clamped = min_fn(pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0)))
            s_h_clamped = clamped_halite_saturation(self, domains)

            for phase in self.fluid.phases:
                if phase.name.lower() == "halite":
                    saturation = s_h_clamped
                elif phase == self.fluid.reference_phase:
                    saturation = s_ref_clamped
                else:
                    inferred = pp.ad.Scalar(1.0) - (s_ref_clamped + s_h_clamped)
                    saturation = min_fn(pp.ad.Scalar(1.0), max_fn(inferred, pp.ad.Scalar(0.0)))

                kappa = phase.thermal_conductivity(domains)
                ops.append(saturation * kappa)

            op = pp.ad.sum_operator_list(ops, name="fluid_thermal_conductivity")
        else:
            op = self.fluid.reference_phase.thermal_conductivity(domains)
            op.set_name("fluid_thermal_conductivity")

        return op
    

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
        # Example reduction: perm_eff = perm_0 * (1 - s_halite)^2
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm*reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    
    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # return self.matrix_permeability(subdomains)
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


class LiquidPhaseFlowModelConfiguration2D(
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
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        if len(halite_phase) != 1:
            raise ValueError("Expected exactly one halite phase.")
    
        max = pp.ad.Function(pp.ad.maximum, "maximum_function")

        # name = phase.name
        s = phase.saturation(domains)

        # s_halite = halite_phase[0].saturation(domains)
        # Total mobile pore volume
        mobile_pore_volume = pp.ad.Scalar(1.0) #* (1-s_halite)

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
        
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(1e-12)  # small floor
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        if len(halite_phase) != 1:
            raise ValueError("Expected exactly one halite phase.")

        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")

        s = phase.saturation(domains)
        # s_halite = halite_phase[0].saturation(domains)
        s_halite = clamped_halite_saturation(self, domains)

        # Effective pore volume available for fluids
        mobile_pore_volume = 1.0 - s_halite

        # Residual saturations (scalable by mobile volume)
        s_l_res = pp.ad.Scalar(0.3) * mobile_pore_volume
        s_v_res = pp.ad.Scalar(0.0) * mobile_pore_volume

        # if phase.name == "halite":
        #     return max_fn(pp.ad.Scalar(1.0e-12)* s, epsilon)

        if phase == self.fluid.reference_phase:  # say liquid
            s_eff = (s - s_l_res) / (mobile_pore_volume - s_l_res - s_v_res)
            s_eff = max_fn(s_eff, epsilon)
            return s_eff ** pp.ad.Scalar(1.5)  # Corey-type curve
        else:  # vapor
            s_eff = (s - s_v_res) / (mobile_pore_volume - s_l_res - s_v_res)
            s_eff = max_fn(s_eff, epsilon)
            return s_eff ** pp.ad.Scalar(1.0)
        


