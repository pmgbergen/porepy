"""
Unified geothermal compositional flow model.
This file is the single model implementation for examples 1--3.

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
- All outer boundaries: **no-flow** (impermeable).
- Wells: injection well imposes temperature/composition of the injected fluid;
  production well is controlled by peaceman well flux.

"""

import porepy as pp
import numpy as np

from porepy.models.compositional_flow import CompositionalFlowTemplate
from porepy.models.constitutive_laws import PeacemanWellFlux
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler
from porepy.examples.geothermal_flow.model_configuration.constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem,
    PhaseMode,
)

from typing import Sequence, ClassVar
from dataclasses import dataclass


class NoFlowAdiabaticBoundary(pp.PorePyModel):
    """No-flow, adiabatic boundary conditions used by the example suite."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, cond="neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, cond="neu")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, cond="neu")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, cond="neu")


class UniformInitialConditions(pp.PorePyModel):
    """Uniform initial conditions, with injection composition at injection wells."""

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        if sd.dim == 0 and "production_well" in sd.tags:
            return np.ones(sd.num_cells) * self._p_OUT
        return np.ones(sd.num_cells) * self._p_INIT

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        if "injection_well" in sd.tags:
            z_NaCl = np.ones(sd.num_cells) * self._z_INJ["NaCl"]
        else:
            z_NaCl = self._z_INIT["NaCl"] * np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        return self.vtk_sampler_ptz.sampled_cloud.point_data["H"]

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        if "injection_well" in sd.tags:
            return np.ones(sd.num_cells) * self._T_INJ
        return np.ones(sd.num_cells) * self._T_INIT

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        if "injection_well" in sd.tags:
            return np.ones(sd.num_cells) * self._z_INJ.get(component.name, 0.0)
        z = self._z_INIT.get(component.name, 0.0)
        return np.full(sd.num_cells, z)


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


class ThreePhaseSecondaryEquation2D(SecondaryEquations):
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

    max_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

    def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
        return -max_fn(-a, -b)

    min_val = min_fn(pp.ad.Scalar(0.8), max_fn(s_h_raw, pp.ad.Scalar(0.0)))
    return min_val


# =============================================================================
# WELL FLOW DATA WITH RAMPING
# =============================================================================
class WellFlowData(pp.PorePyModel):
    """Helper class to bundle configuration of pressure, temperature, and injected mass
    for a multiphase and salt-water geothermal problem with one injector and one producer."""

    vtk_sampler_ptz: VTKSampler

    # Initial reservoir conditions (representative of ~2 km depth).
    _p_INIT: float = 10.5e6  # Pa
    _T_INIT: float = 586.651  # K
    _z_INIT: dict[str, float] = {"H2O": 0.6, "NaCl": 0.4}

    # In- and outflow values.
    _T_INJ: float = 300.651  # 300.15   # K (injection temperature)
    _z_INJ: dict[str, float] = {"H2O": 0.9999, "NaCl": 1.0e-4}

    _INJECTION_FRACTION: float = 1.0  # kg/kg
    _aperture_clogging_exponent: float = 0.1

    _p_OUT: float = 7.0e6  # Pa
    _well_radius: float = 0.1  # m

    # Initial and injected fluid composition.
    _fracture_aperture = 1.0e-3  # m (initial fracture aperture)

    # Injection schedule (can extend later with time-dependent keys)
    _T_INJECTION: dict[int, float] = {0: _T_INJ}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
    _p_INJECTION: dict[int, float] = {0: _p_INIT}

    def _get_fluid_density(
        self, temperature: float, pressure: float, z_NaCl: float
    ) -> float:
        """Sample bulk fluid density from the VTK table."""
        par_point = np.array([[z_NaCl, temperature, pressure]])
        self.vtk_sampler_ptz.sample_at(par_point)
        data = self.vtk_sampler_ptz.sampled_cloud.point_data
        rho = data["Rho"][0]
        return rho

    def _get_total_injected_mass_rate(self) -> float:
        """Calculate total injected mass (in kg/s)."""

        T_inj = self._T_INJ
        z_NaCl = self._z_INJ["NaCl"]
        rho = self._get_fluid_density(
            temperature=T_inj, pressure=self._p_INIT, z_NaCl=z_NaCl
        )

        base_rate = self._INJECTION_FRACTION * rho / 3600.0  # kg/m^3/s
        return base_rate

    def _injected_component_mass(
        self, component: pp.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Return injected mass source density for a given component [kg/m^3/s]."""

        # total injected mass rate [kg/m3/s]
        _total_injected_mass = self._get_total_injected_mass_rate()

        z_H2O = self._z_INJ["H2O"]
        z_NaCl = self._z_INJ["NaCl"]

        # total injected mass rate [kg/m^3/s]
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


# =============================================================================
# COMPONENT SOURCE MIXIN
# =============================================================================
class ModifyComponentSourceMixin:
    """
    Adjusts the component source terms for the mass balance equations.
    """

    def component_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
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
            subdomain_projections.cell_prolongation(injection_wells_grid)
            @ injected_mass
        )

        production_wells_grid, _ = self._filter_wells(subdomains, "production")
        source -= subdomain_projections.cell_prolongation(production_wells_grid) @ (
            subdomain_projections.cell_restriction(production_wells_grid) @ source
        )

        return source


# =============================================================================
# FLUID SOURCE MIXIN WITH RAMPED PRODUCTION BHP
# =============================================================================
class ModifyFluidSourceMixin(WellFlowData):
    """Modify the fluid source term for the pressure equations at the injection and production wells ."""

    def WI(self) -> pp.ad.Operator:
        # Pick the hosting matrix grid (first 2D subdomain)
        # Reservoir properties
        prod_intf = [
            intf
            for intf in self.mdg.interfaces()
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
        well_geo = (
            pp.ad.Scalar(2 * np.pi)
            * h
            * projection.primary_to_mortar_avg()
            @ (isotropic_permeability / (f_log(r_e / r_w) + skin_factor))
        )

        well_index = self.volume_integral(well_geo, prod_intf, 1)
        return well_index

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Source term in the total mass balance.
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
        # Production wells
        # -------------------
        prod_wells_grid, _ = self._filter_wells(subdomains, "production")
        WI = self.WI()
        lambda_f = self.total_mass_mobility(prod_wells_grid)

        p_bhp = self._p_OUT
        q_prod = -lambda_f * WI * (self.pressure(prod_wells_grid) - p_bhp)
        q_prod.set_name("produced_fluid_mass")

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
        Adjust energy source term.

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
# PRIMARY EQUATIONS MIXIN WITH TEMPERATURE CONSTRAINT
# =============================================================================
class ModifiedPrimaryEquationsMixin:
    # Adjusting PDEs
    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not injection wells."""

        _, no_injection_well_grids = self._filter_wells(subdomains, "injection")
        eq: pp.ad.Operator = super().energy_balance_equation(no_injection_well_grids)  # type:ignore[misc]
        return eq

    def set_equations(self):
        """Introduces temperature constraint at injection well."""
        super().set_equations()

        subdomains = self.mdg.subdomains()

        injection_well_grid, _ = self._filter_wells(subdomains, "injection")
        t_constraint = self.temperature_constraint_at_injection_wells(
            injection_well_grid
        )
        self.equation_system.set_equation(
            t_constraint, injection_well_grid, {"cells": 1}
        )

    def temperature_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        T_injection_value = self._T_INJ
        T_injection = pp.wrap_as_dense_ad_array(
            np.hstack([np.ones(sd.num_cells) * T_injection_value for sd in subdomains]),
            name="injection_temperature",
        )

        temperature_constraint_injection = self.temperature(subdomains) - T_injection
        temperature_constraint_injection.set_name("injection_temperature_constraint")
        return temperature_constraint_injection


class HaliteDependentApertureMixin:
    """
    Update fracture aperture as halite precipitates.

    Effective aperture:
        b_eff = b0 * (1 - alpha * S_halite)^aperture_clogging_exponent
    """

    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0
        well_tags = {"injection_well", "production_well"}
        fractures_and_intersections = [
            sd
            for sd in subdomains
            if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]
        wells_grids = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        for i, sd in enumerate(subdomains):
            if sd.dim == self.nd:
                a_loc = super().aperture([sd])
            elif sd in fractures_and_intersections:
                a_loc = self.fracture_or_intersection_aperture([sd])
            elif sd in wells_grids:
                a_loc = np.ones(sd.num_cells) * self._well_radius
            else:
                a_loc = super().aperture([sd])
            a_glob = projection.cell_prolongation([sd]) @ a_loc
            if i == 0:
                aperture = a_glob
            else:
                aperture += a_glob
        aperture.set_name("aperture")
        return aperture

    def fracture_or_intersection_aperture(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        b_min = getattr(
            self, "_minimum_aperture", 1.0e-4
        )  # m, minimum aperture to avoid zero permeability
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0

        s_halite_array = clamped_halite_saturation(self, subdomains)
        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")
        b0 = pp.ad.Scalar(self._fracture_aperture)
        effective_b = b0 * (pp.ad.Scalar(1.0) - s_halite_array) ** pp.ad.Scalar(
            self._aperture_clogging_exponent
        )
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
        subdomains_lower = [
            sd
            for sd in subdomains
            if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]
        subdomains_wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        phi_nd = projection.cell_prolongation(subdomains_nd) @ self.porosity_matrix(
            subdomains_nd
        )
        phi_lower = projection.cell_prolongation(
            subdomains_lower
        ) @ self.porosity_fracture_and_intersection(subdomains_lower)
        phi_wells = projection.cell_prolongation(
            subdomains_wells
        ) @ self.porosity_wells(subdomains_wells)
        phi = phi_nd + phi_lower + phi_wells
        phi.set_name("porosity")

        return phi

    def porosity_fracture_and_intersection(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Porosity specifically for fractures and intersections."""

        return self.porosity_matrix(subdomains)

    def porosity_wells(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity specifically for OD grid"""
        return self.porosity_matrix(subdomains)

    def porosity_matrix(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity in the nd-dimensional matrix [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].

        """

        phi_0 = pp.ad.Scalar(self.solid.porosity, name="porosity")
        s_h_clamped = clamped_halite_saturation(self, subdomains)

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
            s_ref_clamped = min_fn(
                pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0))
            )
            s_h_clamped = clamped_halite_saturation(self, domains)

            for phase in self.fluid.phases:
                if phase.name.lower() == "halite":
                    saturation = s_h_clamped
                elif phase == self.fluid.reference_phase:
                    saturation = s_ref_clamped
                else:
                    total_sat = min_fn(pp.ad.Scalar(1.0), s_ref_clamped + s_h_clamped)
                    inferred = pp.ad.Scalar(1.0) - total_sat
                    saturation = min_fn(
                        pp.ad.Scalar(1.0), max_fn(inferred, pp.ad.Scalar(0.0))
                    )

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
            sd
            for sd in subdomains
            if sd.dim == self.nd - 2 and not well_tags.intersection(sd.tags)
        ]
        wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        permeability = (
            projection.cell_prolongation(matrix) @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(wells) @ self.well_permeability(wells)
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
            self.solid.permeability, size, name="permeability"
        )

        s_h_clamped = clamped_halite_saturation(self, subdomains)
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm * reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)

    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.matrix_permeability(subdomains)

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
            self.solid.fracture_permeability, size, name="permeability"
        )
        return self.isotropic_second_order_tensor(subdomains, base_perm)


# =============================================================================
# FINAL MODEL CONFIGURATION - WITH VariableClampingMixin
# =============================================================================
class ThreePhaseFlowModelConfiguration2D(
    HaliteDependentApertureMixin,
    PorosityWithHaliteMixin2D,
    PermeabilityWithHaliteMixin2D,
    ThermalConductivityMixinWithClampedHalite,
    FluidMixture,
    ModifyComponentSourceMixin,
    ModifyFluidSourceMixin,
    ModifyEnergySourceMixin,
    ModifiedPrimaryEquationsMixin,
    ThreePhaseSecondaryEquation2D,
    CompositionalFlowTemplate,
    VTKSamplerMixin,
):
    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        epsilon = pp.ad.Scalar(1e-12)  # small floor

        max_fn = pp.ad.Function(pp.ad.maximum, "maximum_function")

        s = phase.saturation(domains)

        s_halite = clamped_halite_saturation(self, domains)

        # Effective pore volume available for fluids
        mobile_pore_volume = 1.0 - s_halite

        # Residual saturations (scalable by mobile volume)
        s_l_res = pp.ad.Scalar(0.3) * mobile_pore_volume
        s_v_res = pp.ad.Scalar(0.00) * mobile_pore_volume

        if phase.name == "halite":
            return pp.ad.Scalar(0.0) * s

        if phase == self.fluid.reference_phase:  # liquid
            s_eff = (s - s_l_res) / (mobile_pore_volume - s_l_res - s_v_res)
            s_eff = max_fn(s_eff, epsilon)
            return s_eff ** pp.ad.Scalar(1.5)  # Corey-type curve
        else:  # vapor
            s_eff = (s - s_v_res) / (mobile_pore_volume - s_l_res - s_v_res)
            return s_eff
