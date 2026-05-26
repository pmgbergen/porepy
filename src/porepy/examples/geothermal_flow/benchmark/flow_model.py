"""Benchmark flow model for the 1D CSMP--PorePy comparison.

This module contains only the benchmark model definition:

- horizontal 1D-like rectangular geometry,
- low-pressure three-phase boundary conditions,
- low-pressure three-phase initial conditions,
- three-phase H2O--NaCl compositional flow model.

The simulation driver, solver settings, material constants, VTK-file loading,
and plotting should live outside this file.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from ..io_utils import as_float
import porepy as pp
from porepy.models.compositional_flow import CompositionalFlowTemplate

from ..model_configuration.constitutive_description.mixture_constitutive_description import (
    ComponentSystem,
    FluidMixture,
    PhaseMode,
    SecondaryEquations,
)
from ..vtk_sampler import VTKSampler

from scipy.optimize import root_scalar


# =============================================================================
# Helper utilities
# =============================================================================


def find_z_for_target_sh(
    p_values: np.ndarray,
    T0: float,
    sh_target: float,
    sampler: "VTKSampler",
    z_bounds: Tuple[float, float] = (0.0, 0.2),
    z_size: int = 2
) -> np.ndarray:
    """
    For each pressure value, find the NaCl mass fraction z such that the halite saturation S_h equals a target value.

    Parameters:
        p_values: Array of pressure values for each grid cell.
        T0: Fixed temperature (K) used for all cells.
        sh_target: Target halite saturation value to match.
        sampler: VTKSampler
            Sampler instance that provides S_h and its gradient.
        z_bounds: Bounds within which to search for z (default: (0.0, 0.2)).

    Returns:
        np.ndarray of shape (N,)
            Array of z values where S_h ≈ sh_target, or np.nan if no root is found.
    """
    N: int = len(p_values)
    z_solutions: np.ndarray = np.full(N, np.nan)

    for i, p_i in enumerate(p_values):
        def f(z: float) -> float:
            par_point: np.ndarray = np.array([[z, T0, p_i]])
            sampler.sample_at(par_point)
            return sampler.sampled_cloud.point_data["S_h"][0] - sh_target

        try:
            z_vals = np.linspace(z_bounds[0], z_bounds[1], z_size)
            s_vals = np.array([
                f(z) for z in z_vals
            ])
            idx = np.where(np.diff(np.sign(s_vals)))[0]
            z_low = z_vals[idx[0]]
            z_high = z_vals[idx[0] + 1]
            sol = root_scalar(
                f,
                bracket=[z_low, z_high],
                method="brentq",
                xtol=1.0e-6
            )
            if sol.converged:
                z_solutions[i] = sol.root
        except ValueError:
            # No root in bracket: leave as np.nan
            continue

    return z_solutions


# =============================================================================
# VTK sampler mixin
# =============================================================================


class VTKSamplerMixin:
    """Store p-h-z and p-T-z thermodynamic VTK samplers on the model."""

    @property
    def vtk_sampler(self) -> VTKSampler:
        """Return the p-h-z VTK sampler."""
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler: VTKSampler) -> None:
        """Set the p-h-z VTK sampler."""
        self._vtk_sampler = vtk_sampler

    @property
    def vtk_sampler_ptz(self) -> VTKSampler:
        """Return the p-T-z VTK sampler."""
        return self._vtk_sampler_ptz

    @vtk_sampler_ptz.setter
    def vtk_sampler_ptz(self, vtk_sampler: VTKSampler) -> None:
        """Set the p-T-z VTK sampler."""
        self._vtk_sampler_ptz = vtk_sampler


# =============================================================================
# Geometry
# =============================================================================


class BenchmarkHorizontalGeometry(pp.PorePyModel):
    """Horizontal benchmark domain used for the CSMP comparison.

    The domain is a 2000 m by 10 m rectangle. The west boundary is the inlet
    and the east boundary is the outlet.
    """

    _dist_from_ref_point: float = 5.0
    _inlet_centre: np.ndarray = np.array([0.0, 5.0, 0.0])
    _outlet_centre: np.ndarray = np.array([2000.0, 5.0, 0.0])

    def set_domain(self) -> None:
        """Set the rectangular benchmark domain."""
        x_length = self.units.convert_units(
            self.params.get("domain", {}).get("x_length", 2000.0),
            "m",
        )
        y_length = self.units.convert_units(
            self.params.get("domain", {}).get("y_length", 10.0),
            "m",
        )

        self._domain = pp.Domain({"xmax": x_length, "ymax": y_length})

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        """Return the grid type used for the benchmark."""
        return self.params.get("domain", {}).get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict[str, float]:
        """Return meshing arguments for the benchmark grid."""
        cell_size = self.units.convert_units(
            self.params.get("domain", {}).get("cell_size", 10.0),
            "m",
        )
        return {"cell_size": cell_size}

    def get_inlet_outlet_sides(
        self,
        sd: pp.Grid | pp.BoundaryGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return west inlet and east outlet facets/cells."""
        domain_sides = self.domain_boundary_sides(sd)
        inlet_facets = np.where(domain_sides.west)[0]
        outlet_facets = np.where(domain_sides.east)[0]
        return inlet_facets, outlet_facets


# =============================================================================
# Boundary conditions
# =============================================================================


class BenchmarkThreePhaseBoundaryConditions(pp.PorePyModel):
    """Boundary conditions for the low-pressure benchmark case."""

    vtk_sampler_ptz: VTKSampler

    def _bc_config(self) -> dict:
        """Return boundary-condition configuration dictionary."""
        return self.params.get("boundary_conditions", {})

    def _pressure_config(self) -> dict:
        """Return pressure boundary-condition values."""
        return self._bc_config().get("pressure", {})

    def _temperature_config(self) -> dict:
        """Return temperature boundary-condition values."""
        return self._bc_config().get("temperature", {})

    def _composition_config(self) -> dict:
        """Return composition boundary-condition values."""
        return self._bc_config().get("z_nacl", {})

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Use Dirichlet temperature on west/east and no-flux elsewhere."""
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond="neu")
        bc.is_dir[sides.west] = True
        bc.is_dir[sides.east] = True
        bc.is_neu[sides.west] = False
        bc.is_neu[sides.east] = False
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Use Dirichlet pressure on west/east and no-flow elsewhere."""
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond="neu")
        bc.is_dir[sides.west] = True
        bc.is_dir[sides.east] = True
        bc.is_neu[sides.west] = False
        bc.is_neu[sides.east] = False
        return bc

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Use Dirichlet component composition on west/east."""
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond="neu")
        bc.is_dir[sides.west] = True
        bc.is_dir[sides.east] = True
        bc.is_neu[sides.west] = False
        bc.is_neu[sides.east] = False
        return bc

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Use Dirichlet enthalpy on west/east."""
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond="neu")
        bc.is_dir[sides.west] = True
        bc.is_dir[sides.east] = True
        bc.is_neu[sides.west] = False
        bc.is_neu[sides.east] = False
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Return west/east pressure boundary values."""
        p_inlet = as_float(self._pressure_config().get("inlet", 4.0e6))
        p_outlet = as_float(self._pressure_config().get("outlet", 1.0e6))

        sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[sides.west] = p_inlet
        values[sides.east] = p_outlet
        return values

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Return west/east temperature boundary values."""
        t_inlet = as_float(self._temperature_config().get("inlet", 573.15))
        t_outlet = as_float(self._temperature_config().get("outlet", 423.15))

        sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[sides.west] = t_inlet
        values[sides.east] = t_outlet
        return values

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Evaluate boundary enthalpy from p-T-z conditions."""
        values = np.zeros(boundary_grid.num_cells)

        sides = self.domain_boundary_sides(boundary_grid)
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        dirichlet = sides.west | sides.east

        if np.any(dirichlet):
            nacl_component = next(
                component
                for component in self.fluid.components
                if component.name == "NaCl"
            )
            z_nacl = self.bc_values_overall_fraction(nacl_component, boundary_grid)

            points = np.array((z_nacl[dirichlet], t[dirichlet], p[dirichlet])).T
            self.vtk_sampler_ptz.sample_at(points)
            values[dirichlet] = self.vtk_sampler_ptz.sampled_cloud.point_data["H"]

        return values

    def bc_values_overall_fraction(
        self,
        component: pp.Component,
        boundary_grid: pp.BoundaryGrid,
    ) -> np.ndarray:
        """Return H2O or NaCl mass fraction on the west/east boundaries."""
        sides = self.domain_boundary_sides(boundary_grid)
        pressure = self.bc_values_pressure(boundary_grid)
        temperature = self.bc_values_temperature(boundary_grid)

        z_inlet = as_float(self._composition_config().get("inlet", 0.0))
        sh_target = as_float(self._composition_config().get(
            "outlet_target_halite_saturation",
            0.1,
        ))
        z_bounds = tuple(self._composition_config().get("search_bounds", [0.3, 0.42]))

        z_outlet_values = find_z_for_target_sh(
            T0=float(temperature[sides.east][0]),
            sh_target=float(sh_target),
            p_values=pressure[sides.east],
            sampler=self.vtk_sampler_ptz,
            z_bounds=z_bounds,
        )
        z_outlet = float(z_outlet_values[0])

        if component.name == "NaCl":
            values = np.zeros(boundary_grid.num_cells)
            values[sides.west] = z_inlet
            values[sides.east] = z_outlet
            return values

        if component.name == "H2O":
            values = np.zeros(boundary_grid.num_cells)
            values[sides.west] = 1.0 - z_inlet
            values[sides.east] = 1.0 - z_outlet
            return values

        raise ValueError(f"Unsupported component: {component.name}")


# =============================================================================
# Initial conditions
# =============================================================================


class BenchmarkThreePhaseInitialConditions(pp.PorePyModel):
    """Initial conditions for the low-pressure benchmark case."""

    vtk_sampler_ptz: VTKSampler

    def _ic_config(self) -> dict:
        """Return initial-condition configuration dictionary."""
        return self.params.get("initial_conditions", {})

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Return a linear pressure profile from inlet to outlet."""
        pressure_cfg = self._ic_config().get("pressure", {})
        p_inlet = as_float(pressure_cfg.get("inlet", 4.0e6))
        p_outlet = as_float(pressure_cfg.get("outlet", 1.0e6))

        domain_length = self.params.get("domain", {}).get("x_length", 2000.0)
        x = sd.cell_centers[0]
        pressure_gradient = (p_outlet - p_inlet) / domain_length

        return p_inlet + pressure_gradient * x

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        """Return uniform initial temperature."""
        temperature = as_float(self._ic_config().get("temperature", 423.15))
        return np.ones(sd.num_cells) * temperature

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate initial enthalpy from p-T-z conditions."""
        pressure = self.ic_values_pressure(sd)
        temperature = self.ic_values_temperature(sd)

        nacl_component = next(
            component for component in self.fluid.components if component.name == "NaCl"
        )
        z_nacl = self.ic_values_overall_fraction(nacl_component, sd)

        points = np.array((z_nacl, temperature, pressure)).T
        self.vtk_sampler_ptz.sample_at(points)

        return self.vtk_sampler_ptz.sampled_cloud.point_data["H"]

    def ic_values_overall_fraction(
        self,
        component: pp.Component,
        sd: pp.Grid,
    ) -> np.ndarray:
        """Return initial H2O or NaCl mass fraction."""
        if component.name == "NaCl":
            pressure = self.ic_values_pressure(sd)
            temperature = self.ic_values_temperature(sd)

            composition_cfg = self._ic_config().get("z_nacl", {})
            sh_target = as_float(composition_cfg.get("target_halite_saturation", 0.1))
            z_bounds = tuple(as_float(x) for x in composition_cfg.get("search_bounds", [0.3, 0.42]))

            return find_z_for_target_sh(
                T0=float(temperature[0]),
                sh_target=float(sh_target),
                p_values=pressure,
                sampler=self.vtk_sampler_ptz,
                z_bounds=z_bounds,
            )

        if component.name == "H2O":
            nacl_component = next(
                c for c in self.fluid.components if c.name == "NaCl"
            )
            z_nacl = self.ic_values_overall_fraction(nacl_component, sd)
            return 1.0 - z_nacl

        raise ValueError(f"Unsupported component: {component.name}")


# =============================================================================
# Secondary equations and flow model
# =============================================================================


class BenchmarkThreePhaseSecondaryEquations(SecondaryEquations):
    """Three-phase H2O--NaCl secondary-equation configuration."""

    component_system = ComponentSystem.WATER_SALT
    phase_mode = PhaseMode.THREE_PHASE


class BenchmarkPorosityWithHaliteMixin(pp.PorePyModel):
    """Reduce effective porosity according to halite saturation."""

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return porosity reduced by clamped halite saturation."""
        phi_0 = pp.ad.Scalar(self.solid.porosity, name="porosity")

        halite_phase = [phase for phase in self.fluid.phases if phase.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required.")

        s_halite_raw = halite_phase[0].saturation(subdomains)
        maximum = pp.ad.Function(pp.ad.maximum, "maximum_function")

        def minimum(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum(-a, -b)

        s_halite_clamped = minimum(
            pp.ad.Scalar(0.5),
            maximum(s_halite_raw, pp.ad.Scalar(0.0)),
        )

        return phi_0 * (1.0 - s_halite_clamped)


class BenchmarkPermeabilityWithHaliteMixin(pp.PorePyModel):
    """Reduce permeability according to halite saturation."""

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return permeability tensor reduced by clamped halite saturation."""
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability,
            size,
            name="permeability",
        )

        halite_phase = [phase for phase in self.fluid.phases if phase.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required.")

        s_halite_raw = halite_phase[0].saturation(subdomains)
        maximum = pp.ad.Function(pp.ad.maximum, "maximum_function")

        def minimum(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum(-a, -b)

        s_halite_clamped = minimum(
            pp.ad.Scalar(0.5),
            maximum(s_halite_raw, pp.ad.Scalar(0.0)),
        )

        reduction = (1.0 - s_halite_clamped) ** 2
        corrected_perm = base_perm * reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)


class SolverStatisticsMixin(pp.PorePyModel):
    """Mixin to print nonlinear solver statistics after each converged solve."""
    def after_nonlinear_convergence(self) -> None:
        """Print benchmark progress after each converged nonlinear solve."""
        super().after_nonlinear_convergence()

        if not self.params.get("print_nonlinear_statistics", True):
            return

        print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
        print(f"Time value (years): {self.time_manager.time / (365.0 * pp.DAY):.4f}")
        print(f"Time value (days): {self.time_manager.time / pp.DAY:.4f}")
        print(f"Time index: {self.time_manager.time_index}\n")

    def after_simulation(self) -> None:
        """Write benchmark PVD output after the simulation."""
        super().after_simulation()
        self.exporter.write_pvd()
        print("Benchmark PVD output written.")


class BenchmarkThreePhaseFlowModel(
    SolverStatisticsMixin,
    BenchmarkPorosityWithHaliteMixin,
    BenchmarkPermeabilityWithHaliteMixin,
    BenchmarkHorizontalGeometry,
    BenchmarkThreePhaseBoundaryConditions,
    BenchmarkThreePhaseInitialConditions,
    FluidMixture,
    BenchmarkThreePhaseSecondaryEquations,
    CompositionalFlowTemplate,
    VTKSamplerMixin,
):
    """Complete benchmark model for the 1D CSMP--PorePy comparison."""

    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> pp.ad.Operator:
        """Return phase relative permeability.

        The halite phase is immobile. The reference liquid phase uses residual
        liquid saturation r_l = 0.3, while the vapor phase uses r_v = 0.
        """
        epsilon = pp.ad.Scalar(0.0)

        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Expected exactly one halite phase.")

        maximum = pp.ad.Function(pp.ad.maximum, "maximum_function")
        saturation = phase.saturation(domains)

        residual_liquid = pp.ad.Scalar(
            self.params.get("relative_permeability", {}).get(
                "residual_liquid_saturation",
                0.3,
            )
        )
        residual_vapor = pp.ad.Scalar(
            self.params.get("relative_permeability", {}).get(
                "residual_vapor_saturation",
                0.0,
            )
        )

        if phase.name == "halite":
            return pp.ad.Scalar(0.0) * saturation

        if phase == self.fluid.reference_phase:
            s_eff = (saturation - residual_liquid) / (
                1.0 - residual_liquid - residual_vapor
            )
            return maximum(s_eff, epsilon)

        return (saturation - residual_vapor) / (
            1.0 - residual_liquid - residual_vapor
        )