from typing import Callable, Literal, Union, cast, Optional, Any

import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc

from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
)
from .flow_model_base import FlowModelBase, FractionalFlowModelBase

from ..obl_sampler import VTKSampler
from .constitutive_description.BrineConstitutiveDescription import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry


class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler
    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = boundary_grid.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p = np.array(list(map(p_linear, xc)))
        return p

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 623.15
        t_outlet = 423.15
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p = self.bc_values_pressure(boundary_grid)
        t = self.bc_values_temperature(boundary_grid)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        h = self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.0
        z_inlet = 0.0
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl


class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    obl_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 50.0
        p_outlet = 25.0
        xc = sd.cell_centers.T
        p_linear = lambda x: (x[0] * p_outlet + (2000.0 - x[0]) * p_inlet) / 2000.0
        p_init = np.array(list(map(p_linear, xc)))
        return p_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        t_init = 423.15
        return np.ones(sd.num_cells) * t_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = np.zeros_like(p)
        par_points = np.array((z_NaCl, t, p)).T
        self.obl_sampler_ptz.sample_at(par_points)
        h_init = self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        return h_init

    def ic_values_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.0
        return z * np.ones(sd.num_cells)


class _DriesnerBrineBase(  # type:ignore[misc]
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
):

    def _liquid_relative_permeability(self, s_liq: pp.ad.Operator) -> pp.ad.Operator:
        """Weis (2014) liquid relative permeability with residual liquid saturation S_rl = 0.3:
        k_rl = max((s_l - 0.3)/0.7, 0), written as the smooth ReLU 0.5*(sqrt(x^2) + x)."""
        sr = pp.ad.Scalar(0.3)
        s_red = (s_liq - sr) / (pp.ad.Scalar(1.0) - sr)
        return pp.ad.Scalar(0.5) * ((s_red**2) ** 0.5 + s_red)

    def relative_permeability(
        self, phase: pp.ad.Operator, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        # Weis (2014) fig-5 pure-water two-phase (liq/gas) relative permeabilities:
        #   k_rl = max((s_l - 0.3)/0.7, 0),   k_rv = 1 - k_rl   (residual vapor R_v = 0, so
        #   k_rl + k_rv = 1). The previous vapor branch returned s_gas, which is NOT the Weis
        #   closure and delayed/widened the vapor front.
        if phase.name == "liq":
            return self._liquid_relative_permeability(phase.saturation(domains))
        # Vapor: complement of the liquid curve at s_l = 1 - s_gas (exact two-phase closure).
        s_liq = pp.ad.Scalar(1.0) - phase.saturation(domains)
        return pp.ad.Scalar(1.0) - self._liquid_relative_permeability(s_liq)

    @property
    def obl_sampler(self):
        return self._obl_sampler

    @obl_sampler.setter
    def obl_sampler(self, obl_sampler):
        self._obl_sampler = obl_sampler

    @property
    def obl_sampler_ptz(self):
        return self._obl_sampler_ptz

    @obl_sampler_ptz.setter
    def obl_sampler_ptz(self, obl_sampler):
        self._obl_sampler_ptz = obl_sampler

    # def after_simulation(self):
    #     self.exporter.write_pvd()

    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.refresh_buoyancy_direction()


    def check_convergence(
        self,
        model: pp.PorePyModel,
        nonlinear_increment: np.ndarray,
    ) -> tuple[
        ConvergenceStatusCollection,
        ConvergenceStatusCollection,
        ConvergenceInfoCollection,
    ]:
        """Check convergence and divergence based on passed criteria.

        Parameters:
            model: The model instance specifying the problem to be solved, knowing
                of its metrics for measuring states and residuals.
            nonlinear_increment: Newly obtained solution increment vector.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceStatusCollection,
            ConvergenceInfoCollection]: Status and
                info about convergence and divergence.

        """
        # Fetch the residual and current iterate.
        # Update derived quantities
        model.update_derived_quantities()
        # Update buoyancy-driven fluxes (skipped when the direction is lagged/frozen)
        model.refresh_buoyancy_direction()
        # Rediscretize
        model.rediscretize()

        residual = model.equation_system.assemble(evaluate_jacobian=False)
        iterate = model.equation_system.get_variable_values(iterate_index=0)

        # Check convergence status based on current iteration.
        convergence_status, convergence_info = self.convergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
        )

        # Check divergence status based on current iteration.
        divergence_status = self.divergence_criteria.check(
            increment=nonlinear_increment,
            reference_increment=iterate,
            residual=residual,
            reference_residual=residual,
            num_iterations=self.iteration_index,
        )

        return convergence_status, divergence_status, convergence_info

    def get_equation_indices_by_category(self):
        """
        Get equation indices organized by category: differential and algebraic.

        Returns:
            tuple: (diff_eq_indices, alg_eq_indices)
                - diff_eq_indices: dict of differential equation indices
                - alg_eq_indices: dict of algebraic equation indices
        """
        eq_indices = self.equation_system.assembled_equation_indices

        # Find equation names dynamically
        try:
            temp_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_temperature"))
            s_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_s_gas"))
            x_nacl_gas_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_gas"))
            x_nacl_liq_elim_name = next(name for name in eq_indices if name.startswith("elimination_of_x_NaCl_liq"))
        except StopIteration as e:
            raise KeyError(f"A required elimination equation was not found in the equation system. {e}")

        # Differential equations (conservation laws)
        diff_eq_indices = {
            'pressure': eq_indices['mass_balance_equation'],
            'composition_NaCl': eq_indices['component_mass_balance_equation_NaCl'],
            'enthalpy': eq_indices['energy_balance_equation'],
        }

        # Algebraic equations (elimination/closure relations)
        alg_eq_indices = {
            'temperature': eq_indices[temp_elim_name],
            'saturation': eq_indices[s_gas_elim_name],
            'mass_fraction_NaCl_gas': eq_indices[x_nacl_gas_elim_name],
            'mass_fraction_NaCl_liquid': eq_indices[x_nacl_liq_elim_name],
        }

        return diff_eq_indices, alg_eq_indices

    def compute_residuals_by_category(self, residual: np.ndarray) -> tuple[dict, dict, float, float, dict]:
        """
        Compute residual norms organized by category.

        Parameters:
            residual: Full residual vector

        Returns:
            tuple: (diff_residuals, alg_residuals, diff_norm, alg_norm, alg_exceeds)
                - diff_residuals: dict of individual differential equation norms
                - alg_residuals: dict of individual algebraic equation norms
                - diff_norm: combined differential equations norm
                - alg_norm: combined algebraic equations norm
                - alg_exceeds: dict mapping algebraic variable name to an array of
                  cell indices where the algebraic residual magnitude exceeds the
                  combined differential residual magnitude for the same cell.
        """
        diff_eq_indices, alg_eq_indices = self.get_equation_indices_by_category()

        # Compute individual differential equation norms
        diff_residuals = {}
        diff_components_arrays = {}
        for name, indices in diff_eq_indices.items():
            arr = residual[indices]
            diff_components_arrays[name] = np.asarray(arr).ravel()
            diff_residuals[name] = np.linalg.norm(diff_components_arrays[name])

        # Compute individual algebraic equation norms
        alg_residuals = {}
        alg_components_arrays = {}
        for name, indices in alg_eq_indices.items():
            arr = residual[indices]
            alg_components_arrays[name] = np.asarray(arr).ravel()
            alg_residuals[name] = np.linalg.norm(alg_components_arrays[name])

        # Compute combined norms by category
        differential_components = [diff_components_arrays[name] for name in diff_eq_indices.keys()]
        algebraic_components = [alg_components_arrays[name] for name in alg_eq_indices.keys()]

        # Combined global norms
        try:
            differential_norm = np.linalg.norm(np.concatenate(differential_components))
        except Exception:
            differential_norm = float(np.linalg.norm(np.hstack(list(diff_components_arrays.values()))))
        try:
            algebraic_norm = np.linalg.norm(np.concatenate(algebraic_components))
        except Exception:
            algebraic_norm = float(np.linalg.norm(np.hstack(list(alg_components_arrays.values()))))

        # Now compute per-cell exceedance: for each algebraic variable, find cell
        # indices where |alg_residual(cell)| > combined_diff_cell_norm(cell).
        # This requires that the per-variable residual arrays have the same length
        # (number of cells). If lengths don't match, we fall back to empty masks
        # and print a diagnostic.
        alg_exceeds: dict = {}
        # Determine if we can compute per-cell combined differential norm
        diff_lengths = [arr.size for arr in differential_components]
        alg_lengths = {name: arr.size for name, arr in alg_components_arrays.items()}

        if len(set(diff_lengths)) == 1:
            # All differential components have same length -> can compute per-cell vector
            n_cells = diff_lengths[0]
            # build per-cell combined differential norm (Euclidean over components)
            diff_stack = np.vstack([diff_components_arrays[name] for name in diff_eq_indices.keys()])
            # shape (n_components, n_cells) -> compute norm over axis=0
            combined_diff_per_cell = np.linalg.norm(diff_stack, axis=0)

            # For each algebraic variable, compare if lengths match and compute mask
            for name, arr in alg_components_arrays.items():
                if arr.size == n_cells:
                    mask = np.abs(arr) > combined_diff_per_cell
                    alg_exceeds[name] = np.nonzero(mask)[0]
                else:
                    # Length mismatch; cannot compute per-cell comparison reliably
                    alg_exceeds[name] = np.array([], dtype=int)
                    print(
                        f"Warning: cannot compare per-cell residuals for '{name}' (size={arr.size}) to differential components (n_cells={n_cells})."
                    )
        else:
            # Differential components have different sizes -- cannot build per-cell comparison
            for name in alg_components_arrays.keys():
                alg_exceeds[name] = np.array([], dtype=int)
            print(
                "Warning: differential residual components have inconsistent lengths; skipping per-cell exceedance check."
            )

        return diff_residuals, alg_residuals, differential_norm, algebraic_norm, alg_exceeds

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        """
        Compute residual norm for convergence check.

        NOTE: Only differential equations are checked for convergence since
        algebraic equations (temperature, saturation, phase fractions) can
        always be reconstructed from the differential variables.

        Parameters:
            residual: Current residual vector
            reference_residual: Reference residual (not used currently)

        Returns:
            Residual norm based on differential equations only
        """
        if residual is None:
            return np.nan

        # Use unified method to compute residuals by category
        diff_residuals, alg_residuals, differential_norm, algebraic_norm, _ = \
            self.compute_residuals_by_category(residual)

        # Return only differential equations norm for convergence check
        # Algebraic equations can be reconstructed and don't need to be converged
        return differential_norm

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system, report residuals by category, and apply brine-specific
        post-processing. The actual step control (None / LS / TR / TR-LS) is delegated to
        :meth:`FlowModelBase.solve_linear_system`.

        Returns:
            np.ndarray: The solution vector (nonlinear increment).
        """
        start_time = time.time()

        _, residual_vector = self.linear_system

        diff_residuals, alg_residuals, differential_residual_norm, algebraic_residual_norm, alg_exceeds = \
            self.compute_residuals_by_category(residual_vector)

        print("\n Report Residuals ")
        print(f"Overall residual norm: {np.linalg.norm(residual_vector):.4e}")
        print("Residual norms for differential equations:")
        for name, norm in diff_residuals.items():
            print(f"  - {name.capitalize()}: {norm:.4e}")
        print("Residual norms for algebraic equations:")
        for name, norm in alg_residuals.items():
            print(f"  - {name.capitalize()}: {norm:.4e}")
        print("\nAlgebraic residuals exceeding combined differential residual per cell:")
        for name, idxs in alg_exceeds.items():
            try:
                count = int(idxs.size)
            except Exception:
                count = len(idxs)
            sample = idxs[:10].tolist() if count > 0 else []
            print(f"  - {name}: {count} cells (examples: {sample})")

        # Step control (None / LS / TR / TR-LS) handled by FlowModelBase.
        solution = super().solve_linear_system()

        end_time = time.time()
        print(f"Elapsed time for linear solve: {end_time - start_time:.4f} seconds\n")

        # Post-processing solution overshoots.
        self.postprocessing_overshoots(solution)

        # Conditional thermal overshoot post-processing: apply when the differential
        # residual is smaller than the algebraic residual.
        if differential_residual_norm < algebraic_residual_norm:
            print("\nThermal overshoot condition triggered:")
            print(f"  Differential norm ({differential_residual_norm:.4e}) < "
                  f"Algebraic norm ({algebraic_residual_norm:.4e})")
            self.postprocessing_thermal_overshoots(solution, alg_exceeds)
        else:
            print("\nThermal overshoot condition NOT triggered:")
            print(f"  Differential norm ({differential_residual_norm:.4e}) >= "
                  f"Algebraic norm ({algebraic_residual_norm:.4e})")

        return solution




    def postprocessing_overshoots(self, delta_x):

        # Define the lambda expression
        inside_ratio = lambda arr, min_v, max_v: 1.0 - np.mean((arr < min_v) | (arr > max_v))


        _, _, tmin, tmax, _, _ = self.obl_sampler_ptz.search_space.bounds
        tmin -= self.obl_sampler_ptz.translation_factors[1]
        tmax -= self.obl_sampler_ptz.translation_factors[1]

        zmin, zmax, hmin, hmax, pmin, pmax = self.obl_sampler.search_space.bounds
        z_scale, h_scale, p_scale = self.obl_sampler.conversion_factors
        zmin /= z_scale
        zmax /= z_scale
        hmin /= h_scale
        hmax /= h_scale
        pmin /= p_scale
        pmax /= p_scale

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        p_scale = inside_ratio(new_p, pmin, pmax)
        new_p = np.clip(new_p, pmin, pmax)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        z_scale = inside_ratio(new_z, 0.0, zmax)
        new_z = np.clip(new_z, 0.0, zmax)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        h_scale = inside_ratio(new_h, hmin, hmax)
        new_h = np.clip(new_h, hmin, hmax)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        t_scale = inside_ratio(new_t, tmin, tmax)
        new_t = np.clip(new_t, tmin, tmax)
        delta_x[t_dof_idx] = new_t - t_0

        # secondary fractions
        for dof_idx in [s_dof_idx, xs_v_dof_idx, xs_l_dof_idx]:
            new_q = delta_x[dof_idx] + x0[dof_idx]
            new_q = np.clip(new_q, 0.0, 1.0)
            delta_x[dof_idx] = new_q - x0[dof_idx]

        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return np.min([p_scale,z_scale,h_scale,t_scale])

    def postprocessing_thermal_overshoots(self, delta_x, alg_exceeds: dict):

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]
        s_0 = x0[s_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        delta_x[t_dof_idx] = new_t - t_0

        # saturation
        new_s = delta_x[s_dof_idx] + s_0
        idx_mp = np.where(np.abs(new_s * (1 - new_s)) > 0.0)[0]
        idx_sp = np.where(np.isclose(np.abs(new_s * (1 - new_s)), 0.0))[0]

        # if idx_mp.size != 0:
        #     # correct saturation and temperature from enthalpy
        #     par_points = np.array((z_0[idx_mp], h_0[idx_mp], p_0[idx_mp])).T
        #     self.obl_sampler.sample_at(par_points)
        #
        #     star_t = self.obl_sampler.sampled_could.point_data["Temperature"]
        #     delta_x[t_dof_idx[idx_mp]] = star_t - t_0[idx_mp]
        #     star_s = self.obl_sampler.sampled_could.point_data["S_v"]
        #     delta_x[s_dof_idx[idx_mp]] = star_s - s_0[idx_mp]

        if len(alg_exceeds) != 0:
            # correct from enthalpy using  temperature idxs
            idx_temp = alg_exceeds.get('temperature', np.array([], dtype=int))
            if len(idx_temp) != 0:
                par_points = np.array((new_z[idx_temp], new_h[idx_temp], new_p[idx_temp])).T
                self.obl_sampler.sample_at(par_points)
                star_t = self.obl_sampler.sampled_could.point_data["Temperature"]
                delta_x[t_dof_idx[idx_temp]] = star_t - t_0[idx_temp]
                star_s = self.obl_sampler.sampled_could.point_data["S_v"]
                delta_x[s_dof_idx[idx_temp]] = star_s - s_0[idx_temp]

            idx_sat = alg_exceeds.get('saturation', np.array([], dtype=int))
            if len(idx_sat) !=0:
                par_points = np.array((new_z[idx_sat], new_h[idx_sat], new_p[idx_sat])).T
                self.obl_sampler.sample_at(par_points)
                star_s = self.obl_sampler.sampled_could.point_data["S_v"]
                delta_x[s_dof_idx[idx_sat]] = star_s - s_0[idx_sat]

        # if idx_sp.size != 0:
        #     # correct enthalpy from temperature
        #     par_points = np.array((new_z[idx_sp], new_t[idx_sp], new_p[idx_sp])).T
        #     self.obl_sampler_ptz.sample_at(par_points)
        #     star_h = self.obl_sampler_ptz.sampled_could.point_data["H"] * 1.0e-3
        #     delta_x[h_dof_idx[idx_sp]] = star_h - h_0[idx_sp]
        #     star_s = self.obl_sampler_ptz.sampled_could.point_data["S_v"]
        #     delta_x[s_dof_idx[idx_sp]] = star_s - s_0[idx_sp]


        te = time.time()
        print("Elapsed time for postprocessing thermal overshoots: ", te - tb)
        return


class DriesnerBrineFlowModel(_DriesnerBrineBase, FlowModelBase):  # type: ignore[misc]
    """Driesner brine model with the STANDARD primary equations -- HU (upwinded total mobility).
    Inherits FlowModelBase (= core + CompositionalFlowTemplate) so the core's solver overrides
    keep MRO precedence over the template's SolutionStrategy. Public name kept for backward compat."""


class DriesnerBrineFractionalFlowModel(_DriesnerBrineBase, FractionalFlowModelBase):  # type: ignore[misc]
    """Driesner brine model with the FRACTIONAL-FLOW primary equations -- HU-mw (mobility-weighted);
    inherits FractionalFlowModelBase (= core + CompositionalFractionalFlowTemplate). weighted_perm=True."""

