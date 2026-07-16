from __future__ import annotations
import logging
import time
import csv
import os
import json
import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, cast, Any

import porepy as pp
from porepy.models.compositional_flow import (
    CompositionalFlowTemplate,
    CompositionalFractionalFlowTemplate,
)
from .transport_predictor import ReorderedTransportPredictor

# PETSc imports (only if available)
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    logging.warning("*** ITERATIVE SOLVER NOT AVAILABLE ***")
    logging.warning("PETSc not available. All linear systems will use direct solver (MUMPS/UMFPACK).")
    logging.warning("For large systems, consider installing PETSc for iterative solver options.")

# Configure logging to show info messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure specific loggers are enabled for linear solver information
logging.getLogger('porepy.models.solution_strategy').setLevel(logging.INFO)
logging.getLogger('porepy').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

to_Mega = 1.0e-6

class _CachingSurrogateFactory(pp.ad.SurrogateFactory):
    """A ``SurrogateFactory`` that memoizes ``__call__`` per domain set.

    A phase property -- notably ``phase.density`` -- is referenced by many equation builders
    (the accumulation ``Sum_j s_j rho_j``, the mobilities, the fractional-flow density, the
    buoyancy), and each ``phase.density(domains)`` call otherwise mints a FRESH
    ``SurrogateOperator``. Because the AD parser keys on object identity, those structurally
    identical duplicates are each re-evaluated -- a data fetch plus a sparse Jacobian assembly
    over all cells -- on every assembly. Returning one shared operator per ``domains`` collapses
    them to a single evaluation. Bit-exact: same operator, same data.
    """

    def __call__(self, domains):
        key = tuple(id(g) for g in domains)
        cache = self.__dict__.setdefault("_call_cache", {})
        op = cache.get(key)
        if op is None:
            op = super().__call__(domains)
            cache[key] = op
        return op


@dataclass
class NonlinearRunStats:
    """Picklable summary of a simulation's nonlinear-solver behaviour.

    Collected by :class:`_FlowModelBaseCore` over the course of a run and returned by
    :meth:`_FlowModelBaseCore.collect_run_stats`. It holds only plain Python types (ints, a float
    property and a list), so it pickles cleanly and carries no reference to the model or to PorePy
    internals -- unlike ``pp``'s own ``NonlinearSolverStatistics``, which embeds dict-subclass
    convergence histories and whose JSON persistence is currently broken."""

    n_accepted_steps: int = 0
    """Number of accepted time steps (each contributes one entry to ``iterations_per_step``)."""
    n_time_step_cuts: int = 0
    """Failed Newton loops -- how many times a step was rejected and the time step cut."""
    total_newton_iterations: int = 0
    """Newton iterations summed over the accepted steps (failed attempts are not counted)."""
    max_newton_iterations: int = 0
    """Newton iterations of the worst accepted step (0 if there are none)."""
    iterations_per_step: list[int] = field(default_factory=list)
    """Per-accepted-step Newton-iteration counts, in solve order."""

    @property
    def avg_newton_iterations(self) -> float:
        """Mean Newton iterations per accepted step (0.0 if no steps were accepted)."""
        return self.total_newton_iterations / self.n_accepted_steps if self.n_accepted_steps else 0.0

    def as_text(self) -> str:
        """Render a self-documenting, human-readable summary (used for ``.txt`` dumps)."""
        cut = " => dt WAS cut" if self.n_time_step_cuts else "; no dt-cuts"
        lines = [
            "# nonlinear-solver run statistics",
            f"# accepted steps: {self.n_accepted_steps} "
            f"(rejected/cut loops: {self.n_time_step_cuts}{cut})  "
            f"total Newton iterations (accepted): {self.total_newton_iterations}",
            f"accepted_steps    {self.n_accepted_steps}",
            f"time_step_cuts    {self.n_time_step_cuts}",
            f"total_newton_it   {self.total_newton_iterations}",
            f"avg_newton_it     {self.avg_newton_iterations:.3f}",
            f"max_newton_it     {self.max_newton_iterations}",
            "# step_index  newton_iterations",
        ]
        lines += [f"{i} {it}" for i, it in enumerate(self.iterations_per_step)]
        return "\n".join(lines) + "\n"


@dataclass
class DofSummary:
    """Picklable summary of the model's degrees of freedom.

    Holds the cells per subdomain dimension and, per variable, its total dof count and whether it
    is a PRIMARY unknown or a locally-eliminated SECONDARY (algebraic) variable.  Built by
    :meth:`_FlowModelBaseCore.dof_summary` and printed at the start and end of a run.  Only plain
    Python types, so it pickles cleanly and carries no reference to the model."""

    n_dofs: int = 0
    n_subdomains: int = 0
    n_interfaces: int = 0
    cells_per_dim: dict[int, tuple[int, int]] = field(default_factory=dict)
    """``dimension -> (number of subdomains, total number of cells)``."""
    variables: list[tuple[str, int, str]] = field(default_factory=list)
    """``(variable name, total ndof over all grids, 'primary' | 'secondary')`` per variable."""

    @property
    def n_primary_dofs(self) -> int:
        """Total dof over the primary (non-eliminated) variables."""
        return sum(nd for _, nd, kind in self.variables if kind == "primary")

    @property
    def n_secondary_dofs(self) -> int:
        """Total dof over the locally-eliminated (secondary/algebraic) variables."""
        return sum(nd for _, nd, kind in self.variables if kind == "secondary")

    def as_text(self) -> str:
        """Render a self-documenting, human-readable summary (used for logging / ``.txt`` dumps)."""
        lines = [
            "# degrees-of-freedom summary",
            f"total DoF: {self.n_dofs}   "
            f"(subdomains: {self.n_subdomains}, interfaces: {self.n_interfaces})",
            "# cells per subdomain, by dimension:",
            "  dim   n_subdomains      n_cells",
        ]
        for d in sorted(self.cells_per_dim, reverse=True):
            n_sub, n_cell = self.cells_per_dim[d]
            lines.append(f"  {d}D    {n_sub:>10}   {n_cell:>10}")
        lines += ["# variables:", f"  {'name':<26} {'ndof':>10}   type"]
        for name, ndof, kind in self.variables:
            lines.append(f"  {name:<26} {ndof:>10}   {kind}")
        lines.append(
            f"# primary dof: {self.n_primary_dofs}   "
            f"secondary (eliminated) dof: {self.n_secondary_dofs}")
        return "\n".join(lines) + "\n"


class _FlowModelBaseCore(ReorderedTransportPredictor):
    """Template-agnostic core of the flow model (all solver/discretisation logic). It is combined
    with one of the two compositional-flow templates below to form a concrete base; its ``super()``
    calls resolve to whichever template is mixed in after it in the concrete class's MRO."""

    # Trust-region state, persistent across nonlinear iterations (reset each time step).
    _trust_radius: float = None

    def __init__(self, params):
        super().__init__(params)
        self.newton_iterations_per_timestep = []
        self.total_newton_iterations = 0
        # Rejected nonlinear loops (time-step cuts); incremented in after_nonlinear_failure.
        self.n_time_step_cuts = 0
        # Flag to use PETSc with MUMPS solver
        self.use_petsc = params.get("use_petsc", False)

        # Linear solver selection for the PETSc path.  Only two options are supported:
        #   "cpr" -- Schur-reduced CPR (iterative; default), and
        #   "lu"  -- direct LU via MUMPS.
        self.petsc_preconditioner = params.get("petsc_preconditioner", "cpr")
        valid_preconditioners = {"lu", "cpr"}
        if self.petsc_preconditioner not in valid_preconditioners:
            logger.warning(f"Invalid linear solver '{self.petsc_preconditioner}'. Using 'cpr' as default.")
            self.petsc_preconditioner = "cpr"

        # Flag to enable Cuthill-McKee permutation for bandwidth reduction
        self.use_cuthill_mckee = params.get("use_cuthill_mckee", True)

        # Check if PETSc is available when requested
        if self.use_petsc and not PETSC_AVAILABLE:
            logger.warning("*** SOLVER CONFIGURATION MISMATCH ***")
            logger.warning("PETSc iterative solver was requested (use_petsc=True) but PETSc is not available.")
            logger.warning("All linear systems will use the default direct solver instead.")
            logger.warning("To use iterative solvers, install PETSc with: pip install petsc petsc4py")
            self.use_petsc = False

    # --- AD-graph dedup: cache the operator-builders SHARED across the mass / component /
    #     energy equations. The AD parser keys on object identity (id(op)), so each equation's
    #     ``advective_flux`` rebuilding these as fresh (structurally identical) objects makes
    #     the parser re-evaluate the same subtree. ``@cached_method`` returns ONE shared object
    #     per (args) -> evaluated once. Bit-exact (same operator, same math); model-level
    #     overrides so porepy core stays untouched.
    #     ``darcy_flux`` is the big one: ``advective_flux`` (constitutive_laws.py) calls it for
    #     the mass flux, each component flux and the energy flux -> ~4 duplicate copies.
    @pp.ad.cached_method
    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        return super().darcy_flux(domains)

    @pp.ad.cached_method
    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        return super().fluid_flux(domains)

    @pp.ad.cached_method
    def advection_weight_energy_balance(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return super().advection_weight_energy_balance(domains)

    @pp.ad.cached_method
    def advection_weight_component_mass_balance(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return super().advection_weight_component_mass_balance(component, domains)

    def density_of_phase(self, phase: pp.Phase) -> pp.ad.SurrogateFactory:
        """Make the phase-density surrogate factory memoize its calls (retag to
        :class:`_CachingSurrogateFactory`). ``phase.density`` is the only phase property
        referenced many times per assembly (the diagnostic shows 12x/phase: accumulation,
        mobilities, fractional-flow density, buoyancy); sharing one operator node per domain
        set collapses those to a single surrogate evaluation. Bit-exact."""
        factory = super().density_of_phase(phase)
        if isinstance(factory, pp.ad.SurrogateFactory):
            factory.__class__ = _CachingSurrogateFactory
        return factory

    def update_derived_quantities(self) -> None:
        """Install (once) a memoized full-iterate fetch on the equation system, then update.

        Every per-grid "flash" inside the after-iteration update re-fetches the ENTIRE system
        state.  There are TWO such grid-by-grid loops, and they run in different classes:
          * the phase-property update (density, enthalpy) in ``compositional_flow.py``, and
          * the locally-eliminated secondaries (T, s, x) in ``abstract_equations.LocalElimination``.
        Both evaluate their dependencies per grid with ``state=None``, so ``_ad_parser.evaluate``
        calls ``equation_system.get_variable_values(iterate_index=0)`` -- ALL variables on ALL
        subdomains, a ``numpy.copy`` per sub-variable -- for EVERY grid.  On a many-subdomain
        fracture network this is O(n_subdomains^2) and dominates the step (profiled: ~4.3s of a ~5s
        Newton step on the 62-subdomain Cartesian MD case; millions of array copies).

        ``LocalElimination.update_derived_quantities`` is the MRO entry point and runs its loop
        AFTER its ``super()`` call, so we cannot wrap it from here with a scoped patch.  Instead we
        install a PERSISTENT memoization of the full-iterate fetch (:meth:`_install_full_iterate_cache`),
        invalidated by a generation counter bumped on every variable-value write.  The primaries do
        not change during a single update and every flash dependency is a primary, so all flashes
        in one update share one fetch -- bit-exact, O(n_subdomains^2) -> O(n_subdomains).
        """
        self._install_full_iterate_cache()
        super().update_derived_quantities()

    def _install_full_iterate_cache(self) -> None:
        """Wrap ``equation_system.get_variable_values`` so the full-iterate fetch
        (``iterate_index=0``, no variable subset) is memoized until the next variable-value write.

        ``set_variable_values`` / ``shift_iterate_values`` bump a generation counter that
        invalidates the cache; all other fetches (subsets, other indices, reference values) pass
        through unchanged.  Idempotent (installs once per equation system).  A fresh copy is
        returned per call, so callers that mutate the result stay correct."""
        es = self.equation_system
        if getattr(es, "_full_iterate_cache", None) is not None:
            return
        cache = {"gen": 0, "cached_gen": -1, "value": None}
        es._full_iterate_cache = cache
        _get, _set, _shift = (
            es.get_variable_values, es.set_variable_values, es.shift_iterate_values)

        def get_variable_values(variables=None, time_step_index=None, iterate_index=None,
                                reference=False):
            if (variables is None and time_step_index is None
                    and iterate_index == 0 and not reference):
                if cache["cached_gen"] != cache["gen"]:
                    cache["value"] = _get(iterate_index=0)
                    cache["cached_gen"] = cache["gen"]
                return cache["value"].copy()
            return _get(variables=variables, time_step_index=time_step_index,
                        iterate_index=iterate_index, reference=reference)

        def set_variable_values(*args, **kwargs):
            cache["gen"] += 1
            return _set(*args, **kwargs)

        def shift_iterate_values(*args, **kwargs):
            cache["gen"] += 1
            return _shift(*args, **kwargs)

        es.get_variable_values = get_variable_values     # type: ignore[method-assign]
        es.set_variable_values = set_variable_values     # type: ignore[method-assign]
        es.shift_iterate_values = shift_iterate_values   # type: ignore[method-assign]

    def solve_linear_system_petsc(self, A: sps.spmatrix, b: np.ndarray, preconditioner: str = "lu") -> np.ndarray:
        """
        Solve linear system using PETSc with selectable preconditioners and detailed logging.
        """
        if not PETSC_AVAILABLE:
            raise RuntimeError("PETSc is not available")

        # Only two linear solvers are supported: "cpr" (Schur-reduced CPR, iterative) and
        # "lu" (direct LU via MUMPS).
        if preconditioner not in {"lu", "cpr"}:
            logger.warning(f"Invalid linear solver '{preconditioner}'. Using 'cpr' as default.")
            preconditioner = "cpr"

        # CPR is a self-contained Schur reduction + CPR (its own DOF partition, so it needs neither
        # the equation permutation nor the matrix scaling below).  It converges well when the
        # transport is not strongly advection-dominated (fracture-free / low-flow cases); on the
        # high-contrast fractured MD system the coupled advection-diffusion transport defeats the ILU
        # smoother, so fall back to the direct MUMPS LU rather than fail the Newton step.  (Manually
        # Schur-eliminating the local secondaries before the LU does NOT help: MUMPS already handles
        # the identity secondary block with zero fill, and the Schur complement only adds fill.)
        if preconditioner == "cpr":
            try:
                return self._schur_cpr_solve(A.tocsr(), np.asarray(b, dtype=float))
            except Exception as exc:
                logger.warning("Schur-CPR did not converge; falling back to direct LU (MUMPS).")
                logger.warning("  reason: %s", exc)
                preconditioner = "lu"

        logger.info(f"Solving linear system with PETSc {preconditioner.upper()}")

        # 1. Convert to CSR and prepare working vector
        A_csr = A.tocsr()
        b_working = b.copy()

        # Initialize permutation variables to None for safety
        perm = None
        eq_perm = None
        var_perm = None
        field_split = None

        # 1.5. Apply equation permutation
        try:
            A_csr, b_working, eq_perm, var_perm, field_split = self.apply_equation_permutation(A_csr, b_working)
        except Exception as e:
            logger.warning(f"Equation permutation failed: {e}. Continuing with original ordering.")
            if preconditioner == "cpr":
                logger.warning("CPR requires successful equation permutation. Falling back to 'asm'.")
                preconditioner = "asm"

        # 2. Apply Cuthill-McKee permutation
        if self.use_cuthill_mckee and preconditioner not in ["lu", "cpr"]:
            try:
                perm = reverse_cuthill_mckee(A_csr, symmetric_mode=False)
                A_csr = A_csr[perm, :][:, perm]
                b_working = b_working[perm]
            except Exception as e:
                logger.warning(f"Cuthill-McKee permutation failed: {e}. Continuing with original ordering.")
                perm = None

        # 3. Regularize Diagonal
        if preconditioner not in ["lump_colsum", "lu"]:
            diagonal = A_csr.diagonal()
            zero_diag_indices = np.where(np.abs(diagonal) < 1e-14)[0]
            if len(zero_diag_indices) > 0:
                logger.info(f"Regularizing {len(zero_diag_indices)} zero diagonal entries")
                A_lil = A_csr.tolil()
                matrix_norm = np.mean(np.abs(A_csr.data))
                regularization_value = max(1e-12, matrix_norm * 1e-8)
                for idx in zero_diag_indices:
                    A_lil[idx, idx] = regularization_value
                A_csr = A_lil.tocsr()

        # 4. Apply Matrix Scaling
        row_scaling, col_scaling, A_csr, b_scaled = self._apply_matrix_scaling(A_csr, b_working)

        # 5. Create PETSc Matrices/Vectors
        petsc_A = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        petsc_A.assemblyBegin()
        petsc_A.assemblyEnd()

        petsc_b = PETSc.Vec().createWithArray(b_scaled)
        petsc_x = PETSc.Vec().createWithArray(np.zeros_like(b_scaled))

        # 6. Setup KSP
        ksp = PETSc.KSP().create()
        ksp_prefix = "fluid_buoyancy_"
        ksp.setOptionsPrefix(ksp_prefix)

        # Initialize explicit references for cleanup
        petsc_M = None
        is_p = None
        is_t = None

        # 7. Configure Solver
        if preconditioner == "lu":
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            ksp.setOperators(A=petsc_A, P=petsc_A)
        else:
            ksp.setType(PETSc.KSP.Type.FGMRES)
            ksp.setGMRESRestart(50)

            # Setup Operators
            if preconditioner == "lump_colsum":
                col_sums = np.array(np.abs(A_csr).sum(axis=0)).flatten()
                zero_cols = np.where(col_sums < 1e-14)[0]
                if len(zero_cols) > 0:
                    col_sums[zero_cols] = 1e-12
                diag_vals = 1.0 / col_sums

                petsc_M = PETSc.Mat().createAIJ(size=A_csr.shape)
                petsc_M.setUp()
                for i in range(len(diag_vals)):
                    petsc_M.setValue(i, i, diag_vals[i])
                petsc_M.assemblyBegin()
                petsc_M.assemblyEnd()
                ksp.setOperators(A=petsc_A, P=petsc_M)
            else:
                ksp.setOperators(A=petsc_A, P=petsc_A)

            # Setup PC
            pc = ksp.getPC()
            opts = PETSc.Options()

            if preconditioner == "cpr":
                if not field_split:
                    raise RuntimeError("CPR preconditioner requires 'field_split' data.")

                try:
                    n_pressure = field_split.get('pressure',
                                                 field_split.get('pressure_size', list(field_split.values())[0]))
                except (AttributeError, IndexError):
                    raise RuntimeError("Could not parse 'field_split' dictionary.")

                n_total = A_csr.shape[0]
                is_p = PETSc.IS().createStride(n_pressure, first=0, step=1)
                is_t = PETSc.IS().createStride(n_total - n_pressure, first=n_pressure, step=1)

                pc.setType(PETSc.PC.Type.FIELDSPLIT)
                pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
                pc.setFieldSplitIS(('pressure', is_p), ('transport', is_t))

                # --- Block 0: Pressure ---
                # Protection: Use LU for small matrices (<10k rows) to avoid AMG setup failures
                if n_pressure < 10000:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "preonly")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "lu")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_factor_shift_type", "nonzero")
                else:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "preonly")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "lu")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_factor_mat_solver_type", "mumps")

                # --- Block 1: Transport ---
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_ksp_type", "richardson")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_type", "ilu")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_levels", "0")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_amount", "1e-10")

            elif preconditioner == "ilu0":
                pc.setType(PETSc.PC.Type.ILU)
                pc.setFactorLevels(0)
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_amount", "1e-12")

            elif preconditioner == "amg_hypre":
                pc.setType(PETSc.PC.Type.HYPRE)
                pc.setHYPREType("boomeramg")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_strong_threshold", "0.25")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_coarsen_type", "HMIS")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_interp_type", "ext+i")

            elif preconditioner == "bjacobi":
                pc.setType(PETSc.PC.Type.BJACOBI)
            elif preconditioner == "asm":
                pc.setType(PETSc.PC.Type.ASM)
                pc.setASMOverlap(1)
            elif preconditioner == "jacobi":
                pc.setType(PETSc.PC.Type.JACOBI)
            elif preconditioner == "lump_colsum":
                pc.setType(PETSc.PC.Type.MAT)

        # 8. Finalize Options
        ksp.setFromOptions()

        # Apply tolerances AFTER setFromOptions to strictly enforce them.
        # This overrides any command-line defaults or database presets.
        ksp.setTolerances(rtol=1.0e-5, atol=1.0e-8, max_it=500)

        # Optional: Log the actual tolerances PETSc is using to be 100% sure
        r_tol, a_tol, div_tol, max_its = ksp.getTolerances()
        logger.info(f"KSP Tolerances Enforced | rtol: {r_tol}, atol: {a_tol}, max_it: {max_its}")

        # 9. Solve and Log
        solution = None
        try:
            # Step A: Explicitly time the Preconditioner Setup
            t_setup_start = time.time()
            ksp.setUp()
            t_setup_end = time.time()
            setup_dur = t_setup_end - t_setup_start

            # Step B: Time the Solve
            t_solve_start = time.time()
            ksp.solve(petsc_b, petsc_x)
            t_solve_end = time.time()
            solve_dur = t_solve_end - t_solve_start

            # Step C: Retrieve Metrics
            iters = ksp.getIterationNumber()
            resid = ksp.getResidualNorm()

            # Step D: Log Report
            logger.info(
                f"PETSc {preconditioner.upper()} Report | Setup: {setup_dur:.4f}s | Solve: {solve_dur:.4f}s | Iters: {iters} | Residual: {resid:.4e}")

            if ksp.getConvergedReason() < 0:
                logger.warning(f"Solver failed. Reason: {ksp.getConvergedReason()}")
            else:
                # 10. Unscale and Reverse Permutations
                scaled_sol = petsc_x.getArray().copy()
                unscaled_sol = col_scaling * scaled_sol

                if perm is not None:
                    cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                    cuthill_reversed_sol[perm] = unscaled_sol
                    unscaled_sol = cuthill_reversed_sol

                if var_perm is not None:
                    solution = np.zeros_like(unscaled_sol)
                    solution[var_perm] = unscaled_sol
                else:
                    solution = unscaled_sol

        except Exception as e:
            # Fallback for LU
            if preconditioner == "lu" and "mumps" in str(e).lower():
                logger.warning("MUMPS failed. Retrying with PETSc native LU...")
                try:
                    pc.setFactorSolverType("petsc")
                    ksp.setFromOptions()
                    ksp.solve(petsc_b, petsc_x)
                    if ksp.getConvergedReason() >= 0:
                        scaled_sol = petsc_x.getArray().copy()
                        unscaled_sol = col_scaling * scaled_sol

                        if perm is not None:
                            cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                            cuthill_reversed_sol[perm] = unscaled_sol
                            unscaled_sol = cuthill_reversed_sol

                        if var_perm is not None:
                            solution = np.zeros_like(unscaled_sol)
                            solution[var_perm] = unscaled_sol
                        else:
                            solution = unscaled_sol
                except Exception as e2:
                    logger.error(f"Fallback solver failed: {e2}")
            else:
                logger.error(f"Solver execution error: {e}")
        # Cleanup
        petsc_A.destroy()
        petsc_b.destroy()
        petsc_x.destroy()
        ksp.destroy()
        if petsc_M: petsc_M.destroy()
        if is_p: is_p.destroy()
        if is_t: is_t.destroy()

        return solution

    # ----------------------------------------------------------------------------------------- #
    #  Schur-reduced CPR -- the iterative solver.  Ported from subsection_4_2/porepy_2d_solver.py.
    #  The constant-pressure null-mean gauge is OPTIONAL, controlled by
    #  ``params["null_mean_pressure"]`` (default False): OFF for the usual Dirichlet inlet/outlet
    #  problem (non-singular pressure block -> plain CPR); ON for a fully closed / all-Neumann
    #  domain (singular constant-pressure mode -> Sum(dp_matrix)=0 pinned on the matrix rows).
    # ----------------------------------------------------------------------------------------- #
    _ELLIPTIC_VARS = ("pressure",)   # only pressure is elliptic; enthalpy is advective (-> ILU)

    @staticmethod
    def _equation_for_variable(varname: str, eq_names: list):
        """Equation that determines ``varname`` (PorePy names equations independently of the
        variables): pressure<->mass_balance, enthalpy<->energy_balance,
        z_<c><->component_mass_balance_<c>, each interface flux <-> its <var>_equation, and each
        locally-eliminated variable <-> its elimination_of_<var>_on_grids_... equation."""
        if varname == "pressure":
            return "mass_balance_equation"
        if varname == "enthalpy":
            return "energy_balance_equation"
        if varname.startswith("z_"):
            return "component_mass_balance_equation_" + varname[2:]
        if varname.startswith("interface_"):
            return varname + "_equation"
        cands = [e for e in eq_names if e.startswith(f"elimination_of_{varname}_on_grids")]
        return cands[0] if cands else None

    def _primary_secondary_indices(self, n: int):
        """Partition assembled DOFs into three groups, each equation-row aligned with its variable
        column: SUBDOMAIN primaries (pressure -- elliptic, FIRST -- then enthalpy and the overall
        fractions z), the INTERFACE mortar fluxes (interface_darcy/enthalpy/fourier -- the
        mixed-dimensional coupling block), and the local SECONDARY closures (T, s, x). Returns
        ``(subdomain_cols, subdomain_rows, interface_cols, interface_rows, secondary_cols,
        secondary_rows, n_pressure, matrix_p_pos)``."""
        es = self.equation_system
        aei = es.assembled_equation_indices
        eq_names = list(aei.keys())
        vars_by_name: dict = {}
        for v in es.variables:                          # atomic Variable objects, one per grid
            vars_by_name.setdefault(v.name, []).append(v)
        var_names = sorted(vars_by_name)

        def eq_of(v):
            return self._equation_for_variable(v, eq_names)

        def is_secondary(v):
            eq = eq_of(v)
            return eq is not None and eq.startswith("elimination_of_")

        # SUBDOMAIN primaries: pressure (elliptic, first, -> AMG field) + enthalpy + z (-> ILU).
        # INTERFACE: every mortar flux (darcy/enthalpy/fourier), eliminated in the second Schur.
        interface_vars = [v for v in var_names if v.startswith("interface_")]
        elliptic = [v for v in self._ELLIPTIC_VARS if v in var_names]
        middle = [v for v in var_names
                  if v not in elliptic and v not in interface_vars and not is_secondary(v)]
        subdomain_vars = elliptic + middle
        secondary_vars = [v for v in var_names if is_secondary(v)]

        def cols(vs):
            # Gather each variable's global DOFs by NAME across ALL grids (subdomains AND interfaces).
            return [np.asarray(es.dofs_of(vars_by_name[v]), dtype=int) for v in vs]

        def rows(vs):
            return [np.asarray(aei[eq_of(v)], dtype=int) for v in vs]

        d_cols, d_rows = cols(subdomain_vars), rows(subdomain_vars)
        i_cols, i_rows = cols(interface_vars), rows(interface_vars)
        s_cols, s_rows = cols(secondary_vars), rows(secondary_vars)

        def cat(parts):
            return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

        subdomain_cols, subdomain_rows = cat(d_cols), cat(d_rows)
        interface_cols, interface_rows = cat(i_cols), cat(i_rows)
        secondary_cols, secondary_rows = cat(s_cols), cat(s_rows)
        if not (np.array_equal(
                    np.sort(np.concatenate([subdomain_cols, interface_cols, secondary_cols])),
                    np.arange(n))
                and np.array_equal(
                    np.sort(np.concatenate([subdomain_rows, interface_rows, secondary_rows])),
                    np.arange(n))):
            raise RuntimeError("Schur partition: subdomain+interface+secondary do not partition [0,n)")
        n_pressure = len(d_cols[0])
        # Positions of the equidimensional-MATRIX pressure DOFs within the pressure block; the
        # optional null-mean gauge pins Sum(dp_matrix)=0 on these.
        matrix_p = np.asarray(
            es.dofs_of([es.md_variable("pressure", [self.mdg.subdomains(dim=self.nd)[0]])]), dtype=int)
        matrix_p_pos = np.nonzero(np.isin(d_cols[0], matrix_p))[0]
        return (subdomain_cols, subdomain_rows, interface_cols, interface_rows,
                secondary_cols, secondary_rows, n_pressure, matrix_p_pos)

    def _schur_cpr_solve(self, A, b) -> np.ndarray:
        """Two exact Schur reductions to a clean per-variable mixed-dimensional (p, h, z) subdomain
        system, then CPR, then back-substitution.

        (1) Eliminate the LOCAL secondary closures (T, s, x): ``A_ss`` is block-diagonal per cell
        (== I when the closures depend only on primaries), so its factorization is cheap.
        (2) Eliminate the INTERFACE mortar fluxes (interface_darcy/enthalpy/fourier) via a sparse LU
        of their (near-block-diagonal per interface) self-block.  Folding this coupling into the
        SUBDOMAIN blocks makes pressure a CONNECTED mixed-dimensional Darcy Laplacian (matrix +
        fractures + intersections) that AMG coarsens, and leaves enthalpy/z as clean advection
        operators for ILU.  Left in place, the interface fluxes are saddle-point constraints that
        make even an exact pressure solve diverge."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu, spsolve

        t0 = time.perf_counter()
        n = A.shape[0]
        dc, dr, ic, ir, sc, sr, n_p, matrix_p_pos = self._primary_secondary_indices(n)
        null_mean = bool(self.params.get("null_mean_pressure", False))
        n_i = len(ic)
        # PRIMARY = subdomain + interface (interface trailing); SECONDARY = local closures.
        pc = np.concatenate([dc, ic]); pr = np.concatenate([dr, ir])
        A = A.tocsc()
        App = A[pr][:, pc].tocsr()
        Aps = A[pr][:, sc].tocsr()
        Asp = A[sr][:, pc].tocsc()
        Ass = A[sr][:, sc].tocsr()
        bp, bs = b[pr], b[sr]

        # (1) A_ss = Jacobian of ``var - func(primary) = 0``: diagonal identity, no secondary<->
        # secondary coupling -> A_ss == I. Detect and skip the factorization; LU only if non-trivial.
        is_identity = (Ass.shape[0] == Ass.nnz
                       and np.allclose(Ass.data, 1.0)
                       and np.array_equal(Ass.indices, np.arange(Ass.shape[0])))
        if is_identity:
            logger.info("  secondary block A_ss is diagonal (identity): LU skipped")
            Ainv_Asp = Asp
            Ainv_bs = bs
            lu = None
        else:
            logger.info("  secondary block A_ss is not diagonal: LU factorization needed")
            lu = splu(Ass.tocsc())
            Ainv_Asp = spsolve(Ass.tocsc(), Asp)
            if not sps.issparse(Ainv_Asp):
                Ainv_Asp = sps.csc_matrix(Ainv_Asp)
            Ainv_bs = lu.solve(bs)

        S = (App - Aps @ Ainv_Asp).tocsr()                  # (subdomain + interface) system
        g = bp - Aps @ Ainv_bs

        # (2) Eliminate the trailing INTERFACE block via a sparse LU of its self-block (unit diagonal
        # + local coupling -> invertible; near-block-diagonal per interface -> cheap). n_i == 0 (no
        # fractures) -> no-op.
        m = S.shape[0] - n_i
        if n_i:
            Skl = S[:m, m:]
            Slk = S[m:, :m].tocsc()
            Sll = S[m:, m:].tocsc()
            lu_i = splu(Sll)
            # Sll^-1 @ Slk (needed as a SPARSE matrix): scipy spsolve(Sll, Slk) with a sparse RHS is
            # catastrophically slow (~80s), and a dense back-solve of every nonzero column is also
            # slow (~18s) since it densifies a result that is actually sparse.  Instead exploit that
            # ``Sll`` is unit-diagonal + a small off-diagonal coupling (near diagonally dominant): a
            # few Jacobi/Richardson sweeps (all sparse mat-mats) give ``Sll^-1 @ Slk`` exactly and
            # fast.  (Vector solves -- the RHS fold and the back-substitution -- still use ``lu_i``.)
            Slk = Slk.tocsr()
            d_inv = sps.diags(1.0 / Sll.diagonal())
            W = (d_inv @ Slk).tocsr()                                  # Jacobi initial guess
            scale = np.abs(Slk.data).max() if Slk.nnz else 1.0
            converged = False
            for _ in range(30):
                R = Slk - Sll @ W
                if R.nnz == 0 or np.abs(R.data).max() <= 1.0e-12 * scale:
                    converged = True
                    break
                W = (W + d_inv @ R).tocsr()
            if not converged:                                          # not diagonally dominant
                raise RuntimeError("interface Schur fold (Jacobi) did not converge")
            Sc = (S[:m, :m] - Skl @ W).tocsr()
            gk, gl = g[:m], g[m:]
            gc = gk - Skl @ lu_i.solve(gl)
        else:
            Sc, gc = S, g

        xk, cpr_its = self._cpr_petsc_solve(
            Sc, gc, n_p, matrix_p_pos, null_mean,
            lu_pressure_max=int(self.params.get("cpr_lu_pressure_max", 60000)),
            rtol=float(self.params.get("cpr_rtol", 1.0e-8)),
            maxit=int(self.params.get("cpr_maxit", 300)))
        xp = np.concatenate([xk, lu_i.solve(gl - Slk @ xk)]) if n_i else xk

        # Accuracy gate.  Dirichlet: plain residual.  Null-mean: the system is singular AND
        # inconsistent (the mass imbalance lies along the constant-p mode), so measure the residual
        # with that direction projected out (matches the bordered null-mean solution).
        r = S @ xp - g
        if null_mean:
            vp = np.zeros(len(xp)); vp[matrix_p_pos] = 1.0
            r = r - (vp @ r) / (vp @ vp) * vp
        rel = np.linalg.norm(r) / max(np.linalg.norm(g), 1.0e-30)
        acc_tol = float(self.params.get("cpr_accuracy_tol", 1.0e-6))
        if rel > acc_tol:
            raise RuntimeError(
                f"CPR residual too large for Newton (rel={rel:.1e} > {acc_tol:.1e})")

        rhs_s = bs - Asp @ xp
        xs = rhs_s if lu is None else lu.solve(rhs_s)       # back-substitute the secondaries

        logger.info(
            "Schur-CPR solve: %.3fs (%d KSP its, res %.1e%s)",
            time.perf_counter() - t0, cpr_its, rel, ", null-mean" if null_mean else "")
        logger.info("  reduced %d = p %d (AMG) + transport %d (ILU)", m, n_p, m - n_p)
        logger.info("  eliminated: %d interface + %d secondary", n_i, len(sr))

        x = np.empty(n, dtype=float)
        x[pc] = xp
        x[sc] = xs
        return x

    @staticmethod
    def _cpr_petsc_solve(S, g, n_p, matrix_p_pos, null_mean=False,
                         lu_pressure_max=60000, rtol=1.0e-8, maxit=300):
        """CPR on the reduced subdomain system ``S x = g`` (interface mortar fluxes and local
        secondaries already Schur-eliminated): a THREE-field (pressure | enthalpy | composition)
        block preconditioner with a per-cell block decoupling, driven by full GMRES.

        The reduced system carries one DOF per cell for each subdomain variable -- pressure
        (elliptic Darcy), enthalpy (advection-diffusion), the overall fractions z (pure advection) --
        so it is ``nvar = 2 + n_comp`` blocks of ``N = n_p`` cells each.

        (1) ABF DECOUPLING: left-multiply by the inverse of the per-cell ``nvar x nvar`` block of
            LOCAL couplings -- the compressible EOS makes density (hence every equation) depend on
            p, h and z WITHIN each cell, and that is the dominant coupling that makes a naive block
            split diverge.  The block is nonsingular, so the solution is unchanged.
        (2) THREE-field MULTIPLICATIVE block preconditioner: pressure -> AMG (direct LU/MUMPS when
            small, else BoomerAMG -- the connected MD Darcy Laplacian over 3D+2D+1D pressures);
            enthalpy -> direct LU (advection-diffusion, the hard block); composition z -> ILU(0)
            (a pure-advection DAG, for which ILU is ~exact).
        (3) FULL (un-restarted) GMRES -- essential: a restart discards the Krylov modes that resolve
            the residual inter-field (spatial advective) coupling, and the solve stalls at ~1e-1.

        ``null_mean`` handles a singular (closed / all-Neumann) pressure block: pin
        ``Sum(dp_matrix)=0`` on the matrix rows and attach the constant-pressure null space.
        Returns ``(x, n_iterations)``."""
        import scipy.sparse as sps
        from petsc4py import PETSc

        S = S.tocsr(); S.sort_indices()
        n = S.shape[0]
        g = np.array(g, dtype=float)
        if null_mean:
            g[matrix_p_pos] -= g[:n_p].sum() / len(matrix_p_pos)

        N = n_p                                          # cells per subdomain variable
        nvar = n // N if N else 0                        # 2 + n_comp  (pressure, enthalpy, z...)

        # (1) ABF: invert the per-cell block of local (EOS) couplings and left-multiply.
        if nvar >= 2 and n == nvar * N:
            ar = np.arange(N)
            blk = np.empty((N, nvar, nvar))
            for a in range(nvar):
                for b in range(nvar):
                    blk[:, a, b] = S[a * N:(a + 1) * N, b * N:(b + 1) * N].diagonal()
            binv = np.linalg.inv(blk)
            rows, cols, data = [], [], []
            for a in range(nvar):
                for b in range(nvar):
                    rows.append(a * N + ar); cols.append(b * N + ar); data.append(binv[:, a, b])
            dinv = sps.csr_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
            S = (dinv @ S).tocsr(); S.sort_indices()
            g = dinv @ g

        M = S.tocsr(); M.sort_indices()
        mat = PETSc.Mat().createAIJ(
            size=(n, n),
            csr=(M.indptr.astype(PETSc.IntType), M.indices.astype(PETSc.IntType),
                 np.ascontiguousarray(M.data, dtype=PETSc.ScalarType)),
            comm=PETSc.COMM_SELF)
        mat.assemble()

        def const_p_vec(operator):
            w = operator.createVecRight()
            a = w.getArray(); a[:] = 0.0; a[:n_p] = 1.0
            w.assemble(); w.normalize()
            return w

        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("gmres")
        ksp.setGMRESRestart(maxit)                       # (3) FULL GMRES -- no restart
        ksp.setTolerances(rtol=rtol, atol=1.0e-50, max_it=maxit)
        nsp = None
        if null_mean:
            nsp = PETSc.NullSpace().create(constant=False, vectors=[const_p_vec(mat)],
                                           comm=PETSc.COMM_SELF)
            mat.setNullSpace(nsp)

        # (2) three-field multiplicative block preconditioner.
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        fields = [("p", PETSc.IS().createStride(N, first=0, step=1, comm=PETSc.COMM_SELF))]
        if nvar >= 2:
            fields.append(("h", PETSc.IS().createStride(N, first=N, step=1, comm=PETSc.COMM_SELF)))
        if nvar >= 3:
            fields.append(("z", PETSc.IS().createGeneral(
                np.arange(2 * N, n, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)))
        pc.setFieldSplitIS(*fields)
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        pc.setUp()
        subksps = pc.getFieldSplitSubKSP()

        # pressure (elliptic Darcy) -> LU (MUMPS) when small, else BoomerAMG; GAMG + null space if the
        # block is singular (null_mean).
        kp = subksps[0]; kp.setType("preonly")
        App_block = kp.getOperators()[0]
        if null_mean:
            p_nsp = PETSc.NullSpace().create(constant=False, vectors=[const_p_vec(App_block)],
                                             comm=PETSc.COMM_SELF)
            kp.getPC().setType("gamg")
            App_block.setNullSpace(p_nsp); App_block.setNearNullSpace(p_nsp)
        elif n_p < lu_pressure_max:
            kp.getPC().setType("lu"); kp.getPC().setFactorSolverType("mumps")
        else:
            kp.getPC().setType("hypre"); kp.getPC().setHYPREType("boomeramg")
            App_block.setNearNullSpace(PETSc.NullSpace().create(
                constant=False, vectors=[const_p_vec(App_block)], comm=PETSc.COMM_SELF))
        # enthalpy (advection-diffusion, the hard block) -> direct LU.
        if len(subksps) >= 2:
            kh = subksps[1]; kh.setType("preonly")
            kh.getPC().setType("lu"); kh.getPC().setFactorSolverType("mumps")
        # composition z (pure advection -- a flow DAG for which ILU is ~exact) -> ILU(0).
        for kz in subksps[2:]:
            kz.setType("preonly"); kz.getPC().setType("ilu")

        xv = mat.createVecRight()
        bv = mat.createVecLeft()
        bv.setArray(np.ascontiguousarray(g, dtype=PETSc.ScalarType))
        if null_mean:
            nsp.remove(bv)
        ksp.solve(bv, xv)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"PETSc CPR KSP diverged (reason {ksp.getConvergedReason()}, "
                               f"its={ksp.getIterationNumber()})")
        return xv.getArray().copy(), ksp.getIterationNumber()

    def _apply_matrix_scaling(self, A_csr, b):
        """
        Apply row and column scaling to improve matrix conditioning.

        Parameters:
        -----------
        A_csr : scipy sparse matrix
            Input matrix in CSR format
        b : numpy array
            Right-hand side vector

        Returns:
        --------
        tuple
            (row_scaling, col_scaling, scaled_A_csr, scaled_b) where scaling factors and scaled matrix/vector
        """

        # Compute row and column norms for scaling
        row_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=1))).flatten()
        col_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=0))).flatten()

        # Avoid division by zero
        row_norms = np.where(row_norms < 1e-16, 1.0, row_norms)
        col_norms = np.where(col_norms < 1e-16, 1.0, col_norms)

        # Create scaling factors (inverse of norms for better conditioning)
        row_scaling = 1.0 / np.sqrt(row_norms)
        col_scaling = 1.0 / np.sqrt(col_norms)

        # Apply scaling: D_r * A * D_c where D_r, D_c are diagonal scaling matrices
        A_scaled = sps.diags(row_scaling) @ A_csr @ sps.diags(col_scaling)

        # Scale right-hand side: D_r * b
        b_scaled = row_scaling * b

        logger.debug(f"Matrix scaling applied. Row norm range: [{np.min(row_norms):.2e}, {np.max(row_norms):.2e}], "
                    f"Col norm range: [{np.min(col_norms):.2e}, {np.max(col_norms):.2e}]")

        return row_scaling, col_scaling, A_scaled.tocsr(), b_scaled

    def _solve_linear_system_core(self) -> np.ndarray:
        """
        Core linear solve (no step control): PETSc GMRES with a selectable preconditioner,
        or the default direct solver. This is wrapped by :meth:`solve_linear_system`,
        which adds the optional line-search / trust-region step control.

        Preconditioner options (set via petsc_preconditioner parameter):
        - 'bjacobi': Block Jacobi preconditioner (default)
        - 'asm': Additive Schwarz Method
        - 'jacobi': Point Jacobi preconditioner
        - 'lump_colsum': Lumped column sum diagonal preconditioner
        - 'amg_hypre': Algebraic Multigrid with Hypre BoomerAMG

        Returns:
            np.ndarray: Solution vector (the nonlinear increment).
        """
        if self.use_petsc and PETSC_AVAILABLE:
            # Use PETSc solver with selected preconditioner
            A, b = self.linear_system
            solution = self.solve_linear_system_petsc(A, b, preconditioner=self.petsc_preconditioner)
            if solution is None:
                logger.warning(f"PETSc iterative solver with {self.petsc_preconditioner.upper()} preconditioner failed to converge.")
                return super().solve_linear_system()
            return solution
        else:
            # Check if PETSc was requested but not available
            if self.use_petsc and not PETSC_AVAILABLE:
                logger.info("*** SOLVER FALLBACK ***")
                logger.info("PETSc was requested but not available. Using default direct solver.")

            # Use default solver
            solution = super().solve_linear_system()
            if solution is None:
                raise RuntimeError("Linear solver returned None - this should not happen")
            return solution

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system and apply the configured nonlinear step control.

        Selected by ``params["step_control_method"]`` (default ``"None"``):

        - ``"None"``  : plain Newton, no step control.
        - ``"LS"``    : backtracking line search (Armijo), applied only when the full
          Newton step would increase the residual.
        - ``"TR"``    : CFL-based trust region.
        - ``"TR-LS"`` : trust region followed by a line-search refinement.

        Residual reporting and solution post-processing are intentionally left to the
        model (see the overridable hooks :meth:`compute_residuals_by_category`,
        :meth:`postprocessing_overshoots`, :meth:`postprocessing_thermal_overshoots`);
        this method only chooses and applies the step.
        """
        _, residual_vector = self.linear_system
        residual_norm_current = np.linalg.norm(residual_vector)

        step_control_method = self.params.get("step_control_method", "None")
        step_control_alpha_min = self.params.get("step_control_alpha_min", 0.01)
        activate_after_iteration = self.params.get("activate_step_control_after_iter", 1)
        activate_step_control_Q = (
            self.nonlinear_solver_statistics.num_iterations > activate_after_iteration
        )

        # Reset the trust radius at the start of each time step (iteration 0).
        if self.nonlinear_solver_statistics.num_iterations == 0:
            self._trust_radius = 1.0

        if step_control_method == "None":
            solution = self._solve_linear_system_core()

        elif step_control_method == "LS":
            solution = self._solve_linear_system_core()
            residual_future = self.compute_residual_from_increment(solution, restore_state=True)
            increasing_residual_Q = np.linalg.norm(residual_future) > residual_norm_current
            if increasing_residual_Q and activate_step_control_Q:
                print("Step control: Line Search (LS)")
                alpha = self.backtracking_line_search(
                    solution, residual_vector, alpha_min=step_control_alpha_min
                )
                solution *= alpha
                print(f"Line search: accepted alpha = {alpha:.4f}")

        elif step_control_method == "TR":
            if activate_step_control_Q:
                print("Step control: Trust Region (TR)")
                solution, self._trust_radius = self.trust_region_solve(
                    trust_radius=self._trust_radius
                )
            else:
                solution = self._solve_linear_system_core()

        elif step_control_method == "TR-LS":
            if activate_step_control_Q:
                print("Step control: Trust Region + Line Search (TR-LS)")
                solution, self._trust_radius = self.trust_region_solve(
                    trust_radius=self._trust_radius
                )
                residual_after_tr = self.compute_residual_from_increment(
                    solution, restore_state=True
                )
                if np.linalg.norm(residual_after_tr) > residual_norm_current * 0.9:
                    alpha = self.backtracking_line_search(
                        solution, residual_vector, alpha_min=step_control_alpha_min
                    )
                    solution *= alpha
                    print(f"  TR-LS: line search alpha = {alpha:.4f}")
            else:
                solution = self._solve_linear_system_core()

        else:
            raise ValueError(
                f"Unknown step_control_method: {step_control_method}. "
                f"Valid options are: 'None', 'LS', 'TR', 'TR-LS'"
            )

        if self.params.get("reduce_linear_system_q", False):
            raise NotImplementedError(
                "The 'reduce_linear_system_q' case is not yet implemented."
            )

        return solution

    def compute_residual_from_increment(
        self, nonlinear_increment: np.ndarray, restore_state: bool = True
    ) -> np.ndarray:
        """
        Compute the residual after applying a nonlinear increment.

        This method follows the logic for residual evaluation:
        1. Save current state (if restore_state=True)
        2. Apply the nonlinear increment
        3. Update derived quantities
        4. Update buoyancy-driven fluxes
        5. Rediscretize
        6. Assemble the residual
        7. Restore original state (if restore_state=True)
        8. Return the residual vector

        Parameters:
            nonlinear_increment: The increment to apply to current variable values
            restore_state: If True, restore the original state after computing residual.
                          Set to False when this is the final accepted increment.

        Returns:
            The residual vector
        """
        # Save current state if we need to restore it later
        if restore_state:
            x_current = self.equation_system.get_variable_values(iterate_index=0).copy()

        # Apply the nonlinear increment additively to the current iterate
        self.equation_system.set_variable_values(
            values=nonlinear_increment, additive=True, iterate_index=0
        )

        # Update derived quantities
        self.update_derived_quantities()

        # Update buoyancy-driven fluxes (skipped when the direction is lagged/frozen)
        self.refresh_buoyancy_direction()

        # Rediscretize
        self.rediscretize()

        # Assemble the current nonlinear residual
        current_nonlinear_residual = self.equation_system.assemble(evaluate_jacobian=False)

        # Restore original state if requested
        if restore_state:
            try:
                self.equation_system.set_variable_values(x_current, iterate_index=0)
            except TypeError:
                self.equation_system.set_variable_values(x_current)

            # CRITICAL: Must also restore derived quantities and discretization
            # Otherwise the state is corrupted for the next iteration
            self.update_derived_quantities()
            self.refresh_buoyancy_direction()
            self.rediscretize()

        return current_nonlinear_residual

    def estimate_mixed_dimensional_cfl_number(self) -> tuple[float, float, float]:
        """
        Estimate the MD-CFL number

        Uses the same divergence operator as mass balance equations:
            ∂(φρ)/∂t + ∇·(ρ q) = 0

        where div = pp.ad.Divergence(subdomains, dim=1)

        Returns:
            tuple: (cfl_max, div_max, dx_min)
                - cfl_max: Maximum CFL number over all cells
                - div_max: Maximum divergence magnitude [1/s]
                - dx_min: Minimum cell size [m]
        """

        # Get the subdomains
        subdomains = self.mdg.subdomains(dim=self.nd)
        # Get current time step
        dt = self.time_manager.dt

        # Get characteristic cell size
        cell_diameters = self.volume_integral(1,subdomains,dim=1).value(self.equation_system)
        dx_min = np.min(cell_diameters)

        # === Use PorePy's AD operators (same as mass balance equations) ===

        # 1. Get Darcy flux using AD operator
        darcy_flux_ad = self.darcy_flux(subdomains)

        # 2. Get density on cells
        density_ad = self.fluid.density(subdomains)
        density_values = density_ad.value(self.equation_system)

        # 3. Use PorePy's Divergence operator (consistent with mass balance)
        div_operator = pp.ad.Divergence(subdomains, dim=1)

        # Compute divergence of Darcy flux [m³/s/m³ = 1/s]
        div_darcy_ad = div_operator @ darcy_flux_ad
        div_mass_flux = div_darcy_ad.value(self.equation_system)

        # Absolute divergence
        abs_div = np.abs(div_mass_flux)

        # 4. Get accumulation density: φρ [kg/m³]
        porosity_op = self.porosity(subdomains)
        porosity = porosity_op.value(self.equation_system)

        accumulation_density = porosity * density_values

        # 5. CFL number: CFL = dt * |∇·(ρq)| / (φρ)
        cfl_per_cell = np.nan_to_num(dt * abs_div / (accumulation_density) , nan=0.0, posinf=0.0)
        cfl_max = np.max(cfl_per_cell)
        div_max = np.max(np.nan_to_num(abs_div / (accumulation_density) , nan=0.0, posinf=0.0))

        return cfl_per_cell, cfl_max, div_max, dx_min

    def trust_region_solve(
            self,
            trust_radius: float = 1.0,
            eta: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """
        Simplified CFL-based Trust Region solver.

        Strategy:
        - Calculate effective trust radius as: cfl_target / cfl_current
        - Trust pressure Newton step completely (parabolic, well-behaved)
        - Apply CFL-based trust region to hyperbolic variables (enthalpy, composition)
        """
        # Get target CFL parameter
        cfl_target = self.params.get("trust_region_cfl_max_target", 1.0)

        # Get Jacobian and residual
        jacobian_matrix, residual_vector = self.linear_system
        residual_norm_current = np.linalg.norm(residual_vector)

        # Estimate current CFL number
        cfl_per_cell, cfl_current, div_max, dx_min = self.estimate_mixed_dimensional_cfl_number()
        print(f"  TR-CFL: Current CFL={cfl_current:.4f}, div_max={div_max:.2e} 1/s, dx_min={dx_min:.2e} m")

        h_op = self.enthalpy(self.mdg.subdomains())
        h_values = h_op.value(self.equation_system)
        CFL_energy = np.max(cfl_per_cell * np.abs(h_values))

        # Calculate effective trust radius: cfl_target / cfl_current
        if CFL_energy > 1e-3:
            trust_radius = cfl_target / CFL_energy
            print(f"  TR-CFL: Effective trust_radius = {cfl_target:.2f} / {cfl_current:.4f} = {trust_radius:.4e}")
        else:
            trust_radius = 1.0
            print(f"  TR-CFL: Low CFL, using trust_radius = 1.0")

        # Compute pure Newton step
        pk_newton = self._solve_linear_system_core()
        self.postprocessing_overshoots(pk_newton)

        # Get DOF indices for each variable
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])

        # Compute norms for each block
        p_step_norm = np.linalg.norm(pk_newton[p_dof_idx])
        h_step_norm = np.linalg.norm(pk_newton[h_dof_idx])
        z_step_norm = np.linalg.norm(pk_newton[z_dof_idx])
        hyperbolic_step_norm = np.sqrt(h_step_norm**2 + z_step_norm**2)

        print(f"  TR: ||Δp||={p_step_norm:.2e}, ||Δh||={h_step_norm:.2e}, ||Δz||={z_step_norm:.2e}")
        print(f"  TR: ||Δ_hyperbolic||={hyperbolic_step_norm:.2e}, trust_radius={trust_radius:.2e}")

        # # Trust parabolic (pressure), limit hyperbolic (enthalpy, composition)
        pk_solution = pk_newton.copy()
        residual_full_vec = self.compute_residual_from_increment(pk_newton, restore_state=True)
        residual_norm_full_vec = np.linalg.norm(residual_full_vec)

        #
        # if hyperbolic_step_norm > trust_radius:
        #     # Scale back ONLY the hyperbolic components
        #     scaling_factor = trust_radius / hyperbolic_step_norm
        #     pk_solution[h_dof_idx] *= scaling_factor
        #     pk_solution[z_dof_idx] *= scaling_factor
        #     print(f"  TR: Scaling hyperbolic by {scaling_factor:.4f} (CFL limit)")
        #     print(f"  TR: Pressure step UNTOUCHED (parabolic)")
        # else:
        #     print(f"  TR: Full Newton step (hyperbolic within CFL-based radius)")

        pk_solution *= trust_radius
        # Evaluate step quality
        residual_new_vec = self.compute_residual_from_increment(pk_solution, restore_state=True)
        residual_norm_new = np.linalg.norm(residual_new_vec)

        print(f"  TR: ||R_full_step||={residual_norm_full_vec:.4e}, ||R_new||={residual_norm_new:.4e}")

        # Accept step if residual decreased or near convergence
        accept_step = residual_norm_new <  residual_norm_full_vec

        if accept_step:
            print(f"  TR: ✓ ACCEPTING")
        else:
            print(f"  TR: ✗ REJECTING")
            pk_solution = np.zeros_like(pk_solution)

        # Return solution and new trust radius (recalculate next iteration)
        return pk_solution, trust_radius

    def backtracking_line_search(
        self,
        delta_x: np.ndarray,
        current_residual: np.ndarray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        max_iterations: int = 25,
        alpha_min: float = 0.01,  # Minimum acceptable step length
    ) -> float:
        """
        Backtracking line search with Armijo condition (robust, Jacobian-predicted).

        Uses the Jacobian to cheaply predict residuals for candidate alphas and
        only performs the expensive full residual assembly + postprocessing for
        promising alphas. If the Jacobian-based prediction rejects many alphas
        in a row, a full evaluation is forced after a configurable threshold to
        avoid never confirming a valid step.
        """

        # Basic metrics
        residual_norm_current = np.linalg.norm(current_residual)
        phi_current = 0.5 * (residual_norm_current ** 2)

        alpha = alpha_init
        best_alpha: Optional[float] = None
        best_residual = np.inf

        # Parameters
        c_armijo = self.params.get("line_search_armijo", 1e-3)
        use_jacobian_prediction = self.params.get("line_search_use_jacobian_prediction", True)
        force_full_after = int(self.params.get("line_search_force_full_after_predicted_rejects", 3))

        # Try to obtain Jacobian and a J @ delta_x product for cheap prediction
        jacobian_matrix = None
        Jp_full = None
        if use_jacobian_prediction:
            try:
                jacobian_matrix, _ = self.linear_system
                # Compute once (may fail for some sparse/delayed matrix types)
                try:
                    Jp_full = jacobian_matrix @ delta_x
                except Exception:
                    Jp_full = None
            except Exception:
                jacobian_matrix = None
                Jp_full = None

        # Counters and bookkeeping
        full_evals = 0
        predicted_rejects = 0
        tried_alphas: list[float] = []

        # enforce strictly positive alpha_min
        alpha_min = max(alpha_min, 1e-12)

        # Generate alpha sequence deterministically to avoid floating underflow to zero
        for i in range(max_iterations):
            # deterministic alpha sequence: alpha_i = alpha_init * rho**i
            alpha_i = alpha_init * (rho ** i)
            # clamp to alpha_min if below
            if alpha_i < alpha_min:
                alpha = float(alpha_min)
                last_alpha = True
            else:
                alpha = float(alpha_i)
                last_alpha = False

            tried_alphas.append(alpha)
            scaled_increment = alpha * delta_x

            # Jacobian-based cheap prediction
            predicted_ok = False
            if (Jp_full is not None) and use_jacobian_prediction:
                r_pred = current_residual + alpha * Jp_full
                dphi_pred = float(np.dot(current_residual, Jp_full))
                if dphi_pred < 0.0:
                    phi_pred = 0.5 * (np.linalg.norm(r_pred) ** 2)
                    predicted_ok = phi_pred <= phi_current + c_armijo * alpha * dphi_pred
                else:
                    predicted_ok = np.linalg.norm(r_pred) < residual_norm_current

            # Decide whether to run a full expensive residual assembly
            force_full = (predicted_rejects >= force_full_after) and use_jacobian_prediction
            do_full_eval = (Jp_full is None) or predicted_ok or (not use_jacobian_prediction) or force_full

            if not do_full_eval:
                predicted_rejects += 1
                print(f"  Line search iter {i+1}: alpha={alpha:.4f} rejected by Jacobian prediction")
                # If this was the last allowable alpha, break to finalization
                if last_alpha:
                    break
                continue

            # Full evaluation (may be expensive)
            try:
                # Raw residual after applying the scaled increment
                residual_trial = self.compute_residual_from_increment(scaled_increment, restore_state=True)

                # If residual contains non-finite values, treat as failed eval
                if not np.all(np.isfinite(residual_trial)):
                    print(f"  Line search iter {i+1}: non-finite residual from raw eval (alpha={alpha:.4f}), skipping")
                    predicted_rejects = 0
                    if last_alpha:
                        break
                    continue

                # Category breakdown and per-cell exceedences
                _, _, diff_norm_trial, alg_norm_trial, alg_exceeds_trial = self.compute_residuals_by_category(
                    residual_trial
                )

                # Apply same postprocessing that will be applied to accepted steps
                trial_increment_post = scaled_increment.copy()
                try:
                    self.postprocessing_overshoots(trial_increment_post)
                except Exception:
                    pass
                if diff_norm_trial < alg_norm_trial:
                    try:
                        self.postprocessing_thermal_overshoots(trial_increment_post, alg_exceeds_trial)
                    except Exception:
                        pass

                # Residual after postprocessing
                residual_after_post = self.compute_residual_from_increment(trial_increment_post, restore_state=True)

                # If residual after postprocessing contains non-finite values, treat as failed eval
                if not np.all(np.isfinite(residual_after_post)):
                    print(f"  Line search iter {i+1}: non-finite residual after postprocessing (alpha={alpha:.4f}), skipping")
                    predicted_rejects = 0
                    if last_alpha:
                        break
                    continue

                # Norm of the residual after postprocessing
                residual_norm_new = float(np.linalg.norm(residual_after_post))

                # Update counters and best-known (only accept alphas >= alpha_min)
                full_evals += 1
                if (residual_norm_new < best_residual) and (alpha >= alpha_min):
                    best_residual = residual_norm_new
                    best_alpha = float(alpha)

                # Compute directional derivative dphi using Jacobian (if available)
                if Jp_full is not None:
                    Jp_for_dphi = Jp_full
                elif jacobian_matrix is not None:
                    try:
                        Jp_for_dphi = jacobian_matrix @ delta_x
                    except Exception:
                        Jp_for_dphi = None
                else:
                    Jp_for_dphi = None

                dphi = float(np.dot(current_residual, Jp_for_dphi)) if (Jp_for_dphi is not None) else 0.0
                phi_new = 0.5 * (residual_norm_new ** 2)

                # Armijo acceptance
                accepted = False
                if (Jp_for_dphi is not None) and (dphi < 0.0):
                    if phi_new <= phi_current + c_armijo * alpha * dphi:
                        accepted = True
                else:
                    # fallback: require strict residual norm decrease
                    if residual_norm_new < residual_norm_current:
                        accepted = True

                if accepted:
                    reduction_factor = residual_norm_new / residual_norm_current if residual_norm_current > 0 else 0.0
                    print(
                        f"  Line search iter {i+1}: alpha={alpha:.4f}, ||r||={residual_norm_new:.4e} (accepted, factor: {reduction_factor:.4f})"
                    )
                    return alpha

                print(
                    f"  Line search iter {i+1}: alpha={alpha:.4f}, ||r||={residual_norm_new:.4e} (rejected, factor: {residual_norm_new/residual_norm_current:.4f})"
                )

                # reset predicted_rejects if we performed a full eval
                predicted_rejects = 0

            except Exception as e:
                print(f"  Line search iter {i+1}: full evaluation failed at alpha={alpha:.4f}: {e}")

        # End loop: if we never performed any full evals, try a final full eval at the initial alpha
        if full_evals == 0:
            final_alpha = tried_alphas[0] if len(tried_alphas) > 0 else alpha_init
            final_alpha = max(alpha_min, final_alpha)
            print(f"  Line search: no full evals performed; doing final full eval at alpha={final_alpha:.4e}")
            try:
                scaled_increment = final_alpha * delta_x
                residual_trial = self.compute_residual_from_increment(scaled_increment, restore_state=True)
                _, _, diff_norm_trial, alg_norm_trial, alg_exceeds_trial = self.compute_residuals_by_category(
                    residual_trial
                )
                trial_increment_post = scaled_increment.copy()
                try:
                    self.postprocessing_overshoots(trial_increment_post)
                except Exception:
                    pass
                if diff_norm_trial < alg_norm_trial:
                    try:
                        self.postprocessing_thermal_overshoots(trial_increment_post, alg_exceeds_trial)
                    except Exception:
                        pass
                residual_after_post = self.compute_residual_from_increment(trial_increment_post, restore_state=True)
                residual_norm_new = float(np.linalg.norm(residual_after_post))
                if residual_norm_new < residual_norm_current:
                    print(f"  Line search: final eval accepted alpha={final_alpha:.4e}, ||r||={residual_norm_new:.4e}")
                    return final_alpha
                else:
                    print(f"  Line search: final eval rejected; falling back to alpha={alpha_min:.4e}")
                    return alpha_min
            except Exception as e:
                print(f"  Line search: final full eval failed: {e}; falling back to alpha_min={alpha_min:.4e}")
                return alpha_min

        # We have at least one confirmed full evaluation; return best confirmed alpha or fallback
        print(f"  Line search summary: full_evals={full_evals}, predicted_rejects={predicted_rejects}, tried_alphas={len(tried_alphas)}")
        if (best_alpha is not None) and np.isfinite(best_residual) and (best_alpha >= alpha_min):
            print(f"  Line search: using best alpha={best_alpha:.4f} with ||r||={best_residual:.4e}")
            return best_alpha
        # No valid best found above alpha_min -> fallback to conservative minimum
        print(f"  Line search: no good confirmed step found, using fallback alpha={alpha_min:.4f}")
        return float(alpha_min)

    # ----------------------------------------------------------------------------------
    #  Overridable hooks used by the step control above. Base implementations are generic
    #  / no-ops; models with a differential-vs-algebraic structure or variable clamping
    #  (e.g. the Driesner brine model) override them.
    # ----------------------------------------------------------------------------------
    def compute_residuals_by_category(self, residual):
        """Split the residual into differential vs algebraic categories.

        Base default: a single ``"overall"`` differential category (algebraic norm = 0,
        no per-cell exceedances), which is enough for the generic line search. Override
        to expose a physics-specific breakdown.

        Returns:
            ``(diff_residuals, alg_residuals, diff_norm, alg_norm, alg_exceeds)``.
        """
        norm = float(np.linalg.norm(residual))
        return {"overall": norm}, {}, norm, 0.0, {}

    def postprocessing_overshoots(self, delta_x):
        """Clamp/limit overshoots in the increment. No-op by default; override per model."""

    def postprocessing_thermal_overshoots(self, delta_x, alg_exceeds):
        """Clamp thermal/algebraic overshoots. No-op by default; override per model."""

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def lag_buoyancy_direction(self) -> bool:
        """Whether to freeze the buoyancy upwind direction over each time step.

        When ``params["lag_buoyancy_direction"]`` is True (default False) the buoyancy
        upwind direction -- the hybrid inter-phase gravity flux (HU) or the per-phase
        phase potentials (PPU) -- is evaluated once per time step from the previous
        converged state (in :meth:`before_time_step`) and held fixed through the step's
        Newton iterations, instead of being refreshed every iteration. This follows Weis
        et al. (2014, Geofluids 14:347-371, p.353), who use the old velocity field to
        define the upwind nodes for the whole step (cheaper, no visible effect on the
        results, and it removes the upwind-direction flip-flop at flow reversal). The
        option applies to BOTH the hybrid and phase-potential schemes.
        """
        return bool(self.params.get("lag_buoyancy_direction", False))

    def refresh_buoyancy_direction(self) -> None:
        """Per-iteration refresh of the buoyancy upwind direction, unless it is lagged."""
        if not self.lag_buoyancy_direction():
            self.update_buoyancy_driven_fluxes()

    def before_time_step(self) -> None:
        super().before_time_step()
        # Lagged scheme: freeze the buoyancy upwind direction at the previous converged
        # state (now the current iterate) for the whole time step, then rediscretize so
        # the frozen direction is in place before the first nonlinear assembly.
        if self.lag_buoyancy_direction():
            self.update_buoyancy_driven_fluxes()
            self.rediscretize()

    def  after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        self.refresh_buoyancy_direction()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        # ``params["gravity"]=False`` (or 0) sets g=0 -- removes BOTH the buoyant phase
        # segregation and the hydrostatic term from the Darcy flux (gravity-free flow).
        g_constant = pp.GRAVITY_ACCELERATION if self.params.get("gravity", True) else 0.0
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def _get_non_reference_component(self) -> str | None:
        """
        Return the non-reference component name for binary mixtures.

        Convention: self.fluid.components is expected to be an indexable sequence
        where the reference component is at index 0 and the other (active)
        component is at index 1. If there are at least two components, return
        the second one. Otherwise return None.
        """
        return self.get_components()[1].name

    def _local_elimination_pairs(self, equation_keys) -> list[tuple[str, str]]:
        """``(variable, equation)`` for every local-elimination (algebraic) equation.

        Discovered from the ``elimination_of_<variable>_on_grids_...`` naming, so this
        captures the eliminated saturations, partial fractions and temperature for ANY
        number of phases/components -- no variable names are hardcoded. For example
        ``elimination_of_x_CH4_gas_on_grids_[0]`` yields the variable ``x_CH4_gas``.
        """
        prefix = "elimination_of_"
        pairs: list[tuple[str, str]] = []
        for equation in equation_keys:
            if equation.startswith(prefix):
                variable = equation[len(prefix):].rsplit("_on_grids", 1)[0]
                pairs.append((variable, equation))
        return pairs

    def permute_equations_and_variables(self):
        """Reorder the (equation, variable) dofs into a CPR-friendly two-field split.

        * **elliptic / pressure field** -- pressure, the interface Darcy flux, and all
          local algebraic eliminations (eliminated saturations, partial fractions and
          temperature); solved (near-)exactly inside the CPR preconditioner.
        * **transport field** -- enthalpy, the thermal interface fluxes, and one overall
          fraction ``z_<comp>`` per NON-reference component.

        The split is derived from the fluid mixture and the equation system, so it works
        unchanged for two-phase/two-component, three-phase/three-component, and beyond --
        no variable or equation names are hardcoded.

        Returns:
            ``(equation_permutation, variable_permutation, field_sizes)`` where the
            permutations are index arrays and ``field_sizes`` is
            ``{'elliptic': n_e, 'transport': n_t}``.
        """
        assembled = self.equation_system.assembled_equation_indices
        equation_keys = list(assembled.keys())

        def equation_named(keyword: str, exclude: str | None = None) -> str | None:
            """First assembled equation whose key contains ``keyword`` (and not
            ``exclude``). ``exclude`` disambiguates the global 'mass_balance_equation'
            from the per-component 'component_mass_balance_equation_*'."""
            return next(
                (eq for eq in equation_keys
                 if keyword in eq and (exclude is None or exclude not in eq)),
                None,
            )

        def variable_dofs(name: str):
            domains = self.mdg.interfaces() if "interface" in name else self.mdg.subdomains()
            md_var = self.equation_system.md_variable(name, domains)
            return self.equation_system.dofs_of(md_var.sub_vars)

        # One overall-fraction variable + component balance per NON-reference component
        # (the reference component, index 0, is fixed by the unity closure).
        component_pairs = [
            (f"z_{c.name}", equation_named(f"component_mass_balance_equation_{c.name}"))
            for c in self.get_components()[1:]
        ]

        # (variable, equation) pairs, in permutation order, for each field.
        elliptic_pairs = [
            ("pressure", equation_named("mass_balance_equation", exclude="component")),
            ("interface_darcy_flux", equation_named("interface_darcy_flux")),
            *self._local_elimination_pairs(equation_keys),
        ]
        transport_pairs = [
            ("enthalpy", equation_named("energy_balance_equation")),
            ("interface_enthalpy_flux", equation_named("interface_enthalpy_flux")),
            ("interface_fourier_flux", equation_named("interface_fourier_flux")),
            *component_pairs,
        ]

        def collect(pairs):
            equation_idx: list[int] = []
            variable_idx: list[int] = []
            for variable, equation in pairs:
                if equation is None or equation not in assembled:
                    continue
                rows = assembled[equation]
                dofs = variable_dofs(variable)
                assert len(rows) == len(dofs), (
                    f"{variable!r}: {len(rows)} equation rows vs {len(dofs)} variable dofs"
                )
                equation_idx.extend(rows)
                variable_idx.extend(dofs)
            return equation_idx, variable_idx

        elliptic_eq, elliptic_var = collect(elliptic_pairs)
        transport_eq, transport_var = collect(transport_pairs)
        return (
            np.array(elliptic_eq + transport_eq),
            np.array(elliptic_var + transport_var),
            {"elliptic": len(elliptic_eq), "transport": len(transport_eq)},
        )

    def apply_equation_permutation(self, A: sps.spmatrix, b: np.ndarray) -> tuple[sps.spmatrix, np.ndarray, np.ndarray | None, np.ndarray | None, dict | None]:
        """
        Apply equation and variable permutation to the linear system.

        Args:
            A: Jacobian matrix
            b: Right-hand side vector

        Returns:
            tuple: (permuted_A, permuted_b, equation_permutation, variable_permutation)
        """
        try:
            eq_perm, var_perm, field_split = self.permute_equations_and_variables()

            # Permute rows (equations) and columns (variables) of the matrix
            A_permuted = A[eq_perm, :][:, var_perm]

            # Permute the right-hand side vector
            b_permuted = b[eq_perm]

            logger.info(f"Applied equation permutation: {len(eq_perm)} equations, {len(var_perm)} variables")

            return A_permuted, b_permuted, eq_perm, var_perm, field_split

        except Exception as e:
            logger.warning(f"Failed to apply equation permutation: {e}. Using original ordering.")
            return A, b, None, None, None

    def assemble_linear_system(self) -> None:
        """Custom assemble linear system that updates Jacobian every 0, 3, 6, 9... Newton iterations.

        This method implements a dedicated solution strategy that:
        - Assembles the full linear system (Jacobian + residual) at iterations 0, 3, 6, 9, etc.
        - Updates only the residual part for other iterations (1, 2, 4, 5, 7, 8, etc.)
        """
        t_0 = time.time()

        # Get current Newton iteration number
        iteration_num = self.nonlinear_solver_statistics.num_iterations

        if iteration_num % 2 == 0 or iteration_num < 10:
            # Update both Jacobian and residual at iterations 0, 3, 6, 9, ...
            logger.info(f"Newton iteration {iteration_num}: Updating Jacobian and residual")
            self.linear_system = self.equation_system.assemble(evaluate_jacobian=True)
        else:
            # Update only residual at iterations 1, 2, 4, 5, 7, 8, ...
            logger.info(f"Newton iteration {iteration_num}: Updating residual only")
            if hasattr(self, 'linear_system') and self.linear_system is not None:
                # Keep the existing Jacobian, update only the residual
                new_residual = self.equation_system.assemble(evaluate_jacobian=False)
                # Update the residual part of the linear system (tuple format: (matrix, rhs))
                self.linear_system = (
                    self.linear_system[0],  # Keep existing Jacobian
                    -new_residual  # Update residual with new evaluation
                )
            else:
                # Fallback: if no previous linear system exists, assemble full system
                logger.warning("No previous linear system found, assembling full system")
                if self._apply_schur_complement_reduction():
                    assert self.schur_complement_primary_variables, (
                        "Primary column block for Schur technique not found."
                    )
                    assert self.schur_complement_primary_equations, (
                        "Primary row block for Schur technique not defined."
                    )
                    self.linear_system = self.equation_system.assemble_schur_complement_system(
                        self.schur_complement_primary_equations,
                        self.schur_complement_primary_variables,
                        inverter=cast(
                            Callable[[sps.spmatrix], sps.spmatrix],
                            self.params.get("schur_complement_inverter", None),
                        ),
                    )
                else:
                    self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        mode = (
            "Jacobian + residual"
            if (iteration_num % 2 == 0 or iteration_num < 10)
            else "residual only"
        )
        logger.info(f"Assembled {mode} in {t_1 - t_0:.2e} seconds.")

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Track Newton iterations for this timestep
        current_iterations = self.nonlinear_solver_statistics.num_iterations
        self.newton_iterations_per_timestep.append(current_iterations)
        self.total_newton_iterations += current_iterations

        # Print Newton iteration info for current timestep
        current_time = self.time_manager.time
        timestep_number = self.time_manager.time_index
        logger.info(f"Timestep {timestep_number} (t={current_time:.2e}): {current_iterations} Newton iterations")

        print("*" * 60)
        day_to_second = 86400
        second_to_year = 1.0 / (365 * day_to_second)
        super().after_nonlinear_convergence()  # type:ignore[safe-super]
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iterations)
        print("Time value (year): ", self.time_manager.time * second_to_year)
        print("Time index: ", self.time_manager.time_index)
        print("*" * 60)
        print("")

    def after_nonlinear_failure(self) -> None:
        """Count a rejected nonlinear loop (a time-step cut) before deferring to the template."""
        self.n_time_step_cuts = getattr(self, "n_time_step_cuts", 0) + 1
        super().after_nonlinear_failure()

    def collect_run_stats(self) -> NonlinearRunStats:
        """Return a picklable :class:`NonlinearRunStats` snapshot of the run.

        Any model deriving from this base gets the feature for free; call it after the time loop
        (e.g. in ``after_simulation`` or right after ``run_time_dependent_model``) to persist or
        inspect the solver behaviour without touching PorePy's non-picklable statistics object."""
        hist = list(self.newton_iterations_per_timestep)
        return NonlinearRunStats(
            n_accepted_steps=len(hist),
            n_time_step_cuts=getattr(self, "n_time_step_cuts", 0),
            total_newton_iterations=int(self.total_newton_iterations),
            max_newton_iterations=max(hist) if hist else 0,
            iterations_per_step=hist,
        )

    def dof_summary(self) -> DofSummary:
        """Return a :class:`DofSummary` of the current equation system.

        Available to every derived model.  Reports the cells per subdomain dimension and, per
        variable, its total dof count and PRIMARY/SECONDARY type -- SECONDARY meaning locally
        eliminated (algebraic), discovered from the ``elimination_of_<var>_on_grids_...`` equation
        names, so no variable names are hardcoded.  Requires the equation system to be set up (call
        after ``prepare_simulation``)."""
        es = self.equation_system
        mdg = self.mdg

        # Cells per subdomain, grouped by dimension: dim -> (n_subdomains, total cells).
        cells_per_dim: dict[int, tuple[int, int]] = {}
        for d in range(mdg.dim_max() + 1):
            sds = mdg.subdomains(dim=d)
            if sds:
                cells_per_dim[d] = (len(sds), int(sum(sd.num_cells for sd in sds)))

        # Locally-eliminated (secondary) variable names, from the elimination equations.
        prefix = "elimination_of_"
        secondary = {
            name[len(prefix):].rsplit("_on_grids", 1)[0]
            for name in es.equations if name.startswith(prefix)
        }

        # Total dof per variable NAME (summed over its grids), preserving first-seen order.
        vars_by_name: dict[str, list] = {}
        for var in es.variables:
            vars_by_name.setdefault(var.name, []).append(var)
        variables = [
            (name, int(es.dofs_of(vs).size),
             "secondary" if name in secondary else "primary")
            for name, vs in vars_by_name.items()
        ]

        return DofSummary(
            n_dofs=int(es.num_dofs()),
            n_subdomains=mdg.num_subdomains(),
            n_interfaces=mdg.num_interfaces(),
            cells_per_dim=cells_per_dim,
            variables=variables,
        )

    def report_dof_summary(self, label: str = "") -> DofSummary:
        """Build and print the :class:`DofSummary`; also returns it for further use."""
        summary = self.dof_summary()
        header = f" DoF summary{(' -- ' + label) if label else ''} "
        print("\n" + header.center(64, "=") + "\n" + summary.as_text(), flush=True)
        return summary

    def save_run_statistics(self, filename: str = "run_statistics") -> str | None:
        """Persist the run's DoF + nonlinear-solver statistics to the output folder.

        Writes ``<folder>/<filename>.txt`` (human-readable: the :class:`DofSummary` and
        :class:`NonlinearRunStats` renderings, plus the transport-predictor cost when it ran) and
        ``<filename>.json`` (the same data structured for downstream tabulation, with the derived
        averages included).  ``<folder>`` is ``params['folder_name']`` -- the SAME directory the VTU
        exporter writes to -- so the statistics live beside the visualization.  No-op (returns
        ``None``) if no output folder is configured.  Available to every derived model."""
        folder = self.params.get("folder_name")
        if not folder:
            return None
        os.makedirs(folder, exist_ok=True)

        dof = self.dof_summary()
        stats = self.collect_run_stats()
        # Curated json-safe subset of the run configuration (skip time managers, tensors, etc.).
        config = {k: v for k, v in self.params.items()
                  if isinstance(v, (str, int, float, bool)) or v is None}
        predictor = None
        if getattr(self, "_predictor_cum_time", 0.0):
            predictor = {"cumulative_seconds": round(self._predictor_cum_time, 4),
                         "n_sweeps": int(getattr(self, "_predictor_n_calls", 0))}

        txt_path = os.path.join(folder, filename + ".txt")
        with open(txt_path, "w") as fh:
            fh.write(dof.as_text())
            fh.write("\n")
            fh.write(stats.as_text())
            if predictor:
                fh.write(f"\n# transport predictor: {predictor['cumulative_seconds']} s "
                         f"over {predictor['n_sweeps']} sweeps\n")

        dof_json = asdict(dof)
        dof_json.update(n_primary_dofs=dof.n_primary_dofs, n_secondary_dofs=dof.n_secondary_dofs)
        stats_json = asdict(stats)
        stats_json["avg_newton_iterations"] = stats.avg_newton_iterations
        payload = {"config": config, "dof_summary": dof_json, "run_stats": stats_json}
        if predictor:
            payload["transport_predictor"] = predictor
        with open(os.path.join(folder, filename + ".json"), "w") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("run statistics -> %s (+ .json)", txt_path)
        return txt_path

    def prepare_simulation(self) -> None:
        """Set up the model, then report the initial DoF summary."""
        super().prepare_simulation()
        self.report_dof_summary("initial")

    def after_simulation(self) -> None:
        """Report the final DoF summary and persist the run statistics to the output folder."""
        super().after_simulation()
        self.report_dof_summary("final")
        self.save_run_statistics()

    def write_newton_iterations_to_csv(self, filename="newton_iterations.csv"):
        """Write Newton iteration data to CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Timestep', 'Time', 'Newton_Iterations'])

            # Write data for each timestep
            for i, iterations in enumerate(self.newton_iterations_per_timestep):
                timestep_number = i + 1
                # Calculate time value based on time manager
                time_value = self.time_manager.schedule[0] + (timestep_number * self.time_manager.dt_init)
                writer.writerow([timestep_number, f"{time_value:.6e}", iterations])

            # Write summary row
            writer.writerow(['', '', ''])
            writer.writerow(['SUMMARY', '', ''])
            writer.writerow(['Total_Timesteps', len(self.newton_iterations_per_timestep), ''])
            writer.writerow(['Total_Newton_Iterations', self.total_newton_iterations, ''])
            if self.newton_iterations_per_timestep:
                avg_iterations = self.total_newton_iterations / len(self.newton_iterations_per_timestep)
                writer.writerow(['Average_Iterations_Per_Timestep', f"{avg_iterations:.2f}", ''])
                writer.writerow(['Max_Iterations', max(self.newton_iterations_per_timestep), ''])
                writer.writerow(['Min_Iterations', min(self.newton_iterations_per_timestep), ''])

        print(f"Newton iteration data written to {filename}")
        print(f"Total Newton iterations: {self.total_newton_iterations}")
        print(f"Total timesteps: {len(self.newton_iterations_per_timestep)}")
        avg_iterations = 0.0
        if self.newton_iterations_per_timestep:
            avg_iterations = self.total_newton_iterations / len(self.newton_iterations_per_timestep)
        print(f"Average iterations per timestep: {avg_iterations:.2f}")


class FlowModelBase(_FlowModelBaseCore, CompositionalFlowTemplate):
    """Flow-model base with the STANDARD primary equations (upwinded total mobility) -- the HU
    discretisation. Public name kept for backward compatibility; other example scripts inherit it."""


class FractionalFlowModelBase(_FlowModelBaseCore, CompositionalFractionalFlowTemplate):
    """Flow-model base with the FRACTIONAL-FLOW primary equations (mobility-weighted) -- the HU-mw
    discretisation. Select this template for ``mass_mobility_weighted_permeability``/HU-mw runs."""
