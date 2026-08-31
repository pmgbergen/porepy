from __future__ import annotations
import logging
import time
import csv
import os
import json
import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, cast, Any

import porepy as pp
import porepy.compositional as pc
from porepy.models.compositional_flow import (
    CompositionalFlowTemplate,
    CompositionalFractionalFlowTemplate,
    update_phase_properties,
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


def _slice_phase_properties(
    props: pc.PhaseProperties, sl: slice, n_total: int
) -> pc.PhaseProperties:
    """Return a view of ``props`` restricted to the cells in ``sl``.

    Every stored array whose last axis spans the full cell count ``n_total`` (values along
    ``(N,)``, derivatives along ``(..., N)``) is sliced along that axis; scalar fields and the
    empty defaults are left untouched. Used to scatter a batched (all-subdomain) property
    computation back to a single grid, so the ``*_ext`` chain-rule properties -- computed on
    demand from the sliced ``x`` / derivatives -- stay consistent per grid.
    """
    kw = {}
    for f in dataclasses.fields(props):
        v = getattr(props, f.name)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[-1] == n_total:
            kw[f.name] = v[..., sl]
    return dataclasses.replace(props, **kw)


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
class StepTiming:
    """Per-accepted-step cost breakdown [ms] and counts, filled live by :class:`_FlowModelBaseCore`.

    The five ``t_*`` buckets cover one time step's whole Newton loop (summed over its iterations):
    ``before``/``after`` = the before_/after_nonlinear_iteration hooks (rediscretize + buoyancy /
    the flash update_derived_quantities); ``assembly`` = Jacobian+residual; ``linear`` = the reduced
    linear solve; ``linesearch`` = the weis backtracking (residual re-evaluations + clip)."""

    newton_iterations: int = 0
    n_line_searches: int = 0      # total backtracking trials (residual re-evals) over the step
    n_cuts: int = 0               # dt-cuts (failed attempts) preceding this accepted step
    t_before_ms: float = 0.0
    t_assembly_ms: float = 0.0
    t_linear_ms: float = 0.0
    t_linesearch_ms: float = 0.0
    t_after_ms: float = 0.0
    wall_ms: float = 0.0          # total wall time of the accepted step's Newton loop
    dt_begin_s: float = 0.0       # time-step size [s] at the start of the step
    dt_end_s: float = 0.0         # time-step size [s] recorded at the end of the step

    @property
    def t_total_ms(self) -> float:
        return (self.t_before_ms + self.t_assembly_ms + self.t_linear_ms
                + self.t_linesearch_ms + self.t_after_ms)


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
    step_timings: list[StepTiming] = field(default_factory=list)
    """Per-accepted-step cost breakdown (:class:`StepTiming`), in solve order."""
    t_cut_ms: float = 0.0
    """Wall time [ms] burned in rejected (dt-cut) Newton loops (not tied to an accepted step)."""

    @property
    def avg_newton_iterations(self) -> float:
        """Mean Newton iterations per accepted step (0.0 if no steps were accepted)."""
        return self.total_newton_iterations / self.n_accepted_steps if self.n_accepted_steps else 0.0

    def _timing_totals_ms(self) -> dict[str, float]:
        """Sum each cost bucket [ms] over the accepted steps."""
        keys = ("t_before_ms", "t_assembly_ms", "t_linear_ms", "t_linesearch_ms", "t_after_ms")
        tot = {k: sum(getattr(s, k) for s in self.step_timings) for k in keys}
        tot["t_total_ms"] = sum(tot.values())
        return tot

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

        if self.step_timings:
            t = self._timing_totals_ms()
            tot = t["t_total_ms"] or 1.0
            wall_total = sum(s.wall_ms for s in self.step_timings)

            def _pct(x):  # noqa: E306
                return f"{100.0 * x / tot:4.1f}%"
            lines += [
                "",
                "# cost breakdown over accepted steps [s]  (fraction of the summed step cost)",
                f"before_newton     {t['t_before_ms'] / 1e3:10.2f}   {_pct(t['t_before_ms'])}",
                f"assembly          {t['t_assembly_ms'] / 1e3:10.2f}   {_pct(t['t_assembly_ms'])}",
                f"linear_solver     {t['t_linear_ms'] / 1e3:10.2f}   {_pct(t['t_linear_ms'])}",
                f"line_search       {t['t_linesearch_ms'] / 1e3:10.2f}   {_pct(t['t_linesearch_ms'])}",
                f"after_newton      {t['t_after_ms'] / 1e3:10.2f}   {_pct(t['t_after_ms'])}",
                f"accepted_total    {t['t_total_ms'] / 1e3:10.2f}",
                f"accepted_wall     {wall_total / 1e3:10.2f}   (measured step wall; overhead = "
                f"wall - buckets)",
                f"cut_loops_wasted  {self.t_cut_ms / 1e3:10.2f}   (time in rejected dt-cut loops)",
                "",
                "# per-step: cost [ms] (before assembly linear line_search after) | wall [ms] | "
                "dt_beg=this step's dt, dt_end=next step's dt [yr] | it, n_ls trials, n_cut",
                "# step  before assembly   linear linesrch    after      wall     dt_beg    dt_end"
                "   it n_ls n_cut",
            ]
            _YR = 365.0 * 86400.0
            for i, s in enumerate(self.step_timings):
                lines.append(
                    f"{i:<6d} {s.t_before_ms:6.1f} {s.t_assembly_ms:8.1f} {s.t_linear_ms:8.1f} "
                    f"{s.t_linesearch_ms:8.1f} {s.t_after_ms:8.1f} {s.wall_ms:9.1f} "
                    f"{s.dt_begin_s / _YR:9.4g} {s.dt_end_s / _YR:9.4g} {s.newton_iterations:4d} "
                    f"{s.n_line_searches:4d} {s.n_cuts:4d}")
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


class GeothermalLinearSolver(pp.solvers.LinearSolverBase):
    """Adapter routing the new solver stack's linear solves through the MODEL's own solve.

    The current ``pp.solvers.NewtonSolver`` delegates linear solves to a ``LinearSolverBase``
    object and never calls ``model.solve_linear_system()`` -- which is where all the geothermal
    machinery lives (Schur-CPR, PETSc LU/MUMPS, the 2D solver's bordered Lagrange solve).  This
    adapter restores that dispatch: it stores the assembled system where the model's methods
    expect it (``model.linear_system`` as a (matrix, rhs) tuple) and calls the model's
    ``solve_linear_system()``, so every existing override keeps working unchanged.

    Use by passing to the Newton solver, e.g.::

        solver = pp.solvers.NewtonSolver(params=solver_params,
                                         linear_solver=GeothermalLinearSolver())
        runner = pp.ModelRunner(model, solver_params, nonlinear_solver=solver)
    """

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        self._model = model

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        t0 = time.time()
        model = self._model
        model.linear_system = (linear_system.matrix, linear_system.rhs)
        try:
            x = np.asarray(model.solve_linear_system(), dtype=float)
        except Exception as exc:
            logger.warning("model linear solve failed; zero increment returned")
            logger.warning("  reason: %r", exc)
            return (np.zeros_like(linear_system.rhs),
                    pp.solvers.LinearSolverStatusFailure(reason=str(exc)))
        return x, pp.solvers.LinearSolverStatusSuccess(solve_time=time.time() - t0)


def geothermal_nonlinear_solver(solver_params: dict) -> "pp.solvers.NewtonSolver":
    """A NewtonSolver wired to the model-dispatching :class:`GeothermalLinearSolver`.

    Every subsection script builds its runner as
    ``pp.ModelRunner(model, solver_params, nonlinear_solver=geothermal_nonlinear_solver(solver_params))``.
    """
    return pp.solvers.NewtonSolver(
        params=solver_params, linear_solver=GeothermalLinearSolver())


class RelativeStorageLebesgueMetric(pp.EquationBasedLebesgueMetric):
    """Per-equation Lebesgue metric with each PHYSICAL row divided by its storage/throughput scale --
    the same ms/es row-normalization the weis reference uses. Turns PorePy's ABSOLUTE residual bound
    into a RELATIVE one: ``tol`` then means "imbalance relative to the cell's stored fluid mass
    (mass/component rows) or rock heat (energy row) per step", identical to weis, instead of an
    absolute value in unit-scaled kg/s, MJ/s. Scales come from ``model.residual_row_scales()``;
    equations with no scale (the eliminated-secondary closures, ~0 under the slave) stay unscaled."""

    def __call__(self, values: np.ndarray) -> dict:
        norms = super().__call__(values)                       # absolute Lebesgue L2 per equation
        scales = self.model.residual_row_scales()
        return {name: (v / scales[name] if scales.get(name, 0.0) > 0.0 else v)
                for name, v in norms.items()}


class _FlowModelBaseCore(ReorderedTransportPredictor):
    """Template-agnostic core of the flow model (all solver/discretisation logic). It is combined
    with one of the two compositional-flow templates below to form a concrete base; its ``super()``
    calls resolve to whichever template is mixed in after it in the concrete class's MRO."""

    def __init__(self, params):
        super().__init__(params)
        self.newton_iterations_per_timestep = []
        self.total_newton_iterations = 0
        # Rejected nonlinear loops (time-step cuts); incremented in after_nonlinear_failure.
        self.n_time_step_cuts = 0
        # Per-step cost instrumentation (see StepTiming): the live accumulator for the current
        # attempt, the finished per-accepted-step records, dt-cuts pending since the last accepted
        # step, and wall time [ms] burned in rejected loops.
        self._cur_step_timing = StepTiming()
        self.step_timings: list[StepTiming] = []
        self._pending_cuts = 0
        self._cut_time_ms = 0.0
        self._step_wall_t0 = time.perf_counter()
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

    def default_nonlinear_criteria(self, tol: float = 1.0e-4, max_iterations: int = 20) -> dict:
        """Shared stopping criterion for every Driesner solver -- the weis-matched relative-storage
        Lebesgue residual bar at ``tol`` (each equation's imbalance relative to its stored mass / rock
        heat per step) plus a max-iteration divergence guard. Centralised here so all inheriting
        solvers use ONE criterion instead of each re-specifying it (which is how Fig 4/5/6 drifted). A
        solver with a special need overrides by building its own dict (e.g. the Fig-6 salt column's
        absolute 1e-5 bar). Pairs with the ``slave_eliminated_secondaries`` default (now on), so the
        closures are exact each iteration and the PDE residuals are what this bar actually measures."""
        return {
            "nl_convergence_criteria": {
                "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                    tol=tol, metric=RelativeStorageLebesgueMetric(self))},
            "nl_divergence_criteria": {
                "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=max_iterations)},
        }

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
        self._clip_fraction_variables()
        self._install_full_iterate_cache()
        super().update_derived_quantities()
        # Weis-faithful explicit flash: after the surrogate flash, slave every eliminated secondary
        # (T, saturations, partial fractions) to its exact table value. Default ON now -- every model
        # inheriting this base gets it; a solver that truly wants the lagged flash sets it False.
        if self.params.get("slave_eliminated_secondaries", True):
            self._slave_eliminated_secondaries()

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """Batched phase-property flash over all subdomains (opt-in, bit-exact).

        The CF template flashes each phase per grid (``update_thermodynamic_properties_of_phases_on_grid``);
        ``compute_properties`` is cellwise and ``evaluate`` concatenates subdomains in order, so
        evaluating each dependency over the whole md-domain once, calling the EoS once, and
        scattering the per-cell property blocks back per grid is identical -- it just collapses
        ``n_subdomains`` AD tree walks + table samples per phase into one (the dominant flash cost
        on many-subdomain / mixed-dimensional runs). Gated by ``batch_phase_property_flash``;
        default off falls back to the per-grid template method.
        """
        if not self.params.get("batch_phase_property_flash", False):
            super().update_thermodynamic_properties_of_phases(state=state)
            return

        subdomains = self.mdg.subdomains()
        if not subdomains:
            return
        n_total = sum(g.num_cells for g in subdomains)
        equilibrium_defined = pc.has_equilibrium_specified(self)
        is_persistent = pc.is_persistent_variable_form(self)

        for phase in self.fluid.phases:
            dep_vals = [
                self.equation_system.evaluate(d(subdomains), state=state)
                for d in self.dependencies_of_phase_properties(phase)
            ]
            phase_state = phase.compute_properties(
                *cast(list[np.ndarray], dep_vals),
                params=self.params.get("phase_property_params", None),
            )
            offset = 0
            for grid in subdomains:
                sl = slice(offset, offset + grid.num_cells)
                update_phase_properties(
                    grid,
                    phase,
                    _slice_phase_properties(phase_state, sl, n_total),
                    0,
                    use_extended_derivatives=is_persistent,
                    update_fugacities=equilibrium_defined,
                )
                offset += grid.num_cells

    # Locally-eliminated secondary variable -> its OBL flash function (Driesner brine model). Slaving
    # ALL of them each iteration is the full Weis-style explicit flash (T alone is insufficient: the
    # saturations / NaCl partial fractions limit-cycle at later phase fronts just as T does at early
    # ones). Unknown names are skipped, so this is a no-op for models lacking these funcs/variables.
    _CFL_SLAVE_EPS = 1.0e-6              # keeps 1 - s_h and s_liq strictly positive (see the rel-perm)

    def _slave_eliminated_secondaries(self) -> None:
        """Overwrite each locally-eliminated secondary ITERATE with its exact OBL value f(p,h,z).

        Every secondary (T, s_gas, s_halite, x_NaCl_liq/gas/halite) is closed by an elimination
        equation ``var - f_obl(p,h,z) = 0``, but per Newton iteration its DOF is moved only by the
        LINEARIZED back-substituted increment -- a first-order extrapolation of the C0 multilinear
        OBL table. Where the primaries carry a cell across a table-cell / phase-front kink, that
        lagged value overshoots, the elimination residual flips sign, and Newton limit-cycles (the
        stall: temperature drives it at the boiling / halite fronts, the saturations / NaCl fractions
        at later fronts). Re-evaluating every secondary on the table each iteration -- exactly the
        Weis reference's explicit flash T,s,x = f(p,h,z) -- removes every lagged residual, so the
        differential system converges instead of oscillating.

        The elimination equations/Jacobians are untouched (A_ss = I Schur fast-path and the Fourier
        grad(T) column intact); only the iterate VALUES are slaved, so the converged fixed point (all
        elimination residuals 0) is unchanged -- only the Newton PATH. Gated by
        ``params['slave_eliminated_secondaries']``.

        ONE ``obl_sampler.sample_at`` per iteration for ALL secondaries (they share the (z,h,p)
        points) -- not the 5 redundant samples of the per-variable BrineConstitutiveDescription funcs.
        The read fields carry the same physical-bound clips as those funcs (S_h ceiling 1-eps; s_gas
        capped so s_liq = 1 - s_gas - s_halite stays >= eps; X in [0,1]; halite is pure NaCl -> 1)."""
        es = self.equation_system
        present = {v.name for v in es.variables}
        sampler = getattr(self, "obl_sampler", None)
        if "z_NaCl" not in present or sampler is None:
            return
        # secondaries are cell variables in the same subdomain/cell order as the primaries.
        p = es.get_variable_values([self.pressure_variable], iterate_index=0)
        h = es.get_variable_values([self.enthalpy_variable], iterate_index=0)
        z = es.get_variable_values(["z_NaCl"], iterate_index=0)
        try:
            sampler.sample_at(np.column_stack([z, h, p]))         # single OBL sample -> all fields
            pd = sampler.sampled_could.point_data
        except Exception:
            return
        eps = self._CFL_SLAVE_EPS
        s_h = np.clip(np.asarray(pd["S_h"], float), 0.0, 1.0 - eps)
        s_v = np.clip(np.asarray(pd["S_v"], float), 0.0, np.clip(1.0 - s_h - eps, 0.0, 1.0))
        slaved = {
            "temperature": np.asarray(pd["Temperature"], float),
            "s_halite": s_h,
            "s_gas": s_v,
            "x_NaCl_liq": np.clip(np.asarray(pd["Xl"], float), 0.0, 1.0),
            "x_NaCl_gas": np.clip(np.asarray(pd["Xv"], float), 0.0, 1.0),
            "x_NaCl_halite": np.ones_like(s_h),                   # halite is pure NaCl
        }
        for vname, val in slaved.items():
            if vname in present:
                es.set_variable_values(val, [vname], iterate_index=0)

    _FRACTION_VARIABLE_NAMES = (
        "s_gas", "s_halite", "x_NaCl_liq", "x_NaCl_gas", "x_NaCl_halite",
    )

    def _clip_fraction_variables(self) -> None:
        """Clamp saturation / partial-fraction ITERATE values into [0, 1].

        The Newton increment is distributed to all variables before the derived
        quantities are refreshed, so downstream evaluations (in particular the
        mobility-weighted permeability tensor of the fractional-flow template) can
        otherwise see negative saturations from a single overshooting update."""
        es = self.equation_system
        present = {v.name for v in es.variables}
        for name in self._FRACTION_VARIABLE_NAMES:
            if name not in present:
                continue
            vals = es.get_variable_values([name], iterate_index=0)
            clipped = np.clip(vals, 0.0, 1.0)
            if not np.array_equal(clipped, vals):
                es.set_variable_values(clipped, [name], iterate_index=0)
        # joint consistency: s_gas + s_halite <= 1, so s_liq >= 0 and the total
        # mobility stays strictly positive (kr_l + kr_v = 1 - s_h); otherwise the
        # fractional-flow weights divide 0/0 at cells where both hit their caps
        if "s_gas" in present and "s_halite" in present:
            sg = es.get_variable_values(["s_gas"], iterate_index=0)
            sh = es.get_variable_values(["s_halite"], iterate_index=0)
            tot = sg + sh
            over = tot > 1.0 - 1.0e-8
            if np.any(over):
                f = np.where(over, (1.0 - 1.0e-8) / np.maximum(tot, 1.0e-30), 1.0)
                es.set_variable_values(sg * f, ["s_gas"], iterate_index=0)
                es.set_variable_values(sh * f, ["s_halite"], iterate_index=0)

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
    #  These models impose Dirichlet inlet/outlet PRESSURE, which fixes the constant-pressure mode:
    #  the pressure block is non-singular, so no null-mean gauge / null space is needed here.
    #  (subsection_4_2's closed-domain solver keeps its own bordered null-mean path.)
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
        dc, dr, ic, ir, sc, sr, n_p, _ = self._primary_secondary_indices(n)
        n_i = len(ic)
        # PRIMARY = subdomain + interface (interface trailing); SECONDARY = local closures.
        pc = np.concatenate([dc, ic]); pr = np.concatenate([dr, ir])
        A = A.tocsc()
        App = A[pr][:, pc].tocsr()
        Aps = A[pr][:, sc].tocsr()
        Asp = A[sr][:, pc].tocsc()
        Ass = A[sr][:, sc].tocsr()
        bp, bs = b[pr], b[sr]
        t_extract = time.perf_counter() - t0

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
        t_step1 = time.perf_counter() - t0 - t_extract
        t1 = time.perf_counter()

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
        t_step2 = time.perf_counter() - t1

        t2 = time.perf_counter()
        reduced_solver = self.params.get("reduced_solver", "cpr")
        use_cpr = reduced_solver == "cpr"
        if use_cpr:
            xk, cpr_its, cpr_blocks = self._cpr_petsc_solve(
                Sc, gc, n_p,
                lu_pressure_max=int(self.params.get("cpr_lu_pressure_max", 60000)),
                rtol=float(self.params.get("cpr_rtol", 1.0e-8)),
                maxit=int(self.params.get("cpr_maxit", 300)))
        else:
            # Direct sparse LU of the reduced primary system. The Dirichlet inlet/outlet pressure
            # makes it non-singular, and below ~1e5 DOF a single LU beats CPR by ~10x: the iterative
            # machinery's fixed per-solve overhead dwarfs the actual work at these sizes.
            if reduced_solver == "auto":
                reduced_solver = "splu" if Sc.shape[0] < 20000 else "pardiso"
            xk, actual_solver = self._direct_reduced_solve(Sc, gc, reduced_solver)
            cpr_its, cpr_blocks = 0, [("direct", Sc.shape[0], actual_solver)]
        xp = np.concatenate([xk, lu_i.solve(gl - Slk @ xk)]) if n_i else xk
        t_cpr = time.perf_counter() - t2

        # Accuracy gate: the inlet/outlet Dirichlet pressure fixes the constant-pressure mode, so the
        # reduced system is non-singular and a plain relative residual is the right measure.
        r = S @ xp - g
        rel = np.linalg.norm(r) / max(np.linalg.norm(g), 1.0e-30)
        acc_tol = float(self.params.get("cpr_accuracy_tol", 1.0e-6))
        if rel > acc_tol:
            raise RuntimeError(
                f"CPR residual too large for Newton (rel={rel:.1e} > {acc_tol:.1e})")

        rhs_s = bs - Asp @ xp
        xs = rhs_s if lu is None else lu.solve(rhs_s)       # back-substitute the secondaries

        if use_cpr:
            logger.info("Schur-CPR solve: %.3fs (%d KSP its, res %.1e)",
                        time.perf_counter() - t0, cpr_its, rel)
        else:
            logger.info("Schur-direct solve: %.3fs (%s, res %.1e)",
                        time.perf_counter() - t0, cpr_blocks[0][2], rel)
        logger.info("  reduced %d = %s", m,
                    " + ".join(f"{nm} {sz} ({pcname})" for nm, sz, pcname in cpr_blocks))
        logger.info("  eliminated: %d interface + %d secondary", n_i, len(sr))
        logger.info("  cost: extract %.3fs + step1 %.3fs + step2 %.3fs + solve %.3fs",
                    t_extract, t_step1, t_step2, t_cpr)

        x = np.empty(n, dtype=float)
        x[pc] = xp
        x[sc] = xs
        return x

    @staticmethod
    def _direct_reduced_solve(S, g, kind="splu"):
        """Exact sparse LU of the reduced primary system (non-singular under Dirichlet inlet/outlet
        pressure). ``'splu'`` = scipy, fastest below ~2e4 DOF; ``'pardiso'`` = MKL PARDISO, faster
        above. Returns ``(x, actual_solver)`` -- ``actual_solver`` reports what really ran, so a
        pypardiso failure/absence (falls back to scipy splu) is visible in the log, not silent."""
        g = np.asarray(g, dtype=float)
        if kind == "pardiso":
            try:
                from pypardiso import spsolve as _pardiso
                return np.asarray(_pardiso(S.tocsr(), g), dtype=float), "pardiso"
            except Exception as exc:
                logger.warning("pypardiso unavailable/failed (%s); falling back to scipy splu.", exc)
        from scipy.sparse.linalg import splu
        return splu(S.tocsc()).solve(g), "splu"

    @staticmethod
    def _cpr_petsc_solve(S, g, n_p, lu_pressure_max=60000, rtol=1.0e-8, maxit=300):
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

        The inlet/outlet Dirichlet pressure of these models fixes the constant-pressure mode, so the
        pressure block is non-singular and no null-space gauge is needed.  Returns
        ``(x, n_iterations)``."""
        import scipy.sparse as sps
        from petsc4py import PETSc

        S = S.tocsr(); S.sort_indices()
        n = S.shape[0]
        g = np.array(g, dtype=float)

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

        # pressure (elliptic Darcy) -> LU (MUMPS) when small, else BoomerAMG.  ``blocks`` records the
        # preconditioner ACTUALLY selected per field, so the caller's log reports the real choice
        # (the pressure block is only AMG above ``lu_pressure_max``).
        blocks: list[tuple[str, int, str]] = []
        kp = subksps[0]; kp.setType("preonly")
        App_block = kp.getOperators()[0]
        if n_p < lu_pressure_max:
            kp.getPC().setType("lu"); kp.getPC().setFactorSolverType("mumps")
            blocks.append(("p", N, "LU/MUMPS"))
        else:
            kp.getPC().setType("hypre"); kp.getPC().setHYPREType("boomeramg")
            App_block.setNearNullSpace(PETSc.NullSpace().create(
                constant=False, vectors=[const_p_vec(App_block)], comm=PETSc.COMM_SELF))
            blocks.append(("p", N, "hypre/BoomerAMG"))
        # enthalpy (advection-diffusion, the hard block) -> direct LU.
        if len(subksps) >= 2:
            kh = subksps[1]; kh.setType("preonly")
            kh.getPC().setType("lu"); kh.getPC().setFactorSolverType("mumps")
            blocks.append(("h", N, "LU/MUMPS"))
        # composition z (pure advection -- a flow DAG for which ILU is ~exact) -> ILU(0).
        for kz in subksps[2:]:
            kz.setType("preonly"); kz.getPC().setType("ilu")
            blocks.append(("z", N, "ILU"))

        xv = mat.createVecRight()
        bv = mat.createVecLeft()
        bv.setArray(np.ascontiguousarray(g, dtype=PETSc.ScalarType))
        ksp.solve(bv, xv)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"PETSc CPR KSP diverged (reason {ksp.getConvergedReason()}, "
                               f"its={ksp.getIterationNumber()})")
        return xv.getArray().copy(), ksp.getIterationNumber(), blocks

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
                return self._direct_sparse_solve()
            return solution
        else:
            # Check if PETSc was requested but not available
            if self.use_petsc and not PETSC_AVAILABLE:
                logger.info("*** SOLVER FALLBACK ***")
                logger.info("PETSc was requested but not available. Using default direct solver.")

            # Use default solver
            return self._direct_sparse_solve()

    def _direct_sparse_solve(self) -> np.ndarray:
        """Direct sparse solve of ``self.linear_system`` (pypardiso if available, else scipy).

        Replaces the old ``super().solve_linear_system()`` fall-through: on the current solver
        stack ``SolutionStrategy.solve_linear_system`` raises (linear solvers were moved out of
        the model into ``pp.solvers`` LinearSolver objects), so the model must solve directly.
        """
        A, b = self.linear_system
        try:
            from pypardiso import spsolve as _spsolve
        except ImportError:
            from scipy.sparse.linalg import spsolve as _spsolve
        return np.atleast_1d(np.asarray(_spsolve(A.tocsr(), np.asarray(b)), dtype=float))

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system and apply the configured nonlinear step control.

        Selected by ``params["step_control_method"]`` (default ``"None"``):

        - ``"None"``  : plain Newton, no step control.
        - ``"LS"``    : backtracking line search (Armijo), applied only when the full
          Newton step would increase the residual.

        Residual reporting and solution post-processing are intentionally left to the
        model (see the overridable hooks :meth:`compute_residuals_by_category`,
        :meth:`postprocessing_overshoots`, :meth:`postprocessing_thermal_overshoots`);
        this method only chooses and applies the step.
        """
        _, residual_vector = self.linear_system
        step_control_method = self.params.get("step_control_method", "None")

        if step_control_method == "None":
            t = time.perf_counter()
            solution = self._solve_linear_system_core()
            self._accum_step("t_linear_ms", (time.perf_counter() - t) * 1e3)

        elif step_control_method == "LS":
            # weis (2014) globalization: full Newton step, then backtrack EVERY iteration
            # until the physically-clipped step reduces the residual L2 norm. alpha=1 is a
            # no-op when the full step already reduces it (the smooth single-phase case), so
            # this is dormant on Fig 4/5 and only bites at the Fig-6 salt front -- exactly
            # like weis_1d_solver.newton_step_brine. See backtracking_line_search.
            t = time.perf_counter()
            delta_x = self._solve_linear_system_core()
            self._accum_step("t_linear_ms", (time.perf_counter() - t) * 1e3)
            t = time.perf_counter()
            alpha = self.backtracking_line_search(delta_x, residual_vector)
            self._accum_step("t_linesearch_ms", (time.perf_counter() - t) * 1e3)
            solution = alpha * delta_x

        else:
            raise ValueError(
                f"Unknown step_control_method: {step_control_method}. "
                f"Valid options are: 'None', 'LS'"
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

        # Rediscretize the state-dependent flux operators (upwind directions, mobility-weighted
        # permeability). ``lag_discretization_in_line_search`` freezes them at the base-state
        # discretization (from the Newton-iteration assemble) during backtracking: a trial residual is
        # then evaluated against the frozen upwind matrices -- an approximation of the true residual
        # that keeps the monotone-decrease line search valid while avoiding a full re-discretization
        # over every subdomain on every trial (the dominant cost on many-subdomain / MD runs). Default
        # off (exact per-trial re-discretization); it changes the Newton PATH, not the fixed point.
        if not self.params.get("lag_discretization_in_line_search", False):
            self.rediscretize()

        # Assemble the current nonlinear residual
        current_nonlinear_residual = self.equation_system.assemble(evaluate_jacobian=False)

        # Restore original state if requested
        if restore_state:
            try:
                self.equation_system.set_variable_values(x_current, iterate_index=0)
            except TypeError:
                self.equation_system.set_variable_values(x_current)

            # The variable VALUES above are always restored (the accepted step is applied additively,
            # so the base iterate must be back in place). The derived quantities + discretization,
            # however, are rebuilt for the BASE state only to be overwritten by the next line-search
            # trial (or, after the search, by after_nonlinear_iteration / check_convergence) before
            # anything reads them -- so for a line-search caller this rebuild is dead work (~39% of the
            # eval). ``lazy_residual_restore`` skips it; default keeps the full restore for any caller
            # that does read the base derived state back (Fig-8 / 3D stay byte-identical by default).
            if not self.params.get("lazy_residual_restore", False):
                self.update_derived_quantities()
                self.refresh_buoyancy_direction()
                self.rediscretize()

        return current_nonlinear_residual

    def backtracking_line_search(
        self,
        delta_x: np.ndarray,
        current_residual: np.ndarray,
        rho: float = 0.5,
        max_iterations: int = 10,
    ) -> float:
        """weis (2014) backtracking line search -- monotone residual decrease.

        Start at ``alpha = 1`` and halve by ``rho`` until the physically-clipped trial step
        reduces the residual L2 norm below the current one; accept the FIRST alpha that does.
        If none of ``max_iterations`` trials reduce it, accept the smallest step tried -- weis
        never rejects a step, it always advances. The per-trial clip is the model's
        ``postprocessing_overshoots`` (the same physical-bound clip weis applies inside its own
        loop). No Armijo constant, no Jacobian prediction: a faithful port of
        ``weis_1d_solver.newton_step_brine``'s inner ``for _ in range(10)`` backtracking.

        Returns the accepted step length ``alpha``; the caller forms ``alpha * delta_x`` and the
        model's post-processing re-applies the same clip to that accepted increment.

        The trial cap is ``params["line_search_max_iterations"]`` (default = the ``max_iterations``
        argument): a smaller cap means fewer (expensive) residual re-evaluations per Newton
        iteration and a larger minimum step (``rho**(cap-1)``), at the cost of a coarser search.
        """
        max_iterations = int(self.params.get("line_search_max_iterations", max_iterations))
        nrm_current = self._line_search_merit(current_residual)
        alpha = 1.0
        for i in range(max_iterations):
            alpha = rho ** i                                   # 1, 1/2, 1/4, ...
            trial = alpha * delta_x
            try:
                self.postprocessing_overshoots(trial)          # physical clip, per trial
            except Exception:
                pass
            try:
                residual_trial = self.compute_residual_from_increment(
                    trial, restore_state=True)
            except Exception:
                continue
            if np.all(np.isfinite(residual_trial)):
                nrm_trial = self._line_search_merit(residual_trial)
                if nrm_trial < nrm_current:
                    if i > 0:
                        print(f"  Line search (weis): alpha={alpha:.4f}, "
                              f"merit={nrm_trial:.4e} < {nrm_current:.4e}")
                    self._cur_step_timing.n_line_searches += i + 1   # backtracking trials this call
                    return alpha
        self._cur_step_timing.n_line_searches += max_iterations
        return alpha

    def _line_search_merit(self, residual: np.ndarray) -> float:
        """Scalar the backtracking line search descends: the SAME row-scaled RelativeStorageLebesgueMetric
        max the convergence criterion tests, so the search and the stopping test are aligned on the same
        (binding) equation -- fixing the case where the LS reduced raw-L2 (dominated by an already-small
        energy row) while the mass row bound.  Falls back to raw L2 only if the scaled metric is
        unavailable (no row scales yet / empty)."""
        try:
            per_eq = RelativeStorageLebesgueMetric(self)(np.asarray(residual, float))
            if per_eq:
                return float(max(per_eq.values()))
        except Exception:
            pass
        return float(np.linalg.norm(residual))

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
        # check_convergence (called immediately after this by the nonlinear solver) re-runs
        # update_derived_quantities + refresh_buoyancy_direction + rediscretize on the FRESH flash and
        # nothing reads the discretization in between -- so this refresh+rediscretize is on stale flash
        # and is fully overwritten (dead work). ``skip_after_iteration_discretization`` drops it; kept
        # by default for any model whose check_convergence does NOT rediscretize.
        if not self.params.get("skip_after_iteration_discretization", False):
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

    def assemble_linear_system(self) -> pp.solvers.LinearSystem:
        """Custom assemble linear system that updates Jacobian every 0, 3, 6, 9... Newton iterations.

        This method implements a dedicated solution strategy that:
        - Assembles the full linear system (Jacobian + residual) at iterations 0, 3, 6, 9, etc.
        - Updates only the residual part for other iterations (1, 2, 4, 5, 7, 8, etc.)

        Returns a ``pp.solvers.LinearSystem`` (the new solver stack's contract) while keeping the
        internal ``self.linear_system`` (matrix, rhs) tuple the custom solve methods read.
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
        self._accum_step("t_assembly_ms", (t_1 - t_0) * 1e3)
        mode = (
            "Jacobian + residual"
            if (iteration_num % 2 == 0 or iteration_num < 10)
            else "residual only"
        )
        logger.info(f"Assembled {mode} in {t_1 - t_0:.2e} seconds.")
        matrix, rhs = self.linear_system
        return pp.solvers.LinearSystem(matrix=matrix.tocsr(), rhs=np.asarray(rhs))

    def _accum_step(self, field: str, ms: float) -> None:
        """Add ``ms`` to bucket ``field`` of the current step's :class:`StepTiming`."""
        st = self._cur_step_timing
        setattr(st, field, getattr(st, field) + ms)

    def before_nonlinear_loop(self) -> None:
        # Fresh cost accumulator for every step ATTEMPT (incl. retries after a dt-cut); the accepted
        # attempt's record is the one after_nonlinear_convergence keeps.
        self._cur_step_timing = StepTiming()
        self._cur_step_timing.dt_begin_s = float(getattr(self.time_manager, "dt", 0.0))
        self._step_wall_t0 = time.perf_counter()
        super().before_nonlinear_loop()

    def before_nonlinear_iteration(self) -> None:
        t = time.perf_counter()
        super().before_nonlinear_iteration()
        self._accum_step("t_before_ms", (time.perf_counter() - t) * 1e3)

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        t = time.perf_counter()
        super().after_nonlinear_iteration(nonlinear_increment)
        self._accum_step("t_after_ms", (time.perf_counter() - t) * 1e3)

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Track Newton iterations for this timestep
        current_iterations = self.nonlinear_solver_statistics.num_iterations
        self.newton_iterations_per_timestep.append(current_iterations)
        self.total_newton_iterations += current_iterations

        # Finalise this accepted step's cost record (wall time, dt, iterations, preceding dt-cuts).
        self._cur_step_timing.wall_ms = (time.perf_counter() - self._step_wall_t0) * 1e3
        self._cur_step_timing.dt_end_s = float(getattr(self.time_manager, "dt", 0.0))
        self._cur_step_timing.newton_iterations = current_iterations
        self._cur_step_timing.n_cuts = self._pending_cuts
        self.step_timings.append(self._cur_step_timing)
        self._pending_cuts = 0

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
        print("Delta t (year): ", self.time_manager.dt * second_to_year)
        print("Time index: ", self.time_manager.time_index)
        print("*" * 60)
        print("")

    def after_nonlinear_failure(self) -> None:
        """Count a rejected nonlinear loop (a time-step cut) before deferring to the template."""
        self.n_time_step_cuts = getattr(self, "n_time_step_cuts", 0) + 1
        if self.params.get("diagnose_binding", False):        # STEP-0 probe on the stalled iterate
            try:
                self._diagnose_binding_mechanism()
            except Exception as exc:                          # a diagnostic must never break the run
                logger.warning("binding diagnostic skipped: %s", exc)
        # The rejected attempt's cost is not attributed to any accepted step; tally it separately
        # and remember the cut so the next accepted step records how many cuts preceded it.
        self._cut_time_ms += self._cur_step_timing.t_total_ms
        self._pending_cuts += 1
        super().after_nonlinear_failure()

    def darcy_flux_discretization(self, subdomains):
        """TPFA by default; MPFA when ``params["consistent_discretization"]`` is set (the
        consistent discretization on non-K-orthogonal grids)."""
        Ad = (pp.ad.MpfaAd if self.params.get("consistent_discretization", False)
              else pp.ad.TpfaAd)
        return Ad(self.darcy_keyword, list(subdomains))

    def fourier_flux_discretization(self, subdomains):
        """Same TPFA/MPFA switch as :meth:`darcy_flux_discretization`."""
        Ad = (pp.ad.MpfaAd if self.params.get("consistent_discretization", False)
              else pp.ad.TpfaAd)
        return Ad(self.fourier_keyword, list(subdomains))

    def data_to_export(self):
        """Base export plus ``delta_<q> = q(t) - q(0)`` for pressure, T_C, enthalpy and
        every independent overall fraction (the reference state is captured at the first
        export, i.e. t = 0)."""
        data = super().data_to_export()  # type: ignore[misc]
        ev = self.equation_system.evaluate
        components = [c for c in self.fluid.components
                      if c != self.fluid.reference_component]
        if not hasattr(self, "_delta_export_ref"):
            self._delta_export_ref = {}
        for sd in self.mdg.subdomains():
            cur = {
                "pressure": np.asarray(ev(self.pressure([sd])), dtype=float),
                "T_C": np.asarray(ev(self.temperature([sd])), dtype=float) - 273.15,
                "enthalpy": np.asarray(ev(self.enthalpy([sd])), dtype=float),
            }
            for c in components:
                cur[f"z_{c.name}"] = np.asarray(ev(c.fraction([sd])), dtype=float)
            ref = self._delta_export_ref.setdefault(
                sd.id, {k: v.copy() for k, v in cur.items()})
            for name, v in cur.items():
                data.append((sd, f"delta_{name}", v - ref[name]))
        return data

    # weis-matched reference normalizers for the RELATIVE residual bar (RelativeStorageLebesgueMetric)
    _RESIDUAL_RHO_REF = 800.0     # reference fluid density [kg/m^3]  (weis RHO_REF)
    _RESIDUAL_T_REF = 500.0       # reference temperature   [K]       (weis T_REF)

    def residual_row_scales(self) -> dict:
        """Per-equation residual row-scale for a RELATIVE convergence bar mirroring weis ms/es:
        fluid-mass storage for the total-mass (pressure) + component rows, rock-heat storage for the
        energy row. The volume-weighted Lebesgue norm behaves as ``N_e ~ r_bar * sqrt(V_tot)`` (with
        ``r_bar`` the intensive imbalance), so the scale ``S_e = sqrt(V_tot) * (storage density)/dt0``
        makes ``N_e / S_e = r_bar * dt0 / storage`` -- the dimensionless imbalance-per-stored-quantity-
        per-step, i.e. exactly weis's row-scaled residual. Storage densities use the model's own
        porosity / rock constants (``c_s`` already in the model's scaled units) and the weis
        reference ``rho_ref`` / ``T_ref``. Empty dict -> no scaling.

        ``dt0``: with ``params['residual_scale_current_dt']`` (weis' convention) it is the CURRENT step,
        so the bar TRACKS dt -- a step cut at a stiff front loosens the bar proportionally and still
        converges, instead of being judged against the nominal step and stalling. Default (unset) keeps
        the fixed nominal ``dt_init`` bar the 1D/3D benchmarks were validated with."""
        try:
            Vtot = float(sum(float(np.sum(sd.cell_volumes)) for sd in self.mdg.subdomains()))
            if not (Vtot > 0.0):
                return {}
            if self.params.get("residual_scale_current_dt", False):     # weis: bar tracks the current dt
                dt0 = float(getattr(self.time_manager, "dt", 0.0)
                            or getattr(self.time_manager, "dt_init", 0.0) or 1.0)
            else:                                                        # fixed nominal-dt bar (default)
                dt0 = float(getattr(self.time_manager, "dt_init", 0.0)
                            or getattr(self.time_manager, "dt", 0.0) or 1.0)
            phi = float(self.solid.porosity)
            rho_s = float(self.solid.density)
            c_s = float(self.solid.specific_heat_capacity)            # model-scaled [MJ/(kg K)]
            sqrtV = np.sqrt(Vtot)
            S_mass = sqrtV * (phi * self._RESIDUAL_RHO_REF) / dt0                        # [kg/s]-scale
            S_energy = sqrtV * ((1.0 - phi) * rho_s * c_s * self._RESIDUAL_T_REF) / dt0  # [MJ/s]-scale
            scales = {"mass_balance_equation": S_mass, "energy_balance_equation": S_energy}
            for c in self.fluid.components:
                if self.has_independent_fraction(c):
                    scales[f"component_mass_balance_equation_{c.name}"] = S_mass
            return scales
        except Exception:
            return {}                    # any issue -> fall back to the absolute Lebesgue bar

    # ---- STEP-0 binding-mechanism diagnostic (kink vs negative compressibility) ----------------
    def _diagnose_binding_mechanism(self, top_k: int = 20) -> None:
        """Offline diagnostic at a STALLED Newton iterate: why does the mass_balance (pressure) row bind?

        No solve -- pure post-processing on the current iterate.  It (1) identifies the FAILING CELLS --
        there is no per-cell 'fail' (Newton converges globally), so a failing cell is operationally a top
        contributor to the row-scaled mass_balance residual, i.e. a cell holding the binding norm above tol
        (these are typically the phase-front cells); (2) folds in the sign(c_cell) test for NEGATIVE
        COMPRESSIBILITY -- the mass accumulation is phi*rho*V, so its pressure diagonal is phi*V*dRho/dp and
        dRho/dp|_{h,z} is exactly grad_Rho's p-component; dRho/dp<0 makes that diagonal negative -> the
        pressure operator loses diagonal dominance (indefinite), which no kink smoothing fixes; (3) checks
        whether the line-search merit (raw L2) binds the SAME row the criterion (row-scaled) does.
        Gated by params['diagnose_binding'], fired from after_nonlinear_failure so it lands on the failing
        state (PorePy reverts the iterate only in the outer time-step loop, after this hook)."""
        import porepy as pp                                              # local: avoid import cycle noise
        es = self.equation_system
        sampler = getattr(self, "obl_sampler", None)
        if sampler is None or "z_NaCl" not in {v.name for v in es.variables}:
            return
        try:
            p = es.get_variable_values([self.pressure_variable], iterate_index=0)
            h = es.get_variable_values([self.enthalpy_variable], iterate_index=0)
            z = es.get_variable_values(["z_NaCl"], iterate_index=0)
            sampler.sample_at(np.column_stack([z, h, p]))
            pd = sampler.sampled_could.point_data
            s_v = np.asarray(pd["S_v"], float)
            s_h = np.asarray(pd["S_h"], float)
            dRho_dp = np.asarray(pd["grad_Rho"], float)[:, 2]           # dRho/dp|_{h,z}; sign robust to scaling
            rho_l = np.asarray(pd["Rho_l"], float)
            rho_v = np.asarray(pd["Rho_v"], float)
            T = np.asarray(pd["Temperature"], float)
            pr = np.asarray(pd["phase_region"], float) if "phase_region" in pd else None
        except Exception as exc:
            logger.warning("binding diagnostic: flash/grad read failed (%s)", exc)
            return
        sds = list(self.mdg.subdomains())
        vol = np.concatenate([sd.cell_volumes for sd in sds]) if sds else np.zeros(0)
        dims = (np.concatenate([np.full(sd.num_cells, sd.dim) for sd in sds])
                if sds else np.zeros(0, int))
        n = vol.size

        # FAILING CELLS: top per-cell contribution to the (row-scaled) mass residual (scale is a global
        # constant, so it is ranking-invariant; use the raw per-cell Lebesgue contribution |r|*sqrt(V)).
        mass_eq = pp.compositional_flow.get_primary_equations_cf(self)[0]
        try:
            r_mass = np.abs(np.asarray(
                es.assemble(evaluate_jacobian=False, equations={mass_eq: sds}), float))
        except Exception as exc:
            logger.warning("binding diagnostic: mass residual assemble failed (%s)", exc)
            return
        if r_mass.size != n:
            logger.warning("binding diagnostic: residual/cell size mismatch (%d vs %d)", r_mass.size, n)
            return
        contrib = r_mass * np.sqrt(np.maximum(vol, 0.0))
        k = int(min(top_k, n))
        fail = np.argsort(contrib)[::-1][:k]

        neg = dRho_dp < 0.0
        neg_fail = int(np.sum(neg[fail]))
        two_phase_fail = int(np.sum((s_v[fail] > 1e-9) & (s_v[fail] < 1.0 - 1e-9)))
        salt_fail = int(np.sum(s_h[fail] > 1e-9))
        imin = int(np.argmin(dRho_dp)) if n else -1
        # CRITICAL-POINT proximity: the phase densities MERGE at the mixture critical point, so
        # rho_v/rho_l -> 1 (sub-critical two-phase has rho_v << rho_l). Near critical the EOS derivatives
        # diverge -> the Jacobian is near-singular and NEITHER a smaller dt NOR a trust region helps
        # (alpha collapses with no residual progress -- a thermodynamic singularity, not a step-size issue).
        dens_ratio = rho_v / np.maximum(rho_l, 1e-30)
        near_crit_fail = int(np.sum(dens_ratio[fail] > 0.5))
        yr = 365.0 * 86400.0

        logger.info("=" * 68)
        logger.info("BINDING DIAGNOSTIC @ stalled iterate (t=%.1f yr, dt=%.3g yr, cells=%d)",
                    self.time_manager.time / yr, self.time_manager.dt / yr, n)
        logger.info("failing cells = top %d by row-scaled mass_balance residual", k)
        logger.info("  of those:  two-phase=%d  salt(s_h>0)=%d  dRho/dp<0=%d   |   global dRho/dp<0 frac=%.2f",
                    two_phase_fail, salt_fail, neg_fail, float(np.mean(neg)) if n else 0.0)
        logger.info("  min dRho/dp = %+.3e (cell %d, s_v=%.3f s_h=%.3f)",
                    float(dRho_dp[imin]) if n else 0.0, imin,
                    float(s_v[imin]) if n else 0.0, float(s_h[imin]) if n else 0.0)
        logger.info("  CRITICAL proximity: failing cells with rho_v/rho_l>0.5 = %d/%d | max ratio in fail = %.3f",
                    near_crit_fail, k, float(np.max(dens_ratio[fail])) if k else 0.0)
        for j in fail[:min(8, k)]:
            tag = ("CRITICAL" if dens_ratio[j] > 0.5 else
                   ("NEG-COMPR" if dRho_dp[j] < 0 else
                    ("two-phase" if 1e-9 < s_v[j] < 1 - 1e-9 else ("salt" if s_h[j] > 1e-9 else "single"))))
            pr_str = "" if pr is None else "  pr=%.1f" % float(pr[j])
            logger.info("    cell %6d dim%d | mass=%.3e  s_v=%.4f  T=%.1fK  rho_v/rho_l=%.3f  dRho/dp=%+.3e  %s%s",
                        int(j), int(dims[j]), float(contrib[j]), float(s_v[j]), float(T[j]),
                        float(dens_ratio[j]), float(dRho_dp[j]), tag, pr_str)

        if near_crit_fail >= max(1, k // 2):
            verdict = ("CRITICAL-POINT crossing (rho_v/rho_l -> 1) -> SINGULAR EOS derivatives; the Jacobian "
                       "is near-singular, so dt / LS / TR cannot help. Regularize the critical band, not the solver")
        elif neg_fail >= max(1, k // 2):
            verdict = "NEGATIVE COMPRESSIBILITY -> indefinite pressure diagonal; PTC / compressibility floor"
        elif two_phase_fail + salt_fail >= max(1, k // 2):
            verdict = "KINK at a phase front (definite, discontinuous) -> gradient recovery / step limiting"
        else:
            verdict = "single-phase, positive compressibility -> elliptic / CPR-AMG linear-solver issue"
        logger.info("  VERDICT: %s", verdict)

        # MERIT ALIGNMENT: does the raw-L2 line-search merit bind the same row as the row-scaled criterion?
        try:
            r_full = np.asarray(es.assemble(evaluate_jacobian=False), float)
            raw = pp.EquationBasedLebesgueMetric(self)(r_full)
            scaled = RelativeStorageLebesgueMetric(self)(r_full)
            rb = max(raw, key=raw.get) if raw else "?"
            sb = max(scaled, key=scaled.get) if scaled else "?"
            logger.info("  norms: raw-L2 binds '%s'(%.3e); row-scaled (= LS merit = criterion) binds "
                        "'%s'(%.3e)  [%s]", rb, raw.get(rb, 0.0), sb, scaled.get(sb, 0.0),
                        "same row" if rb == sb
                        else "differ: LS now descends the row-scaled (binding) row")
        except Exception:
            pass
        logger.info("=" * 68)

    def collect_run_stats(self) -> NonlinearRunStats:
        """Return a picklable :class:`NonlinearRunStats` snapshot of the run.

        Any model deriving from this base gets the feature for free; call it after the time loop
        (e.g. in ``after_simulation`` or right after ``run_time_dependent_model``) to persist or
        inspect the solver behaviour without touching PorePy's non-picklable statistics object."""
        hist = list(self.newton_iterations_per_timestep)
        timings = list(getattr(self, "step_timings", []))
        # dt is constant within a step (the TimeManager adapts it only AFTER convergence), so
        # dt_end = the dt the NEXT accepted step actually ran with -- this is what exposes the
        # adaptive dt trajectory (growth after easy steps, collapse after cuts). The last step keeps
        # its own dt.
        for a, b in zip(timings, timings[1:]):
            a.dt_end_s = b.dt_begin_s
        return NonlinearRunStats(
            n_accepted_steps=len(hist),
            n_time_step_cuts=getattr(self, "n_time_step_cuts", 0),
            total_newton_iterations=int(self.total_newton_iterations),
            max_newton_iterations=max(hist) if hist else 0,
            iterations_per_step=hist,
            step_timings=timings,
            t_cut_ms=float(getattr(self, "_cut_time_ms", 0.0)),
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
        stats_json["timing_totals_ms"] = stats._timing_totals_ms()
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

