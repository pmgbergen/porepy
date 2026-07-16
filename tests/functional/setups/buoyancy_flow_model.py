"""
Module implementing compositional flow models for multi-phase, multi-component systems
with buoyancy effects.

Supports both:
- 2-phase, 2-component systems (e.g., water and methane)
- 3-phase, 3-component systems (e.g., water, oil, and methane)

The module defines 2D and 3D geometry setup, equations of state, initial and boundary
conditions, and solution procedures for compositional fluid flow problems with
gravitational effects.

Fixed- and mixed-dimensional meshes are defined on cartesian grids.

"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFlowTemplate,
    CompositionalFractionalFlowTemplate,
)


# Constants for fluid phase densities (kg/m^3)
rho_w = 1000.0  #: Density of water (H2O)
rho_o = 700.0  #: Density of oil (C5H12)
rho_g = 200.0  #: Density of gas (CH4)

# Constants for fluid phase viscosities (Pa * second)
mu_w = 1.0e-3  #: Viscosity of water (H2O)
mu_o = 1.0e-4  #: Viscosity of oil (C5H12)
mu_g = 1.0e-5  #: Viscosity of gas (CH4)

# Specific enthalpies (physical units MJ/kg)
h_w = 1.0  # Water
h_o = 1.5  # Oil
h_g = 2.0  # Gas

# Conversion factor to Mega (1e-6)
to_Mega = 1.0e-6  #: Unit conversion factor to Mega units


class Geometry(pp.PorePyModel):
    """
    Abstract base class defining geometry.

    Subclasses must implement:
        - dirichlet_facets: Identifies Dirichlet boundary facets on the domain.
    """

    @abstractmethod
    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Return Dirichlet facet indices."""
        pass

    def _dirichlet_anchor_facet(
        self, sd: pp.Grid | pp.BoundaryGrid, axis: int
    ) -> np.ndarray:
        """Single pressure-anchor facet on the domain's maximum-``axis`` plane.

        A pure coordinate bounding box on the face centers selects the facets lying on
        that plane -- which, unlike PorePy's ``domain_boundary_sides``, also captures
        fracture-tip boundary facets -- and the one nearest the plane center is returned.

        The opening is deliberately kept to a single facet: this incompressible
        fractional-flow system needs a pressure anchor, but any larger open Dirichlet
        boundary lets buoyancy drive mass across it and pollutes the conservation checks.

        Parameters:
            sd: A subdomain grid or its boundary grid.
            axis: Coordinate axis (0=x, 1=y, 2=z) whose maximum defines the plane.

        Returns:
            The index of the anchor facet (empty if the subdomain has no facet on the
            plane, e.g. a fracture not reaching the boundary).

        """
        if isinstance(sd, pp.Grid):
            coords = sd.face_centers
        elif isinstance(sd, pp.BoundaryGrid):
            coords = sd.cell_centers
        else:
            raise ValueError("Type not expected.")

        bounding_box = self._domain.bounding_box
        on_plane = np.where(
            np.isclose(coords[axis], bounding_box[("xmax", "ymax", "zmax")[axis]])
        )[0]
        if on_plane.size <= 1:
            return on_plane
        # Of the plane facets, pick the one closest to the plane center. The transverse
        # axes are exactly those below the (last) maximum axis.
        dist2 = np.zeros(on_plane.size)
        for a in range(axis):
            center = 0.5 * (
                bounding_box[("xmin", "ymin", "zmin")[a]]
                + bounding_box[("xmax", "ymax", "zmax")[a]]
            )
            dist2 += (coords[a, on_plane] - center) ** 2
        return on_plane[[int(np.argmin(dist2))]]


class ModelGeometry2D(Geometry):
    """2D Cartesian domain."""

    def set_domain(self) -> None:
        """Set square domain."""
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Single pressure-anchor facet at the center of the ``ymax`` edge."""
        return self._dirichlet_anchor_facet(sd, axis=1)


class ModelMDGeometry2D(ModelGeometry2D):
    """2D mixed-dimensional domain: [0, 2]^2 (2x2), two fractures crossing at (1, 1)."""

    def set_domain(self) -> None:
        length = self.units.convert_units(2.0, "m")
        self._domain = pp.Domain({"xmax": length, "ymax": length})

    def set_fractures(self) -> None:
        points = np.array([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [2.0, 1.0]]).T
        fracs = np.array([[0, 1], [2, 3]]).T
        self._fractures = pp.frac_utils.pts_edges_to_linefractures(points, fracs)


class ModelGeometry3D(Geometry):
    """3D Cartesian domain."""

    def set_domain(self) -> None:
        """Set a 3D cubic domain."""
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        z_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length,
            "ymax": y_length,
            "zmax": z_length,
        }
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Single pressure-anchor facet at the center of the ``zmax`` face."""
        return self._dirichlet_anchor_facet(sd, axis=2)


class ModelMDGeometry3D(ModelGeometry3D):
    """3D mixed-dimensional domain: [0, 2]^3 (2x2x2), three planes crossing at (1, 1, 1)."""

    def set_domain(self) -> None:
        length = self.units.convert_units(2.0, "m")
        self._domain = pp.Domain(
            {"xmax": length, "ymax": length, "zmax": length}
        )

    def set_fractures(self) -> None:
        fx = np.vstack([[1.0, 1.0, 1.0, 1.0], [0, 2, 2, 0], [0, 0, 2, 2]])
        fy = np.vstack([[0, 2, 2, 0], [1.0, 1.0, 1.0, 1.0], [0, 0, 2, 2]])
        fz = np.vstack([[0, 2, 2, 0], [0, 0, 2, 2], [1.0, 1.0, 1.0, 1.0]])
        self._fractures = [pp.PlaneFracture(f) for f in (fx, fy, fz)]


class BaseEOS(pp.compositional.EquationOfState):
    """Simple constant-property EOS base.

    Provides placeholder implementations of phase property functions
    including dynamic viscosity, enthalpy, thermal conductivity, and density,
    with methods to return values and derivatives for compositional flow.

    Subclasses should override:
        -h_func to provide phase-enthalpy.
        -rho_func to provide phase-specific density.
        -mu_func to provide phase-viscosity.
    """

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Thermal conductivity function.

        Args:
            thermodynamic_dependencies: Variable number of arrays representing
                thermodynamic inputs.
        """
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        """Compile phase properties as a pp.compositional.PhaseProperties."""

        nc = len(thermodynamic_input[0])
        rho, drho = self.rho_func(*thermodynamic_input)  # mass density and derivatives
        h, dh = self.h(*thermodynamic_input)  # specific enthalpy and derivatives
        mu, dmu = self.mu_func(*thermodynamic_input)  # viscosity and derivatives
        kappa, dkappa = self.kappa(
            *thermodynamic_input
        )  # thermal conductivity and derivatives

        return pp.compositional.PhaseProperties(
            state=phase_state,
            rho=rho,
            drho=drho,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=np.empty((2, nc)),
            dphis=np.empty((2, 3, nc)),
        )


class WaterEOS(BaseEOS):
    """Water EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_w * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_w * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_w * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class OilEOS(BaseEOS):
    """Oil EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_o * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_o * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_o * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class GasEOS(BaseEOS):
    """Gas EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_g * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_g * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class BoundaryConditions(pp.PorePyModel):
    """Boundary conditions: the external boundary is CLOSED (all-Neumann, no-flow).

    A closed domain is the PREMISE of the conservation checks: with no flux across the external
    boundary the total mass and energy are exactly invariant, so any drift measured in
    ``after_nonlinear_convergence`` is discretization error of the buoyancy term rather than
    genuine throughput across an open boundary.  ``pp.BoundaryCondition(sd)`` with no facets
    flagged is Neumann everywhere, and the default zero Neumann values make it no-flow.

    The price is a SINGULAR pressure block: a closed, incompressible domain determines pressure
    only up to one additive constant.  That kernel is removed by the null-mean constraint
    assembled in :class:`NullMeanPressureSolve` -- NOT by opening a Dirichlet anchor facet, since
    any open facet lets buoyancy drive mass across the boundary and pollutes the very conservation
    checks this setup exists to make.
    """

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """All-Neumann: zero conductive heat flux across the external boundary."""
        return pp.BoundaryCondition(sd)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """All-Neumann: no-flow across the external boundary."""
        return pp.BoundaryCondition(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_top = 10.0e6 * to_Mega
        p = p_top * np.ones(boundary_grid.num_cells)
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = 1.0
        h = h_inlet * np.ones(boundary_grid.num_cells)
        return h

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        return np.zeros(boundary_grid.num_cells)


class NullMeanPressureSolve(pp.PorePyModel):
    """Make the CLOSED (all-Neumann) problem solvable via a null-mean pressure constraint.

    With no-flow everywhere the Jacobian is singular: adding a constant to the pressure changes no
    flux, so the constant-pressure vector spans a 1-D kernel and a plain direct solve is ill-posed.
    Rather than puncturing the domain with a Dirichlet anchor (which would break conservation), we
    fix the gauge algebraically by appending ONE constraint row ``Sum(dp_matrix) = 0`` and solving
    the bordered saddle-point system

        [[A, C^T], [C, 0]] [dx; lam] = [b; 0],

    then discarding the multiplier ``lam``.  ``C`` is the indicator of the MATRIX pressure DOFs.

    Why matrix-only: the matrix + fracture network floats by ONE common constant (a single kernel
    vector), and the fracture pressures are pinned to the matrix through the Robin-type interface
    (mortar) Darcy coupling -- they are not independently singular.  So one matrix-indicator row
    removes the kernel (the indicator has a non-zero component along it); in the fixed-dimensional
    case the matrix is the only subdomain and this is exactly a mean-zero condition on all
    pressures.

    Self-contained: implemented here with scipy only, so this setup depends on nothing beyond
    PorePy itself.

    The bordered solve itself lives in :class:`NullMeanPressureLinearSolver`: on this branch
    the nonlinear solver delegates linear solves to a ``pp.solvers.LinearSolverBase`` object
    (a model-level ``solve_linear_system`` override is never called), so the gauge constraint
    must be expressed as a linear solver and passed to the ``NewtonSolver``.  This mixin
    contributes the model-side piece: which DOFs span the pressure kernel.
    """

    def null_mean_pressure_dofs(self) -> np.ndarray:
        """Global DOF indices of the pressure on the EQUIDIMENSIONAL matrix subdomain."""
        es = self.equation_system
        matrix = self.mdg.subdomains(dim=self.nd)
        return np.asarray(es.dofs_of([es.md_variable("pressure", matrix)]), dtype=int)


class NullMeanPressureLinearSolver(pp.solvers.LinearSolverBase):
    """Direct solver for the null-mean-bordered system of :class:`NullMeanPressureSolve`,
    with an internal Schur elimination of the local secondary block.

    The local-elimination equations (``elimination_of_*``) couple each eliminated variable
    only to unknowns in its OWN cell, so the secondary block ``A_ss`` is a permuted
    block-diagonal matrix with tiny local blocks -- cheap to invert (the equation system's
    default inverter caches the permutation).  The full system is assembled ONCE through
    the standard fast path (per-equation assembly of the framework's Schur flag would
    defeat the shared-subtree AD evaluation); this solver then reduces algebraically:

        ``S = A_pp - A_ps inv(A_ss) A_sp``,  ``rhs = b_p - A_ps inv(A_ss) b_s``,

    solves the null-mean-bordered reduced system ``[[S, C^T], [C, 0]] [dx_p; lam] =
    [rhs; 0]`` (``C`` indexing the matrix pressure DOFs in reduced numbering), discards
    the multiplier, and back-substitutes ``dx_s = inv(A_ss)(b_s - A_sp dx_p)``.

    The model (pressure-kernel DOFs, elimination bookkeeping) is provided by the
    ``NewtonSolver`` through :meth:`initialize_with_model`.
    """

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        self._model = model

    @staticmethod
    def _bordered_null_mean_solve(A, b: np.ndarray, dofs: np.ndarray) -> np.ndarray:
        """Solve ``[[A, C^T], [C, 0]] [dx; lam] = [b; 0]`` and return ``dx``.

        ``C`` is one row summing the entries at ``dofs`` (the null-mean gauge).  The
        multiplier absorbs the iterate's own mass drift and vanishes as Newton converges;
        its converged size is controlled by :class:`NullSpaceDriftCriterion`, so no
        per-solve heuristic check on ``lam`` is needed here.
        """
        from scipy.sparse import bmat, csr_matrix
        from scipy.sparse.linalg import spsolve

        n = A.shape[0]
        C = csr_matrix(
            (np.ones(dofs.size), (np.zeros(dofs.size, dtype=int), dofs)), shape=(1, n)
        )
        M = bmat([[A, C.T], [C, None]], format="csc")
        rhs = np.concatenate([np.asarray(b, dtype=float), np.zeros(1)])
        x = np.atleast_1d(np.asarray(spsolve(M, rhs), dtype=float))
        return x[:n]  # drop the Lagrange multiplier

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        """Schur-reduce the local secondary block, solve the bordered reduced system,
        back-substitute; returns the full increment ``dx``."""
        import time

        t_0 = time.time()
        A, b = linear_system.matrix, linear_system.rhs
        assert A is not None, "Cannot solve a linear system whose matrix was released."
        A = A.tocsr()
        b = np.asarray(b, dtype=float)
        n = A.shape[0]
        model = self._model
        es = model.equation_system
        p_dofs = model.null_mean_pressure_dofs()

        eliminations = getattr(model, "_LocalElimination__local_eliminations", {})
        if not eliminations:
            dx = self._bordered_null_mean_solve(A, b, p_dofs)
            return dx, pp.solvers.LinearSolverStatusSuccess(solve_time=time.time() - t_0)

        # Partition into primary/secondary rows (equations) and columns (variables).
        # ``assembled_equation_indices`` reflects the system just assembled for this solve
        # (the plain full assembly rebuilds it on every call).
        sec_rows = np.sort(
            np.concatenate([es.assembled_equation_indices[name] for name in eliminations])
        )
        sec_cols = np.sort(
            np.concatenate([es.dofs_of([var]) for var, *_ in eliminations.values()])
        )
        prim_rows = np.setdiff1d(np.arange(n), sec_rows, assume_unique=True)
        prim_cols = np.setdiff1d(np.arange(n), sec_cols, assume_unique=True)

        A_pp = A[prim_rows][:, prim_cols]
        A_ps = A[prim_rows][:, sec_cols]
        A_sp = A[sec_rows][:, prim_cols]
        A_ss = A[sec_rows][:, sec_cols].tocsr()
        # Permuted block-diagonal inverse; the equation system caches the permutation.
        inv_ss = es.default_schur_complement_inverter(A_ss)

        S = (A_pp - A_ps @ inv_ss @ A_sp).tocsr()
        b_p, b_s = b[prim_rows], b[sec_rows]
        rhs = b_p - A_ps @ (inv_ss @ b_s)

        # Pressure DOFs in the reduced numbering: rank among the sorted primary columns.
        red = np.searchsorted(prim_cols, p_dofs)
        assert np.all(prim_cols[red] == p_dofs), (
            "pressure DOFs are not part of the primary block"
        )

        dx_p = self._bordered_null_mean_solve(S, rhs, red)
        dx = np.zeros(n)
        dx[prim_cols] = dx_p
        dx[sec_cols] = inv_ss @ (b_s - A_sp @ dx_p)
        return dx, pp.solvers.LinearSolverStatusSuccess(solve_time=time.time() - t_0)


class NullSpaceDriftCriterion(pp.solvers.ConvergenceCriterion):
    """Converge the residual's NULL-SPACE component: the dt-scaled total-mass drift.

    In the closed all-Neumann domain the constant-pressure vector is a left null vector of
    the Jacobian, so the summed total-mass residual -- ``(M_cur - M_old)/dt`` -- is
    invariant under any linear update and decays only quadratically.  Moreover it is a mass
    RATE: the conservation checks accumulate MASS, so the drift is scaled by ``dt`` (which
    amplifies a metric-converged rate residual by ~1e5 for day-sized steps) and normalized
    by the total volume before comparison with ``tol``.  This criterion iterates Newton
    until the accumulated-mass drift per step is genuinely below tolerance.

    The drift is only ALIVE during the nonlinear transient: once Newton enters the
    quadratic basin the increments vanish and the drift FREEZES at its current value --
    further iterations cannot change it.  If the frozen drift is above tolerance, blocking
    convergence would hang Newton at ``max_iter``; instead the criterion detects the
    stagnation and stops objecting, so the run proceeds and the test's
    ``assert_null_space_residual_converged`` fails FAST with the frozen drift in the
    message rather than burning 50 iterations and a retry.
    """

    #: Consecutive checks with relative drift change below this are considered frozen.
    _stagnation_rtol: float = 1.0e-3
    _stagnation_checks: int = 3

    def __init__(self, model: pp.PorePyModel, tol: float) -> None:
        self._model = model
        self.tol = tol
        self._history: list[float] = []
        self._total_volume: float | None = None

    def reset(self) -> None:
        self._history = []

    def check(
        self, residual: np.ndarray, **kwargs
    ) -> tuple[pp.solvers.ConvergenceStatus, float]:
        model = self._model
        rows = model.equation_system.assembled_equation_indices["mass_balance_equation"]
        if self._total_volume is None:
            # The geometry is fixed, so the normalization volume is computed once.
            self._total_volume = sum(
                np.sum(
                    model.equation_system.evaluate(
                        model.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                    )
                )
                for sd in model.mdg.subdomains()
            )
        total_volume = self._total_volume
        drift = float(
            abs(np.sum(np.asarray(residual, dtype=float)[rows]))
            * model.time_manager.dt
            / total_volume
        )
        self._history.append(drift)
        if drift <= self.tol:
            return pp.solvers.ConvergenceStatus.CONVERGED, drift
        # Stagnation escape: the drift has frozen (quadratic basin) and cannot improve.
        recent = self._history[-self._stagnation_checks :]
        if len(recent) == self._stagnation_checks and all(
            abs(a - b) <= self._stagnation_rtol * max(abs(b), 1e-300)
            for a, b in zip(recent[:-1], recent[1:])
        ):
            return pp.solvers.ConvergenceStatus.CONVERGED, drift
        return pp.solvers.ConvergenceStatus.CONTINUE_ITERATING, drift


class SecondaryEquations(LocalElimination):
    """Base class for Secondary relations (2N or 3N)."""

    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def __init__(
        self,
        *args,
        saturation_functions_map: dict[str, Callable],
        chi_functions_map: dict[str, Callable],
        temperature_function: Callable,
        **kwargs,
    ):
        # Pass positional + keyword args upstream
        super().__init__(*args, **kwargs)

        self._saturation_functions_map = saturation_functions_map
        self._chi_functions_map = chi_functions_map
        self._temperature_function = temperature_function

    def set_equations(self) -> None:
        """Register eliminations."""
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
        # other phases independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        # Saturation eliminations
        for phase in independent_phases:
            if phase.name in self._saturation_functions_map:
                self.eliminate_locally(
                    phase.saturation,
                    self.dependencies_of_phase_properties(phase),
                    self._saturation_functions_map[phase.name],
                    subdomains_and_matrix,
                )

        # Partial fractions eliminations
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    key = f"{comp.name}_{phase.name}"
                    if key in self._chi_functions_map:
                        self.eliminate_locally(
                            phase.partial_fraction_of[comp],
                            self.dependencies_of_phase_properties(phase),
                            self._chi_functions_map[key],
                            subdomains_and_matrix,
                        )

        # Temperature elimination
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),
            self._temperature_function,
            subdomains_and_matrix,
        )


class _MemoizedSurrogateFactory(pp.ad.SurrogateFactory):
    """SurrogateFactory returning one shared operator per domain set.

    A phase property (density, enthalpy, ...) referenced many times then appears as a
    single AD subtree instead of a rebuilt one. Bit-exact: values live in the data and
    are re-read at parse time, so the shared operator always reflects the current state.
    """

    def __call__(self, domains):
        cache = self.__dict__.setdefault("_op_cache", {})
        key = tuple(domains)
        if key not in cache:
            cache[key] = super().__call__(domains)
        return cache[key]


class BaseFlowModel(pp.PorePyModel):
    """Template-agnostic flow behaviour; the flow template is attached by the concrete
    ``BuoyancyFlowModel*`` classes (see :func:`buoyancy_flow_model`), one statically
    declared per ``fractional_flow`` template."""

    def __init__(self, params: dict):
        """Initialize flow model."""
        super().__init__(params)
        self.expected_order_loss = params.get("expected_order_loss", 10)

    # ------------------------------------------------------------------ conservation checks
    @staticmethod
    def conservation_order(loss: float) -> float:
        """Decades by which a NORMALIZED conservation ``loss`` sits below one (1e-4 -> 4).

        Mind the sign.  An earlier version used ``abs(floor(log10(loss)))``, which is symmetric
        about ``loss = 1``: it reported a catastrophic imbalance of 1e+4 as "order 4" -- passing
        the *same* assertion as a perfectly conserved run -- and, as a loss grew through
        1e-3 ... 1e+3 and out the other side, flipped the test from failing back to PASSING.  That
        non-monotonicity is what made these tests fail only intermittently.  Without the ``abs``
        the metric is monotone: a larger loss always yields a smaller order.
        """
        if loss <= 0.0:
            return np.inf                       # exact conservation
        return -np.floor(np.log10(loss))

    def assert_conserved(self, name: str, loss: float) -> None:
        """Assert the normalized conservation ``loss`` meets ``expected_order_loss``."""
        order = self.conservation_order(loss)
        assert order >= self.expected_order_loss, (
            f"{name} not conserved: normalized loss {loss:.6e} -> order {order:.0f}, "
            f"required order >= {self.expected_order_loss} "
            f"(i.e. loss below {10.0 ** -(self.expected_order_loss):.0e})"
        )

    def assert_buoyancy_reciprocal(self) -> None:
        """The component buoyancy fluxes must cancel -- they only redistribute mass internally.

        Summed over ALL components (a subset need not cancel), and measured RELATIVE to the size
        of the fluxes themselves: comparing against ``np.isclose(..., 0.0)``'s absolute 1e-8 is
        meaningless once the fluxes carry physical magnitude.
        """
        sds = self.mdg.subdomains()
        vals = [
            np.asarray(self.equation_system.evaluate(self.component_buoyancy(c, sds)), float)
            for c in self.fluid.components
        ]
        residual = float(np.max(np.abs(sum(vals))))
        scale = max((float(np.max(np.abs(v))) for v in vals), default=0.0)
        tol = 1.0e-8 * max(scale, 1.0)
        assert residual <= tol, (
            f"component buoyancy fluxes are not reciprocal: max|sum_c b_c| = {residual:.6e} "
            f"> {tol:.3e} (individual flux scale {scale:.3e})"
        )

    def ic_saturations(self, sd: pp.Grid) -> dict[str, np.ndarray]:
        """Initial saturation per NON-reference phase (the reference phase is by-unity).

        Implemented per phase count, since the initial saturations are named differently there.
        """
        raise NotImplementedError

    def assert_saturations_evolved(self, min_change: float = 1.0e-3) -> None:
        """The buoyant overturning must actually REDISTRIBUTE the phases.

        Without this the conservation assertions are vacuous: a state frozen at the initial
        condition conserves mass and energy *exactly*, so every check above would pass while
        proving nothing about the buoyancy discretization.  Asserting the saturation distribution
        has moved away from the initial state is what makes the conservation checks meaningful.
        """
        per_phase: dict[str, float] = {}
        for sd in self.mdg.subdomains():
            ic = self.ic_saturations(sd)
            for phase in self.fluid.phases:
                if phase.name not in ic:
                    continue
                cur = np.asarray(
                    self.equation_system.evaluate(phase.saturation([sd])), float
                )
                change = float(np.max(np.abs(cur - np.asarray(ic[phase.name], float))))
                per_phase[phase.name] = max(per_phase.get(phase.name, 0.0), change)
        max_change = max(per_phase.values(), default=0.0)
        assert max_change > min_change, (
            f"saturations did not evolve away from the initial state: max |s - s_ic| = "
            f"{max_change:.3e} <= {min_change:.0e} (per phase: "
            f"{ {k: f'{v:.2e}' for k, v in per_phase.items()} }). The buoyancy driver is inert, "
            f"so the conservation checks are vacuous."
        )

    def assert_external_bcs_are_neumann(self) -> None:
        """Every external boundary facet must be Neumann (no-flow) -- a CLOSED domain.

        The conservation checks compare the current mass/energy against the INITIAL state, which is
        only a valid reference for a closed system: a single Dirichlet facet lets buoyancy drive
        mass across the boundary, and the measured "loss" would then be real throughput rather than
        the discretization error the test is meant to quantify.
        """
        for sd in self.mdg.subdomains():
            external = sd.tags["domain_boundary_faces"]
            if not np.any(external):
                continue
            for name, bc in (
                ("darcy_flux", self.bc_type_darcy_flux(sd)),
                ("fourier_flux", self.bc_type_fourier_flux(sd)),
            ):
                n_open = int(np.count_nonzero(np.asarray(bc.is_dir)[external]))
                assert n_open == 0, (
                    f"bc_type_{name} on subdomain dim={sd.dim}: {n_open} external boundary facets "
                    f"are Dirichlet, but the conservation checks require a CLOSED (all-Neumann) "
                    f"domain"
                )

    def assert_initial_pressure_is_null_mean(self, tol: float = 1.0e-10) -> None:
        """The initial matrix pressure must satisfy the same gauge the solve pins.

        :class:`NullMeanPressureSolve` enforces ``Sum(dp_matrix) = 0`` on every Newton increment,
        which only holds the field at a null mean if the INITIAL field already has one.
        """
        for sd in self.mdg.subdomains(dim=self.nd):
            mean = float(np.mean(self.ic_values_pressure(sd)))
            assert abs(mean) <= tol, (
                f"initial matrix pressure is not null-mean (mean = {mean:.3e} > {tol:.0e}); it must "
                f"match the Sum(p_matrix) = 0 gauge fixed by NullMeanPressureSolve"
            )

    def conservation_integrals(self) -> dict[str, tuple[float, float]]:
        """``{quantity: (reference, numerical)}`` volume integrals; implemented per phase count."""
        raise NotImplementedError

    def null_space_mass_drift(self) -> float:
        """The dt-scaled, volume-normalized total-mass drift of the CURRENT residual.

        Summing the rows of the total-mass balance telescopes all interior and interface
        fluxes in the closed domain, leaving ``(M_cur - M_old)/dt`` -- a mass RATE.  The
        conservation checks accumulate MASS, so the residual is scaled by ``dt`` and
        normalized by the total pore-weighted volume to be comparable with them.  Measured
        here (2N/3N): the COMPONENT mass equations telescope to machine precision; the
        entire conservation loss stems from this total-mass row sum, which sits far below
        the Newton residual metric yet is amplified by ``dt`` (a day) into the loss.
        """
        eq = self.equation_system.equations["mass_balance_equation"]
        r = np.asarray(self.equation_system.evaluate(eq), dtype=float)
        total_volume = sum(
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                )
            )
            for sd in self.mdg.subdomains()
        )
        return float(abs(np.sum(r)) * self.time_manager.dt / total_volume)

    def assert_pressure_null_mean_converged(self) -> None:
        """The CONVERGED pressure must satisfy the gauge: null mean on the matrix.

        The Lagrange multiplier of the bordered solve enforces ``Sum(dp_matrix) = 0`` on
        every Newton increment, and the initial pressure is null-mean, so the converged
        pressure must have a null matrix mean to solver precision.  A violation would mean
        the gauge constraint is not actually holding the solution's pressure level.
        """
        matrix = self.mdg.subdomains(dim=self.nd)
        p = np.concatenate(
            [
                np.asarray(self.equation_system.evaluate(self.pressure([sd])), float)
                for sd in matrix
            ]
        )
        mean = float(abs(np.mean(p)))
        tol = float(self.params["residual_tolerance"])
        assert mean <= tol, (
            f"converged matrix pressure is not null-mean: |mean(p_matrix)| = {mean:.6e} > "
            f"tol = {tol:.1e}; the gauge constraint is not holding the pressure level"
        )

    def assert_null_space_residual_converged(self) -> None:
        """The NULL-SPACE (total-mass drift) component of the CONVERGED residual must be
        below the Newton tolerance IN THE UNITS THE CONSERVATION CHECKS MEASURE.

        In the closed all-Neumann domain the constant-pressure vector is a LEFT null vector
        of the Jacobian, so the drift is invariant under any linear update and decays only
        through second-order (quadratic Newton) effects -- and, crucially, the residual is a
        mass RATE while conservation measures accumulated MASS: a rate residual far below
        the metric tolerance still allows a drift of ``dt * sum(r)`` per step.  If this
        assertion fails, the convergence criteria must be extended with
        :class:`NullSpaceDriftCriterion`.
        """
        drift = self.null_space_mass_drift()
        tol = float(self.params["residual_tolerance"])
        assert drift <= tol, (
            f"null-space residual (total-mass drift) of the converged state is not below the "
            f"Newton tolerance: |sum of mass-balance residuals| * dt / V = {drift:.6e} > "
            f"tol = {tol:.1e}. The convergence criteria control the residual metric (a mass "
            f"RATE) but not the dt-scaled drift the conservation checks accumulate -- add "
            f"NullSpaceDriftCriterion to the convergence criteria."
        )

    def assert_reference_matches_state(self, tol: float | None = None) -> None:
        """At t=0 every REFERENCE must equal its NUMERICAL counterpart EXACTLY.

        The conservation checks measure ``|ref - num|`` and attribute it to the buoyancy
        discretization.  That attribution is only valid if the two agree at t=0, before a single
        step: the reference is hand-built from the initial condition (``sum_i s_i_ic * rho_i``),
        while the numerical value comes from the model (``self.fluid.density``).  Any mismatch in
        the mixing rule, the by-unity reference phase, or the initial fields shows up as a CONSTANT
        offset that is present from the start and has nothing to do with conservation -- it would
        make the test blame the scheme for a bug in its own reference.  Checking it at t=0 is what
        separates "the reference is wrong" from "the scheme does not conserve".

        The default tolerance is MACHINE PRECISION (a few ULPs, relative): at t=0 reference and
        numerical integrals are computed from the same initial arrays, differing only in
        floating-point summation order, so EVERY quantity entering the conservation assertions
        must match to rounding error -- anything above a few ULPs is a genuine inconsistency in
        the reference (mixing rule, by-unity phase, or initial fields), not rounding.
        """
        if tol is None:
            tol = 4.0 * np.finfo(float).eps
        for name, (ref, num) in self.conservation_integrals().items():
            mismatch = abs(ref - num)
            assert mismatch <= tol * max(abs(ref), 1.0), (
                f"{name}: the t=0 reference does not match the initial state EXACTLY: "
                f"|ref - num| = {mismatch:.6e} (ref {ref:.16e}, num {num:.16e}). The conservation "
                f"reference is inconsistent with the model's own initial fields, so any later "
                f"'loss' measures this offset rather than the buoyancy discretization."
            )

    def prepare_simulation(self) -> None:
        """Validate the closed-domain premise, and the reference itself, before spending the run."""
        super().prepare_simulation()
        self.assert_external_bcs_are_neumann()
        self.assert_initial_pressure_is_null_mean()
        self.assert_reference_matches_state()

    def after_simulation(self) -> None:
        """Conservation only means something if the phases actually moved."""
        super().after_simulation()
        self.assert_saturations_evolved()

    def update_derived_quantities(self) -> None:
        """Install (once) a memoized full-iterate fetch on the equation system, then update.

        Every per-grid "flash" inside the after-iteration update re-fetches the ENTIRE system
        state.  There are TWO such grid-by-grid loops, and they run in different classes:
          * the phase-property update (density, enthalpy) in ``compositional_flow.py``, and
          * the locally-eliminated secondaries (T, s, x) in
            ``abstract_equations.LocalElimination``.
        Both evaluate their dependencies per grid with ``state=None``, so ``_ad_parser.evaluate``
        calls ``equation_system.get_variable_values(iterate_index=0)`` -- ALL variables on ALL
        subdomains, a ``numpy.copy`` per sub-variable -- for EVERY grid: O(n_subdomains^2).

        ``LocalElimination.update_derived_quantities`` is the MRO entry point and runs its loop
        AFTER its ``super()`` call, so a scoped (install/teardown) patch placed here would be torn
        down too early.  Instead a PERSISTENT memoization of the full-iterate fetch is installed
        (:meth:`_install_full_iterate_cache`), invalidated by a generation counter bumped on every
        variable-value write.  The primaries do not change during a single update and every flash
        dependency is a primary, so all flashes in one update share one fetch -- bit-exact,
        O(n_subdomains^2) -> O(n_subdomains).
        """
        self._install_full_iterate_cache()
        super().update_derived_quantities()

    def _install_full_iterate_cache(self) -> None:
        """Wrap ``equation_system.get_variable_values`` so the full-iterate fetch
        (``iterate_index=0``, no variable subset) is memoized until the next variable-value
        write.

        ``set_variable_values`` / ``shift_iterate_values`` bump a generation counter that
        invalidates the cache (``shift_time_step_values`` only moves iterate storage to
        time-step storage and never alters iterate 0, so it needs no wrap); all other fetches
        (subsets, other indices, reference values) pass through unchanged.  Idempotent
        (installs once per equation system).  A fresh copy is returned per call, so callers
        that mutate the result stay correct.
        """
        es = self.equation_system
        if getattr(es, "_full_iterate_cache", None) is not None:
            return
        cache: dict = {"gen": 0, "cached_gen": -1, "value": None}
        es._full_iterate_cache = cache
        _get, _set, _shift = (
            es.get_variable_values,
            es.set_variable_values,
            es.shift_iterate_values,
        )

        def get_variable_values(
            variables=None, time_step_index=None, iterate_index=None, reference=False
        ):
            if (
                variables is None
                and time_step_index is None
                and iterate_index == 0
                and not reference
            ):
                if cache["cached_gen"] != cache["gen"]:
                    cache["value"] = _get(iterate_index=0)
                    cache["cached_gen"] = cache["gen"]
                return cache["value"].copy()
            return _get(
                variables=variables,
                time_step_index=time_step_index,
                iterate_index=iterate_index,
                reference=reference,
            )

        def set_variable_values(*args, **kwargs):
            cache["gen"] += 1
            return _set(*args, **kwargs)

        def shift_iterate_values(*args, **kwargs):
            cache["gen"] += 1
            return _shift(*args, **kwargs)

        es.get_variable_values = get_variable_values  # type: ignore[method-assign]
        es.set_variable_values = set_variable_values  # type: ignore[method-assign]
        es.shift_iterate_values = shift_iterate_values  # type: ignore[method-assign]

    def assign_thermodynamic_properties_to_phases(self) -> None:
        """Memoize each phase-property surrogate so it is a single shared subtree."""
        super().assign_thermodynamic_properties_to_phases()
        for phase in self.fluid.phases:
            for attr in vars(phase).values():
                if isinstance(attr, pp.ad.SurrogateFactory):
                    attr.__class__ = _MemoizedSurrogateFactory

    @pp.ad.cached_method
    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """kr = saturation.

        Cached so every reference to a phase's relative permeability (mobility, fractional
        mobility and buoyancy terms all pull it in) shares one operator subtree, keeping
        the AD graph a DAG instead of duplicating the saturation subtree per use.
        """
        return phase.saturation(domains)

    def set_equations(self):
        """Set equations + buoyancy params."""
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        """Register nonlinear discretizations."""
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        """Update buoyancy fluxes."""
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    @pp.ad.cached_method
    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Gravity magnitude field.

        Cached so the (constant) gravity array is built once per subdomain set and shared
        by every consumer (Darcy vector source and each phase-pair buoyancy flux) instead
        of allocating an identical dense array per reference.
        """
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field


# constitutive description for N=2
def temperature_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature correlation (zeroed)."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    # Set temperature to zero to isolate
    # the effect of energy convection driven by buoyancy.
    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def gas_saturation_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gas saturation correlation."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4 * rho_w) / (z_CH4 * rho_w + rho_g - z_CH4 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_w * rho_g) / (
        (z_CH4 * (rho_w - rho_g) + rho_g) * (z_CH4 * (rho_w - rho_g) + rho_g)
    )
    return vals, diffs


def CH4_water_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in water fraction."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_gas_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in gas fraction."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map_2N = {
    "CH4_water": CH4_water_2N,
    "CH4_gas": CH4_gas_2N,
}


# Two phases Two components case
class FluidMixture2N(pp.PorePyModel):
    """2-phase (water-gas), 2-component mixture."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        component_1 = pp.FluidComponent(name="H2O")
        component_2 = pp.FluidComponent(name="CH4")
        return [component_1, component_2]

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_W = WaterEOS(components)
        eos_G = GasEOS(components)
        configuration_W = (pp.compositional.PhysicalState.liquid, "water", eos_W)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_W, configuration_G]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z  # type:ignore[return-value]


class SecondaryEquations2N(SecondaryEquations):
    """Secondary (eliminated) relations 2N."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            saturation_functions_map={"gas": gas_saturation_2N},
            chi_functions_map=chi_functions_map_2N,
            temperature_function=temperature_2N,
            *args,
            **kwargs,
        )


class InitialConditions2N(pp.PorePyModel):
    """Initial conditions 2N."""

    def initial_condition(self) -> None:
        """Set initial fields."""
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_saturation(sd)
            x_CH4_liq_v = np.zeros_like(s_gas_val)
            x_CH4_gas_v = np.ones_like(s_gas_val)

            x_CH4_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CH4_liq_v, [x_CH4_liq], 0, 0)
            self.equation_system.set_variable_values(x_CH4_gas_v, [x_CH4_gas], 0, 0)

    def ic_values_saturation(self, sd: pp.Grid) -> np.ndarray:
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_w) / (z_v * rho_w + rho_g - z_v * rho_g)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """NULL-MEAN initial pressure (identically zero).

        The closed all-Neumann domain fixes pressure only up to an additive constant, and the solve
        pins that gauge with ``Sum(p_matrix) = 0`` (see :class:`NullMeanPressureSolve`).  The
        initial state must satisfy the SAME gauge, otherwise step one would shift the whole field
        by a constant to reach it.  The fluid is incompressible with a constant-property EOS, so
        only ``grad(p)`` is physical and the level is a free gauge -- a uniform field with zero mean
        is simply zero.
        """
        return np.zeros(sd.num_cells)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        ic_s = self.ic_values_saturation(sd)
        ic_rho = rho_g * ic_s + rho_w * (1.0 - ic_s)
        h = (ic_s * h_g * rho_g + (1.0 - ic_s) * h_w * rho_w) / ic_rho
        return np.ones(sd.num_cells) * h

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        # Horizontally layered initial condition: the composition depends only on the
        # vertical (gravity) coordinate -- y in 2D, z in 3D -- so every horizontal layer
        # is laterally uniform. This keeps the density constant along any horizontal
        # boundary plane, which makes the conservation checks agnostic to the number of
        # fixed-pressure facets on that plane.
        vert = sd.cell_centers[self.nd - 1]
        z = np.where((vert >= 1.0) & (vert <= 2.0), 0.5, 0.0) + np.where(
            (vert >= 3.0) & (vert <= 4.0), 0.5, 0.0
        )
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel2N(
    BaseFlowModel,
):
    def after_nonlinear_convergence(self) -> None:
        """Post-convergence diagnostics."""
        # The gauge (null-mean pressure) and the drift must be measured BEFORE super()
        # shifts the time-step solutions (``x_prev <- x_cur``): after the shift the
        # accumulation term of the reassembled residual is zero and the drift measurement
        # is vacuous.
        self.assert_pressure_null_mean_converged()
        self.assert_null_space_residual_converged()
        super().after_nonlinear_convergence()

        # Buoyancy flux reciprocity (summed over ALL components, relative to the flux scale).
        self.assert_buoyancy_reciprocal()
        for name, (ref, num) in self.conservation_integrals().items():
            self.assert_conserved(name, abs(ref - num))
        # Every CONVERGED state must have genuinely redistributed the phases: conservation of
        # a state frozen at the initial condition would be vacuous.
        self.assert_saturations_evolved()

    def conservation_integrals(self) -> dict[str, tuple[float, float]]:
        """``{quantity: (reference, numerical)}`` volume integrals, normalized by total volume.

        The REFERENCE is rebuilt from the initial condition every call (the EOS is
        constant-property, so it is a genuine time-invariant); the NUMERICAL one is the current
        state.  Evaluated through one code path so the very same expressions can be checked for
        exact agreement at t=0 (:meth:`BaseFlowModel.assert_reference_matches_state`) -- if they
        disagree there, the "loss" reported later is a broken reference, not a conservation defect.
        """
        subdomains = self.mdg.subdomains()
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)

        # Total volume
        total_volume = sum(
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                )
            )
            for sd in subdomains
        )

        def norm_vol_int(op: pp.ad.Operator, sd: pp.Grid) -> float:
            return (
                np.sum(
                    self.equation_system.evaluate(self.volume_integral(op, [sd], dim=1))
                )
                / total_volume
            )

        # Reference and numerical accumulators
        ref_rho = ref_rho_z = ref_energy = 0.0
        num_rho = num_rho_z = num_energy = 0.0

        for sd in subdomains:
            ic_sg = self.ic_values_saturation(sd)
            rho_l = phases[0].density([sd])
            rho_g = phases[1].density([sd])

            ic_rho = (
                pp.wrap_as_dense_ad_array(1.0 - ic_sg) * rho_l
                + pp.wrap_as_dense_ad_array(ic_sg) * rho_g
            )
            ref_rho += norm_vol_int(ic_rho, sd)

            ic_z = self.ic_values_overall_fraction(components[1], sd)
            ic_rho_z = ic_rho * pp.wrap_as_dense_ad_array(ic_z)
            ref_rho_z += norm_vol_int(ic_rho_z, sd)

            ic_p = self.ic_values_pressure(sd)
            ic_h = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(
                ic_h
            ) - pp.wrap_as_dense_ad_array(ic_p)
            ref_energy += norm_vol_int(ic_energy, sd)

            cur_rho = self.fluid.density([sd])
            num_rho += norm_vol_int(cur_rho, sd)

            cur_rho_z = cur_rho * components[1].fraction([sd])
            num_rho_z += norm_vol_int(cur_rho_z, sd)

            cur_energy = cur_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy += norm_vol_int(cur_energy, sd)

        return {
            "total mass": (ref_rho, num_rho),
            f"component mass ({components[1].name})": (ref_rho_z, num_rho_z),
            "energy": (ref_energy, num_energy),
        }

    def ic_saturations(self, sd: pp.Grid) -> dict[str, np.ndarray]:
        """Initial gas saturation; the liquid (reference) phase follows by unity."""
        return {list(self.fluid.phases)[1].name: self.ic_values_saturation(sd)}


# The concrete 2N/3N buoyancy models are assembled by :func:`buoyancy_flow_model` at the
# end of this module, so the flow template can be selected via ``fractional_flow``.


# constitutive description for N=3
def temperature_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature correlation (zeroed)."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    # Set temperature to zero to isolate
    # the effect of energy convection driven by buoyancy.
    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def oil_saturation_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Oil saturation correlation."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_C5H12 * rho_g * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -(
        (z_C5H12 * rho_g * rho_w * (-(rho_g * rho_o) + rho_g * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    ) + (rho_g * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    diffs[3, :] = -(
        (z_C5H12 * rho_g * rho_w * (-(rho_g * rho_o) + rho_o * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    )
    return vals, diffs


def gas_saturation_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gas saturation correlation."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4 * rho_o * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -(
        (z_CH4 * rho_o * rho_w * (-(rho_g * rho_o) + rho_g * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    )
    diffs[3, :] = -(
        (z_CH4 * rho_o * rho_w * (-(rho_g * rho_o) + rho_o * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    ) + (rho_o * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    return vals, diffs


saturation_functions_map_3N = {
    "oil": oil_saturation_3N,
    "gas": gas_saturation_3N,
}


def C5H12_water_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in water."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def C5H12_oil_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in oil."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def C5H12_gas_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in gas."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_water_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in water."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_oil_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in oil."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_gas_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in gas."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map_3N = {
    "C5H12_water": C5H12_water_3N,
    "C5H12_oil": C5H12_oil_3N,
    "C5H12_gas": C5H12_gas_3N,
    "CH4_water": CH4_water_3N,
    "CH4_oil": CH4_oil_3N,
    "CH4_gas": CH4_gas_3N,
}


class FluidMixture3N(pp.PorePyModel):
    """3-phase (water-oil-gas), 3-component mixture."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        component_1 = pp.FluidComponent(name="H2O")
        component_2 = pp.FluidComponent(name="C5H12")
        component_3 = pp.FluidComponent(name="CH4")
        return [component_1, component_2, component_3]

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = WaterEOS(components)
        eos_O = OilEOS(components)
        eos_G = GasEOS(components)
        configuration_W = (pp.compositional.PhysicalState.liquid, "water", eos_L)
        configuration_O = (pp.compositional.PhysicalState.liquid, "oil", eos_O)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_W, configuration_O, configuration_G]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z  # type:ignore[return-value]


class SecondaryEquations3N(SecondaryEquations):
    """Secondary relations 3N."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            saturation_functions_map=saturation_functions_map_3N,
            chi_functions_map=chi_functions_map_3N,
            temperature_function=temperature_3N,
            *args,
            **kwargs,
        )


class InitialConditions3N(pp.PorePyModel):
    """Initial conditions 3N."""

    def initial_condition(self) -> None:
        """Set initial fields."""
        super().initial_condition()

        # set the values to be the custom functions
        water, oil, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_oil_val = self.ic_values_saturation_oil(sd)
            s_gas_val = self.ic_values_saturation_gas(sd)
            self.equation_system.set_variable_values(
                s_oil_val, [oil.saturation([sd])], 0, 0
            )
            self.equation_system.set_variable_values(
                s_gas_val, [gas.saturation([sd])], 0, 0
            )

            x_inactive_v = np.zeros_like(s_oil_val)
            x_active_v = np.ones_like(s_gas_val)

            x_C5H12_water = water.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_oil = oil.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            x_CH4_water = water.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_oil = oil.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[2]]([sd])

            self.equation_system.set_variable_values(
                x_inactive_v, [x_C5H12_water], 0, 0
            )
            self.equation_system.set_variable_values(x_active_v, [x_C5H12_oil], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_C5H12_gas], 0, 0)

            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_water], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_oil], 0, 0)
            self.equation_system.set_variable_values(x_active_v, [x_CH4_gas], 0, 0)

    def ic_values_saturation_oil(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        so_val = (z_C5H12 * rho_g * rho_w) / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        return so_val

    def ic_values_saturation_gas(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        sg_val = (z_CH4 * rho_o * rho_w) / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        return sg_val

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """NULL-MEAN initial pressure (identically zero).

        The closed all-Neumann domain fixes pressure only up to an additive constant, and the solve
        pins that gauge with ``Sum(p_matrix) = 0`` (see :class:`NullMeanPressureSolve`).  The
        initial state must satisfy the SAME gauge, otherwise step one would shift the whole field
        by a constant to reach it.  The fluid is incompressible with a constant-property EOS, so
        only ``grad(p)`` is physical and the level is a free gauge -- a uniform field with zero mean
        is simply zero.
        """
        return np.zeros(sd.num_cells)

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # Mass-weighted mixture specific enthalpy, consistent with the initial
        # saturations: h = (Σ s_i ρ_i h_i) / (Σ s_i ρ_i). A constant value would be
        # inconsistent with the initial phase distribution and spoils energy
        # conservation.
        s_o = self.ic_values_saturation_oil(sd)
        s_g = self.ic_values_saturation_gas(sd)
        s_w = 1.0 - s_o - s_g
        ic_rho = s_w * rho_w + s_o * rho_o + s_g * rho_g
        return (
            s_w * h_w * rho_w + s_o * h_o * rho_o + s_g * h_g * rho_g
        ) / ic_rho

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        # Horizontally layered initial condition: the composition depends only on the
        # vertical (gravity) coordinate -- y in 2D, z in 3D -- so every horizontal layer
        # is laterally uniform (see the 2N counterpart for the rationale).
        vert = sd.cell_centers[self.nd - 1]
        z = np.where((vert >= 1.0) & (vert <= 2.0), 1 / 6.0, 0.0) + np.where(
            (vert >= 3.0) & (vert <= 4.0), 1 / 6.0, 0.0
        )
        return z * np.ones(sd.num_cells)


class FlowModel3N(
    BaseFlowModel,
):
    def after_nonlinear_convergence(self) -> None:
        """Post-convergence diagnostics."""
        # The gauge (null-mean pressure) and the drift must be measured BEFORE super()
        # shifts the time-step solutions (``x_prev <- x_cur``): after the shift the
        # accumulation term of the reassembled residual is zero and the drift measurement
        # is vacuous.
        self.assert_pressure_null_mean_converged()
        self.assert_null_space_residual_converged()
        super().after_nonlinear_convergence()

        # Buoyancy flux reciprocity (summed over ALL components, relative to the flux scale).
        self.assert_buoyancy_reciprocal()
        for name, (ref, num) in self.conservation_integrals().items():
            self.assert_conserved(name, abs(ref - num))
        # Every CONVERGED state must have genuinely redistributed the phases: conservation of
        # a state frozen at the initial condition would be vacuous.
        self.assert_saturations_evolved()

    def conservation_integrals(self) -> dict[str, tuple[float, float]]:
        """``{quantity: (reference, numerical)}`` volume integrals; see the 2N counterpart."""
        phases = list(self.fluid.phases)  # water, oil, gas
        components = list(self.fluid.components)  # H2O (ref), C5H12, CH4

        # Total volume for normalization
        total_volume = 0.0
        for sd in self.mdg.subdomains():
            vol_op = self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
            total_volume += np.sum(self.equation_system.evaluate(vol_op))

        # Reference (initial) and numerical integrals
        ref_rho_integral = 0.0
        num_rho_integral = 0.0

        ref_rho_c1_integral = 0.0
        ref_rho_c2_integral = 0.0
        num_rho_c1_integral = 0.0
        num_rho_c2_integral = 0.0

        ref_energy_integral = 0.0
        num_energy_integral = 0.0

        # Loop subdomains
        for sd in self.mdg.subdomains():
            # Initial saturations
            ic_so = self.ic_values_saturation_oil(sd)
            ic_sg = self.ic_values_saturation_gas(sd)
            ic_sw = 1.0 - ic_so - ic_sg

            # Phase densities (AD operators)
            rho_w = phases[0].density([sd])
            rho_o = phases[1].density([sd])
            rho_g = phases[2].density([sd])

            # Initial mixture density (AD)
            ic_rho = (
                pp.wrap_as_dense_ad_array(ic_sw) * rho_w
                + pp.wrap_as_dense_ad_array(ic_so) * rho_o
                + pp.wrap_as_dense_ad_array(ic_sg) * rho_g
            )

            # Initial overall fractions for non-reference components
            ic_z_c1 = self.ic_values_overall_fraction(components[1], sd)  # C5H12
            ic_z_c2 = self.ic_values_overall_fraction(components[2], sd)  # CH4

            # Reference mass integrals (normalized)
            ref_rho_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(ic_rho, [sd], dim=1)
                    )
                )
                / total_volume
            )

            ref_rho_c1_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(
                            ic_rho * pp.wrap_as_dense_ad_array(ic_z_c1), [sd], dim=1
                        )
                    )
                )
                / total_volume
            )
            ref_rho_c2_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(
                            ic_rho * pp.wrap_as_dense_ad_array(ic_z_c2), [sd], dim=1
                        )
                    )
                )
                / total_volume
            )

            # Initial energy (rho*h - p)
            ic_p = self.ic_values_pressure(sd)
            ic_h = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(
                ic_h
            ) - pp.wrap_as_dense_ad_array(ic_p)
            ref_energy_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(ic_energy, [sd], dim=1)
                    )
                )
                / total_volume
            )

            # Current mixture density and integrals
            num_rho = self.fluid.density([sd])
            num_rho_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho, [sd], dim=1)
                    )
                )
                / total_volume
            )

            num_rho_c1 = num_rho * components[1].fraction([sd])
            num_rho_c2 = num_rho * components[2].fraction([sd])
            num_rho_c1_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho_c1, [sd], dim=1)
                    )
                )
                / total_volume
            )
            num_rho_c2_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho_c2, [sd], dim=1)
                    )
                )
                / total_volume
            )

            num_energy = num_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_energy, [sd], dim=1)
                    )
                )
                / total_volume
            )

        return {
            "total mass": (ref_rho_integral, num_rho_integral),
            f"component mass ({components[1].name})": (
                ref_rho_c1_integral,
                num_rho_c1_integral,
            ),
            f"component mass ({components[2].name})": (
                ref_rho_c2_integral,
                num_rho_c2_integral,
            ),
            "energy": (ref_energy_integral, num_energy_integral),
        }

    def ic_saturations(self, sd: pp.Grid) -> dict[str, np.ndarray]:
        """Initial oil and gas saturations; water (reference) follows by unity."""
        phases = list(self.fluid.phases)                       # water, oil, gas
        return {
            phases[1].name: self.ic_values_saturation_oil(sd),
            phases[2].name: self.ic_values_saturation_gas(sd),
        }


# Statically-declared buoyancy models: one concrete class per (phase count, template)
# combination, so the MRO of each configuration is explicit and fixed at import time.
# In every class the parts are ordered fluid -> IC -> BC -> gauge -> secondary ->
# FlowModel*N -> template, so ``FlowModel*N`` -> ``BaseFlowModel`` (whose
# ``set_equations`` registers the buoyancy discretization parameters) precedes the
# template's equation setters in the MRO.
#
# NullMeanPressureSolve contributes null_mean_pressure_dofs, consumed by
# NullMeanPressureLinearSolver (which must be passed to the NewtonSolver): the closed
# all-Neumann domain leaves a singular constant-pressure mode that the default direct
# solver cannot handle.


class BuoyancyFlowModelFF2N(
    FluidMixture2N,
    InitialConditions2N,
    BoundaryConditions,
    NullMeanPressureSolve,
    SecondaryEquations2N,
    FlowModel2N,
    CompositionalFractionalFlowTemplate,
):
    """Two-phase buoyancy model on the fractional-flow template.

    Requires ``params['fractional_flow'] = True``: the flag (read by
    ``is_fractional_flow``) and the template must agree.
    """


class BuoyancyFlowModelCF2N(
    FluidMixture2N,
    InitialConditions2N,
    BoundaryConditions,
    NullMeanPressureSolve,
    SecondaryEquations2N,
    FlowModel2N,
    CompositionalFlowTemplate,
):
    """Two-phase buoyancy model on the standard compositional-flow template.

    Requires ``params['fractional_flow'] = False``: the flag (read by
    ``is_fractional_flow``) and the template must agree.
    """


class BuoyancyFlowModelFF3N(
    FluidMixture3N,
    InitialConditions3N,
    BoundaryConditions,
    NullMeanPressureSolve,
    SecondaryEquations3N,
    FlowModel3N,
    CompositionalFractionalFlowTemplate,
):
    """Three-phase buoyancy model on the fractional-flow template.

    Requires ``params['fractional_flow'] = True``: the flag (read by
    ``is_fractional_flow``) and the template must agree.
    """


class BuoyancyFlowModelCF3N(
    FluidMixture3N,
    InitialConditions3N,
    BoundaryConditions,
    NullMeanPressureSolve,
    SecondaryEquations3N,
    FlowModel3N,
    CompositionalFlowTemplate,
):
    """Three-phase buoyancy model on the standard compositional-flow template.

    Requires ``params['fractional_flow'] = False``: the flag (read by
    ``is_fractional_flow``) and the template must agree.
    """


_BUOYANCY_MODELS: dict[tuple[int, bool], type] = {
    (2, True): BuoyancyFlowModelFF2N,
    (2, False): BuoyancyFlowModelCF2N,
    (3, True): BuoyancyFlowModelFF3N,
    (3, False): BuoyancyFlowModelCF3N,
}


def buoyancy_flow_model(n_phases: int, fractional_flow: bool = True) -> type:
    """Return the statically-declared N-phase buoyancy model for the requested template.

    ``fractional_flow=True`` selects the ``CompositionalFractionalFlowTemplate`` variant,
    ``False`` the ``CompositionalFlowTemplate`` one.  The caller must set the matching
    ``params['fractional_flow']`` flag on the model.
    """
    return _BUOYANCY_MODELS[(n_phases, fractional_flow)]