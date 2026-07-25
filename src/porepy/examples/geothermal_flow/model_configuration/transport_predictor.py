"""Flow-order (reordered) transport PREDICTOR for the geothermal CF models.

This is NOT a solver and does NOT replace the fully-implicit (FI) Newton loop.  It is a cheap,
standalone sweep run once at the start of each time step (``before_nonlinear_loop``) that advances
the transported primaries ``enthalpy`` and ``z_<comp>`` along the flow direction, so the FI Newton
starts from a state whose fronts are already roughly in place -- the cure for the high-CFL
fractures/intersections/points stiffness (the fronts race through the fast pathways, so the
previous-step state is a poor Newton guess there).

Why it is cheap and stable:
  * The **flux graph is read, not built** -- PorePy's upwind discretization already stores, per face,
    the upstream cell (matrix ``"transport"`` under keyword ``"mobility"``), and per mortar the
    upwind-primary/secondary selection.  We reuse those matrices as-is (never re-discretize).
  * Without gravity ``u = -Kλ_t∇p`` is a gradient field ⇒ the cell-to-cell graph is ACYCLIC, so a
    topological order exists (Kahn).  Cells in the same topological LEVEL are mutually independent,
    so each level is one **vectorised** update -- cost is O(DAG depth), not O(cells) Python calls.
  * The per-cell update is the closed-form implicit upwind balance
        q_i = (acc_i·q_iᵒˡᵈ - Σ_{j upstream} A_ij·q_j) / (acc_i + A_ii)
    (monotone, unconditionally stable, one division), followed by one batched OBL flash for the
    eliminated secondaries T, s, x.

It is a PREDICTOR, so it need not be discretisation-exact; the FI Newton corrects it.  Enable with
``params["transport_predictor"] = True`` (default OFF ⇒ FI is byte-identical).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sps

import porepy as pp

_LOG = logging.getLogger(__name__)

_MOBILITY_KW = "mobility"          # keyword under which the upwind matrix + darcy_flux are stored
_UPWIND_KEY = "transport"          # Upwind.upwind_matrix_key
_FLUX_KEY = "darcy_flux"           # Upwind.flux_array_key


class ReorderedTransportPredictor:
    """Mixin providing a flow-order transport predictor for the FI Newton loop.

    Mixed into :class:`_FlowModelBaseCore`; every derived model gets it.  Inert unless
    ``params["transport_predictor"]`` is truthy."""

    # ---- toggle -------------------------------------------------------------------------------
    def transport_predictor_enabled(self) -> bool:
        return bool(self.params.get("transport_predictor", False))

    # ---- global cell indexing ----------------------------------------------------------------
    def _predictor_cell_offsets(self) -> tuple[dict, int]:
        """Contiguous global cell index per subdomain: ``{sd: offset}``, and the total cell count."""
        offset, off = {}, 0
        for sd in self.mdg.subdomains():
            offset[sd] = off
            off += sd.num_cells
        return offset, off

    # ---- cell-to-cell advective coupling from the stored upwind matrices ----------------------
    def _predictor_coupling(self, offset: dict, n_cells: int) -> tuple[sps.csr_matrix, dict]:
        """Assemble the global cell-to-cell advective mass-flux coupling ``A`` (n_cells x n_cells)
        and return it together with the per-subdomain total-mass-mobility ``mob_by_sd`` (reused for
        the Dirichlet boundary source).

        For each subdomain, ``A_sd = div @ diag(mass_flux) @ upwind`` where ``upwind`` and
        ``darcy_flux`` are read from the data dict (as last re-discretized), and
        ``mass_flux[f] = darcy_flux[f] · (upwind @ total_mass_mobility)[f]`` is the upwind total
        MASS flux on face f.  This equals ``div @ diag(darcy) @ upwind @ (mob·q)`` -- the model's own
        advection matrix applied to the passively-advected primary ``q`` (h or z), so it already
        includes the OUTFLOW boundary faces (their upstream is the interior cell).  ``A[i,j]`` (i≠j)
        ≠ 0 ⟺ j is upstream of i; ``A[i,i]`` is i's outflux.  The mortar coupling adds
        matrix↔fracture↔line↔point edges the same way (upwind selects the upstream side).  The
        Dirichlet INFLOW faces are missing here (their upstream is the boundary, not a cell) and are
        added separately by :meth:`_predictor_boundary_source`."""
        es = self.equation_system
        rows, cols, data = [], [], []
        mob_by_sd: dict = {}

        # (a) intra-subdomain coupling from the stored upwind matrix.
        for sd in self.mdg.subdomains():
            if sd.num_cells == 0:
                continue
            dm = self.mdg.subdomain_data(sd)[pp.DISCRETIZATION_MATRICES]
            par = self.mdg.subdomain_data(sd)[pp.PARAMETERS]
            if _MOBILITY_KW not in dm or _UPWIND_KEY not in dm[_MOBILITY_KW]:
                continue
            upwind = dm[_MOBILITY_KW][_UPWIND_KEY].tocsr()          # (n_faces, n_cells)
            darcy = np.asarray(par[_MOBILITY_KW][_FLUX_KEY], float)  # (n_faces,)
            mob = np.asarray(es.evaluate(self.total_mass_mobility([sd])), float)  # (n_cells,)
            mob_by_sd[sd] = mob
            face_mob = upwind @ mob                                  # upstream mobility per face
            mass_flux = darcy * face_mob                             # total mass flux per face
            div = sd.cell_faces.transpose().tocsr()                  # (n_cells, n_faces)
            A_sd = (div @ sps.diags(mass_flux) @ upwind).tocoo()
            b = offset[sd]
            rows.append(A_sd.row + b); cols.append(A_sd.col + b); data.append(A_sd.data)

        # (b) interface coupling (matrix↔fracture↔line↔point), from the mortar flux + projections.
        for intf in self.mdg.interfaces():
            lam = np.asarray(es.evaluate(self.interface_darcy_flux([intf])), float)  # mortar flux
            sd_a, sd_b = self.mdg.interface_to_subdomain_pair(intf)
            # primary_to_mortar maps to the HIGHER-dim faces -> sd_hi must be the higher-dim one.
            sd_hi, sd_lo = (sd_a, sd_b) if sd_a.dim > sd_b.dim else (sd_b, sd_a)
            p2m = intf.primary_to_mortar_int().tocsr()               # (n_mortar, n_hi_faces)
            s2m = intf.secondary_to_mortar_int().tocsr()             # (n_mortar, n_lo_cells)
            hi_cf = sd_hi.cell_faces.tocsr()   # (n_faces, n_cells): index by face (row) -> cells
            bh, bl = offset[sd_hi], offset[sd_lo]
            for mc in range(intf.num_cells):
                hf = p2m.indices[p2m.indptr[mc]:p2m.indptr[mc + 1]]
                lc = s2m.indices[s2m.indptr[mc]:s2m.indptr[mc + 1]]
                if hf.size == 0 or lc.size == 0:
                    continue
                hcells = hi_cf.indices[hi_cf.indptr[hf[0]]:hi_cf.indptr[hf[0] + 1]]
                if hcells.size == 0:
                    continue
                gh, gl = bh + hcells[0], bl + lc[0]
                w = abs(lam[mc])
                # lam>0: higher->lower dim ; lam<0: lower->higher (convention checked empirically).
                if lam[mc] > 0:                       # edge gh -> gl : gl's inflow from gh
                    rows += [gl, gh]; cols += [gh, gh]; data += [-w, +w]
                elif lam[mc] < 0:
                    rows += [gh, gl]; cols += [gl, gl]; data += [-w, +w]

        A = sps.csr_matrix(
            (np.concatenate(data) if data else np.zeros(0),
             (np.concatenate(rows) if rows else np.zeros(0, int),
              np.concatenate(cols) if cols else np.zeros(0, int))),
            shape=(n_cells, n_cells))
        A.sum_duplicates()
        return A, mob_by_sd

    # ---- Dirichlet inflow boundary source (rhs_dir/rhs_neu) -----------------------------------
    def _predictor_boundary_source(self, offset: dict, n_cells: int, mob_by_sd: dict,
                                   kind: str) -> np.ndarray:
        """Global per-cell source from the Dirichlet INFLOW boundary for the transported primary
        ``kind`` ("h" or "z"), mirroring the model's own upwind DIRICHLET rhs
        ``div @ rhs_dir @ diag(darcy) @ bc_entity``.

        The injected face entity is ``bc_entity[f] = mob_bc[f] · q_bc[f]`` with the boundary primary
        ``q_bc`` (enthalpy/overall-fraction BC values scattered to a face array) and the boundary
        mobility approximated by the adjacent cell's ``total_mass_mobility`` (``|cell_faces| @ mob``
        is exact on boundary faces).  Only the DIRICHLET inflow term is used: ``rhs_dir`` selects the
        few inflow faces (its other columns are zero), whereas the Neumann faces carry NO imposed
        advective flux (outflow is already handled by the interior ``transport`` upwind), so
        including ``rhs_neu @ bc_entity`` would inject a bogus flux on every temperature-BC face.
        Inlet cells -- DAG roots with no upstream cell -- get their sustaining inflow here; without
        it the sweep would drain them toward zero."""
        b = np.zeros(n_cells)
        comp = self.get_components()[1] if kind == "z" else None
        for sd in self.mdg.subdomains():
            if sd.num_cells == 0 or sd not in mob_by_sd:
                continue
            dm = self.mdg.subdomain_data(sd)[pp.DISCRETIZATION_MATRICES]
            par = self.mdg.subdomain_data(sd)[pp.PARAMETERS]
            if _MOBILITY_KW not in dm or "rhs_dir" not in dm[_MOBILITY_KW]:
                continue
            darcy = np.asarray(par[_MOBILITY_KW][_FLUX_KEY], float)
            rhs_dir = dm[_MOBILITY_KW]["rhs_dir"].tocsr()
            # boundary values of the transported primary, scattered onto a face array.
            q_bc = np.zeros(sd.num_faces)
            for bg in self.subdomains_to_boundary_grids([sd]):
                if bg.num_cells == 0:
                    continue
                vals = (np.asarray(self.bc_values_enthalpy(bg), float) if kind == "h"
                        else np.asarray(self.bc_values_overall_fraction(comp, bg), float))
                q_bc += bg.projection().transpose() @ vals
            if not np.any(q_bc):
                continue
            face_mob = np.abs(sd.cell_faces) @ mob_by_sd[sd]         # mob at boundary faces
            bc_entity = face_mob * q_bc
            div = sd.cell_faces.transpose().tocsr()
            src = div @ (rhs_dir @ sps.diags(darcy) @ bc_entity)
            b[offset[sd]:offset[sd] + sd.num_cells] = np.asarray(src, float).ravel()
        return b

    # ---- topological LEVELS (Kahn) ------------------------------------------------------------
    @staticmethod
    def _predictor_levels(A: sps.csr_matrix, n_cells: int) -> list[np.ndarray]:
        """Topological levels of the flux DAG (off-diagonal of ``A``): a cell is in level k once all
        its upstream cells (its off-diagonal columns) are in levels < k.  Returns a list of index
        arrays.  A residual cycle (numerical near-zero flux) is broken by releasing the lowest
        remaining in-degree, so the sweep always terminates."""
        off = A - sps.diags(A.diagonal())
        off.eliminate_zeros()
        # in-degree = number of upstream cells = nnz per ROW of off (i depends on its columns j).
        indeg = np.diff(off.tocsr().indptr).astype(int)
        # adjacency upstream j -> downstream i : the transpose (columns become rows).
        downstream = off.transpose().tocsr()
        indeg_w = indeg.copy()
        done = np.zeros(n_cells, bool)
        levels, frontier = [], np.where(indeg == 0)[0]
        seen = 0
        while seen < n_cells:
            if frontier.size == 0:                              # cycle: release min in-degree
                rem = np.where(~done)[0]
                frontier = np.array([rem[np.argmin(indeg_w[rem])]])
            levels.append(frontier)
            done[frontier] = True
            seen += frontier.size
            # decrement in-degree of everything downstream of the frontier.
            nxt = downstream[frontier]
            np.add.at(indeg_w, nxt.indices, -1)
            new = np.unique(nxt.indices)
            frontier = new[(indeg_w[new] <= 0) & (~done[new])]
        return levels

    # ---- the sweep --------------------------------------------------------------------------
    def run_transport_predictor(self) -> None:
        """Advance (h, z) one flow-order sweep and write the result (+ flashed secondaries) into the
        current iterate.  Cheap, standalone, monotone; a predictor for the FI Newton.

        Logs its own cost at INFO: the graph/assembly build (coupling + boundary + topological
        levels) vs. the forward-substitution sweep, plus a cumulative total over the run, so the
        predictor's overhead can be weighed against the Newton-iteration savings it buys."""
        offset, n_cells = self._predictor_cell_offsets()
        dt = float(self.time_manager.dt)

        t0 = time.perf_counter()
        # --- build: cell-to-cell coupling, topological levels, Dirichlet inflow sources ----------
        A, mob_by_sd = self._predictor_coupling(offset, n_cells)
        A_diag = np.clip(A.diagonal(), 0.0, None)                 # outflux coefficient per cell
        A_off = (A - sps.diags(A.diagonal())).tocsr()
        levels = self._predictor_levels(A, n_cells)
        b_h = self._predictor_boundary_source(offset, n_cells, mob_by_sd, "h")
        b_z = self._predictor_boundary_source(offset, n_cells, mob_by_sd, "z")
        t_build = time.perf_counter() - t0

        # --- sweep: gather state, level-scheduled forward substitution, clamp, scatter -----------
        t1 = time.perf_counter()
        p = self._predictor_gather("pressure", offset, n_cells)
        h = self._predictor_gather(self.enthalpy_variable, offset, n_cells)
        zname = self._predictor_overall_fraction_name()
        z = self._predictor_gather(zname, offset, n_cells)
        h0, z0 = h.copy(), z.copy()
        acc_z, acc_h = self._predictor_accumulation(p, h, z, offset, n_cells, dt)

        # Level-scheduled forward substitution (vectorised within each level).  Per cell i:
        #   q_i = (acc_i·q0_i - (A_off·q)_i - B_i) / (acc_i + A_diag_i),
        # with the boundary source B carrying the Dirichlet inflow (sign from div/rhs_dir).
        for lvl in levels:
            infl_z = np.asarray(A_off[lvl] @ z).ravel() + b_z[lvl]
            infl_h = np.asarray(A_off[lvl] @ h).ravel() + b_h[lvl]
            denom = acc_z[lvl] + A_diag[lvl]
            z[lvl] = np.where(denom > 0, (acc_z[lvl] * z0[lvl] - infl_z) / denom, z0[lvl])
            denom_h = acc_h[lvl] + A_diag[lvl]
            h[lvl] = np.where(denom_h > 0, (acc_h[lvl] * h0[lvl] - infl_h) / denom_h, h0[lvl])

        # Clamp to the OBL table bounds and write back the primaries; the FI Newton + the flash
        # recompute the eliminated secondaries at the first assembly.
        z = np.clip(z, *self._predictor_z_bounds())
        h = np.clip(h, *self._predictor_h_bounds())
        self._predictor_scatter(self.enthalpy_variable, h, offset)
        self._predictor_scatter(zname, z, offset)
        t_sweep = time.perf_counter() - t1

        total = t_build + t_sweep
        self._predictor_cum_time = getattr(self, "_predictor_cum_time", 0.0) + total
        self._predictor_n_calls = getattr(self, "_predictor_n_calls", 0) + 1
        _LOG.info(
            "transport predictor: %.1f ms (build %.1f + sweep %.1f)",
            total * 1e3, t_build * 1e3, t_sweep * 1e3)
        _LOG.info(
            "  cells=%d levels=%d | cumulative %.2f s over %d steps",
            n_cells, len(levels), self._predictor_cum_time, self._predictor_n_calls)

    # ---- residual gate: keep the sweep only where it is consistent with the FI solver ---------
    def _apply_gated_predictor(self) -> None:
        """Run the reordered sweep and keep it only where it is CONSISTENT with the fully-implicit
        solver: accept the largest damping along the predictor direction that REDUCES the FI residual
        (backtracking line search), otherwise fall back to the previous state.  This makes the warm
        start provably non-harmful -- a guess that would drive Newton to diverge (and force a
        time-step cut) raises the residual and is rejected or damped before Newton ever sees it.

        The residual is the model's OWN assembled residual with the eliminated secondaries flashed
        consistent, so acceptance is judged by exactly the quantity the FI Newton drives to zero."""
        es = self.equation_system
        if not self._predictor_gate_enabled():
            self.run_transport_predictor()
            return

        r_prev = self._predictor_residual_norm()               # flash-consistent baseline residual
        x0 = es.get_variable_values(iterate_index=0).copy()
        self.run_transport_predictor()                         # advective candidate (writes h, z)
        d = es.get_variable_values(iterate_index=0) - x0       # nonzero only in h, z
        if not np.any(d):
            es.set_variable_values(x0, iterate_index=0)
            self.update_derived_quantities()                   # keep derived <-> variables (see below)
            return

        accepted, trials = None, []
        for alpha in self._predictor_line_search_steps():
            es.set_variable_values(x0 + alpha * d, iterate_index=0)
            r = self._predictor_residual_norm()
            trials.append((alpha, r))
            if r < r_prev:                                     # first damping that helps -> accept
                accepted = (alpha, r)
                break
        if accepted is not None:
            _LOG.info("transport predictor: accepted alpha=%.3g (FI residual %.3e -> %.3e)",
                      accepted[0], r_prev, accepted[1])
        else:
            es.set_variable_values(x0, iterate_index=0)        # reject: keep the FI's own guess
            tried = ", ".join(f"a={a:g}:{rr:.2e}" for a, rr in trials)
            _LOG.info("transport predictor: rejected (r_prev=%.3e; tried %s)", r_prev, tried)
        # Re-flash so the eliminated secondaries + surrogate density match the FINAL primaries: the
        # base Newton loop flashes in initialize_nonlinear_solution (before this gate) and AFTER each
        # iteration, but NOT in before_nonlinear_iteration -- so iteration 0 would otherwise assemble
        # against the stale derived state left by the last line-search probe.
        self.update_derived_quantities()

    def _predictor_gate_enabled(self) -> bool:
        """Whether to residual-gate the sweep (default True).  Disable with
        ``params['transport_predictor_gate'] = False`` to apply the raw (ungated) sweep."""
        return bool(self.params.get("transport_predictor_gate", True))

    def _predictor_line_search_steps(self):
        """Backtracking dampings tried along the sweep direction, largest first: full sweep, then
        half, quarter, eighth (override with ``params['transport_predictor_backtrack']``)."""
        return self.params.get("transport_predictor_backtrack", (1.0, 0.5, 0.25, 0.125))

    def _predictor_residual_norm(self) -> float:
        """L2 norm of the model's residual at the current iterate, with the eliminated secondaries
        flashed consistent -- the exact quantity the FI Newton reduces, so the gate is consistent
        with the fully-implicit solver rather than with the predictor's own approximate balance."""
        self.update_derived_quantities()                       # flash T, s, x at the current (p,h,z)
        r = self.equation_system.assemble(evaluate_jacobian=False)
        return float(np.linalg.norm(np.asarray(r, float)))

    # ---- hook: run once per time step as a predictor -----------------------------------------
    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        if self.transport_predictor_enabled():
            try:
                self._apply_gated_predictor()
            except Exception as exc:  # a predictor must never break the FI solve
                _LOG.warning("transport predictor skipped; FI Newton proceeds unwarmed.")
                _LOG.warning("  reason: %s", exc)

    # ---- helpers ----------------------------------------------------------------------------
    def _predictor_overall_fraction_name(self) -> str:
        """Name of the overall-fraction transport variable (z_<comp>) -- the non-reference one."""
        comp = self._get_non_reference_component()
        return f"z_{comp}"

    def _predictor_gather(self, varname: str, offset: dict, n_cells: int) -> np.ndarray:
        out = np.zeros(n_cells)
        for var in self.equation_system.variables:
            if var.name == varname:
                sd = var.domain
                dofs = self.equation_system.dofs_of([var])
                vals = self.equation_system.get_variable_values(iterate_index=0)[dofs]
                out[offset[sd]:offset[sd] + sd.num_cells] = vals
        return out

    def _predictor_scatter(self, varname: str, values: np.ndarray, offset: dict) -> None:
        full = self.equation_system.get_variable_values(iterate_index=0)
        for var in self.equation_system.variables:
            if var.name == varname:
                sd = var.domain
                dofs = self.equation_system.dofs_of([var])
                full[dofs] = values[offset[sd]:offset[sd] + sd.num_cells]
        self.equation_system.set_variable_values(full, iterate_index=0)

    def _predictor_accumulation(self, p, h, z, offset, n_cells, dt):
        """Per-cell accumulation coefficients ``acc_z`` (fluid mass/dt) and ``acc_h`` (fluid mass +
        rock thermal inertia)/dt, from the OBL flash density and the solid heat capacity."""
        rho = self._predictor_sample_density(p, h, z)                 # mixture density per cell
        phi = float(self.solid.porosity)
        cp_fluid = 4.0e3 * 1.0e-6      # ~water specific heat, Mega-scaled like the model units
        c_rock = float(self.solid.specific_heat_capacity)
        rho_rock = float(self.solid.density)
        vol = np.zeros(n_cells)
        for sd in self.mdg.subdomains():
            vol[offset[sd]:offset[sd] + sd.num_cells] = sd.cell_volumes * self.specific_volume_values(sd)
        mass = phi * rho * vol
        rock_inertia = (1.0 - phi) * rho_rock * c_rock / max(cp_fluid, 1e-30) * vol
        return mass / dt, (mass + rock_inertia) / dt

    def specific_volume_values(self, sd) -> np.ndarray:
        """Aperture^(nd-dim) specific volume per cell (1 in the matrix)."""
        try:
            return np.asarray(self.equation_system.evaluate(self.specific_volume([sd])), float)
        except Exception:
            return np.ones(sd.num_cells)

    def _predictor_sample_density(self, p, h, z) -> np.ndarray:
        s = self.obl_sampler
        pts = np.vstack([z, h, p]).T
        s.sample_at(pts)
        return np.asarray(s.sampled_could.point_data["Rho"], float)

    def _predictor_z_bounds(self):
        zmin, zmax, *_ = self.obl_sampler.bounds
        sc = self.obl_sampler.conversion_factors[0]
        return (max(zmin / sc, 0.0), zmax / sc)

    def _predictor_h_bounds(self):
        _, _, hmin, hmax, *_ = self.obl_sampler.bounds
        sc = self.obl_sampler.conversion_factors[1]
        return (hmin / sc, hmax / sc)
