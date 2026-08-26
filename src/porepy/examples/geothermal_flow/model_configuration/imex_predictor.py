"""Fully-explicit IMEX predictor for the geothermal CF (Driesner H2O-NaCl) models.

A predictor run once per time step (``before_nonlinear_loop``) that advances the state with an
IMPES / sequential-implicit split so the fully-implicit (FI) Newton starts on the correct side of
the two-phase (boiling) kink -- the cure for the phase-transition stiffness that forces the FI
solver to cut the time step to ``dt_min``.

    IMPLICIT pressure  --  the mass-balance (pressure) equation is SPD-elliptic (its definiteness
        comes from the d(rho)/dp compressibility on the accumulation diagonal).  We assemble the
        mass residual and its Jacobian, drop the enthalpy/z columns (freeze transport), and -- on
        a mixed-dimensional grid -- Schur-eliminate the ``interface_darcy_flux`` mortar block (the
        naive pressure block is SINGULAR on any --md grid because the fracture<->matrix coupling
        rides that mortar variable).  Solve J_pp dp = -R with PETSc Krylov + hypre/BoomerAMG
        (CG for the symmetric TPFA block, GMRES for the mildly non-symmetric MPFA block).

    EXPLICIT transport + energy  --  with pressure frozen, forward-Euler the EXACT mixed-dimensional
        energy and component balances the FI solver assembles (mortar coupling, boundary/source
        terms and all), sub-cycled at the advective CFL.  The spatial term is read straight from the
        model's own equation residual: at any sub-step state ``S = R_eq - d_t(acc)`` is exactly
        ``div(flux) - source`` for that balance.  The primaries h and z are marched DIRECTLY -- the
        flash is only ever evaluated FORWARD (value + gradient) for the accumulation capacity, so
        there is NO conserved->primary inversion (and none of its two-phase non-uniqueness).

Everything reuses the flash the FI path uses, so the warm start is consistent, and the whole step is
residual-gated: it is kept only if it REDUCES the FI residual, else the FI's own guess is restored --
so the predictor can never itself cause a time-step cut.

Enable with ``params["imex_predictor"] = True`` (default OFF => FI is byte-identical).  Knobs:
``imex_cfl`` (0.25 -- conservative for gravity/buoyancy), ``imex_max_seconds`` (3.0 -- wall-time
safety cap; the explicit part covers as much of the macro dt as fits in this budget),
``imex_pressure_iters`` (1).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

import porepy as pp

_LOG = logging.getLogger(__name__)

_MASS_EQ = pp.fluid_mass_balance.FluidMassBalanceEquations.primary_equation_name()  # "mass_balance_equation"
_MORTAR_VAR = "interface_darcy_flux"
_MORTAR_EQ = "interface_darcy_flux_equation"
_MOBILITY_KW = "mobility"        # keyword under which the upwind matrix + darcy_flux are stored
_UPWIND_KEY = "transport"
_FLUX_KEY = "darcy_flux"


class IMEXTransportPredictor:
    """Mixin providing the implicit-pressure / explicit-transport IMEX predictor.  Mixed into
    :class:`_FlowModelBaseCore` alongside :class:`ReorderedTransportPredictor`, whose small helpers
    (cell offsets, gather/scatter, table-bound clamps, specific volume, the mass-flux coupling
    graph) it reuses.  Inert unless ``params["imex_predictor"]`` is truthy."""

    # ---- toggle -------------------------------------------------------------------------------
    def imex_predictor_enabled(self) -> bool:
        return bool(self.params.get("imex_predictor", False))

    # ---- hook: run once per time step --------------------------------------------------------
    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        if self.imex_predictor_enabled():
            try:
                self._run_imex_predictor()
            except Exception as exc:                    # a predictor must never break the FI solve
                _LOG.warning("IMEX predictor skipped; FI Newton proceeds unwarmed. reason: %s", exc)

    # ---- the gated IMEX step -----------------------------------------------------------------
    def _run_imex_predictor(self) -> None:
        es = self.equation_system
        t0 = time.perf_counter()
        x0 = es.get_variable_values(iterate_index=0).copy()
        self.update_all_boundary_conditions()
        self.update_derived_quantities()
        r_prev = self._imex_residual_norm()

        self._imex_step()

        self.update_derived_quantities()
        r_new = self._imex_residual_norm()
        if (not np.isfinite(r_new)) or r_new > r_prev:          # residual gate: accept only if it helps
            es.set_variable_values(x0, iterate_index=0)
            self.update_derived_quantities()
            _LOG.info("IMEX predictor rejected (FI residual %.3e -> %.3e), %.0f ms",
                      r_prev, r_new, (time.perf_counter() - t0) * 1e3)
        else:
            _LOG.info("IMEX predictor accepted (FI residual %.3e -> %.3e), %.0f ms",
                      r_prev, r_new, (time.perf_counter() - t0) * 1e3)

    def _imex_residual_norm(self) -> float:
        """Convergence measure the gate accepts/rejects on -- the SAME per-equation, storage-relative
        RelativeStorageLebesgueMetric the FI Newton uses (not a plain L2), so the gate is consistent
        with the quantity the implicit solver actually drives to zero.  Aggregated as the max over
        equations (convergence <=> that max < tol), i.e. the binding distance-to-convergence."""
        from .flow_model_base import RelativeStorageLebesgueMetric   # lazy: avoid the import cycle
        r = np.asarray(self.equation_system.assemble(evaluate_jacobian=False), dtype=float)
        try:
            per_eq = RelativeStorageLebesgueMetric(self)(r)          # {equation: storage-relative L2}
            return max(per_eq.values()) if per_eq else float(np.linalg.norm(r))
        except Exception:
            return float(np.linalg.norm(r))

    def _imex_step(self) -> None:
        """One IMPES step over the full dt: implicit SPD pressure, then a FULLY EXPLICIT, PURE-NUMPY,
        FULLY-UPDATING transport+energy sub-loop.  Each sub-step re-flashes and rebuilds the ENTIRE
        spatial residual S_e, S_z via :meth:`_imex_spatial_numpy` -- F_total and every buoyancy
        direction recomputed from the static Xi_p/G + the flash, the upstream selection rebuilt from
        their sign, advective + buoyancy + Fourier + boundary + interface + jump assembled and
        divergenced -- with NO AD and NO frozen ``rest``: only the TPFA/MPFA sparse matrices are
        static, everything the fluxes carry is recomputed.  h, z are marched forward-Euler,
        ``x -= dt_sub * S / cap`` with the lumped accumulation capacity, and written back once."""
        offset, n = self._predictor_cell_offsets()
        dt = float(self.time_manager.dt)
        zname = self._predictor_overall_fraction_name()
        vol = self._imex_cell_volumes(offset, n)
        self.update_all_boundary_conditions()                              # time fixed within the step

        p = self._predictor_gather("pressure", offset, n)
        h = self._predictor_gather(self.enthalpy_variable, offset, n)
        z = self._predictor_gather(zname, offset, n)

        # --- 1. IMPLICIT PRESSURE (SPD block + AMG) -------------------------------------------
        self._imex_pressure_solve()
        p = self._predictor_gather("pressure", offset, n)                   # refresh after the solve
        self.update_derived_quantities()                                   # sync model state @ (p, h, z)

        # Static operators (parsed once, whole simulation) + per-step frozen data (pressure, boundary
        # values, interface darcy/fourier fluxes and selectors -- all fixed by the implicit solve).
        C = self._imex_static()
        F = self._imex_frozen()
        frozen = self._imex_freeze_operators(offset, n)                    # advective CFL heuristic only

        # --- 2. EXPLICIT TRANSPORT + ENERGY (pure numpy; full dt, wall-time safety cap) -------
        nmax = int(self.params.get("imex_substeps_max", 100000))           # runaway backstop only
        budget = float(self.params.get(                                    # wall-time safety cap [s];
            "imex_max_seconds", 10.0 if self.mdg.interfaces() else 3.0))    # MD sub-loop is heavier -> 10s
        zlo, zhi = self._predictor_z_bounds()
        hlo, hhi = self._predictor_h_bounds()
        t_done, k, t0 = 0.0, 0, time.perf_counter()
        while t_done < dt * (1.0 - 1e-9) and k < nmax:
            pd = self._imex_sample(p, h, z)                                # flash ONCE per sub-step
            w_m, _, _ = self._imex_weights(pd)
            dt_sub = min(dt - t_done, self._imex_cfl_dt(frozen, w_m, pd["rho"], vol, offset))
            if not (dt_sub > 0.0):
                break
            S_e, S_z = self._imex_spatial_numpy(p, h, z, C, F, pd=pd)      # updating spatial residual
            _, _, capE_h, capZ_z = self._imex_terms(pd, p, h, z, vol)      # lumped accumulation capacity
            h = np.clip(h - dt_sub * S_e / capE_h, hlo, hhi)              # forward-Euler march
            z = np.clip(z - dt_sub * S_z / capZ_z, zlo, zhi)
            t_done += dt_sub
            k += 1
            if time.perf_counter() - t0 > budget:                         # safety cap: wall-time budget hit
                break
        self._predictor_scatter(self.enthalpy_variable, h, offset)        # write the advanced state ONCE
        self._predictor_scatter(zname, z, offset)
        cover = (t_done / dt) if dt > 0.0 else 1.0
        _LOG.info("IMEX explicit: %d sub-steps, %.0f%% of dt in %.2fs (updating flux, pure numpy)",
                  k, 100.0 * cover, time.perf_counter() - t0)

    # ---- implicit pressure -------------------------------------------------------------------
    def _imex_pressure_solve(self) -> None:
        """IMPES pressure: 1-3 Newton iterations of the mass-balance block only.  Each iteration
        re-flashes rho(p,h,z), extracts the SPD (Schur-mortar-reduced) pressure block, AMG-solves
        for dp, adds it to the pressure iterate.  h, z frozen."""
        es = self.equation_system
        for _ in range(int(self.params.get("imex_pressure_iters", 1))):
            self.update_derived_quantities()
            J_pp, R = self._imex_extract_Jpp()
            if np.linalg.norm(R) < 1e-10:
                break
            dp = self._imex_amg_solve(J_pp, -R)
            es.set_variable_values(dp, [self.pressure_variable], additive=True, iterate_index=0)

    def _imex_extract_Jpp(self) -> tuple[sps.csr_matrix, np.ndarray]:
        """Square SPD pressure block J_pp = d(R_mass)/d(pressure) and residual R_mass over the full
        mdg.  On an MD grid the mortar ``interface_darcy_flux`` block is Schur-eliminated (the naive
        pressure-only block is singular there)."""
        es = self.equation_system
        pvar = self.pressure_variable
        has_mortar = (len(self.mdg.interfaces()) > 0
                      and _MORTAR_VAR in {v.name for v in es.variables})
        if not has_mortar:
            A, minus_r = es.assemble(equations=[_MASS_EQ], variables=[pvar])
            return A.tocsr(), -np.asarray(minus_r, float).ravel()
        App, r_p = es.assemble(equations=[_MASS_EQ], variables=[pvar])
        Apm, _ = es.assemble(equations=[_MASS_EQ], variables=[_MORTAR_VAR])
        Amp, r_m = es.assemble(equations=[_MORTAR_EQ], variables=[pvar])
        Amm, _ = es.assemble(equations=[_MORTAR_EQ], variables=[_MORTAR_VAR])
        lu = splu(Amm.tocsc())
        J_pp = (App.tocsc() - Apm @ sps.csc_matrix(lu.solve(Amp.toarray()))).tocsr()
        r_p = np.asarray(r_p, float).ravel()
        r_m = np.asarray(r_m, float).ravel()
        R = -(-r_p - Apm @ lu.solve(-r_m))
        return J_pp, np.asarray(R, float).ravel()

    def _imex_amg_solve(self, J: sps.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve J dp = rhs with PETSc GMRES + hypre/BoomerAMG.  The pressure block is treated as
        (mildly) NON-symmetric unconditionally -- MPFA makes it so, and even the TPFA block is not
        exactly symmetric once the mortar is Schur-folded -- so GMRES + BoomerAMG (classical AMG,
        robust for non-symmetric operators where GAMG stalls) is used in every case."""
        from petsc4py import PETSc

        J = J.tocsr()
        J.sort_indices()
        mat = PETSc.Mat().createAIJ(
            size=J.shape,
            csr=(J.indptr.astype(PETSc.IntType), J.indices.astype(PETSc.IntType),
                 J.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_SELF)
        mat.assemble()
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-8, atol=1e-50, max_it=500)
        pc = ksp.getPC()
        try:
            pc.setType("hypre")
            pc.setHYPREType("boomeramg")
        except Exception:                                          # hypre not built -> smoothed aggregation
            pc.setType("gamg")
        x = mat.createVecRight()
        b = mat.createVecRight()
        b.setArray(np.ascontiguousarray(rhs, dtype=PETSc.ScalarType))
        ksp.solve(b, x)
        return np.array(x.getArray(), dtype=float)

    # ---- flash + accumulation (ONE forward flash, reused across CFL/capacity/accumulation) ---
    def _imex_cell_volumes(self, offset, n) -> np.ndarray:
        """Per-cell volume * specific-volume over the mdg -- geometry, so computed once and cached."""
        vol = self.__dict__.get("_imex_vol")
        if vol is not None and vol.shape[0] == n:
            return vol
        vol = np.zeros(n)
        for sd in self.mdg.subdomains():
            vol[offset[sd]:offset[sd] + sd.num_cells] = (
                sd.cell_volumes * self.specific_volume_values(sd))
        self._imex_vol = vol
        return vol

    def _imex_sample(self, p, h, z) -> dict:
        """One forward flash at (p, h, z) -- all per-phase fields needed for the mobility weights,
        plus mixture density/temperature and their h,z gradients.  Densities kg/m^3, enthalpies ->
        MJ/kg (table kJ/kg x 1e-3), viscosities -> MPa*s (table Pa*s x 1e-6): the model's unit system,
        so the frozen-graph fluxes stay consistent with the stored darcy flux."""
        s = self.obl_sampler
        s.sample_at(np.column_stack([z, h, p]))
        pd = s.sampled_could.point_data
        g = lambda k: np.asarray(pd[k], float)                             # noqa: E731
        gR, gT = g("grad_Rho"), g("grad_Temperature")
        return {
            "rho": g("Rho"), "T": g("Temperature"),
            "dR_dz": gR[:, 0], "dR_dh": gR[:, 1], "dT_dh": gT[:, 1],
            "rho_l": g("Rho_l"), "rho_v": g("Rho_v"),
            "h_l": g("H_l") * 1e-3, "h_v": g("H_v") * 1e-3,
            "s_v": g("S_v"), "s_h": g("S_h"), "s_l": g("S_l"),
            "x_l": g("Xl"), "x_v": g("Xv"),
            "mu_l": g("mu_l") * 1e-6, "mu_v": g("mu_v") * 1e-6,
        }

    def _imex_phase_mobilities(self, pd):
        """Per-cell liquid/vapor MASS mobilities m_j = rho_j*kr_j/mu_j with the Weis (2014) halite-aware
        relative permeability (option A/B, matching the model).  Halite is immobile (m=0)."""
        s_v = np.clip(pd["s_v"], 0.0, 1.0)
        s_h = np.clip(pd["s_h"], 0.0, 1.0 - s_v)
        s_l = np.clip(1.0 - s_v - s_h, 0.0, 1.0)
        one_sh = np.maximum(1.0 - s_h, 1.0e-12)
        try:
            opt = self._halite_perm_option()
        except Exception:
            opt = "B"
        sr = 0.3
        if opt == "A":                                                    # halite in rel-perm
            s_red = (s_l - sr * (1.0 - s_h)) / (1.0 - sr)
            perm, total = np.ones_like(s_l), (1.0 - s_h)
        else:                                                             # B: halite in abs-perm (Eq. 28)
            s_red = (s_l / one_sh - sr) / (1.0 - sr)
            perm, total = one_sh ** 2, np.ones_like(s_l)
        kr_l = np.maximum(s_red, 0.0)
        kr_v = np.maximum(total - kr_l, 0.0)
        m_l = pd["rho_l"] * (perm * kr_l) / np.maximum(pd["mu_l"], 1.0e-30)
        m_v = pd["rho_v"] * (perm * kr_v) / np.maximum(pd["mu_v"], 1.0e-30)
        return m_l, m_v

    def _imex_weights(self, pd):
        """The three advection weights the balances carry: total mass (mass/CFL), enthalpy (energy),
        NaCl (component), each summed from the per-phase mass mobilities."""
        m_l, m_v = self._imex_phase_mobilities(pd)
        w_m = m_l + m_v
        w_e = pd["h_l"] * m_l + pd["h_v"] * m_v
        w_z = pd["x_l"] * m_l + pd["x_v"] * m_v
        return w_m, w_e, w_z

    # ---- hand-coded upwind (directions recomputed every sub-step; only the topology is static) --
    def _imex_upwind_topology(self, sds):
        """Static per-face upstream candidates for the single-point upwind (Upwind._single_point_
        upwind_matrices convention): global ``cf0``/``cf1`` are the positive-/negative-normal-side
        cells of each face (-1 exterior), so the upstream cell is ``cf0 where dir>=0 else cf1``.
        Geometry only -- built once and cached."""
        cache = self.__dict__.get("_imex_up_topo")
        if cache is not None:
            return cache
        cf0, cf1, off = [], [], 0
        for sd in sds:
            if sd.num_faces == 0:
                continue
            cfd = sd.cell_faces_as_dense()                              # (2, num_faces): +/- side cells
            c0, c1 = cfd[0].astype(np.intp).copy(), cfd[1].astype(np.intp).copy()
            c0[c0 >= 0] += off
            c1[c1 >= 0] += off                                          # globalize interior cells
            cf0.append(c0)
            cf1.append(c1)
            off += sd.num_cells
        cache = (np.concatenate(cf0), np.concatenate(cf1))
        self._imex_up_topo = cache
        return cache

    def _imex_upwind(self, direction, w, cf0, cf1, neu_mask):
        """Single-point upstream weighting of the cell quantity ``w`` along a signed face
        ``direction`` (>=0 -> positive-side cell, else negative-side), matching PorePy's Upwind: drop
        the exterior side (upstream cell -1) and the Neumann faces (handled by bound_transport_neu).
        Pure numpy gather -- the direction (hence the upstream choice) is recomputed every sub-step."""
        up = np.where(direction >= 0.0, cf0, cf1)
        keep = (up >= 0) & (~neu_mask)
        out = np.zeros_like(direction)
        out[keep] = w[up[keep]]
        return out

    def _imex_neu_mask(self, sds, keyword):
        """Static per-face Neumann flag for one advective keyword (the bc classification is fixed)."""
        masks = []
        for sd in sds:
            if sd.num_faces == 0:
                continue
            bc = self.mdg.subdomain_data(sd)[pp.PARAMETERS][keyword].get("bc")
            masks.append(np.asarray(bc.is_neu, bool) if bc is not None
                         else np.zeros(sd.num_faces, bool))
        return np.concatenate(masks)

    def _imex_verify_upwind(self, sds, tol) -> bool:
        """Confirm the hand-coded upwind reproduces PorePy's ``discr.upwind() @ w`` bit-for-bit at the
        current (rediscretized) state, for the mobility and enthalpy keywords -- so the runtime may
        rebuild the upstream choice from sign(direction) instead of the frozen matrix."""
        es = self.equation_system
        cf0, cf1 = self._imex_upwind_topology(sds)
        w = np.asarray(self.advection_weight_mass_balance(sds).value(es), float)  # a real cell field
        ok = True
        for kw, discr in (("mobility", self.mobility_discretization(sds)),
                          (self.enthalpy_keyword, self.enthalpy_discretization(sds))):
            direction = np.concatenate([                                # the flux PorePy upwinded along
                np.asarray(self.mdg.subdomain_data(sd)[pp.PARAMETERS][kw]["darcy_flux"], float)
                for sd in sds if sd.num_faces > 0])
            ad = discr.upwind().parse(self.mdg).tocsr() @ w
            my = self._imex_upwind(direction, w, cf0, cf1, self._imex_neu_mask(sds, kw))
            d = float(np.max(np.abs(ad - my))); s = float(np.max(np.abs(ad))) or 1.0
            ok &= d <= tol * s
            _LOG.info("  upwind[%s]: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      kw, float(np.max(np.abs(ad))), d, d / s, "OK" if d <= tol * s else "MISMATCH")
        return ok

    def _imex_verify_directions(self, sds, tol) -> bool:
        """Confirm the flux DIRECTIONS are reconstructible from the static matrices + the flash, so the
        sub-loop can recompute them (not freeze them):
          F_total  = flux()@p + bound_flux()@(bc + M2p@interface_darcy_flux) + vector_source()@(g*rho)
          nu_pair  = vector_source()@(g*(rho_delta - rho_gamma))
        Only the gravity source (g*rho) moves with the flash; everything else (p, bc, interface darcy)
        is frozen by the implicit pressure solve.  The gravity coefficient g is calibrated once from
        vector_source_darcy_flux / rho_flow (a per-dim constant)."""
        es = self.equation_system
        intf = self.subdomains_to_interfaces(sds, [1])
        P = lambda op: op.parse(self.mdg).tocsr()                        # noqa: E731
        base = self.darcy_flux_discretization(sds)
        Xi, Bf, G = P(base.flux()), P(base.bound_flux()), P(base.vector_source())
        p = np.asarray(self.pressure(sds).value(es), float)
        bc_d = np.asarray(self.combine_boundary_operators_darcy_flux(sds).value(es), float)
        vs = np.asarray(self.vector_source_darcy_flux(sds).value(es), float)          # g*rho_flow
        bnd = bc_d
        if len(intf) != 0:
            M2p = P(pp.ad.MortarProjections(self.mdg, sds, intf, dim=1).mortar_to_primary_int())
            bnd = bnd + M2p @ np.asarray(self.interface_darcy_flux(intf).value(es), float)
        F_recon = Xi @ p + Bf @ bnd + G @ vs
        F_ad = np.asarray(self.darcy_flux(sds).value(es), float)
        d = float(np.max(np.abs(F_ad - F_recon))); s = float(np.max(np.abs(F_ad))) or 1.0
        ok = d <= tol * s
        _LOG.info("  darcy_flux decomposition: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                  float(np.max(np.abs(F_ad))), d, d / s, "OK" if d <= tol * s else "MISMATCH")

        # nu reconstruction: density_driven_flux builds gravity_flux = -e_n . (drho * g_field), i.e. the
        # gravity source is -drho*g in the VERTICAL component (dim nd-1) only, zero elsewhere.
        n = int(sum(sd.num_cells for sd in sds))
        nd = self.nd
        g_field = np.asarray(self.gravity_field(sds).value(es), float)                # per-cell g (const)
        pd = self._imex_sample(self._predictor_gather("pressure", *self._predictor_cell_offsets()),
                               self._predictor_gather(self.enthalpy_variable, *self._predictor_cell_offsets()),
                               self._predictor_gather(self._predictor_overall_fraction_name(),
                                                      *self._predictor_cell_offsets()))
        for gamma, delta in self._imex_ordered_pairs():
            if not (self._imex_is_mobile_phase(gamma) and self._imex_is_mobile_phase(delta)):
                continue                                                 # solids never drive nu (immobile)
            nu_ad = np.asarray(self.pair_density_driven_flux(gamma, delta, sds).value(es), float)
            drho = self._imex_phase_density(pd, gamma) - self._imex_phase_density(pd, delta)  # rho_g - rho_d
            gflux = np.zeros((n, nd)); gflux[:, nd - 1] = -drho * g_field
            nu_np = G @ gflux.ravel()
            d = float(np.max(np.abs(nu_ad - nu_np))); s = float(np.max(np.abs(nu_ad))) or 1.0
            ok &= d <= tol * s
            _LOG.info("  nu[%s->%s] reconstruction: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      gamma.name, delta.name, float(np.max(np.abs(nu_ad))), d, d / s,
                      "OK" if d <= tol * s else "MISMATCH")
            break                                                        # one mobile pair suffices
        return ok

    @staticmethod
    def _imex_is_mobile_phase(phase) -> bool:
        name = phase.name.lower()
        return not ("hal" in name or "sol" in name)                     # halite/solid is immobile

    def _imex_phase_density(self, pd, phase) -> np.ndarray:
        """Map a model phase to its flash density (liquid/vapor; immobile solids never drive nu)."""
        name = phase.name.lower()
        if "vap" in name or "gas" in name:
            return np.asarray(pd["rho_v"], float)
        return np.asarray(pd["rho_l"], float)

    # =====================================================================================
    #  Fully-updating pure-numpy spatial residual S_e, S_z (no AD in the hot path).
    #  Static (parsed once): the TPFA/MPFA matrices, div, mortar projections, trace, the upwind
    #  topology and bc classification.  Recomputed every call from the flash: F_total and each
    #  buoyancy direction nu (via static Xi_p/G), the upstream selection (sign of those), and all
    #  weights.  Frozen per MACRO step by the implicit solve: p, the boundary data, and the interface
    #  darcy/fourier mortar variables.
    # =====================================================================================
    def _imex_grav_src(self, d, g_field, n, nd):
        """Cell gravity source for a scalar density field d, matching gravity_force / density_driven_
        flux: ``-e_n . (d * g)`` -- i.e. ``-d*g`` in the vertical component (dim nd-1), zero else."""
        gs = np.zeros((n, nd))
        gs[:, nd - 1] = -d * g_field
        return gs.ravel()

    def _imex_static(self):
        """Parse & cache every STATIC operator once (geometry + permeability only)."""
        c = self.__dict__.get("_imex_static_cache")
        if c is not None:
            return c
        es = self.equation_system
        mdg = self.mdg
        sds = mdg.subdomains()
        intf = self.subdomains_to_interfaces(sds, [1])
        P = lambda op: op.parse(mdg).tocsr()                             # noqa: E731
        base = self.darcy_flux_discretization(sds)
        mob = self.mobility_discretization(sds)
        enth = self.enthalpy_discretization(sds)
        cf0, cf1 = self._imex_upwind_topology(sds)
        off, n = self._predictor_cell_offsets()
        c = dict(
            sds=sds, intf=intf, n=n, nd=self.nd, offset=off,
            nfaces=int(sum(sd.num_faces for sd in sds)),
            Xi=P(base.flux()), Bf=P(base.bound_flux()), G=P(base.vector_source()),
            div=P(pp.ad.Divergence(sds, dim=1)),
            Bdir_m=P(mob.bound_transport_dir()), Bneu_m=P(mob.bound_transport_neu()),
            Bdir_e=P(enth.bound_transport_dir()), Bneu_e=P(enth.bound_transport_neu()),
            neu_m=self._imex_neu_mask(sds, "mobility"),
            neu_e=self._imex_neu_mask(sds, self.enthalpy_keyword),
            cf0=cf0, cf1=cf1,
            g_field=np.broadcast_to(np.asarray(self.gravity_field(sds).value(es), float),
                                    (n,)).copy(),
            vol=self._imex_cell_volumes(off, n),
        )
        if len(intf) != 0:
            mp = pp.ad.MortarProjections(mdg, sds, intf, dim=1)
            c.update(Mavg=P(mp.primary_to_mortar_avg()), S2m=P(mp.secondary_to_mortar_avg()),
                     Tr=P(pp.ad.Trace(sds).trace), M2p=P(mp.mortar_to_primary_int()),
                     M2s=P(mp.mortar_to_secondary_int()))
        self._imex_static_cache = c
        return c

    def _imex_frozen(self):
        """Read the per-MACRO-step frozen data (pressure + boundary data + interface mortar
        variables) ONCE at the step start (a consistent, converged state), so the sub-loop never
        touches AD.  Boundary values are frozen within a step; the interface darcy/fourier fluxes come
        from the implicit pressure/elliptic solve."""
        es = self.equation_system
        C = self._imex_static()
        sds, intf = C["sds"], C["intf"]
        comp = list(self.fluid.components)[1:][0]
        en_bc = self._combine_boundary_operators(
            subdomains=sds, dirichlet_operator=self.advection_weight_energy_balance,
            neumann_operator=self.enthalpy_flux, robin_operator=None,
            bc_type=self.bc_type_enthalpy_flux, name="bc_values_enthalpy")
        fd = self.fourier_flux_discretization(sds)
        P0 = lambda op: op.parse(self.mdg).tocsr()                       # noqa: E731
        F = dict(
            p=self._predictor_gather("pressure", *self._predictor_cell_offsets()),
            bc_darcy=np.asarray(self.combine_boundary_operators_darcy_flux(sds).value(es), float),
            bc_fourier=np.asarray(self.combine_boundary_operators_fourier_flux(sds).value(es), float),
            bc_e=np.asarray(en_bc.value(es), float),
            bc_z=np.asarray(self.boundary_component_flux(comp, sds).value(es), float),
            # Fourier flux matrices (kept per-step: conductivity may depend on the state) and the
            # interface buoyancy operators (their selectors follow the density-driven direction).
            Kflux=P0(fd.flux()), Bflux_f=P0(fd.bound_flux()),
            buoy=self._imex_parse_buoyancy_ops(sds),
        )
        if len(intf) != 0:
            P = lambda op: op.parse(self.mdg).tocsr()                   # noqa: E731
            F["idf"] = np.asarray(self.interface_darcy_flux(intf).value(es), float)
            F["iff"] = np.asarray(self.interface_fourier_flux(intf).value(es), float)
            F["bnd_darcy"] = F["bc_darcy"] + C["M2p"] @ F["idf"]
            # the interface advective UPWIND selectors (frozen with interface_darcy_flux); the
            # enthalpy and mobility keywords can upwind along different fluxes, so keep both pairs.
            cpl_m = self.interface_mobility_discretization(intf)
            cpl_e = self.interface_enthalpy_discretization(intf)
            F["Vp_m"], F["Vs_m"] = P(cpl_m.upwind_primary()), P(cpl_m.upwind_secondary())
            F["Vp_e"], F["Vs_e"] = P(cpl_e.upwind_primary()), P(cpl_e.upwind_secondary())
        else:
            F["bnd_darcy"] = F["bc_darcy"]
        return F

    def _imex_phase_maps(self, pd, m_l, m_v, n):
        """Map each model phase to its flash (mobility, specific enthalpy, NaCl fraction); halite/solid
        is immobile with zero advected quantities (they vanish from every pair flux)."""
        z = np.zeros(n)
        mob, enth, frac = {}, {}, {}
        for ph in self.fluid.phases:
            nm = ph.name.lower()
            if "vap" in nm or "gas" in nm:
                mob[ph], enth[ph], frac[ph] = m_v, np.asarray(pd["h_v"], float), np.asarray(pd["x_v"], float)
            elif "hal" in nm or "sol" in nm:
                mob[ph], enth[ph], frac[ph] = z, z, z
            else:
                mob[ph], enth[ph], frac[ph] = m_l, np.asarray(pd["h_l"], float), np.asarray(pd["x_l"], float)
        return mob, enth, frac

    def _imex_spatial_numpy(self, p, h, z, C=None, F=None, pd=None):
        """Fully-updating spatial residual S_e, S_z = div@flux - source for the energy and component
        balances, PURE numpy/scipy: recompute the flash, F_total and every buoyancy nu from the static
        Xi_p/G, rebuild the upstream selection with :meth:`_imex_upwind`, assemble advective + buoyancy
        + Fourier + boundary + interface, and take the mixed-dimensional divergence.  No AD."""
        C = C if C is not None else self._imex_static()
        F = F if F is not None else self._imex_frozen()
        n, nd, g = C["n"], C["nd"], C["g_field"]
        cf0, cf1 = C["cf0"], C["cf1"]
        pd = pd if pd is not None else self._imex_sample(p, h, z)
        m_l, m_v = self._imex_phase_mobilities(pd)
        lam = np.maximum(m_l + m_v, 1e-30)
        w_e = np.asarray(pd["h_l"], float) * m_l + np.asarray(pd["h_v"], float) * m_v
        w_z = np.asarray(pd["x_l"], float) * m_l + np.asarray(pd["x_v"], float) * m_v
        rho_l, rho_v = np.asarray(pd["rho_l"], float), np.asarray(pd["rho_v"], float)

        # --- recomputed total-flux direction ---
        rho_flow = (m_l * rho_l + m_v * rho_v) / lam
        F_total = C["Xi"] @ p + C["Bf"] @ F["bnd_darcy"] + C["G"] @ self._imex_grav_src(rho_flow, g, n, nd)

        mob, enth, frac = self._imex_phase_maps(pd, m_l, m_v, n)

        # --- viscous advective flux (interior + boundary) per balance ---
        def visc(w, neu, Bdir, Bneu, bc):
            return (F_total * self._imex_upwind(F_total, w, cf0, cf1, neu)
                    + Bdir @ (F_total * bc) + Bneu @ bc)
        flux_e = visc(w_e, C["neu_e"], C["Bdir_e"], C["Bneu_e"], F["bc_e"])
        flux_z = visc(w_z, C["neu_m"], C["Bdir_m"], C["Bneu_m"], F["bc_z"])

        # --- subdomain buoyancy (directions recomputed every call) ---
        for ga, de in self._imex_ordered_pairs():
            if not (self._imex_is_mobile_phase(ga) and self._imex_is_mobile_phase(de)):
                continue                                                 # immobile pair -> zero flux
            nu = C["G"] @ self._imex_grav_src(
                self._imex_phase_density(pd, ga) - self._imex_phase_density(pd, de), g, n, nd)
            l_g, l_d = mob[ga], mob[de]
            Ug = self._imex_upwind(nu, l_g, cf0, cf1, C["neu_m"])
            Ud = self._imex_upwind(-nu, l_d, cf0, cf1, C["neu_m"])
            den = Ug + Ud + 1.0e-15
            flux_e += self._imex_upwind(nu, enth[ga] * l_g, cf0, cf1, C["neu_m"]) * Ud / den * nu
            flux_z += self._imex_upwind(nu, frac[ga] * l_g, cf0, cf1, C["neu_m"]) * Ud / den * nu

        # --- Fourier conduction (interior + Neumann heat flux) ---
        flux_e += F["Kflux"] @ np.asarray(pd["T"], float) + F["Bflux_f"] @ F["bc_fourier"]

        src_e = np.zeros(n)
        src_z = np.zeros(n)
        if len(C["intf"]) != 0:
            # interface advective flux (interface_darcy_flux frozen -> its upwind selectors frozen);
            # enthalpy and mobility keywords may upwind differently, so use each one's own selectors.
            def iadv(w, Vp, Vs):
                return F["idf"] * (Vp @ (C["Mavg"] @ (C["Tr"] @ w)) + Vs @ (C["S2m"] @ w))
            ie = iadv(w_e, F["Vp_e"], F["Vs_e"])
            iz = iadv(w_z, F["Vp_m"], F["Vs_m"])
            flux_e += C["Bneu_e"] @ (C["M2p"] @ ie); src_e += C["M2s"] @ ie
            flux_z += C["Bneu_m"] @ (C["M2p"] @ iz); src_z += C["M2s"] @ iz
            # interface Fourier flux (frozen mortar variable)
            flux_e += F["Bflux_f"] @ (C["M2p"] @ F["iff"]); src_e += C["M2s"] @ F["iff"]
            # interface buoyancy (frozen selectors + interface nu, recomputed flash weights)
            for pr in F["buoy"]:
                if pr["intf"] is None or not (self._imex_is_mobile_phase(pr["gamma"])
                                              and self._imex_is_mobile_phase(pr["delta"])):
                    continue
                l_g, l_d = mob[pr["gamma"]], mob[pr["delta"]]
                ce = self._imex_interface_coupling(pr, l_g, l_d, enth[pr["gamma"]])
                cz = self._imex_interface_coupling(pr, l_g, l_d, frac[pr["gamma"]])
                flux_e += pr["intf"]["flux_proj"] @ ce; src_e += pr["intf"]["jump_proj"] @ ce
                flux_z += pr["intf"]["flux_proj"] @ cz; src_z += pr["intf"]["jump_proj"] @ cz

        return C["div"] @ flux_e - src_e, C["div"] @ flux_z - src_z

    def _imex_freeze_operators(self, offset, n) -> dict:
        """Freeze the transport operators for the sub-loop (pressure fixed): per subdomain the stored
        darcy flux, the upwind matrix and the divergence.  Touched zero times by PorePy inside the
        loop -- reused for every sub-step's advective divergence and the CFL."""
        frozen = {}
        for sd in self.mdg.subdomains():
            if sd.num_cells == 0:
                continue
            data = self.mdg.subdomain_data(sd)
            dm = data.get(pp.DISCRETIZATION_MATRICES, {})
            par = data.get(pp.PARAMETERS, {})
            if _MOBILITY_KW not in dm or _UPWIND_KEY not in dm[_MOBILITY_KW]:
                continue
            frozen[sd] = (np.asarray(par[_MOBILITY_KW][_FLUX_KEY], float),
                          dm[_MOBILITY_KW][_UPWIND_KEY].tocsr(),
                          sd.cell_faces.transpose().tocsr())
        return frozen

    # =====================================================================================
    #  Static-operator pure-numpy MD explicit integrator + built-in residual verification.
    #
    #  The mixed-dimensional TPFA/MPFA discretization is FIXED for the whole simulation
    #  (rock permeability, geometry).  So every operator the component/energy balances need is
    #  a static sparse matrix, parsed ONCE from the model's own AD operators (the "slow stupid
    #  AD", used offline), and the sub-loop is then pure numpy: frozen matrices @ flash weights.
    #  The one nonlinearity in a gravity term -- the cell source g*rho -- is LINEAR in rho
    #  (residual/value mode), so the consistent-gravity operator G freezes as a plain matrix too.
    #
    #  Buoyancy (the hybrid-upwind pairwise term) is reproduced EXACTLY from the model's kernel
    #  (fluid_property_library.__entity_buoyancy_flux, mobility-product / HU branch, N=2 -> empty
    #  background): per ordered phase pair (gamma, delta),
    #       b = Ug @ (a_gamma * l_gamma) * (Ud @ l_delta) / (Ug @ l_gamma + Ud @ l_delta + eps) * nu
    #  with the HU upwind matrices Ug, Ud and the pair density-driven flux nu = G @ grav(rho_d-rho_g)
    #  FROZEN over the macro step (the model's own lag_buoyancy policy), and the flash weights
    #  l_gamma, a_gamma recomputed each sub-step.  `imex_verify_residuals` checks the numpy
    #  assembly against the AD operator value, per pair and per equation.
    # =====================================================================================
    @staticmethod
    def _imex_is_fractional_flow(model) -> bool:
        from porepy.models.compositional_flow import is_fractional_flow
        return bool(is_fractional_flow(model))

    def _imex_ordered_pairs(self) -> list:
        """Every ordered buoyancy pair (gamma, delta) the model sums over (N=2 -> two)."""
        pairs = []
        for phase in self.fluid.phases:
            for gd in self.phase_pairs_for(phase):
                pairs.append(gd)
        return pairs

    def _imex_parse_buoyancy_ops(self, sds) -> list:
        """Parse, ONCE, the frozen per-pair buoyancy operators from the model's AD kernel: the two HU
        upwind matrices (Ug, Ud) and the pair density-driven face flux nu on subdomains, plus -- when
        interfaces are present (--md) -- the mortar bundle (the four HUpwindCoupling selectors, the
        primary trace + primary/secondary mortar averages, the interface density-driven flux, and the
        flux/jump projections).  These are exactly the matrices the FI path re-discretizes each
        iteration; here they are frozen at the reference (lag-buoyancy)."""
        es = self.equation_system
        interfaces = self.subdomains_to_interfaces(sds, [1])
        out = []
        for gamma, delta in self._imex_ordered_pairs():
            discr = self.hybrid_upwind_discretization(gamma, delta, sds)
            Ug = discr.upwind_gamma().parse(self.mdg).tocsr()
            Ud = discr.upwind_delta().parse(self.mdg).tocsr()
            nu = np.asarray(self.pair_density_driven_flux(gamma, delta, sds).value(es), float)
            intf = None
            if len(interfaces) != 0:
                idc = self.hybrid_interface_upwind_discretization(gamma, delta, interfaces)
                mp = pp.ad.MortarProjections(self.mdg, sds, interfaces, dim=1)
                P = lambda op: op.parse(self.mdg).tocsr()                            # noqa: E731
                intf = {
                    "Upg": P(idc.upwind_primary_gamma()), "Usg": P(idc.upwind_secondary_gamma()),
                    "Upd": P(idc.upwind_primary_delta()), "Usd": P(idc.upwind_secondary_delta()),
                    "Mavg": P(mp.primary_to_mortar_avg()), "Tr": P(pp.ad.Trace(sds).trace),
                    "S2m": P(mp.secondary_to_mortar_avg()),
                    "flux_proj": P(discr.bound_transport_neu_gamma()) @ P(mp.mortar_to_primary_int()),
                    "jump_proj": P(mp.mortar_to_secondary_int()),
                    "nu": np.asarray(
                        self.pair_interface_density_driven_flux(gamma, delta, interfaces).value(es),
                        float),
                }
            out.append({"gamma": gamma, "delta": delta, "Ug": Ug, "Ud": Ud, "nu": nu, "intf": intf})
        return out

    def _imex_interface_coupling(self, pair, l_gamma, l_delta, a_gamma) -> np.ndarray:
        """Mortar-cell interface buoyancy coupling of one pair (HU mobility-product branch, empty
        N=2 background) -- the shared quantity the flux and the jump both project.  Exact replica of
        ``__interface_mp_coupling``: upwind (a*l_gamma) and l_delta onto the mortar from BOTH the
        primary (trace) and secondary sides, normalise by the interface-upwinded total mobility, and
        scale by the interface density-driven flux."""
        d = pair["intf"]
        Mtr = lambda x: d["Mavg"] @ (d["Tr"] @ x)                                    # primary -> mortar
        gi = d["Upg"] @ Mtr(a_gamma * l_gamma) + d["Usg"] @ (d["S2m"] @ (a_gamma * l_gamma))
        di = d["Upd"] @ Mtr(l_delta) + d["Usd"] @ (d["S2m"] @ l_delta)
        lam = ((d["Upg"] @ Mtr(l_gamma) + d["Usg"] @ (d["S2m"] @ l_gamma))
               + (d["Upd"] @ Mtr(l_delta) + d["Usd"] @ (d["S2m"] @ l_delta)) + 1.0e-15)
        return gi * di / lam * d["nu"]

    def _imex_numpy_buoyancy(self, sds, advected_of, buoy_ops) -> np.ndarray:
        """Assemble the subdomain buoyancy FACE flux over ``sds`` in pure numpy, from the frozen
        per-pair operators and the flash weights.  ``advected_of(gamma)`` returns the advected cell
        quantity a_gamma (specific enthalpy for energy, partial fraction for a component).  Exact
        replica of ``__entity_buoyancy_flux`` (HU mobility-product branch, empty N=2 background),
        INCLUDING the --md interface term ``flux_proj @ interface_coupling``."""
        es = self.equation_system
        ff = self._imex_is_fractional_flow(self)
        nfaces = int(sum(sd.num_faces for sd in sds))
        out = np.zeros(nfaces)
        for pair in buoy_ops:
            gamma, delta, Ug, Ud, nu = pair["gamma"], pair["delta"], pair["Ug"], pair["Ud"], pair["nu"]
            l_gamma = np.asarray(self._phase_mass_mobility(gamma, sds).value(es), float)
            l_delta = np.asarray(self._phase_mass_mobility(delta, sds).value(es), float)
            a_gamma = np.asarray(advected_of(gamma), float)
            if ff:
                lam_T = np.asarray(self.total_mass_mobility(sds).value(es), float)
                out += (Ug @ (a_gamma * l_gamma / lam_T)) * (Ud @ (l_delta / lam_T)) * nu
            else:
                num = (Ug @ (a_gamma * l_gamma)) * (Ud @ l_delta)
                den = (Ug @ l_gamma) + (Ud @ l_delta) + 1.0e-15
                out += num / den * nu
            if pair["intf"] is not None:                                             # --md flux side
                out += pair["intf"]["flux_proj"] @ self._imex_interface_coupling(
                    pair, l_gamma, l_delta, a_gamma)
        return out

    def _imex_numpy_buoyancy_jump(self, sds, advected_of, buoy_ops) -> np.ndarray:
        """Assemble the subdomain buoyancy JUMP source over ``sds`` in pure numpy (--md only): the
        SAME interface coupling as the flux, projected to the secondary (lower-dim) cells.  Exact
        replica of ``__entity_buoyancy_jump``."""
        es = self.equation_system
        ncells = int(sum(sd.num_cells for sd in sds))
        out = np.zeros(ncells)
        for pair in buoy_ops:
            if pair["intf"] is None:
                continue
            l_gamma = np.asarray(self._phase_mass_mobility(pair["gamma"], sds).value(es), float)
            l_delta = np.asarray(self._phase_mass_mobility(pair["delta"], sds).value(es), float)
            a_gamma = np.asarray(advected_of(pair["gamma"]), float)
            out += pair["intf"]["jump_proj"] @ self._imex_interface_coupling(
                pair, l_gamma, l_delta, a_gamma)
        return out

    def _imex_force_two_phase_state(self) -> None:
        """Push the state into the two-phase, salt-bearing region so the buoyancy is actually
        exercised: the physical IC is single-phase with z_NaCl=0 (both buoyancy terms identically 0,
        a trivial check).  A cell-indexed enthalpy ramp lands cells across the two-phase envelope, and
        a nonzero overall salt fraction gives the component buoyancy something to advect.  Then the
        flash, the gravity buoyancy DIRECTION and the upwind stencils are refreshed to this state.
        Destructive -- only ever called by the offline verifier."""
        offset, n = self._predictor_cell_offsets()
        zlo, zhi = self._predictor_z_bounds()
        hlo, hhi = self._predictor_h_bounds()
        h_ramp = np.clip(np.linspace(1.2, 2.6, n), hlo, hhi)          # MJ/kg: across the two-phase dome
        z_val = np.clip(np.full(n, 0.1), zlo, zhi)                    # nonzero salt to advect
        self._predictor_scatter(self.enthalpy_variable, h_ramp, offset)
        self._predictor_scatter(self._predictor_overall_fraction_name(), z_val, offset)
        # The interface Darcy flux is a mortar VARIABLE solved implicitly; it is 0 at a hand-built
        # state, which would make the viscous interface-flux check a trivial 0==0.  Set a synthetic
        # nonzero mortar flux so that check is exercised (the assembly is an identity in its value).
        intfs = self.mdg.interfaces()
        if intfs:
            for vname, scale in ((_MORTAR_VAR, 1.0e-2), ("interface_fourier_flux", 1.0e-3)):
                try:
                    mvar = self.equation_system.md_variable(vname, intfs)
                    n_m = int(self.equation_system.dofs_of(mvar.sub_vars).size)
                    self.equation_system.set_variable_values(
                        scale * np.linspace(-1.0, 1.0, n_m), [mvar], iterate_index=0)
                except (KeyError, ValueError):
                    pass                                                 # variable may not exist
        self.update_derived_quantities()                             # re-flash at the new state
        self.refresh_buoyancy_direction()                            # refresh nu = G[g(rho_d-rho_g)]
        self.rediscretize()                                          # refresh the HU upwind stencils

    def _imex_balance_specs(self, sds, intf):
        """Per-balance operator bundle -- weight, the correct SUBDOMAIN advective discretization
        (energy upwinds under the enthalpy keyword, not mobility), the interface coupling, the
        combined boundary operator, and (for the full-residual assembly) the accumulation operator,
        the interface-flux mortar variable and the buoyancy advected-quantity callback -- for mass,
        energy and each non-reference component, exactly as the FI path builds them."""
        es = self.equation_system
        has_i = len(intf) != 0
        specs = [dict(name="mass", w=self.advection_weight_mass_balance(sds),
                      sub=self.mobility_discretization(sds),
                      cpl=self.interface_mobility_discretization(intf) if has_i else None,
                      bc=self.boundary_fluid_flux(sds),
                      acc=self.volume_integral(self.fluid_mass(sds), sds, 1),
                      iflux=self.interface_fluid_flux(intf) if has_i else None,
                      adv=None)]                                       # mass: no buoyancy
        en_bc = self._combine_boundary_operators(
            subdomains=sds, dirichlet_operator=self.advection_weight_energy_balance,
            neumann_operator=self.enthalpy_flux, robin_operator=None,
            bc_type=self.bc_type_enthalpy_flux, name="bc_values_enthalpy")
        specs.append(dict(name="energy", w=self.advection_weight_energy_balance(sds),
                          sub=self.enthalpy_discretization(sds),
                          cpl=self.interface_enthalpy_discretization(intf) if has_i else None,
                          bc=en_bc,
                          acc=self.volume_integral(self.total_internal_energy(sds), sds, 1),
                          iflux=self.interface_enthalpy_flux(intf) if has_i else None,
                          adv=lambda g: self._advected_specific_enthalpy(g, sds).value(es)))
        for comp in list(self.fluid.components)[1:]:
            specs.append(dict(name="comp[%s]" % comp.name,
                              w=self.advection_weight_component_mass_balance(comp, sds),
                              sub=self.mobility_discretization(sds),
                              cpl=self.interface_mobility_discretization(intf) if has_i else None,
                              bc=self.boundary_component_flux(comp, sds),
                              acc=self.volume_integral(self.component_mass(comp, sds), sds, 1),
                              iflux=self.interface_component_flux(comp, intf) if has_i else None,
                              adv=lambda g, c=comp: self._advected_partial_fraction(c, g, sds).value(es)))
        return specs

    def _imex_verify_viscous(self, sds, tol) -> bool:
        """Verify the pure-numpy viscous advective flux against AD, per balance:
          interior   darcy_flux * (upwind @ w)                          -- subdomain faces
          boundary   Bdir @ (darcy_flux * bc) + Bneu @ bc               -- Dirichlet/Neumann faces
          interface  interface_darcy_flux * (Vp@Mavg@Tr@w + Vs@S2m@w)   -- mortar cells
        Each is an identity in w and the (frozen) boundary data / interface darcy, so they hold at any
        state -- exactly what the explicit integrator recomputes with pressure/interface-darcy frozen
        and w read from the flash."""
        es = self.equation_system
        intf = self.subdomains_to_interfaces(sds, [1])
        P = lambda op: op.parse(self.mdg).tocsr()                        # noqa: E731
        darcy = np.asarray(self.darcy_flux(sds).value(es), float)        # global faces (frozen)
        if len(intf) != 0:
            mp = pp.ad.MortarProjections(self.mdg, sds, intf, dim=1)
            Mavg, Tr, S2m = P(mp.primary_to_mortar_avg()), P(pp.ad.Trace(sds).trace), \
                P(mp.secondary_to_mortar_avg())
            idf = np.asarray(self.interface_darcy_flux(intf).value(es), float)  # frozen mortar darcy
        ok = True

        def _chk(label, ad, npv):
            nonlocal ok
            ad = np.asarray(ad, float); npv = np.asarray(npv, float)
            d = float(np.max(np.abs(ad - npv))) if ad.size else 0.0
            s = float(np.max(np.abs(ad))) or 1.0
            ok &= d <= tol * s
            _LOG.info("  %s: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      label, float(np.max(np.abs(ad))) if ad.size else 0.0, d, d / s,
                      "OK" if d <= tol * s else "MISMATCH")

        nfaces = int(sum(sd.num_faces for sd in sds))
        bc_synth = np.linspace(0.5, 1.5, nfaces)                         # synthetic boundary data
        bc_synth_op = pp.wrap_as_dense_ad_array(bc_synth, name="imex_bc_synth")
        for spec in self._imex_balance_specs(sds, intf):
            name, w_op, sub, cpl = spec["name"], spec["w"], spec["sub"], spec["cpl"]
            w = np.asarray(w_op.value(es), float)
            Upw = P(sub.upwind())
            # interior
            _chk("viscous interior[%s]" % name,
                 (self.darcy_flux(sds) * (sub.upwind() @ w_op)).value(es), darcy * (Upw @ w))
            # boundary  Bdir @ (darcy * bc) + Bneu @ bc  (assembly identity: synthetic bc data,
            # since the real per-balance bc operator == frozen data the integrator reads verbatim).
            Bdir, Bneu = P(sub.bound_transport_dir()), P(sub.bound_transport_neu())
            _chk("viscous boundary[%s]" % name,
                 (sub.bound_transport_dir() @ (self.darcy_flux(sds) * bc_synth_op)
                  + sub.bound_transport_neu() @ bc_synth_op).value(es),
                 Bdir @ (darcy * bc_synth) + Bneu @ bc_synth)
            # interface
            if cpl is not None:
                _chk("viscous interface[%s]" % name,
                     self.interface_advective_flux(intf, w_op, cpl).value(es),
                     idf * (P(cpl.upwind_primary()) @ (Mavg @ (Tr @ w))
                            + P(cpl.upwind_secondary()) @ (S2m @ w)))

        # Neumann HEAT flux: the bottom Q enters energy via the Fourier boundary term
        # discr.bound_flux() @ combine_boundary_operators_fourier_flux (the real boundary data,
        # carrying the prescribed bottom heat influx).
        fdiscr = self.fourier_flux_discretization(sds)
        bc_f = self.combine_boundary_operators_fourier_flux(sds)
        _chk("Neumann heat flux (Fourier bnd)",
             (fdiscr.bound_flux() @ bc_f).value(es), P(fdiscr.bound_flux()) @ bc_f.value(es))
        return ok

    def _imex_full_residual_numpy(self, sds, spec, buoy_ops, extra_flux=None,
                                  extra_src=None):
        """Assemble one balance's full residual  dt(acc) + div@flux - source  in numpy, from the
        validated flux pieces + the model's interface mortar-variable values + accumulation operator.
        ``spec`` is a _imex_balance_specs entry (mass has no buoyancy -> buoy_ops filtered by caller).
        ``extra_flux``/``extra_src`` add the energy-only Fourier face flux / interface source."""
        es = self.equation_system
        intf = self.subdomains_to_interfaces(sds, [1])
        P = lambda op: op.parse(self.mdg).tocsr()                        # noqa: E731
        div = pp.ad.Divergence(sds, dim=1).parse(self.mdg).tocsr()       # cells x faces
        darcy = np.asarray(self.darcy_flux(sds).value(es), float)
        name, w_op, sub, cpl, bc_op = (spec["name"], spec["w"], spec["sub"], spec["cpl"], spec["bc"])
        w = np.asarray(w_op.value(es), float)
        # ---- face flux: interior + boundary + interface(primary) + buoyancy ----
        flux = darcy * (P(sub.upwind()) @ w)
        bc = np.asarray(bc_op.value(es), float)
        flux = flux + P(sub.bound_transport_dir()) @ (darcy * bc) + P(sub.bound_transport_neu()) @ bc
        src = np.zeros(int(sum(sd.num_cells for sd in sds)))
        if cpl is not None:
            mp = pp.ad.MortarProjections(self.mdg, sds, intf, dim=1)
            icf = np.asarray(spec["iflux"].value(es), float)             # interface mortar variable
            flux = flux + P(sub.bound_transport_neu()) @ (P(mp.mortar_to_primary_int()) @ icf)
            src = src + P(mp.mortar_to_secondary_int()) @ icf
        if buoy_ops is not None:
            adv = spec["adv"]
            flux = flux + self._imex_numpy_buoyancy(sds, adv, buoy_ops)
            src = src + self._imex_numpy_buoyancy_jump(sds, adv, buoy_ops)
        if extra_flux is not None:
            flux = flux + extra_flux
        if extra_src is not None:
            src = src + extra_src
        # ---- accumulation dt(acc) ----
        dt_acc = np.asarray(
            pp.ad.time_derivatives.dt(spec["acc"], self.ad_time_step).value(es), float)
        divflux = div @ flux
        scale = max(float(np.max(np.abs(dt_acc))), float(np.max(np.abs(divflux))),
                    float(np.max(np.abs(src))), 1e-30)                   # O(1) term size
        return dt_acc + divflux - src, scale

    def _imex_fourier_extra(self, sds, intf):
        """Energy-only Fourier terms for the full residual: face flux  kappa grad(T) + bound_flux@bc
        + bound_flux@(M2p@interface_fourier_flux); secondary source  M2s@interface_fourier_flux."""
        es = self.equation_system
        P = lambda op: op.parse(self.mdg).tocsr()                        # noqa: E731
        fd = self.fourier_flux_discretization(sds)
        T = np.asarray(self.temperature(sds).value(es), float)
        bc_f = np.asarray(self.combine_boundary_operators_fourier_flux(sds).value(es), float)
        flux = P(fd.flux()) @ T + P(fd.bound_flux()) @ bc_f
        src = np.zeros(int(sum(sd.num_cells for sd in sds)))
        if len(intf) != 0:
            mp = pp.ad.MortarProjections(self.mdg, sds, intf, dim=1)
            iff = np.asarray(self.interface_fourier_flux(intf).value(es), float)
            flux = flux + P(fd.bound_flux()) @ (P(mp.mortar_to_primary_int()) @ iff)
            src = src + P(mp.mortar_to_secondary_int()) @ iff
        return flux, src

    def _imex_verify_full(self, sds, tol, buoy_ops) -> bool:
        """End-to-end: assemble the FULL residual  dt(acc) + div@flux - source  in pure numpy for the
        energy and component balances and compare to the model's AD residual (-equation_system.assemble)
        at the current state.  All flux/source pieces were validated bit-exact above; this checks that
        they combine (div, interface primary/secondary projections, accumulation, sign of source) into
        the same residual the FI solver assembles."""
        es = self.equation_system
        intf = self.subdomains_to_interfaces(sds, [1])
        eqs = pp.compositional_flow.get_primary_equations_cf(self)       # [mass, energy, comp...]
        specs = self._imex_balance_specs(sds, intf)
        ok = True
        for spec, eq in zip(specs, eqs):
            if spec["name"] == "mass":
                continue                                                 # pressure is implicit, not marched
            try:
                extra = self._imex_fourier_extra(sds, intf) if spec["name"] == "energy" else (None, None)
                buoy = buoy_ops if spec["adv"] is not None else None
                R_np, scale = self._imex_full_residual_numpy(sds, spec, buoy, extra[0], extra[1])
                R_ad = -np.asarray(es.assemble(evaluate_jacobian=False, equations=[eq]), float)
            except Exception as exc:                                     # e.g. secondaries unset on a hand-built state
                _LOG.info("  FULL residual[%s]: SKIPPED (%s)", spec["name"], exc)
                continue
            d = float(np.max(np.abs(R_ad - R_np)))                       # rel to the O(1) term size
            ok &= d <= tol * scale
            _LOG.info("  FULL residual[%s]: |R_AD|max=%.3e  term=%.3e  max|R_AD-R_np|=%.3e (rel %.3e)  %s",
                      spec["name"], float(np.max(np.abs(R_ad))), scale, d, d / scale,
                      "OK" if d <= tol * scale else "MISMATCH")
        return ok

    def _imex_verify_spatial(self, tol=1e-8) -> bool:
        """Validate the FULLY-UPDATING spatial residual (``_imex_spatial_numpy``, directions recomputed
        from the flash via the static matrices) against AD: at the current state ``S_AD = R_AD -
        dt(acc)``, and the recomputed directions equal PorePy's, so ``S_numpy`` must match.  This is
        the check that the runtime kernel -- hand-coded upwind + recomputed F_total/nu, no AD --
        reproduces the FI spatial residual."""
        es = self.equation_system
        C = self._imex_static()
        off = self._predictor_cell_offsets()
        p = self._predictor_gather("pressure", *off)
        h = self._predictor_gather(self.enthalpy_variable, *off)
        z = self._predictor_gather(self._predictor_overall_fraction_name(), *off)
        S_e, S_z = self._imex_spatial_numpy(p, h, z)
        # isolate: flash weights vs the model's AD advection weights
        pd = self._imex_sample(p, h, z)
        w_m, w_e, w_z = self._imex_weights(pd)
        for nm, wnp, wop in (("w_m", w_m, self.advection_weight_mass_balance(C["sds"])),
                             ("w_e", w_e, self.advection_weight_energy_balance(C["sds"])),
                             ("w_z", w_z, self.advection_weight_component_mass_balance(
                                 list(self.fluid.components)[1:][0], C["sds"]))):
            wad = np.asarray(wop.value(es), float)
            dw = float(np.max(np.abs(wad - wnp))); sw = float(np.max(np.abs(wad))) or 1.0
            _LOG.info("  (flash %s vs AD: rel=%.3e)", nm, dw / sw)
        Tdiff = float(np.max(np.abs(np.asarray(pd["T"], float)
                                    - np.asarray(self.temperature(C["sds"]).value(es), float))))
        _LOG.info("  (flash T vs model T at this state: max|dT|=%.3e)", Tdiff)
        if len(C["intf"]) != 0:
            # The interface enthalpy flux is ALGEBRAIC (= interface_advective_flux(w_e)); the FI leaves
            # a small slop on that variable, which my runtime recomputes consistently.  Sync the
            # variable to the consistent value so the AD reference matches the runtime's recompute.
            F = self._imex_frozen()
            def _iadv(w, Vp, Vs):
                return F["idf"] * (Vp @ (C["Mavg"] @ (C["Tr"] @ w)) + Vs @ (C["S2m"] @ w))
            ie = _iadv(w_e, F["Vp_e"], F["Vs_e"])
            ief_var = es.md_variable("interface_enthalpy_flux", C["intf"])
            rel_before = (float(np.max(np.abs(ie - np.asarray(ief_var.value(es), float))))
                          / (float(np.max(np.abs(ief_var.value(es)))) or 1.0))
            es.set_variable_values(ie, [ief_var], iterate_index=0)
            _LOG.info("  (synced interface_enthalpy_flux to consistency; FI slop was rel=%.3e)",
                      rel_before)
        eqs = pp.compositional_flow.get_primary_equations_cf(self)       # [mass, energy, comp...]
        specs = self._imex_balance_specs(C["sds"], C["intf"])
        ok = True
        for spec, eq, S_np in ((specs[1], eqs[1], S_e), (specs[2], eqs[2], S_z)):
            R_ad = -np.asarray(es.assemble(evaluate_jacobian=False, equations=[eq]), float)
            dt_acc = np.asarray(
                pp.ad.time_derivatives.dt(spec["acc"], self.ad_time_step).value(es), float)
            S_ad = R_ad - dt_acc
            term = float(np.max(np.abs(S_ad))) or 1.0
            diff = np.abs(S_ad - S_np)
            d = float(np.max(diff))
            ok &= d <= tol * term
            _LOG.info("  SPATIAL updating[%s]: |S_AD|max=%.3e  max|S_AD-S_np|=%.3e (rel %.3e)  %s",
                      spec["name"], float(np.max(np.abs(S_ad))), d, d / term,
                      "OK" if d <= tol * term else "MISMATCH")
            if d > tol * term:                                          # localize the mismatch
                i, csum, loc = int(np.argmax(diff)), 0, "?"
                for sd in C["sds"]:
                    if i < csum + sd.num_cells:
                        loc = "dim%d local-cell %d/%d" % (sd.dim, i - csum, sd.num_cells)
                        break
                    csum += sd.num_cells
                _LOG.info("      -> worst at global cell %d (%s): S_AD=%.3e  S_np=%.3e",
                          i, loc, float(S_ad[i]), float(S_np[i]))
        return ok

    def imex_verify_full_residual(self) -> bool:
        """Run the full-residual end-to-end check at the CURRENT (real, converged) state -- parses the
        buoyancy operators fresh and compares numpy vs AD for energy + component.  Meant to be fired
        mid-run from a two-phase converged step (see after_nonlinear_convergence), where every
        eliminated secondary is consistently stored so the AD residual assembles."""
        sds = self.mdg.subdomains()
        self.update_derived_quantities()                                # sync the stored flash secondaries
        buoy_ops = self._imex_parse_buoyancy_ops(sds)
        _LOG.info("IMEX FULL-residual check at real state (t=%.1f yr)",
                  self.time_manager.time / (365.0 * 86400.0))
        ok = self._imex_verify_full(sds, 1e-8, buoy_ops)
        ok &= self._imex_verify_spatial()                               # the updating-direction kernel
        _LOG.info("IMEX FULL-residual: %s", "PASS" if ok else "FAIL")
        return ok

    def after_nonlinear_convergence(self, *args, **kwargs) -> None:
        super().after_nonlinear_convergence(*args, **kwargs)
        # Fire the full-residual end-to-end check ONCE, at the first REAL converged step (every
        # eliminated secondary consistently stored, real interface fluxes solved), then keep running.
        if self.params.get("imex_verify_full", False) and not self.__dict__.get("_imex_full_done"):
            self._imex_full_done = True
            self.imex_verify_full_residual()

    def _imex_verify_fourier(self, sds, tol) -> bool:
        """Fourier conduction, per the model's discretization:
          interior   discr.flux() @ T                                   -- kappa grad(T), T from the flash
          interface  bound_flux @ mortar_to_primary_int @ interface_fourier_flux
        The interface Fourier flux is a Robin mortar VARIABLE (its trace reconstruction depends on the
        variable itself -- genuinely implicit, like interface_darcy_flux), so the explicit march reads
        it frozen from the implicit block; validated here as an assembly identity in a synthetic value.
        The Fourier vector source is identically zero.  Also confirms the flash temperature the
        integrator would use equals the model's own temperature."""
        es = self.equation_system
        intf = self.subdomains_to_interfaces(sds, [1])
        P = lambda op: op.parse(self.mdg).tocsr()                        # noqa: E731
        fd = self.fourier_flux_discretization(sds)
        ok = True

        def _chk(label, ad, npv):
            nonlocal ok
            ad = np.asarray(ad, float); npv = np.asarray(npv, float)
            d = float(np.max(np.abs(ad - npv))) if ad.size else 0.0
            s = float(np.max(np.abs(ad))) or 1.0
            ok &= d <= tol * s
            _LOG.info("  %s: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      label, float(np.max(np.abs(ad))) if ad.size else 0.0, d, d / s,
                      "OK" if d <= tol * s else "MISMATCH")

        # interior  kappa grad(T)
        T = np.asarray(self.temperature(sds).value(es), float)
        _chk("Fourier interior (k grad T)", (fd.flux() @ self.temperature(sds)).value(es),
             P(fd.flux()) @ T)
        # the explicit integrator uses the flash temperature; confirm it equals the model's T
        off = self._predictor_cell_offsets()
        pd = self._imex_sample(self._predictor_gather("pressure", *off),
                               self._predictor_gather(self.enthalpy_variable, *off),
                               self._predictor_gather(self._predictor_overall_fraction_name(), *off))
        _chk("flash T vs model T", T, pd["T"])
        # interface Fourier flux contribution (frozen mortar variable)
        if len(intf) != 0:
            mp = pp.ad.MortarProjections(self.mdg, sds, intf, dim=1)
            iff_op = self.interface_fourier_flux(intf)
            _chk("Fourier interface", (fd.bound_flux() @ (mp.mortar_to_primary_int() @ iff_op)).value(es),
                 P(fd.bound_flux()) @ (P(mp.mortar_to_primary_int()) @ np.asarray(iff_op.value(es), float)))
        return ok

    def imex_verify_residuals(self, tol: float = 1e-8, two_phase: bool = True) -> bool:
        """Built-in test (your step 3): verify the pure-numpy buoyancy assembly reproduces the
        model's AD operator value BIT-EXACTLY, per equation, at a reference state.  Uses the frozen
        operators parsed from AD and the model's own flash weights, so a pass proves the
        static-operator numpy kernel == the FI discretization.  With ``two_phase`` the state is first
        driven across the phase boundary (the IC is single-phase, hence buoyancy 0).  Returns True on
        pass; run offline via ``--imex-verify``, never in the hot path."""
        es = self.equation_system
        sds = self.mdg.subdomains()
        self.update_derived_quantities()                # fresh flash + buoyancy direction + rediscretize
        if two_phase:
            self._imex_force_two_phase_state()          # exercise the buoyancy (IC is single-phase)
        buoy_ops = self._imex_parse_buoyancy_ops(sds)
        ok = True
        # Reference-state diagnostics: fraction of two-phase cells and the pair gravity-flux scale,
        # so a "0 == 0" pass can be told apart from a genuine non-degenerate match.
        pd = self._imex_sample(self._predictor_gather("pressure", *self._predictor_cell_offsets()),
                               self._predictor_gather(self.enthalpy_variable, *self._predictor_cell_offsets()),
                               self._predictor_gather(self._predictor_overall_fraction_name(),
                                                      *self._predictor_cell_offsets()))
        sv = np.asarray(pd["s_v"], float)
        n_2ph = int(np.sum((sv > 1e-9) & (sv < 1.0 - 1e-9)))
        nu_scale = max((float(np.max(np.abs(p["nu"]))) for p in buoy_ops), default=0.0)
        drho = float(np.max(np.abs(np.asarray(pd["rho_v"], float) - np.asarray(pd["rho_l"], float))))
        _LOG.info("IMEX verify | %d subdomains, %d ordered pairs, is_fractional_flow=%s, two_phase=%s",
                  len(sds), len(buoy_ops), self._imex_is_fractional_flow(self), two_phase)
        _LOG.info("  reference state | two-phase cells=%d/%d, max|rho_v-rho_l|=%.1f, max|pair nu|=%.3e",
                  n_2ph, sv.size, drho, nu_scale)

        # -- energy buoyancy --
        ad_e = np.asarray(self.enthalpy_buoyancy(sds).value(es), float)
        np_e = self._imex_numpy_buoyancy(
            sds, lambda g: self._advected_specific_enthalpy(g, sds).value(es), buoy_ops)
        d_e = float(np.max(np.abs(ad_e - np_e))) if ad_e.size else 0.0
        s_e = float(np.max(np.abs(ad_e))) or 1.0
        ok &= d_e <= tol * s_e
        _LOG.info("  enthalpy_buoyancy : |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                  float(np.max(np.abs(ad_e))) if ad_e.size else 0.0,
                  d_e, d_e / s_e, "OK" if d_e <= tol * s_e else "MISMATCH")

        # -- component buoyancy (per non-reference component; canonical component objects) --
        for comp in list(self.fluid.components)[1:]:
            ad_c = np.asarray(self.component_buoyancy(comp, sds).value(es), float)
            np_c = self._imex_numpy_buoyancy(
                sds, lambda g, c=comp: self._advected_partial_fraction(c, g, sds).value(es),
                buoy_ops)
            d_c = float(np.max(np.abs(ad_c - np_c))) if ad_c.size else 0.0
            s_c = float(np.max(np.abs(ad_c))) or 1.0
            ok &= d_c <= tol * s_c
            _LOG.info("  component_buoyancy[%s]: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      comp.name, float(np.max(np.abs(ad_c))) if ad_c.size else 0.0,
                      d_c, d_c / s_c, "OK" if d_c <= tol * s_c else "MISMATCH")

        # -- viscous advective flux (interior darcy*upwind(w)) + interface advective flux --
        ok &= self._imex_verify_viscous(sds, tol)

        # -- hand-coded upwind (directions recomputed from sign) vs PorePy's frozen upwind --
        ok &= self._imex_verify_upwind(sds, tol)

        # -- flux DIRECTIONS (F_total, nu) reconstructed from static matrices + flash --
        ok &= self._imex_verify_directions(sds, tol)

        # -- Fourier conduction (interior kappa grad(T) + interface Fourier flux) --
        ok &= self._imex_verify_fourier(sds, tol)

        # -- FULL residual dt(acc)+div@flux-source vs the model's AD residual --
        ok &= self._imex_verify_full(sds, tol, buoy_ops)

        # -- mortar JUMP source terms (--md only; a no-op assert of 0==0 on a fixed-dim grid) --
        has_intf = any(p["intf"] is not None for p in buoy_ops)
        if has_intf:
            ad_ej = np.asarray(self.enthalpy_buoyancy_jump(sds).value(es), float)
            np_ej = self._imex_numpy_buoyancy_jump(
                sds, lambda g: self._advected_specific_enthalpy(g, sds).value(es), buoy_ops)
            d_ej = float(np.max(np.abs(ad_ej - np_ej))) if ad_ej.size else 0.0
            s_ej = float(np.max(np.abs(ad_ej))) or 1.0
            ok &= d_ej <= tol * s_ej
            _LOG.info("  enthalpy_buoyancy_JUMP : |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                      float(np.max(np.abs(ad_ej))) if ad_ej.size else 0.0,
                      d_ej, d_ej / s_ej, "OK" if d_ej <= tol * s_ej else "MISMATCH")
            for comp in list(self.fluid.components)[1:]:
                ad_cj = np.asarray(self.component_buoyancy_jump(comp, sds).value(es), float)
                np_cj = self._imex_numpy_buoyancy_jump(
                    sds, lambda g, c=comp: self._advected_partial_fraction(c, g, sds).value(es),
                    buoy_ops)
                d_cj = float(np.max(np.abs(ad_cj - np_cj))) if ad_cj.size else 0.0
                s_cj = float(np.max(np.abs(ad_cj))) or 1.0
                ok &= d_cj <= tol * s_cj
                _LOG.info("  component_buoyancy_JUMP[%s]: |AD|max=%.3e  max|AD-np|=%.3e (rel %.3e)  %s",
                          comp.name, float(np.max(np.abs(ad_cj))) if ad_cj.size else 0.0,
                          d_cj, d_cj / s_cj, "OK" if d_cj <= tol * s_cj else "MISMATCH")

        _LOG.info("IMEX verify | %s", "ALL PASS" if ok else "FAILURES ABOVE")
        return ok

    def _imex_adv_div(self, frozen, w, offset) -> np.ndarray:
        """Advective flux divergence of a balance, ``div @ (darcy * (upwind @ weight))``, over the
        mdg -- pure numpy + sparse matvecs.  Captures interior + outflow-boundary faces; the
        inflow-boundary / conduction / mortar terms live in the frozen ``rest``."""
        out = np.zeros_like(w)
        for sd, (darcy, upwind, div) in frozen.items():
            b = offset[sd]
            face = darcy * (upwind @ w[b:b + sd.num_cells])
            out[b:b + sd.num_cells] = div @ face
        return out

    def _imex_terms(self, pd, p, h, z, vol):
        """Energy/salt accumulation and the capacities d(acc_e)/dh, d(acc_z)/dz from a PRE-COMPUTED
        flash ``pd`` (no re-flash).  Rock T_ref cancels in the d_t difference and in the capacity."""
        rho, T = pd["rho"], pd["T"]
        phi = float(self.solid.porosity)
        rock = (1.0 - phi) * float(self.solid.density) * float(self.solid.specific_heat_capacity)
        acc_e = vol * (phi * (rho * h - p) + rock * T)
        acc_z = vol * (phi * rho * z)
        eps = 1.0e-30
        cap_e_h = vol * (phi * (rho + h * pd["dR_dh"]) + rock * pd["dT_dh"])   # d(acc_e)/dh
        cap_z_z = vol * (phi * (rho + z * pd["dR_dz"]))                        # d(acc_z)/dz
        cap_e_h = np.where(np.abs(cap_e_h) < eps, eps, cap_e_h)
        cap_z_z = np.where(np.abs(cap_z_z) < eps, eps, cap_z_z)
        return acc_e, acc_z, cap_e_h, cap_z_z

    # ---- advective CFL from the frozen operators + flash mobility (no matrix build, no AD) -----
    def _imex_cfl_dt(self, frozen, w_m, rho, vol, offset) -> float:
        """Weis-Eq.27 limit dt <= cfl * min(cell fluid mass / total mass outflux), from the FROZEN
        darcy/upwind and the flash total-mass mobility ``w_m`` (a bincount over faces)."""
        phi = float(self.solid.porosity)
        best = np.inf
        for sd, (darcy, upwind, _div) in frozen.items():
            nc, b = sd.num_cells, offset[sd]
            face_mass = darcy * (upwind @ w_m[b:b + nc])                   # [kg/s] total mass flux per face
            cf = sd.cell_faces.tocoo()
            contrib = np.maximum(cf.data * face_mass[cf.row], 0.0)        # mass LEAVING each cell
            outflux = np.bincount(cf.col, weights=contrib, minlength=nc)
            acc_mass = phi * rho[b:b + nc] * vol[b:b + nc]
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.where(outflux > 1e-30, acc_mass / outflux, np.inf)
            if lim.size:
                best = min(best, float(np.min(lim)))
        cfl = float(self.params.get("imex_cfl", 0.25))    # 0.25: conservative for buoyant (gravity) runs
        return cfl * best if np.isfinite(best) else np.inf
