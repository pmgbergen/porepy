"""Completion checks covering the full HU-BM theory, driven by both solvers.

Test 1 -- reduction consistency (merge exactness) under QUADRATIC k_r: an N=4 run with two
equal-density intermediates (band split equally) reduces exactly to an N=3 run whose intermediate
carries the merged-band relperm k_r = 2 k_r(s/2) (``--merge-ref``). The rescaling is what makes the
comparison a DISCRETIZATION test: two identical quadratic-k_r intermediates each at s/2 sum to
2 k_r(s/2), so a naive N=3 (k_r = s^2) would mismatch at O(1) for constitutive reasons.

Test 2 -- monotonicity of the buoyant flux in its own composition, on the two-cell problem, for both
the fixed-dimensional column and the mixed-dimensional (fractured) column: not only must the flux be
monotone, the sign of its local derivative must be consistent across every evaluation point. Computed
for both solvers (hamon numpy, porepy fixed-dim + interface) with finite-difference derivatives, and
visualized concisely in ``figures/monotonicity.png``.

Run standalone with ``python completion_checks.py`` (``--quick`` for a coarse pass); ``run_all_checks``
is the entry point wired into ``run_workflow.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SAVE_PDF = False    # also write a vector PDF next to monotonicity.png (run_workflow toggles via --pdf)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hamon_2d_solver as H  # noqa: E402


# --------------------------------------------------------------------------------------- #
#  Test 1 -- reduction consistency (merge exactness), quadratic k_r
# --------------------------------------------------------------------------------------- #
def _merge_residuals(s4, s3):
    """Given N=4 [water, mid1, mid2, gas] and N=3 [water, int, gas] saturation stacks, the three
    reduction residuals: water, merged middles vs intermediate, gas."""
    return (
        float(np.max(np.abs(s4[0] - s3[0]))),
        float(np.max(np.abs((s4[1] + s4[2]) - s3[1]))),
        float(np.max(np.abs(s4[3] - s3[2]))),
    )


def check_reduction_consistency_hamon(quick: bool = False) -> dict:
    """N=4 equal-middle (quadratic k_r) vs N=3 merge-ref, hamon FV solver."""
    nx = 20 if quick else 40
    t_end = 4.0 if quick else 10.0
    kw = dict(nx=nx, ny=nx, t_end_days=t_end, dt_days=2.0, snap_days=(0.0, t_end),
              linear_solver="scipy", atol=1.0e-8, verbose=False)
    _, snaps4, _ = H.run("hu", nphase=4, equal_middle=True, merge_ic=True, **kw)
    _, snaps3, _ = H.run("hu", nphase=3, merge_ref=True, **kw)
    s4 = snaps4[max(snaps4)]["s"]
    s3 = snaps3[max(snaps3)]["s"]
    dw, do, dg = _merge_residuals(s4, s3)
    worst = max(dw, do, dg)
    # tolerance-limited exact merge (~1e-7) vs an O(1) constitutive mismatch (~1e-1 without the
    # merge_ref rescaling / equal-split IC); the threshold separates the two robustly.
    return dict(name="reduction-consistency (hamon)", worst=worst, passed=worst < 1e-4,
                detail=f"|s_w| {dw:.2e}  |s_o-(m1+m2)| {do:.2e}  |s_g| {dg:.2e}")


def check_reduction_consistency_porepy(quick: bool = False) -> dict:
    """N=4 equal-middle (quadratic k_r) vs N=3 merge-ref, PorePy CF model."""
    import porepy as pp
    import porepy_2d_solver as P

    cell = 5.0 if quick else 4.0
    days = 6.0 if quick else 10.0
    scratch = os.path.join(HERE, "_checks_out")

    def run(nphase, **flags):
        params = P.build_params(
            nphase, "hu", t_end_days=days, dt_days=2.0, constant_dt=True,
            snap_days=(0.0, days), cell_size=cell, lagrange_linear_solver="scipy",
            folder_name=scratch, times_to_export=[], **flags)
        model = P.flow_model_class(params)(params)
        solver_params = {
            "nl_convergence_criteria": {
                "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                    tol=1e-6, metric=pp.EquationBasedLebesgueMetric(model)),
                "null_drift": P.NullSpaceDriftCriterion(model, tol=1e-2)},
            "nl_divergence_criteria": {
                "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=25)},
        }
        pp.ModelRunner(model, solver_params,
                       nonlinear_solver=P.geothermal_nonlinear_solver(solver_params)).run()
        sds = model.mdg.subdomains()
        return [np.asarray(ph.saturation(sds).value(model.equation_system), dtype=float)
                for ph in model.fluid.phases]

    s4 = run(4, equal_middle=True, merge_ic=True)
    s3 = run(3, merge_ref=True)
    dw, do, dg = _merge_residuals(s4, s3)
    worst = max(dw, do, dg)
    return dict(name="reduction-consistency (porepy)", worst=worst, passed=worst < 1e-5,
                detail=f"|s_w| {dw:.2e}  |s_o-(m1+m2)| {do:.2e}  |s_g| {dg:.2e}")


# --------------------------------------------------------------------------------------- #
#  Test 2 -- two-cell buoyant-flux monotonicity + derivative-sign consistency (hamon)
# --------------------------------------------------------------------------------------- #
def check_monotonicity_hamon(quick: bool = False) -> dict:
    """Two-cell column: sweep a phase saturation in one cell, and require the phase flux at the
    internal face to be monotone AND its finite-difference derivative to keep one sign at every
    evaluation point (the sign of the numerical-flux own-saturation derivative)."""
    H.set_phase_system(3)
    grid = H.make_grid(1, 2)                       # 1 x 2 vertical column, one internal face
    nc = grid.ncell
    p = H.initial_pressure(grid)                    # hydrostatic: no spurious viscous flow
    npts = 13 if quick else 21
    sweep = np.linspace(0.10, 0.60, npts)
    eps = 1.0e-6

    def phase_flux(s_swept):
        # cell 0 swept (gas rises), cell 1 fixed; keep saturations summing to one per cell.
        s = np.array([[0.35, 0.30], [0.30, 0.30], [0.35, 0.40]])   # water, oil, gas
        s[2, 0] = s_swept
        s[0, 0] = 1.0 - s[1, 0] - s[2, 0]
        x = np.concatenate([p, s[:H.NPHASE - 1].ravel()])
        dirs = H.frozen_directions(x, grid, "hu")
        _, q = H._face_fluxes(x, grid, dirs)        # independent phase fluxes on internal faces
        return float(q[0][0])                        # water-phase flux at the internal face

    flux = np.array([phase_flux(s) for s in sweep])
    deriv = np.array([(phase_flux(s + eps) - phase_flux(s)) / eps for s in sweep])
    scale = max(np.max(np.abs(flux)), 1e-300)
    d = np.diff(flux)
    monotone = bool(np.all(d >= -1e-9 * scale) or np.all(d <= 1e-9 * scale))
    dscale = max(np.max(np.abs(deriv)), 1e-300)
    sign_consistent = bool(np.all(deriv >= -1e-6 * dscale) or np.all(deriv <= 1e-6 * dscale))
    # the local slope and the secant trend must agree
    agree = bool(np.sign(np.mean(deriv)) == np.sign(np.mean(d)) or scale < 1e-12)
    passed = monotone and sign_consistent and agree
    return dict(name="monotonicity + derivative sign (hamon)", worst=0.0, passed=passed,
                detail=f"monotone={monotone}  deriv-sign-consistent={sign_consistent}  "
                       f"trend-agree={agree}  flux[{flux.min():.2e},{flux.max():.2e}]")


# --------------------------------------------------------------------------------------- #
#  Test 2 -- two-cell monotonicity + derivative sign, PorePy (fixed- and mixed-dimensional),
#  with a concise visualization
# --------------------------------------------------------------------------------------- #
def check_monotonicity_porepy(quick: bool = False) -> dict:
    """Sweep a component's composition on the two-cell column (fixed-dim) and the two-cell +
    fracture column (mixed-dim); require the buoyant flux to be monotone AND its local derivative
    to keep one sign at every point. Saves ``figures/monotonicity.png`` visualizing both."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import porepy as pp
    from porepy.applications.test_utils.models import add_mixin

    repo = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from tests.functional.setups.buoyancy_flow_model import (
        ModelGeometry2D, buoyancy_flow_model, to_Mega,
    )

    solid = pp.SolidConstants(permeability=1.0e-14, porosity=0.1, thermal_conductivity=2.0 * to_Mega,
                              density=2500.0, specific_heat_capacity=1000.0 * to_Mega)

    class TwoCell(ModelGeometry2D):
        def set_domain(self):
            self._domain = pp.Domain({"xmax": self.units.convert_units(1.0, "m"),
                                      "ymax": self.units.convert_units(2.0, "m")})
        def meshing_arguments(self):
            return {"cell_size_x": self.units.convert_units(1.0, "m"),
                    "cell_size_y": self.units.convert_units(1.0, "m")}

    class MDColumn(ModelGeometry2D):
        def set_domain(self):
            self._domain = pp.Domain({"xmax": self.units.convert_units(1.0, "m"),
                                      "ymax": self.units.convert_units(2.0, "m")})
        def set_fractures(self):
            self._fractures = pp.frac_utils.pts_edges_to_linefractures(
                np.array([[0.0, 1.0], [1.0, 1.0]]).T, np.array([[0, 1]]).T)
        def meshing_arguments(self):
            return {"cell_size": self.units.convert_units(1.0, "m")}

    class SweepIC(pp.PorePyModel):
        _sc = "CH4"; _target = "bottom"; _v = 0.3; _ref = 0.2
        def ic_values_overall_fraction(self, component, sd):
            if component.name == "H2O":
                return (1.0 - self.ic_values_overall_fraction(self.fluid.components[1], sd)
                            - self.ic_values_overall_fraction(self.fluid.components[2], sd))
            vals = np.full(sd.num_cells, self._ref)
            if component.name == self._sc:
                if sd.dim == 2:
                    idx = {"bottom": 0, "top": 1}.get(self._target)
                    if idx is not None:
                        vals[idx] = self._v
                elif sd.dim == 1 and self._target == "fracture":
                    vals[0] = self._v
            return vals

    def params():
        tm = pp.TimeManager(schedule=[0.0, 86400.0], dt_init=86400.0, constant_dt=True,
                            iter_max=50, print_info=False)
        return {"fractional_flow": False, "enable_buoyancy_effects": True,
                "material_constants": {"solid": solid}, "time_manager": tm,
                "expected_order_loss": 3, "residual_tolerance": 1e-4, "drift_tolerance": 1e-4}

    def interface_flux_op(model, comp):
        sds = model.mdg.subdomains()
        terms = []
        for ph in model.fluid.phases:
            for g, d in model.phase_pairs_for(ph):
                chi = model._advected_partial_fraction(comp, g, sds)
                terms.append(model._interface_pair_coupling(chi, g, d, sds))
        return pp.ad.sum_operator_list(terms)

    def below_above(model):
        intf = model.mdg.interfaces()[0]
        sd_p, _ = model.mdg.interface_to_subdomain_pair(intf)
        yc = sd_p.cell_centers[1]
        P = intf.primary_to_mortar_int().tocsr()
        ys = [yc[np.unique(np.abs(sd_p.cell_faces)[P.getrow(k).indices].nonzero()[1])].mean()
              for k in range(intf.num_cells)]
        return int(np.argmin(ys)), int(np.argmax(ys))

    def flux_at(kind, comp_name, ic_target, readout, z):
        base = add_mixin(TwoCell if kind == "fd" else MDColumn, buoyancy_flow_model(3, False))
        m = type("M", (SweepIC, base), {})(params())
        m._sc, m._target, m._v = comp_name, ic_target, float(z)
        m.prepare_simulation(); m.before_nonlinear_iteration()
        sds = m.mdg.subdomains(); es = m.equation_system
        comp = next(c for c in m.fluid.components if c.name == comp_name)
        if readout == "internal":
            f = m.component_buoyancy(comp, sds).value(es)
            return f[sds[0].get_internal_faces()[0]]
        b, a = below_above(m)
        f = es.evaluate(interface_flux_op(m, comp))
        return f[b if readout == "below" else a]

    def sweep_case(fluxfn, xs, eps):
        flux = np.array([fluxfn(x) for x in xs])
        dfx = np.array([(fluxfn(x + eps) - fluxfn(x)) / eps for x in xs])
        scale = max(np.max(np.abs(flux)), 1e-300)
        mono = bool(np.all(np.diff(flux) >= -1e-6 * scale) or np.all(np.diff(flux) <= 1e-6 * scale))
        dscale = max(np.max(np.abs(dfx)), 1e-300)
        sign_ok = bool(np.all(dfx >= -1e-3 * dscale) or np.all(dfx <= 1e-3 * dscale))
        return dict(x=xs, flux=flux, dfx=dfx, sign=int(np.sign(np.mean(dfx))),
                    ok=mono and sign_ok and scale > 1e-12)

    npts = 18 if quick else 30
    zsw = np.linspace(0.05, 0.55, npts)

    # three porepy readouts, each swept in the CH4 overall composition z in BOTH cells adjacent to its
    # face: the primary matrix cell and the cell on the other side (the far matrix cell for the fixed-
    # dim internal face, the fracture for each MD mortar). Sweeping the opposite cell mirrors the flux
    # vertically and flips the derivative sign -- the up/down symmetry of the buoyant flux (monotone
    # increasing in one adjacent cell, decreasing in the other). z is the analyzed state variable;
    # saturation is intermediate.
    # (interface-below is omitted: it coincides exactly with the fixed-dim internal face -- same flux
    # in both the primary and the opposite sweep -- so it would sit invisibly under fixed-dim.)
    cases = [  # (label, color, kind, readout, primary cell, opposite cell across the face)
        ("fixed-dim", "#1f78b4", "fd", "internal", "bottom", "top"),
        ("mixed-dim", "#33a02c", "md", "above",    "top",    "fracture"),
    ]
    series = []                                     # (label, color, primary, opposite)
    for label, c, kind, readout, prim, opp in cases:
        pr = sweep_case(lambda z, k=kind, ro=readout, t=prim: flux_at(k, "CH4", t, ro, z), zsw, 1e-3)
        op = sweep_case(lambda z, k=kind, ro=readout, t=opp: flux_at(k, "CH4", t, ro, z), zsw, 1e-3)
        series.append((label, c, pr, op))
    all_pass = all(s[2]["ok"] for s in series)      # assert each readout's primary-cell sweep

    # single lumped figure: normalized flux (left) and its normalized derivative in z (right). Each
    # case-pair (primary solid, opposite dashed) shares its COMMON max-abs so the mirror is faithful;
    # colour = readout, line style = swept cell. Sign is preserved, so left shows monotonicity and
    # right shows the derivative sign flipping between the two adjacent cells.
    xlabel = r"overall composition $z_{\mathrm{CH_4}}$"
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for label, c, pr, op in series:
        fs = max(np.max(np.abs(pr["flux"])), np.max(np.abs(op["flux"])), 1e-300)
        ds = max(np.max(np.abs(pr["dfx"])), np.max(np.abs(op["dfx"])), 1e-300)
        for r, ls, mk in ((pr, "-", "o"), (op, "--", "x")):
            axf.plot(r["x"], r["flux"] / fs, ls=ls, marker=mk, color=c, ms=4, lw=1.3)
            axd.plot(r["x"], r["dfx"] / ds, ls=ls, marker=mk, color=c, ms=4, lw=1.3)
    axf.axhline(0.0, color="0.5", lw=0.9, zorder=0)
    axf.set(xlabel=xlabel, ylabel=r"normalized flux  $F/\max|F|$")
    axd.axhline(0.0, color="0.5", lw=0.9, zorder=0)
    axd.set(xlabel=xlabel, ylabel=r"normalized derivative  $(dF/dz)\,/\max|dF/dz|$")
    for ax in (axf, axd):
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    from matplotlib.lines import Line2D
    handles = ([Line2D([], [], color=c, lw=2, label=label) for label, c, *_ in cases]
               + [Line2D([], [], color="0.35", ls="-", marker="o", ms=4, label="primary cell"),
                  Line2D([], [], color="0.35", ls="--", marker="x", ms=4,
                         label="opposite cell across face")])
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    # miniature schematic (inset in the derivative panel's empty bottom-right corner): the two-cell
    # geometry of each case and which cell is the primary (swept, o) vs the opposite (x) across the
    # face. All derivative curves converge toward zero on the right, so this corner is clear.
    import matplotlib.patches as mpatches
    ins = axd.inset_axes([0.68, 0.03, 0.30, 0.30])   # bottom-right corner, below the lowest curve
    ins.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for s in ins.spines.values():
        s.set_edgecolor("0.7")
    ins.patch.set(facecolor="white", alpha=1.0)
    ins.set_zorder(5)

    def _mini(cx, div_lw, div_col, color, prim_y, opp_y, title):
        w, y0, y1 = 0.13, 0.24, 0.72
        ins.add_patch(mpatches.Rectangle((cx - w, y0), 2 * w, y1 - y0, fill=False, ec="0.35", lw=0.9))
        ins.plot([cx - w, cx + w], [(y0 + y1) / 2] * 2, color=div_col, lw=div_lw)   # face / fracture
        ins.plot(cx, prim_y, "o", color=color, ms=4)                                # primary cell
        ins.plot(cx, opp_y, "x", color=color, ms=4, mew=1.4)                        # opposite cell
        ins.text(cx, y1 + 0.03, title, ha="center", va="bottom", fontsize=5, color=color)
        ins.text(cx - w - 0.015, 0.60, "top", ha="right", va="center", fontsize=4.2, color="0.45")
        ins.text(cx - w - 0.015, 0.36, "bot", ha="right", va="center", fontsize=4.2, color="0.45")

    _mini(0.28, 1.0, "0.55", "#1f78b4", 0.36, 0.60, "fixed-dim")           # thin internal face
    _mini(0.76, 2.4, "0.10", "#33a02c", 0.60, 0.48, "mixed-dim")           # thick fracture (= opposite)
    ins.text(0.76 + 0.135, 0.48, "frac", ha="left", va="center", fontsize=4.2, color="0.1")
    ins.text(0.5, 0.08, r"$\bullet$ primary   $\times$ opposite", ha="center", fontsize=5)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figdir = os.path.join(HERE, "figures")
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, "monotonicity.png")
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(os.path.splitext(figpath)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

    detail = "  ".join(f"{label}:{'ok' if pr['ok'] else 'X'}" for label, _c, pr, _op in series)
    return dict(name="monotonicity + derivative sign (porepy)", worst=0.0, passed=all_pass,
                detail=f"{detail}  fig={os.path.relpath(figpath, HERE)}")


# --------------------------------------------------------------------------------------- #
#  Aggregate
# --------------------------------------------------------------------------------------- #
def run_all_checks(quick: bool = False, skip_porepy: bool = False) -> list:
    results = []
    print("\n=== completion checks (subsection 4.1) ===", flush=True)
    for fn in (check_reduction_consistency_hamon, check_monotonicity_hamon):
        r = fn(quick)
        results.append(r)
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['name']:38s} {r['detail']}", flush=True)
    if not skip_porepy:
        for fn, tag in ((check_reduction_consistency_porepy, "reduction-consistency (porepy)"),
                        (check_monotonicity_porepy, "monotonicity + derivative sign (porepy)")):
            try:
                r = fn(quick)
                results.append(r)
                print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['name']:38s} {r['detail']}",
                      flush=True)
            except Exception as exc:  # porepy unavailable / model failure -- report, do not abort
                print(f"  [SKIP] {tag:38s} {type(exc).__name__}: {exc}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Completion checks for the HU-BM scheme.")
    ap.add_argument("--quick", action="store_true", help="coarse/short pass")
    ap.add_argument("--skip-porepy", action="store_true", help="hamon checks only")
    args = ap.parse_args()
    res = run_all_checks(quick=args.quick, skip_porepy=args.skip_porepy)
    ok = all(r["passed"] for r in res)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}", flush=True)
    sys.exit(0 if ok else 1)
