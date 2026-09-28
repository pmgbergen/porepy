#!/usr/bin/env python
"""PorePy figures for subsection 4.2, from the exported VTU snapshots (no simulations).

Per N in --nphase (default 3 4) and per case (fixed-dim, --md):
  saturation_maps_pp_hu[_md]  -- diverging composite maps sum_k c_k s_k, c = linspace(-1, 1, N)
                                 (one row over --days), layout mirroring plot_reference.plot_maps
  saturation_grid_pp_hu[_md]  -- per-phase saturation grid (rows = days, cols = phases),
                                 layout mirroring plot_reference.plot_grid
  conservation_pp_hu[_md]     -- RELATIVE losses of the volume-averaged totals vs time, both
                                 referenced to the initial state:
                                     mass   |<rho>(t) - <rho>(0)| / <rho>(0),
                                     energy |<E>(t) - <E>(0)| / <E>(0),
                                     rho_mix = sum_i s_i rho_i,
                                     E = phi (rho h - p) + (1-phi) rho_s c_ps T,
                                 with the actual specific volumes A (2D), len*a (1D), a^2 (0D).
  saturation_conservation_pp_hu[_md] -- per-phase RELATIVE pore-volume losses
                                 |V_i(t) - V_i(0)| / V_pore, V_i = phi int s_i dV,
                                 V_pore = phi V_tot (uniform phi cancels); these decompose the
                                 mass drift: M(t) - M(0) = sum_i rho_i (V_i(t) - V_i(0)).

Reads visualization_barriers[_frac]_hu_N<n>/ ; writes into figures/n<N>/ (+ fd_md_figures/
subfolder; cross-N comparisons into figures/comparison/).
Usage: python plot_porepy.py [--nphase 3 4] [--days 0 78 571]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import meshio
import numpy as np

import hamon_2d_solver as H                       # barrier_mask + scheme conventions (no porepy)
import plot_reference as PR                       # EXACT figure conventions (cmap/headers/axes)

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0
LX = LY = 100.0
PHI = 0.3                                          # = porepy_2d_solver porosity
APERTURE = 1.0e-1                                  # = porepy_2d_solver FRACTURE_APERTURE
H_MIN, H_MAX = 1.0, 3.0                            # = porepy_2d_solver enthalpy bounds
RHO_S = 2500.0                                     # = solid_constants density
CP_S = 0.0035                                      # = solver C_P (caloric T = h/C_P AND rock c_p)

_PHASE_FIELDS = {3: ("s_water", "s_oil", "s_gas"),
                 4: ("s_water", "s_oil1", "s_oil2", "s_gas")}

FRAC_LW = 0.25                                     # md overlay: 1D fracture line thickness [pt]
POINT_SIZE = 1.0                                  # md overlay: 0D point diameter [pt]


def _case_dir(n, md):
    return os.path.join(HERE, f"visualization_barriers{'_frac' if md else ''}_hu_N{n}")


_PVD_CACHE: dict[str, list] = {}


def read_pvd(case_dir):
    """Master .pvd -> sorted list of (t_days, {dim: vtu_path}).

    Grid selection is DATA-driven, the ParaView threshold recipe: each file's dimension is
    read from its ``grid_dim`` cell field (verified grid_dim-pure) and interface grids are
    dropped by their ``is_mortar`` field; file names are only addresses.  Cached per dir
    (the classification reads every VTU once)."""
    if case_dir in _PVD_CACHE:
        return _PVD_CACHE[case_dir]
    master = [f for f in os.listdir(case_dir)
              if f.endswith(".pvd") and not re.search(r"_\d{6}\.pvd$", f)][0]
    by_time: dict[float, dict[int, str]] = {}
    for ds in ET.parse(os.path.join(case_dir, master)).getroot().iter("DataSet"):
        path = os.path.join(case_dir, ds.attrib["file"])
        mesh = meshio.read(path)
        mortar = mesh.cell_data.get("is_mortar")
        if mortar is not None and np.any(np.concatenate([np.asarray(b) for b in mortar])):
            continue                                             # interface grids: excluded
        gd = np.unique(np.concatenate([np.asarray(b) for b in mesh.cell_data["grid_dim"]]))
        assert gd.size == 1, f"{path}: mixed grid_dim {gd}"
        files = by_time.setdefault(float(ds.attrib["timestep"]) / DAY, {})
        assert int(gd[0]) not in files, f"{path}: duplicate grid_dim {gd[0]} at one instant"
        files[int(gd[0])] = path
    _PVD_CACHE[case_dir] = sorted(by_time.items())
    return _PVD_CACHE[case_dir]


def _cell_measures(mesh, dim):
    """Per-cell SPECIFIC volume: area (2D quads), length*a (1D lines), a^2 (0D vertices)."""
    pts = mesh.points
    if dim == 2:
        quads = mesh.cells_dict["quad"]
        x, y = pts[:, 0][quads], pts[:, 1][quads]
        # shoelace: abs of the SIGNED sum (per-edge abs would weight area by elevation)
        return 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))
    if dim == 1:
        lines = mesh.cells_dict["line"]
        return np.linalg.norm(pts[lines[:, 1]] - pts[lines[:, 0]], axis=1) * APERTURE
    return np.full(sum(len(b) for b in mesh.cells), APERTURE ** 2)


def _field(mesh, name):
    return np.concatenate([np.asarray(b, float) for b in mesh.cell_data[name]])


def totals(files, n):
    """Volume-AVERAGED mixture mass and energy for one snapshot, as in
    tests/functional/setups/buoyancy_flow_model.py:

        mass   = sum_grids int rho_mix dV / V_tot,        rho_mix = sum_i s_i rho_i,
        energy = sum_grids int [ phi (rho_mix h - p) + (1 - phi) rho_s c_ps T ] dV / V_tot,

    with the ACTUAL specific volumes of every dimension (area, len*a, a^2).  The energy is the
    TOTAL internal energy: fluid phi(rho h - p) (h, p = exported mixture fields) + rock
    (1-phi) rho_s c_ps T, with the solver's solid constants."""
    rho = np.linspace(1500.0, 500.0, n)
    m_int = e_int = v_tot = 0.0
    for dim, path in files.items():
        mesh = meshio.read(path)
        v = _cell_measures(mesh, dim)
        s = [_field(mesh, key) for key in _PHASE_FIELDS[n]]
        rho_mix = sum(rho[i] * s[i] for i in range(n))
        e_fluid = PHI * (rho_mix * _field(mesh, "enthalpy") - _field(mesh, "pressure"))
        e_rock = (1.0 - PHI) * RHO_S * CP_S * _field(mesh, "temperature")
        m_int += float(np.sum(rho_mix * v))
        e_int += float(np.sum((e_fluid + e_rock) * v))
        v_tot += float(v.sum())
    return m_int / v_tot, e_int / v_tot


def phase_volumes(files, n):
    """Phase volumes V_i = int s_i dV for one snapshot (s_i is already intensive; its
    volume integral IS the conserved measure -- no rho, phi, or V_tot factors)."""
    vol = np.zeros(n)
    for dim, path in files.items():
        mesh = meshio.read(path)
        v = _cell_measures(mesh, dim)
        for i, key in enumerate(_PHASE_FIELDS[n]):
            vol[i] += float(np.sum(_field(mesh, key) * v))
    return vol


def saturation_conservation(n, md, out_dir):
    """Per-phase PORE-volume losses |V_i(t) - V_i(0)| / V_pore vs time, with
    V_i = phi int s_i dV and V_pore = phi V_tot (phi is uniform, so it cancels and the
    ratio is the relative loss |int s_i dV| / V_tot).  rho_i is constant, so these
    decompose the mass drift exactly: M(t) - M(0) = sum_i rho_i (V_i(t) - V_i(0))."""
    series = read_pvd(_case_dir(n, md))
    t = np.array([s[0] for s in series])
    S = np.array([phase_volumes(s[1], n) for s in series])      # (nt, n) bulk int s_i dV
    v_tot = float(S[0].sum())                                   # sum_i s_i = 1 -> V_tot at t=0
    loss, t = np.abs(S[1:] - S[0]) / v_tot, t[1:]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, key in enumerate(_PHASE_FIELDS[n]):
        ax.semilogy(t, loss[:, i], "o-", ms=3, label=PR.phase_labels(n)[i])
    for level in (1.0e-3, 1.0e-4):
        ax.axhline(level, color="0.6", lw=1.0, ls="--", zorder=0)
        ax.text(t[-1], level, f" {level:.0e}", color="0.45", fontsize=8, va="bottom")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"pore-volume loss  $|V_i(t)-V_i(0)| / V_{pore}$,"
                  "\n"
                  r"$V_i = \phi\int s_i\,dV$,  $V_{pore} = \phi V_{tot}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    tag = "mixed-dimensional" if md else "fixed-dimensional"
    ax.set_title(f"PorePy HU-BM(mp), N={n}, {tag}: saturation conservation")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"saturation_conservation_pp_hu{'_md' if md else ''}.png"))


CMP_SCHEMES = (("hu-mp", "HU"), ("ppu", "PPU"))    # independent-solver curves: (token, label)


def _hamon_vtr_dir(n):
    return os.path.join(HERE, "vtr" if n == 3 else f"vtr_n{n}")


def _hamon_frac_vtr_dir(n):
    """Directory of the independent solver's EQUI-DIMENSIONAL (fractured) snapshots."""
    return os.path.join(HERE, "vtr_frac" if n == 3 else f"vtr_n{n}_frac")


def _hamon_days(n, scheme):
    """Snapshot days available in the hamon vtr dir for ``scheme`` (sorted)."""
    tag = scheme.replace("-", "_")
    days = []
    for f in glob.glob(os.path.join(_hamon_vtr_dir(n), f"hamon_{tag}_*d.vtr")):
        m = re.search(r"_(\d+)d\.vtr$", f)
        if m:
            days.append(float(m.group(1)))
    return sorted(days)


def _hamon_saturations(n, scheme, day):
    """Per-phase saturation fields (heavy -> light) from one hamon vtr snapshot."""
    _, _, f = PR.load_vtr(PR._vtr_path(_hamon_vtr_dir(n), scheme, day))
    return [np.asarray(f[k], float) for k in PR._phase_fields(f)]


def conservation_comparison(n, out_dir, include_md=True):
    """Relative mass loss: porepy HU vs the independent solver's HU (= HU-BM(mp)) and PPU on
    the LEFT axis; porepy total energy on the right.  fd case (plus the md porepy curves,
    dashed, when ``include_md``); matched snapshot instants."""
    series = read_pvd(_case_dir(n, False))
    t = np.array([s[0] for s in series])
    q = np.array([totals(s[1], n) for s in series])
    rho = np.linspace(1500.0, 500.0, n)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    handles = []
    l, = ax1.semilogy(t[1:], np.abs(q[1:, 0] - q[0, 0]) / abs(q[0, 0]), "o-", color="C0",
                      ms=3, label="Mass HU (PorePy)")
    handles.append(l)
    for (scheme, lab), color in zip(CMP_SCHEMES, ("C2", "C1")):
        days = _hamon_days(n, scheme)
        if len(days) < 2:
            continue
        m = np.array([sum(float(r * s.mean()) for r, s in
                          zip(rho, _hamon_saturations(n, scheme, d))) for d in days])
        l, = ax1.semilogy(np.array(days)[1:], np.abs(m[1:] - m[0]) / abs(m[0]), "s-",
                          color=color, ms=3, label=f"Mass {lab}")
        handles.append(l)
    l2, = ax2.semilogy(t[1:], np.abs(q[1:, 1] - q[0, 1]) / abs(q[0, 1]), "^-", color="C3",
                       ms=3, label="Energy HU (PorePy)")
    handles.append(l2)
    if include_md and os.path.isdir(_case_dir(n, True)):
        series_md = read_pvd(_case_dir(n, True))
        tm = np.array([s[0] for s in series_md])
        qm = np.array([totals(s[1], n) for s in series_md])
        lm, = ax1.semilogy(tm[1:], np.abs(qm[1:, 0] - qm[0, 0]) / abs(qm[0, 0]), ".--",
                           color="C0", ms=4, alpha=0.8, label="Mass HU (PorePy, md)")
        handles.insert(1, lm)
        le, = ax2.semilogy(tm[1:], np.abs(qm[1:, 1] - qm[0, 1]) / abs(qm[0, 1]), ".--",
                           color="C3", ms=4, alpha=0.8, label="Energy HU (PorePy, md)")
        handles.append(le)
    ax1.axhline(1.0e-4, color="C0", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax2.axhline(1.0e-4, color="C3", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel(
        r"$|\langle\rho\rangle(t)-\langle\rho\rangle(0)| / \langle\rho\rangle(0)$,"
        "\n"
        r"$\langle\rho\rangle = \frac{1}{V_{tot}}\int_\Omega \rho\; dV$", color="C0")
    ax2.set_ylabel(
        r"$|\langle E\rangle(t)-\langle E\rangle(0)| / \langle E\rangle(0)$,"
        "\n"
        r"$\langle E\rangle = \frac{1}{V_{tot}}\int_\Omega E\; dV$", color="C3")
    ax1.tick_params(axis="y", colors="C0"); ax2.tick_params(axis="y", colors="C3")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               ncol=min(4, len(handles)), fontsize=9, frameon=True, fancybox=True)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"conservation_comparison_{n}_phases.png"))


def l2_difference(nphases, out_dir):
    """Volume-averaged RMS saturation difference of the independent solver's HU (= HU-BM(mp))
    and PPU w.r.t. the porepy HU solution,

        [ (1/N) sum_i int (s_i^indep - s_i^pp)^2 dV / V_tot ]^{1/2},

    at the matched snapshot instants (t = 0 excluded: identical ICs make it zero).
    One figure covering every phase count in ``nphases`` (N=3 solid, N=4 dashed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    handles = []
    for n, ls in zip(nphases, ("-", "--", ":")):
        if not os.path.isdir(_case_dir(n, False)) or not os.path.isdir(_hamon_vtr_dir(n)):
            continue
        series = {round(s[0]): s[1] for s in read_pvd(_case_dir(n, False))}
        for (scheme, lab), color in zip(CMP_SCHEMES, ("C2", "C1")):
            days = [d for d in _hamon_days(n, scheme) if d > 0 and round(d) in series]
            if not days:
                continue
            vals = []
            for d in days:
                mesh = meshio.read(series[round(d)][2])
                v = _cell_measures(mesh, 2)
                s_pp = [_field(mesh, k) for k in _PHASE_FIELDS[n]]
                s_h = _hamon_saturations(n, scheme, d)
                num = sum(float(np.sum((a - b) ** 2 * v)) for a, b in zip(s_h, s_pp))
                vals.append(np.sqrt(num / (n * float(v.sum()))))
            phase_word = {3: "three-phases", 4: "four-phases"}.get(n, f"{n}-phases")
            st = PR.parse_stats(os.path.join(
                _hamon_vtr_dir(n), f"stats_{scheme.replace('-', '_')}.txt"))
            it_tag = (f", total it. {st['total_newton_iters']}"
                      if "total_newton_iters" in st else "")
            l, = ax.semilogy(days, vals, marker="o", ls=ls, color=color, ms=3,
                             label=f"{lab} ({phase_word}{it_tag})")
            handles.append(l)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\left[\frac{1}{N V_{tot}}\sum_i \int (s_i - s_i^{pp})^2\, dV\right]^{1/2}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9, frameon=True, fancybox=True)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "l2_difference_fd.png"))


def comparison_saturation_maps(n, days, out_dir):
    """Vertical stack of the diverging saturation maps -- rows: PPU (independent),
    HU (PorePy, fd), HU (PorePy, md); columns: the requested days.  The same panels as
    saturation_maps_ppu / _pp_hu / _pp_hu_md, without the (a)/(b)/(c) letters."""
    cm = PR._cmap("vlag")
    comp = _composite_of(n)
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    fig, axes = plt.subplots(3, len(days), figsize=(4.4 * len(days), 4.2 * 3), squeeze=False)
    im = None
    for k, day in enumerate(days):                    # row 0: independent PPU (vtr)
        ax = axes[0][k]
        nx, ny, f = PR.load_vtr(PR._vtr_path(_hamon_vtr_dir(n), "ppu", day))
        im = ax.imshow(PR._composite(f, nx, ny), extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm, vmin=-1.0, vmax=1.0, interpolation="nearest")
        PR._overlay_barriers(ax, f["barrier"], nx, ny)
        PR._style_axes(ax, "")
    for row, md in ((1, False), (2, True)):           # rows 1-2: porepy HU, fd then md
        snaps = dict(read_pvd(_case_dir(n, md)))
        for k, day in enumerate(days):
            ax = axes[row][k]
            t = min(snaps, key=lambda s: abs(s - day))
            mesh = meshio.read(snaps[t][2])
            im = ax.imshow(PR._image(comp(mesh), 100, 100), extent=[0, LX, LY, 0],
                           aspect="equal", cmap=cm, vmin=-1.0, vmax=1.0,
                           interpolation="nearest")
            PR._overlay_barriers(ax, barrier, 100, 100)
            if md:
                _fracture_layer(ax, snaps[t], comp, cm, vmin=-1.0, vmax=1.0)
            PR._style_axes(ax, f"Saturation map at {int(round(t))} days" if row == 2 else "")
    for row, lab in enumerate(("PPU", "HU (PorePy)", "HU (PorePy, md)")):
        axes[row][0].set_ylabel(lab, fontsize=12)
    cbar = fig.colorbar(im, ax=[a for r in axes for a in r], fraction=0.02, pad=0.02,
                        ticks=[-1, 0, 1])
    _label_composite_cbar(cbar, n)
    _save(fig, os.path.join(out_dir, f"comparison_saturation_maps_{n}_phases.png"), dpi=360)


SAVE_PDF = True     # also write a vector PDF next to each PNG (run_workflow toggles this via --pdf)


def _save(fig, png, dpi=180):
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(png, HERE)}" + (" (+ .pdf)" if SAVE_PDF else ""))


def _fracture_layer(ax, files, values_of, cm, vmin=0.0, vmax=1.0):
    """md overlay: the 1D fracture cells as line segments and the 0D intersection points,
    colored by ``values_of(mesh)`` on the SAME scale as the 2D cells (grids selected per
    grid_dim), with a dark outline that keeps the network visible where fracture and matrix
    values coincide."""
    def color(mesh):
        v = np.clip(np.asarray(values_of(mesh), float), vmin, vmax)
        return cm((v - vmin) / (vmax - vmin))
    if 1 in files:
        m1 = meshio.read(files[1])
        segs = m1.points[m1.cells_dict["line"]][:, :, :2].copy()
        segs[:, :, 1] = LY - segs[:, :, 1]                      # depth axis (0 at top)
        ax.add_collection(LineCollection(segs, colors="0.15", linewidths=FRAC_LW + 1.2,
                                         capstyle="projecting", zorder=4))
        ax.add_collection(LineCollection(segs, colors=color(m1), linewidths=FRAC_LW,
                                         capstyle="projecting", zorder=5))
    if 0 in files:
        m0 = meshio.read(files[0])
        ax.scatter(m0.points[:, 0], LY - m0.points[:, 1], c=color(m0),
                   s=POINT_SIZE ** 2, edgecolors="0.15", linewidths=0.4, zorder=6)


def _composite_of(n):
    """Callable mesh -> diverging composite sum_k c_k s_k, c = linspace(-1, +1, N)
    (= plot_reference._composite: heaviest -> -1 blue, lightest -> +1 red)."""
    c = np.linspace(-1.0, 1.0, n)
    return lambda mesh: sum(ck * _field(mesh, key) for ck, key in zip(c, _PHASE_FIELDS[n]))


def _overlay_fractures(ax, frac1d, nx, ny):
    """Overlay the equi-dimensional fracture-band cells (teal) -- the standalone solver's
    resolved high-k fractures, distinct from the grey barriers."""
    F = PR._image(frac1d, nx, ny) > 0.5
    rgba = np.zeros(F.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.0, 0.5, 0.5      # teal
    rgba[..., 3] = np.where(F, 0.9, 0.0)
    ax.imshow(rgba, extent=[0, LX, LY, 0], aspect="equal", interpolation="nearest")


def _overlay_fracture_network(ax):
    """Draw the conformal fractures as dark line segments (from H._FRACTURES_REF), matching the md
    row's network rendering -- used on the equi-dim row, where the resampled 0.1 m band is sub-grid."""
    segs = [[(x0, LY - y0), (x1, LY - y1)] for (x0, y0, x1, y1) in H._FRACTURES_REF]   # elev -> depth
    ax.add_collection(LineCollection(segs, colors="0.15", linewidths=FRAC_LW + 1.2,
                                     capstyle="projecting", zorder=5))


def _hamon_it(vtr_dir, scheme="ppu"):
    """Total accepted Newton iterations of an independent-solver run, or None if unavailable."""
    st = PR.parse_stats(os.path.join(vtr_dir, f"stats_{scheme.replace('-', '_')}.txt"))
    return st.get("total_newton_iters")


def _pp_it(case_dir):
    """Total accepted Newton iterations of a PorePy run (run_statistics.txt), or None."""
    try:
        txt = open(os.path.join(case_dir, "run_statistics.txt")).read()
        return int(re.search(r"total Newton iterations \(accepted\): (\d+)", txt).group(1))
    except Exception:
        return None


def _it_label(base, it):
    return base if it is None else f"{base}\n({it} iterations)"


def _label_composite_cbar(cbar, n):
    """Make the diverging colorbar unambiguous: the first two rows show a SATURATION composite
    sum_k c_k s_k (c_k = density rank in [-1, +1]), NOT densities. Blue end = heaviest phase
    dominant, red end = lightest phase dominant. Ticks name the phases (heavy -> light); no bare
    density numbers (those were the source of the 'saturations or densities?' confusion)."""
    labs = PR.phase_labels(n)                       # heavy -> light: [s_gamma, ..., s_alpha]
    cbar.ax.set_yticklabels([labs[0] + "\n(heaviest phase)",
                             ", ".join(labs[1:-1]),
                             labs[-1] + "\n(lightest phase)"])
    cbar.set_label(r"saturation composite $\sum_k c_k\, s_k$", fontsize=10)


def comparison_saturation_maps_fixed_dim(n, days, out_dir):
    """Fixed-dimensional maps (columns = days): the MIXTURE DENSITY rho = sum_k s_k rho_k [kg/m^3]
    for the independent PPU (row 0) and the PorePy HU (row 1), and their absolute difference
    |rho_PPU - rho_HU| (row 2), on the unfractured domain. The first two row labels report each
    run's total Newton iterations."""
    cm = PR._cmap("vlag")
    cm_rho = cm.reversed()                                # heavy (high rho) -> blue, light -> red
    rho = np.linspace(1500.0, 500.0, n)                  # phase densities, heavy -> light [kg/m^3]
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    fig, axes = plt.subplots(3, len(days), figsize=(4.4 * len(days), 4.2 * 3), squeeze=False)
    rho_ppu, rho_hu = [], []                              # per-day mixture-density maps [kg/m^3]
    im = None
    for k, day in enumerate(days):                        # row 0: independent PPU (vtr)
        ax = axes[0][k]
        nx, ny, f = PR.load_vtr(PR._vtr_path(_hamon_vtr_dir(n), "ppu", day))
        s = [PR._image(np.asarray(f[key], float), nx, ny) for key in PR._phase_fields(f)]
        rho_ppu.append(sum(rho[j] * s[j] for j in range(n)))     # rho = sum_k s_k rho_k
        im = ax.imshow(rho_ppu[-1], extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm_rho, vmin=500.0, vmax=1500.0, interpolation="nearest")
        PR._overlay_barriers(ax, f["barrier"], nx, ny)
        PR._style_axes(ax, "")
    snaps = dict(read_pvd(_case_dir(n, False)))           # row 1: porepy HU (fixed-dim)
    for k, day in enumerate(days):
        ax = axes[1][k]
        t = min(snaps, key=lambda s: abs(s - day))
        mesh = meshio.read(snaps[t][2])
        s = [PR._image(_field(mesh, key), 100, 100) for key in _PHASE_FIELDS[n]]   # heavy->light
        rho_hu.append(sum(rho[j] * s[j] for j in range(n)))
        im = ax.imshow(rho_hu[-1], extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm_rho, vmin=500.0, vmax=1500.0, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        PR._style_axes(ax, "")
    # row 2: |rho_PPU - rho_HU| / ||rho_PPU|| -- mixture-density difference RELATIVE to the
    # VOLUME-AVERAGED L2 norm ||rho_PPU|| = sqrt(int rho^2 dV / V_tot) = RMS(rho) (a scalar per
    # snapshot, O(the density scale ~1000), not sqrt(sum) ~ 1e5). Uniform grid -> V_c cancels.
    diffs = [np.abs(rho_ppu[k] - rho_hu[k]) / np.sqrt(np.mean(rho_ppu[k] ** 2))
             for k in range(len(days))]
    dmax = max((float(d.max()) for d in diffs), default=1.0) or 1.0
    dim = None
    for k, day in enumerate(days):
        ax = axes[2][k]
        dim = ax.imshow(diffs[k], extent=[0, LX, LY, 0], aspect="equal",
                        cmap=cm, vmin=0.0, vmax=dmax, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        PR._style_axes(ax, f"$t = {int(round(day))}$ days")
    axes[0][0].set_ylabel(_it_label("PPU (independent)", _hamon_it(_hamon_vtr_dir(n))), fontsize=12)
    axes[1][0].set_ylabel(_it_label("HU (PorePy)", _pp_it(_case_dir(n, False))), fontsize=12)
    axes[2][0].set_ylabel(
        r"$|\rho_{\mathrm{PPU}}-\rho_{\mathrm{HU}\,(\mathrm{PorePy})}| \,/\, \|\rho_{\mathrm{PPU}}\|$",
        fontsize=12)
    cbar = fig.colorbar(im, ax=[axes[r][j] for r in (0, 1) for j in range(len(days))],
                        fraction=0.02, pad=0.02, ticks=[500, 1000, 1500])
    cbar.set_label(r"mixture density $\rho=\sum_k s_k\,\rho_k$  [kg m$^{-3}$]", fontsize=10)
    dcbar = fig.colorbar(dim, ax=[axes[2][j] for j in range(len(days))], fraction=0.02, pad=0.02)
    dcbar.set_label(
        r"$|\rho_{\mathrm{PPU}}-\rho_{\mathrm{HU}\,(\mathrm{PorePy})}| \,/\, \|\rho_{\mathrm{PPU}}\|$",
        fontsize=9)
    _save(fig, os.path.join(out_dir, f"comparison_saturation_maps_fixed_dim_{n}_phases.png"), dpi=360)


def _equidim_density_uniform(n, day, rho, nu=100):
    """Load the equi-dim (graded) PPU snapshot and resample its mixture density + fracture mask
    onto a uniform ``nu x nu`` grid, position-aware via the graded edges (piecewise-constant).
    Returns ``(rho_img, frac_img)`` as top-oriented (nu, nu) images, aligned with the 100x100 maps."""
    import pyvista as pv
    m = pv.read(PR._vtr_path(_hamon_frac_vtr_dir(n), "ppu", day))
    xe, ye = np.asarray(m.x, float), np.asarray(m.y, float)
    nx, ny = xe.size - 1, ye.size - 1
    cell = lambda key: np.asarray(m.cell_data[key], float).reshape(ny, nx)   # (j,i), j=0 bottom
    rho_g = sum(rho[k] * cell(f"s_{k}") for k in range(n))
    frac_g = cell("fracture")
    xu, yu = (np.arange(nu) + 0.5) * (LX / nu), (np.arange(nu) + 0.5) * (LY / nu)
    gi = np.clip(np.searchsorted(xe, xu) - 1, 0, nx - 1)   # graded col covering each uniform col
    gj = np.clip(np.searchsorted(ye, yu) - 1, 0, ny - 1)
    idx = np.ix_(gj, gi)
    return np.flipud(rho_g[idx]), np.flipud(frac_g[idx])   # flip: j=0 bottom -> row 0 top


def comparison_saturation_maps_mixed_dim(n, days, out_dir):
    """Mixed-dimensional maps (columns = days): the independent PPU with an EQUI-DIMENSIONAL fracture
    representation (row 0; fractures as refined high-k bands, resampled onto the 100x100 grid, teal
    overlay), the PorePy HU mixed-dimensional computation (row 1; conformal 1D fractures overlaid),
    both as MIXTURE DENSITY rho = sum_k s_k rho_k [kg/m^3], and their relative difference
    |rho_PPU - rho_HU|/||rho_PPU|| (row 2). The first two row labels report total Newton iterations."""
    cm = PR._cmap("vlag")
    cm_rho = cm.reversed()                                # heavy (high rho) -> blue, light -> red
    rho = np.linspace(1500.0, 500.0, n)
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    frac_dir = _hamon_frac_vtr_dir(n)
    fig, axes = plt.subplots(3, len(days), figsize=(4.4 * len(days), 4.2 * 3), squeeze=False)
    rho_ppu, rho_hu, frac_ppu = [], [], []
    im = None
    for k, day in enumerate(days):                        # row 0: independent PPU, equi-dimensional
        ax = axes[0][k]
        r, _ = _equidim_density_uniform(n, day, rho)      # resampled to 100x100
        rho_ppu.append(r)
        im = ax.imshow(r, extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm_rho, vmin=500.0, vmax=1500.0, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        _overlay_fracture_network(ax)                     # fracture network as dark lines (like md)
        PR._style_axes(ax, "")
    snaps = dict(read_pvd(_case_dir(n, True)))            # row 1: porepy HU (mixed-dimensional)
    comp = _composite_of(n)
    for k, day in enumerate(days):
        ax = axes[1][k]
        t = min(snaps, key=lambda s: abs(s - day))
        mesh = meshio.read(snaps[t][2])
        s = [PR._image(_field(mesh, key), 100, 100) for key in _PHASE_FIELDS[n]]
        rho_hu.append(sum(rho[j] * s[j] for j in range(n)))
        im = ax.imshow(rho_hu[-1], extent=[0, LX, LY, 0], aspect="equal",
                       cmap=cm_rho, vmin=500.0, vmax=1500.0, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        _fracture_layer(ax, snaps[t], comp, cm, vmin=-1.0, vmax=1.0)   # 1D fracture lines
        PR._style_axes(ax, "")
    diffs = [np.abs(rho_ppu[k] - rho_hu[k]) / np.sqrt(np.mean(rho_ppu[k] ** 2))
             for k in range(len(days))]
    dmax = max((float(d.max()) for d in diffs), default=1.0) or 1.0
    dim = None
    for k, day in enumerate(days):
        ax = axes[2][k]
        dim = ax.imshow(diffs[k], extent=[0, LX, LY, 0], aspect="equal",
                        cmap=cm, vmin=0.0, vmax=dmax, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        PR._style_axes(ax, f"$t = {int(round(day))}$ days")
    axes[0][0].set_ylabel(_it_label("PPU (equi-dim.)", _hamon_it(frac_dir)), fontsize=12)
    axes[1][0].set_ylabel(_it_label("HU (PorePy, md)", _pp_it(_case_dir(n, True))), fontsize=12)
    axes[2][0].set_ylabel(
        r"$|\rho_{\mathrm{PPU}}-\rho_{\mathrm{HU}\,(\mathrm{PorePy})}| \,/\, \|\rho_{\mathrm{PPU}}\|$",
        fontsize=12)
    cbar = fig.colorbar(im, ax=[axes[r][j] for r in (0, 1) for j in range(len(days))],
                        fraction=0.02, pad=0.02, ticks=[500, 1000, 1500])
    cbar.set_label(r"mixture density $\rho=\sum_k s_k\,\rho_k$  [kg m$^{-3}$]", fontsize=10)
    dcbar = fig.colorbar(dim, ax=[axes[2][j] for j in range(len(days))], fraction=0.02, pad=0.02)
    dcbar.set_label(
        r"$|\rho_{\mathrm{PPU}}-\rho_{\mathrm{HU}\,(\mathrm{PorePy})}| \,/\, \|\rho_{\mathrm{PPU}}\|$",
        fontsize=9)
    _save(fig, os.path.join(out_dir, f"comparison_saturation_maps_mixed_dim_{n}_phases.png"), dpi=360)


def enthalpy_temperature_maps_mixed_dim(n, days, out_dir):
    """Enthalpy [MJ/kg] (row 0) and temperature [degC] (row 1) of the PorePy HU MIXED-DIMENSIONAL
    solution; columns = the requested days. Barriers overlaid (grey) and the conformal 1D fractures
    coloured on the SAME scale as the matrix. Each row shares one colour scale across the columns
    (matrix + 1D fracture cells). ``enthalpy`` is already MJ/kg and ``T_C`` already degC in the VTU."""
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    snaps = dict(read_pvd(_case_dir(n, True)))
    picks = [min(snaps, key=lambda s: abs(s - d)) for d in days]
    meshes = [meshio.read(snaps[t][2]) for t in picks]
    cm = PR._cmap("vlag").reversed()                    # SAME map as the density maps (high -> blue)
    rows = (("enthalpy", cm, r"enthalpy [MJ kg$^{-1}$]"),
            ("T_C",      cm, r"temperature [$^{\circ}$C]"))
    # one colour scale per row, over matrix AND 1D fracture cells across every column
    scales = {}
    for key, _, _ in rows:
        vals = [_field(m, key) for m in meshes]
        vals += [_field(meshio.read(snaps[t][1]), key) for t in picks if 1 in snaps[t]]
        allv = np.concatenate(vals)
        scales[key] = (float(allv.min()), float(allv.max()))
    fig, axes = plt.subplots(2, len(days), figsize=(4.4 * len(days), 4.2 * 2), squeeze=False)
    ims = [None, None]
    for r, (key, cm, _) in enumerate(rows):
        vmin, vmax = scales[key]
        for k, (t, m) in enumerate(zip(picks, meshes)):
            ax = axes[r][k]
            ims[r] = ax.imshow(PR._image(_field(m, key), 100, 100), extent=[0, LX, LY, 0],
                               aspect="equal", cmap=cm, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
            PR._overlay_barriers(ax, barrier, 100, 100)
            _fracture_layer(ax, snaps[t], lambda mm, kk=key: _field(mm, kk),
                            cm, vmin=vmin, vmax=vmax)
            PR._style_axes(ax, f"$t = {int(round(t))}$ days" if r == 1 else "")
    for r, (_, _, label) in enumerate(rows):
        axes[r][0].set_ylabel(label, fontsize=12)
        cb = fig.colorbar(ims[r], ax=[axes[r][j] for j in range(len(days))],
                          fraction=0.02, pad=0.02)
        cb.set_label(label, fontsize=10)
    _save(fig, os.path.join(out_dir, f"enthalpy_temperature_mixed_dim_{n}_phases.png"), dpi=360)


def _pp_stats_line(case_dir):
    """The plot_maps stats line, parsed from the solver's run_statistics.txt."""
    try:
        txt = open(os.path.join(case_dir, "run_statistics.txt")).read()
        steps = int(re.search(r"accepted steps: (\d+)", txt).group(1))
        its = int(re.search(r"total Newton iterations \(accepted\): (\d+)", txt).group(1))
        cuts = int(re.search(r"rejected/cut loops: (\d+)", txt).group(1))
        return (f"total iterations: {its}     avg iterations/step: {its / steps:.2f}     "
                f"dt-cuts: {cuts}")
    except Exception:
        return ""


def saturation_maps(n, md, days, out_dir):
    """EXACTLY plot_reference.plot_maps: one row of diverging composite maps
    (sum_k c_k s_k in [-1, 1], heavy blue -> light red) at the requested days, barrier
    overlay, per-panel captions and the run-statistics line; md adds the fracture network
    colored by the same composite."""
    snaps = dict(read_pvd(_case_dir(n, md)))
    cm = PR._cmap("vlag")
    comp = _composite_of(n)
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    fig, axes = plt.subplots(1, len(days), figsize=(4.4 * len(days), 4.6))
    axes = [axes] if len(days) == 1 else list(axes)
    im = None
    for k, (ax, day) in enumerate(zip(axes, days)):
        t = min(snaps, key=lambda s: abs(s - day))
        mesh = meshio.read(snaps[t][2])
        im = ax.imshow(PR._image(comp(mesh), 100, 100), extent=[0, LX, LY, 0],
                       aspect="equal", cmap=cm, vmin=-1.0, vmax=1.0, interpolation="nearest")
        PR._overlay_barriers(ax, barrier, 100, 100)
        if md:
            _fracture_layer(ax, snaps[t], comp, cm, vmin=-1.0, vmax=1.0)
        PR._style_axes(ax, f"({PR._ABC[k]}) Saturation map at {int(round(t))} days")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
    _label_composite_cbar(cbar, n)
    stat_line = _pp_stats_line(_case_dir(n, md))
    if stat_line:
        fig.text(0.5, 0.935, stat_line, ha="center", va="top", fontsize=10, color="0.35")
    _save(fig, os.path.join(out_dir, f"saturation_maps_pp_hu{'_md' if md else ''}.png"),
          dpi=360)


def saturation_grid(n, md, days, out_dir):
    """EXACTLY plot_reference.plot_grid's layout (vlag cmap, density headers, barrier overlay,
    0..100 axes), fed from the porepy VTUs; md adds the fracture network as a
    saturation-colored layer (:func:`_fracture_layer`) on top of the 2D cells."""
    snaps = dict(read_pvd(_case_dir(n, md)))
    keys = _PHASE_FIELDS[n]
    headers = PR._phase_header(n)
    cm = PR._cmap("vlag")
    barrier = np.asarray(H.barrier_mask(100, 100), float)
    fig, axes = plt.subplots(len(days), n, figsize=(3.1 * n, 3.3 * len(days)), squeeze=False)
    im = None
    for i, day in enumerate(days):
        t = min(snaps, key=lambda s: abs(s - day))
        mesh = meshio.read(snaps[t][2])
        for j, key in enumerate(keys):
            ax = axes[i][j]
            im = ax.imshow(PR._image(_field(mesh, key), 100, 100), extent=[0, LX, LY, 0],
                           aspect="equal", cmap=cm, vmin=0.0, vmax=1.0, interpolation="nearest")
            PR._overlay_barriers(ax, barrier, 100, 100)
            if md:
                _fracture_layer(ax, snaps[t], lambda m, k=key: _field(m, k), cm)
            PR._style_axes(ax, "")
            if i == 0:
                ax.set_title(headers[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{int(round(t))} days", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, ticks=[0, 0.5, 1])
    cbar.set_label("phase saturation $s_k$")
    _save(fig, os.path.join(out_dir, f"saturation_grid_pp_hu{'_md' if md else ''}.png"),
          dpi=360)                                    # 2x the default raster resolution


def conservation(n, md, out_dir):
    """Mass/energy losses |q(t) - q(0)| of the volume-averaged quantities (test-style,
    absolute, NOT relative): mass on the left axis, energy on the right, gray order
    lines at 1e-3 and 1e-4 marking the expected mass-preservation decades."""
    series = read_pvd(_case_dir(n, md))
    t = np.array([s[0] for s in series])
    q = np.array([totals(s[1], n) for s in series])             # (nt, 2): mass, energy
    # RELATIVE losses (dimensionless), both referenced to the INITIAL state (the t=0 point
    # itself is zero by definition and dropped)
    loss_m, t_m = np.abs(q[1:, 0] - q[0, 0]) / abs(q[0, 0]), t[1:]
    loss_e, t_e = np.abs(q[1:, 1] - q[0, 1]) / abs(q[0, 1]), t[1:]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    l1, = ax1.semilogy(t_m, loss_m, "o-", color="C0", ms=3, label="mass")
    l2, = ax2.semilogy(t_e, loss_e, "s-", color="C3", ms=3, label="energy")
    # order line at 1e-4 on EACH axis, colored like its curve (left/mass blue, right/energy red)
    ax1.axhline(1.0e-4, color="C0", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax1.text(t_m[-1], 1.0e-4, " 1e-04", color="C0", fontsize=8, va="bottom")
    ax2.axhline(1.0e-4, color="C3", lw=1.0, ls="--", alpha=0.6, zorder=0)
    ax2.text(t_m[0], 1.0e-4, " 1e-04", color="C3", fontsize=8, va="bottom", ha="left")
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel(r"relative mass loss  "
                   r"$|\langle\rho\rangle(t)-\langle\rho\rangle(0)| / \langle\rho\rangle(0)$",
                   color="C0")
    ax2.set_ylabel(r"relative total-energy loss  "
                   r"$|\langle E \rangle(t) - \langle E \rangle(0)| / \langle E \rangle(0)$,"
                   "\n"
                   r"$E = \phi(\rho h - p) + (1{-}\phi)\rho_s c_{ps} T$", color="C3")
    ax1.tick_params(axis="y", colors="C0"); ax2.tick_params(axis="y", colors="C3")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)
    tag = "mixed-dimensional" if md else "fixed-dimensional"
    ax1.set_title(f"PorePy HU-BM(mp), N={n}, {tag}: conservation vs time")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"conservation_pp_hu{'_md' if md else ''}.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nphase", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--days", type=float, nargs="+", default=[0.0, 78.0, 571.0])
    args = ap.parse_args(argv)
    for n in args.nphase:
        out_dir = os.path.join(HERE, "figures", f"n{n}")
        os.makedirs(out_dir, exist_ok=True)
        for md in (False, True):
            if not os.path.isdir(_case_dir(n, md)):
                print(f"  (skip N={n} md={md}: no output dir)")
                continue
            saturation_maps(n, md, args.days, out_dir)
            saturation_grid(n, md, args.days, out_dir)
            conservation(n, md, out_dir)
            saturation_conservation(n, md, out_dir)
        if os.path.isdir(_hamon_vtr_dir(n)):          # porepy-vs-independent comparisons
            conservation_comparison(n, out_dir)
            if os.path.isdir(_case_dir(n, False)):    # FIXED-dimensional: independent PPU + PorePy HU
                comparison_saturation_maps_fixed_dim(n, args.days, out_dir)
            if os.path.isdir(_case_dir(n, True)) and os.path.isdir(_hamon_frac_vtr_dir(n)):
                comparison_saturation_maps_mixed_dim(n, args.days, out_dir)  # MIXED-dim: equi-dim + md
    cmp_dir = os.path.join(HERE, "figures", "comparison")       # cross-N figures
    os.makedirs(cmp_dir, exist_ok=True)
    l2_difference(args.nphase, cmp_dir)


if __name__ == "__main__":
    main()
