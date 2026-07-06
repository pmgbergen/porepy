"""WA-HU Fig. 5 benchmark: barrier-cell geometry + saturation maps.

Reference: S.B.M. Bosma, F.P. Hamon, B.T. Mallison & H.A. Tchelepi, "Smooth implicit hybrid
upwinding for compositional multiphase flow in porous media", Computer Methods in Applied
Mechanics and Engineering 388 (2022) 114288 -- Example 6.3, "Immiscible three-phase gravity
segregation through barriers" (Fig. 5).

Grid / convention
-----------------
100 x 100 Cartesian cells, 1 m each -> a 100 m x 100 m vertical box. Figure convention used
throughout this module: ``row 0 = TOP`` and gravity points downward (depth increases with the
row index). PorePy's Cartesian ``y`` increases UPWARD, so a PorePy row ``prow`` maps to a
figure row ``(n - 1) - prow`` (see :func:`barrier_mask_porepy`).

Barriers
--------
Seven impermeable horizontal layers, RE-EXTRACTED from Fig. 5(a) (``fig5_raw_a.png``) by
calibrating the plot box off the blue (top, depth 0) and red (bottom, depth 100) bands and
thresholding the dark barrier lines. ``BARRIER_LAYERS_FIG`` maps each barrier's figure
depth-row to its INCLUSIVE ``(col_start, col_end)`` filled (impermeable) spans; columns not
covered by any span are the openings.

    NOTE: the depth-23 layer (``23: [(18, 44), (62, 84)]``) was ABSENT from the earlier
    digitization -- the raw figure clearly shows it (two segments) in panels (a) and (c). It
    is restored here, giving the full seven layers.

Saturations
-----------
* t = 0 (initial): analytic -- top 10 % = water (heavy, rho = 1500), bottom 10 % = gas
  (light, rho = 500), middle 80 % = oil (intermediate, rho = 1000).
* t = 78 d and t = 571 d (reference): digitized from Fig. 5(b, c) and stored per cell as
  ``s_water`` / ``s_oil`` / ``s_gas`` in ``fig5_ref_t{1,2}.vtu`` (loaded on request).
"""
from __future__ import annotations

import os

import numpy as np

NX = NY = 100                      # cells per side (1 m each) -> 100 m x 100 m box
_HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------
#  Barrier geometry.  key = figure depth-row (0 = top);  value = inclusive impermeable spans.
# ------------------------------------------------------------------------------------------
# Re-extracted at PIXEL resolution (2024 pass) so every opening is resolved -- the earlier
# read over-merged segments. Subregion (segment) counts, bottom -> top: 3, 6, 4, 2, 5, 2, 5.
BARRIER_LAYERS_FIG: dict[int, list[tuple[int, int]]] = {
    16: [(5, 19), (23, 25), (40, 59), (70, 79), (82, 99)],                     # 5 subregions
    23: [(18, 44), (62, 84)],                                                  # 2  (restored)
    38: [(0, 9), (18, 25), (38, 49), (55, 74), (90, 94)],                      # 5
    45: [(23, 59), (63, 70)],                                                  # 2
    58: [(2, 17), (22, 29), (48, 59), (70, 99)],                              # 4
    74: [(0, 15), (19, 22), (24, 53), (58, 70), (75, 76), (84, 92)],           # 6
    82: [(5, 18), (24, 42), (58, 94)],                                        # 3  (bottom)
}


def barrier_mask_figure(nx: int = NX, ny: int = NY) -> np.ndarray:
    """``(ny, nx)`` boolean mask in FIGURE orientation (row 0 = top). True = barrier cell.

    Layers are scaled if the grid is not exactly 100 cells per side.
    """
    scale = ny / 100.0
    m = np.zeros((ny, nx), dtype=bool)
    for fig_row, spans in BARRIER_LAYERS_FIG.items():
        r = int(round(fig_row * scale))
        if not 0 <= r < ny:
            continue
        for a, b in spans:
            m[r, int(round(a * scale)):int(round(b * scale)) + 1] = True
    return m


def barrier_mask_porepy(nx: int = NX, ny: int = NY) -> np.ndarray:
    """Same mask flipped to PorePy row order (row 0 = BOTTOM), as a flat ``(nx*ny,)`` vector
    in PorePy Cartesian cell order (``cell = col + prow * nx``). Directly comparable to
    ``GeometryBarriers2D.barrier_cell_mask``.
    """
    fig = barrier_mask_figure(nx, ny)             # row 0 = top
    prp = fig[::-1, :]                            # row 0 = bottom
    return prp.reshape(-1)


# ------------------------------------------------------------------------------------------
#  Saturation maps.
# ------------------------------------------------------------------------------------------
def initial_saturations(nx: int = NX, ny: int = NY) -> dict[str, np.ndarray]:
    """t = 0 initial condition, each ``(ny, nx)`` in FIGURE orientation (row 0 = top).

    Top 10 % water (heavy), bottom 10 % gas (light), middle 80 % oil (intermediate).
    """
    sw = np.zeros((ny, nx))
    so = np.zeros((ny, nx))
    sg = np.zeros((ny, nx))
    top = int(round(0.10 * ny))
    bot = int(round(0.90 * ny))
    sw[:top, :] = 1.0
    sg[bot:, :] = 1.0
    so[top:bot, :] = 1.0
    return {"s_water": sw, "s_oil": so, "s_gas": sg}


def reference_saturations(which: str, nx: int = NX, ny: int = NY) -> dict[str, np.ndarray]:
    """Digitized reference maps from the ``.vtu`` files, each ``(ny, nx)`` in FIGURE
    orientation (row 0 = top).

    Parameters:
        which: ``"t0"`` (0 d), ``"t1"`` (78 d) or ``"t2"`` (571 d).

    Returns dict with ``s_water`` / ``s_oil`` / ``s_gas`` (and ``barrier`` / ``dominant_phase``
    if present). Requires the ``vtk`` package.
    """
    import vtk  # noqa: PLC0415
    from vtk.util.numpy_support import vtk_to_numpy  # noqa: PLC0415

    path = os.path.join(_HERE, f"fig5_ref_{which}.vtu")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    ug = reader.GetOutput()
    n = ug.GetNumberOfCells()
    ctr = np.array([ug.GetCell(i).GetBounds() for i in range(n)])
    cx = 0.5 * (ctr[:, 0] + ctr[:, 1])
    cy = 0.5 * (ctr[:, 2] + ctr[:, 3])
    col = np.clip(np.floor(cx).astype(int), 0, nx - 1)
    row = np.clip(ny - 1 - np.floor(cy).astype(int), 0, ny - 1)  # y ascending -> row0 = top
    cd = ug.GetCellData()
    out: dict[str, np.ndarray] = {}
    for k in range(cd.GetNumberOfArrays()):
        name = cd.GetArrayName(k)
        g = np.full((ny, nx), np.nan)
        g[row, col] = vtk_to_numpy(cd.GetArray(k))
        out[name] = g
    return out


def write_barrier_to_vtu(which_list=("t0", "t1", "t2")) -> None:
    """Overwrite the ``barrier`` cell-data array in each ``fig5_ref_{which}.vtu`` with the
    corrected 7-layer mask, leaving the ``s_water`` / ``s_oil`` / ``s_gas`` / ``dominant_phase``
    arrays untouched. The barrier geometry is static, so the same mask is written to every
    snapshot. Requires the ``vtk`` package.
    """
    import vtk  # noqa: PLC0415
    from vtk.util.numpy_support import numpy_to_vtk  # noqa: PLC0415

    mask = barrier_mask_figure()                      # (ny, nx), row 0 = top
    for which in which_list:
        path = os.path.join(_HERE, f"fig5_ref_{which}.vtu")
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        ug = reader.GetOutput()
        n = ug.GetNumberOfCells()
        ctr = np.array([ug.GetCell(i).GetBounds() for i in range(n)])
        cx = 0.5 * (ctr[:, 0] + ctr[:, 1])
        cy = 0.5 * (ctr[:, 2] + ctr[:, 3])
        col = np.clip(np.floor(cx).astype(int), 0, mask.shape[1] - 1)
        row = np.clip(mask.shape[0] - 1 - np.floor(cy).astype(int), 0, mask.shape[0] - 1)
        bar = mask[row, col].astype(np.float64)       # vtu cell order; y ascending -> flip
        arr = numpy_to_vtk(bar, deep=1)
        arr.SetName("barrier")
        cd = ug.GetCellData()
        cd.RemoveArray("barrier")
        cd.AddArray(arr)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(path)
        writer.SetInputData(ug)
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib()
        writer.Write()
        print(f"  updated {os.path.basename(path)}: barrier cells = {int(bar.sum())}")


# ------------------------------------------------------------------------------------------
#  Verification: render barriers + saturation maps and compare vs the digitized reference.
# ------------------------------------------------------------------------------------------
def _rgb(sw, so, sg, barrier=None):
    """water -> blue, gas -> red, oil -> white; barrier cells -> black."""
    sw, so, sg = (np.nan_to_num(x) for x in (sw, so, sg))
    rgb = np.stack([1 - sw, 1 - sw - sg, 1 - sg], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    if barrier is not None:
        rgb[np.nan_to_num(barrier) > 0.5] = (0, 0, 0)
    return rgb


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bar = barrier_mask_figure()
    counts = {r: len(s) for r, s in sorted(BARRIER_LAYERS_FIG.items())}
    print(f"barriers: {len(BARRIER_LAYERS_FIG)} layers, {int(bar.sum())} cells")
    print(f"  subregions per layer (top->bottom): {counts}")
    print(f"  subregions bottom->top: {[len(BARRIER_LAYERS_FIG[r]) for r in sorted(BARRIER_LAYERS_FIG, reverse=True)]}")

    # rewrite the barrier array in the reference vtu files (saturations untouched)
    try:
        print("updating reference vtu files:")
        write_barrier_to_vtu()
    except Exception as exc:
        print(f"  [skip vtu update] {exc}")

    ini = initial_saturations()

    panels = [("0 d (analytic init)", ini["s_water"], ini["s_oil"], ini["s_gas"], bar)]
    for which, label in (("t1", "78 d (ref)"), ("t2", "571 d (ref)")):
        try:
            r = reference_saturations(which)
            panels.append((label, r["s_water"], r["s_oil"], r["s_gas"], r.get("barrier")))
        except Exception as exc:  # vtk missing or file absent
            print(f"  [skip {which}] {exc}")

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (label, sw, so, sg, b) in zip(axes, panels):
        ax.imshow(_rgb(sw, so, sg, b), origin="upper", extent=[0, 100, 100, 0])
        ax.set_title(label)
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("depth [m]")
    fig.suptitle("WA-HU Fig. 5 -- barriers (black) + saturations (blue=water, red=gas, white=oil)")
    fig.tight_layout()
    out = os.path.join(_HERE, "fig5_reconstructed.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
