"""Canonical output tag for one porepy_2d_solver parametrization.

Shared by the solver (output folder ``visualization_<tag>/``) and the figure script
(input folder + ``fig_8_plume_<tag>.png``), so distinct parametrizations never
overwrite each other.  Components appear only when they differ from the defaults:

    <scheme>[_mpfa][_<grid_type>][_cs<cell_size>][_q<q_anomaly>][_z<z_init>]
            [_dt<dt_nominal>][_dtmin<dt_min>][_dtmax<dt_max>][_tf<final_years>]
    e.g.  hu  |  hu_mw_mpfa  |  hu_simplex_cs200  |  hu_q10_dt5  |  hu_z0.1_tf20000

The intermediate snapshot instants are deliberately NOT part of the tag -- only the
final time is; runs differing only in intermediate snapshots share a folder.
"""
from __future__ import annotations


def case_tag(scheme: str, consistent: bool = False, grid_type: str | None = None,
             cell_size: float | None = None, q_anomaly: float | None = None,
             z_init: float | None = None, dt_nominal: float | None = None,
             dt_min: float | None = None, dt_max: float | None = None,
             tf_years: float | None = None) -> str:
    parts = [scheme.replace("-", "_")]
    if consistent:
        parts.append("mpfa")
    if grid_type not in (None, "cartesian"):
        parts.append(grid_type)
    if cell_size is not None:
        parts.append(f"cs{cell_size:g}")
    if q_anomaly not in (None, 5.0):
        parts.append(f"q{q_anomaly:g}")
    if z_init not in (None, 0.0):
        parts.append(f"z{z_init:g}")
    if dt_nominal not in (None, 5.0):
        parts.append(f"dt{dt_nominal:g}")
    if dt_min not in (None, 0.01):
        parts.append(f"dtmin{dt_min:g}")
    if dt_max not in (None, 25.0):
        parts.append(f"dtmax{dt_max:g}")
    if tf_years not in (None, 50000.0):
        parts.append(f"tf{tf_years:g}")
    return "_".join(parts)
