"""Save and load the per-level results of the manufactured-solution test as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import porepy as pp


def subdomain_geometry(mdg: pp.MixedDimensionalGrid) -> dict:
    """Matrix cell polygons and fracture cell centres, for plotting fields.

    Parameters:
        mdg: Mixed-dimensional grid.

    Returns:
        ``{"matrix": {"polygons", "centers"}, "fracture": {"centers"}}``.

    """

    geometry = {}
    for sd in mdg.subdomains():
        if sd.dim == mdg.dim_max():
            cell_nodes = sd.cell_nodes().tocsc()
            polygons = []
            for c in range(sd.num_cells):
                nodes = cell_nodes.indices[
                    cell_nodes.indptr[c] : cell_nodes.indptr[c + 1]
                ]
                xy = sd.nodes[:2, nodes].T
                centre = xy.mean(axis=0)
                order = np.argsort(
                    np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
                )
                polygons.append(xy[order])
            geometry["matrix"] = {
                "polygons": polygons,
                "centers": sd.cell_centers[:2].T,
            }
        elif sd.dim == mdg.dim_max() - 1:
            geometry["fracture"] = {"centers": sd.cell_centers[:2].T}
    return geometry


def _to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


def save_results_json(
    results: list, path: str | Path, meta: Optional[dict] = None
) -> Path:
    """Write the per-level results to JSON, coarsest level first.

    Parameters:
        results: Per-level result objects.
        path: Output file.
        meta: Metadata stored with the results.

    Returns:
        The path written.

    """
    levels = []
    for result in sorted(results, key=lambda r: -r.cell_diameter):
        entry = {
            "cell_diameter": result.cell_diameter,
            "dt": getattr(result, "dt", None),
            "time": getattr(result, "time", None),
            "errors": {k: v for k, v in vars(result).items() if k.startswith("error_")},
            "values": getattr(result, "values", {}),
            "geometry": getattr(result, "geometry", {}),
        }
        levels.append(_to_json(entry))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": _to_json(meta or {}), "levels": levels}))
    return path


def save_ensemble_json(
    analysis,
    results: list,
    path: str | Path,
    meta: Optional[dict] = None,
    bootstrap_samples: int = 2000,
) -> Path:
    """Write a mesh-ensemble study to JSON: level averages, members and fitted orders.

    Parameters:
        analysis: Ensemble analysis that produced the results.
        results: Level averages from ``analysis.run_analysis()``.
        path: Output file.
        meta: Metadata stored with the results.
        bootstrap_samples: Number of resamples for the order intervals.

    Returns:
        The path written.

    """
    names = sorted(k for k in vars(results[0]) if k.startswith("error_"))
    full = analysis.order_of_convergence(results)
    last_two = analysis.order_of_convergence(results, data_range=slice(-2, None))

    # Fields cannot be averaged across meshes; store those of the unperturbed mesh (jitter 0),
    # or of the member closest to it, so plot_exact_vs_computed can draw them.
    shown = [
        min(members, key=lambda m: abs(getattr(m, "mesh_jitter", 0.0) or 0.0))
        for members in analysis.ensemble_results
    ]

    levels = []
    for level, (members, mean) in enumerate(zip(analysis.ensemble_results, results)):
        levels.append(
            {
                "cell_diameter": mean.cell_diameter,
                "dt": getattr(mean, "dt", None),
                "time": getattr(mean, "time", None),
                "errors": {n: getattr(mean, n) for n in names},
                "values": getattr(shown[level], "values", {}),
                "geometry": getattr(shown[level], "geometry", {}),
                "fields_from_jitter": getattr(shown[level], "mesh_jitter", None),
                "spread": getattr(mean, "ensemble_spread", {}),
                "members": [
                    {
                        "jitter": getattr(m, "mesh_jitter", None),
                        "cell_diameter": m.cell_diameter,
                        "errors": {n: getattr(m, n) for n in names},
                    }
                    for m in members
                ],
                "duplicates_dropped": [
                    j for lvl, j in analysis.ensemble_duplicates if lvl == level
                ],
            }
        )
    orders = {}
    for n in names:
        q = n.removeprefix("error_")
        orders[q] = {
            "all_levels": full.get("ooc_" + q),
            "last_two_levels": last_two.get("ooc_" + q),
            "bootstrap95_all_levels": analysis.bootstrap_order_of_convergence(
                q, samples=bootstrap_samples
            ),
            "bootstrap95_last_two_levels": analysis.bootstrap_order_of_convergence(
                q, samples=bootstrap_samples, data_range=slice(-2, None)
            ),
            "pooled": analysis.pooled_order_of_convergence(q),
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_json({"meta": meta or {}, "levels": levels, "orders": orders}))
    )
    return path


def load_results_json(path: str | Path) -> list:
    """Read results written by :func:`save_results_json`, coarsest level first.

    Parameters:
        path: Results file.

    Returns:
        The per-level results.

    """
    data = json.loads(Path(path).read_text())
    results = []
    for entry in data["levels"]:
        values = {
            name: {key: np.asarray(array) for key, array in pair.items()}
            for name, pair in entry["values"].items()
        }
        geometry = {
            name: {
                key: [np.asarray(p) for p in item]
                if key == "polygons"
                else np.asarray(item)
                for key, item in grid.items()
            }
            for name, grid in entry["geometry"].items()
        }
        results.append(
            SimpleNamespace(
                cell_diameter=entry["cell_diameter"],
                dt=entry["dt"],
                time=entry["time"],
                values=values,
                geometry=geometry,
                **entry["errors"],
            )
        )
    return results
