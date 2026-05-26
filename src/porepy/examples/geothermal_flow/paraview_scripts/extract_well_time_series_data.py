from pathlib import Path

import meshio
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional

import argparse

from porepy.examples.geothermal_flow.plotting.plot_production_diagnostics import (
    compute_production_diagnostics,
    save_production_diagnostics_cache,
)

# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def extract_well_timeseries(
    pvd_file: str,
    cell_id: int,
    variables: list[str],
    shape: str = "triangle",
    block_filter: str = "data_2",
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract time series of variables at a specific cell from a PVD output.

    Parameters
    ----------
    pvd_file : str
        Path to the PVD index file.
    cell_id : int
        Cell index in the chosen subdomain block (e.g. matrix cell adjacent to
        the producer).
    variables : list[str]
        Cell-data variable names to extract (e.g. ["pressure", "temperature"]).
    shape : str
        VTK cell type stored in cell_data_dict (typically "triangle" for 2D).
    block_filter : str
        Substring used to select the right block from the PVD (e.g. "data_2"
        for the 2D matrix block).

    Returns
    -------
    timesteps : np.ndarray
        Array of times in days.
    results : dict[str, np.ndarray]
        Dict mapping variable name to time-series array.
    """
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    datasets = root.findall(".//DataSet")

    timesteps = np.array(
        [
            float(ds.attrib["timestep"]) / (3600 * 24)
            for ds in datasets
            if block_filter in ds.attrib["file"]
        ]
    )
    filenames = [
        ds.attrib["file"] for ds in datasets if block_filter in ds.attrib["file"]
    ]
    base_dir = os.path.dirname(os.path.abspath(pvd_file))
    filenames = [os.path.join(base_dir, f) for f in filenames]

    results = {var: [] for var in variables}
    for fname in filenames:
        mesh = meshio.read(fname)
        for var in variables:
            if var in mesh.cell_data_dict:
                arr = mesh.cell_data_dict[var][shape]
                results[var].append(arr[cell_id] if cell_id < len(arr) else np.nan)
            else:
                results[var].append(np.nan)

    results = {var: np.array(vals) for var, vals in results.items()}
    return timesteps, results


def extract_or_load_well_timeseries(
    pvd_file: str,
    cell_id: int,
    variables: list[str],
    cache_path: str,
    shape: str = "triangle",
    block_filter: str = "data_2",
    force_reextract: bool = False,
    max_time_days: Optional[float] = None,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load extracted well time series from cache, or extract from PVD if
    not cached.
    """
    if os.path.exists(cache_path) and not force_reextract:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        timesteps = df["time_days"].values
        results = {var: df[var].values for var in variables if var in df.columns}
        # return timesteps, results
    else:
        print("Extracting from PVD (this may take a while)...")
        timesteps, results = extract_well_timeseries(
            pvd_file=pvd_file,
            cell_id=cell_id,
            variables=variables,
            shape=shape,
            block_filter=block_filter,
        )

        # Cache the full dataset for future use
        df = pd.DataFrame({"time_days": timesteps, **results})
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached data saved to {cache_path}")
    
    # --- 2. Filter by Time if Requested ---
    if max_time_days is not None:
        # Create a boolean mask where timesteps are less than or equal to the max time
        mask = timesteps <= max_time_days
        
        # Apply mask to timesteps and all variable arrays
        timesteps = timesteps[mask]
        results = {var: vals[mask] for var, vals in results.items()}

    return timesteps, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pvd", required=True)
    parser.add_argument("--cell-id", type=int, required=True)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--diagnostics-cache", required=True)
    parser.add_argument("--shape", default="triangle")
    parser.add_argument("--block-filter", default="data_2")
    parser.add_argument("--max-time-days", type=float, default=None)
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timesteps, results = extract_or_load_well_timeseries(
        pvd_file=args.pvd,
        cell_id=args.cell_id,
        variables=args.variables,
        cache_path=args.cache,
        shape=args.shape,
        block_filter=args.block_filter,
        force_reextract=args.force_reextract,
        max_time_days=args.max_time_days,
    )

    vtk_file = (
        Path.cwd()
        / "geothermal_flow"
        / "model_configuration"
        / "constitutive_description"
        / "driesner_vtk_files"
        / "XHP_l2_original_salt_new.vtk"
    )

    timesteps, q_prod, energy_prod = compute_production_diagnostics(
        timesteps=timesteps,
        results=results,
        vtk_file=vtk_file,
    )

    save_production_diagnostics_cache(
        output_csv=args.diagnostics_cache,
        timesteps=timesteps,
        q_prod=q_prod,
        energy_prod=energy_prod,
    )


if __name__ == "__main__":
    main()