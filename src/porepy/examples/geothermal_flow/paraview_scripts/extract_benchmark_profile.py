"""Extract 1D benchmark PorePy profiles from a PVD file into CSV."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0


def read_pvd_datasets(pvd_file: str | Path) -> list[dict]:
    """Read DataSet entries from a PVD file."""
    pvd_file = Path(pvd_file).resolve()
    tree = ET.parse(pvd_file)
    root = tree.getroot()

    datasets = []
    for dataset in root.findall(".//DataSet"):
        datasets.append(
            {
                "time": float(dataset.attrib["timestep"]),
                "file": dataset.attrib["file"],
            }
        )

    if not datasets:
        raise RuntimeError(f"No DataSet entries found in {pvd_file}")

    return datasets


def closest_dataset(
    datasets: list[dict],
    *,
    time_years: float | None,
    time_days: float | None,
    time_index: int | None,
    block_filter: str | None,
    pvd_time_unit: str,
) -> dict:
    """Choose one dataset from a PVD file."""
    candidates = datasets

    if block_filter is not None:
        candidates = [entry for entry in datasets if block_filter in entry["file"]]

    if not candidates:
        raise RuntimeError(f"No PVD datasets matched block_filter={block_filter!r}.")

    # if time_years is not None:
    #     target = time_years * SECONDS_PER_YEAR
    #     return min(candidates, key=lambda entry: abs(entry["time"] - target))

    # if time_days is not None:
    #     target = time_days * SECONDS_PER_DAY
    #     return min(candidates, key=lambda entry: abs(entry["time"] - target))

    target = convert_requested_time_to_pvd_units(
        time_years=time_years,
        time_days=time_days,
        pvd_time_unit=pvd_time_unit,
    )

    if target is not None:
        return min(candidates, key=lambda entry: abs(entry["time"] - target))

    if time_index is None:
        time_index = -1

    return candidates[time_index]


def load_mesh_from_pvd(
    pvd_file: str | Path,
    *,
    time_years: float | None = None,
    time_days: float | None = None,
    time_index: int | None = -1,
    block_filter: str | None = None,
    pvd_time_unit: str = "seconds",
) -> tuple[pv.DataSet, float, Path]:
    """Load one mesh referenced by a PVD file."""
    pvd_file = Path(pvd_file).resolve()
    datasets = read_pvd_datasets(pvd_file)

    selected = closest_dataset(
        datasets,
        time_years=time_years,
        time_days=time_days,
        time_index=time_index,
        block_filter=block_filter,
        pvd_time_unit=pvd_time_unit,
    )

    mesh_path = pvd_file.parent / selected["file"]
    if not mesh_path.exists():
        raise FileNotFoundError(f"Referenced VTK/VTU file not found: {mesh_path}")

    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    return mesh, selected["time"], mesh_path


def extract_profile(
    mesh: pv.DataSet, *, gas_zero_threshold: float | None = None
) -> pd.DataFrame:
    """Extract benchmark cell-centered profile data."""
    centers = mesh.cell_centers().points
    x_m = np.asarray(centers[:, 0], dtype=float)
    x_km = x_m * 1.0e-3

    cell_data = mesh.cell_data

    required = ["pressure", "temperature", "s_gas", "s_halite"]
    missing = [name for name in required if name not in cell_data]
    if missing:
        raise ValueError(
            f"Missing required cell-data arrays: {missing}. "
            f"Available arrays: {list(cell_data.keys())}"
        )

    pressure = np.asarray(cell_data["pressure"], dtype=float)
    temperature = np.asarray(cell_data["temperature"], dtype=float)
    s_gas = np.asarray(cell_data["s_gas"], dtype=float)
    s_halite = np.asarray(cell_data["s_halite"], dtype=float)

    if gas_zero_threshold is not None:
        s_gas_for_liq = np.where(
            (0.0 < s_gas) & (s_gas <= gas_zero_threshold),
            0.0,
            s_gas,
        )
    else:
        s_gas_for_liq = s_gas

    s_liq = 1.0 - (s_gas_for_liq + s_halite)

    data = {
        "x_m": x_m,
        "x_km": x_km,
        "pressure": pressure,
        "pressure_MPa": pressure * 1.0e-6,
        "temperature": temperature,
        "temperature_C": temperature - 273.15,
        "s_gas": s_gas,
        "s_halite": s_halite,
        "s_liq": s_liq,
    }

    if "z_NaCl" in cell_data:
        data["z_NaCl"] = np.asarray(cell_data["z_NaCl"], dtype=float)

    if "enthalpy" in cell_data:
        data["enthalpy"] = np.asarray(cell_data["enthalpy"], dtype=float)

    df = pd.DataFrame(data)
    df = df.sort_values("x_m").reset_index(drop=True)

    return df


def convert_requested_time_to_pvd_units(
    *,
    time_years: float | None,
    time_days: float | None,
    pvd_time_unit: str,
) -> float | None:
    """Convert requested physical time to the unit used in the PVD file."""
    if time_years is None and time_days is None:
        return None

    if time_years is not None:
        if pvd_time_unit == "years":
            return time_years
        if pvd_time_unit == "days":
            return time_years * 365.0
        return time_years * 365.0 * 24.0 * 3600.0

    if time_days is not None:
        if pvd_time_unit == "years":
            return time_days / 365.0
        if pvd_time_unit == "days":
            return time_days
        return time_days * 24.0 * 3600.0

    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract benchmark 1D PorePy profile from a PVD file."
    )
    parser.add_argument("--pvd", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--time-years", type=float, default=None)
    parser.add_argument("--time-days", type=float, default=None)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument(
        "--block-filter",
        default=None,
        help="Optional substring used to select one PVD block/file.",
    )
    parser.add_argument(
        "--gas-zero-threshold",
        type=float,
        default=0.1,
        help=(
            "If set, gas saturation values in (0, threshold] are treated as zero "
            "when computing liquid saturation."
        ),
    )
    parser.add_argument(
        "--pvd-time-unit",
        choices=["seconds", "days", "years"],
        default="seconds",
        help="Unit used by timestep values stored in the PVD file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run extraction."""
    args = parse_args()

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mesh, selected_time, mesh_path = load_mesh_from_pvd(
        args.pvd,
        time_years=args.time_years,
        time_days=args.time_days,
        time_index=args.time_index,
        block_filter=args.block_filter,
        pvd_time_unit=args.pvd_time_unit,
    )

    df = extract_profile(mesh, gas_zero_threshold=args.gas_zero_threshold)
    df.to_csv(out_csv, index=False)

    print(f"Loaded mesh: {mesh_path}")
    print(f"Selected time: {selected_time:g} seconds")
    print(f"Selected time: {selected_time / SECONDS_PER_YEAR:g} years")
    print(f"Saved benchmark profile: {out_csv}")


if __name__ == "__main__":
    main()
