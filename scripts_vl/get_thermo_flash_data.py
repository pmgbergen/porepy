"""Runscript to calculate the data using thermo for the water-CO2, 2-phase mixture."""
import pathlib
import sys

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from thermo_comparison import (
    H_LIMITS,
    P_LIMITS,
    RESOLUTION,
    T_LIMITS,
    read_px_data,
    read_px_identifiers,
    read_results,
    thermo_ph_flash,
    thermo_pT_flash,
    write_px_identifier,
    write_results,
)

if __name__ == "__main__":

    result_file_pT = f"data/thermodata_pT.csv"
    result_file_ph = f"data/thermodata_ph.csv"

    pT_identifier_file = f"data/thermodata_pT_id.csv"
    ph_identifier_file = f"data/thermodata_ph_id.csv"

    eps = 0.01
    p_limits = [0.01e6, 100.0e6]  # [Pa]
    h_limits = [-30000.0, 10000.0]  # [J/mol]
    T_limits = [280, 700]  # [K]

    print("Computing thermo data: calculating", flush=True)
    results_pT = thermo_pT_flash(P_LIMITS, T_LIMITS, p_res=RESOLUTION, T_res=RESOLUTION)

    results_ph = thermo_ph_flash(P_LIMITS, H_LIMITS, p_res=RESOLUTION, h_res=RESOLUTION)

    print("Computing thermo data: writing result files", flush=True)
    write_results(result_file_pT, results_pT)
    write_results(result_file_ph, results_ph)

    print("Computing thermo data: writing identifier files", flush=True)
    p_points, T_points, pT_id = read_px_data([result_file_pT], "T")
    _, h_points, ph_id = read_px_data([result_file_ph], "h")
    write_px_identifier(pT_identifier_file, pT_id, "T")
    write_px_identifier(ph_identifier_file, ph_id, "h")

    # sanity check for I/O to see if the data was stored in a readable
    # way and that nothing messed with the precision
    assert set(results_ph.keys()) == set(
        results_pT.keys()
    ), "Data header inconsistency detected."
    data = read_results([result_file_pT, result_file_ph], list(results_pT.keys()))
    px_id = read_px_identifiers([pT_identifier_file, ph_identifier_file])

    print("Computing thermo data: done", flush=True)
