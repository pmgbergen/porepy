"""Runscript to calculate the data using thermo for the water-CO2, 2-phase mixture."""
import pathlib
import sys
import time

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

    result_file_pT = f"data/thermodata_pT100.csv"
    result_file_ph = f"data/thermodata_ph100.csv"

    pT_identifier_file = f"data/thermodata_pT100_id.csv"
    ph_identifier_file = f"data/thermodata_ph100_id.csv"

    print("Computing thermo data: calculating", flush=True)
    start_time = time.time()
    results_pT = thermo_pT_flash(P_LIMITS, T_LIMITS, p_res=RESOLUTION, T_res=RESOLUTION)
    end_time = time.time()
    print(f"Computing thermo data: p-T data calculated after {end_time - start_time} seconds", flush=True)
    write_results(result_file_pT, results_pT)

    start_time = time.time()
    results_ph = thermo_ph_flash(P_LIMITS, H_LIMITS, p_res=RESOLUTION, h_res=RESOLUTION)
    end_time = time.time()
    print(f"Computing thermo data: p-h data calculated after {end_time - start_time} seconds", flush=True)
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
