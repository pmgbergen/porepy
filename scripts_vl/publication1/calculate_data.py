"""Script for calculating data used for comparison with thermo and figures in general.

Warning:
    The calculation using the PorePy flash might take a long time, depending on the
    set resolution.

"""
from __future__ import annotations
import pathlib
import sys
import time

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    calculate_thermo_pT_data,
    write_results,
    read_px_data,
    read_results,
    THERMO_DATA_PATH,
    PT_FLASH_DATA_PATH,
    PH_FLASH_DATA_PATH,
    logger
)

if __name__ == "__main__":

    total_time_start = time.time()

    logger.info("Starting thermo p-T-calculations ..\n")
    start_time = time.time()
    results_thermo = calculate_thermo_pT_data()
    end_time = time.time()
    logger.info(f"Finished thermo p-T-calculations ({end_time - start_time} seconds).")
    write_results(THERMO_DATA_PATH, results_thermo)

    logger.info("Reading p-T data for PorePy flash ..\n")
    p_points, T_points = read_px_data(THERMO_DATA_PATH, 'T')

    # logger.info("Starting PorePy p-T-calculations ..\n")
    # start_time = time.time()

    # end_time = time.time()
    # logger.info(f"Finished PorePy p-T-calculations ({end_time - start_time} seconds).")
    # write_results(PT_FLASH_DATA_PATH, results_pT)

    r = read_results(THERMO_DATA_PATH)

    total_time_end = time.time()
    logger.info(f"Data computed (total {total_time_end - total_time_start} seconds)")