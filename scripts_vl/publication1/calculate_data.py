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
    ISOTHERM_DATA_PATH,
    PH_FLASH_DATA_PATH,
    PT_FLASH_DATA_PATH,
    PT_QUICKSHOT_DATA_PATH,
    THERMO_DATA_PATH,
    calculate_porepy_isotherm_data,
    calculate_porepy_ph_data,
    calculate_porepy_pT_data,
    calculate_thermo_pT_data,
    logger,
    read_px_data,
    write_results,
)

# Flags for which data should be computed, to avoid long waiting for re-computations
COMPUTE_THERMO_DATA = False
COMPUTE_PT_DATA = False
COMPUTE_PH_DATA = True

if __name__ == "__main__":

    total_time_start = time.time()

    if COMPUTE_THERMO_DATA:
        logger.info("Starting thermo p-T-calculations ..\n")
        start_time = time.time()
        results_thermo = calculate_thermo_pT_data()
        end_time = time.time()
        logger.info(f"Finished thermo p-T-calculations ({end_time - start_time} seconds).")
        write_results(THERMO_DATA_PATH, results_thermo)

    if COMPUTE_PT_DATA:
        logger.info("Reading p-T data for PorePy flash ..\n")
        p_points, T_points = read_px_data(THERMO_DATA_PATH, "T")

        logger.info("Starting PorePy p-T-quickshort calculations ..\n")
        start_time = time.time()
        results_pT = calculate_porepy_pT_data(p_points, T_points, quickshot=True)
        end_time = time.time()
        logger.info(
            f"Finished PorePy p-T-quickshot calculations ({end_time - start_time} seconds)."
        )
        write_results(PT_QUICKSHOT_DATA_PATH, results_pT)

        logger.info("Starting PorePy p-T-calculations ..\n")
        start_time = time.time()
        results_pT = calculate_porepy_pT_data(p_points, T_points)
        end_time = time.time()
        logger.info(f"Finished PorePy p-T-calculations ({end_time - start_time} seconds).")
        write_results(PT_FLASH_DATA_PATH, results_pT)

    if COMPUTE_PH_DATA:
        logger.info("Starting PorePy p-T calculations along isotherms ..\n")
        start_time = time.time()
        results_pT_isotherms = calculate_porepy_isotherm_data()
        end_time = time.time()
        logger.info(f"Finished isotherm-calculations ({end_time - start_time} seconds).")
        write_results(ISOTHERM_DATA_PATH, results_pT_isotherms)

        logger.info("Reading data for isothermal flash ..\n")
        p_points, h_points = read_px_data(ISOTHERM_DATA_PATH, "h")
        logger.info("Starting PorePy p-h calculations along isotherms ..\n")
        start_time = time.time()
        results_ph_isotherms = calculate_porepy_ph_data(p_points, h_points)
        end_time = time.time()
        logger.info(f"Finished p-h-calculations ({end_time - start_time} seconds).")
        write_results(PH_FLASH_DATA_PATH, results_ph_isotherms)

    total_time_end = time.time()
    logger.info(f"Data computed (total {total_time_end - total_time_start} seconds)")
