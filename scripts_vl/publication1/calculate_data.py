"""Script for calculating data used for comparison with thermo and figures in general.

Warning:
    The calculation using the PorePy flash might take a long time, depending on the
    set resolution.

"""
from __future__ import annotations

import pathlib
import sys
import os
import time

import numpy as np

import porepy as pp

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    HV_FLASH_DATA_PATH,
    HV_ISOBAR,
    HV_ISOBAR_DATA_PATH,
    HV_ISOBAR_T_LIMITS,
    HV_ISOTHERM,
    HV_ISOTHERM_DATA_PATH,
    HV_ISOTHERM_P_LIMITS,
    ISOTHERM_DATA_PATH,
    ISOTHERMS,
    P_LIMITS_ISOTHERMS,
    PH_FLASH_DATA_PATH,
    PT_FLASH_DATA_PATH,
    SPECIES,
    T_HEADER,
    THERMO_DATA_PATH,
    RESOLUTION_hv,
    RESOLUTION_ph,
    calculate_porepy_data,
    calculate_thermo_pT_data,
    h_HEADER,
    logger,
    p_HEADER,
    read_data_column,
    v_HEADER,
    write_results,
    path,
)

# Flags for which data should be computed, to avoid long waiting for re-computations
COMPUTE_THERMO_DATA = True
COMPUTE_PT_DATA = True
COMPUTE_PH_DATA = True
COMPUTE_HV_DATA = True

if __name__ == "__main__":

    total_time_start = time.time()

    logger.info("Fetching constant parameters ..\n")
    species = pp.composite.load_species(SPECIES)

    comps = [
        pp.composite.peng_robinson.H2O.from_species(species[0]),
        pp.composite.peng_robinson.CO2.from_species(species[1]),
    ]
    eos = pp.composite.peng_robinson.PengRobinson(True)
    eos.components = comps
    logger.info("Name\tT_crit\tp_crit\tomega\n")
    for c in comps:
        logger.info(f"{c.name}\t{c.T_crit}\t{c.p_crit}\t{c.omega}\n")
    logger.info("Binary interaction parameters:\n")
    bip = pp.composite.peng_robinson.load_bip(
        comps[0].CASr_number, comps[1].CASr_number
    )
    logger.info(f"{comps[0].name} - {comps[1].name}: {bip}\n")

    data_path = f"{path()}/data/"
    if not os.path.isdir(data_path):
        logger.info("Creating data directory ..\n")
        os.mkdir(data_path)

    if COMPUTE_THERMO_DATA:
        logger.info("Starting thermo calculations ..\n")
        start_time = time.time()
        results = calculate_thermo_pT_data()
        end_time = time.time()
        logger.info(f"Finished thermo calculations ({end_time - start_time} seconds).")
        write_results(THERMO_DATA_PATH, results)

    if COMPUTE_PT_DATA:
        logger.info("Reading p-T data for PorePy flash ..")
        p_points = read_data_column(THERMO_DATA_PATH, p_HEADER)
        T_points = read_data_column(THERMO_DATA_PATH, T_HEADER)
        logger.info("Starting PorePy p-T-calculations ..\n")
        start_time = time.time()
        results = calculate_porepy_data(p_points, T_points, "p-T", quickshot=False)
        end_time = time.time()
        logger.info(
            f"Finished PorePy p-T-calculations ({end_time - start_time} seconds)."
        )
        write_results(PT_FLASH_DATA_PATH, results)

    if COMPUTE_PH_DATA:
        logger.info("Starting PorePy p-T calculations along isotherms ..\n")
        start_time = time.time()
        p_points = list()
        T_points = list()
        p_ = np.linspace(
            P_LIMITS_ISOTHERMS[0],
            P_LIMITS_ISOTHERMS[1],
            RESOLUTION_ph,
            endpoint=True,
            dtype=float,
        )
        for T in ISOTHERMS:
            for p in p_:
                T_points.append(T)
                p_points.append(p)
        results = calculate_porepy_data(p_points, T_points, "p-T", quickshot=False)
        end_time = time.time()
        logger.info(
            f"Finished isotherm-calculations ({end_time - start_time} seconds)."
        )
        write_results(ISOTHERM_DATA_PATH, results)

        logger.info("Reading p-h data for isothermal flash ..")
        p_points = read_data_column(ISOTHERM_DATA_PATH, p_HEADER)
        h_points = read_data_column(ISOTHERM_DATA_PATH, h_HEADER)
        logger.info("Starting PorePy p-h calculations along isotherms ..\n")
        start_time = time.time()
        results = calculate_porepy_data(p_points, h_points, "p-h", quickshot=False)
        end_time = time.time()
        logger.info(f"Finished p-h-calculations ({end_time - start_time} seconds).")
        write_results(PH_FLASH_DATA_PATH, results)

    if COMPUTE_HV_DATA:
        logger.info("Starting PorePy p-T calculations along isobar and isotherm ..\n")
        start_time = time.time()
        p_points = list()
        T_points = list()
        T_ = np.linspace(
            HV_ISOBAR_T_LIMITS[0],
            HV_ISOBAR_T_LIMITS[1],
            RESOLUTION_hv,
            endpoint=True,
            dtype=float,
        )
        for T in T_:
            T_points.append(T)
            p_points.append(HV_ISOBAR)
        results = calculate_porepy_data(p_points, T_points, "p-T", quickshot=False)
        end_time = time.time()
        logger.info(f"Finished isobar-calculations ({end_time - start_time} seconds).")
        write_results(HV_ISOBAR_DATA_PATH, results)

        start_time = time.time()
        p_points = list()
        T_points = list()
        p_ = np.linspace(
            HV_ISOTHERM_P_LIMITS[0],
            HV_ISOTHERM_P_LIMITS[1],
            RESOLUTION_hv,
            endpoint=True,
            dtype=float,
        )
        for p in p_:
            T_points.append(HV_ISOTHERM)
            p_points.append(p)
        results = calculate_porepy_data(p_points, T_points, "p-T", quickshot=False)
        end_time = time.time()
        logger.info(
            f"Finished isotherm-calculations ({end_time - start_time} seconds)."
        )
        write_results(HV_ISOTHERM_DATA_PATH, results)

        logger.info("Starting PorePy h-v calculations ..\n")
        h1 = read_data_column(HV_ISOBAR_DATA_PATH, h_HEADER)
        v1 = read_data_column(HV_ISOBAR_DATA_PATH, v_HEADER)
        h2 = read_data_column(HV_ISOTHERM_DATA_PATH, h_HEADER)
        v2 = read_data_column(HV_ISOTHERM_DATA_PATH, v_HEADER)
        h_points = h1 + h2
        v_points = v1 + v2
        start_time = time.time()
        results = calculate_porepy_data(h_points, v_points, "h-v", quickshot=False)
        end_time = time.time()
        logger.info(f"Finished p-h-calculations ({end_time - start_time} seconds).")
        write_results(HV_FLASH_DATA_PATH, results)

    total_time_end = time.time()
    logger.info(f"Data computed (total {total_time_end - total_time_start} seconds)")
