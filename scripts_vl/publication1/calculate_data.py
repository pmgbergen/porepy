"""Script for calculating data used for comparison with thermo and figures in general.

Warning:
    The calculation using the PorePy flash might take a long time, depending on the
    set resolution.

"""
from __future__ import annotations

import os
import pathlib
import sys
import time
import logging

import numpy as np

from porepy.composite.composite_utils import COMPOSITE_LOGGER as logger

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    GEO_DATA_PATH,
    GEO_THERMO_DATA_PATH,
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
    T_HEADER,
    THERMO_DATA_PATH,
    EXAMPLE_2_flash_type,
    RESOLUTION_hv,
    RESOLUTION_ph,
    calculate_example_2_thermo,
    calculate_porepy_data,
    calculate_example_1_thermo,
    h_HEADER,
    p_HEADER,
    path,
    read_data_column,
    v_HEADER,
    write_results,
    create_mixture,
    SPECIES,
    SPECIES_geo,
    GEO_P_LIMITS,
    GEO_H_LIMITS,
    GEO_T_LIMITS,
    RESOLUTION_geo,
    create_mixture_geo,
)

# Flags for which data should be computed, to avoid long waiting for re-computations
COMPUTE_THERMO_DATA = True
COMPUTE_PT_DATA = True
COMPUTE_PH_DATA = True
COMPUTE_HV_DATA = True
COMPUTE_GEO_THERMO_DATA = False
COMPUTE_GEO_DATA = False

if __name__ == "__main__":

    logger.setLevel(logging.INFO)

    total_time_start = time.time()

    compile_start = time.time()
    mix, flash = create_mixture(verbosity=2)
    mix_2, flash_2 = create_mixture_geo(verbosity=2)
    compile_end = time.time()

    logger.warning(f"2-2 mixture created and compiled in {compile_end - compile_start} seconds.\n")

    data_path = f"{path()}/data/"
    if not os.path.isdir(data_path):
        logger.info("Creating data directory ..")
        os.mkdir(data_path)

    if COMPUTE_THERMO_DATA:
        logger.info("Starting thermo calculations ..")
        start_time = time.time()
        results = calculate_example_1_thermo()
        end_time = time.time()
        logger.info(f"Finished thermo calculations ({end_time - start_time} seconds).\n")
        write_results(THERMO_DATA_PATH, results)

    if COMPUTE_PT_DATA:
        logger.info("Reading p-T data for PorePy flash ..")
        p_points = read_data_column(THERMO_DATA_PATH, p_HEADER)
        T_points = read_data_column(THERMO_DATA_PATH, T_HEADER)
        logger.info("Starting PorePy p-T-calculations ..")
        start_time = time.time()
        results = calculate_porepy_data(p_points, T_points, "p-T", SPECIES, flash)
        end_time = time.time()
        logger.info(
            f"Finished PorePy p-T-calculations ({end_time - start_time} seconds).\n"
        )
        write_results(PT_FLASH_DATA_PATH, results)

    if COMPUTE_PH_DATA:
        logger.info("Starting PorePy p-T calculations along isotherms ..")
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
        results = calculate_porepy_data(
            np.array(p_points), np.array(T_points), "p-T", SPECIES, flash
        )
        end_time = time.time()
        logger.info(
            f"Finished isotherm-calculations ({end_time - start_time} seconds).\n"
        )
        write_results(ISOTHERM_DATA_PATH, results)

        logger.info("Reading p-h data for isothermal flash ..")
        p_points = read_data_column(ISOTHERM_DATA_PATH, p_HEADER)
        h_points = read_data_column(ISOTHERM_DATA_PATH, h_HEADER)
        logger.info("Starting PorePy p-h calculations along isotherms ..")
        start_time = time.time()
        results = calculate_porepy_data(
            np.array(p_points), np.array(h_points), "p-h", SPECIES, flash)
        end_time = time.time()
        logger.info(f"Finished p-h-calculations ({end_time - start_time} seconds).\n")
        write_results(PH_FLASH_DATA_PATH, results)

    if COMPUTE_HV_DATA:
        logger.info("Starting PorePy p-T calculations along isobar and isotherm ..")
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
        results = calculate_porepy_data(
            np.array(p_points), np.array(T_points), "p-T", SPECIES, flash
        )
        end_time = time.time()
        logger.info(f"Finished isobar-calculations ({end_time - start_time} seconds).\n")
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
        results = calculate_porepy_data(
            np.array(p_points), np.array(T_points), "p-T", SPECIES, flash
        )
        end_time = time.time()
        logger.info(
            f"Finished isotherm-calculations ({end_time - start_time} seconds).\m"
        )
        write_results(HV_ISOTHERM_DATA_PATH, results)

        logger.info("Starting PorePy h-v calculations ..")
        h1 = read_data_column(HV_ISOBAR_DATA_PATH, h_HEADER)
        v1 = read_data_column(HV_ISOBAR_DATA_PATH, v_HEADER)
        h2 = read_data_column(HV_ISOTHERM_DATA_PATH, h_HEADER)
        v2 = read_data_column(HV_ISOTHERM_DATA_PATH, v_HEADER)
        h_points = h1 + h2
        v_points = v1 + v2
        start_time = time.time()
        results = calculate_porepy_data(
            np.array(v_points), np.array(h_points), "v-h", SPECIES, flash
        )
        end_time = time.time()
        logger.info(f"Finished p-h-calculations ({end_time - start_time} seconds).\n")
        write_results(HV_FLASH_DATA_PATH, results)

    if COMPUTE_GEO_THERMO_DATA:
        logger.info("Starting thermo calculations for geothermal fluid ..")

        start_time = time.time()
        results = calculate_example_2_thermo(EXAMPLE_2_flash_type)
        end_time = time.time()
        logger.info(f"Finished thermo calculations ({end_time - start_time} seconds).\n")
        write_results(GEO_THERMO_DATA_PATH, results)

    if COMPUTE_GEO_DATA:
        logger.info("Starting PorePy calculations for geothermal fluid ..")
        p_ = np.linspace(
            GEO_P_LIMITS[0],
            GEO_P_LIMITS[1],
            RESOLUTION_geo,
            endpoint=True,
            dtype=float,
        )

        if EXAMPLE_2_flash_type == "p-h":
            x_ = np.linspace(
                GEO_H_LIMITS[0],
                GEO_H_LIMITS[1],
                RESOLUTION_geo,
                endpoint=True,
                dtype=float,
            )
        elif EXAMPLE_2_flash_type == "p-T":
            x_ = np.linspace(
                GEO_T_LIMITS[0],
                GEO_T_LIMITS[1],
                RESOLUTION_geo,
                endpoint=True,
                dtype=float,
        )
        x, p = np.meshgrid(x_, p_)
        start_time = time.time()
        results = calculate_porepy_data(p.flat, x.flat, SPECIES_geo, EXAMPLE_2_flash_type, flash_2)
        end_time = time.time()
        logger.info(f"Finished {EXAMPLE_2_flash_type}-calculations ({end_time - start_time} seconds).\n")
        write_results(GEO_DATA_PATH, results)

    total_time_end = time.time()
    logger.info(f"Data computed (total {total_time_end - total_time_start} seconds)\n")
