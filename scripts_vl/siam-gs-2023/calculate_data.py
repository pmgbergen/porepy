"""Script for calculating data used for comparison with thermo and figures in general.

Warning:
    The calculation using the PorePy flash might take a long time, depending on the
    set resolution.

"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    DATA_PATH,
    P_LIMITS,
    SALINITIES,
    T_LIMITS,
    RESOLUTION_pT,
    calculate_porepy_data,
    logger,
    sal_path,
    write_results,
)

if __name__ == "__main__":

    p_points = np.linspace(P_LIMITS[0], P_LIMITS[1], num=RESOLUTION_pT).tolist()
    T_points = np.linspace(T_LIMITS[0], T_LIMITS[1], num=RESOLUTION_pT).tolist()

    T_points, p_points = np.meshgrid(T_points, p_points)
    T_points = T_points.flatten().tolist()
    p_points = p_points.flatten().tolist()

    for sal in SALINITIES:
        logger.info(f"Starting computations for salinity {sal}\n")
        p = sal_path(DATA_PATH, sal)
        results = calculate_porepy_data(
            p_points, T_points, "p-T", salinity=sal, quickshot=False
        )
        write_results(p, results)
