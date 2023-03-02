"""Experimental script for parallelyzed processing of thermo data.

Important:
    This script relies on some local modifications of the Flash to store condition
    numbers. Make sure to comment/uncomment respective lines.
"""
import pathlib
import sys

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

import time
from ctypes import c_double, c_uint8
from multiprocessing import Array, Pool, RawArray

import numpy as np
import psutil
from thermo_comparison import FEED

import porepy as pp

NUM_PHYS_CPU_CORS = psutil.cpu_count(logical=False)


def get_MIX_AD_FLASH(num_cells):

    h2o_frac = FEED["water"]
    co2_frac = FEED["CO2"]

    MIX = pp.composite.PengRobinsonMixture(nc=num_cells)
    ADS = MIX.AD.system

    # components
    H2O = pp.composite.H2O(ADS)
    CO2 = pp.composite.CO2(ADS)
    # phases
    LIQ = pp.composite.PR_Phase(ADS, False, name="L")
    GAS = pp.composite.PR_Phase(ADS, True, name="G")

    MIX.add([H2O, CO2], [LIQ, GAS])

    # setting feed fractions
    ADS.set_variable_values(
        h2o_frac * np.ones(num_cells),
        variables=[H2O.fraction.name],
        to_iterate=True,
        to_state=True,
    )
    ADS.set_variable_values(
        co2_frac * np.ones(num_cells),
        variables=[CO2.fraction.name],
        to_iterate=True,
        to_state=True,
    )

    MIX.AD.set_up()
    # Setting zero enthalpy to get the AD framework going (h is irrelevant here)
    ADS.set_variable_values(
        0 * np.ones(num_cells),
        variables=[MIX.AD.h.name],
        to_iterate=True,
        to_state=True,
    )

    # instantiating Flasher, without auxiliary variables V and W
    FLASH = pp.composite.Flash(MIX, auxiliary_npipm=False)
    FLASH.use_armijo = True
    FLASH.armijo_parameters["rho"] = 0.99
    FLASH.armijo_parameters["j_max"] = 55  # cap the number of Armijo iterations
    FLASH.armijo_parameters[
        "return_max"
    ] = True  # return max Armijo iter, even if not min
    FLASH.flash_tolerance = 1e-8
    FLASH.max_iter_flash = 140

    return MIX, ADS, FLASH


def par_flash(args):

    # storage, ipT = args

    i, p, T = args
    global arrs_loc
    (
        success,
        num_iter,
        y,
        x_h2o_L,
        x_co2_L,
        x_h2o_G,
        x_co2_G,
        Z_L,
        Z_G,
        cond_start,
        cond_end,
    ) = arrs_loc

    MIX, ADS, FLASH = get_MIX_AD_FLASH(num_cells=1)
    LIQ, GAS = [phase for phase in MIX.phases]
    H2O, CO2 = [comp for comp in MIX.components]

    p_vec = np.array([p], dtype=np.double) * 1e-6
    T_vec = np.array([T])

    ADS.set_variable_values(
        p_vec,
        variables=[MIX.AD.p.name],
        to_iterate=True,
        to_state=True,
    )
    ADS.set_variable_values(
        T_vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
    )
    print(f"Performing flash {i} ...", flush=True)
    try:
        success_ = FLASH.flash(
            flash_type="pT",
            method="npipm",
            initial_guess="rachford_rice",
            copy_to_state=False,  # don't overwrite the state, store as iterate
            do_logging=False,
        )
    except Exception:  # if Flasher fails, flag as failed
        success_ = False
    print(f"Flash {i} done.", flush=True)

    if success_:
        try:
            MIX.precompute(apply_smoother=False)
        except Exception:
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L[i] = 0.0
            Z_G[i] = 0.0
        else:
            # if successful, store values
            Z_L[i] = LIQ.eos.Z.val[0]
            Z_G[i] = GAS.eos.Z.val[0]

        # extract and store results from last iterate
        success[i] = 1
        y[i] = GAS.fraction.evaluate(ADS).val[0]
        x_h2o_L[i] = LIQ.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_co2_L[i] = LIQ.fraction_of_component(CO2).evaluate(ADS).val[0]
        x_h2o_G[i] = GAS.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_co2_G[i] = GAS.fraction_of_component(CO2).evaluate(ADS).val[0]

        num_iter[i] = FLASH.flash_history[-1]["iterations"]
        cond_start[i] = FLASH.cond_start
        cond_end[i] = FLASH.cond_end
    else:
        Z_L[i] = 0.0
        Z_G[i] = 0.0
        success[i] = 0
        y[i] = 0.0
        x_h2o_L[i] = 0.0
        x_co2_L[i] = 0.0
        x_h2o_G[i] = 0.0
        x_co2_G[i] = 0.0

        num_iter[i] = 0
        cond_start[i] = 0.0
        cond_end[i] = 0.0

    return f"Flash {i} finished."


def local_storage(storage: list[Array]):
    global arrs_loc

    arrs_loc = [np.frombuffer(vec.get_obj(), dtype=c_uint8) for vec in storage[:2]]
    arrs_loc += [np.frombuffer(vec.get_obj(), dtype=c_double) for vec in storage[2:]]


if __name__ == "__main__":
    # thermo data files
    files = [
        ("data/pr_data_thermo_isothermal_G_easy.csv", "G"),
        ("data/pr_data_thermo_isothermal_G_hard.csv", "G"),
        ("data/pr_data_thermo_isothermal_L_easy.csv", "L"),
        ("data/pr_data_thermo_isothermal_L_hard.csv", "L"),
        ("data/pr_data_thermo_isothermal_GL_easy.csv", "GL"),
        ("data/pr_data_thermo_isothermal_GL_hard.csv", "GL"),
    ]
    # output files
    version = "test2"
    output_file = f"data/results/pr_result_VL_{version}.csv"  # file with flash data
    identifier_file = (
        f"data/results/pr_result_VL_ID.csv"  # file to identify thermo data
    )

    # reading p-T points from thermo data files
    p_points, T_points, pT_id = get_thermodata(files)
    nc = len(p_points)

    # prepare storage of results
    # raw arrays without lock due to the values being not interdependent
    success = Array(typecode_or_type=c_uint8, size_or_initializer=nc)
    num_iter = Array(typecode_or_type=c_uint8, size_or_initializer=nc)
    cond_start = Array(typecode_or_type=c_double, size_or_initializer=nc)
    cond_end = Array(typecode_or_type=c_double, size_or_initializer=nc)
    y = Array(typecode_or_type=c_double, size_or_initializer=nc)
    x_h2o_L = Array(typecode_or_type=c_double, size_or_initializer=nc)
    x_co2_L = Array(typecode_or_type=c_double, size_or_initializer=nc)
    x_h2o_G = Array(typecode_or_type=c_double, size_or_initializer=nc)
    x_co2_G = Array(typecode_or_type=c_double, size_or_initializer=nc)
    Z_L = Array(typecode_or_type=c_double, size_or_initializer=nc)
    Z_G = Array(typecode_or_type=c_double, size_or_initializer=nc)
    # iterable input for parallelism
    # i indicates which position in the global result array the respective pT point takes
    npT = [(i, p, T) for i, p, T in zip(np.arange(nc), p_points, T_points)]
    storage = [
        success,
        num_iter,
        y,
        x_h2o_L,
        x_co2_L,
        x_h2o_G,
        x_co2_G,
        Z_L,
        Z_G,
        cond_start,
        cond_end,
    ]
    # par_args = itertools.product([storage], npT)

    print("Performing parallel flash ... ", flush=True)
    start_time = time.time()
    # multiprocessing.set_start_method('fork')
    with Pool(
        processes=NUM_PHYS_CPU_CORS, initargs=(storage,), initializer=local_storage
    ) as pool:

        chunksize = NUM_PHYS_CPU_CORS
        # chunksize=int(np.floor(nc / NUM_PHYS_CPU_CORS))
        result = pool.map_async(par_flash, npT, chunksize=chunksize)

        result.wait(5 * 60)
        pool.close()
        pool.terminate()
        pool.join()

    end_time = time.time()
    print(f"Ended parallel Flash after {end_time - start_time} seconds.", flush=True)

    results = [np.frombuffer(vec.get_obj(), dtype=c_uint8) for vec in storage[:2]]
    results += [np.frombuffer(vec.get_obj(), dtype=c_double) for vec in storage[2:]]
    # writing identifier file
    write_identifier_file(identifier_file, pT_id)
    # writing result file
    write_results(output_file, (p_points, T_points), results)
