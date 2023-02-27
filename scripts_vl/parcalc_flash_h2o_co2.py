"""Experimental script for parallelyzed processing of thermo data"""
import csv
import pathlib
import multiprocessing
import itertools
import psutil
import time

import numpy as np
import porepy as pp

from multiprocessing import Array, Pool, RawArray
from ctypes import c_double, c_uint8

NUM_PHYS_CPU_CORS = psutil.cpu_count(logical=False)

def path():
    return str(pathlib.Path(__file__).parent.resolve())


def get_thermodata(files: list[tuple[str, str]]):

    p_points: list[float] = list()
    T_points: list[float] = list()
    # to identify file and row per pT point
    # (p, T) -> (mode, file name, row id)
    pT_id: dict[tuple[float, float], tuple[str, str, int]] = dict()

    print("Reading data ...", end='', flush=True)
    for filename, mode in files:
        with open(f"{path()}/{filename}") as file:
            file_reader = csv.reader(file, delimiter=",")
            _ = next(file_reader)  # get rid of header

            for datarow in file_reader:

                row_id = int(datarow[0])
                p = float(datarow[1])
                T = float(datarow[2])

                pT = (p, T)
                # get only unique points
                if pT not in pT_id:
                    identifier = (mode, filename, row_id)
                    pT_id.update({pT: identifier})

                    p_points.append(p)
                    T_points.append(T)
    print("\rReading data ... DONE", flush=True)
    return p_points, T_points, pT_id

def get_MIX_AD_FLASH(num_cells):

    h2o_frac = 0.99
    co2_frac = 0.01

    MIX = pp.composite.PR_Composition(nc=num_cells)
    AD = MIX.ad_system

    # components
    H2O = pp.composite.H2O(AD)
    CO2 = pp.composite.CO2(AD)
    # phases
    LIQ = pp.composite.PR_Phase(AD, False, name="L")
    GAS = pp.composite.PR_Phase(AD, True, name="G")

    MIX.add_components([H2O, CO2])
    MIX.add_phases([LIQ, GAS])

    # setting feed fractions
    AD.set_variable_values(
        h2o_frac * np.ones(num_cells),
        variables=[H2O.fraction_name],
        to_iterate=True,
        to_state=True,
    )
    AD.set_variable_values(
        co2_frac * np.ones(num_cells),
        variables=[CO2.fraction_name],
        to_iterate=True,
        to_state=True,
    )
    # Setting zero enthalpy to get the AD framework going (h is irrelevant here)
    AD.set_variable_values(
        0 * np.ones(num_cells), variables=[MIX.h_name], to_iterate=True, to_state=True
    )

    MIX.initialize()

    # instantiating Flasher, without auxiliary variables V and W
    FLASH = pp.composite.Flash(MIX, auxiliary_npipm=False)
    FLASH.use_armijo = True
    FLASH.armijo_parameters["rho"] = 0.99
    FLASH.armijo_parameters["j_max"] = 55  # cap the number of Armijo iterations
    FLASH.armijo_parameters["return_max"] = True  # return max Armijo iter, even if not min
    FLASH.flash_tolerance = 1e-8
    FLASH.max_iter_flash = 140

    return MIX, AD, FLASH

def write_identifier_file(filename, pT_id):

    # identifiers
    print("Writing identifier file ...", end='', flush=True)
    with open(f"{path()}/{filename}", "w", newline="") as csvfile:
        id_writer = csv.writer(csvfile, delimiter=",")
        # header labeling column values
        header = ["p [Pa]", "T [K]", "mode", "file", "row-id"]
        id_writer.writerow(header)

        for pT, identifier in pT_id.items():
            p, T = pT
            mode, file, row_id = identifier
            row = [p, T, mode, file, row_id]
            id_writer.writerow(row)
    print("\rWriting identifier file ... DONE", flush=True)

def write_results(filename, pT_points, results):

    p_points, T_points = pT_points
    success, num_iter, y, x_h2o_L, x_co2_L, x_h2o_G, x_co2_G, Z_L, Z_G, cond_start, cond_end = results

    print("Writing results ...", end='', flush=True)
    with open(f"{path()}/{filename}", "w", newline="") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        # header labeling column values
        header = [
            "p [Pa]",
            "T [K]",
            "success",
            "y",
            "x_h2o_L",
            "x_co2_L",
            "x_h2o_G",
            "x_co2_G",
            "Z_L",
            "Z_G",
            "num_iter",
            "cond_start",
            "cond_end",
        ]
        result_writer.writerow(header)

        for i, pT in enumerate(zip(p_points, T_points)):
            p, T = pT
            row = [
                p,
                T,
                success[i],
                y[i],
                x_h2o_L[i],
                x_co2_L[i],
                x_h2o_G[i],
                x_co2_G[i],
                Z_L[i],
                Z_G[i],
                num_iter[i],
                cond_start[i],
                cond_end[i],
            ]
            result_writer.writerow(row)
        print("\rWriting results ... DONE", flush=True)

def par_flash(args):

    # storage, ipT = args

    i, p, T = args
    global arrs_loc
    success, num_iter, y, x_h2o_L, x_co2_L, x_h2o_G, x_co2_G, Z_L, Z_G, cond_start, cond_end = arrs_loc

    MIX, AD, FLASH = get_MIX_AD_FLASH(num_cells=1)
    LIQ, GAS = [phase for phase in MIX.phases]
    H2O, CO2 = [comp for comp in MIX.components]

    p_vec = np.array([p], dtype=np.double) * 1e-6
    T_vec = np.array([T])

    AD.set_variable_values(
        p_vec, variables=[MIX.p_name], to_iterate=True, to_state=True,
    )
    AD.set_variable_values(
        T_vec, variables=[MIX.T_name], to_iterate=True, to_state=True
    )
    print(f"Performing flash {i} ...", flush=True)
    try:
        success_ = FLASH.flash(
            flash_type="isothermal",
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
            MIX.compute_roots()
        except Exception:
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L[i] = 0.
            Z_G[i] = 0.
        else:
            # if successful, store values
            Z_L[i] = LIQ.eos.Z.val[0]
            Z_G[i] = GAS.eos.Z.val[0]

        # extract and store results from last iterate
        success[i] = 1
        y[i] = GAS.fraction.evaluate(AD).val[0]
        x_h2o_L[i] = LIQ.fraction_of_component(H2O).evaluate(AD).val[0]
        x_co2_L[i] = LIQ.fraction_of_component(CO2).evaluate(AD).val[0]
        x_h2o_G[i] = GAS.fraction_of_component(H2O).evaluate(AD).val[0]
        x_co2_G[i] = GAS.fraction_of_component(CO2).evaluate(AD).val[0]

        num_iter[i] = FLASH.flash_history[-1]['iterations']
        cond_start[i] = FLASH.cond_start
        cond_end[i] = FLASH.cond_end
    else:
        Z_L[i] = 0.
        Z_G[i] = 0.
        success[i] = 0
        y[i] = 0.
        x_h2o_L[i] = 0.
        x_co2_L[i] = 0.
        x_h2o_G[i] = 0.
        x_co2_G[i] = 0.

        num_iter[i] = 0
        cond_start[i] = 0.
        cond_end[i] = 0.

    return f"Flash {i} finished."

def local_storage(storage: list[Array]):
    global arrs_loc

    arrs_loc = [np.frombuffer(vec.get_obj(), dtype=c_uint8) for vec in storage[:2]]
    arrs_loc += [np.frombuffer(vec.get_obj(), dtype=c_double) for vec in storage[2:]]

if __name__ == '__main__':
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
    version = "reg-omar-par-cond"
    output_file = f"data/results/pr_result_VL_{version}.csv"  # file with flash data
    identifier_file = (
        f"data/results/pr_result_VL_{version}_ID.csv"  # file to identify thermo data
    )

    # reading p-T points from thermo data files
    p_points, T_points, pT_id = get_thermodata(files)
    nc = len(p_points)

    # writing identifier file
    write_identifier_file(identifier_file, pT_id)

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
    storage = [success, num_iter, y, x_h2o_L, x_co2_L, x_h2o_G, x_co2_G, Z_L, Z_G, cond_start, cond_end]
    # par_args = itertools.product([storage], npT)

    print("Performing parallel flash ... ", flush=True)
    start_time = time.time()
    # multiprocessing.set_start_method('fork')
    with Pool(
        processes=NUM_PHYS_CPU_CORS, initargs=(storage,), initializer=local_storage
    ) as pool:

        chunksize  = NUM_PHYS_CPU_CORS
        # chunksize=int(np.floor(nc / NUM_PHYS_CPU_CORS))
        pool.imap_unordered(par_flash, npT, chunksize=chunksize)

        pool.close()
        pool.join()

    end_time = time.time()
    print(f"Ended parallel Flash after {end_time - start_time} seconds.", flush=True)

    results = [np.frombuffer(vec.get_obj(), dtype=c_uint8) for vec in storage[:2]]
    results += [np.frombuffer(vec.get_obj(), dtype=c_double) for vec in storage[2:]]
    # writing result file
    write_results(output_file, (p_points, T_points), results)