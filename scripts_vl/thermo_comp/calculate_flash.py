"""Script to invoke PorePy's unified flash procedure."""
import pathlib
import sys
import time
import psutil

import numpy as np
import porepy as pp

from multiprocessing import Pool, Array, Queue, Process
from ctypes import c_uint8, c_double
from typing import Any

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from thermo_comparison import (
    FEED,
    read_px_data,
    COMPONENTS,
    PHASES,
    NAN_ENTRY,
    get_result_headers,
    write_results
)

# How to calculate:
# 0 - component-wise
# 1 - vectorized
# 2 - parallelized
# This is critical when it comes to performance on big data files
MODE: int = 2
# flash type: pT or ph
FLASH_TYPE: str = 'pT'
# p-x data from an thermo results file
PX_DATA_FILE = f"data/thermodata_pT10K.csv"
# file to which to write the results
RESULT_FILE = f"data/results/results_pT10k_par_wo-reg.csv"
# Number of physical CPU cores.
# This is used for the number of sub-processes and chunksize in the parallelization
NUM_PHYS_CPU_CORS = psutil.cpu_count(logical=False)


def _access_shared_objects(
        shared_arrays: list[tuple[Array, Any]],
        progress_queue: Queue
):
    """Helper function to be called by subprocesses to provide access to shared-memory
    objects.
    
    ``sharred_arrays`` must be a list of tuples, where the first entry is the
    shared array object and the second entry its data type (as a C-type).

    """
    global arrays_loc, progress_queue_loc

    progress_queue_loc = progress_queue
    arrays_loc = [
        np.frombuffer(vec.get_obj(), dtype=dtype)
        for vec, dtype in shared_arrays
    ]


def _create_shared_arrays(size: int) -> list[tuple[Array, Any]]:
    """Creates shared memory arrays for the parallelized flash and returns a list of
    tuples with arrays and their data types as entries. 
    
    Important:
        The order here in the list determines for which quantities they are used.
        The same order is assumed in the parallelized flash.
        It must not be messed with.

        The order corresponds to the order returned by ``get_result_headers``.

    ``size`` determines the size of the allocated arrays. It must be equal the number
    of flashes performed in parallel.

    """
    shared_arrays: list[tuple[Array, Any]] = list()

    INT_PRECISION = c_uint8
    FLOAT_PRECISION = c_double

    def _double_array():
        return Array(typecode_or_type=FLOAT_PRECISION, size_or_initializer=size)
    def _uint_array():
        return Array(typecode_or_type=INT_PRECISION, size_or_initializer=size)

    # array to store the success flag: 0 or 1
    success = _uint_array()
    shared_arrays.append((success, INT_PRECISION))
    # integer array to store the number of iterations necessary
    num_iter = _uint_array()
    shared_arrays.append((num_iter, INT_PRECISION))
    # array storing 1 if the flash showes a super-critical mixture at converged state.
    is_supercrit = _uint_array()
    shared_arrays.append((is_supercrit, INT_PRECISION))
    # array storing the condition number of the array at the beginning (initial guess)
    cond_start = _double_array()
    shared_arrays.append((cond_start, FLOAT_PRECISION))
    # array showing the condition number at converged state
    cond_end = _double_array()
    shared_arrays.append((cond_end, FLOAT_PRECISION))
    # arrays containing pressure, temperature and enthalpies
    p = _double_array()
    shared_arrays.append((p, FLOAT_PRECISION))
    T = _double_array()
    shared_arrays.append((T, FLOAT_PRECISION))
    h = _double_array()
    shared_arrays.append((h, FLOAT_PRECISION))
    # array containing the vapor fraction
    y = _double_array()
    shared_arrays.append((y, FLOAT_PRECISION))
    # arrays containing the phase compressibility factors
    # NOTE: The unified framework has for now only one liquid phase
    for _ in range(2):
        Z = _double_array()
        shared_arrays.append((Z, FLOAT_PRECISION))
    # arrays containing the phase composition
    for _ in COMPONENTS:
        for _ in PHASES[:2]:
            x = _double_array()
            shared_arrays.append((x, FLOAT_PRECISION))
    # arrays containing the fugacities
    for _ in COMPONENTS:
        for _ in PHASES[:2]:
            phi = _double_array()
            shared_arrays.append((phi, FLOAT_PRECISION))
    
    return shared_arrays



def _progress_counter(q: Queue, NC: int):
    """A function sharing a Queue object with other processes, which receives
    the index of all finished flashes. This function finishes when all indices
    where received."""
    progress_array = Array(typecode_or_type=c_uint8, size_or_initializer=NC)

    while True:
        i = q.get()
        progress_array[i] = 1
        progress = int(sum(progress_array))
        print(f"\rParallel flash: {progress}/{NC}", end='', flush=True)
        if progress == NC:
            break


def get_flash_setup(
        num_cells: int
) -> tuple[pp.composite.Mixture, pp.ad.EquationSystem, pp.composite.Flash]:
    """Returns instances of the modelled mixture, respective AD system and flash from
    PorePy's framework, in that order.
    
    ``num_cells`` is an integer indicating how large AD arrays should be.
    I.e., choose 1 for a point-wise and parallelized flash, choose some number ``n``
    for a vectorized flash.

    Important:
        See here how parameters for the flash solver can be set.

    """

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


def parallel_pT_flash(ipT):
    """Performs a p-T flash (including modelling) and stores the results in shared
    memory.
    
    ``ipT`` must be a tuple containing the index where the results should be stored,
    and the p-T point.

    Warning:
        There are some unresolved issues with parallel subprocesses if
        numpy/scipy/pypardiso throw errors or warnings. It causes the respective
        subprocess unable to join, despite finishing all flashes.

    """

    i, p, T = ipT

    FAILURE_ENTRY = NAN_ENTRY

    # accessing shared memory
    global arrays_loc, progress_queue_loc
    (
        success_arr,
        num_iter_arr,
        is_supercrit_arr,
        cond_start_arr,
        cond_end_arr,
        p_arr,
        T_arr,
        h_arr,
        y_arr,
        Z_L_arr,
        Z_G_arr,
        x_h2o_L_arr,
        x_h2o_G_arr,
        x_co2_L_arr,
        x_co2_G_arr,
        phi_h2o_L_arr,
        phi_h2o_G_arr,
        phi_co2_L_arr,
        phi_co2_G_arr,
    ) = arrays_loc

    MIX, ADS, FLASH = get_flash_setup(num_cells=1)
    LIQ, GAS = [phase for phase in MIX.phases]
    H2O, CO2 = [comp for comp in MIX.components]

    p_vec = np.array([p], dtype=np.double) * 1e-6
    T_vec = np.array([T])
    h_vec = np.zeros(1)

    # setting thermodynamic state, feed fractions assumed to be set
    ADS.set_variable_values(
        p_vec, variables=[MIX.AD.p.name], to_iterate=True, to_state=True,
    )
    ADS.set_variable_values(
        T_vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
    )
    ADS.set_variable_values(
        h_vec, variables=[MIX.AD.h.name], to_iterate=True, to_state=True
    )

    try:
        success_ = FLASH.flash(
            flash_type="pT",
            method="npipm",
            initial_guess="rachford_rice",
            copy_to_state=True,  # don't overwrite the state, store as iterate
            do_logging=False,
        )
    except Exception as err:  # if Flasher fails, flag as failed
        print(f"\nParallel flash: failed at {ipT}\n{str(err)}\n", flush=True)
        success_ = False

    # always available
    num_iter_arr[i] = FLASH.flash_history[-1]["iterations"]
    cond_start_arr[i] = FLASH.cond_start
    cond_end_arr[i] = FLASH.cond_end
    p_arr[i] = p
    T_arr[i] = T

    if success_:
        try:
            MIX.precompute(apply_smoother=False)
            FLASH.evaluate_specific_enthalpy(True)
        except Exception:
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L_arr[i] = FAILURE_ENTRY
            Z_G_arr[i] = FAILURE_ENTRY
            phi_h2o_L_arr[i] = FAILURE_ENTRY
            phi_co2_L_arr[i] = FAILURE_ENTRY
            phi_h2o_G_arr[i] = FAILURE_ENTRY
            phi_co2_G_arr[i] = FAILURE_ENTRY
            h_arr[i] = FAILURE_ENTRY
            is_supercrit_arr[i] = 0
        else:
            # if successful, store values
            Z_L_arr[i] = LIQ.eos.Z.val[0]
            Z_G_arr[i] = GAS.eos.Z.val[0]
            phi_h2o_L_arr[i] = LIQ.eos.phi[H2O].val[0]
            phi_co2_L_arr[i] = LIQ.eos.phi[CO2].val[0]
            phi_h2o_G_arr[i] = LIQ.eos.phi[H2O].val[0]
            phi_co2_G_arr[i] = LIQ.eos.phi[CO2].val[0]
            h_arr[i] = MIX.AD.h.evaluate(ADS).val[0]
            # TODO normally both phases should have the same boolean value here
            is_supercrit_arr[i] = int(LIQ.eos.is_supercritical[0] or GAS.eos.is_supercritical[0])

            

        # extract and store results from last iterate
        success_arr[i] = 1
        y_arr[i] = GAS.fraction.evaluate(ADS).val[0]
        x_h2o_L_arr[i] = LIQ.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_h2o_G_arr[i] = GAS.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_co2_L_arr[i] = LIQ.fraction_of_component(CO2).evaluate(ADS).val[0]
        x_co2_G_arr[i] = GAS.fraction_of_component(CO2).evaluate(ADS).val[0]

    else:
        Z_L_arr[i] = FAILURE_ENTRY
        Z_G_arr[i] = FAILURE_ENTRY
        phi_h2o_L_arr[i] = FAILURE_ENTRY
        phi_h2o_G_arr[i] = FAILURE_ENTRY
        phi_co2_L_arr[i] = FAILURE_ENTRY
        phi_co2_G_arr[i] = FAILURE_ENTRY
        h_arr[i] = FAILURE_ENTRY

        success_arr[i] = FAILURE_ENTRY
        y_arr[i] = FAILURE_ENTRY
        x_h2o_L_arr[i] = FAILURE_ENTRY
        x_h2o_G_arr[i] = FAILURE_ENTRY
        x_co2_L_arr[i] = FAILURE_ENTRY
        x_co2_G_arr[i] = FAILURE_ENTRY

    progress_queue_loc.put(i)


def parallel_ph_flash(iph):
    """Performs a p-h flash (including modelling) and stores the results in shared
    memory.
    
    ``iph`` must be a tuple containing the index where the results should be stored,
    and the p-h point.

    Warning:
        There are some unresolved issues with parallel subprocesses if
        numpy/scipy/pypardiso throw errors or warnings. It causes the respective
        subprocess unable to join, despite finishing all flashes.

    """

    i, p, h = iph

    FAILURE_ENTRY = NAN_ENTRY

    # accessing shared memory
    global arrays_loc, progress_queue_loc
    (
        success_arr,
        num_iter_arr,
        is_supercrit_arr,
        cond_start_arr,
        cond_end_arr,
        p_arr,
        T_arr,
        h_arr,
        y_arr,
        Z_L_arr,
        Z_G_arr,
        x_h2o_L_arr,
        x_h2o_G_arr,
        x_co2_L_arr,
        x_co2_G_arr,
        phi_h2o_L_arr,
        phi_h2o_G_arr,
        phi_co2_L_arr,
        phi_co2_G_arr,
    ) = arrays_loc

    MIX, ADS, FLASH = get_flash_setup(num_cells=1)
    LIQ, GAS = [phase for phase in MIX.phases]
    H2O, CO2 = [comp for comp in MIX.components]

    p_vec = np.array([p], dtype=np.double) * 1e-6
    T_vec = np.zeros(1)
    h_vec = np.array([h])

    # setting thermodynamic state, feed fractions assumed to be set
    ADS.set_variable_values(
        p_vec, variables=[MIX.AD.p.name], to_iterate=True, to_state=True,
    )
    ADS.set_variable_values(
        T_vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
    )
    ADS.set_variable_values(
        h_vec, variables=[MIX.AD.h.name], to_iterate=True, to_state=True
    )

    try:
        success_ = FLASH.flash(
            flash_type="ph",
            method="npipm",
            initial_guess="rachford_rice",
            copy_to_state=True,  # don't overwrite the state, store as iterate
            do_logging=False,
        )
    except Exception as err:  # if Flasher fails, flag as failed
        print(f"\nParallel flash: failed at {iph}\n{str(err)}\n", flush=True)
        success_ = False

    # always available
    num_iter_arr[i] = FLASH.flash_history[-1]["iterations"]
    cond_start_arr[i] = FLASH.cond_start
    cond_end_arr[i] = FLASH.cond_end
    p_arr[i] = p
    h_arr[i] = h

    if success_:
        try:
            MIX.precompute(apply_smoother=False)
            FLASH.evaluate_specific_enthalpy(True)
        except Exception:
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L_arr[i] = FAILURE_ENTRY
            Z_G_arr[i] = FAILURE_ENTRY
            phi_h2o_L_arr[i] = FAILURE_ENTRY
            phi_co2_L_arr[i] = FAILURE_ENTRY
            phi_h2o_G_arr[i] = FAILURE_ENTRY
            phi_co2_G_arr[i] = FAILURE_ENTRY
            is_supercrit_arr[i] = 0
        else:
            # if successful, store values
            Z_L_arr[i] = LIQ.eos.Z.val[0]
            Z_G_arr[i] = GAS.eos.Z.val[0]
            phi_h2o_L_arr[i] = LIQ.eos.phi[H2O].val[0]
            phi_co2_L_arr[i] = LIQ.eos.phi[CO2].val[0]
            phi_h2o_G_arr[i] = LIQ.eos.phi[H2O].val[0]
            phi_co2_G_arr[i] = LIQ.eos.phi[CO2].val[0]
            # TODO normally both phases should have the same boolean value here
            is_supercrit_arr[i] = int(LIQ.eos.is_supercritical[0] or GAS.eos.is_supercritical[0])

            

        # extract and store results from last iterate
        T_arr[i] = MIX.AD.T.evaluate(ADS).val[0]
        success_arr[i] = 1
        y_arr[i] = GAS.fraction.evaluate(ADS).val[0]
        x_h2o_L_arr[i] = LIQ.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_h2o_G_arr[i] = GAS.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_co2_L_arr[i] = LIQ.fraction_of_component(CO2).evaluate(ADS).val[0]
        x_co2_G_arr[i] = GAS.fraction_of_component(CO2).evaluate(ADS).val[0]

    else:
        Z_L_arr[i] = FAILURE_ENTRY
        Z_G_arr[i] = FAILURE_ENTRY
        phi_h2o_L_arr[i] = FAILURE_ENTRY
        phi_h2o_G_arr[i] = FAILURE_ENTRY
        phi_co2_L_arr[i] = FAILURE_ENTRY
        phi_co2_G_arr[i] = FAILURE_ENTRY
        T_arr[i] = FAILURE_ENTRY

        success_arr[i] = FAILURE_ENTRY
        y_arr[i] = FAILURE_ENTRY
        x_h2o_L_arr[i] = FAILURE_ENTRY
        x_h2o_G_arr[i] = FAILURE_ENTRY
        x_co2_L_arr[i] = FAILURE_ENTRY
        x_co2_G_arr[i] = FAILURE_ENTRY

    progress_queue_loc.put(i)


if __name__ == '__main__':

    if FLASH_TYPE == 'pT':
        p_points, x_points, pT_id = read_px_data([PX_DATA_FILE], 'T')
    elif FLASH_TYPE == 'ph':
        p_points, x_points, pT_id = read_px_data([PX_DATA_FILE], 'h')
    else:
        NotImplementedError(f"Unknown flash type: {FLASH_TYPE}.")
    
    # sanity check
    assert len(p_points) == len(x_points
    ), f"Incomplete set of p-T points from file {PX_DATA_FILE}"
    # number of p-T points, i.e. flashes to perform
    NC: int = len(p_points)

    results: dict[str, list]

    if MODE == 0:
        pass
    elif MODE == 1:
        pass
    elif MODE == 2:
        if FLASH_TYPE == 'pT':
            flash_func = parallel_pT_flash
        elif FLASH_TYPE == 'ph':
            flash_func = parallel_ph_flash
        else:
            raise AssertionError("Something went terribly wrong")
        # list of p-T points and which entry they should fill in the shared arrays
        ipx = [(i, p, x) for i, p, x in zip(np.arange(NC), p_points, x_points)]
        shared_arrays = _create_shared_arrays(NC)
        print("Parallel flash: starting ... ", flush=True)
        start_time = time.time()
        # multiprocessing.set_start_method('fork')
        prog_q = Queue(maxsize=NC)
        with Pool(
            processes=NUM_PHYS_CPU_CORS,
            initargs=(shared_arrays, prog_q),
            initializer=_access_shared_objects
        ) as pool:
            prog_process = Process(
                target=_progress_counter,
                args=(prog_q, NC),
                daemon=True
            )
            chunksize = NUM_PHYS_CPU_CORS
            result = pool.map_async(flash_func, ipx, chunksize=chunksize)

            # Wait until all results are here
            prog_process.start()
            prog_process.join()
            # Wait for some time and see if processes terminate as they should
            # we terminate if the processes for some case could not finish
            result.wait(3)
            if result.ready() and result.successful():
                pool.close()
            else:
                print(f"\nParallel flash: terminated", flush=True)
                pool.close()
                pool.terminate()
            pool.join()

        end_time = time.time()
        print(f"\nParallel flash: finished after {end_time - start_time} seconds.", flush=True)

        result_vecs = [
            list(np.frombuffer(vec.get_obj(), dtype=dtype))
            for vec, dtype in shared_arrays
        ]
        results = dict(
            [(header, vec) for header, vec in zip(get_result_headers(), result_vecs)]
        )
    else:
        raise NotImplementedError(f"Flash mode {MODE} not supported. Use (0, 1, 2).")

    write_results(RESULT_FILE, results)
