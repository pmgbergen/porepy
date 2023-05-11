"""Script to invoke PorePy's unified flash procedure."""
import pathlib
import sys
import time
from ctypes import c_double, c_uint8
from multiprocessing import Array, Pool, Process, Queue
from multiprocessing.pool import AsyncResult
from typing import Any

import numpy as np
import psutil

import porepy as pp

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from thermo_comparison import (
    COMPONENTS,
    FEED,
    NAN_ENTRY,
    PHASES,
    T_HEADER,
    composition_HEADER,
    compressibility_HEADER,
    cond_end_HEADER,
    cond_start_HEADER,
    fugacity_HEADER,
    gas_frac_HEADER,
    get_result_headers,
    h_HEADER,
    is_supercrit_HEADER,
    num_iter_HEADER,
    p_HEADER,
    read_px_data,
    success_HEADER,
    write_results,
)

# How to calculate:
# 0 - point-wise
# 1 - vectorized  NOTE: Not implemented
# 2 - parallelized
# This is critical when it comes to performance on big data files
MODE: int = 2
# flash type: pT or ph
FLASH_TYPE: str = "pT"
# p-x data from an thermo results file
PX_DATA_FILE = f"data\\thermodata_pT2k_co2_1e-2.csv"
# file to which to write the results
RESULT_FILE = f"data\\results\\results_pT2k_co2_1e-2_par.csv"
# Number of physical CPU cores.
# This is used for the number of sub-processes and chunksize in the parallelization
NUM_PHYS_CPU_CORS = psutil.cpu_count(logical=False)


def _access_shared_objects(
    shared_arrays: list[tuple[Array, Any]], progress_queue: Queue
):
    """Helper function to be called by subprocesses to provide access to shared-memory
    objects.

    ``sharred_arrays`` must be a list of tuples, where the first entry is the
    shared array object and the second entry its data type (as a C-type).

    """
    global arrays_loc, progress_queue_loc

    progress_queue_loc = progress_queue
    # # access locally as np arrays
    # arrays_loc = [
    #     np.frombuffer(vec.get_obj(), dtype=dtype) for vec, dtype in shared_arrays
    # ]
    # # access locally as Array from multiprocessing
    arrays_loc = [
        vec for vec, _ in shared_arrays
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


def _progress_counter(q: Queue, NC: int, flash_result: AsyncResult):
    """A function sharing a Queue object with other processes, which receives
    the index of all finished flashes. This function finishes when all indices
    where received."""
    progress_array = Array(typecode_or_type=c_uint8, size_or_initializer=NC)

    while True:
        i = q.get()
        progress_array[i] = 1
        progress = int(sum(progress_array))
        print(f"\rParallel flash: {progress}/{NC}", end="", flush=True)
        if progress == NC:
            break
        # if flash_result.ready():
        #     print(f"\rParallel flash: results ready", end='', flush=True)
        #     break


def get_flash_setup(
    num_cells: int,
) -> tuple[pp.composite.NonReactiveMixture, pp.ad.EquationSystem, pp.composite.FlashNR]:
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

    MIX = pp.composite.NonReactiveMixture(nc=num_cells)
    ADS = MIX.AD.system

    # components
    H2O = pp.composite.H2O(ADS)
    CO2 = pp.composite.CO2(ADS)
    # phases
    LIQ = pp.composite.PR_Phase(ADS, False, name="L")
    GAS = pp.composite.PR_Phase(ADS, True, name="G")

    MIX.set([H2O, CO2], [LIQ, GAS])

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
    FLASH = pp.composite.FlashNR(MIX, auxiliary_npipm=False)
    FLASH.use_armijo = True
    FLASH.armijo_parameters["rho"] = 0.99
    FLASH.armijo_parameters["j_max"] = 55  # cap the number of Armijo iterations
    FLASH.armijo_parameters[
        "return_max"
    ] = True  # return max Armijo iter, even if not min
    FLASH.tolerance = 1e-8
    FLASH.max_iter = 140

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
        x_h2o_G_arr,
        x_h2o_L_arr,
        x_co2_G_arr,
        x_co2_L_arr,
        phi_h2o_G_arr,
        phi_h2o_L_arr,
        phi_co2_G_arr,
        phi_co2_L_arr,
    ) = arrays_loc

    MIX, ADS, FLASH = get_flash_setup(num_cells=1)
    LIQ, GAS = [phase for phase in MIX.phases]
    H2O, CO2 = [comp for comp in MIX.components]

    p_vec = np.array([p], dtype=np.double) * 1e-6
    T_vec = np.array([T])
    h_vec = np.zeros(1)

    # setting thermodynamic state, feed fractions assumed to be set
    ADS.set_variable_values(
        p_vec,
        variables=[MIX.AD.p.name],
        to_iterate=True,
        to_state=True,
    )
    ADS.set_variable_values(
        T_vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
    )
    ADS.set_variable_values(
        h_vec, variables=[MIX.AD.h.name], to_iterate=True, to_state=True
    )
    p_arr[i] = p
    T_arr[i] = T

    try:
        success_ = FLASH.flash(
            flash_type="pT",
            method="npipm",
            initial_guess="rachford_rice",
            store_to_iterate=True,  # don't overwrite the state, store as iterate
            verbosity=False,
        )
    except Exception as err:  # if Flasher fails, flag as failed
        print(f"\nParallel flash: failed at {ipT}\n{str(err)}\n", flush=True)
        success_ = False

    if success_:
        try:
            MIX.AD.compute_properties_from_state(apply_smoother=False)
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
            is_supercrit_arr[i] = int(
                LIQ.eos.is_supercritical[0] and GAS.eos.is_supercritical[0]
            )

        # extract and store results from last iterate
        success_arr[i] = 1
        y_arr[i] = GAS.fraction.evaluate(ADS).val[0]
        x_h2o_L_arr[i] = LIQ.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_h2o_G_arr[i] = GAS.fraction_of_component(H2O).evaluate(ADS).val[0]
        x_co2_L_arr[i] = LIQ.fraction_of_component(CO2).evaluate(ADS).val[0]
        x_co2_G_arr[i] = GAS.fraction_of_component(CO2).evaluate(ADS).val[0]

        num_iter_arr[i] = FLASH.history[-1]["iterations"]
        cond_start_arr[i] = FLASH.cond_start
        cond_end_arr[i] = FLASH.cond_end

    else:
        Z_L_arr[i] = FAILURE_ENTRY
        Z_G_arr[i] = FAILURE_ENTRY
        phi_h2o_L_arr[i] = FAILURE_ENTRY
        phi_h2o_G_arr[i] = FAILURE_ENTRY
        phi_co2_L_arr[i] = FAILURE_ENTRY
        phi_co2_G_arr[i] = FAILURE_ENTRY
        h_arr[i] = FAILURE_ENTRY

        success_arr[i] = 0
        is_supercrit_arr[i] = 0
        y_arr[i] = FAILURE_ENTRY
        x_h2o_L_arr[i] = FAILURE_ENTRY
        x_h2o_G_arr[i] = FAILURE_ENTRY
        x_co2_L_arr[i] = FAILURE_ENTRY
        x_co2_G_arr[i] = FAILURE_ENTRY

        num_iter_arr[i] = FAILURE_ENTRY
        cond_start_arr[i] = FAILURE_ENTRY
        cond_end_arr[i] = FAILURE_ENTRY

    progress_queue_loc.put(i, block=False)


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
        p_vec,
        variables=[MIX.AD.p.name],
        to_iterate=True,
        to_state=True,
    )
    ADS.set_variable_values(
        T_vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
    )
    ADS.set_variable_values(
        h_vec, variables=[MIX.AD.h.name], to_iterate=True, to_state=True
    )

    p_arr[i] = p
    h_arr[i] = h

    try:
        success_ = FLASH.flash(
            flash_type="ph",
            method="npipm",
            initial_guess="rachford_rice",
            store_to_iterate=True,  # don't overwrite the state, store as iterate
            verbosity=False,
        )
    except Exception as err:  # if Flasher fails, flag as failed
        print(f"\nParallel flash: failed at {iph}\n{str(err)}\n", flush=True)
        success_ = False

    # always available
    num_iter_arr[i] = FLASH.history[-1]["iterations"]
    cond_start_arr[i] = FLASH.cond_start
    cond_end_arr[i] = FLASH.cond_end

    if success_:
        try:
            MIX.AD.compute_properties_from_state(apply_smoother=False)
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
            is_supercrit_arr[i] = int(
                LIQ.eos.is_supercritical[0] or GAS.eos.is_supercritical[0]
            )

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

        success_arr[i] = 0
        is_supercrit_arr[i] = 0
        y_arr[i] = FAILURE_ENTRY
        x_h2o_L_arr[i] = FAILURE_ENTRY
        x_h2o_G_arr[i] = FAILURE_ENTRY
        x_co2_L_arr[i] = FAILURE_ENTRY
        x_co2_G_arr[i] = FAILURE_ENTRY

    progress_queue_loc.put(i)


def pointwise_pT_flash(p_points: list[float], T_points: list[float]) -> dict[str, list]:
    """Performs point-wise a p-T flash for p and T in passed lists.

    Returns the results stored in a dictionary with column headers as keys.

    """
    NC = len(p_points)
    MIX, ADS, FLASH = get_flash_setup(num_cells=1)
    LIQ, GAS = [p for p in MIX.phases]
    H2O, CO2 = [c for c in MIX.components]

    ADS.set_variable_values(
        np.zeros(1), [MIX.AD.h.name], to_iterate=True, to_state=True
    )

    # result storage
    success: list[int] = list()  # flag if flash succeeded
    num_iter: list[int] = list()  # number of iterations
    cond_start: list[float] = list()
    cond_end: list[float] = list()
    is_supercrit: list[int] = list()
    h: list[float] = list()
    y: list[float] = list()
    Z_L: list[float] = list()
    Z_G: list[float] = list()
    x_h2o_L: list[float] = list()
    x_co2_L: list[float] = list()
    x_h2o_G: list[float] = list()
    x_co2_G: list[float] = list()
    phi_h2o_L: list[float] = list()
    phi_co2_L: list[float] = list()
    phi_h2o_G: list[float] = list()
    phi_co2_G: list[float] = list()

    print(f"Point-wise flash: starting ... ", flush=True)
    for i, pT in enumerate(zip(p_points, T_points)):
        p, T = pT
        ADS.set_variable_values(np.array([p]) * 1e-6, [MIX.AD.p.name], True, True)
        ADS.set_variable_values(np.array([T]), [MIX.AD.T.name], True, True)

        try:
            print(f"\rPoint-wise flash: flash {i + 1}/{NC}", end="", flush=True)
            success_ = FLASH.flash(
                flash_type="pT",
                method="npipm",
                initial_guess="rachford_rice",
                store_to_iterate=False,  # don't overwrite the state, store as iterate
                verbosity=False,
            )
        except Exception as err:  # if Flasher fails, flag as failed
            print(f"\rPoint-wise flash: failed at {i}\n{str(err)}", flush=True)
            success_ = False

        try:
            MIX.AD.compute_properties_from_state(apply_smoother=False)
            FLASH.evaluate_specific_enthalpy(True)
        except Exception as err:
            print(
                f"\rPoint-wise flash: EOS evaluation failed at {i}\n{str(err)}",
                flush=True,
            )
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L.append(np.nan)
            Z_G.append(np.nan)
            phi_h2o_L.append(np.nan)
            phi_h2o_G.append(np.nan)
            phi_co2_L.append(np.nan)
            phi_co2_G.append(np.nan)
            h.append(np.nan)
            is_supercrit.append(0)
        else:
            # if successful, store values
            Z_L.append(LIQ.eos.Z.val[0])
            Z_G.append(GAS.eos.Z.val[0])
            phi_h2o_L.append(LIQ.eos.phi[H2O].val[0])
            phi_h2o_G.append(GAS.eos.phi[H2O].val[0])
            phi_co2_L.append(LIQ.eos.phi[CO2].val[0])
            phi_co2_G.append(GAS.eos.phi[CO2].val[0])
            h.append(MIX.AD.h.evaluate(ADS).val[0])
            is_supercrit.append(
                int(LIQ.eos.is_supercritical[0] and GAS.eos.is_supercritical[0])
            )

        # extract and store results from last iterate
        success.append(int(success_))  # store booleans as 0 and 1
        y.append(GAS.fraction.evaluate(ADS).val[0])
        x_h2o_L.append(LIQ.fraction_of_component(H2O).evaluate(ADS).val[0])
        x_co2_L.append(LIQ.fraction_of_component(CO2).evaluate(ADS).val[0])
        x_h2o_G.append(GAS.fraction_of_component(H2O).evaluate(ADS).val[0])
        x_co2_G.append(GAS.fraction_of_component(CO2).evaluate(ADS).val[0])
        num_iter.append(FLASH.history[-1]["iterations"])
        cond_start.append(FLASH.cond_start)
        cond_end.append(FLASH.cond_end)

    results: dict[str, int] = dict()

    results[p_HEADER] = p_points
    results[T_HEADER] = T_points
    results[h_HEADER] = h

    results[success_HEADER] = success
    results[num_iter_HEADER] = num_iter
    results[cond_start_HEADER] = cond_start
    results[cond_end_HEADER] = cond_end
    results[is_supercrit_HEADER] = is_supercrit
    results[gas_frac_HEADER] = y

    results[compressibility_HEADER[PHASES[0]]] = Z_G
    results[compressibility_HEADER[PHASES[1]]] = Z_L

    h2o = COMPONENTS[0]
    co2 = COMPONENTS[1]
    gas = PHASES[0]
    liq = PHASES[1]

    results[composition_HEADER[h2o][gas]] = x_h2o_G
    results[composition_HEADER[h2o][liq]] = x_h2o_L
    results[composition_HEADER[co2][gas]] = x_co2_G
    results[composition_HEADER[co2][liq]] = x_co2_L

    results[fugacity_HEADER[h2o][gas]] = phi_h2o_G
    results[fugacity_HEADER[h2o][liq]] = phi_h2o_L
    results[fugacity_HEADER[co2][gas]] = phi_co2_G
    results[fugacity_HEADER[co2][liq]] = phi_co2_L

    return results


if __name__ == "__main__":

    if FLASH_TYPE == "pT":
        p_points, x_points, pT_id = read_px_data([PX_DATA_FILE], "T")
    elif FLASH_TYPE == "ph":
        p_points, x_points, pT_id = read_px_data([PX_DATA_FILE], "h")
    else:
        NotImplementedError(f"Unknown flash type: {FLASH_TYPE}.")

    # sanity check
    assert len(p_points) == len(
        x_points
    ), f"Incomplete set of p-T points from file {PX_DATA_FILE}"
    # number of p-T points, i.e. flashes to perform
    NC: int = len(p_points)

    results: dict[str, list]

    if MODE == 0:
        if FLASH_TYPE == "pT":
            results = pointwise_pT_flash(p_points=p_points, T_points=x_points)
        elif FLASH_TYPE == "ph":
            pass
    elif MODE == 1:
        pass
    elif MODE == 2:
        if FLASH_TYPE == "pT":
            flash_func = parallel_pT_flash
        elif FLASH_TYPE == "ph":
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
            initializer=_access_shared_objects,
        ) as pool:

            chunksize = NUM_PHYS_CPU_CORS
            result = pool.map_async(flash_func, ipx, chunksize=chunksize)

            prog_process = Process(
                target=_progress_counter, args=(prog_q, NC, None), daemon=True
            )

            # Wait until all results are here
            prog_process.start()
            # Wait for some time and see if processes terminate as they should
            # we terminate if the processes for some case could not finish
            result.wait(60 * 60 * 5)
            if result.ready():
                prog_process.join(5)
                if prog_process.exitcode != 0:
                    prog_process.terminate()
                pool.close()
            else:
                prog_process.terminate()
                print(f"\nParallel flash: terminated", flush=True)
                pool.close()
                pool.terminate()
            pool.join()

        end_time = time.time()
        print(
            f"\nParallel flash: finished after {end_time - start_time} seconds.",
            flush=True,
        )

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