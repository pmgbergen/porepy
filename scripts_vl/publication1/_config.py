"""Configuration file containing data range values and functionality for calculating
and plotting data."""
from __future__ import annotations

import csv
import logging
import pathlib
import sys
import time
from ctypes import c_char_p, c_double, c_uint8
from multiprocessing import Array, Pool, Process, Queue
from multiprocessing.pool import AsyncResult
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib import figure
from thermo import PR78MIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVLN
from thermo.interaction_parameters import IPDB

import porepy as pp

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

# figure configurations
FIGURE_WIDTH: int = 15  # in inches, 1080 / 1920 ratio applied to height
DPI: int = 400  # Dots per Inch (level of detail per figure)

# fluid mixture configuration
SPECIES: list[str] = ["H2O", "CO2"]
FEED: list[float] = [0.99, 0.01]

# pressure and temperature limits for calculations
P_LIMITS: list[float] = [1e6, 50e6]  # [Pa]
T_LIMITS: list[float] = [450.0, 700.0]  # [K]
# temperature values for isotherms (isenthalpic flash calculations)
ISOTHERMS: list[float] = [500.0, 600, 625.0, 640.0, 645.0, 650.0]
P_LIMITS_ISOTHERMS: list[float] = [1e6, 20e6]
# resolution of p-T limits
RESOLUTION_pT: int = 50
# pressure resolution along isotherms
RESOLUTION_isotherms: int = 25

# Scaling for plots
PRESSURE_SCALE: float = 1e-6  # to MPa
PRESSURE_SCALE_NAME: str = "MPa"

# Calculation modus for PorePy flash
# 1 - point-wise (robust, but possibly very slow),
# 2 - vectorized (not recommended),
# 3 - parallelized (use with care, if system compatible)
CALCULATION_MODE: int = 3

# paths to where results should be stored
THERMO_DATA_PATH: str = "data/thermodata.csv"  # storing results from therm
PT_FLASH_DATA_PATH: str = "data/flash_pT.csv"  # storing p-T results from porepy
PT_QUICKSHOT_DATA_PATH: str = (
    "data/flash_pT.csv"  # storing p-T results from initial guess
)
ISOTHERM_DATA_PATH: str = (
    "data/flash_pT_isotherms.csv"  # storing p-T results on isotherms
)
PH_FLASH_DATA_PATH: str = "data/flash_ph.csv"  # storing p-h results from porepy
FIG_PATH: str = "figs/"  # path to folder containing plots

NUM_COMP: int = len(SPECIES)  # number of components
DELIMITER: str = ","  # delimiter in result files
# entry for a points which are missing, have no meaning or the flash failed
NAN_ENTRY: np.nan = np.nan
# Phase labels, G L1 L2 ...
PHASES: list[str] = ["G"] + [f"L{i}" for i in range(1, NUM_COMP + 1)]
# headers in result files
success_HEADER: str = "success"
num_iter_HEADER: str = "num-iter"
conditioning_HEADER: str = "condition-number"
phases_HEADER: str = "phase-split"
p_HEADER: str = "p [Pa]"
T_HEADER: str = "T [K]"
h_HEADER: str = "h [J/mol]"
gas_frac_HEADER: str = "y"
liq_frac_HEADER: list[str] = [f"y_L{j}" for j in range(1, NUM_COMP + 1)]
compressibility_HEADER: dict[str, str] = dict([(f"{j}", f"Z_{j}") for j in PHASES])
composition_HEADER: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"x_{i}_{j}") for j in PHASES])) for i in SPECIES]
)
fugacity_HEADER: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"phi_{i}_{j}") for j in PHASES])) for i in SPECIES]
)

# Number of physical CPU cores.
# This is used for the number of sub-processes and chunksize in the parallelization
NUM_PHYS_CPU_CORS = psutil.cpu_count(logical=False)


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.terminator = ""
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

del_log = "\r" + " " * 100 + "\r"


def path():
    """Returns path to script calling this function as string."""
    return str(pathlib.Path(__file__).parent.resolve())


def read_px_data(
    file_path: str,
    x: str,
) -> tuple[list[float], list[float]]:
    """Reads pressure and x-data from a file.

    x can be ``'T'`` or ``'h'``.

    Returns two list containing the pressure- and x-data, as given in the file
    (pairs of p-x).

    """
    p_points: list[float] = list()
    x_points: list[float] = list()

    px_double: list[tuple] = list()  # to filter out double entries

    if x == "T":
        x_HEADER = T_HEADER
    elif x == "h":
        x_HEADER = h_HEADER
    else:
        raise NotImplementedError(f"Unknown x for p-x data: {x}")

    logger.info(f"{del_log}Reading p-{x} data: file {file_path}")
    with open(f"{path()}/{file_path}", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER)
        headers = next(reader)  # get rid of header

        p_idx = headers.index(p_HEADER)
        x_idx = headers.index(x_HEADER)

        for row in reader:

            # must always be available
            p = float(row[p_idx])
            # potentially missing or nan
            x_ = row[x_idx]
            if x_ not in [NAN_ENTRY, str(NAN_ENTRY)]:
                x_ = float(x_)
            else:
                x_ = NAN_ENTRY

            px = (p, x_)
            # get only unique points
            if px not in px_double:
                px_double.append(px)
                p_points.append(p)
                x_points.append(x_)
    logger.info(f"{del_log}Reading p-{x} data: done\n")

    # sanity check to see if nothing messed with the precision during I/O
    # and data is readable
    test = list()
    mapping_failure = list()
    for p, x_ in zip(p_points, x_points):
        try:
            test.append(px_double.index((p, x_)))
        except KeyError:
            mapping_failure.append((p, x_))

    if not (len(test) == len(p_points) == len(x_points)):
        logger.warn(f"\np-{x} mapping failed\n")
    if not (len(mapping_failure) == 0):
        logger.warn(f"\np-{x} mapping failed for points: {mapping_failure}\n")

    return p_points, x_points


def create_index_map(x: list[float], y: list[float]) -> dict[tuple[float, float], int]:
    """Creates a index map to associate a tuple of data found in x and y with their
    index.

    Use this in combination with ``read_px_data`` to access specific values in a results
    data structure created by ``read_results``.

    """
    return dict([((xy[0], xy[1]), i) for i, xy in enumerate(zip(x, y))])


def read_results(file_name: str, headers: list[str] | None = None) -> dict[str, list]:
    """Reads data previously written with ``write_results``.

    Use ``headers`` to specify which header to read.
    Otherwise all columns are read.

    The returned dictionary contains per header the data column.

    """

    logger.info(f"{del_log}Reading result data: file {file_name}")
    with open(f"{path()}/{file_name}", mode="r") as file:
        reader = csv.reader(file, delimiter=DELIMITER)

        file_headers = next(reader)
        if headers is None:
            headers = file_headers

        data: dict[str, list] = dict([(head, list()) for head in headers])
        header_idx: dict[str, int] = dict(
            [(head, file_headers.index(head)) for head in headers]
        )

        for row in reader:
            for head in headers:
                idx = header_idx[head]
                d = row[idx]
                data[head].append(d)
    logger.info(f"{del_log}Reading result data: done\n")

    return data


def write_results(filename: str, results: dict[str, list]):
    """Writes results to file. Results must be a dictionary with headers as keys
    and data columns as values."""
    headers = [header for header, _ in results.items()]
    data = [col for _, col in results.items()]

    logger.info(f"{del_log}Writing result data: file {filename}")
    with open(f"{path()}/{filename}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=DELIMITER)
        # wrting header
        writer.writerow(headers)
        # writing data
        for row in zip(*data):
            writer.writerow(row)
    logger.info(f"{del_log}Writing result data: done\n")


def _thermo_init() -> FlashVLN:
    """Helper function to initiate the thermo flash."""
    constants, properties = ChemicalConstantsPackage.from_IDs(SPECIES)
    kijs = IPDB.get_ip_asymmetric_matrix("ChemSep PR", constants.CASs, "kij")
    eos_kwargs = {
        "Pcs": constants.Pcs,
        "Tcs": constants.Tcs,
        "omegas": constants.omegas,
        "kijs": kijs,
    }

    GAS = CEOSGas(
        PR78MIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases
    )
    LIQs = [
        CEOSLiquid(
            PR78MIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        for _ in range(NUM_COMP)
    ]
    flasher = FlashVLN(constants, properties, liquids=LIQs, gas=GAS)

    return flasher


def _init_empty_results() -> dict[str, list]:
    """Initiate and return an results dict with proper headers as needed for the
    comparison."""
    results: dict[str, list] = {
        success_HEADER: list(),
        num_iter_HEADER: list(),
        conditioning_HEADER: list(),
        p_HEADER: list(),
        T_HEADER: list(),
        h_HEADER: list(),
        phases_HEADER: list(),
        gas_frac_HEADER: list(),
    }
    results.update(dict([(liq_frac_HEADER[i], list()) for i in range(NUM_COMP)]))
    results.update(dict([(compressibility_HEADER[j], list()) for j in PHASES]))
    results.update(
        dict([(composition_HEADER[i][j], list()) for j in PHASES for i in SPECIES])
    )
    results.update(
        dict([(fugacity_HEADER[i][j], list()) for j in PHASES for i in SPECIES])
    )

    return results


def _thermo_parse_result(state) -> dict:
    """Helper function to parse a state returned by thermo into processable format."""
    out = _init_empty_results()
    for k in out.keys():
        out[k] = NAN_ENTRY

    out.update(
        {
            success_HEADER: 1,
            gas_frac_HEADER: state.VF,
        }
    )
    # anticipate at max only 1 gas phase and predefined number of liquid phases
    if 0 < state.phase_count <= 1 + NUM_COMP:
        if (
            0.0 < state.VF <= 1.0
        ) and state.gas is not None:  # parse gas phase if present
            j = PHASES[0]
            out.update(
                {
                    phases_HEADER: j,
                    compressibility_HEADER[j]: state.gas.Z(),
                }
            )
            out.update(
                dict(
                    [
                        (composition_HEADER[i][j], state.gas.zs[SPECIES.index(i)])
                        for i in SPECIES
                    ]
                    + [
                        (fugacity_HEADER[i][j], state.gas.phis()[SPECIES.index(i)])
                        for i in SPECIES
                    ]
                )
            )
        # should not happen
        elif (state.gas is None and (0.0 < state.VF <= 1.0)) or (
            state.gas is not None and state.VF == 0.0
        ):
            raise NotImplementedError(
                f"Uncovered thermo phase case: conflicting gas state"
            )
        else:  # gas phase is not present, store nans
            assert state.VF == 0.0, "Uncovered thermo gas phase state."
            j = PHASES[0]
            out.update(
                {
                    phases_HEADER: "",
                    compressibility_HEADER[j]: NAN_ENTRY,
                }
            )
            out.update(
                dict(
                    [(composition_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                    + [(fugacity_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                )
            )

        if (
            state.gas is not None and state.VF == 1.0
        ):  # if only gas, fill liquid entries with nans
            # for anticipated liquid phases
            for liq_frac in liq_frac_HEADER:
                out.update({liq_frac: 0.0})
            for j in PHASES[1:]:
                out.update(
                    {
                        compressibility_HEADER[j]: NAN_ENTRY,
                    }
                )
                out.update(
                    dict(
                        [(composition_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                        + [(fugacity_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                    )
                )
        else:  # parse present liquid phases
            # sanity check
            assert (
                state.VF < 1.0
            ), "Thermo conflicting gas state: Gas saturated with liquid phases"
            if len(state.liquids) == 1:  # if only one liquid phase
                out[phases_HEADER] = out[phases_HEADER] + "L"
                j = PHASES[1]
                out.update({liq_frac_HEADER[0]: 1 - state.VF})
                for yl in liq_frac_HEADER[1:]:
                    out.update({yl: NAN_ENTRY})
                out.update(
                    {
                        compressibility_HEADER[j]: state.liquids[0].Z(),
                    }
                )
                out.update(
                    dict(
                        [
                            (
                                composition_HEADER[i][j],
                                state.liquids[0].zs[SPECIES.index(i)],
                            )
                            for i in SPECIES
                        ]
                        + [
                            (
                                fugacity_HEADER[i][j],
                                state.liquids[0].phis()[SPECIES.index(i)],
                            )
                            for i in SPECIES
                        ]
                    )
                )
                # fill other liquid phases with nans
                for j in PHASES[2:]:
                    out.update(
                        {
                            compressibility_HEADER[j]: NAN_ENTRY,
                        }
                    )
                    out.update(
                        dict(
                            [(composition_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                            + [(fugacity_HEADER[i][j], NAN_ENTRY) for i in SPECIES]
                        )
                    )
            elif 1 < len(state.liquids) <= NUM_COMP:  # get all liquid data
                assert (
                    state.liquids_betas
                ), "Thermo conflicting liquid phase state: no liquid betas"
                for i in range(NUM_COMP):
                    out.update({liq_frac_HEADER[i]: state.liquids_betas[i]})
                for p_idx, j in enumerate(PHASES[1:]):
                    out[phases_HEADER] = out[phases_HEADER] + "L"
                    out.update(
                        {
                            compressibility_HEADER[j]: state.liquids[p_idx].Z(),
                        }
                    )
                    out.update(
                        dict(
                            [
                                (
                                    composition_HEADER[i][j],
                                    state.liquids[p_idx].zs[SPECIES.index(i)],
                                )
                                for i in SPECIES
                            ]
                            + [
                                (
                                    fugacity_HEADER[i][j],
                                    state.liquids[p_idx].phis()[SPECIES.index(i)],
                                )
                                for i in SPECIES
                            ]
                        )
                    )
            else:  # more liquid phases than anticipated
                raise NotImplementedError(
                    f"Uncovered thermo state with:"
                    + f"\nLiquid: {state.liquids}\nGas: {state.gas}"
                )
    else:
        raise NotImplementedError(
            f"Uncovered thermo state phase-count {state.phase_count}"
        )

    return out


def _failed_entry() -> dict[str]:
    """Create a row-entry for failed flashes."""
    failed: dict = _init_empty_results()
    for k in failed.keys():
        failed[k] = NAN_ENTRY
    failed[success_HEADER] = 0
    return failed


def calculate_thermo_pT_data() -> dict[str, list]:
    """Uses thermo to perform the p-T flash for various pressure and temperature ranges.

    Returns a dictionary containing per header (name of some property) respective values
    per p-T point.

    """

    flasher = _thermo_init()
    results = _init_empty_results()

    p_points = np.linspace(P_LIMITS[0], P_LIMITS[1], num=RESOLUTION_pT).tolist()
    T_points = np.linspace(T_LIMITS[0], T_LIMITS[1], num=RESOLUTION_pT).tolist()

    f_num = len(T_points) * len(p_points)
    f_count = 1

    for T in T_points:
        for P in p_points:
            try:
                state = flasher.flash(P=P, T=T, zs=FEED)
            except Exception:
                logger.warn(f"\nThermo p-T-flash failed for p, T = ({P}, {T})\n")
                parsed = _failed_entry()
            else:
                parsed = _thermo_parse_result(state)
                # sanity check
                assert state.P == P, "Thermo p-T result has different pressure."
                assert state.T == T, "Thermo p-T result has different temperature."
                # in the p-T flash, we use the thermo enthalpy also as target enthalpy
                parsed[h_HEADER] = state.H()
            finally:
                parsed[p_HEADER] = P
                parsed[T_HEADER] = T
                for head, val in parsed.items():
                    results[head].append(val)
                print(f"\rFlash: {f_count}/{f_num} done", end="", flush=True)
                logger.info(f"{del_log}Thermo p-T-flash: {f_count}/{f_num}")
                f_count += 1
    logger.info(f"{del_log}Thermo p-T-flash: done\n")

    return results


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
    arrays_loc = [vec for vec, _ in shared_arrays]


def _array_headers() -> list[str]:
    """Returns a list of header (names) for arrays created for the parallel flash."""
    headers = [
        success_HEADER,
        # phases_HEADER,
        num_iter_HEADER,
        conditioning_HEADER,
        p_HEADER,
        T_HEADER,
        h_HEADER,
        gas_frac_HEADER,
        compressibility_HEADER[PHASES[1]],
        compressibility_HEADER[PHASES[0]],
    ]
    for i in SPECIES:
        for j in PHASES[:2]:
            headers += [composition_HEADER[i][j]]
    for i in SPECIES:
        for j in PHASES[:2]:
            headers += [fugacity_HEADER[i][j]]
    return headers


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

    def _string_array():
        return Array(typecode_or_type=c_char_p, size_or_initializer=size)

    # array to store the success flag: 0 or 1
    success = _uint_array()
    shared_arrays.append((success, INT_PRECISION))
    # split = _string_array()
    # shared_arrays.append((split, c_char_p))
    # integer array to store the number of iterations necessary
    num_iter = _uint_array()
    shared_arrays.append((num_iter, INT_PRECISION))
    # array storing 1 if the flash showes a super-critical mixture at converged state.
    # is_supercrit = _uint_array()
    # shared_arrays.append((is_supercrit, INT_PRECISION))
    # array storing the condition number of the array at the beginning (initial guess)
    cond_start = _double_array()
    shared_arrays.append((cond_start, FLOAT_PRECISION))
    # array showing the condition number at converged state
    # cond_end = _double_array()
    # shared_arrays.append((cond_end, FLOAT_PRECISION))
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
    for _ in SPECIES:
        for _ in PHASES[:2]:
            x = _double_array()
            shared_arrays.append((x, FLOAT_PRECISION))
    # arrays containing the fugacities
    for _ in SPECIES:
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
        logger.info(f"{del_log}Parallel flash: {progress}/{NC}")
        if progress == NC:
            break


def create_mixture(
    num_vals: int,
) -> tuple[pp.composite.NonReactiveMixture, pp.composite.FlashNR]:
    """Returns instances of the modelled mixture and flash using PorePy's framework.

    ``num_vals`` is an integer indicating how many DOFs per state function are set.
    This is used for vectorization.
    Especially, choose 1 for a point-wise and parallelized flash.

    Configure flash parameters here.

    """

    species = pp.composite.load_fluid_species(SPECIES)

    comps = [
        pp.composite.peng_robinson.H2O.from_species(species[0]),
        pp.composite.peng_robinson.CO2.from_species(species[1]),
    ]

    phases = [
        pp.composite.Phase(
            pp.composite.peng_robinson.PengRobinsonEoS(gaslike=False), name="L"
        ),
        pp.composite.Phase(
            pp.composite.peng_robinson.PengRobinsonEoS(gaslike=True), name="G"
        ),
    ]

    mix = pp.composite.NonReactiveMixture(comps, phases)

    mix.set_up(num_vals=num_vals)

    # instantiating Flasher, without auxiliary variables V and W
    flash = pp.composite.FlashNR(mix)
    flash.use_armijo = True
    flash.armijo_parameters["rho"] = 0.99
    flash.armijo_parameters["j_max"] = 50
    flash.armijo_parameters["return_max"] = True
    flash.newton_update_chop = 1.0
    flash.tolerance = 1e-6
    flash.max_iter = 120

    return mix, flash


def _porepy_parse_state(state: pp.composite.ThermodynamicState) -> dict:
    """Function to parse the resulting state after a porepy flash into a structure
    ready for writing to csv.

    Only meant for states with 1 value per state function.

    Parses only molar fractional variables and phase split.

    """
    out = _init_empty_results()
    for k in out.keys():
        out[k] = NAN_ENTRY

    y = state.y[1][0]
    if y >= 1:
        out[phases_HEADER] = "G"
    elif y <= 0:
        out[phases_HEADER] = "L"
    else:
        out[phases_HEADER] = "GL"
    out[gas_frac_HEADER] = y
    out[liq_frac_HEADER[0]] = 1 - y

    # liquid phase composition
    for i, s in enumerate(SPECIES):
        x_ij = state.X[0][i][0]
        out[composition_HEADER[s][PHASES[1]]] = x_ij
    # gas phase composition
    for i, s in enumerate(SPECIES):
        x_ij = state.X[1][i][0]
        out[composition_HEADER[s][PHASES[0]]] = x_ij

    return out


def _parallel_pT_flash(ipT):
    """Performs a p-T flash (including modelling) and stores the results in shared
    memory.

    ``ipT`` must be a tuple containing the index where the results should be stored,
    and the p-T point.

    Warning:
        There are some unresolved issues with parallel subprocesses if
        numpy/scipy/pypardiso throw errors or warnings. It causes the respective
        subprocess unable to join, despite finishing all flashes.

    """

    i, p, T, quickshot = ipT

    # accessing shared memory
    global arrays_loc, progress_queue_loc
    (
        success_arr,
        # split_arr,
        num_iter_arr,
        # is_supercrit_arr,
        cond_arr,
        # cond_end_arr,
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

    mix, flash = create_mixture(1)

    p_vec = np.array([p], dtype=np.double)
    T_vec = np.array([T])
    feed = [np.ones(1) * z for z in FEED]

    p_arr[i] = p
    T_arr[i] = T

    try:
        success_, state = flash.flash(
            state={"p": p_vec, "T": T_vec},
            feed=feed,
            eos_kwargs={"apply_smoother": True},
            quickshot=quickshot,
            return_system=True,
        )
    except Exception as err:  # if Flasher fails, flag as failed
        logger.warn(f"\nParallel p-T flash crashed at {ipT}\n{str(err)}\n")

        success_arr[i] = 0
        # split_arr[i] = str(NAN_ENTRY)
        # is_supercrit_arr[i] = 0
        y_arr[i] = NAN_ENTRY
        x_h2o_L_arr[i] = NAN_ENTRY
        x_h2o_G_arr[i] = NAN_ENTRY
        x_co2_L_arr[i] = NAN_ENTRY
        x_co2_G_arr[i] = NAN_ENTRY

        num_iter_arr[i] = NAN_ENTRY
        cond_arr[i] = NAN_ENTRY
        # cond_end_arr[i] = NAN_ENTRY

        Z_L_arr[i] = NAN_ENTRY
        Z_G_arr[i] = NAN_ENTRY
        phi_h2o_L_arr[i] = NAN_ENTRY
        phi_h2o_G_arr[i] = NAN_ENTRY
        phi_co2_L_arr[i] = NAN_ENTRY
        phi_co2_G_arr[i] = NAN_ENTRY
        h_arr[i] = NAN_ENTRY
    else:
        cond_arr[i] = np.linalg.cond(state(with_derivatives=True).jac.todense())
        state = state.export_state()
        if success_ in [0, 3]:
            success_arr[i] = 1
            props = mix.compute_properties(state.p, state.T, state.X, store=False)
            Z_L_arr[i] = props[0].Z[0]
            Z_G_arr[i] = props[1].Z[0]
            phi_h2o_L_arr[i] = props[0].phis[0][0]
            phi_co2_L_arr[i] = props[0].phis[1][0]
            phi_h2o_G_arr[i] = props[1].phis[0][0]
            phi_co2_G_arr[i] = props[1].phis[1][0]
        else:
            logger.warn(f"\nParallel p-T failed to converge at {ipT}\n")
            success_arr[i] = 0
            Z_L_arr[i] = NAN_ENTRY
            Z_G_arr[i] = NAN_ENTRY
            phi_h2o_L_arr[i] = NAN_ENTRY
            phi_co2_L_arr[i] = NAN_ENTRY
            phi_h2o_G_arr[i] = NAN_ENTRY
            phi_co2_G_arr[i] = NAN_ENTRY

        h_arr[i] = state.h[0]
        y_arr[i] = state.y[1][0]
        # if 0 < state.y[1][0] < 1:
        #     split_arr[i] = 'GL'
        # elif state.y[1][0] <= 0:
        #     split_arr[i] = 'L'
        # elif state.y[1][0] >= 1.:
        #     split_arr[i] = 'G'
        x_h2o_L_arr[i] = state.X[0][0][0]
        x_h2o_G_arr[i] = state.X[1][0][0]
        x_co2_L_arr[i] = state.X[0][1][0]
        x_co2_G_arr[i] = state.X[1][1][0]

        if quickshot:
            num_iter_arr[i] = 0
        else:
            num_iter_arr[i] = flash.history[-1]["iterations"]

    progress_queue_loc.put(i, block=False)


def _parallel_ph_flash(iph):
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

    # accessing shared memory
    global arrays_loc, progress_queue_loc
    (
        success_arr,
        # split_arr,
        num_iter_arr,
        # is_supercrit_arr,
        cond_arr,
        # cond_end_arr,
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

    mix, flash = create_mixture(1)

    p_vec = np.array([p], dtype=np.double)
    h_vec = np.array([h])
    feed = [np.ones(1) * z for z in FEED]

    p_arr[i] = p
    h_arr[i] = h

    try:
        success_, state = flash.flash(
            state={"p": p_vec, "h": h_vec},
            feed=feed,
            eos_kwargs={"apply_smoother": True},
        )
    except Exception as err:  # if Flasher fails, flag as failed
        logger.warn(f"\nParallel p-h flash crashed at {iph}\n{str(err)}\n")

        success_arr[i] = 0
        # split_arr[i] = str(NAN_ENTRY)
        # is_supercrit_arr[i] = 0
        y_arr[i] = NAN_ENTRY
        x_h2o_L_arr[i] = NAN_ENTRY
        x_h2o_G_arr[i] = NAN_ENTRY
        x_co2_L_arr[i] = NAN_ENTRY
        x_co2_G_arr[i] = NAN_ENTRY

        num_iter_arr[i] = NAN_ENTRY
        # cond_start_arr[i] = NAN_ENTRY
        # cond_end_arr[i] = NAN_ENTRY

        Z_L_arr[i] = NAN_ENTRY
        Z_G_arr[i] = NAN_ENTRY
        phi_h2o_L_arr[i] = NAN_ENTRY
        phi_h2o_G_arr[i] = NAN_ENTRY
        phi_co2_L_arr[i] = NAN_ENTRY
        phi_co2_G_arr[i] = NAN_ENTRY
        T_arr[i] = NAN_ENTRY
    else:
        if success_ == 0:
            success_arr[i] = 1
            props = mix.compute_properties(state.p, state.T, state.X, store=False)
            Z_L_arr[i] = props[0].Z[0]
            Z_G_arr[i] = props[1].Z[0]
            phi_h2o_L_arr[i] = props[0].phis[0][0]
            phi_co2_L_arr[i] = props[0].phis[1][0]
            phi_h2o_G_arr[i] = props[1].phis[0][0]
            phi_co2_G_arr[i] = props[1].phis[1][0]
        else:
            logger.warn(f"\nParallel p-h failed to converge at {iph}\n")
            success_arr[i] = 0
            Z_L_arr[i] = NAN_ENTRY
            Z_G_arr[i] = NAN_ENTRY
            phi_h2o_L_arr[i] = NAN_ENTRY
            phi_co2_L_arr[i] = NAN_ENTRY
            phi_h2o_G_arr[i] = NAN_ENTRY
            phi_co2_G_arr[i] = NAN_ENTRY

        T_arr[i] = state.T[0]
        y_arr[i] = state.y[1][0]
        # if 0 < state.y[1][0] < 1:
        #     split_arr[i] = 'GL'
        # elif state.y[1][0] <= 0:
        #     split_arr[i] = 'L'
        # elif state.y[1][0] >= 1.:
        #     split_arr[i] = 'G'
        x_h2o_L_arr[i] = state.X[0][0][0]
        x_h2o_G_arr[i] = state.X[1][0][0]
        x_co2_L_arr[i] = state.X[0][1][0]
        x_co2_G_arr[i] = state.X[1][1][0]

        num_iter_arr[i] = flash.history[-1]["iterations"]

    progress_queue_loc.put(i, block=False)


def calculate_porepy_pT_data(
    p_points: list[float], T_points: list[float], quickshot: bool = False
) -> dict:
    """Performs the PorePy flash for given pressure-temperature points and
    returns a result structure similar to that of the thermo computation.

    If ``quickshot`` is True, returns the results from the initial guess."""

    results = _init_empty_results()
    nf = len(p_points)

    if CALCULATION_MODE == 1:  # point-wise flash
        v = np.ones(1)
        logger.info(f"PorePy p-T-flash: initializing point-wise calculations ..")
        mix, flash = create_mixture(1)
        feed = [v * z for z in FEED]

        for f, pT in enumerate(zip(p_points, T_points)):
            p, T = pT
            p_ = v * p
            T_ = v * T

            try:
                success, state = flash.flash(
                    state={"p": p_, "T": T_},
                    feed=feed,
                    eos_kwargs={"apply_smoother": True},
                    quickshot=quickshot,
                    return_system=True,
                )
            except:
                logger.warn(f"\nPorePy p-T-flash crashed for p,T = ({p}, {T})\n")
                res = _failed_entry()
            else:
                cn = np.linalg.cond(state(with_derivatives=True).jac.todense())
                state = state.export_state()
                res = _porepy_parse_state(state)

                if quickshot:
                    res[num_iter_HEADER] = 0
                else:
                    res[num_iter_HEADER] = int(flash.history[-1]["iterations"])
                res[conditioning_HEADER] = cn
                if success not in [0, 3]:
                    logger.warn(
                        f"\nPorePy p-T-flash failed to converge for p,T = ({p}, {T})\n"
                    )
                    res[success_HEADER] = 0
                else:
                    logger.info(f"{del_log}PorePy p-T-flash: {f+1} / {nf} ..")
                    res[success_HEADER] = 1

                    props = mix.compute_properties(
                        state.p, state.T, state.X, store=False
                    )

                    res[h_HEADER] = mix.evaluate_weighed_sum(
                        [prop.h for prop in props], state.y
                    )[0]

                    res[compressibility_HEADER[PHASES[1]]] = props[0].Z[0]
                    for i, s in enumerate(SPECIES):
                        res[fugacity_HEADER[s][PHASES[1]]] = props[0].phis[i][0]
                    res[compressibility_HEADER[PHASES[0]]] = props[1].Z[0]
                    for i, s in enumerate(SPECIES):
                        res[fugacity_HEADER[s][PHASES[0]]] = props[1].phis[i][0]

            res[p_HEADER] = p
            res[T_HEADER] = T

            for key, val in res.items():
                results[key].append(val)

    elif CALCULATION_MODE == 2:  # vectorized flash
        pass
    elif CALCULATION_MODE == 3:  # parallelized flash

        ipx = [
            (i, p, x, quickshot) for i, p, x in zip(np.arange(nf), p_points, T_points)
        ]
        shared_arrays = _create_shared_arrays(nf)
        logger.info("Parallel p-T flash: starting ..")
        start_time = time.time()
        # multiprocessing.set_start_method('fork')
        prog_q = Queue(maxsize=nf)
        with Pool(
            processes=NUM_PHYS_CPU_CORS + 1,
            initargs=(shared_arrays, prog_q),
            initializer=_access_shared_objects,
        ) as pool:

            prog_process = Process(
                target=_progress_counter, args=(prog_q, nf, None), daemon=True
            )
            prog_process.start()

            chunksize = NUM_PHYS_CPU_CORS
            result = pool.map_async(_parallel_pT_flash, ipx, chunksize=chunksize)

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
                logger.warn(f"\nParallel p-T flash: terminated\n")
                pool.close()
                pool.terminate()
            pool.join()

        end_time = time.time()
        logger.info(
            f"\nParallel p-T flash: finished after {end_time - start_time} seconds.\n"
        )

        result_vecs = [
            list(np.frombuffer(vec.get_obj(), dtype=dtype))
            for vec, dtype in shared_arrays
        ]
        results = dict(
            [(header, vec) for header, vec in zip(_array_headers(), result_vecs)]
        )

    else:
        raise ValueError(f"Unknown flash calculation mode {CALCULATION_MODE}.")

    return results


def calculate_porepy_isotherm_data():
    """Calculates p-T data along defined isotherms using defined pressure limits."""

    p_points = list()
    T_points = list()
    p_ = np.linspace(
        P_LIMITS_ISOTHERMS[0],
        P_LIMITS_ISOTHERMS[1],
        RESOLUTION_isotherms,
        endpoint=True,
        dtype=float,
    )

    for T in ISOTHERMS:
        for p in p_:
            T_points.append(T)
            p_points.append(p)

    results = _init_empty_results()
    nf = len(p_points)

    if CALCULATION_MODE == 1:  # point-wise flash
        v = np.ones(1)
        logger.info(f"PorePy p-T-flash: initializing point-wise calculations ..")
        mix, flash = create_mixture(1)
        feed = [v * z for z in FEED]

        for f, pT in enumerate(zip(p_points, T_points)):
            p, T = pT
            p_ = v * p
            T_ = v * T

            try:
                success, state = flash.flash(
                    state={"p": p_, "T": T_},
                    feed=feed,
                    eos_kwargs={"apply_smoother": True},
                )
            except:
                logger.warn(f"\nPorePy p-T-flash crashed for p,T = ({p}, {T})\n")
                res = _failed_entry()
            else:
                res = _porepy_parse_state(state)

                res[num_iter_HEADER] = int(flash.history[-1]["iterations"])
                if success != 0:
                    logger.warn(
                        f"\nPorePy p-T-flash failed to converge for p,T = ({p}, {T})\n"
                    )
                    res[success_HEADER] = 0
                else:
                    logger.info(f"{del_log}PorePy p-T-flash: {f+1} / {nf} ..")
                    res[success_HEADER] = 1

                    props = mix.compute_properties(
                        state.p, state.T, state.X, store=False
                    )

                    res[h_HEADER] = mix.evaluate_weighed_sum(
                        [prop.h for prop in props], state.y
                    )[0]

                    res[compressibility_HEADER[PHASES[1]]] = props[0].Z[0]
                    for i, s in enumerate(SPECIES):
                        res[fugacity_HEADER[s][PHASES[1]]] = props[0].phis[i][0]
                    res[compressibility_HEADER[PHASES[0]]] = props[1].Z[0]
                    for i, s in enumerate(SPECIES):
                        res[fugacity_HEADER[s][PHASES[0]]] = props[1].phis[i][0]

            res[p_HEADER] = p
            res[T_HEADER] = T

            for key, val in res.items():
                results[key].append(val)

    elif CALCULATION_MODE == 2:  # vectorized flash
        pass
    elif CALCULATION_MODE == 3:  # parallelized flash

        ipx = [(i, p, x, False) for i, p, x in zip(np.arange(nf), p_points, T_points)]
        shared_arrays = _create_shared_arrays(nf)
        logger.info("Parallel p-T flash: starting ..")
        start_time = time.time()
        # multiprocessing.set_start_method('fork')
        prog_q = Queue(maxsize=nf)
        with Pool(
            processes=NUM_PHYS_CPU_CORS + 1,
            initargs=(shared_arrays, prog_q),
            initializer=_access_shared_objects,
        ) as pool:

            prog_process = Process(
                target=_progress_counter, args=(prog_q, nf, None), daemon=True
            )
            prog_process.start()

            chunksize = NUM_PHYS_CPU_CORS
            result = pool.map_async(_parallel_pT_flash, ipx, chunksize=chunksize)

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
                logger.warn(f"\nParallel p-T flash: terminated\n")
                pool.close()
                pool.terminate()
            pool.join()

        end_time = time.time()
        logger.info(
            f"\nParallel p-T flash: finished after {end_time - start_time} seconds.\n"
        )

        result_vecs = [
            list(np.frombuffer(vec.get_obj(), dtype=dtype))
            for vec, dtype in shared_arrays
        ]
        results = dict(
            [(header, vec) for header, vec in zip(_array_headers(), result_vecs)]
        )

    else:
        raise ValueError(f"Unknown flash calculation mode {CALCULATION_MODE}.")

    return results


def calculate_porepy_ph_data(p_points: list[float], h_points: list[float]) -> dict:
    """Performs the PorePy flash for given pressure-temperature points and
    returns a result structure similar to that of the thermo computation."""

    results = _init_empty_results()
    nf = len(p_points)

    if CALCULATION_MODE == 1:  # point-wise flash
        v = np.ones(1)
        logger.info(f"PorePy p-h-flash: initializing point-wise calculations ..")
        mix, flash = create_mixture(1)
        feed = [v * z for z in FEED]

        for f, ph in enumerate(zip(p_points, h_points)):
            p, h = ph
            p_ = v * p
            h_ = v * h

            # if the p-T- flash yielding h failed, h is none
            # indicate the p-h flash in this case also as failed
            if h in [NAN_ENTRY, str(NAN_ENTRY)]:
                logger.info(f"\nPorePy p-h-flash: {f + 1} / {nf} skipped (h is nan)\n")
                res = _failed_entry()
            else:
                try:
                    success, state = flash.flash(
                        state={"p": p_, "h": h_},
                        feed=feed,
                        eos_kwargs={"apply_smoother": True},
                    )
                except:
                    logger.warn(f"\nPorePy p-h-flash crashed for p,h = ({p}, {h})\n")
                    res = _failed_entry()
                else:
                    res = _porepy_parse_state(state)

                    res[num_iter_HEADER] = int(flash.history[-1]["iterations"])
                    if success != 0:
                        logger.warn(
                            f"\nPorePy p-h-flash failed to converge for p,h = ({p}, {h})\n"
                        )
                        res[success_HEADER] = 0
                    else:
                        logger.info(f"{del_log}PorePy p-h-flash: {f+1} / {nf} ..")
                        res[success_HEADER] = 1

                        props = mix.compute_properties(
                            state.p, state.T, state.X, store=False
                        )

                        res[T_HEADER] = state.T[0]

                        res[compressibility_HEADER[PHASES[1]]] = props[0].Z[0]
                        for i, s in enumerate(SPECIES):
                            res[fugacity_HEADER[s][PHASES[1]]] = props[0].phis[i][0]
                        res[compressibility_HEADER[PHASES[0]]] = props[1].Z[0]
                        for i, s in enumerate(SPECIES):
                            res[fugacity_HEADER[s][PHASES[0]]] = props[1].phis[i][0]

            res[p_HEADER] = p
            res[h_HEADER] = h

            for key, val in res.items():
                results[key].append(val)

    elif CALCULATION_MODE == 2:  # vectorized flash
        pass
    elif CALCULATION_MODE == 3:  # parallelized flash

        ipx = [(i, p, x) for i, p, x in zip(np.arange(nf), p_points, h_points)]
        shared_arrays = _create_shared_arrays(nf)
        logger.info("Parallel p-h flash: starting ..")
        start_time = time.time()
        # multiprocessing.set_start_method('fork')
        prog_q = Queue(maxsize=nf)
        with Pool(
            processes=NUM_PHYS_CPU_CORS + 1,
            initargs=(shared_arrays, prog_q),
            initializer=_access_shared_objects,
        ) as pool:

            prog_process = Process(
                target=_progress_counter, args=(prog_q, nf, None), daemon=True
            )
            prog_process.start()

            chunksize = NUM_PHYS_CPU_CORS
            result = pool.map_async(_parallel_ph_flash, ipx, chunksize=chunksize)

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
                logger.warn(f"\nParallel p-h flash: terminated\n")
                pool.close()
                pool.terminate()
            pool.join()

        end_time = time.time()
        logger.info(
            f"\nParallel p-h flash: finished after {end_time - start_time} seconds.\n"
        )

        result_vecs = [
            list(np.frombuffer(vec.get_obj(), dtype=dtype))
            for vec, dtype in shared_arrays
        ]
        results = dict(
            [(header, vec) for header, vec in zip(_array_headers(), result_vecs)]
        )

    else:
        raise ValueError(f"Unknown flash calculation mode {CALCULATION_MODE}.")

    return results


def plot_crit_point_pT(axis: plt.Axes):
    """Plot critical pressure and temperature in p-T plot for components H2O and CO2."""

    S = pp.composite.load_fluid_species(SPECIES)

    img = [
        axis.plot(
            s.T_crit, s.p_crit * PRESSURE_SCALE, "*", markersize=7, color="fuchsia"
        )[0]
        for s in S[:1]
    ]

    return img, [f"Crit. point {s.name}" for s in S[:1]]


def plot_phase_split_pT(
    axis: plt.Axes,
    p: np.ndarray,
    T: np.ndarray,
    split: np.ndarray,
) -> figure.Figure:
    """Plots a phase split figure across a range of pressure and temperature values."""

    cmap = mpl.colors.ListedColormap(
        ["firebrick", "royalblue", "mediumturquoise", "forestgreen"]
    )
    img = axis.pcolormesh(
        T,
        p * PRESSURE_SCALE,
        split,
        cmap=cmap,
        vmin=0,
        vmax=3,
        shading="nearest",  # gouraud
    )

    return img


def plot_abs_error_pT(
    axis: plt.Axes,
    p: np.ndarray,
    T: np.ndarray,
    error: np.ndarray,
    norm=None,
) -> figure.Figure:
    """Plots the absolute error in grey scales."""

    if norm:
        kwargs = {"norm": norm}
    else:
        kwargs = {"vmin": error.min(), "vmax": error.max()}

    img = axis.pcolormesh(
        T, p * PRESSURE_SCALE, error, cmap="Greys", shading="nearest", **kwargs
    )

    return img
