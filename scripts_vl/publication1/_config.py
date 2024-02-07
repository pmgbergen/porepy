"""Configuration file containing data range values and functionality for calculating
and plotting data."""
from __future__ import annotations

import csv
import logging
import pathlib
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib import figure
from matplotlib.colors import LinearSegmentedColormap
from thermo import PR78MIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVLN
from thermo.interaction_parameters import IPDB

import porepy as pp

from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from batlow import cm_data as batlow_data

# figure configurations
# in inches, all plots are square, error plots have a aspect ratio of 1:3
FIGURE_WIDTH: int = 10
# height-to-width ratio per plot
# This is not applied to error plots
ASPECT_RATIO: float = 0.8
DPI: int = 400  # Dots per Inch (level of detail per figure)
FIGURE_FORMAT: str = "png"
MARKER_SCALE: int = 2  # Size scaling of markers in legend
MARKER_SIZE: int = 10

# Defining colors for plots
batlow_map = LinearSegmentedColormap.from_list("batlow", batlow_data)
NA_COL = batlow_map(0.0)  # color for not available data
LIQ_COL = batlow_map(0.25)  # color for liquid phase
MPHASE_COL = batlow_map(0.5)  # color for multi-phase region
GAS_COL = batlow_map(0.75)  # color for gas phase
GLL_COL = batlow_map(0.875)
LL_COL = batlow_map(0.375)
GAS_COL_2 = batlow_map(1.0)  # Additional color for gas phase (Widom extension)
WHITE_COL = (1, 1, 1, 1)
BLACK_COL = (0, 0, 0, 1)
GREY_COL = (0.5, 0.5, 0.5, 1)

# Calculation modus for PorePy flash
# 1 - point-wise (robust, but possibly very slow),
# 3 - parallelized (use with care, if system compatible)
CALCULATION_MODE: int = 3

# fluid mixture configuration
SPECIES: list[str] = ["H2O", "CO2"]
FEED: list[float] = [0.99, 0.01]

SPECIES_geo: list[str] = ["H2O", "CO2", "H2S", "N2"]
FEED_geo: list[float] = [0.8, 0.05, 0.1, 0.05]

# pressure and temperature limits for p-T calculations
P_LIMITS: list[float] = [1e6, 50e6]  # [Pa]
T_LIMITS: list[float] = [450.0, 700.0]  # [K]
# resolution of p-T limits
RESOLUTION_pT: int = 80

# temperature values for isotherms for p-h calculations
# more refined around critical temperature of water, up to critical pressure of water
ISOTHERMS: list[float] = [500.0, 550.0, 600, 640.0, 647.14, 650.0]
P_LIMITS_ISOTHERMS: list[float] = [1e6, 23000000.0]
# pressure resolution along isotherms
RESOLUTION_ph: int = 20

# Isobar and isotherm for h-v calculations
HV_ISOBAR: float = 15e6
HV_ISOBAR_T_LIMITS: list[float] = [575, 630]
HV_ISOTHERM: float = 575.0
HV_ISOTHERM_P_LIMITS: list[float] = [5e6, 15e6]
# pressure and temperature resolution for isobar and isotherm for h-v flash
RESOLUTION_hv: int = 10

# Pressure and enthalpy limits for multi-component, geothermal fluid example
GEO_P_LIMITS: list[float] = [22e6, 27e6]  # [Pa]
# GEO_H_LIMITS: list[float] = [-15e3, 15e3]  # [kJ]
GEO_H_LIMITS: list[float] = [-15e3, 8e3]  # [kJ]
GEO_T_LIMITS: list[float] = [500, 820]
EXAMPLE_2_flash_type: str = "p-h"  # p-T or p-h
RESOLUTION_geo: int = 40

# Limits for A and B when plotting the roots
A_LIMITS: list[float] = [0, 2 * pp.composite.peng_robinson.A_CRIT]
B_LIMITS: list[float] = [0, 2 * pp.composite.peng_robinson.B_CRIT]
RESOLUTION_AB: int = 300

# Widom line for water: Pressure and Temperature values
WIDOM_LINE: list[np.ndarray] = [
    np.array([225, 250, 270]) * 1e5,  # bar to Pa
    np.array([646.9, 655.6, 664.9]),
]

# Scaling for plots
PRESSURE_SCALE: float = 1e-6  # Pa to MPa
PRESSURE_SCALE_NAME: str = "MPa"
# scaling of pressure or temperature for plots for second example
X_SCALE: float = 1e-3 if EXAMPLE_2_flash_type == "p-h" else 1.0  # J to kJ for p-h
X_SCALE_NAME: str = "kJ"

# paths to where results should be stored
THERMO_DATA_PATH: str = "data/thermodata.csv"  # storing results from therm
PT_FLASH_DATA_PATH: str = "data/flash_pT.csv"  # storing p-T results from porepy
ISOTHERM_DATA_PATH: str = (
    "data/flash_pT_isotherms.csv"  # storing p-T results on isotherms
)
PH_FLASH_DATA_PATH: str = "data/flash_ph.csv"  # storing p-h results from porepy
HV_ISOTHERM_DATA_PATH: str = (
    "data/flash_hv_isotherm.csv"  # storing p-T results on isotherm for h-v flash
)
HV_ISOBAR_DATA_PATH: str = (
    "data/flash_hv_isobar.csv"  # storing p-T results on isobar for h-v flash
)
HV_FLASH_DATA_PATH: str = "data/flash_hv.csv"  # storing h-v results from porepy
GEO_DATA_PATH: str = "data/flash_geo.csv"  # storing p-h results for geothermal model
GEO_THERMO_DATA_PATH: str = (
    "data/thermo_geo.csv"  # storing thermo results for geothermal model
)
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
h_mix_HEADER: str = "h mix [J/mol]"
v_HEADER: str = "v [m^3]"
v_mix_HEADER: str = "v mix [m^3]"
gas_satur_HEADER: str = "s"
liq_satur_HEADER: str = [f"s_L{j}" for j in range(1, NUM_COMP + 1)]
gas_frac_HEADER: str = "y"
liq_frac_HEADER: list[str] = [f"y_L{j}" for j in range(1, NUM_COMP + 1)]
composition_HEADER: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"x_{i}_{j}") for j in PHASES])) for i in SPECIES]
)
compressibility_HEADER: dict[str, str] = dict([(f"{j}", f"Z_{j}") for j in PHASES])
fugacity_HEADER: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"phi_{i}_{j}") for j in PHASES])) for i in SPECIES]
)

composition_HEADER_geo: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"x_{i}_{j}") for j in PHASES])) for i in SPECIES_geo]
)
fugacity_HEADER_geo: dict[str, dict[str, str]] = dict(
    [(i, dict([(f"{j}", f"phi_{i}_{j}") for j in PHASES])) for i in SPECIES_geo]
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


def read_data_column(
    file_path: str,
    header: str,
) -> tuple[list[float], list[float]]:
    """Reads a data column from a file."""
    points: list[float] = list()

    logger.info(f"{del_log}Reading data column {header} from file {file_path}")
    with open(f"{path()}/{file_path}", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER)
        headers = next(reader)  # get rid of header
        if header not in headers:
            raise ValueError(f"Header {header} not found in file {file_path}")

        idx = headers.index(header)

        for row in reader:

            # must always be available
            x = float(row[idx])
            # potentially missing or nan
            if x not in [NAN_ENTRY, str(NAN_ENTRY)]:
                x = float(x)
            else:
                x = NAN_ENTRY

            points.append(x)
    logger.info(f"{del_log}Reading data column {header}: done\n")
    return points


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


def _thermo_init(species=SPECIES) -> FlashVLN:
    """Helper function to initiate the thermo flash."""
    constants, properties = ChemicalConstantsPackage.from_IDs(species)
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
        for _ in range(len(species))
    ]
    flasher = FlashVLN(constants, properties, liquids=LIQs, gas=GAS)

    return flasher


def _init_empty_results(
    species=SPECIES, comp_header=composition_HEADER, fug_header=fugacity_HEADER
) -> dict[str, list]:
    """Initiate and return an results dict with proper headers as needed for the
    comparison."""
    results: dict[str, list] = {
        success_HEADER: list(),
        num_iter_HEADER: list(),
        conditioning_HEADER: list(),
        p_HEADER: list(),
        T_HEADER: list(),
        h_HEADER: list(),
        v_HEADER: list(),
        phases_HEADER: list(),
        gas_satur_HEADER: list(),
    }
    results.update(dict([(liq_satur_HEADER[i], list()) for i in range(NUM_COMP)]))
    results.update({gas_frac_HEADER: list()})
    results.update(dict([(liq_frac_HEADER[i], list()) for i in range(NUM_COMP)]))
    results.update(dict([(compressibility_HEADER[j], list()) for j in PHASES]))
    results.update(dict([(comp_header[i][j], list()) for j in PHASES for i in species]))
    results.update(dict([(fug_header[i][j], list()) for j in PHASES for i in species]))

    return results


def _thermo_parse_result(state) -> dict:
    """Helper function to parse a state returned by thermo into processable format."""
    out = _init_empty_results()
    for k in out.keys():
        out[k] = NAN_ENTRY

    out.update(
        {
            success_HEADER: 0,
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


def _failed_entry(
    species=SPECIES, comp_header=composition_HEADER, fug_header=fugacity_HEADER
) -> dict[str]:
    """Create a row-entry for failed flashes."""
    failed: dict = _init_empty_results(species, comp_header, fug_header)
    for k in failed.keys():
        failed[k] = NAN_ENTRY
    failed[success_HEADER] = 2
    return failed


def calculate_example_1_thermo() -> dict[str, list]:
    """Uses thermo to perform the p-T flash for various pressure and temperature ranges.

    Returns a dictionary containing per header (name of some property) respective values
    per p-T point.

    """

    flasher = _thermo_init(SPECIES)
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
                logger.info(f"{del_log}Thermo p-T-flash: {f_count}/{f_num}")
                f_count += 1
    logger.info(f"{del_log}Thermo p-T-flash: done\n")

    return results


def calculate_example_2_thermo(flash_type: str = "p-h") -> dict[str, list]:
    """Uses thermo to perform the p-T flash for various pressure and temperature ranges.

    Returns a dictionary containing per header (name of some property) respective values
    per p-T point.

    """

    flasher = _thermo_init(SPECIES_geo)
    results = _init_empty_results(
        SPECIES_geo, composition_HEADER_geo, fugacity_HEADER_geo
    )

    p_points = np.linspace(
        GEO_P_LIMITS[0],
        GEO_P_LIMITS[1],
        RESOLUTION_geo,
        endpoint=True,
        dtype=float,
    ).tolist()

    if flash_type == "p-h":
        x_points = np.linspace(
            GEO_H_LIMITS[0],
            GEO_H_LIMITS[1],
            RESOLUTION_geo,
            endpoint=True,
            dtype=float,
        ).tolist()
    elif flash_type == "p-T":
        x_points = np.linspace(
            GEO_T_LIMITS[0],
            GEO_T_LIMITS[1],
            RESOLUTION_geo,
            endpoint=True,
            dtype=float,
        ).tolist()
    else:
        raise ValueError("Only p-T or p-h flash supported for this example.")

    # p_points, x_points = _test_range_geo(np.array(p_points), np.array(x_points))
    # p_points = p_points.tolist()
    # x_points = x_points.tolist()
    f_num = len(x_points) * len(p_points)
    f_count = 1

    for X in x_points:
        for P in p_points:
            try:
                if flash_type == "p-T":
                    state = flasher.flash(P=P, T=X, zs=FEED_geo)
                else:
                    state = flasher.flash(P=P, H=X, zs=FEED_geo)
            except Exception:
                logger.warn(
                    f"\nThermo {flash_type} flash failed for p, x = ({P}, {X})\n"
                )
                parsed = _failed_entry(
                    SPECIES_geo, composition_HEADER_geo, fugacity_HEADER_geo
                )
            else:
                # parsed = _thermo_parse_result(state)
                parsed = _failed_entry(
                    SPECIES_geo, composition_HEADER_geo, fugacity_HEADER_geo
                )
                parsed[success_HEADER] = 0
                # in the p-T flash, we use the thermo enthalpy also as target enthalpy
                if flash_type == "p-h":
                    parsed[T_HEADER] = state.T
                else:
                    parsed[h_HEADER] = state.H()
                parsed[gas_frac_HEADER] = float(state.VF)
                ph = ""
                pc = state.phase_count
                if (0.0 < state.VF <= 1.0) and state.gas is not None:
                    ph += "G"
                    if pc > 1:
                        for _ in range(pc - 1):
                            ph += "L"

                    # parsing gas phase composition
                    for i, s in enumerate(SPECIES_geo):
                        parsed[composition_HEADER_geo[s][PHASES[0]]] = state.gas.zs[i]
                else:
                    for _ in range(pc):
                        ph += "L"
                # parsing liquid phase (one expected)
                if state.VF < 1.0:
                    if len(state.liquids) == 1:
                        for i, s in enumerate(SPECIES_geo):
                            parsed[
                                composition_HEADER_geo[s][PHASES[1]]
                            ] = state.liquids[0].zs[i]

                parsed[phases_HEADER] = ph
                parsed[num_iter_HEADER] = 0
            finally:
                parsed[p_HEADER] = P
                if flash_type == "p-h":
                    parsed[h_HEADER] = X
                else:
                    parsed[T_HEADER] = X
                for head, val in parsed.items():
                    results[head].append(val)
                logger.info(f"{del_log}Thermo {flash_type}-flash: {f_count}/{f_num}")
                f_count += 1
    logger.info(f"{del_log}Thermo p-T-flash: done\n")

    return results


def create_mixture(
    verbosity: int,
) -> tuple[pp.composite.Mixture, pp.composite.Flash_c]:
    """Returns instances of the modelled mixture and flash using PorePy's framework.

    ``num_vals`` is an integer indicating how many DOFs per state function are set.
    This is used for vectorization.
    Especially, choose 1 for a point-wise and parallelized flash.

    ``flash_type`` is a string containing the flash type in terms of specificiations.
    This influences the solver settings.

    Configure flash parameters here.

    """

    from porepy.composite.composite_utils import COMPOSITE_LOGGER as logger
    logger.setLevel(logging.DEBUG)
    from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
    from porepy.composite.flash_c import Flash_c
    logger.setLevel(logging.WARNING)

    species = pp.composite.load_species(SPECIES)
    comps = [
        pp.composite.peng_robinson.H2O.from_species(species[0]),
        pp.composite.peng_robinson.CO2.from_species(species[1]),
    ]

    eos = PengRobinsonCompiler(comps)

    phases = [
        pp.composite.Phase(eos, 0, 'L'),
        pp.composite.Phase(eos, 0, 'G'),
    ]

    mix = pp.composite.Mixture(comps, phases)

    flash = Flash_c(mix, eos)
    flash.tolerance = 1e-8
    flash.max_iter = 150

    eos.compile(verbosity=verbosity)
    flash.compile(verbosity=verbosity)

    # instantiating Flasher, without auxiliary variables V and W
    flash.armijo_parameters["rho"] = 0.99
    flash.armijo_parameters["kappa"] = 0.4
    flash.armijo_parameters["j_max"] = 50
    flash.npipm_parameters['u1'] = 1.
    flash.npipm_parameters['u2'] = 10.
    flash.npipm_parameters['eta'] = 0.5
    flash.initialization_parameters['N1'] = 3
    flash.initialization_parameters['N2'] = 1
    flash.initialization_parameters['N3'] = 5

    return mix, flash


def create_mixture_geo(
    verbosity: int,
) -> tuple[pp.composite.NonReactiveMixture, pp.composite.FlashNR]:
    """Analogon to ''create_mixture'' for geothermal fluid misture."""

    from porepy.composite.composite_utils import COMPOSITE_LOGGER as logger
    logger.setLevel(logging.DEBUG)
    from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
    from porepy.composite.flash_c import Flash_c
    logger.setLevel(logging.WARNING)

    species = pp.composite.load_species(SPECIES_geo)

    comps = [
        pp.composite.peng_robinson.H2O.from_species(species[0]),
        pp.composite.peng_robinson.CO2.from_species(species[1]),
        pp.composite.peng_robinson.H2S.from_species(species[2]),
        pp.composite.peng_robinson.N2.from_species(species[3]),
    ]

    eos = PengRobinsonCompiler(comps)

    phases = [
        pp.composite.Phase(eos, 0, 'L'),
        pp.composite.Phase(eos, 0, 'G'),
    ]

    mix = pp.composite.Mixture(comps, phases)

    flash = Flash_c(mix, eos)
    flash.tolerance = 1e-8
    flash.max_iter = 150

    eos.compile(verbosity=verbosity)
    flash.compile(verbosity=verbosity)

    return mix, flash


def calculate_porepy_data(
    state_1: list[float],
    state_2: list[float],
    species: list[str],
    flash_type: str,
    flash: pp.composite.Flash_c,
) -> dict:
    """Performs the PorePy flash for given pressure-temperature points and
    returns a result structure similar to that of the thermo computation.

    If ``quickshot`` is True, returns the results from the initial guess."""

    results = _init_empty_results()

    if flash_type == 'p-T':
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["kappa"] = 0.4
        flash.armijo_parameters["j_max"] = 50
        flash.npipm_parameters['u1'] = 1.
        flash.npipm_parameters['u2'] = 10.
        flash.npipm_parameters['eta'] = 0.5
        flash.initialization_parameters['N1'] = 3
        flash.initialization_parameters['N2'] = 1
        flash.initialization_parameters['N3'] = 5

        equilibrium = {
            'z': FEED,
            'p': state_1,
            'T': state_2, 
        }
    elif flash_type == 'p-h':
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["kappa"] = 0.4
        flash.armijo_parameters["j_max"] = 30
        flash.npipm_parameters['u1'] = 1.
        flash.npipm_parameters['u2'] = 1.
        flash.npipm_parameters['eta'] = 0.5
        flash.initialization_parameters['N1'] = 3
        flash.initialization_parameters['N2'] = 1
        flash.initialization_parameters['N3'] = 5

        equilibrium = {
            'z': FEED,
            'p': state_1,
            'h': state_2, 
        }
    elif flash_type == 'v-h':
        flash.armijo_parameters["rho"] = 0.9
        flash.armijo_parameters["kappa"] = 0.4
        flash.armijo_parameters["j_max"] = 150
        flash.npipm_parameters['u1'] = 1.
        flash.npipm_parameters['u2'] = 10.
        flash.npipm_parameters['eta'] = 0.5
        flash.initialization_parameters['N1'] = 2
        flash.initialization_parameters['N2'] = 2
        flash.initialization_parameters['N3'] = 7

        equilibrium = {
            'z': FEED,
            'v': state_1,
            'h': state_2, 
        }
    else:
        raise NotImplementedError(f'Unknown flash type {flash_type}')
    
    out, success, num_iter = flash.flash(**equilibrium, mode='parallel', verbosity=2)

    results[success_HEADER] = success
    results[num_iter_HEADER] = num_iter

    results[p_HEADER] = out.p
    results[T_HEADER] = out.T

    # to ensure proper mapping between state args
    if flash_type == "p-T":
        results[p_HEADER] = state_1
        results[T_HEADER] = state_2
        results[h_HEADER] = out.h
        results[v_HEADER] = out.v
        results[h_mix_HEADER] = out.h
        results[v_mix_HEADER] = out.v
    elif flash_type == "p-h":
        results[p_HEADER] = state_1
        results[T_HEADER] = out.T
        results[h_HEADER] = state_2
        results[v_HEADER] = out.v
        results[h_mix_HEADER] = out.h
        results[v_mix_HEADER] = out.v
    elif flash_type == "h-v":
        results[p_HEADER] = out.P
        results[T_HEADER] = out.T
        results[h_HEADER] = state_1
        results[v_HEADER] = state_2
        results[h_mix_HEADER] = out.h
        results[v_mix_HEADER] = out.V

    y = out.y[1]
    phases = np.array(['GL'] * y.shape[0])
    phases[y >= 1] = 'G'
    phases[y <= 0] = 'L'
    results[phases_HEADER] = phases
    out[gas_frac_HEADER] = y
    out[liq_frac_HEADER[0]] = 1 - y

    s = out.s[1]
    out[gas_satur_HEADER] = s
    out[liq_satur_HEADER[0]] = 1 - s

    # liquid phase
    if species == SPECIES:
        ch = composition_HEADER
        fh = fugacity_HEADER
    elif species == SPECIES_geo:
        ch = composition_HEADER_geo
        fh = fugacity_HEADER_geo

    for i, s in enumerate(species):
        out[ch[s][PHASES[1]]] = out.phases[1].x[i]
        out[fh[s][PHASES[1]]] = out.phases[1].phis[i]
    # gas phase
    for i, s in enumerate(species):
        out[ch[s][PHASES[0]]] = out.phases[0].x[i]
        out[fh[s][PHASES[0]]] = out.phases[0].phis[i]

    # TODO condition numbers and compressibility factors

    return results


def plot_crit_point_H2O(axis: plt.Axes):
    """Plot critical pressure and temperature in p-T plot for components H2O and CO2."""

    S = pp.composite.load_species(SPECIES)

    img = [
        axis.plot(
            s.T_crit,
            s.p_crit * PRESSURE_SCALE,
            "*",
            markersize=MARKER_SIZE,
            color="red",
        )[0]
        for s in S[:1]
    ]

    return img, [f"crit. point {s.name}" for s in S[:1]]


def plot_max_iter_reached(
    axis: plt.Axes,
    p: np.ndarray,
    T: np.ndarray,
    max_iter_reached: np.ndarray,
) -> figure.Figure:
    """Plots markers where the maximal number of iterations is reached."""
    if np.any(max_iter_reached):
        img = axis.plot(
            T[max_iter_reached],
            p[max_iter_reached] * PRESSURE_SCALE,
            "P",
            markersize=MARKER_SIZE,
            color="black",
        )
        return [img[0]], ["max iter reached"]
    else:
        return [], []


def plot_phase_split_GL(
    axis: plt.Axes,
    p: np.ndarray,
    T: np.ndarray,
    split: np.ndarray,
) -> figure.Figure:
    """Plots a phase split figure across a range of pressure and temperature values."""

    # cmap = mpl.colors.ListedColormap(
    #     ["firebrick", "royalblue", "mediumturquoise", "forestgreen"]
    # )
    cmap = mpl.colors.ListedColormap(np.array([NA_COL, LIQ_COL, MPHASE_COL, GAS_COL]))
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


def _plot_critical_line(axis: plt.Axes, A_mesh: np.ndarray):
    A_CRIT = pp.composite.peng_robinson.PengRobinsonEoS.A_CRIT
    B_CRIT = pp.composite.peng_robinson.PengRobinsonEoS.B_CRIT
    slope = B_CRIT / A_CRIT
    x_vals = np.sort(np.unique(A_mesh.flatten()))
    x_vals = x_vals[x_vals <= A_CRIT]
    y_vals = 0.0 + slope * x_vals
    # critical line
    img_line = axis.plot(x_vals, y_vals, "-", color="black", linewidth=3)
    # critical point
    img_point = axis.plot(A_CRIT, B_CRIT, "*", markersize=MARKER_SIZE, color="red")
    return [img_point[0], img_line[0]], ["(A_c, B_c)", "crit. line"]


def _plot_Widom_line(axis: plt.Axes, A_mesh: np.ndarray, B_mesh: np.ndarray):
    """Plots the approximation of the Widom-line."""
    A_CRIT = pp.composite.peng_robinson.PengRobinsonEoS.A_CRIT
    B_CRIT = pp.composite.peng_robinson.PengRobinsonEoS.B_CRIT
    x_vals = np.sort(np.unique(A_mesh.flatten()))
    x_vals = x_vals[x_vals >= A_CRIT]
    y_vals = pp.composite.peng_robinson.PengRobinsonEoS.Widom_line(x_vals)
    cap = y_vals <= B_mesh.max()
    y_vals = y_vals[cap]
    x_vals = x_vals[cap]

    # Widom line
    img_line = axis.plot(x_vals, y_vals, linestyle="dashed", color="black", linewidth=3)
    # subcrit- triangle
    img_b = axis.plot(
        [A_CRIT, A_mesh.max()],
        [B_CRIT, B_CRIT],
        linestyle="dotted",
        color="black",
        linewidth=3,
    )

    return [img_line[0], img_b[0]], ["Widom line", "B=B_c"]


def plot_root_regions(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    regions: np.ndarray,
    liq_root: np.ndarray,
):
    """A discrete plot for plotting the root cases."""
    # cmap = mpl.colors.ListedColormap(["yellow", "green", "blue", "indigo"])
    cmap = mpl.colors.ListedColormap(np.array([WHITE_COL, MPHASE_COL]))
    img = axis.pcolormesh(
        A_mesh,
        B_mesh,
        regions,
        cmap=cmap,
        vmin=0,
        vmax=3,
        shading="nearest",
    )
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)

    violated = (liq_root <= B_mesh) & (
        B_mesh >= pp.composite.peng_robinson.PengRobinsonEoS.critical_line(A_mesh)
    )
    if np.any(violated):
        mr = np.ma.array(regions, mask=np.logical_not(violated))
        hatch = axis.pcolor(
            A_mesh,
            B_mesh,
            mr,
            hatch="//",
            edgecolor="black",
            cmap=mpl.colors.ListedColormap(["none"]),
            facecolor="none",
            vmin=0,
            vmax=3,
            shading="nearest",
            lw=0,
            zorder=2,
        )
        img_v = [hatch]
        leg_v = [f"Z_L <= B"]
    else:
        img_v = []
        leg_v = []

    axis.legend(
        imgs_c + img_v,
        legs_c + leg_v,
        loc="upper right",
        markerscale=MARKER_SCALE,
    )

    return img


def plot_root_extensions(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    root_extensions: np.ndarray,
):
    """A discrete plot for plotting the root cases."""
    # cmap = mpl.colors.ListedColormap(["white", "royalblue", "orange", "forestgreen"])
    cmap = mpl.colors.ListedColormap(
        np.array([MPHASE_COL, LIQ_COL, GAS_COL_2, GAS_COL])
    )
    img = axis.pcolormesh(
        A_mesh,
        B_mesh,
        root_extensions,
        cmap=cmap,
        vmin=0,
        vmax=3,
        shading="nearest",
    )

    img_w, leg_w = _plot_Widom_line(axis, A_mesh, B_mesh)
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)

    axis.legend(
        imgs_c + img_w, legs_c + leg_w, loc="upper left", markerscale=MARKER_SCALE
    )

    return img


def plot_Widom_points_experimental(axis: plt.Axes):
    """Plots the three points corresponding to the experimental Widom line
    (Maxim et al. 2019)"""

    img = axis.plot(
        WIDOM_LINE[1],
        WIDOM_LINE[0] * PRESSURE_SCALE,
        "D-",
        markersize=MARKER_SIZE,
        color="black",
    )
    return [img[0]], ["Widom-line data"]


def plot_hv_iso(
    axis: plt.Axes,
    x: np.ndarray,
    p_err: np.ndarray,
    T_err: np.ndarray,
    s_err: np.ndarray,
    y_err: np.ndarray,
):
    """Plots the pressure, temperature, saturation and molar fraction error after the
    h-v flash."""
    marker_size = int(np.floor(MARKER_SIZE / 2))
    marker_size = MARKER_SIZE
    img_p = axis.plot(
        x, p_err, "--o", fillstyle="none", color="red", markersize=marker_size
    )[0]
    img_s = axis.plot(
        x, s_err, "--P", fillstyle="none", color="black", markersize=marker_size
    )[0]
    img_T = axis.plot(x, T_err, "-s", color="red", markersize=marker_size)[0]
    img_y = axis.plot(x, y_err, "-D", color="black", markersize=marker_size)[0]

    return [img_p, img_s, img_T, img_y], ["p err", "s err", "T err", "y err"]


def plot_phase_split_GnL(
    axis: plt.Axes,
    p: np.ndarray,
    T: np.ndarray,
    split: np.ndarray,
) -> figure.Figure:
    """Plots a phase split figure across a range of pressure and temperature values."""

    cmap = mpl.colors.ListedColormap(
        np.array([NA_COL, LIQ_COL, MPHASE_COL, GAS_COL, LL_COL, GLL_COL])
    )
    img = axis.pcolormesh(
        T,
        p * PRESSURE_SCALE,
        split,
        cmap=cmap,
        vmin=0,
        vmax=5,
        shading="nearest",  # gouraud
    )

    return img


def plot_conjugate_x_for_px_flash(
    axis: plt.Axes,
    p: np.ndarray,
    x: np.ndarray,
    conjugate_x: np.ndarray,
):
    """Color mesh plot for temperature values in the p-h space."""

    img = axis.pcolormesh(
        x * X_SCALE,
        p * PRESSURE_SCALE,
        conjugate_x,
        cmap=batlow_map,
        shading="nearest",
    )

    return img
