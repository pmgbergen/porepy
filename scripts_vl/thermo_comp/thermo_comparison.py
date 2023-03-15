"""Module containing functionality to compute thermo data, and read/write results.

It is set-up for a water-CO2 VLE problem, with fixed CO2 feed fractions.

"""

import csv
import pathlib
from typing import Any, Literal

import numpy as np
from thermo import (  # PRMIX,; FlashVL,
    PR78MIX,
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVLN,
)
from thermo.interaction_parameters import IPDB

PhaseType = Literal["L", "G", "GL", "GL+", "L+"]
"""Phases at equilibrium:

- Liquid,
- Gas,
- Gas-Liquid,
- Gas-Liquid+ with multiple Liquids,
- Liquid+ multiple but only Liquids.

"""

# CSV files: headers for data columns
DELIMITER: str = ","
FAILED_ENTRY: Any = "failed"  # entry for a failed flash
NAN_ENTRY: np.nan = np.nan  # entry for a points where thermo delivers no meaning
MISSING_ENTRY: str = "missing"  # entry for missing entries, sanity check for scripts
COMPONENTS: list[str] = ["water", "CO2"]
MAX_LIQ_PHASES: int = 2  # maximal number of anticipated liquid phases
FEED: dict[str, float] = {
    "water": 0.99,
    "CO2": 0.01,
}
PHASES: list[str] = ["G"] + [f"L{i}" for i in range(1, MAX_LIQ_PHASES + 1)]

# All files produced by these scripts should have at least the following headers
# for identifier files
file_name_HEADER: str = "file name"
row_id_HEADER: str = "f-row"
phases_HEADER: str = "phase-split"
# for result data files
success_HEADER: str = "success"
p_HEADER: str = "p [Pa]"
T_HEADER: str = "T [K]"
h_HEADER: str = "h [J/mol]"
h_thermo_HEADER: str = "h (thermo) [J/mol]"
gas_frac_HEADER: str = "y"
liq_frac_HEADER: list[str] = [f"y_L{i}" for i in range(1, MAX_LIQ_PHASES + 1)]
compressibility_HEADER: dict[str, str] = dict([(f"{p}", f"Z_{p}") for p in PHASES])
composition_HEADER: dict[str, dict[str, str]] = dict(
    [(c, dict([(f"{p}", f"x_{c}_{p}") for p in PHASES])) for c in COMPONENTS]
)
fugacity_HEADER: dict[str, dict[str, str]] = dict(
    [(c, dict([(f"{p}", f"phi_{c}_{p}") for p in PHASES])) for c in COMPONENTS]
)
# for files containing own results for analysis
num_iter_HEADER: str = "num-iter"
cond_start_HEADER: str = "cond-start"
cond_end_HEADER: str = "cond-end"
is_supercrit_HEADER: str = "is-supercrit"

RESOLUTION: int = 50

P_LIMITS: list[float] = [0.01e6, 100.0e6]  # [Pa]
T_LIMITS: list[float] = [280, 700]  # [K]
H_LIMITS: list[float] = [-30000.0, 10000.0]  # [J/mol]


def path():
    """Returns path to script calling this function as string."""
    return str(pathlib.Path(__file__).parent.resolve())


def get_result_headers() -> list[str]:
    """Returns the headers for results computed using PorePy's flash."""
    headers: list[str] = [success_HEADER, num_iter_HEADER, is_supercrit_HEADER]
    headers += [cond_start_HEADER, cond_end_HEADER]
    headers += [p_HEADER, T_HEADER, h_HEADER, gas_frac_HEADER]
    # only one liquid phase
    headers += [compressibility_HEADER[p] for p in PHASES[:2]]
    for c in COMPONENTS:
        for p in PHASES[:2]:
            headers.append(composition_HEADER[c][p])
    for c in COMPONENTS:
        for p in PHASES[:2]:
            headers.append(fugacity_HEADER[c][p])
    return headers


def read_headers(filename: str) -> list[str]:
    """Reads and returns the first line of a file."""
    print(f"Reading headers: file {filename}", flush=True)
    with open(f"{path()}/{filename}", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER)
        headers = next(reader)
    print(f"Reading headers: done", flush=True)
    return headers


def read_px_data(
    files: list[str],
    x: str,
) -> tuple[
    list[float],
    list[float],
    dict[tuple[float, float], tuple[str, int, PhaseType]],
]:
    """Reads from a list of files containing correct headers for pressure,
    x and row-id the respective data.

    Returns a list of pressure values, a list of x and an identification map
    which gives the file name, row-id and phases for each p-x point.

    Use this to read p-T or p-h points from a file, i.e. ``x='T'`` or ``x='h'``

    The files are assumed to be created by this module.

    """
    p_points: list[float] = list()
    x_points: list[float] = list()
    # to identify file and row per pT point
    # (p, x) -> (file name, row id, phases)
    px_id: dict[tuple[float, float], tuple[str, int, PhaseType]] = dict()

    if x == "T":
        x_HEADER = T_HEADER
    elif x == "h":
        x_HEADER = h_HEADER
    else:
        raise NotImplementedError(f"Unknown x for p-x data: {x}")

    for filename in files:
        print(f"Reading p-{x} data: file {filename}", flush=True)
        with open(f"{path()}/{filename}", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=DELIMITER)
            headers = next(reader)  # get rid of header

            id_idx = headers.index(row_id_HEADER)
            p_idx = headers.index(p_HEADER)
            x_idx = headers.index(x_HEADER)
            phase_idx = headers.index(phases_HEADER)

            for row in reader:

                # must always be available
                row_id = int(row[id_idx])
                p = float(row[p_idx])
                # potentially missing or nan
                x_ = row[x_idx]
                if x_ not in [MISSING_ENTRY, NAN_ENTRY]:
                    x_ = float(x_)
                phases = str(row[phase_idx])

                px = (p, x_)
                # get only unique points
                if px not in px_id:
                    identifier = (filename, row_id, phases)
                    px_id.update({px: identifier})

                    p_points.append(p)
                    x_points.append(x_)
    print(f"Reading p-{x} data: done", flush=True)

    # sanity check to see if nothing messed with the precision during I/O
    # and data is readable
    test = list()
    mapping_failure = list()
    for p, x_ in zip(p_points, x_points):
        try:
            test.append(px_id[(p, x_)])
        except KeyError:
            mapping_failure.append((p, x_))
    assert len(test) == len(p_points) == len(x_points), f"p-{x} mapping failed"
    assert (
        len(mapping_failure) == 0
    ), f"p-{x} mapping failed for points: {mapping_failure}"

    return p_points, x_points, px_id


def write_px_identifier(
    filename: str,
    px_data: dict[tuple[float, float], tuple[str, int, PhaseType]],
    x: str,
):
    """Write the identification map obtained from ``read_thermo_px_data`` into a file.

    This is useful when data from multiple files is drawn.

    ``x`` must be either ``'T'`` or ``'h'``.

    """
    if x == "T":
        x_HEADER = T_HEADER
    elif x == "h":
        x_HEADER = h_HEADER
    else:
        raise NotImplementedError(f"Unknown x for p-x data: {x}")

    with open(f"{path()}/{filename}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=DELIMITER)

        headers = [p_HEADER, x_HEADER, file_name_HEADER, row_id_HEADER, phases_HEADER]
        writer.writerow(headers)

        print(f"Writing p-{x} identifier map: file {filename}", flush=True)
        for px, identifier in px_data.items():
            p, x_ = px
            fname, row_id, phases = identifier
            row = [p, x_, fname, row_id, phases]
            writer.writerow(row)
    print(f"Writing p-{x} identifier map: done", flush=True)


def read_px_identifiers(
    files: list[str],
) -> dict[tuple[float, float], tuple[str, int, PhaseType]]:
    """Read identification maps obtained from ``read_thermo_px_data`` and written
    by ``write_px_identifier``.

    This is useful when data from multiple files is drawn.

    """
    supported_x_data = [T_HEADER, h_HEADER]

    px_id: dict[tuple[float, float], tuple[str, int, PhaseType]] = dict()

    print("Reading p-x identifier maps: ...", end="", flush=True)
    for filename in files:
        with open(f"{path()}/{filename}", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=DELIMITER)

            headers = next(reader)

            # sanity check
            assert (
                p_HEADER in headers
                and file_name_HEADER in headers
                and row_id_HEADER in headers
                and phases_HEADER in headers
            ), f"Cannot parse identifier file: Unknown headers {headers}"

            p_idx = headers.index(p_HEADER)
            f_idx = headers.index(file_name_HEADER)
            row_idx = headers.index(row_id_HEADER)
            phases_idx = headers.index(phases_HEADER)

            if T_HEADER in headers:
                x_HEADER = T_HEADER
                x = "T"
            elif h_HEADER in headers:
                x_HEADER = h_HEADER
                x = "h"
            else:
                raise NotImplementedError(
                    f"Cannot parse identifier file {filename}:"
                    + f"\nUnable to detect supported x-data {supported_x_data}"
                )

            x_idx = headers.index(x_HEADER)

            print(
                f"\rReading p-{x} identifier map: file {filename}", end="", flush=True
            )
            for row in reader:
                p = float(row[p_idx])
                x = float(row[x_idx])
                fname = str(row[f_idx])
                row_id = str(row[row_idx])
                phases = str(row[phases_idx])

                px = (p, x)
                if px not in px_id:
                    px_id.update({px: (fname, row_id, phases)})
        print(f"\rReading p-{x} identifier map: done", end="", flush=True)
    print("\rReading p-x identifier maps: done", flush=True)
    return px_id


def write_results(filename: str, results: dict[str, list]):
    """Writes results to file. Results must be a dictionary with headers as keys
    and data columns as values."""
    headers = [header for header, _ in results.items()]
    data = [col for _, col in results.items()]

    print(f"Writing result data: file {filename}", flush=True)
    with open(f"{path()}/{filename}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=DELIMITER)
        # wrting header
        writer.writerow(headers)
        # writing data
        for row in zip(*data):
            writer.writerow(row)
    print("Writing result data: done", flush=True)


def read_results(files: list[str], headers: list[str]) -> dict[str, list]:
    """Reads data previously written with ``write_results``.

    Use ``headers`` to specify which header to read.
    They must be consistent throughout all files.

    The returned dictionary contains per header the data column.

    """
    data: dict[str, list] = dict([(header, list()) for header in headers])

    print("Reading result data: ...", end="", flush=True)
    for filename in files:
        with open(f"{path()}/{filename}", mode="r") as file:
            reader = csv.reader(file, delimiter=DELIMITER)
            print(f"\rReading result data: file {filename}", end="", flush=True)
            file_headers = next(reader)
            try:
                header_idx: dict[str, int] = dict(
                    [(header, file_headers.index(header)) for header in headers]
                )
            except KeyError as err:
                raise KeyError(f"Unable to find headers in file {filename}: {str(err)}")

            for row in reader:
                for header in headers:
                    idx = header_idx[header]
                    d = row[idx]
                    data[header].append(d)
    print(f"\rReading result data: done", flush=True)
    return data


def _parse_thermo_state(state) -> dict[str, Any]:
    """Helper function to parse a state returned by thermo into processable format."""
    out: dict[str, Any] = create_results_structure()
    for k in out.keys():
        out[k] = MISSING_ENTRY

    out.update(
        {
            success_HEADER: 1,
            gas_frac_HEADER: state.VF,
        }
    )
    # anticipate at max only 1 gas phase and predefined number of liquid phases
    if 0 < state.phase_count <= 1 + MAX_LIQ_PHASES:
        if (
            0.0 < state.VF <= 1.0
        ) and state.gas is not None:  # parse gas phase if present
            p = PHASES[0]
            out.update(
                {
                    phases_HEADER: p,
                    compressibility_HEADER[p]: state.gas.Z(),
                }
            )
            out.update(
                dict(
                    [
                        (composition_HEADER[c][p], state.gas.zs[COMPONENTS.index(c)])
                        for c in COMPONENTS
                    ]
                    + [
                        (fugacity_HEADER[c][p], state.gas.phis()[COMPONENTS.index(c)])
                        for c in COMPONENTS
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
            p = PHASES[0]
            out.update(
                {
                    phases_HEADER: "",
                    compressibility_HEADER[p]: NAN_ENTRY,
                }
            )
            out.update(
                dict(
                    [(composition_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                    + [(fugacity_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                )
            )

        if (
            state.gas is not None and state.VF == 1.0
        ):  # if only gas, fill liquid entries with nans
            # for anticipated liquid phases
            for liq_frac in liq_frac_HEADER:
                out.update({liq_frac: 0.0})
            for p in PHASES[1:]:
                out.update(
                    {
                        compressibility_HEADER[p]: NAN_ENTRY,
                    }
                )
                out.update(
                    dict(
                        [(composition_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                        + [(fugacity_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                    )
                )
        else:  # parse present liquid phases
            # sanity check
            assert (
                state.VF < 1.0
            ), "Thermo conflicting gas state: Gas saturated with liquid phases"
            if len(state.liquids) == 1:  # if only one liquid phase
                out[phases_HEADER] = out[phases_HEADER] + "L"
                p = PHASES[1]
                out.update({liq_frac_HEADER[0]: 1 - state.VF})
                for yl in liq_frac_HEADER[1:]:
                    out.update({yl: 0})
                out.update(
                    {
                        compressibility_HEADER[p]: state.liquids[0].Z(),
                    }
                )
                out.update(
                    dict(
                        [
                            (
                                composition_HEADER[c][p],
                                state.liquids[0].zs[COMPONENTS.index(c)],
                            )
                            for c in COMPONENTS
                        ]
                        + [
                            (
                                fugacity_HEADER[c][p],
                                state.liquids[0].phis()[COMPONENTS.index(c)],
                            )
                            for c in COMPONENTS
                        ]
                    )
                )
                # fill other liquid phases with nans
                for p in PHASES[2:]:
                    out.update(
                        {
                            compressibility_HEADER[p]: NAN_ENTRY,
                        }
                    )
                    out.update(
                        dict(
                            [(composition_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                            + [(fugacity_HEADER[c][p], NAN_ENTRY) for c in COMPONENTS]
                        )
                    )
            elif 1 < len(state.liquids) <= MAX_LIQ_PHASES:  # get all liquid data
                assert (
                    state.liquids_betas
                ), "Thermo conflicting liquid phase state: no liquid betas"
                for i in range(MAX_LIQ_PHASES):
                    out.update({liq_frac_HEADER[i]: state.liquids_betas[i]})
                for p_idx, p in enumerate(PHASES[1:]):
                    out[phases_HEADER] = out[phases_HEADER] + "L"
                    out.update(
                        {
                            compressibility_HEADER[p]: state.liquids[p_idx].Z(),
                        }
                    )
                    out.update(
                        dict(
                            [
                                (
                                    composition_HEADER[c][p],
                                    state.liquids[p_idx].zs[COMPONENTS.index(c)],
                                )
                                for c in COMPONENTS
                            ]
                            + [
                                (
                                    fugacity_HEADER[c][p],
                                    state.liquids[p_idx].phis()[COMPONENTS.index(c)],
                                )
                                for c in COMPONENTS
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


def _init_thermo() -> FlashVLN:
    """Helper function to initiate the thermo flasher and the results data structure."""
    constants, properties = ChemicalConstantsPackage.from_IDs(COMPONENTS)
    kijs = IPDB.get_ip_asymmetric_matrix("ChemSep PR", constants.CASs, "kij")
    eos_kwargs = {
        "Pcs": constants.Pcs,
        "Tcs": constants.Tcs,
        "omegas": constants.omegas,
        "kijs": kijs,
    }

    print("---Thermo flash---")
    print(f"Binary system with components: {COMPONENTS}")
    print(f"Overal composition: {FEED}")
    print(f"Critical pressures: {constants.Pcs}")
    print(f"Critical temperatures: {constants.Tcs}")
    print(f"Acentric factors: {constants.omegas}")
    print(f"BIPs: {kijs}")

    GAS = CEOSGas(
        PR78MIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases
    )
    LIQs = [
        CEOSLiquid(
            PR78MIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        for _ in range(MAX_LIQ_PHASES)
    ]
    flasher = FlashVLN(constants, properties, liquids=LIQs, gas=GAS)

    return flasher


def create_results_structure() -> dict[str, list]:
    """Initiate and return an results dict with proper headers as needed for the
    comparison."""
    results: dict[str, list] = {
        row_id_HEADER: list(),
        success_HEADER: list(),
        p_HEADER: list(),
        T_HEADER: list(),
        h_HEADER: list(),
        h_thermo_HEADER: list(),
        phases_HEADER: list(),
        gas_frac_HEADER: list(),
    }
    results.update(dict([(liq_frac_HEADER[i], list()) for i in range(MAX_LIQ_PHASES)]))
    results.update(dict([(compressibility_HEADER[p], list()) for p in PHASES]))
    results.update(
        dict([(composition_HEADER[c][p], list()) for p in PHASES for c in COMPONENTS])
    )
    results.update(
        dict([(fugacity_HEADER[c][p], list()) for p in PHASES for c in COMPONENTS])
    )

    return results


def get_failed_entry() -> dict[str, Any]:
    """Create a row-entry for failed flashes."""
    failed: dict[str, Any] = create_results_structure()
    for k in failed.keys():
        failed[k] = FAILED_ENTRY
    failed[success_HEADER] = 0
    return failed


def thermo_ph_flash(
    p_limits: list[float], h_limits: list[float], p_res: int = 10, h_res: int = 10
) -> dict[str, list]:
    """Uses thermo to perform the p-h flash for various pressure and enthalpy ranges.

    Returns a dictionary containing per header (name of some property) respective values
    per p-h point.

    The tolerance for how far the resulting thermo enthalpies are allowed to deviate
    from the input enthalpy can be specified by ``h_tol``.

    """

    flasher = _init_thermo()
    results = create_results_structure()

    p_points = np.linspace(p_limits[0], p_limits[1], num=p_res).tolist()
    h_points = np.linspace(h_limits[0], h_limits[1], num=h_res).tolist()

    flash_num = len(h_points) * len(p_points)
    flash_count = 1
    zs = list(FEED.values())

    for H in h_points:
        for P in p_points:
            try:
                state = flasher.flash(P=P, H=H, zs=zs)
            except Exception as err:
                print(
                    f"Flash: {flash_count}/{flash_num} FAILED\n{str(err)}", flush=True
                )
                parsed = get_failed_entry()
            else:
                parsed = _parse_thermo_state(state)
                # sanity check
                assert state.P == P, "Thermo p-h result has different pressure."
                # store additionally the target enthalpy
                # thermo enthalpy has some slight deviations
                parsed[h_thermo_HEADER] = state.H()
                parsed[T_HEADER] = state.T
            finally:
                parsed[p_HEADER] = P
                parsed[h_HEADER] = H
                parsed[row_id_HEADER] = flash_count
                for head, val in parsed.items():
                    results[head].append(val)
                print(f"\rFlash: {flash_count}/{flash_num} done", end="", flush=True)
                flash_count += 1
    print("", flush=True)

    return results


def thermo_pT_flash(
    p_limits: list[float], T_limits: list[float], p_res: int = 10, T_res: int = 10
) -> dict[str, list]:
    """Uses thermo to perform the p-T flash for various pressure and enthalpy ranges.

    Returns a dictionary containing per header (name of some property) respective values
    per p-T point
    """

    flasher = _init_thermo()
    results = create_results_structure()

    p_points = np.linspace(p_limits[0], p_limits[1], num=p_res).tolist()
    T_points = np.linspace(T_limits[0], T_limits[1], num=T_res).tolist()

    flash_num = len(T_points) * len(p_points)
    flash_count = 1
    zs = list(FEED.values())

    for T in T_points:
        for P in p_points:
            try:
                state = flasher.flash(P=P, T=T, zs=zs)
            except Exception as err:
                print(
                    f"Flash: {flash_count}/{flash_num} FAILED\n{str(err)}", flush=True
                )
                parsed = get_failed_entry()
            else:
                parsed = _parse_thermo_state(state)
                # sanity check
                assert state.P == P, "Thermo p-T result has different pressure."
                assert state.T == T, "Thermo p-T result has different temperature."
                # in the p-T flash, we use the thermo enthalpy also as target enthalpy
                parsed[h_thermo_HEADER] = state.H()
                parsed[h_HEADER] = state.H()
            finally:
                parsed[p_HEADER] = P
                parsed[T_HEADER] = T
                parsed[row_id_HEADER] = flash_count
                for head, val in parsed.items():
                    results[head].append(val)
                print(f"\rFlash: {flash_count}/{flash_num} done", end="", flush=True)
                flash_count += 1
    print("", flush=True)

    return results
