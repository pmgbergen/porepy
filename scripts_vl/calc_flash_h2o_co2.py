"""Script for performing the pT flash for all test cases defined in thermo data files.
"""
import csv
import pathlib
import time

import numpy as np

import porepy as pp

### General settings and data import

# General configuration of this test
h2o_frac = 0.99
co2_frac = 0.01
vectorize = True

# results from which to draw data
# list of (filename, mode),
# where mode indicates if only L (liquid), G (gas) or both (GL) data are available
files = [
    ("data/pr_data_thermo_isothermal_G_easy.csv", "G"),
    ("data/pr_data_thermo_isothermal_G_hard.csv", "G"),
    ("data/pr_data_thermo_isothermal_L_easy.csv", "L"),
    ("data/pr_data_thermo_isothermal_L_hard.csv", "L"),
    ("data/pr_data_thermo_isothermal_GL_easy.csv", "GL"),
    ("data/pr_data_thermo_isothermal_GL_hard.csv", "GL"),
]
# results stored here
version = "w-o-reg-vectorized"
output_file = f"data/results/pr_result_VL_{version}.csv"  # file with flash data
identifier_file = (
    f"data/results/pr_result_VL_{version}_ID.csv"  # file to identify thermo data
)
path = pathlib.Path(__file__).parent.resolve()  # path to script for file i/o

# lists containing pressure and Temperature data for test cases
p_points: list[float] = list()
T_points: list[float] = list()
# to identify file and row per pT point
# (p, T) -> (mode, file name, row id)
pT_id: dict[tuple[float, float], tuple[str, str, int]] = dict()

print("Reading data ...", flush=True)
for filename, mode in files:
    with open(f"{path}/{filename}") as file:
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

### Modelling the composition and performing the Flash

def get_MIX_AD_FLASH(num_cells):
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

if vectorize:
    num_cells = len(p_points)
else:
    num_cells = 1

MIX, AD, FLASH = get_MIX_AD_FLASH(num_cells)
LIQ, GAS = [phase for phase in MIX.phases]
H2O, CO2 = [comp for comp in MIX.components]

# prepare storage of results
success: list[int] = list()  # flag if flash succeeded
y: list[float] = list()  # gas fraction
x_h2o_L: list[float] = list()  # fraction h2o in liquid
x_co2_L: list[float] = list()  # fraction co2 in liquid
x_h2o_G: list[float] = list()  # fraction h2o in gas
x_co2_G: list[float] = list()  # fraction co2 in gas
Z_L: list[float] = list()  # liquid compressibility factor
Z_G: list[float] = list()  # gas compressibility factor

# perform the Flash per pT point
if vectorize:
    print("Performing vectorized flash ...", flush=True)
    start_time = time.time()
    p_vec = np.array(p_points, dtype=np.double) * 1e-6
    T_vec = np.array(T_points)

    AD.set_variable_values(
        p_vec, variables=[MIX.p_name], to_iterate=True, to_state=True,
    )
    AD.set_variable_values(
        T_vec, variables=[MIX.T_name], to_iterate=True, to_state=True
    )
    
    try:
        success_ = FLASH.flash(
            flash_type="isothermal",
            method="npipm",
            initial_guess="rachford_rice",
            copy_to_state=False,  # don't overwrite the state, store as iterate
            do_logging=True,
        )
    except Exception:  # if Flasher fails, flag as failed
            success_ = False

    end_time = time.time()
    print(f"Ended vectorized Flash after {end_time - start_time} seconds.", flush=True)

    # for the vectorized flash, there is no easy way to see which cell failed
    success = [success_ for _ in range(num_cells)]

    try:
        MIX.compute_roots()
    except Exception:
        # if the flash failed, the root computation can fail too
        # store nans as compressibility factors
        Z_L = [np.nan for _ in range(num_cells)]
        Z_G = [np.nan for _ in range(num_cells)]
    else:
        # if successful, store values
        Z_L = list(LIQ.eos.Z.val)
        Z_G = list(GAS.eos.Z.val)

    y = list(GAS.fraction.evaluate(AD).val)
    x_h2o_L = list(LIQ.fraction_of_component(H2O).evaluate(AD).val)
    x_co2_L = list(LIQ.fraction_of_component(CO2).evaluate(AD).val)
    x_h2o_G = list(GAS.fraction_of_component(H2O).evaluate(AD).val)
    x_co2_G = list(GAS.fraction_of_component(CO2).evaluate(AD).val)
else:
    print("Performing point-wise flash ...", flush=True)
    nf = len(p_points)
    start_time = time.time()
    for f, pT in enumerate(zip(p_points, T_points)):
        p, T = pT
        # set thermodynamic state
        # scale from Pa to MPa
        AD.set_variable_values(
            (1e-6 * p) * np.ones(num_cells, dtype=np.double),
            variables=[MIX.p_name],
            to_iterate=True,
            to_state=True,
        )
        AD.set_variable_values(
            T * np.ones(num_cells), variables=[MIX.T_name], to_iterate=True, to_state=True
        )

        # flashing
        try:
            print(f"\r... flash {f}/{nf}", end="", flush=True)
            success_ = FLASH.flash(
                flash_type="isothermal",
                method="npipm",
                initial_guess="rachford_rice",
                copy_to_state=False,  # don't overwrite the state, store as iterate
                do_logging=False,
            )
        except Exception:  # if Flasher fails, flag as failed
            success_ = False

        try:
            MIX.compute_roots()
        except Exception:
            # if the flash failed, the root computation can fail too
            # store nans as compressibility factors
            Z_L.append(np.nan)
            Z_G.append(np.nan)
        else:
            # if successful, store values
            Z_L.append(LIQ.eos.Z.val[0])
            Z_G.append(GAS.eos.Z.val[0])

        # extract and store results from last iterate
        success.append(int(success_))  # store booleans as 0 and 1
        y.append(GAS.fraction.evaluate(AD).val[0])
        x_h2o_L.append(LIQ.fraction_of_component(H2O).evaluate(AD).val[0])
        x_co2_L.append(LIQ.fraction_of_component(CO2).evaluate(AD).val[0])
        x_h2o_G.append(GAS.fraction_of_component(H2O).evaluate(AD).val[0])
        x_co2_G.append(GAS.fraction_of_component(CO2).evaluate(AD).val[0])
    
    end_time = time.time()
    print(f"Ended point-wise Flash after {end_time - start_time} seconds.", flush=True)

### Storing results in files

# flash results
print("Writing results ...", flush=True)
with open(f"{path}/{output_file}", "w", newline="") as csvfile:
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
        ]
        result_writer.writerow(row)

# identifiers
with open(f"{path}/{identifier_file}", "w", newline="") as csvfile:
    id_writer = csv.writer(csvfile, delimiter=",")
    # header labeling column values
    header = ["p [Pa]", "T [K]", "mode", "file", "row-id"]
    id_writer.writerow(header)

    for pT, identifier in pT_id.items():
        p, T = pT
        mode, file, row_id = identifier
        row = [p, T, mode, file, row_id]
        id_writer.writerow(row)
