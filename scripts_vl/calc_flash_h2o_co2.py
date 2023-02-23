"""Script for performing the pT flash for all test cases defined in thermo data files.
"""
import csv
import pathlib
import numpy as np
import porepy as pp

### General settings and data import

# General configuration of this test
h2o_frac = 0.99
co2_frac = 0.01

# results from which to draw data
# list of (filename, mode),
# where mode indicates if only L (liquid), G (gas) or both (GL) data are available
files = [
    ('data/pr_data_thermo_isothermal_G_easy.csv', 'G'),
    ('data/pr_data_thermo_isothermal_G_hard.csv', 'G'),
    ('data/pr_data_thermo_isothermal_L_easy.csv', 'L'),
    ('data/pr_data_thermo_isothermal_L_hard.csv', 'L'),
    ('data/pr_data_thermo_isothermal_GL_easy.csv', 'GL'),
    ('data/pr_data_thermo_isothermal_GL_hard.csv', 'GL'),
    # ('data/testdata.csv', 'G')
]
# results stored here
output_file = 'data/results/pr_result_VL_reg-v-smooth.csv'  # file with flash data
identifier_file = 'data/results/pr_result_VL_reg-v-smooth_ID.csv'  # file to identify thermo data
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
            p = float(datarow[1]) * 1e-6  # scale from Pa to MPa
            T = float(datarow[2])

            pT = (p, T)
            # get only unique points
            if pT not in pT_id:
                identifier = (mode, filename, row_id)
                pT_id.update({
                    pT: identifier
                })

                p_points.append(p)
                T_points.append(T)

### Modelling the composition and performing the Flash

nc = 1  # single cell domain, one flash calc per pT point
MIX = pp.composite.PR_Composition(nc=nc)
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
    h2o_frac * np.ones(nc), variables=[H2O.fraction_name], to_iterate=True, to_state=True
)
AD.set_variable_values(
    co2_frac * np.ones(nc), variables=[CO2.fraction_name], to_iterate=True, to_state=True
)
# Setting zero enthalpy to get the AD framework going (h is irrelevant here)
AD.set_variable_values(
    0 * np.ones(nc), variables=[MIX.h_name], to_iterate=True, to_state=True
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
print("Performing flash ...", flush=True)
for p, T in zip(p_points, T_points):
    # set thermodynamic state
    AD.set_variable_values(
        p * np.ones(nc), variables=[MIX.p_name], to_iterate=True, to_state=True
    )
    AD.set_variable_values(
        T * np.ones(nc), variables=[MIX.T_name], to_iterate=True, to_state=True
    )

    # flashing
    try:
        success_ = FLASH.flash(
            flash_type='isothermal',
            method='npipm',
            initial_guess='rachford_rice',
            copy_to_state=False,  # don't overwrite the state, store as iterate
            do_logging=False
        )
    except Exception as err:  # if Flasher fails, flag as failed
        success_ = False

    # compute thermodynamic properties from last iterate
    try:
        MIX.compute_roots()
    except Exception as err:
        # if the flash failed, the root computation can fail too
        # store nans as compressibility factors
        Z_L.append(np.nan)
        Z_G.append(np.nan)
    else:
        # if successful, store values
        Z_L.append(
            LIQ.eos.Z.val[0]
        )
        Z_G.append(
            GAS.eos.Z.val[0]
        )
    
    # extract and store results from last iterate
    success.append(int(success_))  # store booleans as 0 and 1
    y.append(
        GAS.fraction.evaluate(AD).val[0]
    )
    x_h2o_L.append(
        LIQ.fraction_of_component(H2O).evaluate(AD).val[0]
    )
    x_co2_L.append(
        LIQ.fraction_of_component(CO2).evaluate(AD).val[0]
    )
    x_h2o_G.append(
        GAS.fraction_of_component(H2O).evaluate(AD).val[0]
    )
    x_co2_G.append(
        GAS.fraction_of_component(CO2).evaluate(AD).val[0]
    )


### Storing results in files

# flash results
print("Writing results ...", flush=True)
with open(f"{path}/{output_file}", 'w', newline='') as csvfile:
    result_writer = csv.writer(csvfile, delimiter=',')
    # header labeling column values
    header = [
        'p [MPa]', 'T [K]', 'success', 'y', 'x_h2o_L', 'x_co2_L', 'x_h2o_G', 'x_co2_G', 'Z_L', 'Z_G'
    ]
    result_writer.writerow(header)

    for i, pT in enumerate(zip(p_points, T_points)):
        p, T = pT
        row = [
            p, T, success[i], y[i], x_h2o_L[i], x_co2_L[i], x_h2o_G[i], x_co2_G[i], Z_L[i], Z_G[i]
        ]
        result_writer.writerow(row)

# identifiers
with open(f"{path}/{identifier_file}", 'w', newline='') as csvfile:
    id_writer = csv.writer(csvfile, delimiter=',')
    # header labeling column values
    header = [
        'p [MPa]', 'T [K]', 'mode', 'file', 'row-id'
    ]
    id_writer.writerow(header)

    for pT, identifier in pT_id.items():
        p, T = pT
        mode, file, row_id = identifier
        row = [
            p, T, mode, file, row_id
        ]
        id_writer.writerow(row)