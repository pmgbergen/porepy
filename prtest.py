import numpy as np

import porepy as pp

M = pp.composite.PR_Composition()
adsys = M.ad_system
nc = adsys.mdg.num_subdomain_cells()
vec = np.ones(nc)
h2o = pp.composite.H2O(adsys)
co2 = pp.composite.CO2(adsys)
# n2 = pp.composite.N2(sys)

LIQ = pp.composite.PR_Phase(adsys, False, name="L")
GAS = pp.composite.PR_Phase(adsys, True, name="G")

M.add_components([h2o, co2])
M.add_phases([LIQ, GAS])

temperature = 300
pressure = 0.1  # 1 10 20 23
co2_fraction = 0.6
n2_fraction = 0.0
h2o_fraction = 1 - co2_fraction - n2_fraction

adsys.set_variable_values(
    h2o_fraction * vec, variables=[h2o.fraction_name], to_iterate=True, to_state=True
)
adsys.set_variable_values(
    co2_fraction * vec, variables=[co2.fraction_name], to_iterate=True, to_state=True
)
# sys.set_variable_values(
#     n2_fraction * vec, variables=[n2.fraction_name], to_iterate=True, to_state=True
# )

adsys.set_variable_values(
    temperature * vec, variables=[M.T_name], to_iterate=True, to_state=True
)
adsys.set_variable_values(
    pressure * vec, variables=[M.p_name], to_iterate=True, to_state=True
)
adsys.set_variable_values(0 * vec, variables=[M.h_name], to_iterate=True, to_state=True)

M.initialize()

FLASH = pp.composite.Flash(M, auxiliary_npipm=False)
FLASH.use_armijo = True
FLASH.armijo_parameters["rho"] = 0.9
FLASH.armijo_parameters["j_max"] = 150
FLASH.newton_update_chop = 0.5

FLASH.flash("isothermal", "npipm", "feed", True, True)
# evaluate enthalpy after pT flash
FLASH.evaluate_specific_enthalpy()
FLASH.evaluate_saturations()
# print thermodynamic state stored as STATE in ad
FLASH.print_state()

# modifying enthalpy for isenthalpic flash
h = adsys.get_variable_values(variables=[M.h_name]) * 1.25
adsys.set_variable_values(h, variables=[M.h_name], to_iterate=True, to_state=False)

# isenthalpic procedure, storing only as ITERATE
FLASH.use_armijo = False
FLASH.flash("isenthalpic", "npipm", "iterate", False, True)
FLASH.evaluate_saturations(False)
# print thermodynamic state stored as ITERATE in ad
FLASH.print_state(True)  # print state with temperature values after isenthalpic flash

print("DONE")
