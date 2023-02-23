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

temperature = 344.44444444444446
pressure = 90178.2993540684 * 1e-6
co2_fraction = 0.01
# n2_fraction = 0.0
h2o_fraction = 0.99

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
FLASH.armijo_parameters["rho"] = 0.99
FLASH.armijo_parameters["j_max"] = 50
FLASH.armijo_parameters["return_max"] = True
FLASH.newton_update_chop = 1.0
FLASH.flash_tolerance = 1e-8
FLASH.max_iter_flash = 140

FLASH.flash("isothermal", "npipm", "rachford_rice", True, True)
# evaluate enthalpy after pT flash
FLASH.evaluate_specific_enthalpy()
FLASH.evaluate_saturations()
# print thermodynamic state stored as STATE in AD
FLASH.print_state()
print("Z LIQ: ", M._phases[0].eos.Z.val)
print("Z GAS: ", M._phases[1].eos.Z.val)
print("---")
print("PHI LIQ: ", [phi.val for phi in M._phases[0].eos.phi.values()])
print("PHI GAS: ", [phi.val for phi in M._phases[1].eos.phi.values()])

# # modifying enthalpy for isenthalpic flash
# h = adsys.get_variable_values(variables=[M.h_name]) * 1.25
# adsys.set_variable_values(h, variables=[M.h_name], to_iterate=True, to_state=False)

# # isenthalpic procedure, storing only as ITERATE
# FLASH.use_armijo = False
# FLASH.flash("isenthalpic", "npipm", "iterate", False, True)
# FLASH.evaluate_saturations(False)
# # print thermodynamic state stored as ITERATE in ad
# FLASH.print_state(True)

print("DONE")
