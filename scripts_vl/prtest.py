import numpy as np

import porepy as pp

MIX = pp.composite.PengRobinsonMixture()
ads = MIX.AD.system
nc = ads.mdg.num_subdomain_cells()

h2o = pp.composite.H2O(ads)
co2 = pp.composite.CO2(ads)
LIQ = pp.composite.PR_Phase(ads, False, name="L")
GAS = pp.composite.PR_Phase(ads, True, name="G")

MIX.add([h2o, co2], [LIQ, GAS])

temperature = 700
pressure = 23965794.8140 * 1e-6
co2_fraction = 0.01
h2o_fraction = 0.99
vec = np.ones(nc)

ads.set_variable_values(
    h2o_fraction * vec, variables=[h2o.fraction.name], to_iterate=True, to_state=True
)
ads.set_variable_values(
    co2_fraction * vec, variables=[co2.fraction.name], to_iterate=True, to_state=True
)

MIX.AD.set_up()

ads.set_variable_values(
    temperature * vec, variables=[MIX.AD.T.name], to_iterate=True, to_state=True
)
ads.set_variable_values(
    pressure * vec, variables=[MIX.AD.p.name], to_iterate=True, to_state=True
)
ads.set_variable_values(
    0 * vec, variables=[MIX.AD.h.name], to_iterate=True, to_state=True
)

FLASH = pp.composite.Flash(MIX, auxiliary_npipm=False)
FLASH.use_armijo = True
FLASH.armijo_parameters["rho"] = 0.99
FLASH.armijo_parameters["j_max"] = 50
FLASH.armijo_parameters["return_max"] = True
FLASH.newton_update_chop = 1.0
FLASH.flash_tolerance = 1e-8
FLASH.max_iter_flash = 140

FLASH.flash("pT", "npipm", "rachford_rice", True, True)
MIX.precompute(apply_smoother=False)
# evaluate enthalpy after pT flash
FLASH.post_process_fractions(True)
FLASH.evaluate_specific_enthalpy(True)
# print thermodynamic state stored as STATE in AD
FLASH.print_state()
print("Z LIQ: ", MIX._phases[0].eos.Z.val)
print("Z GAS: ", MIX._phases[1].eos.Z.val)
print("---")
print("PHI LIQ: ", [phi.val for phi in MIX._phases[0].eos.phi.values()])
print("PHI GAS: ", [phi.val for phi in MIX._phases[1].eos.phi.values()])

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
