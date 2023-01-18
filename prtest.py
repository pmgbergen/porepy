import porepy as pp
import numpy as np

import iapws


M = pp.composite.PR_Composition()
sys = M.ad_system
dm = M.ad_system.dof_manager
nc = dm.mdg.num_subdomain_cells()
vec = np.ones(nc)
h2o = pp.composite.H2O(sys)
co2 = pp.composite.CO2(sys)
# n2 = pp.composite.N2(sys)

L, G = tuple([p for p in M.phases])

M.add_component(h2o)
M.add_component(co2)
# M.add_component(n2)

temperature = 273.15
pressure = 0.1  # 1 10 20 23
h2o_fraction = 0.99
co2_fraction = 0.01
n22_fraction = 0.005
salt_fraction = 0.01
salt_molality = 3.

sys.set_var_values(h2o.fraction_name, h2o_fraction * vec, True)
sys.set_var_values(co2.fraction_name, co2_fraction * vec, True)
# sys.set_var_values(n2.fraction_name, n22_fraction * vec, True)
# brine.set_solute_fractions({brine.NaCl: salt_fraction})
# brine.set_solute_fractions_with_molality({brine.NaCl: salt_molality})

sys.set_var_values(M.T_name, temperature * vec, True)
sys.set_var_values(M.p_name, pressure * vec, True)
sys.set_var_values(M.h_name, 0 * vec, True)

M.initialize()

FLASH = pp.composite.Flash(M)

M.roots.compute_roots()

print("ROOTS")
print("Liquid: ", M.roots.liquid_root.evaluate(dm).val)
print("Gas: ", M.roots.gas_root.evaluate(dm).val)
print("---")
print("Liquid Log(PHI)")
print("H2O: ", M.log_fugacity_coeffs[h2o][L].evaluate(dm).val)
print("CO2: ", M.log_fugacity_coeffs[co2][L].evaluate(dm).val)
# print("N2: ", M.log_fugacity_coeffs[n2][L].evaluate(dm).val)
print("---")
print("GAS Log(PHI)")
print("H2O: ", M.log_fugacity_coeffs[h2o][G].evaluate(dm).val)
print("CO2: ", M.log_fugacity_coeffs[co2][G].evaluate(dm).val)
# print("N2: ", M.log_fugacity_coeffs[n2][G].evaluate(dm).val)

FLASH.flash("isothermal", 'npipm', 'feed', False, True)
FLASH.print_state(True)
FLASH.print_state(False)
FLASH.post_process_fractions()
FLASH.evaluate_specific_enthalpy()
FLASH.evaluate_saturations()
FLASH.print_state(True)
FLASH.print_state(False)
# W = iapws.IAPWS95(P=pressure, T=temperature)

print("DONE")
