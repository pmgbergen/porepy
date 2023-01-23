import porepy as pp
import numpy as np

import iapws


M = pp.composite.PR_Composition()
sys = M.ad_system
nc = sys.mdg.num_subdomain_cells()
vec = np.ones(nc)
h2o = pp.composite.H2O(sys)
co2 = pp.composite.CO2(sys)
n2 = pp.composite.N2(sys)

L, G = tuple([p for p in M.phases])

M.add_component(h2o)
M.add_component(co2)
M.add_component(n2)

temperature = 273.15
pressure = 0.1  # 1 10 20 23
h2o_fraction = 0.99
co2_fraction = 0.005
n2_fraction = 0.005
salt_fraction = 0.01
salt_molality = 3.

sys.set_variable_values(h2o_fraction * vec, variables=[h2o.fraction_name], to_iterate=True, to_state=True)
sys.set_variable_values(co2_fraction * vec, variables=[co2.fraction_name], to_iterate=True, to_state=True)
sys.set_variable_values(n2_fraction * vec, variables=[n2.fraction_name], to_iterate=True, to_state=True)
# brine.set_solute_fractions({brine.NaCl: salt_fraction})
# brine.set_solute_fractions_with_molality({brine.NaCl: salt_molality})

sys.set_variable_values(temperature * vec, variables=[M.T_name], to_iterate=True, to_state=True)
sys.set_variable_values(pressure * vec, variables=[M.p_name], to_iterate=True, to_state=True)
sys.set_variable_values(0 * vec, variables=[M.h_name], to_iterate=True, to_state=True)

M.initialize()

FLASH = pp.composite.Flash(M)
FLASH.use_armijo = False

M.roots.compute_roots()

FLASH.flash("isothermal", 'npipm', 'feed', False, True)
FLASH.post_process_fractions()
FLASH.evaluate_specific_enthalpy()
FLASH.evaluate_saturations()
# W = iapws.IAPWS95(P=pressure, T=temperature)

h = sys.get_variable_values(variables=[M.h_name], from_iterate=False) * 1.25
sys.set_variable_values(h, variables=[M.h_name], to_iterate=True, to_state=True)
FLASH.print_state()
print("--------------------------")
FLASH.use_armijo = False
FLASH.flash("isenthalpic", 'npipm', 'iterate', False, True)
FLASH.print_state(True)
FLASH.print_state(False)
FLASH.post_process_fractions()
FLASH.evaluate_saturations()
FLASH.print_state(True)
FLASH.print_state(False)

print("DONE")
