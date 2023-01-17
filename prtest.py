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
n2 = pp.composite.N2(sys)

M.add_component(h2o)
M.add_component(co2)
M.add_component(n2)

temperature = 273.15
pressure = 0.1  # 1 10 20 23
h2o_fraction = 0.99
co2_fraction = 0.005
n22_fraction = 0.005
salt_fraction = 0.01
salt_molality = 3.

sys.set_var_values(h2o.fraction_name, h2o_fraction * vec, True)
sys.set_var_values(co2.fraction_name, co2_fraction * vec, True)
sys.set_var_values(n2.fraction_name, n22_fraction * vec, True)
# brine.set_solute_fractions({brine.NaCl: salt_fraction})
# brine.set_solute_fractions_with_molality({brine.NaCl: salt_molality})

sys.set_var_values(M.T_name, temperature * vec, True)
sys.set_var_values(M.p_name, pressure * vec, True)
sys.set_var_values(M.h_name, 0 * vec, True)

M.initialize()

FLASH = pp.composite.Flash(M)

M.roots.compute_roots()

FLASH.flash("isothermal", 'npipm', 'feed', False, True)
# W = iapws.IAPWS95(P=pressure, T=temperature)
