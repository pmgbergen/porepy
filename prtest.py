import porepy as pp
import numpy as np

import iapws


M = pp.composite.PR_Composition()
sys = M.ad_system
dm = M.ad_system.dof_manager
nc = dm.mdg.num_subdomain_cells()
vec = np.ones(nc)
brine = pp.composite.NaClBrine(sys)
co2 = pp.composite.CO2(sys)

M.add_component(brine)
M.add_component(co2)

temperature = 400
pressure = 20  # 1 10 20 23
brine_fraction = 0.8
salt_fraction = 0.01
salt_molality = 3.

sys.set_var_values(brine.fraction_name, brine_fraction * vec, True)
brine.set_solute_fractions({brine.NaCl: salt_fraction})
# brine.set_solute_fractions_with_molality({brine.NaCl: salt_molality})
sys.set_var_values(co2.fraction_name, (1 - brine_fraction) * vec, True)

sys.set_var_values(M.T_name, temperature * vec, True)
sys.set_var_values(M.p_name, pressure * vec, True)
sys.set_var_values(M.h_name, 0 * vec, True)

M.initialize()

FLASH = pp.composite.Flash(M)

M.roots.compute_roots()

FLASH.flash("isothermal", 'npipm', 'feed', False, True)
# W = iapws.IAPWS95(P=pressure, T=temperature)
