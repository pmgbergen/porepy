import sys

sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import numpy as np
from iapws import IAPWS95
from matplotlib import pyplot as plt

import porepy as pp

### PARAMETRIZATION
p = 101.320  # 101 kPa
T = 343.15  # 100 deg C
h = T + p
salt_fraction = 0.5
k_salt = 0.1
k_water = 10

### CALCULATION
c = pp.composite.Composition()
ad_system = c.ad_system
dm = ad_system.dof_manager
mdg = dm.mdg
nc = mdg.num_subdomain_cells()

water = pp.composite.H2O(ad_system)
salt = pp.composite.NaCl(ad_system)
c.add_component(water)
c.add_component(salt)

ad_system.set_var_values(water.fraction_name, (1-salt_fraction) * np.ones(nc), True)
ad_system.set_var_values(salt.fraction_name, salt_fraction * np.ones(nc), True)
ad_system.set_var_values(c._p_var, p * np.ones(nc), True)
ad_system.set_var_values(c._T_var, T * np.ones(nc), True)
ad_system.set_var_values(c._h_var, np.zeros(nc), True)

c.k_values = {
    water: k_water,
    salt: k_salt
}

c.initialize()

success = c.isothermal_flash(copy_to_state=False, initial_guess="feed")
c.evaluate_saturations(False)
c.evaluate_specific_enthalpy(False)

c.print_state()
c.print_state(True)

print("Done")

# k_vals = [0.1, 0.5, 1., 1.001, 1.01, 1.1, 1.5, 2., 5., 10., 50., 100.,]
# s_l = list()
# s_v = list()
# y_l = list()
# y_v = list()
# x_w_l = list()
# x_s_l = list()
# x_w_v = list()
# h = list()

# iter_nums = list()

# for k in k_vals:

#     equilibrium = vapor.ext_fraction_of_component(water) - k * brine.ext_fraction_of_component(water)
#     c.add_equilibrium_equation(water, equilibrium, "k_water")
    
#     c.initialize()

#     c.isothermal_flash(copy_to_state=True, initial_guess="feed")
#     print("\nk value: ", k)
#     c.print_last_flash()
#     c.evaluate_saturations()
#     c.evaluate_specific_enthalpy()

#     X = c.ad_system.dof_manager.assemble_variable()

#     s_l.append(X[c.ad_system.dof_manager.dof_var([brine.saturation_name])])
#     s_v.append(X[c.ad_system.dof_manager.dof_var([vapor.saturation_name])])
#     y_l.append(X[c.ad_system.dof_manager.dof_var([brine.fraction_name])])
#     y_v.append(X[c.ad_system.dof_manager.dof_var([vapor.fraction_name])])
#     x_s_l.append(X[c.ad_system.dof_manager.dof_var([brine.component_fraction_name(salt)])])
#     x_w_l.append(X[c.ad_system.dof_manager.dof_var([brine.component_fraction_name(water)])])
#     x_w_v.append(X[c.ad_system.dof_manager.dof_var([vapor.component_fraction_name(water)])])

#     enthalpy = c.h * c.density()
#     enthalpy = enthalpy.evaluate(dm).val
#     h.append(enthalpy[0])

#     i = c.flash_history[-1]['iterations']
#     iter_nums.append(i)

# plt.figure()
# plt.loglog(k_vals, y_l, color="green", marker="o", linestyle="solid", label="frac L")
# plt.loglog(k_vals, y_v, color="red", marker="o", linestyle="solid", label="frac V")
# plt.loglog(k_vals, s_l, color="green", marker="v", linestyle="dashed", label="satur L")
# plt.loglog(k_vals, s_v, color="red", marker="v", linestyle="dashed", label="satur V")
# plt.loglog(k_vals, x_w_v, color="grey", marker=".", linestyle="dotted", label="W in V")
# plt.loglog(k_vals, x_w_l, color="blue", marker="*", linestyle="dotted", label="W in L")
# plt.loglog(k_vals, x_s_l, color="black", marker="*", linestyle="dotted", label="S in L")
# plt.loglog(k_vals, h, color="orange", marker="^", linestyle="dashdot", label="enthalpy")

# plt.xlabel("k-value")
# plt.ylabel("fractions")
# plt.grid(True, "both")
# plt.title(f"Unified p-T-flash; p={p} Pa, T={T} K, salt={salt_fraction} %")
# # plt.xticks(k_vals)
# plt.legend(loc="upper right", framealpha=1.)

# plt.show()