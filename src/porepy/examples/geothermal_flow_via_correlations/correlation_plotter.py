import numpy as np
import matplotlib.pyplot as plt
from DriesnerBrineOBL import DriesnerBrineOBL

folder_name = 'figures_wp1'
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

taylor_extended = True
file_name = "binary_files/PHX_l0_with_gradients.vtk"
brine_obl = DriesnerBrineOBL(file_name, taylor_extended)
brine_obl.conversion_factors = (1.0, 1.0, 10.0)  # (z,h,p)

def compose_figure_name(folder_name, p_val, z_val, suffix):
    fig_name = folder_name + '/'
    fig_name += 'p_' + str(p_val)
    fig_name += '_z_' + str(z_val)
    fig_name += suffix
    return fig_name


pressure_val = 20.0
z_NaCl_val = 0.02

at_label = '(p,z_NaCl) = ' + '(' + str(pressure_val) + ', ' + str(z_NaCl_val) + ')'

h = np.arange(0.1e3, 3.4e3, 0.05e3)
p = pressure_val * np.ones_like(h)

z_NaCl = z_NaCl_val * np.ones_like(h)
par_points = np.array((z_NaCl, h, p)).T
brine_obl.sample_at(par_points)

T = brine_obl.sampled_could.point_data["Temperature"]
plt.plot(h, T, label='T at ' + at_label, color='blue', linestyle='-', marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend()
plt.xlabel("Mixture specific enthalpy [J/Kg]")
plt.ylabel("Temperature [K]")
fig_temp = compose_figure_name(folder_name,pressure_val, z_NaCl_val, '_temperature.png')
plt.savefig(fig_temp)
plt.clf()

dT_DH = brine_obl.sampled_could.point_data["grad_Temperature"][:,1]
plt.plot(h, dT_DH, label='dT/dH at ' + at_label, color='blue', linestyle='-', marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend()
plt.xlabel("Mixture specific enthalpy [J/Kg]")
plt.ylabel("Rate [K/J/Kg]")
fig_dtdH = compose_figure_name(folder_name,pressure_val, z_NaCl_val, '_dt_dH.png')
plt.savefig(fig_dtdH)
plt.clf()

s_l = brine_obl.sampled_could.point_data["S_l"]
s_v = brine_obl.sampled_could.point_data["S_v"]
plt.plot(h, s_l, label='Liquid', color='blue', linestyle='-', marker='o',
         markerfacecolor='blue', markersize=5)
plt.plot(h, s_v, label='Vapor', color='red', linestyle='-', marker='o',
         markerfacecolor='red', markersize=5)
fig_sat = compose_figure_name(folder_name, pressure_val, z_NaCl_val, '_saturations.png')
plt.legend()
plt.xlabel("Mixture specific enthalpy [J/Kg]")
plt.ylabel("Saturations [-]")
plt.savefig(fig_sat)
plt.clf()


Rho = brine_obl.sampled_could.point_data["Rho"]
plt.plot(h, Rho, label='Rho at ' + at_label, color='blue', linestyle='-', marker='o',
         markerfacecolor='blue', markersize=5)
fig_rho = compose_figure_name(folder_name, pressure_val, z_NaCl_val, '_mixture_rho.png')
plt.legend()
plt.xlabel("Mixture specific enthalpy [J/Kg]")
plt.ylabel("Mixture mass density [Kg/m3]")
plt.savefig(fig_rho)
plt.clf()
