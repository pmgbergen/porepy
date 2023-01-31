import numpy as np

import porepy as pp

M = pp.composite.PR_Composition()
sys = M.ad_system
nc = sys.mdg.num_subdomain_cells()
h2o = pp.composite.H2O(sys)
co2 = pp.composite.CO2(sys)

components = [h2o, co2]

# EOS with gas or liquid role
GAS = pp.composite.PR_EoS(gaslike=True)
LIQ = pp.composite.PR_EoS(gaslike=False)

# setting components
GAS.components = components
LIQ.components = components


# Here, any combination of floats, arrays and Ad_arrays is possible
# as long as numpy can broadcast them properly!
# with below we mimic basically a computation in 3 cells
# where each cell has a different temperature, the same pressure and the same fractions

# Arrays with different length, and Ad_arrays with different lengths in vals and
# different row numbers won't work, since they cannot be broadcasted by numpy

pressure = np.array([0.1])  # MPa
temperature = np.array([300, 310, 400])  # K
# Enthalpy in kJ

x_co2 = np.array(
    [1.0]
)  # here an array with length 2 or >=4 would cause a broadcasting error
x_h2o = 1 - x_co2

# computing everything for gas
GAS.compute(pressure, temperature, x_h2o, x_co2)
# print state
print(GAS.Z)
print(GAS.h_dep)
print(list(GAS.phi.values()))

# computing everything for liquid
LIQ.compute(pressure, temperature, x_h2o, x_co2)
# print state
print(LIQ.Z)
print(LIQ.h_dep)
print(list(LIQ.phi.values()))
