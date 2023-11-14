import os

# os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"
os.environ["NUMBA_CACHE_DIR"] = f"{str(os.path.dirname(__file__))}/__pycache__/"
# os.environ['NUMBA_DEBUG_CACHE'] = str(1)

import numpy as np

import porepy as pp
import time
import matplotlib.pyplot as plt

from porepy.composite.peng_robinson.eos_compiler import PengRobinson_c
from porepy.composite.flash_c import Flash_c

vec = np.ones(3)
z = [vec * 0.99, vec * 0.01]
p = vec * 5e6
T = vec * 500
verbosity = 3

x_test = np.array([0.01, 5e6, 500, 0.9, 0.1, 0.2, 0.3, 0.4])

chems = ["H2O", "CO2"]
species = pp.composite.load_species(chems)
comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

phases = [
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinson(gaslike=False), name="L"
    ),
    pp.composite.Phase(pp.composite.peng_robinson.PengRobinson(gaslike=True), name="G"),
]

mix = pp.composite.NonReactiveMixture(comps, phases)

mix.set_up()
[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

eos_c = PengRobinson_c(mix)
flash = Flash_c(mix, eos_c)

flash.armijo_parameters["rho"] = 0.99
flash.armijo_parameters["j_max"] = 150
flash.tolerance = 1e-7
flash.max_iter = 150

flash.compile(verbosity=verbosity)

result, success, num_iter = flash.flash(z, p = p, T= T, mode='parallel', verbosity=verbosity)
result, success, num_iter = flash.flash(z, p = p, T= T, mode='parallel', verbosity=verbosity)

print(success)
print(num_iter)
print(result)