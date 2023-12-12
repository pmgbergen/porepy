# import os

# os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"
# os.environ["NUMBA_CACHE_DIR"] = f"{str(os.path.dirname(__file__))}/__pycache__/"
# os.environ['NUMBA_DEBUG_CACHE'] = str(1)
import logging
import numpy as np
import porepy as pp

from porepy.composite.composite_utils import COMPOSITE_LOGGER as logger
logger.setLevel(logging.DEBUG)
from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
from porepy.composite.flash_c import Flash_c
logger.setLevel(logging.WARNING)

chems = ["H2O", "CO2"]
feed = [0.99, 0.01]
p_range = [1e6, 50e6]
T_range = [450., 700.]
refinement = 80

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

vec = np.ones(1)
p = vec * 1e6
T = vec * 453.16455696
verbosity = 2
z = [vec * _ for _ in feed]
[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

eos_c = PengRobinsonCompiler(mix, verbosity=verbosity)
flash_c = Flash_c(mix, eos_c)

flash_c.armijo_parameters["rho"] = 0.99
flash_c.armijo_parameters["j_max"] = 50
flash_c.npipm_parameters['u2'] = 10.
flash_c.tolerance = 1e-8
flash_c.max_iter = 150

eos_c.compile(verbosity=verbosity)
flash_c.compile(verbosity=verbosity)

print('--- Test runs ')
result, success, num_iter = flash_c.flash(z, p = p, T= T, mode='linear', verbosity=verbosity)
result, success, num_iter = flash_c.flash(z, p = p, T= T, mode='parallel', verbosity=verbosity)
print("---\n")

p_vec = np.linspace(p_range[0], p_range[1], refinement, endpoint=True, dtype=np.float64)
T_vec = np.linspace(T_range[0], T_range[1], refinement, endpoint=True, dtype=np.float64)

T_mesh, p_mesh = np.meshgrid(T_vec, p_vec)

T = T_mesh.flatten()
p = p_mesh.flatten()
z = [np.ones(p.shape[0]) * _ for _ in feed]

result, success, num_iter = flash_c.flash(z, p = p, T= T, mode='parallel', verbosity=verbosity)

investigate = success == 1
print("Investigate p-T:")
print(p[investigate])
print(T[investigate])

# flash = pp.composite.FlashNR(mix)
# flash.use_armijo = True
# flash.armijo_parameters["rho"] = 0.99
# flash.armijo_parameters["j_max"] = 50
# flash.armijo_parameters["return_max"] = True
# flash.npipm_parameters["u2"] = 10.0
# flash.newton_update_chop = 1.0
# flash.tolerance = 1e-8
# flash.max_iter = 150


# success_o, result_o = flash.flash(
#     state={'p': p[investigate], 'T': T[investigate]},
#     feed=[z_[investigate] for z_ in z],
#     eos_kwargs={"apply_smoother": True},
#     quickshot=True,
#     return_system=False,
#     verbosity=2,
# )

# flash_c.max_iter = 0
# result, success, num_iter = flash_c.flash([z_[investigate] for z_ in z], p = p[investigate], T= T[investigate], mode='parallel', verbosity=verbosity)

# diff = result.diff(result_o)

# tol = 1e-3
# print('diff p', np.any(diff.p > tol))
# print('diff T', np.any(diff.T > tol))
# for j in range(2):
#     idx = diff.y[j] > tol
#     print(f'diff y_{j}', np.any(idx))
