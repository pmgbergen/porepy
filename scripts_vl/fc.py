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

vec = np.ones(1)
verbosity = 2
chems = ["H2O", "CO2"]
feed = [0.99, 0.01]
z = [vec * _ for _ in feed]

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

eos_c = PengRobinsonCompiler(mix, verbosity=verbosity)
flash_c = Flash_c(mix, eos_c)
flash_c.tolerance = 1e-8
flash_c.max_iter = 150

eos_c.compile(verbosity=verbosity)
flash_c.compile(verbosity=verbosity)


# test p-T
p = vec * 1e6
T = vec * 450.
print('--- Test runs ')
flash_c.armijo_parameters["rho"] = 0.99
flash_c.armijo_parameters["kappa"] = 0.4
flash_c.armijo_parameters["j_max"] = 50
flash_c.npipm_parameters['u1'] = 1.
flash_c.npipm_parameters['u2'] = 10.
flash_c.npipm_parameters['eta'] = 0.5
flash_c.initialization_parameters['N1'] = 3
flash_c.initialization_parameters['N2'] = 1
flash_c.initialization_parameters['N3'] = 5
# result, success, num_iter = flash_c.flash(z, p = p, T= T, mode='parallel', verbosity=verbosity)
result, success, num_iter = flash_c.flash(z, p = p, T= T, mode='linear', verbosity=verbosity)
print("---\n")

# test p-h
p = vec * 1e6
h = vec * 7335.055860939756
flash_c.armijo_parameters["rho"] = 0.99
flash_c.armijo_parameters["kappa"] = 0.4
flash_c.armijo_parameters["j_max"] = 30
flash_c.npipm_parameters['u1'] = 1.
flash_c.npipm_parameters['u2'] = 1.
flash_c.npipm_parameters['eta'] = 0.5
flash_c.initialization_parameters['N1'] = 3
flash_c.initialization_parameters['N2'] = 1
flash_c.initialization_parameters['N3'] = 5
result, success, num_iter = flash_c.flash(z, p=p, h=h, verbosity=verbosity)

# test v-h
v = vec * 3.267067077646246e-05
h = vec * (-18911.557739855507)
flash_c.armijo_parameters["rho"] = 0.9
flash_c.armijo_parameters["kappa"] = 0.4
flash_c.armijo_parameters["j_max"] = 150
flash_c.npipm_parameters['u1'] = 1.
flash_c.npipm_parameters['u2'] = 10.
flash_c.npipm_parameters['eta'] = 0.5
flash_c.initialization_parameters['N1'] = 2
flash_c.initialization_parameters['N2'] = 2
flash_c.initialization_parameters['N3'] = 7
result, success, num_iter = flash_c.flash(z, h=h, v=v, verbosity=verbosity)


# parallel p-T test
p_range = [1e6, 50e6]
T_range = [450., 700.]
refinement = 80

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
