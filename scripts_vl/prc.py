import numpy as np

import porepy as pp
import numba

numba.warnings.simplefilter('ignore', numba.NumbaExperimentalFeatureWarning)

from typing import Callable

vec = np.ones(1)
z = [vec * 0.99,  vec * 0.01]
p = vec * 5e6
T = vec * 500
verbosity = 2

t = np.array([0.01, 5e6, 500, 0.9, 0.1, 0.2, 0.3, 0.4])

chems = ["H2O", "CO2"]
species = pp.composite.load_species(chems)
comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

phases = [
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinsonEoS(gaslike=False), name="L"
    ),
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinsonEoS(gaslike=True), name="G"
    ),
]

mix = pp.composite.NonReactiveMixture(comps, phases)

mix.set_up()
[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

PRC = pp.composite.peng_robinson.PR_Compiler(mix)

print("test call ---")
print(PRC.equations['p-T'](t))
print(PRC.jacobians['p-T'](t))
print("-------------")

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.9
flash.armijo_parameters["j_max"] = 150
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-5
flash.max_iter = 150

# success, results_pT = flash.flash(
_, oldsys = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
    quickshot=True,
    return_system=True,
)
# print("Results p-T:\n" + "------------")
# print(str(results_pT))
# print("------------")

initstate = oldsys.state
oldsys_eval = oldsys(initstate, True)
print("old value", oldsys_eval.val)
print("old matrix")
print(oldsys_eval.jac.todense()[:-1, :-1])

t[3:] = initstate[:-1]

newval = PRC.equations['p-T'](t)
newjac = PRC.jacobians['p-T'](t)
print('new val')
print(newval)
print('new matrix')
print(newjac)

print("diff vals", np.linalg.norm(newval - oldsys_eval.val[:-1]))
print("diff jac", np.linalg.norm(newjac - oldsys_eval.jac.todense()[:-1, :-1]))


@numba.njit
def newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo step-size."""

    num_iter = 0
    success = 1  # 1 means max iter reached

    X = X0.copy()
    f_i = F(X)

    if np.linalg.norm(f_i) <= tol:
        success = 0
        return X, success, num_iter
    else:
        for i in range(1, max_iter + 1):
            num_iter += i

            df_i = DF(X)

            X -= np.linalg.solve(df_i, -f_i)

            f_i = F(X)

            if np.linalg.norm(f_i) <= tol:
                success = 0
                break

        return X, success, num_iter


@numba.njit(parallel=True)
def par_newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    F_dim: int,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Newton, assuming each row in ``X0`` is a starting point to find a root of ``F``.
    
    Note that ``X0`` can contain parameters for the evaluation of ``F``.
    Therefore the dimension of the image of ``F`` must be defined ``F_dim``.
    
    I.e., ``len(F(X0[i])) == F_dim`` and ``DF(X0[i]).shape == (F_dim, F_dim)``"""

    N = X0.shape[0]
    result = np.empty((N, F_dim))
    num_iter = np.zeros(N, dtype=int)
    converged = np.zeros(N, dtype=int)

    for n in numba.prange(N):

        res_i, conv_i, n_i = newton(X0[n], F, DF, tol, max_iter)
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i
    
    return result, converged, num_iter
