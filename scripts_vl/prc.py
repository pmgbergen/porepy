import os

# os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"
os.environ["NUMBA_CACHE_DIR"] = f"{str(os.path.dirname(__file__))}/__pycache__/"
# os.environ['NUMBA_DEBUG_CACHE'] = str(1)

import numpy as np

import porepy as pp
import sympy as sm
import numba
import time
import matplotlib.pyplot as plt

from typing import Callable

from porepy.composite import safe_sum

from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
from porepy.composite.flash_c import parse_xyz, Flash_c

vec = np.ones(1)
z = [vec * 0.99, vec * 0.01]
p = vec * 5e6
T = vec * 500
verbosity = 0

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

eos_c = PengRobinsonCompiler(mix)
PRC = Flash_c(mix, eos_c)

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.99
flash.armijo_parameters["j_max"] = 150
flash.newton_update_chop = 1.0
flash.tolerance = 1e-7
flash.max_iter = 150

# NPIPM
ncomp = mix.num_components
nphase = mix.num_phases
mpmc = (nphase, ncomp)

u = 1.0
eta = 0.5

F_pT = PRC.residuals["p-T"]
DF_pT = PRC.jacobians["p-T"]


def pos_symbolic(a: sm.Expr) -> sm.Expr:
    return sm.Piecewise((a, a > 0.0), (0, a <= 0.0))


def neg_symbolic(a: sm.Expr) -> sm.Expr:
    return sm.Piecewise((a, a < 0.0), (0, a >= 0.0))


V_s = [sm.Symbol(f"v_{j}") for j in range(1, nphase + 1)]
W_s = [sm.Symbol(f"w_{j}") for j in range(1, nphase + 1)]
nu_s = sm.Symbol("nu")

slack_arg = [V_s, W_s, nu_s]  # two vector args, one scalar arg

# decrease of nu
slack_equ_e = eta * nu_s + nu_s**2
# punishment of negative parts (original inequality cosntraint)
slack_equ_e += (
    safe_sum([neg_symbolic(v) ** 2 + neg_symbolic(w) ** 2 for v, w in zip(V_s, W_s)])
    / 2
)
# punishment of positive parts (CC violation)
slack_equ_e += (
    (pos_symbolic(safe_sum([v * w for v, w in zip(V_s, W_s)]))) ** 2
    * u
    / nphase**2
    / 2
)

d_slack_equ_e = (
    [slack_equ_e.diff(_) for _ in V_s]
    + [slack_equ_e.diff(_) for _ in W_s]
    + [slack_equ_e.diff(nu_s)]
)

slack_equ_c = sm.lambdify(slack_arg, slack_equ_e, "math")
slack_equ_c = numba.njit("float64(float64[:], float64[:], float64)", fastmath=True)(
    slack_equ_c
)


def _njit_diff(f, **njit_kwargs):
    """Helper function for special wrapping of some compiled derivative of slack equation."""

    f = numba.njit(f, **njit_kwargs)

    @numba.njit("float64[:](float64[:], float64[:], float64)", **njit_kwargs)
    def inner(a, b, c):
        return np.array(f(a, b, c))

    return inner


d_slack_equ_c = sm.lambdify(slack_arg, d_slack_equ_e, "math")
d_slack_equ_c = _njit_diff(d_slack_equ_c, fastmath=True)


@numba.njit("float64[:](float64[:])")
def F_npipm(X: np.ndarray) -> np.ndarray:
    X_thd = X[:-1]
    nu = X[-1]
    x, y, _ = parse_xyz(X_thd, mpmc)

    f_flash = F_pT(X_thd)

    # couple complementary conditions with nu
    f_flash[-nphase:] -= nu

    # NPIPM equation
    unity_j = np.zeros(nphase)
    for j in range(nphase):
        unity_j[j] = 1.0 - np.sum(x[j])

    # complete vector of fractions
    slack = slack_equ_c(y, unity_j, nu)

    # NPIPM system has one equation more at end
    f_npipm = np.zeros(f_flash.shape[0] + 1)
    f_npipm[:-1] = f_flash
    f_npipm[-1] = slack

    return f_npipm


@numba.njit("float64[:,:](float64[:])")
def DF_npipm(X: np.ndarray) -> np.ndarray:
    X_thd = X[:-1]
    nu = X[-1]
    x, y, _ = parse_xyz(X_thd, mpmc)

    df_flash = DF_pT(X_thd)

    # NPIPM matrix has one row and one column more
    df_npipm = np.zeros((df_flash.shape[0] + 1, df_flash.shape[1] + 1))
    df_npipm[:-1, :-1] = df_flash
    # relaxed complementary conditions read as y * (1 - sum x) - nu
    # add the -1 for the derivative w.r.t. nu
    df_npipm[-(nphase + 1) : -1, -1] = np.ones(nphase) * (-1)

    # derivative NPIPM equation
    unity_j = np.zeros(nphase)
    for j in range(nphase):
        unity_j[j] = 1.0 - np.sum(x[j])

    # complete vector of fractions
    d_slack = d_slack_equ_c(y, unity_j, nu)
    # d slack has derivatives w.r.t. y_j and w_j
    # d w_j must be expanded since w_j = 1 - sum x_j
    # d y_0 must be expanded since reference phase is eliminated by unity
    expand_yr = np.ones(nphase - 1) * (-1)
    expand_x_in_j = np.ones(ncomp) * (-1)

    # expansion of y_0 and cut of redundant value
    d_slack[1 : nphase - 1] += d_slack[0] * expand_yr
    d_slack = d_slack[1:]

    # expand it also to include possibly other derivatives
    d_slack_expanded = np.zeros(df_npipm.shape[1])
    # last derivative is w.r.t. nu
    d_slack_expanded[-1] = d_slack[-1]

    for j in range(nphase):
        # derivatives w.r.t. x_ij, +2 because nu must be skipped and j starts with 0
        vec = expand_x_in_j * d_slack[-(j + 2)]
        d_slack_expanded[-(1 + (j + 1) * ncomp) : -(1 + j * ncomp)] = vec

    # derivatives w.r.t y_j. j != r
    d_slack_expanded[
        -(1 + nphase * ncomp + nphase - 1) : -(1 + nphase * ncomp)
    ] = d_slack[: nphase - 1]

    df_npipm[-1] = d_slack_expanded

    return df_npipm


@numba.njit(
    "Tuple((float64[:,:], float64[:]))(float64[:], float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
)
def npipm_regularization(X, A, b):
    x, y, _ = parse_xyz(X[:-1], mpmc)

    reg = 0.0
    for j in range(nphase):
        reg += y[j] * (1 - np.sum(x[j]))

    reg = 0.0 if reg < 0 else reg
    reg *= u / ncomp**2

    # subtract all relaxed complementary conditions multiplied with reg from the slack equation
    b[-1] = b[-1] - reg * np.sum(b[-(nphase + 1) : -1])
    # do the same for respective rows in the Jacobian
    for j in range(nphase):
        # +2 to skip slack equation and because j start with 0
        v = A[-(j + 2)] * reg
        A[-1] = A[-1] - v

    return A, b


# Numerical solvers
@numba.njit("float64(float64[:])", fastmath=True, cache=True)
def l2_potential(vec: np.ndarray) -> float:
    return np.sum(vec * vec) / 2.0


@numba.njit
def Armijo_line_search(
    Xk: np.ndarray,
    dXk: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
) -> float:
    fk = F(Xk)
    potk = l2_potential(fk)
    rho_j = 0.99
    kappa = 0.4
    j_max = 150

    for j in range(1, j_max + 1):
        rho_j = rho_j**j

        try:
            fk_j = F(Xk + rho_j * dXk)
        except:
            continue

        potk_j = l2_potential(fk_j)

        if potk_j <= (1 - 2 * kappa * rho_j) * potk:
            return rho_j

    # return max
    return rho_j


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
    DX = np.zeros_like(X0)

    f_i = F(X)

    if np.linalg.norm(f_i) <= tol:
        success = 0
        return X, success, num_iter
    else:
        for i in range(1, max_iter + 1):
            num_iter += 1

            df_i = DF(X)

            A, b = npipm_regularization(X, df_i, -f_i)

            dx = np.linalg.solve(A, b)

            # X contains also parameters (p, T, z_i, ...)
            # exactly ncomp - 1 feed fractions and 2 state definitions (p-T, p-h, ...)
            # for broadcasting insert solution into new vector
            DX[ncomp - 1 + 2 :] = dx

            s = Armijo_line_search(X, DX, F)

            X = X + s * DX

            f_i = F(X)

            if np.linalg.norm(f_i) <= tol:
                success = 0
                break

    return X0, success, num_iter


@numba.njit(
    parallel=True,
    nogil=True,
    boundscheck=False,
)
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
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in numba.prange(N):

        res_i, conv_i, n_i = newton(X0[n], F, DF, tol, max_iter)
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-F_dim:]

    return result, converged, num_iter


_, oldsys = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
    quickshot=True,
    return_system=True,
)

initstate = oldsys.state
oldsys_eval = oldsys(initstate, True)
x_test[3:] = initstate[:-1]

# dimension of  per flash: 1 z_i, 2 thd state (p, T), 1 y_j, 4 x_ij, 1 nu (npipm)
xdim = 1 + 2 + 1 + 4 + 1

x_, y_, _ = parse_xyz(x_test, mpmc)
y_ = np.concatenate([np.array([1 - np.sum(y_)]), y_])
nu0 = safe_sum([y__ * (1 - np.sum(x__)) for y__, x__ in zip(y_, x_)]) / nphase

x_test = np.concatenate([x_test, np.array([nu0])])

newval = F_npipm(x_test)
newjac = DF_npipm(x_test)
# print('new val')
# print(newval)
# print('old val')
# print(oldsys_eval.val)
# print('diff')
# print(np.abs(newval - oldsys_eval.val))
# print('new matrix')
# print(newjac)
# print('old matrix')
# print(oldsys_eval.jac.todense())
# print('diff')
# print(np.abs(newjac - oldsys_eval.jac.todense()))

### PAR TESTING

success, results_old = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
old_iter = flash.history[-1]["iterations"]


tolerance = 1e-4
NF_l = [10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000]
# NF_l = [10, 100, 1000, 10000, 100000, 1000000]
# NF_l = [10]
times_l = []

for n in NF_l:
    X_total = np.repeat(x_test.reshape((1, len(x_test))), n, axis=0)

    print(f"------ CALL PAR n={n}")
    start = time.time()
    result, conv, num_iter = par_newton(X_total, F_npipm, DF_npipm, 6, 1e-7, 150)
    end = time.time()
    times_l.append(end - start)

    print(f"----- RESULTS FOR n={n}, tol = {tolerance}, time = {times_l[-1]} s")

    if not np.all(conv == 0):
        print(f"{len(conv[np.logical_not(conv==0)])} cases not converged")
    else:
        print(f"All cases converged.")

    idx = num_iter == old_iter
    if not np.all(idx):
        print(
            f"{len(num_iter[np.logical_not(idx)])} mismatches in number of iterations."
        )
        print(f"Should be {old_iter}, got {num_iter[:4]} ...")
    else:
        print("Number of iterations matches")

    if len(np.unique(num_iter)) != 1:
        print(
            f"Heterogeneous number of iterations: {len(np.unique(num_iter))} different results."
        )

    idx = np.isclose(result[:, 0], results_old.y[1][0], rtol=0, atol=tolerance)
    if not np.all(idx):
        print(f"{len(result[:, 0][np.logical_not(idx)])} mismatches in results for y.")
        print(f"Max error: {np.max(np.abs(result[:, 0] - results_old.y[1][0]))}")
    else:
        print("No mismatch for y")

    idx = np.isclose(result[:, 1], results_old.X[0][0][0], rtol=0, atol=tolerance)
    if not np.all(idx):
        print(
            f"{len(result[:, 1][np.logical_not(idx)])} mismatches in results for x_11."
        )
        print(f"Max error: {np.max(np.abs(result[:, 1] - results_old.X[0][0][0]))}")
    else:
        print("No mismtach for x_11.")

    idx = np.isclose(result[:, 2], results_old.X[0][1][0], rtol=0, atol=tolerance)
    if not np.all(idx):
        print(
            f"{len(result[:, 2][np.logical_not(idx)])} mismatches in results for x_21."
        )
        print(f"Max error: {np.max(np.abs(result[:, 2] - results_old.X[0][1][0]))}")
    else:
        print("No mismtach for x_21.")

    idx = np.isclose(result[:, 3], results_old.X[1][0][0], rtol=0, atol=tolerance)
    if not np.all(idx):
        print(
            f"{len(result[:, 3][np.logical_not(idx)])} mismatches in results for x_12."
        )
        print(f"Max error: {np.max(np.abs(result[:, 3] - results_old.X[1][0][0]))}")
    else:
        print("No mismtach for x_12.")

    idx = np.isclose(result[:, 4], results_old.X[1][1][0], rtol=0, atol=tolerance)
    if not np.all(idx):
        print(
            f"{len(result[:, 4][np.logical_not(idx)])} mismatches in results for x_22."
        )
        print(f"Max error: {np.max(np.abs(result[:, 4] - results_old.X[1][1][0]))}")
    else:
        print("No mismtach for x_22.")

    idx = np.isclose(result[:, 5], 0.0, rtol=0, atol=tolerance)
    if not np.all(idx):
        print(f"{len(result[:, 5][np.logical_not(idx)])} violations for NPIPM slack.")
        print(f"Max error: {np.max(np.abs(result[:, 5] - 0.))}")
    else:
        print("No violation for NPIPM slack.")

    print(f"-----")

print("Computational times:")
print(times_l)
print("for N as")
print(NF_l)

# copied from output of regular run
NF_l_test = [10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000]
times_test = [
    60.24044609069824,
    0.0008132457733154297,
    0.00570988655090332,
    0.062392473220825195,
    0.6264514923095703,
    6.295071363449097,
    36.79759645462036,
    76.59307980537415,
]

print("\nPlotting ..", flush=True)
fig = plt.figure(figsize=(10, 10))
fig.suptitle(f"Residual tolerance: 1e-7")
axis = fig.add_subplot(1, 1, 1)
axis.set_box_aspect(1)
img_a = axis.plot(NF_l_test, times_test, "-*", color="black")
img_m = axis.plot(NF_l, times_l, "--*", color="red")
axis.legend(img_a + img_m, ["test case - 3 conv", "worst case - max conv"], loc='lower right')

xt = axis.get_xticks()

for i, v in zip(NF_l_test, times_test):
    axis.text(i, 1.5 * v, "%.4f" % v, ha="center")

for i, v in zip(NF_l, times_l):
    axis.text(i, 1.5 * v, "%.4f" % v, ha="center")

axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel('Number of flashes')
axis.set_ylabel('Total execution time [s]')

fig.tight_layout()
fig.savefig(
    "parflash_times.png",
    format="png",
    dpi=300,
)
