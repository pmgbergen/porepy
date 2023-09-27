"""DEPRECATED. Contains only example code for later development of tests."""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import sympy as sm

from ..mixture import NonReactiveMixture


class PR_Compiler:
    """Class implementing JIT-compiled representation of the equilibrium equations
    using numba and sympy, based on the Peng-Robinson EoS.

    It uses the no-python mode of numba to produce compiled code with near-C-efficiency.

    """

    def __init__(self, mixture: NonReactiveMixture) -> None:
        self._n_p = mixture.num_phases
        self._n_c = mixture.num_components

        self._sysfuncs: dict[str, Callable | list[Callable]] = dict()
        """A collection of relevant functions, which must not be dereferenced."""

        self._gaslike: list[int] = []
        """List containing gaslike flags per phase as integers."""

        self._Z_cfuncs: dict[str, Callable] = dict()
        """A map containing compiled functions for the compressibility factor,
        dependeng on A and B.

        The functions represent computations for each root-case of the characteristic
        polynomial, as well as the gradient w.r.t. A and B."""

        self.unknowns: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        independent variables as symbols."""

        self.arguments: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        arguments required for the evaluation of the equations and Jacobians.

        The arguments include the fixed values and the unknowns.
        The fixed values for each flash type are the respective specifications e.g.,
        for the p-T flash it includes the symbol for p and T, followed by the symbolic
        unknowns for the p-T flash."""

        self.residuals: dict[Literal["p-T", "p-h", "v-h"], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the equilibrium equations
        represented by multivariate, vector-valued callables."""

        self.jacobians: dict[Literal["p-T", "p-h", "v-h"], Callable] = dict()
        """A map between flash-types (p-T, p-h,...) and the Jacobian of respective
        equilibrium equations."""

        self.cfuncs: dict[str, Callable] = dict()
        """Contains a collection of numba-compiled callables representing thermodynamic
        properties. (nopython, just-in-time compilation)

        Keys are names of the properties. Standard symbols from literature are used.

        Several naming conventions apply:

        - ``_cv``: compiled and vectorized. Arguments can be numpy arrays and the return
          value is an array
          (see `here <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_).
          If ``_cv`` is not indicated, the function takes only scalar input.
        - ``dx_y``: The partial derivative of ``y`` w.r.t. ``x``.
        - ``d_y``: The complete derivative of ``y``. If ``y`` is multivariate,
          the return value is an array of length equal the number of dependencies
          (in respective order).

        Important:
            The functions are assembled and compiled at runtime.
            For general multiphase, multicomponent mixtures, this implies that the
            number of input args varies. The user must be aware of the dependencies
            explained in the class documentation.

        """

        self.ufuncs: dict[str, Callable] = dict()
        """Generalized numpy-ufunc represenatation of some thermodynamic properties.

        See :attr:`cfuncs` for more information.

        Callable contained here describe generalized numpy functions, meaning they
        operatore on scalar and vectorized input (nummpy arrays).
        They exploit numba.guvectorize (among others) for efficient computation.

        """

    def test_compiled_functions(self, tol: float = 1e-12, n: int = 100000):
        """Performs some tests on assembled functions.

        Warning:
            This triggers numba's just-in-time compilation!

            I.e., the execution of this function takes a considerable amount of time.

        Warning:
            This method raises AssertionErrors if any test failes.

        Parameters:
            tol: ``default=1e-12``

                Tolerance for numerical zero.
            n: ``default=100000``

                Number for testing of vectorized computations.

        """

        ncomp = self._n_c

        p_1 = 1.0
        T_1 = 1.0
        X0 = np.array([0.0] * ncomp)

        A_c = self.cfuncs["A"]
        d_A_c = self.cfuncs["d_A"]
        B_c = self.cfuncs["B"]
        d_B_c = self.cfuncs["d_B"]
        Z_c = self.cfuncs["Z"]
        d_Z_c = self.cfuncs["d_Z"]

        Z_double_g_c = self._Z_cfuncs["double-root-gas"]
        d_Z_double_g_c = self._Z_cfuncs["d-double-root-gas"]
        Z_double_l_c = self._Z_cfuncs["double-root-liq"]
        d_Z_double_l_c = self._Z_cfuncs["d-double-root-liq"]

        # if compositions are zero, A and B are zero
        assert (
            B_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional covolume failed."
        assert (
            A_c(p_1, T_1, X0) < tol
        ), "Value-test of compiled call to non-dimensional cohesion failed."

        # if A,B are zero, this should give the double-root case
        z_test_g = Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        z_test_l = Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        assert (
            np.abs(z_test_g - Z_double_g_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, gas-like compressibility factor failed."
        assert (
            np.abs(z_test_l - Z_double_l_c(0.0, 0.0)) < tol
        ), "Value-test for compiled, liquid-like compressibility factor failed."

        d_z_test_g = d_Z_c(True, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        d_z_test_l = d_Z_c(False, p_1, T_1, X0, eps=1e-14, smooth_e=0.0, smooth_3=0.0)
        da_ = d_A_c(p_1, T_1, X0)
        db_ = d_B_c(p_1, T_1, X0)
        dzg_ = d_Z_double_g_c(0.0, 0.0)
        dzl_ = d_Z_double_l_c(0.0, 0.0)
        dzg_ = dzg_[0] * da_ + dzg_[1] * db_
        dzl_ = dzl_[0] * da_ + dzl_[1] * db_
        assert (
            np.linalg.norm(d_z_test_g - dzg_) < tol
        ), "Derivative-test for compiled, gas-like compressibility factor failed."
        assert (
            np.linalg.norm(d_z_test_l - dzl_) < tol
        ), "Derivative-test for compiled, liquid-like compressibility factor failed."

        # p = np.random.rand(n) * 1e6 + 1
        # T = np.random.rand(n) * 1e2 + 1
        # X = np.random.rand(n, 2)
        # X = normalize_fractions(X)
