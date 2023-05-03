"""Testing module for the Peng-Robinson EoS class."""
import numbers
import unittest
from itertools import product

import numpy as np
import scipy.sparse as sps

import porepy as pp


def res_compressibility(Z, A, B):
    """Returns the evaluation of the cubic compressibility polynomial p(A,B)[Z].

    If Z is a root, this should return zero.
    """
    return (
        Z * Z * Z
        + (B - 1) * Z * Z
        + (A - 2 * B - 3 * B * B) * Z
        + (B * B + B * B * B - A * B)
    )


class TestPREoS(unittest.TestCase):
    """Test Case for the implementation of the Peng-Robinson Equation of state.

    The valuation of various properties and formulas is tested here, including

    - compressibility factor ``Z``

    """

    Z_refinement: int = 1000
    """Interval refinement for the testing of the compressibility factor Z."""

    Z_eps: float = 2e-3
    """Numerical zero to check if computed Z is actual root of polynomial."""

    @unittest.skip("Broadcasting not supported by PorePy's Ad-array.")
    def test_compressibility_factor_broadcasting(self):
        """Tests the ability of the root evaluation to take input in form of
        real numbers, numpy arrays and PorePy's AD-arrays in any combination,
        with agreeing dimension.

        """
        ### Test Type broadcasting with equal dimensions
        def tested_formats_1(x):
            return [
                x,
                np.array([x]),
                pp.ad.Ad_array(np.array([x]), sps.csr_matrix(np.array([1]))),
            ]

        def format_info(X):
            msg = ""
            if isinstance(X, numbers.Real):
                msg += str(type(X))
            elif isinstance(X, np.ndarray):
                msg += f"{str(type(X))}, shape={X.shape}"
            elif isinstance(X, pp.ad.Ad_array):
                msg += f"Ad_array; val={str(type(X.val))}"
                if isinstance(X.val, np.ndarray):
                    msg += f" with shape={X.val.shape}"

                msg += f"; jac={str(type(X.jac))}"
                if isinstance(X.jac, (np.ndarray, sps.spmatrix)):
                    msg += f" with shape={X.jac.shape}"

            return msg

        # taking values in the middle of the 3-root-region
        A = tested_formats_1(0.2)
        B = tested_formats_1(0.1)

        EOS = pp.composite.PengRobinsonEoS(True)  # flag does not change results

        for A_i, B_i in product(A, B):

            a_type = f"A: {format_info(A_i)}"
            b_type = f"B: {format_info(B_i)}"

            with self.subTest(msg=f"Types:\nA={a_type}\nB={b_type}", A_i=A_i, B_i=B_i):

                Z = EOS._Z(A_i, B_i)
                if isinstance(A_i, (numbers.Real, np.ndarray)) and isinstance(
                    B_i, (numbers.Real, np.ndarray)
                ):
                    self.assertTrue(
                        isinstance(Z, np.ndarray),
                        msg=f"Broadcast-reversion failed for types:\n{a_type}\n{b_type}",
                    )

        ### Test Type broadcasting with unequal dimensions
        def tested_formats_2(x):
            return [
                x,
                np.array([x, x, x]),
                pp.ad.Ad_array(np.array([x, x, x]), sps.csr_matrix(np.eye(3))),
            ]

        A = tested_formats_2(0.2)
        B = tested_formats_2(0.1)

        for A_i, B_i in product(A, B):

            a_type = f"A: {format_info(A_i)}"
            b_type = f"B: {format_info(B_i)}"

            with self.subTest(msg=f"Types:\nA={a_type}\nB={b_type}", A_i=A_i, B_i=B_i):

                Z = EOS._Z(A_i, B_i)
                if isinstance(A_i, (numbers.Real, np.ndarray)) and isinstance(
                    B_i, (numbers.Real, np.ndarray)
                ):
                    self.assertTrue(
                        isinstance(Z, np.ndarray),
                        msg=f"Broadcast-reversion failed for types:\n{a_type}\n{b_type}",
                    )

    def test_compressibility_factor_values_subcritical(self):
        """Tests the evaluation of roots of the cubic EoS implemented using
        Cardano formulas, in the subcritical region.

        The test case consists of dimensionless cohesion and covolume terms ``A`` and
        ``B`` being fet to respective function, where ``A`` in ``[0, A_crit]`` and
        ``B`` in ``[0, B_crit]``.

        The refinement of the intervals can be set in class attribute ``Z_refinement``.

        """
        # requested number of refinements, excluding 0 and the critical value
        all_A = np.linspace(0, pp.composite.A_CRIT, self.Z_refinement + 2)[1:-1]
        all_B = np.linspace(0, pp.composite.B_CRIT, self.Z_refinement + 2)[1:-1]

        GAS = pp.composite.PengRobinsonEoS(True)
        LIQ = pp.composite.PengRobinsonEoS(False)

        liquid_residuals = list()
        gas_residuals = list()

        for A, B in zip(all_A, all_B):

            with self.subTest(msg=f"A={A}\n; B={B}", A=A, B=B):

                # AD-fy
                A_ = pp.ad.Ad_array(np.array([A]), sps.lil_matrix((1, 1)))
                B_ = pp.ad.Ad_array(np.array([B]), sps.lil_matrix((1, 1)))

                # compute and return to scalar
                Z_G = GAS._Z(A_, B_).val[0]
                Z_L = LIQ._Z(A_, B_).val[0]

                # check if it is a root
                residual_liquid = abs(res_compressibility(Z_L, A, B))
                residual_gas = abs(abs(res_compressibility(Z_G, A, B)))
                liquid_residuals.append(residual_liquid)
                gas_residuals.append(residual_gas)

                self.assertTrue(
                    residual_liquid < self.Z_eps,
                    f"Liquid-like root not close to zero. Residual: {residual_liquid}",
                )

                self.assertTrue(
                    residual_gas < self.Z_eps,
                    f"Gas-like root not close to zero. Residual: {residual_gas}",
                )

    def test_compressibility_factor_values_supercritical(self):
        """Tests the evaluation of roots of the cubic EoS implemented using
        Cardano formulas, in the supercritical region.

        The test case consists of dimensionless cohesion and covolume terms ``A`` and
        ``B`` being fet to respective function,
        where ``A`` in ``[A_crit, 2 * A_crit]`` and
        ``B`` in ``[B_crit, 2 * B_crit]``

        The refinement of the intervals can be set in class attribute ``Z_refinement``.

        """
        pass

    def test_compressibility_factor_values_superheated(self):
        """Tests the evaluation of roots of the cubic EoS implemented using
        Cardano formulas, in the supercritical region.

        The test case consists of 2 subtest for dimensionless
        cohesion and covolume terms ``A`` and ``B``:

        1. ``A`` in ``[0, A_crit]``and ``B`` in ``[B_crit, 2 * B_crit]``
        2. ``A`` in ``[A_crit, 2 * A_crit]``and ``B`` in ``[0, B_crit]``

        The refinement of the intervals can be set in class attribute ``Z_refinement``.

        """
        pass


if __name__ == "__main__":
    unittest.main()
