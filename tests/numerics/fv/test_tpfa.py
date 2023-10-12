"""Tests for the TPFA discretization scheme."""
import pytest

import porepy as pp
from porepy.applications.test_utils import common_xpfa_tests as xpfa_tests

"""Local utility functions."""


def _discretization_matrices(g, perm, bound):
    kw = "flow"
    data = pp.initialize_data(
        g, {}, kw, {"second_order_tensor": perm, "bc": bound, "inverter": "python"}
    )
    discr = pp.Tpfa(kw)

    discr.discretize(g, data)
    flux = data[pp.DISCRETIZATION_MATRICES][kw][discr.flux_matrix_key]
    bound_flux = data[pp.DISCRETIZATION_MATRICES][kw][discr.bound_flux_matrix_key]
    vector_source = data[pp.DISCRETIZATION_MATRICES][kw][discr.vector_source_matrix_key]
    div = g.cell_faces.T
    return div, flux, bound_flux, vector_source


"""Tests below.

The tests are identical to the ones in test_mpfa.py, except for the discretization.
They are therefore defined in test_utils.common_xpfa_tests.py, and simply run here.
This is to avoid code duplication while adhering to the contract that code is tested
in its mirror file in the test directories.
"""


def test_laplacian_stencil_cart_2d():
    """Apply MPFA on Cartesian grid, should obtain Laplacian stencil.

    See test_tpfa.py for the original test. This test is identical, except for the
    discretization method used.
    """
    xpfa_tests._test_laplacian_stencil_cart_2d(_discretization_matrices)


def test_symmetric_bc_common_with_mpfa():
    """Outsourced to helper functions for convenient reuse in test_mpfa.py."""
    xpfa_tests._test_symmetry_field_2d_periodic_bc(_discretization_matrices)
    xpfa_tests._test_laplacian_stensil_cart_2d_periodic_bcs(_discretization_matrices)


@pytest.mark.parametrize(
    "test_method",
    [
        xpfa_tests._test_gravity_1d_ambient_dim_1,
        xpfa_tests._test_gravity_1d_ambient_dim_2,
        xpfa_tests._test_gravity_1d_ambient_dim_3,
        xpfa_tests._test_gravity_1d_ambient_dim_2_nodes_reverted,
        xpfa_tests._test_gravity_2d_horizontal_ambient_dim_3,
        xpfa_tests._test_gravity_2d_horizontal_ambient_dim_2,
        xpfa_tests._test_gravity_2d_horizontal_periodic_ambient_dim_2,
    ],
)
def test_tpfa_gravity_common_with_mpfa(test_method):
    """See test_utils.common_xpfa_tests.py for the original tests."""
    test_method("tpfa")


discr_instance = pp.Tpfa("flow")


class TestTpfaBoundaryPressure(xpfa_tests.XpfaBoundaryPressureTests):
    """Tests for the boundary pressure computation in MPFA. Accesses the fixture
    discr_instance, otherwise identical to the tests in test_utils.common_xpfa_tests.py
    and used in test_tpfa.py.

    """

    @property
    def discr_instance(self):
        """Return a tpfa instance."""
        return discr_instance
