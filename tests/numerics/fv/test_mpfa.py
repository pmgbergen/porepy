"""Tests for the MPFA discretization scheme."""
import random

import numpy as np
import pytest
import scipy
import scipy.sparse as sps
import sympy

import porepy as pp
from porepy.applications.test_utils import common_xpfa_tests as xpfa_tests
from porepy.applications.test_utils.partial_discretization import (
    perform_partial_discretization_specified_nodes,
)

"""Utility methods."""


def _discretization_matrices(g, perm, bound):
    flux, bound_flux, _, _, vector_source, _ = pp.Mpfa("flow")._flux_discretization(
        g, perm, bound, inverter="python"
    )
    div = g.cell_faces.T
    return div, flux, bound_flux, vector_source


def _grid_and_discretization_matrices(nx, dir_faces=None):
    g, perm, bound = xpfa_tests._setup_cart_2d(nx, dir_faces)
    div, flux, bound_flux, vector_source = _discretization_matrices(g, perm, bound)
    return g, perm, bound, div, flux, bound_flux, vector_source


def _setup_random_pressure_field(g):
    gx = random.random()
    gy = random.random()
    xf = g.face_centers
    xc = g.cell_centers

    pr_bound = gx * xf[0] + gy * xf[1]
    pr_cell = gx * xc[0] + gy * xc[1]
    return pr_bound, pr_cell, gx, gy


"""Tests on discretization stensils."""


def test_laplacian_stencil_cart_2d():
    """Apply MPFA on Cartesian grid, should obtain Laplacian stencil.

    See test_tpfa.py for the original test. This test is identical, except for the
    discretization method used.
    """
    xpfa_tests._test_laplacian_stencil_cart_2d(_discretization_matrices)


def test_uniform_flow_cart_2d():
    # Structured Cartesian grid
    nx = np.array([10, 10])
    g, _, _, div, flux, bound_flux, _ = _grid_and_discretization_matrices(nx)

    a = div * flux

    pr_bound, pr_cell, _, _ = _setup_random_pressure_field(g)

    rhs = div * bound_flux * pr_bound
    pr = np.linalg.solve(a.todense(), -rhs)

    p_diff = pr - pr_cell
    assert np.max(np.abs(p_diff)) < 1e-8


def test_uniform_flow_cart_2d_structured_pert():
    g, perm, bound = xpfa_tests._setup_cart_2d(np.array([2, 2]))
    g.nodes[0, 4] = 1.5
    g.compute_geometry()
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    div, flux, bound_flux, _ = _discretization_matrices(g, perm, bound)

    a = div * flux

    xf = np.zeros_like(g.face_centers)
    xf[:, bound_faces.ravel()] = g.face_centers[:, bound_faces.ravel()]
    xc = g.cell_centers
    pr_bound = xf.sum(axis=0)
    pr_cell = xc.sum(axis=0)

    rhs = div * bound_flux * pr_bound
    pr = np.linalg.solve(a.todense(), -rhs)

    p_diff = pr - pr_cell
    assert np.max(np.abs(p_diff)) < 1e-8


def test_uniform_flow_cart_2d_pert():
    # Randomly perturbed grid, with random linear pressure field
    g, perm, bound = xpfa_tests._setup_cart_2d(np.array([10, 10]))
    dx = 1
    pert = 0.4
    g.nodes = g.nodes + dx * pert * (
        0.5 - np.random.rand(g.nodes.shape[0], g.num_nodes)
    )
    # Cancel perturbations in z-coordinate.
    g.nodes[2, :] = 0
    g.compute_geometry()

    div, flux, bound_flux, _ = _discretization_matrices(g, perm, bound)

    # Compute solution
    a = div * flux
    pr_bound, pr_cell, *_ = _setup_random_pressure_field(g)

    rhs = div * bound_flux * pr_bound
    pr = np.linalg.solve(a.todense(), -rhs)

    p_diff = pr - pr_cell
    assert np.max(np.abs(p_diff)) < 1e-8


"""Tests for periodic boundary condition implementation in MPFA."""


def setup_periodic_pressure_field(g, kxx):
    xf = g.face_centers
    xc = g.cell_centers

    pr_bound = np.sin(2 * np.pi * xf[0]) + np.cos(2 * np.pi * xf[1])
    pr_cell = np.sin(2 * np.pi * xc[0]) + np.cos(2 * np.pi * xc[1])

    src = kxx * (2 * np.pi) ** 2 * pr_cell
    return pr_bound, pr_cell, src


def test_symmetric_bc_common_with_tpfa():
    """See test_tpfa.py for the original tests."""
    xpfa_tests._test_symmetry_field_2d_periodic_bc(_discretization_matrices)
    xpfa_tests._test_laplacian_stensil_cart_2d_periodic_bcs(_discretization_matrices)


"""Partial discretization tests for the MPFA discretization scheme below."""


@pytest.fixture
def grid_and_discretization_matrices_partial():
    """Return a grid and the corresponding mpfa discretization matrices.

    Used throughout the partial discretization tests. Note that these tests don't use
    div, which is therefore not returned as opposed to _grid_and_discretization_matrices.
    """
    g, perm, bound = xpfa_tests._setup_cart_2d(np.array([5, 5]), np.array([]))
    _, flux, bound_flux, vector_source = _discretization_matrices(g, perm, bound)
    return g, perm, bound, flux, bound_flux, vector_source


def specified_parameters(perm, bnd):
    """Return parameter dictionary for partial update tests."""
    specified_data = {
        "second_order_tensor": perm,
        "bc": bnd,
        "inverter": "python",
    }
    return specified_data


@pytest.mark.parametrize(
    "cell_id",
    [
        10,  # cell at domain boundary
        12,  # internal cell
    ],
)
def test_partial_discretization_specified_nodes(
    cell_id: int, grid_and_discretization_matrices_partial
):
    """Test that the discretization matrices are correct for a partial update.

    Parameters:
        cell_id: The cell whose node will be used to define the partial update. The
            nodes are identified based on the provided cell_id outside of the
            discretization method.

    """
    (
        g,
        perm,
        bnd,
        flux_full,
        bound_flux_full,
        vector_src_full,
    ) = grid_and_discretization_matrices_partial
    specified_data = specified_parameters(perm, bnd)
    keyword = "flow"
    discr = pp.Mpfa(keyword)
    data = perform_partial_discretization_specified_nodes(
        g, discr, specified_data, cell_id
    )

    partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
    partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_flux_matrix_key
    ]
    partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.vector_source_matrix_key
    ]

    active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

    for partial, full in zip(
        [partial_flux, partial_bound, partial_vector_source],
        [flux_full, bound_flux_full, vector_src_full],
    ):
        assert pp.test_utils.arrays.compare_matrices(
            partial[active_faces], full[active_faces], 1e-10
        )
        # For partial update, only the active faces should be nonzero. Force these to
        # zero and check that the rest is zero.
        pp.matrix_operations.zero_rows(partial, active_faces)
        assert np.allclose(partial.data, 0)


def test_partial_discretization_one_cell_at_a_time():
    g = pp.CartGrid([3, 3])
    g.compute_geometry()

    # Assign random permeabilities, for good measure
    np.random.seed(42)
    kxx = np.random.random(g.num_cells)
    kyy = np.random.random(g.num_cells)
    # Ensure positive definiteness
    kxy = np.random.random(g.num_cells) * kxx * kyy
    perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)
    bnd = pp.BoundaryCondition(g)
    specified_data = specified_parameters(perm, bnd)
    keyword = "flow"
    discr = pp.Mpfa(keyword)

    # Initialize matrices to be filled one cell at a time
    flux = sps.csr_matrix((g.num_faces, g.num_cells))
    bound_flux = sps.csr_matrix((g.num_faces, g.num_faces))
    vector_src = sps.csr_matrix((g.num_faces, g.num_cells * g.dim))
    faces_covered = np.zeros(g.num_faces, bool)

    for cell_id in range(g.num_cells):
        data = perform_partial_discretization_specified_nodes(
            g, discr, specified_data, cell_id
        )
        partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_flux_matrix_key
        ]
        partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.vector_source_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        if np.any(faces_covered):
            fi = np.where(faces_covered)[0]
            pp.fvutils.remove_nonlocal_contribution(
                fi, 1, partial_flux, partial_bound, partial_vector_source
            )

        faces_covered[active_faces] = True

        flux += partial_flux
        bound_flux += partial_bound
        vector_src += partial_vector_source

    # Use direct discretization for all cells.
    flux_full, bound_flux_full, *_, vector_src_full, _ = pp.Mpfa(
        "flow"
    )._flux_discretization(g, perm, bnd, inverter="python")

    # Compare results
    for partial, full in zip(
        [flux, bound_flux, vector_src], [flux_full, bound_flux_full, vector_src_full]
    ):
        assert pp.test_utils.arrays.compare_matrices(partial, full, 1e-10)


"""Gravity related tests."""


class _SolutionHomogeneousDomainFlowWithGravity(object):
    """Convenience class for representing an analytical solution, and its
    derivatives"""

    def __init__(self, p, x, y):
        p_f = sympy.lambdify((x, y), p, "numpy")
        gx = sympy.diff(p, x)
        gy = sympy.diff(p, y)
        gx_f = sympy.lambdify((x, y), gx, "numpy")
        gy_f = sympy.lambdify((x, y), gy, "numpy")
        self.p_f = p_f
        self.gx_f = gx_f
        self.gy_f = gy_f


class _Solution1DFlowWithGravity(object):
    """Convenience class for representing an analytical solution, and its
    derivatives"""

    def __init__(self, p, y):
        p_f = sympy.lambdify(y, p, "numpy")
        g = sympy.diff(p, y)
        g_f = sympy.lambdify(y, g, "numpy")
        self.p_f = p_f
        self.g_f = g_f


def test_hydrostatic_pressure_1d():
    # Test mpfa_gravity in 1D grid
    # Solver uses TPFA + standard method
    # Should be exact for hydrostatic pressure
    # with stepwise gravity variation

    x = sympy.symbols("x")
    g1 = 10
    g2 = 1
    p0 = 1  # reference pressure
    p = p0 + sympy.Piecewise(
        ((1 - x) * g1, x >= 0.5), (0.5 * g1 + (0.5 - x) * g2, x < 0.5)
    )
    an_sol = _Solution1DFlowWithGravity(p, x)

    g = pp.CartGrid(8, 1)
    g.compute_geometry()
    xc = g.cell_centers
    xf = g.face_centers

    k = pp.SecondOrderTensor(np.ones(g.num_cells))

    # Gravity
    gforce = an_sol.g_f(xc[0])

    # Set type of boundary conditions
    # 'dir' left, 'neu' right
    p_bound = np.zeros(g.num_faces)
    dir_faces = np.array([0])

    bound_cond = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)

    # set value of boundary condition
    p_bound[dir_faces] = an_sol.p_f(xf[0, dir_faces])

    # GCMPFA discretization, and system matrix
    div, flux, bound_flux, div_g = _discretization_matrices(g, k, bound_cond)
    a = div * flux

    flux_g = div_g * gforce
    # assemble rhs
    b = -div * bound_flux * p_bound - div * flux_g
    # solve system
    p = scipy.sparse.linalg.spsolve(a, b)
    q = flux * p + bound_flux * p_bound + flux_g
    p_ex = an_sol.p_f(xc[0])
    q_ex = np.zeros(g.num_faces)
    assert np.allclose(p, p_ex)
    assert np.allclose(q, q_ex)


@pytest.mark.parametrize("grid_class", [pp.CartGrid, pp.StructuredTriangleGrid])
def test_hydrostatic_pressure_2d(grid_class):
    # Test mpfa_gravity in 2D Cartesian
    # and triangular grids
    # Should be exact for hydrostatic pressure
    # with stepwise gravity variation

    x, y = sympy.symbols("x y")
    g1 = 10
    g2 = 1
    p0 = 1  # reference pressure
    p = p0 + sympy.Piecewise(
        ((1 - y) * g1, y >= 0.5), (0.5 * g1 + (0.5 - y) * g2, y < 0.5)
    )
    an_sol = _SolutionHomogeneousDomainFlowWithGravity(p, x, y)

    domain = np.array([1, 1])
    basedim = np.array([4, 4])
    pert = 0.5
    g = grid_class(basedim, domain)
    g.compute_geometry()
    dx = np.max(domain / basedim)
    g = xpfa_tests.perturb_grid(g, pert, dx)
    g.compute_geometry()
    xc = g.cell_centers
    xf = g.face_centers

    k = pp.SecondOrderTensor(np.ones(g.num_cells))

    # Gravity
    gforce = np.zeros((2, g.num_cells))
    gforce[0, :] = an_sol.gx_f(xc[0], xc[1])
    gforce[1, :] = an_sol.gy_f(xc[0], xc[1])
    gforce = gforce.ravel("F")

    # Set type of boundary conditions
    p_bound = np.zeros(g.num_faces)
    left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
    right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
    dir_faces = np.concatenate((left_faces, right_faces))

    bound_cond = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)

    # set value of boundary condition
    p_bound[dir_faces] = an_sol.p_f(xf[0, dir_faces], xf[1, dir_faces])

    # GCMPFA discretization, and system matrix
    div, flux, bound_flux, div_g = _discretization_matrices(g, k, bound_cond)
    a = div * flux
    flux_g = div_g * gforce
    b = -div * bound_flux * p_bound - div * flux_g
    p = scipy.sparse.linalg.spsolve(a, b)
    q = flux * p + bound_flux * p_bound + flux_g
    p_ex = an_sol.p_f(xc[0], xc[1])
    q_ex = np.zeros(g.num_faces)
    assert np.allclose(p, p_ex)
    assert np.allclose(q, q_ex)


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
def test_mpfa_gravity_common_with_tpfa(test_method):
    """See test_utils.common_xpfa_tests.py for the original tests."""
    test_method("mpfa")
