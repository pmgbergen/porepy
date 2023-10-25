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


discr_instance = pp.Mpfa("flow")


class TestMpfaBoundaryPressure(xpfa_tests.XpfaBoundaryPressureTests):
    """Tests for the boundary pressure computation in MPFA.
    Provides access to the fixture discr_instance and adds two tests for simplices.
    Otherwise identical to the tests in test_utils.common_xpfa_tests.py and used in
    test_tpfa.py.

    """

    @property
    def discr_instance(self):
        """Return a tpfa instance."""
        return discr_instance

    def test_linear_flow_simplex_grid(self):
        mesh_size = {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}
        network = pp.create_fracture_network(
            None, pp.grids.standard_grids.utils.unit_domain(2)
        )
        mdg = network.mesh(mesh_size)
        g = mdg.subdomains(dim=2)[0]
        # Flow from right to left
        bf = g.get_boundary_faces()
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[xright]] = 1 * g.face_areas[bf[xright]]

        data = self.make_dictionary(g, bound, bc_val)

        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_structured_simplex_linear_flow(self):
        g = self.simplex_grid()
        bc_val = np.zeros(g.num_faces)
        # Flow from rght to left
        bf = g.get_boundary_faces()
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[xright]] = -1

        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])


class TestMpfaPressureReconstructionMatrices:
    def _make_true_2d(self, g):
        if g.dim == 2:
            g = g.copy()
            g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
            g.face_centers = np.delete(g.face_centers, (2), axis=0)
            g.face_normals = np.delete(g.face_normals, (2), axis=0)
            g.nodes = np.delete(g.nodes, (2), axis=0)
        return g

    @property
    def reference_dense_arrays(self):
        return pp.test_utils.reference_dense_arrays.test_mpfa[
            "TestMpfaPressureReconstructionMatrices"
        ]

    def test_cart_2d(self):
        """
        Test that mpfa gives out the correct matrices for reconstruction of the
        pressures at the faces. Also check those returned by the helper function
        reconstruct_pressure.
        """
        g = pp.CartGrid(np.array([1, 1]), physdims=[2, 2])
        g.compute_geometry()

        k = pp.SecondOrderTensor(np.array([2]))

        bc = pp.BoundaryCondition(g)

        _, _, grad_cell, grad_bound, *_ = discr_instance._flux_discretization(
            g, k, bc, inverter="python"
        )

        reference_grad_bound = self.reference_dense_arrays["test_cart_2d"]["grad_bound"]
        reference_grad_cell = np.array([[1.0], [1.0], [1.0], [1.0]])

        assert np.all(np.abs(grad_bound - reference_grad_bound) < 1e-7)
        assert np.all(np.abs(grad_cell - reference_grad_cell) < 1e-12)

        # The reconstruction function requires a "true 2d" grid without z-coordinates.
        g = self._make_true_2d(g)
        sc_top = pp.fvutils.SubcellTopology(g)
        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=0)

        D_g_ref = self.reference_dense_arrays["test_cart_2d"]["D_g"]
        CC_ref = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
        assert np.all(np.abs(D_g - D_g_ref) < 1e-12)
        assert np.all(np.abs(CC - CC_ref) < 1e-12)

    def test_simplex_2d(self):
        """Test reconstruct_pressure matrices"""
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        g = self._make_true_2d(g)
        sc_top = pp.fvutils.SubcellTopology(g)

        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=0)

        references = self.reference_dense_arrays["test_simplex_2d"]
        assert np.all(np.abs(D_g - references["D_g"]).A < 1e-12)
        assert np.all(np.abs(CC - references["CC"]) < 1e-12)

    def test_cart_3d(self):
        """Test reconstruct_pressure matrices"""

        g = pp.CartGrid([1, 1, 1], physdims=[2, 2, 2])
        g.compute_geometry()
        sc_top = pp.fvutils.SubcellTopology(g)

        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=1)
        references = self.reference_dense_arrays["test_cart_3d"]
        assert np.all(np.abs(D_g - references["D_g"]).A < 1e-12)
        assert np.all(np.abs(CC - references["CC"]) < 1e-12)

    def test_simplex_3d_dirichlet(self):
        """
        Test that we retrieve a linear solution exactly
        """
        num_cells = 2 * np.ones(3, dtype=int)
        g = pp.StructuredTetrahedralGrid(num_cells, physdims=[1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        k = pp.SecondOrderTensor(kxx)

        bc = pp.BoundaryCondition(g)
        bc.is_dir[g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False

        p0 = 1
        p_b = np.sum(g.face_centers, axis=0) + p0

        flux, bound_flux, p_t_cell, p_t_bound, *_ = discr_instance._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        P = sps.linalg.spsolve(div * flux, -div * bound_flux * p_b)

        P_f = p_t_cell * P + p_t_bound * p_b

        assert np.all(np.abs(P - np.sum(g.cell_centers, axis=0) - p0) < 1e-10)
        assert np.all(np.abs(P_f - np.sum(g.face_centers, axis=0) - p0) < 1e-10)

    def test_simplex_3d_boundary(self):
        """
        Even if we do not get exact solution at interiour we should be able to
        retrieve the boundary conditions
        """
        num_cells = 2 * np.ones(3, dtype=int)
        g = pp.StructuredTetrahedralGrid(num_cells, physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        kxx = 10 * np.random.rand(g.num_cells)
        k = pp.SecondOrderTensor(kxx)

        bc = pp.BoundaryCondition(g)
        dir_ind = g.get_all_boundary_faces()[[0, 2, 5, 8, 10, 13, 15, 21]]

        bc.is_dir[dir_ind] = True
        bc.is_neu[bc.is_dir] = False

        p_b = np.random.randn(g.face_centers.shape[1])

        flux, bound_flux, p_t_cell, p_t_bound, *_ = discr_instance._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        P = sps.linalg.spsolve(div * flux, -div * bound_flux * p_b.ravel("F"))

        P_f = p_t_cell * P + p_t_bound * p_b

        assert np.all(np.abs(P_f[dir_ind] - p_b[dir_ind]) < 1e-10)

    def test_simplex_3d_sub_face(self):
        """
        Test that we reconstruct the exact solution on subfaces
        """
        num_cells = 2 * np.ones(3, dtype=int)
        g = pp.StructuredTetrahedralGrid(num_cells, physdims=[1, 1, 1])
        g.compute_geometry()
        s_t = pp.fvutils.SubcellTopology(g)

        k = pp.SecondOrderTensor(10 * np.ones(g.num_cells))

        bc = pp.BoundaryCondition(g)
        bc.is_dir[g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False
        bc = pp.fvutils.boundary_to_sub_boundary(bc, s_t)

        p_b = np.sum(-g.face_centers, axis=0)
        p_b = p_b[s_t.fno_unique]

        flux, bound_flux, p_t_cell, p_t_bound, *_ = discr_instance._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        hf2f = pp.fvutils.map_hf_2_f(nd=1, sd=g)
        P = sps.linalg.spsolve(div * hf2f * flux, -div * hf2f * bound_flux * p_b)

        P_hf = p_t_cell * P + p_t_bound * p_b
        _, IA = np.unique(s_t.fno_unique, True)
        P_f = P_hf[IA]

        assert np.all(np.abs(P + np.sum(g.cell_centers, axis=0)) < 1e-10)
        assert np.all(np.abs(P_f + np.sum(g.face_centers, axis=0)) < 1e-10)


class TestRobinBoundaryCondition:
    """Test Robin boundary conditions."""

    @pytest.mark.parametrize("nx", [1, 3])  # Number of cells in x-direction
    @pytest.mark.parametrize("ny", [1, 3])  # Number of cells in y-direction
    def test_flow_left_right(self, nx, ny):
        """Dirichlet at left, Robin at right, results in flow to the right."""
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        k = pp.SecondOrderTensor(np.ones(g.num_cells))
        robin_weight = 1.5

        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        rob_ind = np.ravel(np.argwhere(right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        p_bound = 1
        rob_bound = 0
        C = (rob_bound - robin_weight * p_bound) / (robin_weight - p_bound)
        u_ex = -C * g.face_normals[0]
        p_ex = C * g.cell_centers[0] + p_bound
        self.solve_robin(
            g,
            k,
            bnd,
            robin_weight,
            p_bound,
            rob_bound,
            dir_ind,
            rob_ind,
            p_ex,
            u_ex,
        )

    @pytest.mark.parametrize("nx", [1, 3])  # Number of cells in x-direction
    @pytest.mark.parametrize("ny", [1, 3])  # Number of cells in y-direction
    def test_flow_nonzero_rhs(self, nx, ny):
        """Dirichlet at left, Robin at right, results in flow to the right.

        This test is similar to the previous one, but with a nonzero right hand side.
        """
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        k = pp.SecondOrderTensor(np.ones(g.num_cells))
        robin_weight = 1.5

        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        rob_ind = np.ravel(np.argwhere(right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        p_bound = 1
        rob_bound = 1

        C = (rob_bound - robin_weight * p_bound) / (robin_weight - p_bound)
        u_ex = -C * g.face_normals[0]
        p_ex = C * g.cell_centers[0] + p_bound
        self.solve_robin(
            g,
            k,
            bnd,
            robin_weight,
            p_bound,
            rob_bound,
            dir_ind,
            rob_ind,
            p_ex,
            u_ex,
        )

    @pytest.mark.parametrize("nx", [1, 3])  # Number of cells in x-direction
    @pytest.mark.parametrize("ny", [1, 3])  # Number of cells in y-direction
    def test_flow_down(self, nx, ny):
        """Dirichlet at top, Robin at bottom, results in downward flow."""
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        k = pp.SecondOrderTensor(np.ones(g.num_cells))
        robin_weight = 1.5

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(top))
        rob_ind = np.ravel(np.argwhere(bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        p_bound = 1
        rob_bound = 1
        C = (rob_bound - robin_weight * p_bound) / (robin_weight - p_bound)
        u_ex = C * g.face_normals[1]
        p_ex = C * (1 - g.cell_centers[1]) + p_bound
        self.solve_robin(
            g,
            k,
            bnd,
            robin_weight,
            p_bound,
            rob_bound,
            dir_ind,
            rob_ind,
            p_ex,
            u_ex,
        )

    def test_no_neumann(self):
        """Dirichlet top and left, Robin bottom and right."""
        g = pp.CartGrid([2, 2], physdims=[1, 1])
        g.compute_geometry()
        k = pp.SecondOrderTensor(np.ones(g.num_cells))
        robin_weight = 1.5

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(top + left))
        rob_ind = np.ravel(np.argwhere(bot + right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        bnd.robin_weight = robin_weight * np.ones(g.num_faces)
        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, k, bnd, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        rob_ex = [robin_weight * 0.25, robin_weight * 0.75, 1, 1]
        u_bound = np.zeros(g.num_faces)
        u_bound[dir_ind] = g.face_centers[1, dir_ind]
        u_bound[rob_ind] = rob_ex * g.face_areas[rob_ind]

        a = div * flux
        b = -div * bound_flux * u_bound

        p = np.linalg.solve(a.A, b)

        u_ex = [
            np.dot(g.face_normals[:, f], np.array([0, -1, 0]))
            for f in range(g.num_faces)
        ]
        u_ex = np.array(u_ex).ravel("F")
        p_ex = g.cell_centers[1]

        assert np.allclose(p, p_ex)
        assert np.allclose(flux * p + bound_flux * u_bound, u_ex)

    def solve_robin(
        self, g, k, bnd, robin_weight, p_bound, rob_bound, dir_ind, rob_ind, p_ex, u_ex
    ):
        """Helper function to solve the Robin problem and compare to reference values."""
        bnd.robin_weight = robin_weight * np.ones(g.num_faces)

        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, k, bnd, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        u_bound = np.zeros(g.num_faces)
        u_bound[dir_ind] = p_bound
        u_bound[rob_ind] = rob_bound * g.face_areas[rob_ind]

        a = div * flux
        b = -div * bound_flux * u_bound

        p = np.linalg.solve(a.A, b)
        assert np.allclose(p, p_ex)
        assert np.allclose(flux * p + bound_flux * u_bound, u_ex)
