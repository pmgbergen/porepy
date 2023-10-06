"""Utility functions used here in test_mpfa.py and test_tpfa.py.
Some simple for getting grid, permeability, bc object etc.

Then more specific functions related to specific tests defined both here and for mpfa.

"""
import sympy
import scipy
import numpy as np

import porepy as pp


def _setup_cart_2d(nx, dir_faces=None):
    g = pp.CartGrid(nx)
    g.compute_geometry()
    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)
    if dir_faces is None:
        dir_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
    return g, perm, bound


def perturb_grid(g, rate, dx):
    rand = np.vstack((np.random.rand(g.dim, g.num_nodes), np.repeat(0.0, g.num_nodes)))
    r1 = np.ravel(
        np.argwhere(
            (g.nodes[0] < 1 - 1e-10)
            & (g.nodes[0] > 1e-10)
            & (g.nodes[1] < 0.5 - 1e-10)
            & (g.nodes[1] > 1e-10)
        )
    )
    r2 = np.ravel(
        np.argwhere(
            (g.nodes[0] < 1 - 1e-10)
            & (g.nodes[0] > 1e-10)
            & (g.nodes[1] < 1.0 - 1e-10)
            & (g.nodes[1] > 0.5 + 1e-10)
        )
    )
    pert_nodes = np.concatenate((r1, r2))
    npertnodes = pert_nodes.size
    rand = np.vstack((np.random.rand(g.dim, npertnodes), np.repeat(0.0, npertnodes)))
    g.nodes[:, pert_nodes] += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0
    return g


"""Tests for discretization stensils. Base case + periodic BCs."""


def _test_laplacian_stencil_cart_2d(discr_matrices_func):
    """Apply TPFA or MPFA on Cartesian grid, should obtain Laplacian stencil."""
    nx = np.array([3, 3])
    dir_faces = np.array([0, 3, 12])
    g, perm, bound = _setup_cart_2d(nx, dir_faces)
    div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
    A = div * flux
    b = -(div * bound_flux).A

    # Checks on interior cell
    mid = 4
    assert A[mid, mid] == 4
    assert A[mid - 1, mid] == -1
    assert A[mid + 1, mid] == -1
    assert A[mid - 3, mid] == -1
    assert A[mid + 3, mid] == -1

    # The first cell should have two Dirichlet bnds
    assert A[0, 0] == 6
    assert A[0, 1] == -1
    assert A[0, 3] == -1

    # Cell 3 has one Dirichlet, one Neumann face
    assert A[2, 2] == 4
    assert A[2, 1] == -1
    assert A[2, 5] == -1

    # Cell 2 has one Neumann face
    assert A[1, 1] == 3
    assert A[1, 0] == -1
    assert A[1, 2] == -1
    assert A[1, 4] == -1

    assert b[1, 13] == -1


def _test_laplacian_stensil_cart_2d_periodic_bcs(discr_matrices_func):
    """Apply TPFA and MPFA on a periodic Cartesian grid, should obtain Laplacian stencil."""

    # Structured Cartesian grid and permeability. We need tailored BC object.
    g, perm, _ = _setup_cart_2d(np.array([3, 3]))

    left_faces = [0, 4, 8, 12, 13, 14]
    right_faces = [3, 7, 11, 21, 22, 23]
    periodic_face_map = np.vstack((left_faces, right_faces))
    g.set_periodic_map(periodic_face_map)

    bound = pp.BoundaryCondition(g)
    div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
    a = div * flux
    b = -(div * bound_flux).A

    # Create laplace matrix
    A_lap = np.array(
        [
            [4.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 4.0, -1.0, -1.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, 4.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 4.0, -1.0, -1.0],
            [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 4.0],
        ]
    )
    assert np.allclose(a.A, A_lap)
    assert np.allclose(b, 0)


def _test_symmetry_field_2d_periodic_bc(discr_matrices_func):
    """
    Test that we obtain a symmetric solution accross the periodic boundary.
    The test consider the unit square with periodic boundary conditions
    on the top and bottom boundary. A source is added to the bottom row of
    cells and we test that the solution is periodic.
    Setup, with x denoting the source:
           --------
          |       |
    p = 0 |       | p = 0
          |   x   |
           -------
    """
    # Structured Cartesian grid and permeability. We need tailored BC object.
    g, perm, _ = _setup_cart_2d(np.array([5, 5]))  # No physdims allowed, [1, 1])

    bot_faces = np.argwhere(g.face_centers[1] < 1e-5).ravel()
    top_faces = np.argwhere(g.face_centers[1] > 5 - 1e-5).ravel()

    left_faces = np.argwhere(g.face_centers[0] < 1e-5).ravel()
    right_faces = np.argwhere(g.face_centers[0] > 5 - 1e-5).ravel()

    dir_faces = np.hstack((left_faces, right_faces))

    g.set_periodic_map(np.vstack((bot_faces, top_faces)))

    bound = pp.BoundaryCondition(g, dir_faces, "dir")

    # Solve
    div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
    a = div * flux

    pr_bound = np.zeros(div.shape[1])
    src = np.zeros(div.shape[0])
    src[2] = 1

    rhs = -div * bound_flux * pr_bound + src
    pr = np.linalg.solve(a.todense(), rhs)

    p_diff = pr[5:15] - np.hstack((pr[-5:], pr[-10:-5]))
    assert np.max(np.abs(p_diff)) < 1e-10


"""Gravity related testing."""


def set_params_and_discretize_gravity(g, ambient_dim, method, periodic=False):
    g.compute_geometry()
    keyword = "flow"

    if periodic:
        south = g.face_centers[1] < np.min(g.nodes[1]) + 1e-8
        north = g.face_centers[1] > np.max(g.nodes[1]) - 1e-8
        south_idx = np.argwhere(south).ravel()
        north_idx = np.argwhere(north).ravel()
        g.set_periodic_map(np.vstack((south_idx, north_idx)))

    bc = pp.BoundaryCondition(g)

    k = pp.SecondOrderTensor(np.ones(g.num_cells))

    params = {
        "bc": bc,
        "second_order_tensor": k,
        "mpfa_inverter": "python",
        "ambient_dimension": ambient_dim,
    }

    data = pp.initialize_data(g, {}, keyword, params)
    if method == "mpfa":
        discr = pp.Mpfa(keyword)
    elif method == "tpfa":
        discr = pp.Tpfa(keyword)
    discr.discretize(g, data)

    flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
    vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.vector_source_matrix_key
    ]
    div = pp.fvutils.scalar_divergence(g)
    return flux, vector_source, div


def _test_gravity_1d_ambient_dim_1(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))

    ambient_dim = 1
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength  # /2 * dx
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should decrease with increasing x coordinate, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)


def _test_gravity_1d_ambient_dim_2(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))

    ambient_dim = 2
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength  # /2 * dx
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should decrease with increasing x coordinate, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)

    # introduce a source term in y-direction
    g_y = np.zeros(g.num_cells * ambient_dim)
    g_y[1::ambient_dim] = -1 * grav_strength
    p_y = np.linalg.pinv(A.toarray()).dot(rhs * g_y)
    assert np.allclose(p_y, 0)

    flux_y = flux * p_y + vector_source_discr * g_y
    # The net flux should still be zero
    assert np.allclose(flux_y, 0)


def _test_gravity_1d_ambient_dim_2_nodes_reverted(method):
    # Same test as above, but with the orientation of the grid rotated.
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, -dx, -2 * dx]))

    ambient_dim = 2
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength  # /2 * dx
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should decrease with increasing x coordinate, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], -dx * grav_strength)
    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)

    # introduce a source term in y-direction
    g_y = np.zeros(g.num_cells * ambient_dim)
    g_y[1::ambient_dim] = -1 * grav_strength
    p_y = np.linalg.pinv(A.toarray()).dot(rhs * g_y)
    assert np.allclose(p_y, 0)

    flux_y = flux * p_y + vector_source_discr * g_y
    # The net flux should still be zero
    assert np.allclose(flux_y, 0)


def _test_gravity_1d_ambient_dim_3(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))
    g.nodes[:] = np.array([0, dx, 2 * dx])

    ambient_dim = 3
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength  # /2 * dx
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should decrease with increasing x coordinate, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)

    # introduce a source term in y-direction
    g_y = np.zeros(g.num_cells * ambient_dim)
    g_y[1::ambient_dim] = -1 * grav_strength
    p_y = np.linalg.pinv(A.toarray()).dot(rhs * g_y)
    assert np.allclose(p_y, p_x)

    flux_y = flux * p_y + vector_source_discr * g_y
    # The net flux should still be zero
    assert np.allclose(flux_y, 0)


def _test_gravity_2d_horizontal_ambient_dim_3(method):
    # Cartesian grid in xy-plane. The rotation of the grid in the mpfa discretization
    # will be trivial, leaving one source of error

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # Embed in 3d, this means that the vector source is a 3-vector per cell
    ambient_dim = 3

    # Discretization
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # First set source in z-direction. This should have no impact on the solution
    g_z = np.zeros(g.num_cells * ambient_dim)
    g_z[2::ambient_dim] = -1
    p_z = np.linalg.pinv(A.toarray()).dot(rhs * g_z)
    # all zeros
    assert np.allclose(p_z, 0)
    flux_z = flux * p_z + vector_source_discr * g_z
    assert np.allclose(flux_z, 0)

    # Next a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should be higher in the first x-row of cells, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)
    assert np.allclose(p_x[2] - p_x[3], dx * grav_strength)
    # The solution should be equal for equal x-coordinate
    assert np.allclose(p_x[0], p_x[2])
    assert np.allclose(p_x[1], p_x[3])

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)


def _test_gravity_2d_horizontal_ambient_dim_2(method):
    # Cartesian grid in xy-plane. The rotation of the grid in the mpfa discretization
    # will be trivial, leaving one source of error

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # The vector source is a 2-vector per cell
    ambient_dim = 2

    # Discretization
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should be higher in the first x-row of cells, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)
    assert np.allclose(p_x[2] - p_x[3], dx * grav_strength)
    # The solution should be equal for equal x-coordinate
    assert np.allclose(p_x[0], p_x[2])
    assert np.allclose(p_x[1], p_x[3])

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)


def _test_gravity_2d_horizontal_periodic_ambient_dim_2(method):
    # Cartesian grid in xy-plane with periodic boundary conditions.

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # The vector source is a 2-vector per cell
    ambient_dim = 2

    # Discretization
    flux, vector_source_discr, div = set_params_and_discretize_gravity(
        g, ambient_dim, method, True
    )

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)

    # The solution should be higher in the first x-row of cells, with magnitude
    # controlled by grid size and source stregth
    assert np.allclose(p_x[0] - p_x[1], dx * grav_strength)
    assert np.allclose(p_x[2] - p_x[3], dx * grav_strength)
    # The solution should be equal for equal x-coordinate
    assert np.allclose(p_x[0], p_x[2])
    assert np.allclose(p_x[1], p_x[3])

    flux_x = flux * p_x + vector_source_discr * g_x
    # The net flux should still be zero
    assert np.allclose(flux_x, 0)

    # Check matrices:
    A_known = np.array(
        [
            [3.0, -1.0, -2.0, 0.0],
            [-1.0, 3.0, 0.0, -2.0],
            [-2.0, 0.0, 3.0, -1.0],
            [0.0, -2.0, -1.0, 3.0],
        ]
    )
    # why 0.5?
    vct_src_known = (
        0.5
        * dx
        * np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    assert np.allclose(A.A, A_known)
    assert np.allclose(vector_source_discr.A, vct_src_known)
