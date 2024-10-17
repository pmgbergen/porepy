"""Utility functions used here in test_mpfa.py and test_tpfa.py.
Some simple for getting grid, permeability, bc object etc.

Then more specific functions related to specific tests defined both here and for mpfa.

"""

from typing import Literal, Union

import numpy as np
import scipy.sparse.linalg as spla

import porepy as pp


def _setup_cart_2d(nx, dir_faces=None):
    g = pp.CartGrid(nx)
    g.compute_geometry()
    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)
    if dir_faces is None:
        # If no Dirichlet faces are specified, set Dirichlet conditions on all faces.
        dir_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
    return g, perm, bound


def perturb_grid(g, rate, dx):
    # Perturb the nodes in a grid (assumed to cover the unit square). Nodes on the
    # domain boundary, and along the line `y=0` are kept fixed.
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


def create_grid_mpfa_mpsa_reproduce_known_values(
    grid_type: Literal["cart", "simplex"]
) -> tuple[pp.Grid, pp.Grid]:
    """Create grids for the tests that mpfa and mpsa reproduce known values.

    The construction below is somewhat specific to the test case, and should not be
    changed, since any change in geometry will be reflected in the solution, thus the
    tests will fail.
    """

    # Define a characteristic function which is True in the region
    # x > 0.5, y > 0.5
    def chi_func(xcoord, ycoord):
        return np.logical_and(np.greater(xcoord, 0.5), np.greater(ycoord, 0.5))

    def perturb(h, rate, dx):
        rand = np.vstack(
            (np.random.rand(h.dim, h.num_nodes), np.repeat(0.0, h.num_nodes))
        )
        h.nodes += rate * dx * (rand - 0.5)
        # Ensure there are no perturbations in the z-coordinate
        if h.dim == 2:
            h.nodes[2, :] = 0
        return h

    # Set random seed
    np.random.seed(42)
    nx = np.array([4, 4])
    domain = np.array([1, 1])
    if grid_type == "cart":
        g: pp.Grid = pp.CartGrid(nx, physdims=domain)
    elif grid_type == "simplex":
        g = pp.StructuredTriangleGrid(nx, physdims=domain)
    # Perturbation rates, same notation as in setup_grids.py
    pert = 0.5
    dx = 0.25
    g_nolines = perturb(g, pert, dx)
    g_nolines.compute_geometry()

    # Create a new grid, which will not have faces along the
    # discontinuity perturbed
    if grid_type == "cart":
        g = pp.CartGrid(nx, physdims=domain)
    elif grid_type == "simplex":
        g = pp.StructuredTriangleGrid(nx, physdims=domain)

    g.compute_geometry()
    old_nodes = g.nodes.copy()
    dx = np.max(domain / nx)
    np.random.seed(42)
    g = perturb(g, pert, dx)

    # Characteristic function for all cell centers
    xc = g.cell_centers
    chi = chi_func(xc[0], xc[1])
    # Detect faces on the discontinuity by applying g.cell_faces (this
    # is signed, so two cells in the same region will cancel out).
    #
    # Note that positive values also includes boundary faces, these will
    #  not be perturbed.
    chi_face = np.abs(g.cell_faces * chi)
    bnd_face = np.argwhere(chi_face > 0).squeeze(1)
    node_ptr = g.face_nodes.indptr
    node_ind = g.face_nodes.indices
    # Nodes of faces on the boundary
    bnd_nodes = node_ind[
        pp.utils.mcolon.mcolon(node_ptr[bnd_face], node_ptr[bnd_face + 1])
    ]
    g.nodes[:, bnd_nodes] = old_nodes[:, bnd_nodes]
    g.compute_geometry()
    g_lines = g

    return g_nolines, g_lines


"""Tests for discretization stensils. Base case + periodic BCs."""


def _test_laplacian_stencil_cart_2d(discr_matrices_func):
    """Apply TPFA or MPFA on Cartesian grid, should obtain Laplacian stencil."""
    nx = np.array([3, 3])
    dir_faces = np.array([0, 3, 12])
    g, perm, bound = _setup_cart_2d(nx, dir_faces)
    div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
    A = div * flux
    b = -(div * bound_flux).toarray()

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
    b = -(div * bound_flux).toarray()

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
    assert np.allclose(a.toarray(), A_lap)
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

    assert np.allclose(A.toarray(), A_known)
    assert np.allclose(vector_source_discr.toarray(), vct_src_known)


class XpfaBoundaryPressureTests:
    """Expects access to a fixture method "discr_instance" that returns a xpfa
    discretization object. See the pressure method below.

    """

    @property
    def discr_instance(self):
        """Abstract method that return a xpfa instance."""
        pass

    def make_dictionary(self, g, bc, bc_values=None):
        if bc_values is None:
            bc_values = np.zeros(g.num_faces)
        d = {"bc": bc, "bc_values": bc_values, "mpfa_inverter": "python"}
        return pp.initialize_default_data(g, {}, "flow", d)

    def boundary_pressure(self, p, bc_vals, data: dict):
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]

        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_vals
        )
        return bound_p

    def grid(self, nx=[2, 2], physdims=None) -> pp.Grid:
        if physdims is None:
            physdims = nx
        g = pp.CartGrid(nx, physdims)
        g.compute_geometry()
        return g

    def simplex_grid(self, nx=[2, 2]) -> pp.Grid:
        g = pp.StructuredTriangleGrid(nx)
        g.compute_geometry()
        return g

    def pressure(self, g: pp.Grid, data: dict):
        # discr_instance: pp.Mpfa | pp.Tpfa = self.discr_instance
        self.discr_instance.discretize(g, data)
        A, b = self.discr_instance.assemble_matrix_rhs(g, data)
        p = spla.spsolve(A, b)
        return p

    def test_zero_pressure(self):
        g = self.grid()
        bf = g.get_boundary_faces()
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)
        data = self.make_dictionary(g, bound)
        p = self.pressure(g, data)
        assert np.allclose(p, np.zeros_like(p))

        bound_p = self.boundary_pressure(p, np.zeros(g.num_faces), data)
        assert np.allclose(bound_p, np.zeros_like(bound_p))

    def test_constant_pressure(self):
        g = self.grid()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)
        bc_val = np.ones(g.num_faces)
        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)
        assert np.allclose(p, np.ones_like(p))

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.ones(g.num_faces)
        data = self.make_dictionary(g, bound, bc_val)

        p = self.pressure(g, data)

        assert np.allclose(p, np.ones_like(p))

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        data = self.make_dictionary(g, bound, bc_val)

        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)
        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = self.make_dictionary(g, bound, bc_val)

        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_small_domain(self):
        g = self.grid(physdims=[1, 1])

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], -2 * g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.zeros(g.num_faces)
        # Set up pressure gradient in x-direction, with value -1
        bc_val[[2, 5]] = -1
        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = pp.CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_sign_trouble_two_neumann_sides(self):
        g = pp.CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        bc_val[[0, 3]] = 1
        bc_val[[2, 5]] = -1

        data = self.make_dictionary(g, pp.BoundaryCondition(g), bc_val)
        self.discr_instance.discretize(g, data)
        self.discr_instance.assemble_matrix_rhs(g, data)
        # The problem is singular, and spsolve does not work well on all systems.
        # Instead, set a consistent solution, and check that the boundary
        # pressure is recovered.
        x = g.cell_centers[0]

        bound_p = self.boundary_pressure(x, bc_val, data)
        assert bound_p[0] == x[0] - 0.5
        assert bound_p[2] == x[1] + 0.5

    def test_linear_pressure_dirichlet_conditions_perturbed_grid(self):
        g = self.grid()
        g.nodes[:2] = g.nodes[:2] + np.random.random((2, g.num_nodes))
        g.compute_geometry()

        bf = g.get_boundary_faces()
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        data = self.make_dictionary(g, bound, bc_val)
        p = self.pressure(g, data)

        bound_p = self.boundary_pressure(p, bc_val, data)
        assert np.allclose(bound_p[bf], bc_val[bf])


def test_split_discretization_into_subproblems(
    discr_class: Union[pp.Mpfa, pp.Mpsa, pp.Biot]
):
    """Test that the discretization matrices produced by Mpxa are the same if they
    are split into subproblems or not.

    The test is designed to be run with all three discretization classes, Mpfa, Mpsa and
    Biot. This leads to a bit of boilerplate for the two former classes (e.g., we define
    a FourthOrderTensor when testing Mpfa), but it is worth it, since it allows us to
    reuse the test for all three classes.

    Failure of this test indicates that the gradual discretization, by means of
    constructing subgrids, is not working. Provided that the standard tests for
    discretizing do not fail, a likely reason for the failure is that something has
    changed in the extraction of subgrids, including the construction of mappings
    between local and global grids; see the individual discretization classes for
    details.

    """

    # Define keywords
    if isinstance(discr_class, pp.Mpfa):
        # Pick the keyword set for the flow problem, set a default for the mechanics.
        flow_keyword = discr_class.keyword
        mechanics_keyword = "mechanics"
    elif isinstance(discr_class, pp.Biot):
        # Both keywords are set for Biot. We need to deal with this before the Mpsa
        # case, since Biot is a subclass of Mpsa.
        flow_keyword = "flow"
        mechanics_keyword = discr_class.keyword
    elif isinstance(discr_class, pp.Mpsa):
        # Pick the keyword set for the mechanics problem, set a default for the flow.
        flow_keyword = "flow"
        mechanics_keyword = discr_class.keyword

    # Consider four different grids; this should be enough to cover all cases. The size
    # of the grids are a bit random here, what we really want is to allow a division
    # into subgrids that include an overlap, but also cells on each side which is not
    # part of the overlap.
    #
    # Note: While it is tempting to use parametrization for this loop over grids, EK
    # found no easy way to do that which also allows for calling this test from a
    # shallow test wrapper which is unique to each discretization class (we could have
    # used the same parametrization on each of the wrappers, but that would have been a
    # bit messy).
    grid_list = [
        pp.CartGrid(np.array([4, 2])),
        pp.CartGrid(np.array([4, 2, 2])),
        pp.StructuredTriangleGrid(np.array([4, 2])),
        pp.StructuredTetrahedralGrid(np.array([4, 2, 2])),
    ]

    # Loop over the grids, discretize twice (with different data dictionaries): Once
    # wiht an enforced splitting into subdomains, once with no such splitting.
    for g in grid_list:
        g.compute_geometry()
        nc = g.num_cells

        # Parameter dictionaries for the two types of physics, can be common for both
        # the partitioned and non-partitioned schemes. Give boundary conditions and
        # tensors for both types of physics; although we may only use one of them
        # (unless Biot), but the overhead should be minimal.
        tensor = pp.SecondOrderTensor(kxx=np.ones(nc), kyy=10 * np.ones(nc))
        flow_param = {
            "bc": pp.BoundaryCondition(g),
            "second_order_tensor": pp.SecondOrderTensor(np.ones(nc)),
        }
        mechanics_param = {
            "bc": pp.BoundaryConditionVectorial(g),
            "fourth_order_tensor": pp.FourthOrderTensor(np.ones(nc), np.ones(nc)),
            "scalar_vector_mappings": {"foo": 1, "bar": tensor},
        }

        # Set up a data dictionary that will split the discretization into two.
        data_partition = {
            pp.PARAMETERS: {
                flow_keyword: flow_param,
                mechanics_keyword: mechanics_param,
            },
            pp.DISCRETIZATION_MATRICES: {flow_keyword: {}, mechanics_keyword: {}},
        }
        data_partition[pp.PARAMETERS][flow_keyword][  # type: ignore[index]
            "partition_arguments"
        ] = {"num_subproblems": 2}
        data_partition[pp.PARAMETERS][mechanics_keyword][  # type: ignore[index]
            "partition_arguments"
        ] = {"num_subproblems": 2}
        # Discretize
        discr_class.discretize(g, data_partition)

        # Set up a data dictionary that will not split the discretization into two.

        data_no_partition = {
            pp.PARAMETERS: {
                flow_keyword: flow_param,
                mechanics_keyword: mechanics_param,
            },
            pp.DISCRETIZATION_MATRICES: {flow_keyword: {}, mechanics_keyword: {}},
        }
        # Discretize
        discr_class.discretize(g, data_no_partition)

        # Compare the discretization matrices. We should have the same matrices for both
        # types of physics, and for all individual matrices.
        for key in [flow_keyword, mechanics_keyword]:
            for mat_key in data_partition[pp.DISCRETIZATION_MATRICES][
                key  # type: ignore[index]
            ]:
                data_with = data_partition[pp.DISCRETIZATION_MATRICES][
                    key  # type: ignore[index]
                ][mat_key]
                data_without = data_no_partition[pp.DISCRETIZATION_MATRICES][
                    key  # type: ignore[index]
                ][mat_key]

                if isinstance(data_with, dict):
                    # This is a Biot coupling discretization, where the matrices are
                    # stored in a dictionary, compare the individual matrices.
                    for sub_key in data_with:
                        assert np.allclose(
                            data_with[sub_key].toarray(),
                            data_without[sub_key].toarray(),
                        )
                else:
                    # The matrices are stored directly in the data dictionary.
                    assert np.allclose(data_with.toarray(), data_without.toarray())
