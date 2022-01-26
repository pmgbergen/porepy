import unittest

import numpy as np
import pytest
import scipy
import sympy
from scipy.sparse.linalg import spsolve

import porepy as pp


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


def perturb(g, rate, dx):
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
    # r3 = np.ravel(np.argwhere((g.nodes[0] < 1 - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < 0.75 - 1e-10) & (g.nodes[1] > 0.5 + 1e-10)))
    # r4 = np.ravel(np.argwhere((g.nodes[0] < 1 - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < 1.0 - 1e-10) & (g.nodes[1] > 0.75 + 1e-10)))
    pert_nodes = np.concatenate((r1, r2))
    npertnodes = pert_nodes.size
    rand = np.vstack((np.random.rand(g.dim, npertnodes), np.repeat(0.0, npertnodes)))
    g.nodes[:, pert_nodes] += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0
    return g


def make_grid(grid, grid_dims, domain):
    if grid.lower() == "cart":
        return pp.CartGrid(grid_dims, domain)
    elif grid.lower() == "triangular":
        return pp.StructuredTriangleGrid(grid_dims, domain)


class TestMPFAgravity(unittest.TestCase):
    def test_hydrostatic_pressure_1D(self):

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
        flux, bound_flux, _, _, div_g, _ = pp.Mpfa("flow")._flux_discretization(
            g, k, bound_cond, inverter="python"
        )
        div = pp.fvutils.scalar_divergence(g)
        a = div * flux

        flux_g = div_g * gforce
        # assemble rhs
        b = -div * bound_flux * p_bound - div * flux_g
        # solve system
        p = scipy.sparse.linalg.spsolve(a, b)
        q = flux * p + bound_flux * p_bound + flux_g
        p_ex = an_sol.p_f(xc[0])
        q_ex = np.zeros(g.num_faces)
        self.assertTrue(np.allclose(p, p_ex))
        self.assertTrue(np.allclose(q, q_ex))

    def test_hydrostatic_pressure(self):

        # Test mpfa_gravity in 2D Cartesian
        # and triangular grids
        # Should be exact for hydrostatic pressure
        # with stepwise gravity variation

        grids = ["cart", "triangular"]

        x, y = sympy.symbols("x y")
        g1 = 10
        g2 = 1
        p0 = 1  # reference pressure
        p = p0 + sympy.Piecewise(
            ((1 - y) * g1, y >= 0.5), (0.5 * g1 + (0.5 - y) * g2, y < 0.5)
        )
        an_sol = _SolutionHomogeneousDomainFlowWithGravity(p, x, y)

        for gr in grids:

            domain = np.array([1, 1])
            basedim = np.array([4, 4])
            pert = 0.5
            g = make_grid(gr, basedim, domain)
            g.compute_geometry()
            dx = np.max(domain / basedim)
            g = perturb(g, pert, dx)
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
            flux, bound_flux, _, _, div_g, _ = pp.Mpfa("flow")._flux_discretization(
                g, k, bound_cond, inverter="python"
            )
            div = pp.fvutils.scalar_divergence(g)
            a = div * flux
            flux_g = div_g * gforce
            b = -div * bound_flux * p_bound - div * flux_g
            p = scipy.sparse.linalg.spsolve(a, b)
            q = flux * p + bound_flux * p_bound + flux_g
            p_ex = an_sol.p_f(xc[0], xc[1])
            q_ex = np.zeros(g.num_faces)
            self.assertTrue(np.allclose(p, p_ex))
            self.assertTrue(np.allclose(q, q_ex))


def set_params_disrcetize(g, ambient_dim, method, periodic=False):
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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_1d_ambient_dim_1(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))

    ambient_dim = 1
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_1d_ambient_dim_2(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))

    ambient_dim = 2
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_1d_ambient_dim_2_nodes_reverted(method):
    # Same test as above, but with the orientation of the grid rotated.
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, -dx, -2 * dx]))

    ambient_dim = 2
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_1d_ambient_dim_3(method):
    dx = np.random.rand(1)[0]
    g = pp.TensorGrid(np.array([0, dx, 2 * dx]))
    g.nodes[:] = np.array([0, dx, 2 * dx])

    ambient_dim = 3
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

    # Prepare to solve problem
    A = div * flux
    rhs = -div * vector_source_discr

    # Make source strength another random number
    grav_strength = np.random.rand(1)

    # introduce a source term in x-direction
    g_x = np.zeros(g.num_cells * ambient_dim)
    g_x[::ambient_dim] = -1 * grav_strength  # /2 * dx
    p_x = np.linalg.pinv(A.toarray()).dot(rhs * g_x)
    print(dx)
    print(vector_source_discr.toarray())
    print(flux.toarray())

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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_2d_horizontal_ambient_dim_3(method):
    # Cartesian grid in xy-plane. The rotation of the grid in the mpfa discretization
    # will be trivial, leaving one source of error

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # Embed in 3d, this means that the vector source is a 3-vector per cell
    ambient_dim = 3

    # Discretization
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_2d_horizontal_ambient_dim_2(method):
    # Cartesian grid in xy-plane. The rotation of the grid in the mpfa discretization
    # will be trivial, leaving one source of error

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # The vector source is a 2-vector per cell
    ambient_dim = 2

    # Discretization
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method)

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


@pytest.mark.parametrize("method", ["tpfa"])
def test_2d_horizontal_periodic_ambient_dim_2(method):
    # Cartesian grid in xy-plane with periodic boundary conditions.

    # Random size of the domain
    dx = np.random.rand(1)[0]

    # 2x2 grid of the random size
    g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

    # The vector source is a 2-vector per cell
    ambient_dim = 2

    # Discretization
    flux, vector_source_discr, div = set_params_disrcetize(g, ambient_dim, method, True)

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


class TiltedGrids(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TiltedGrids, self).__init__(*args, **kwargs)
        self.keyword = "flow"

    def set_params_disrcetize(self, g, ambient_dim=3):
        g.compute_geometry()

        bc = pp.BoundaryCondition(g)
        k = pp.SecondOrderTensor(np.ones(g.num_cells))

        params = {
            "bc": bc,
            "second_order_tensor": k,
            "mpfa_inverter": "python",
            "ambient_dimension": ambient_dim,
        }

        data = pp.initialize_data(g, {}, self.keyword, params)
        discr = pp.Mpfa(self.keyword)
        discr.discretize(g, data)

        flux = data[pp.DISCRETIZATION_MATRICES][self.keyword][discr.flux_matrix_key]
        vector_source = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            discr.vector_source_matrix_key
        ]
        div = pp.fvutils.scalar_divergence(g)
        return flux, vector_source, div

    def test_assembly(self):
        # Test the assemble_matrix_rhs method, with vector sources included.
        # The rest of the setup is identical to that in
        # self.test_2d_horizontal_ambient_dim_2()

        # Random size of the domain
        dx = np.random.rand(1)[0]

        # 2x2 grid of the random size
        g = pp.CartGrid([2, 2], [2 * dx, 2 * dx])

        # Hhe vector source is a 2-vector per cell
        ambient_dim = 2

        g.compute_geometry()

        bc = pp.BoundaryCondition(g)
        k = pp.SecondOrderTensor(np.ones(g.num_cells))

        # Make source strength another random number
        grav_strength = np.random.rand(1)

        # introduce a source term in x-direction
        g_x = np.zeros(g.num_cells * ambient_dim)
        g_x[::ambient_dim] = -1 * grav_strength

        params = {
            "bc": bc,
            "bc_values": np.zeros(g.num_faces),
            "second_order_tensor": k,
            "mpfa_inverter": "python",
            "ambient_dimension": ambient_dim,
            "vector_source": g_x,
        }

        data = pp.initialize_data(g, {}, self.keyword, params)

        discr = pp.Mpfa(self.keyword)
        discr.discretize(g, data)

        A, b = discr.assemble_matrix_rhs(g, data)

        p_x = np.linalg.pinv(A.toarray()).dot(b)

        # The solution should be higher in the first x-row of cells, with magnitude
        # controlled by grid size and source stregth
        self.assertTrue(np.allclose(p_x[0] - p_x[1], dx * grav_strength))
        self.assertTrue(np.allclose(p_x[2] - p_x[3], dx * grav_strength))
        # The solution should be equal for equal x-coordinate
        self.assertTrue(np.allclose(p_x[0], p_x[2]))
        self.assertTrue(np.allclose(p_x[1], p_x[3]))

        data[pp.STATE] = {"pressure": p_x}
        pp.fvutils.compute_darcy_flux(g, data=data)


if __name__ == "__main__":
    unittest.main()
