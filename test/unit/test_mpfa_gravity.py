import numpy as np
import scipy
import sympy
import unittest
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
        flux, bound_flux, _, _, div_g = pp.Mpfa("flow").mpfa(
            g, k, bound_cond, vector_source=True, inverter="python"
        )
        div = pp.fvutils.scalar_divergence(g)
        a = div * flux

        # take arithmetic mean of cell center gravities
        fi, ci, _ = scipy.sparse.find(g.cell_faces)
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]
        dist_face_cell = np.linalg.norm(fc_cc, 2, axis=0)
        gforce = gforce[ci]
        fg = np.bincount(fi, weights=dist_face_cell * gforce)
        flux_g = fg * div_g
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
            flux, bound_flux, _, _, div_g = pp.Mpfa("flow").mpfa(
                g, k, bound_cond, vector_source=True, inverter="python"
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


if __name__ == "__main__":
    unittest.main()
