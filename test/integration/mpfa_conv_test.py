"""
Unit tests for MPFA methods.

The tests are meant for ensuring detecting if the behavior of the
discretization changes. This is done by comparing the results on relatively
small and simple problems with those obtained at a time when the code was
considered 'correct' - (May 2016). The comparison in itself is rather ugly
(results are hard-coded).

The tests herein should be seen in connection with the jupyter notebook for
convergence tests located in the same folder (mpfa_conv_2d.ipynb)

If the tests fail, it is most likely caused by changes in the grid,
or in the discretization method.
"""
from __future__ import division
import numpy as np
import sympy
import scipy.sparse.linalg
import unittest
from math import pi

from porepy.grids import structured, simplex
from porepy.params import tensor, bc
from porepy.numerics.fv import mpfa, fvutils, mpsa
from porepy.utils.mcolon import mcolon
from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


class _SolutionHomogeneousDomainFlow(object):
    """Convenience class for representing an analytical solution, and its
    derivatives"""

    def __init__(self, u, x, y):
        u_f = sympy.lambdify((x, y), u, "numpy")
        dux = sympy.diff(u, x)
        duy = sympy.diff(u, y)
        dux_f = sympy.lambdify((x, y), dux, "numpy")
        duy_f = sympy.lambdify((x, y), duy, "numpy")
        rhs = -sympy.diff(dux, x) - sympy.diff(duy, y)
        rhs_f = sympy.lambdify((x, y), rhs, "numpy")
        self.u_f = u_f
        self.dux_f = dux_f
        self.duy_f = duy_f
        self.rhs_f = rhs_f


class _SolutionHomogeneousDomainElasticity(object):
    def __init__(self, ux, uy, x, y):
        ux_f = sympy.lambdify((x, y), ux, "numpy")
        uy_f = sympy.lambdify((x, y), uy, "numpy")
        dux_x = sympy.diff(ux, x)
        dux_y = sympy.diff(ux, y)
        duy_x = sympy.diff(uy, x)
        duy_y = sympy.diff(uy, y)
        divu = dux_x + duy_y

        sxx = 2 * dux_x + divu
        sxy = dux_y + duy_x
        syx = duy_x + dux_y
        syy = 2 * duy_y + divu

        sxx_f = sympy.lambdify((x, y), sxx, "numpy")
        sxy_f = sympy.lambdify((x, y), sxy, "numpy")
        syx_f = sympy.lambdify((x, y), syx, "numpy")
        syy_f = sympy.lambdify((x, y), syy, "numpy")

        rhs_x = sympy.diff(sxx, x) + sympy.diff(syx, y)
        rhs_y = sympy.diff(sxy, x) + sympy.diff(syy, y)
        rhs_x_f = sympy.lambdify((x, y), rhs_x, "numpy")
        rhs_y_f = sympy.lambdify((x, y), rhs_y, "numpy")
        self.ux_f = ux_f
        self.uy_f = uy_f
        self.sxx_f = sxx_f
        self.sxy_f = sxy_f
        self.syx_f = syx_f
        self.syy_f = syy_f
        self.rhs_x_f = rhs_x_f
        self.rhs_y_f = rhs_y_f


class MainTester(unittest.TestCase):
    """
    Wrapper for all system solves. One day, this can be replaced by an
    'elliptic solver' of some kind.

    The class contains methods for discretizing problems, and solving for a
    given right hand side. The design of these mehtods is poor (they should
    have been combined into one), but that is how it is.
    """

    def solve_system_homogeneous_perm(self, g, bound_cond, bound_faces, k, an_sol):
        """
        Set up and solve a problem with an analytical solution expressed in
        a simple form (probably this implies no heterogeneity, or some
        rewriting of the way analytical solutions are represented).
        """
        # Discretization. Use python inverter for speed
        flux, bound_flux, _, _ = mpfa.mpfa(g, k, bound_cond, inverter="python", eta=0)
        # Set up linear system
        div = fvutils.scalar_divergence(g)
        a = div * flux

        # Boundary values from analytical solution
        xf = g.face_centers
        u_bound = np.zeros(g.num_faces)
        u_bound[bound_faces] = an_sol.u_f(xf[0, bound_faces], xf[1, bound_faces])
        # Right hand side
        xc = g.cell_centers
        b = an_sol.rhs_f(xc[0], xc[1]) * g.cell_volumes - div * bound_flux * u_bound

        # Solve, compute flux, return
        u_num = scipy.sparse.linalg.spsolve(a, b)
        flux_num = flux * u_num + bound_flux * u_bound
        return u_num, flux_num

    def solve_system_chi_type_perm(self, g, bound_cond, an_sol, chi, kappa):
        """
        Set up and solve a problem where
        1) the permeability is given on the form perm = (1-chi) + chi *
        kappa, with chi a characteristic function of a subdomain,
        2) The solution takes the form u_full = u / ((1-chi) + chi*kappa),
        so that the flux is independent of kappa.

        Note that for this to work, some care is needed when choosing the
        analytical solution: u must be zero on the line of the discontinuity.

        Also note that self.solve_system is a sub-method of this, with a
        single characteristic region.
        """
        # Compute permeability
        char_func_cells = chi(g.cell_centers[0], g.cell_centers[1]) * 1.
        perm_vec = (1 - char_func_cells) + kappa * char_func_cells
        perm = tensor.SecondOrderTensor(2, perm_vec)

        # The rest of the function is similar to self.solve.system, see that
        # for comments.
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        flux, bound_flux, _, _ = mpfa.mpfa(
            g, perm, bound_cond, inverter="python", eta=0
        )

        xc = g.cell_centers
        xf = g.face_centers
        char_func_bound = chi(xf[0, bound_faces], xf[1, bound_faces]) * 1

        u_bound = np.zeros(g.num_faces)
        u_bound[bound_faces] = an_sol.u_f(xf[0, bound_faces], xf[1, bound_faces]) / (
            (1 - char_func_bound) + kappa * char_func_bound
        )

        div = fvutils.scalar_divergence(g)
        a = div * flux

        b = an_sol.rhs_f(xc[0], xc[1]) * g.cell_volumes - div * bound_flux * u_bound

        u_num = scipy.sparse.linalg.spsolve(a, b)
        flux_num = flux * u_num + bound_flux * u_bound
        return u_num, flux_num

    def solve_system_homogeneous_elasticity(
        self, g, bound_cond, bound_faces, k, an_sol
    ):
        stress, bound_stress = mpsa.mpsa(g, k, bound_cond, inverter="python", eta=0)
        div = fvutils.vector_divergence(g)
        a = div * stress

        # Boundary conditions
        xf = g.face_centers
        u_bound = np.zeros((g.dim, g.num_faces))
        u_bound[0, bound_faces] = an_sol.ux_f(xf[0, bound_faces], xf[1, bound_faces])
        u_bound[1, bound_faces] = an_sol.uy_f(xf[0, bound_faces], xf[1, bound_faces])

        # Right hand side - contribution from the solution and the boundary
        # conditions
        xc = g.cell_centers
        rhs = (
            np.vstack((an_sol.rhs_x_f(xc[0], xc[1]), an_sol.rhs_y_f(xc[0], xc[1])))
            * g.cell_volumes
        )
        b = rhs.ravel("F") - div * bound_stress * u_bound.ravel("F")

        # Solve system, derive fluxes
        u_num = scipy.sparse.linalg.spsolve(a, b)
        stress_num = stress * u_num + bound_stress * u_bound.ravel("F")
        return u_num, stress_num

    def solve_system_chi_type_elasticity(self, g, bound_cond, an_sol, chi, kappa):
        char_func_cells = chi(g.cell_centers[0], g.cell_centers[1]) * 1.
        mat_vec = (1 - char_func_cells) + kappa * char_func_cells

        k = tensor.FourthOrderTensor(2, mat_vec, mat_vec)
        stress, bound_stress = mpsa.mpsa(g, k, bound_cond, inverter="python", eta=0)
        div = fvutils.vector_divergence(g)
        a = div * stress

        # Boundary conditions
        xf = g.face_centers
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        char_func_bound = chi(xf[0, bound_faces], xf[1, bound_faces]) * 1
        u_bound = np.zeros((g.dim, g.num_faces))
        u_bound[0, bound_faces] = an_sol.ux_f(
            xf[0, bound_faces], xf[1, bound_faces]
        ) / ((1 - char_func_bound) + kappa * char_func_bound)
        u_bound[1, bound_faces] = an_sol.uy_f(
            xf[0, bound_faces], xf[1, bound_faces]
        ) / ((1 - char_func_bound) + kappa * char_func_bound)

        # Right hand side - contribution from the solution and the boundary
        # conditions
        xc = g.cell_centers
        rhs = (
            np.vstack((an_sol.rhs_x_f(xc[0], xc[1]), an_sol.rhs_y_f(xc[0], xc[1])))
            * g.cell_volumes
        )
        b = rhs.ravel("F") - div * bound_stress * u_bound.ravel("F")

        # Solve system, derive fluxes
        u_num = scipy.sparse.linalg.spsolve(a, b)
        stress_num = stress * u_num + bound_stress * u_bound.ravel("F")
        return u_num, stress_num


class CartGrid2D(MainTester):
    """
    Tests of Cartesian grids in 2D.
    """

    def setUp(self):
        # Set random seed
        np.random.seed(42)
        nx = np.array([4, 4])
        domain = np.array([1, 1])
        g = structured.CartGrid(nx, physdims=domain)

        # Perturbation rates, same notation as in setup_grids.py
        pert = 0.5
        dx = 0.25
        g = setup_grids.perturb(g, pert, dx)
        g.compute_geometry()
        self.g_nolines = g

        # Define a characteristic function which is True in the region
        # x > 0.5, y > 0.5
        def chi(xcoord, ycoord):
            return np.logical_and(np.greater(xcoord, 0.5), np.greater(ycoord, 0.5))

        # Create a new grid, which will not have faces along the
        # discontinuity perturbed
        g = structured.CartGrid(nx, physdims=domain)
        g.compute_geometry()
        old_nodes = g.nodes.copy()
        dx = np.max(domain / nx)
        np.random.seed(42)
        g = setup_grids.perturb(g, pert, dx)

        # Characteristic function for all cell centers
        xc = g.cell_centers
        chi = chi(xc[0], xc[1])
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
        bnd_nodes = node_ind[mcolon(node_ptr[bnd_face], node_ptr[bnd_face + 1])]
        g.nodes[:, bnd_nodes] = old_nodes[:, bnd_nodes]
        g.compute_geometry()
        self.g_lines = g

        # Define boundary faces and conditions
        self.bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        self.bc = bc.BoundaryCondition(
            g, self.bound_faces, ["dir"] * self.bound_faces.size
        )

    def test_homogeneous_mpfa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        x, y = sympy.symbols("x y")
        u = sympy.sin(x) * sympy.cos(y)
        an_sol = _SolutionHomogeneousDomainFlow(u, x, y)

        perm = 1
        k = tensor.SecondOrderTensor(2, perm * np.ones(self.g_nolines.num_cells))
        u_num, flux_num = self.solve_system_homogeneous_perm(
            self.g_nolines, self.bc, self.bound_faces, k, an_sol
        )
        # Hard coded values for potential and flux
        u_precomp = np.array(
            [
                0.11215902,
                0.37777176,
                0.61052237,
                0.77095723,
                0.09728957,
                0.36730807,
                0.56046212,
                0.70182774,
                0.10112506,
                0.31733332,
                0.47708993,
                0.60621791,
                0.06218831,
                0.23226929,
                0.36654522,
                0.49221303,
            ]
        )
        flux_precomp = np.array(
            [
                -0.22614269,
                -0.23671876,
                -0.16849627,
                -0.21205881,
                -0.19724312,
                -0.25489906,
                -0.25400658,
                -0.20711298,
                -0.13397631,
                -0.07618574,
                -0.13563931,
                -0.20535843,
                -0.16753938,
                -0.17611844,
                -0.11203853,
                -0.2116459,
                -0.14134316,
                -0.1733023,
                -0.09103128,
                -0.1022011,
                -0.07195505,
                0.03399637,
                0.00940329,
                -0.04211765,
                -0.0475905,
                0.01355307,
                0.11901831,
                0.06015525,
                -0.0321949,
                0.02277024,
                0.09980502,
                0.07256712,
                0.05134825,
                0.02497919,
                0.15964945,
                0.08262653,
                -0.01126896,
                0.09066114,
                0.12487007,
                0.14794111,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(flux_num, flux_precomp, atol=1e-10).all()

    def test_homogeneous_mpsa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        x, y = sympy.symbols("x y")
        ux = sympy.sin(x) * sympy.cos(y)
        uy = sympy.sin(x) * x ** 2
        an_sol = _SolutionHomogeneousDomainElasticity(ux, uy, x, y)

        muc = np.ones(self.g_nolines.num_cells)
        lambdac = muc
        k = tensor.FourthOrderTensor(2, muc, lambdac)

        u_num, stress_num = self.solve_system_homogeneous_elasticity(
            self.g_nolines, self.bc, self.bound_faces, k, an_sol
        )
        # Hard coded values for potential and flux
        u_precomp = np.array(
            [
                0.11366212,
                0.0016265,
                0.38066164,
                0.05290915,
                0.60764223,
                0.26625161,
                0.76716622,
                0.59179678,
                0.09577507,
                0.00561839,
                0.3626334,
                0.05751547,
                0.56062814,
                0.24723059,
                0.70605554,
                0.55294348,
                0.10303519,
                0.00842774,
                0.32185779,
                0.05971505,
                0.47883107,
                0.23055903,
                0.60543074,
                0.52036681,
                0.06327931,
                0.00702883,
                0.23617388,
                0.04850275,
                0.36972696,
                0.21029376,
                0.49293205,
                0.56432259,
            ]
        )

        stress_precomp = np.array(
            [
                0.68484199,
                0.02895805,
                0.73603478,
                0.1475628,
                0.48215302,
                0.12417911,
                0.62809128,
                0.39723115,
                0.45344911,
                0.69887207,
                0.76189355,
                0.01707399,
                0.71496816,
                -0.07979219,
                0.63322857,
                0.15883014,
                0.49902082,
                0.27421345,
                0.39221742,
                0.36820387,
                0.40893422,
                -0.01739874,
                0.66081767,
                0.08374431,
                0.54320625,
                0.14502388,
                0.46352808,
                0.24266992,
                0.30123855,
                0.34816392,
                0.62928856,
                -0.03865476,
                0.43785531,
                0.00720618,
                0.56430799,
                0.1143649,
                0.29710911,
                0.17885921,
                0.24513327,
                0.34234796,
                0.25399609,
                0.33148205,
                0.00877334,
                0.17425769,
                0.23058976,
                0.17996427,
                0.46815353,
                0.21598288,
                0.17522634,
                0.24058778,
                0.16709671,
                0.32158003,
                -0.04745346,
                0.06141119,
                0.40861851,
                0.14410509,
                0.18957558,
                0.32701767,
                0.13661476,
                0.2044117,
                -0.02109245,
                0.06352552,
                0.37501704,
                0.18515113,
                -0.08995594,
                0.19745797,
                0.17417018,
                0.21385356,
                -0.04394501,
                0.07499957,
                0.38902793,
                0.22864314,
                0.08610625,
                0.09333756,
                -0.01972811,
                0.14977722,
                0.12827059,
                0.11835038,
                0.33271425,
                0.18711232,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(stress_num, stress_precomp, atol=1e-10).all()

    def test_heterogeneous_kappa_1e_neg6_mpfa(self):
        """
        2D cartesian grid (perturbed) with permeability given as

        k(x,y) = 1e-6, x > 0.5, y > 0.5
        k(x,y) = 1 otherwise

        Analytical solution: sin(2 * x * pi) * sin(2 * pi * y).

        Note that the analytical solution is zero along the discontinuity.

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        kappa = 1e-6

        x, y = sympy.symbols("x y")
        u = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
        an_sol = _SolutionHomogeneousDomainFlow(u, x, y)

        def chi(xc, yc):
            return np.logical_and(np.greater(xc, 0.5), np.greater(yc, 0.5))

        u_num, flux_num = self.solve_system_chi_type_perm(
            self.g_lines, self.bc, an_sol, chi, kappa
        )
        u_precomp = np.array(
            [
                3.91688459e-01,
                5.44014582e-01,
                -6.95254908e-01,
                -6.91407727e-01,
                5.27854234e-01,
                8.73164613e-01,
                -3.68081801e-01,
                -2.90171573e-01,
                -5.64748667e-01,
                -5.95331084e-01,
                7.16007640e+05,
                7.52597195e+05,
                -5.24438451e-01,
                -7.22016098e-01,
                4.95904641e+05,
                5.32583312e+05,
            ]
        )
        flux_precomp = np.array(
            [
                -1.11909891,
                -0.45106523,
                0.84944736,
                -0.06652199,
                -2.16807666,
                -1.12677713,
                -0.45344628,
                1.6102826,
                -0.03276262,
                -0.58079751,
                0.71460786,
                0.19652822,
                -1.35594388,
                -0.06197948,
                1.37532039,
                1.79954634,
                0.1518235,
                -1.12637603,
                -0.04696978,
                1.15521742,
                -1.64778558,
                -1.10489621,
                1.19653738,
                0.79433075,
                -0.27014072,
                -0.47589136,
                -0.49921764,
                -0.38658258,
                1.8201197,
                0.88930865,
                -1.29370566,
                -1.32735917,
                -0.10266602,
                0.08101587,
                0.32309499,
                0.33810848,
                -0.61819902,
                -1.65272094,
                1.14414139,
                1.18782153,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(flux_num, flux_precomp, atol=1e-10).all()

    def test_heterogeneous_kappa_1e_pos6_mpfa(self):
        """
        2D cartesian grid (perturbed) with permeability given as

        k(x,y) = 1e6, x > 0.5, y > 0.5
        k(x,y) = 1 otherwise

        Analytical solution: sin(2 * x * pi) * sin(2 * pi * y).

        Note that the analytical solution is zero along the discontinuity.

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        kappa = 1e6

        x, y = sympy.symbols("x y")
        u = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
        an_sol = _SolutionHomogeneousDomainFlow(u, x, y)

        def chi(xc, yc):
            return np.logical_and(np.greater(xc, 0.5), np.greater(yc, 0.5))

        u_num, flux_num = self.solve_system_chi_type_perm(
            self.g_lines, self.bc, an_sol, chi, kappa
        )
        u_precomp = np.array(
            [
                3.86896239e-01,
                5.25655435e-01,
                -7.35593962e-01,
                -7.20356456e-01,
                5.13670140e-01,
                8.14201598e-01,
                -5.31488005e-01,
                -4.11844753e-01,
                -5.75653573e-01,
                -6.28177518e-01,
                9.38100917e-07,
                8.71900958e-07,
                -5.23515554e-01,
                -7.05297017e-01,
                5.28281641e-07,
                5.58253234e-07,
            ]
        )
        flux_precomp = np.array(
            [
                -1.10722384,
                -0.42775993,
                0.85224437,
                -0.09453212,
                -2.26260685,
                -1.10152723,
                -0.41307325,
                1.72965213,
                -0.07004913,
                -0.75104724,
                0.72531379,
                0.2155964,
                -1.30860082,
                0.06961809,
                1.61515011,
                1.79751536,
                0.13811229,
                -1.23375323,
                -0.046755,
                1.21561606,
                -1.62410537,
                -1.06935512,
                1.27398432,
                0.82059427,
                -0.25789075,
                -0.41984198,
                -0.39096356,
                -0.29379901,
                1.81724656,
                0.86636152,
                -1.02879553,
                -1.10161239,
                -0.11390141,
                0.02979385,
                0.50375061,
                0.45562311,
                -0.61775419,
                -1.61027696,
                1.21720503,
                1.24515229,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(flux_num, flux_precomp, atol=1e-10).all()

    def test_heterogeneous_kappa_1e_neg6_mpsa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        kappa = 1e-6
        x, y = sympy.symbols("x y")
        ux = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
        uy = sympy.cos(pi * x) * (y - 0.5) ** 2
        an_sol = _SolutionHomogeneousDomainElasticity(ux, uy, x, y)

        def chi(xc, yc):
            return np.logical_and(np.greater(xc, 0.5), np.greater(yc, 0.5))

        u_num, stress_num = self.solve_system_chi_type_elasticity(
            self.g_lines, self.bc, an_sol, chi, kappa
        )
        # Hard coded values for potential and flux

        u_precomp = np.array(
            [
                4.34828020e-01,
                -4.28604711e-02,
                4.44822207e-01,
                2.38491959e-01,
                -7.44343744e-01,
                -6.57167144e-02,
                -7.00162624e-01,
                -3.00238227e-01,
                5.87213962e-01,
                4.38867039e-01,
                8.23865674e-01,
                -1.13226288e-02,
                -3.95485035e-01,
                -1.99228558e-01,
                -3.86032264e-01,
                1.98718901e-01,
                -6.10107048e-01,
                3.92175454e-01,
                -4.61175691e-01,
                -9.54876903e-02,
                7.16825459e+05,
                -1.16316129e+05,
                7.47596763e+05,
                1.37255259e+05,
                -4.88145864e-01,
                -2.05122149e-02,
                -6.50227511e-01,
                2.10004595e-01,
                5.11591679e+05,
                6.36396262e+04,
                5.40606333e+05,
                -2.95726100e+05,
            ]
        )
        stress_precomp = np.array(
            [
                3.32923060e+00,
                4.80967293e-01,
                4.75468311e-01,
                9.79098447e-01,
                -2.41353425e+00,
                -8.25311645e-02,
                4.92754427e-01,
                -8.70707593e-01,
                6.40562038e+00,
                -6.99251805e-01,
                3.62887576e+00,
                1.24576479e-01,
                7.41411040e-01,
                -1.01589906e+00,
                -4.62752910e+00,
                -1.54194953e-01,
                -8.81738306e-03,
                9.75232315e-01,
                2.03840198e+00,
                3.94212745e-01,
                -2.40312366e+00,
                -9.84146744e-04,
                1.44439248e-01,
                -1.26062693e+00,
                4.01807150e+00,
                -2.18985536e-01,
                6.69214843e-02,
                9.75560080e-01,
                -4.16164304e+00,
                2.61832387e-01,
                -4.78158572e+00,
                4.00765626e-01,
                -4.17553339e-01,
                6.68855436e-01,
                3.43078750e+00,
                1.76856058e-01,
                -3.19126220e-03,
                -8.16523644e-01,
                -3.70740449e+00,
                -3.94622916e-01,
                1.88586720e+00,
                -8.33488229e-01,
                9.75877417e-01,
                3.12724793e-01,
                -1.54364949e+00,
                1.36688693e-06,
                -3.99045087e-01,
                1.75368137e-02,
                7.52661873e-01,
                1.77359758e+00,
                3.06531877e-01,
                -1.61010433e+00,
                9.96675947e-01,
                -9.49588482e-01,
                3.75235100e-01,
                2.15856369e+00,
                -1.84643256e+00,
                -5.28927672e-01,
                -1.02791621e+00,
                -5.48170356e-02,
                1.33910753e+00,
                6.06184771e-02,
                1.32712008e+00,
                7.05495950e-02,
                4.51513684e-01,
                -1.89761162e+00,
                -2.70617671e-01,
                1.52758689e+00,
                -6.39287873e-01,
                1.14606454e+00,
                -6.98695191e-01,
                -2.00244197e+00,
                3.32888439e-01,
                6.13216163e-01,
                1.61829667e+00,
                -1.94315594e-01,
                -1.25632372e+00,
                -3.92829981e-01,
                -1.20246999e+00,
                -2.09484045e-01,
            ]
        )
        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(stress_num, stress_precomp, atol=1e-10).all()

    if __name__ == "__main__":
        unittest.main()


class TriangleGrid2D(MainTester):
    """
    Tests of simplex grids in 2D.
    """

    def setUp(self):
        # Set random seed
        np.random.seed(42)
        nx = np.array([4, 4])
        domain = np.array([1, 1])
        g = simplex.StructuredTriangleGrid(nx, physdims=domain)

        # Perturbation rates, same notation as in setup_grids.py
        pert = 0.5
        dx = 0.25
        g = setup_grids.perturb(g, pert, dx)
        g.compute_geometry()
        self.g_nolines = g

        # Define a characteristic function which is True in the region
        # x > 0.5, y > 0.5
        def chi(xcoord, ycoord):
            return np.logical_and(np.greater(xcoord, 0.5), np.greater(ycoord, 0.5))

        # Create a new grid, which will not have faces along the
        # discontinuity perturbed
        g = simplex.StructuredTriangleGrid(nx, physdims=domain)
        g.compute_geometry()
        old_nodes = g.nodes.copy()
        dx = np.max(domain / nx)
        np.random.seed(42)
        g = setup_grids.perturb(g, pert, dx)

        # Characteristic function for all cell centers
        xc = g.cell_centers
        chi = chi(xc[0], xc[1])
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
        bnd_nodes = node_ind[mcolon(node_ptr[bnd_face], node_ptr[bnd_face + 1])]
        g.nodes[:, bnd_nodes] = old_nodes[:, bnd_nodes]
        g.compute_geometry()
        self.g_lines = g

        # Define boundary faces and conditions
        self.bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        self.bc = bc.BoundaryCondition(
            g, self.bound_faces, ["dir"] * self.bound_faces.size
        )

    def test_homogeneous_mpfa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        x, y = sympy.symbols("x y")
        u = sympy.sin(x) * sympy.cos(y)
        an_sol = _SolutionHomogeneousDomainFlow(u, x, y)
        g = self.g_nolines
        bound_cond = self.bc
        bound_faces = self.bound_faces
        perm = 1
        k = tensor.SecondOrderTensor(2, perm * np.ones(g.num_cells))

        u_num, flux_num = self.solve_system_homogeneous_perm(
            g, bound_cond, bound_faces, k, an_sol
        )

        # Hard coded values for potential and flux
        u_precomp = np.array(
            [
                0.16107635,
                0.04460227,
                0.44388758,
                0.34038129,
                0.63045801,
                0.56741841,
                0.79083512,
                0.73753872,
                0.14570593,
                0.06198251,
                0.39847451,
                0.31697368,
                0.58878597,
                0.52329375,
                0.74148373,
                0.65776383,
                0.12987879,
                0.03270606,
                0.37239417,
                0.26970513,
                0.51268517,
                0.44713568,
                0.64796637,
                0.56339303,
                0.08862535,
                0.03561036,
                0.26618302,
                0.18073226,
                0.41137445,
                0.33476786,
                0.53204111,
                0.44254309,
            ]
        )
        flux_precomp = np.array(
            [
                0.07224276,
                0.22520049,
                0.17438399,
                -0.03106521,
                -0.23518914,
                0.21772969,
                -0.00907209,
                -0.16780779,
                0.26366039,
                0.04119693,
                -0.2113708,
                0.21537036,
                -0.19676077,
                -0.04846246,
                0.25316387,
                0.21519289,
                0.01138746,
                -0.25353288,
                0.25465231,
                0.11863116,
                -0.20781077,
                0.28400782,
                0.06103447,
                -0.1345499,
                0.17170964,
                -0.0746532,
                -0.03170599,
                0.13539543,
                0.18490946,
                0.02175786,
                -0.20407522,
                0.2084673,
                0.09926638,
                -0.16599291,
                0.29765445,
                0.07124824,
                -0.17482109,
                0.21936224,
                -0.11164636,
                0.05092288,
                0.21120468,
                0.19719789,
                0.02376679,
                -0.14079355,
                0.21939472,
                0.15935784,
                -0.17298347,
                0.27041779,
                0.08215586,
                -0.09058382,
                0.21516999,
                -0.10237499,
                -0.01166704,
                0.08984219,
                0.12449397,
                0.14809246,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(flux_num, flux_precomp, atol=1e-10).all()

    def test_homogeneous_mpsa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        x, y = sympy.symbols("x y")
        ux = sympy.sin(x) * sympy.cos(y)
        uy = sympy.sin(x) * x ** 2
        an_sol = _SolutionHomogeneousDomainElasticity(ux, uy, x, y)

        muc = np.ones(self.g_nolines.num_cells)
        lambdac = muc
        k = tensor.FourthOrderTensor(2, muc, lambdac)

        u_num, stress_num = self.solve_system_homogeneous_elasticity(
            self.g_nolines, self.bc, self.bound_faces, k, an_sol
        )
        # Hard coded values for potential and flux
        u_precomp = np.array(
            [
                0.15856319,
                0.0075,
                0.04298072,
                -0.00079293,
                0.43107795,
                0.10715564,
                0.33244902,
                0.05167529,
                0.62022066,
                0.30753443,
                0.5543221,
                0.22721421,
                0.7869957,
                0.6654834,
                0.72869072,
                0.5516277,
                0.1385006,
                0.0157466,
                0.06000193,
                0.0085041,
                0.39103168,
                0.09516664,
                0.3074358,
                0.05323975,
                0.59110494,
                0.28586219,
                0.52769556,
                0.20543369,
                0.74200332,
                0.6640997,
                0.65726519,
                0.47923141,
                0.13084311,
                0.01622213,
                0.03428634,
                0.00352019,
                0.37070982,
                0.09298494,
                0.27081387,
                0.04674414,
                0.51863678,
                0.27388231,
                0.45230105,
                0.20205724,
                0.64715332,
                0.60240166,
                0.56084446,
                0.48053464,
                0.08648362,
                0.01190816,
                0.03544077,
                0.00380538,
                0.25916216,
                0.07420655,
                0.17748621,
                0.03627169,
                0.40720991,
                0.27779005,
                0.32996662,
                0.18466372,
                0.5284689,
                0.64230054,
                0.43615255,
                0.50965552,
            ]
        )

        stress_precomp = np.array(
            [
                -0.26958583,
                -0.33638689,
                -0.667947,
                -0.01742885,
                -0.47458633,
                0.21238819,
                -0.0349944,
                -0.16198909,
                0.72129694,
                0.15320213,
                -0.47553386,
                0.07875239,
                -0.27337858,
                -0.1670963,
                0.47281489,
                0.13315125,
                -0.46861074,
                -0.12624934,
                -0.48304876,
                -0.22723128,
                0.65555433,
                0.39386466,
                -0.15498256,
                -0.34261604,
                0.51841828,
                0.68522363,
                0.18865259,
                0.22811062,
                -0.74386934,
                -0.02560191,
                -0.52968486,
                0.30489755,
                0.18806925,
                0.30525305,
                0.69809253,
                -0.06651859,
                -0.51913659,
                0.23583853,
                -0.04135297,
                0.06045769,
                0.6362977,
                0.14952779,
                -0.60953407,
                -0.14399453,
                0.38650195,
                0.16491128,
                0.50652757,
                0.27122571,
                -0.07659649,
                -0.15409943,
                0.39105449,
                0.38504024,
                0.2016545,
                0.30974013,
                -0.3994646,
                0.01539455,
                -0.4740917,
                0.22705338,
                0.13767907,
                0.21021141,
                0.65066576,
                0.07504187,
                -0.44013249,
                0.09388028,
                -0.01937504,
                0.05587886,
                0.53637851,
                0.1573951,
                -0.53252462,
                -0.13551429,
                0.36175408,
                0.18516825,
                0.46601561,
                0.24146723,
                -0.02101134,
                -0.0909527,
                0.30983013,
                0.34634257,
                -0.07744481,
                0.19081557,
                -0.59843171,
                0.03032643,
                -0.50909979,
                0.15881483,
                0.17178383,
                0.19799585,
                0.42069205,
                0.01178103,
                -0.42643062,
                0.10703985,
                -0.05153871,
                0.08057204,
                0.55292553,
                0.12709268,
                -0.40762997,
                -0.05260906,
                0.36977486,
                0.22871008,
                0.315139,
                0.17973276,
                0.02248408,
                -0.05615675,
                0.28601249,
                0.34985441,
                0.08465242,
                0.09051632,
                -0.0282206,
                0.12533392,
                0.09117625,
                0.13517655,
                0.29061051,
                0.17997061,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(stress_num, stress_precomp, atol=1e-10).all()

    def test_heterogeneous_kappa_1e_neg6(self):
        """
        2D triangulars grid (perturbed) with permeability given as

        k(x,y) = 1e-6, x > 0.5, y > 0.5
        k(x,y) = 1 otherwise

        Analytical solution: sin(2 * x * pi) * sin(2 * pi * y).

        Note that the analytical solution is zero along the discontinuity.

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        kappa = 1e6

        x, y = sympy.symbols("x y")
        u = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
        an_sol = _SolutionHomogeneousDomainFlow(u, x, y)

        def chi(xc, yc):
            return np.logical_and(np.greater(xc, 0.5), np.greater(yc, 0.5))

        u_num, flux_num = self.solve_system_chi_type_perm(
            self.g_lines, self.bc, an_sol, chi, kappa
        )

        # For clarity: The discretization here is done with eta = 0,
        # this gives some oscillations at the heterogeneity
        u_precomp = np.array(
            [
                3.23882374e-01,
                2.00141263e-01,
                4.89788221e-02,
                6.50248262e-01,
                -6.26582286e-01,
                -5.85645414e-01,
                -2.23206214e-01,
                -8.61911020e-01,
                7.68008367e-01,
                1.55728946e-01,
                6.74836809e-01,
                5.99591360e-01,
                -7.17121651e-01,
                -2.80544428e-01,
                -3.19433071e-01,
                -3.13312194e-01,
                -5.27341943e-01,
                -2.21703640e-01,
                -1.48524532e-01,
                -7.68503490e-01,
                9.04646914e-07,
                7.97365667e-07,
                3.23832023e-07,
                1.01162732e-06,
                -6.70554892e-01,
                -1.62657580e-01,
                -5.00531058e-01,
                -4.89809540e-01,
                7.41287855e-07,
                2.59896652e-07,
                4.68358154e-07,
                3.56771478e-07,
            ]
        )
        flux_precomp = np.array(
            [
                1.50682603,
                1.01692398,
                0.0436895,
                0.9552813,
                -0.54725235,
                -1.65342246,
                -1.27117394,
                0.83402579,
                -0.25889507,
                -0.69655085,
                -0.01292093,
                1.91889083,
                -2.01822556,
                -0.46218259,
                0.95069045,
                1.89315773,
                -0.30483903,
                -0.28695981,
                0.11899968,
                -0.32077122,
                1.63225913,
                -1.56757238,
                -0.48490227,
                -0.11766933,
                -0.36985968,
                -0.73307357,
                1.6860483,
                -0.67473777,
                -0.27357371,
                0.8401944,
                0.29960646,
                1.64284045,
                -0.9733459,
                -1.13761907,
                0.51436174,
                -1.13192358,
                0.01456427,
                -1.87705695,
                1.36198364,
                0.02567319,
                -1.68354502,
                -1.67602555,
                -0.11905926,
                0.06053962,
                -0.24926609,
                0.43280923,
                -1.14624988,
                1.68966019,
                0.58267543,
                0.04319606,
                0.47524675,
                1.16926973,
                -0.52939579,
                -1.53356316,
                1.16026059,
                1.21286644,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(flux_num, flux_precomp, atol=1e-10).all()

    def test_heterogeneous_kappa_1e_neg6_mpsa(self):
        """
        2D cartesian grid (perturbed) with constant permeability of unity.

        Analytical solution: sin(x) * cos(y).

        The data is created by the jupyter notebook mpfa_conv_2d. To
        recreate data, use the following parameters:

        np.random.seed(42)
        base = 4
        domain = np.array([1, 1])  [NOTE: base and domain together implies a
                                    resolution dx = 0.25 as used in
                                    perturbations in self.setUp()
        basedim = np.array([base, base])
        num_refs = 1
        grid_type = 'cart' [NOTE: The script will run both Cartesian and
                            Simplex grid, only consider the first values]
        pert = 0.5
        """
        kappa = 1e-6
        x, y = sympy.symbols("x y")
        ux = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
        uy = sympy.cos(pi * x) * (y - 0.5) ** 2
        an_sol = _SolutionHomogeneousDomainElasticity(ux, uy, x, y)

        def chi(xc, yc):
            return np.logical_and(np.greater(xc, 0.5), np.greater(yc, 0.5))

        u_num, stress_num = self.solve_system_chi_type_elasticity(
            self.g_lines, self.bc, an_sol, chi, kappa
        )
        # Hard coded values for potential and flux

        u_precomp = np.array(
            [
                3.09390864e-01,
                1.34641782e-01,
                2.39984582e-01,
                1.40769722e-01,
                -1.01690824e-02,
                1.27214799e-01,
                6.25290354e-01,
                2.93655381e-01,
                -6.21697832e-01,
                -9.98054636e-02,
                -6.05268072e-01,
                1.39309160e-02,
                -2.52539272e-01,
                -3.08198323e-01,
                -8.77588051e-01,
                -1.75971892e-01,
                8.63812650e-01,
                4.18966686e-01,
                1.97526372e-01,
                3.47568703e-01,
                7.81745485e-01,
                9.96400435e-02,
                6.68681050e-01,
                2.46143679e-01,
                -6.17920741e-01,
                -2.36336034e-02,
                -1.17481291e-01,
                2.66781994e-02,
                -3.38240999e-01,
                1.08400498e-01,
                -2.15426044e-01,
                2.31935194e-01,
                -4.78229418e-01,
                3.12982884e-01,
                -2.64775005e-01,
                1.49997413e-01,
                1.07616544e-01,
                -1.05270256e-02,
                -5.35178837e-01,
                -1.53240133e-03,
                5.68991037e+05,
                1.89712057e+04,
                5.04435224e+05,
                -5.96455647e+04,
                2.41985386e+05,
                1.20387001e+05,
                8.92815531e+05,
                1.09523111e+05,
                -6.64055159e-01,
                3.86014521e-02,
                -1.78896900e-01,
                4.60447467e-02,
                -4.23566159e-01,
                2.04419759e-01,
                -4.35057023e-01,
                9.56055537e-02,
                6.83885576e+05,
                3.93671133e+04,
                2.26546428e+05,
                2.56681115e+04,
                4.47894042e+05,
                -1.64651914e+05,
                3.22225459e+05,
                -1.77346295e+05,
            ]
        )
        stress_precomp = np.array(
            [
                -1.61357268e+00,
                2.80488162e-01,
                -3.30186586e+00,
                -3.96733690e-01,
                -1.05855209e+00,
                1.65972580e-01,
                -8.69365614e-01,
                -1.90711867e-01,
                7.59404680e-01,
                1.02477676e+00,
                3.14402627e+00,
                -1.18972497e+00,
                1.52677689e+00,
                -1.16161943e-01,
                -2.42805581e+00,
                -1.44840056e-01,
                1.53171831e+00,
                2.46034913e-03,
                7.41828564e-02,
                1.20733804e-01,
                1.69615641e-01,
                -8.19989892e-01,
                -4.46900740e+00,
                2.76955699e+00,
                6.03830615e+00,
                -8.54287340e-01,
                1.23675204e+00,
                1.65992898e+00,
                -3.31236189e+00,
                6.96835654e-02,
                -3.25788011e+00,
                2.08920433e+00,
                2.91195954e-01,
                -1.48114864e+00,
                3.95226369e-01,
                -8.85307501e-01,
                7.75764611e-01,
                -1.39524507e-01,
                9.74156517e-01,
                -7.72101341e-01,
                -4.46911828e+00,
                -1.20590624e-01,
                3.56645157e+00,
                -1.45219055e+00,
                5.46701436e-01,
                2.23513666e+00,
                1.74449458e-01,
                9.48470755e-01,
                -4.61535220e-01,
                5.76862760e-01,
                2.25326564e+00,
                2.67592513e-01,
                -1.42268191e+00,
                -9.04967778e-01,
                2.31401211e+00,
                -1.04384062e-01,
                1.75957149e+00,
                -1.23059978e+00,
                -8.17646416e-01,
                -1.93107738e-01,
                1.12604490e-01,
                -1.18117811e+00,
                -4.17892656e+00,
                1.64442921e+00,
                1.21959672e+00,
                3.24295237e-01,
                4.00908758e+00,
                -2.61436595e-01,
                -2.32261688e+00,
                5.77455488e-01,
                1.01131873e+00,
                4.80335366e-01,
                4.96331808e-01,
                8.74556628e-01,
                3.55241411e+00,
                -1.90608302e+00,
                -3.78326576e+00,
                3.56825332e-01,
                1.90782184e-01,
                -1.84379643e+00,
                4.58239048e+00,
                -4.07064626e-01,
                3.61352071e+00,
                -1.78284718e+00,
                -9.40026685e-03,
                1.47155162e+00,
                -1.87760460e-01,
                6.83596274e-01,
                -8.73341300e-01,
                1.50132399e-01,
                -4.93138058e-01,
                1.11662381e+00,
                3.27912098e+00,
                2.34849922e-01,
                -3.00459720e+00,
                1.34554429e+00,
                -8.86961989e-01,
                -1.81949318e+00,
                -1.79817871e-01,
                -8.55922163e-01,
                4.57786835e-01,
                -4.04773077e-01,
                -3.50647833e+00,
                -2.74326653e-01,
                7.54153110e-02,
                6.31218484e-01,
                1.48522376e+00,
                9.06961018e-02,
                -1.11722601e+00,
                -3.04826450e-01,
                -1.17965266e+00,
                -4.79898022e-01,
            ]
        )

        assert np.isclose(u_num, u_precomp, atol=1e-10).all()
        assert np.isclose(stress_num, stress_precomp, atol=1e-10).all()


if __name__ == "__main__":
    unittest.main()
