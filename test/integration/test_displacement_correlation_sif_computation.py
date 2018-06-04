"""
In these tests, we set up a displacement field, evaluate stress intensity
factors (by the displacement correlation method) and compare against the
analytical solution. The test is performed in two and three dimensions for
cartesian and simplex grids.
The domain contains a single immersed through-the-thickness fracture. For the
simplex grid, inclination axis is described by the angle beta. A normal
traction of sigma is applied on two sides. Thanks to symmetry, only half of the
original domain needs to be simulated.
The test case is described in more detail as case (i) in section 6.1 of
Nejati et al., On the use of quarter-point tetrahedral finite elements in
linear elastic fracture mechanics, 2015.

In this test case, the normal traction on the top and bottom of the domain
is denoted by
    sigma
The fracture aperture is
    a
and its inclination to the last axis (y in 2d, z in 3d) is
    beta.
The domain is unit square in 2d, and has an (y-direction!) thickness of t in
3d. The fracture is located at the domain center.
2d example with beta = pi / 2:

    ^     ^
    |     |
-----------------
|               |
|               |
|      --       |
|               |
|               |
-----------------
    |     |
    v     v
"""

import scipy.sparse as sps
import numpy as np
import unittest

import porepy as pp

from test.integration.setup_mixed_dimensional_grids import set_bc_mech_tension


#-----------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    #-------------------------------------------------------------------------#

    def test_sif_convergence_cartesian_2d(self):
        _, error = cartesian_2d([20, 20], .05, .001, rm_factor=.5)
        assert max(error) < .08

    def test_sif_convergence_simplex_2d(self):
        sifs, error = simplex_2d(20, .05, .001, np.pi / 2, rm_factor=.8)
        phi = pp.propagate_fracture.propgation_angle(sifs)
        print(phi)
        assert np.max(np.absolute(phi)) < .05
        assert max(error) < .04
        _, error = simplex_2d(30, .05, .001, np.pi / 3, rm_factor=.8)
        assert max(error) < .2

    def test_sif_convergence_cartesian_3d(self):
        nc = [33, 3, 33]
        _, error = cartesian_3d(nc, .05, .001, 0.03, rm_factor=.5)
        assert max(error) < .11

    def test_sif_convergence_simplex_3d(self):
        sifs, error = simplex_3d(20, .05, .001, np.pi / 2, 0.03, rm_factor=.8)
        phi = pp.propagate_fracture.propgation_angle(sifs)
        assert np.max(np.absolute(phi)) < .05
        assert max(error) < .08
        sifs, error = simplex_3d(20, .05, .001, np.pi / 3, 0.03, rm_factor=.8)
        phi = pp.propagate_fracture.propgation_angle(sifs)
        assert max(error) < .13
        assert np.max(np.absolute(np.absolute(phi) - np.pi / 3)) < .05

    def test_two_fractures_cartesian_2d(self):
        a = .1
        n_cells = [10, 4]
        sigma = 1e-5
        rm_factor = .5
        dim_h = 2
        f_1 = np.array([[0, 2 * a],
                        [0.5, 0.5]])
        f_2 = np.array([[1 - a, 1],
                        [0.5, 0.5]])
        gb = pp.meshing.cart_grid([f_1, f_2], n_cells, **{'physdims': [1, 1]})
        assign_parameters(gb, sigma)

        # Discretize, solve and evaluate sifs. Dimension of array correspond to
        # fracture, sif mode and face.
        sifs = np.array(solve(gb, dim_h, rm_factor=rm_factor))

        # First fracture is twice as big as second, so K_I should be larger:
        assert sifs[0, 0, 0] > 1.5 * sifs[1, 0, 0]
        # Pure opening, so mode II should be zero:
        assert np.all(np.isclose(sifs[:, 1], 0))


def cartesian_2d(n_cells, a, sigma, rm_factor):
    # Make grid bucket and assign data
    dim_h = 2
    y_1 = 0.5
    x_0 = .5 - a
    x_1 = .5 + a

    f = np.array([[x_0, x_1],
                  [0.5, y_1]])
    gb = pp.meshing.cart_grid([f], n_cells, **{'physdims': [1, 1]})
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, rm_factor=rm_factor)

    # Analytical solution and error evaluation.
    beta = np.pi / 2
    K = analytical_sifs(a, sigma, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1)) / np.max(K)
    return sifs, e


def simplex_2d(mesh_size, a, sigma, beta, rm_factor):
    # Make grid bucket and assign data
    dim_h = 2
    y_0 = 0.5 - a * np.cos(beta)
    y_1 = 0.5 + a * np.cos(beta)
    x_0 = .5 - a * np.sin(beta)
    x_1 = .5 + a * np.sin(beta)
    f = np.array([[x_0, x_1],
                  [y_0, y_1]])
    box = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}
    mesh_kwargs = {}
    h = 1 / mesh_size
    mesh_kwargs = {'mesh_size_frac': h, 'mesh_size_min': 1 / 2 * h}

    gb = pp.meshing.simplex_grid([f], box, **mesh_kwargs)
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, rm_factor=rm_factor)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, sigma, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1)) / np.max(K)
    return sifs, e


def cartesian_3d(n_cells, a, sigma, t, rm_factor):
    # Make grid bucket and assign data
    dim_h = 3
    x_0 = .5 - a
    x_1 = .5 + a
    f = np.array([[x_0, x_1, x_1, x_0],
                  [0, 0, 2 * t, 2 * t],
                  [0.5, 0.5, 0.5, 0.5]])
    gb = pp.meshing.cart_grid([f], n_cells, **{'physdims': [1, 2 * t, 1]})
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, rm_factor=rm_factor)

    # Analytical solution and error evaluation.
    beta = np.pi / 2
    K = analytical_sifs(a, sigma, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1)) / np.max(K)
    return sifs, e


def simplex_3d(mesh_size, a, sigma, beta, t, rm_factor):
    # Make grid bucket and assign data
    dim_h = 3

    z_0 = 0.5 - a * np.cos(beta)
    z_1 = 0.5 + a * np.cos(beta)
    x_0 = .5 - a * np.sin(beta)
    x_1 = .5 + a * np.sin(beta)
    f = np.array([[x_0, x_1, x_1, x_0],
                  [0, 0, 2 * t, 2 * t],
                  [z_0, z_1, z_1, z_0]])
    box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
           'xmax': 1, 'ymax': 2 * t, 'zmax': 1}
    mesh_kwargs = {}
    h = 1 / mesh_size
    mesh_kwargs = {'mesh_size_frac': h, 'mesh_size_min': 1 / 2 * h}

    gb = pp.meshing.simplex_grid([f], box, **mesh_kwargs)
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, rm_factor=rm_factor)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, sigma, beta, dim_h)
    e = np.absolute(K - np.mean(np.absolute(sifs), axis=1)) / np.max(K)
    return sifs, e


def analytical_sifs(a, sigma, beta, dim):
    """
    Analytical stress intensity factors for the through-the-thickness crack
    problem in question.
    """
    K_I = sigma * np.sqrt(np.pi * a) * np.power(np.sin(beta), 2)
    K_II = sigma * np.sqrt(np.pi * a) * np.cos(beta) * np.sin(beta)
    K = np.array([K_I,  K_II])
    if dim == 3:
        K = np.append(K, 0)
    return K


def assign_parameters(gb, sigma=0.001):
    gb.add_node_props(['param'])
    for g, d in gb:
        param = pp.Parameters(g)
        d['param'] = param
    set_bc_mech_tension(gb, top_tension=sigma, fix_faces=False)


def solve(gb, dim_h, rm_factor):
    """
    Discretize and solve mechanical problem, and evaluate stress intensity
    factors by displacement correlation method.
    """
    discr = pp.FracturedMpsa(given_traction=True)
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)

    lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h)
    u = sps.linalg.spsolve(lhs_2, rhs_2)

    # SIF evaluation by displacemnt correlation
    critical_sifs = [.005, .1, .1]
    _, sifs = pp.displacement_correlation.faces_to_open(gb, u, critical_sifs,
                                                        rm_factor=rm_factor)
    if len(gb.grids_of_dimension(dim_h-1)) < 2:
        return sifs[0]
    else:
        return sifs


if __name__ == '__main__':
    unittest.main()
