#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:18:33 2018

@author: ivar

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
"""

import scipy.sparse as sps
import numpy as np
import unittest
import matplotlib.pyplot as plt

from porepy.numerics.fv.mpsa import FracturedMpsa
from porepy.fracs.propagate_fracture import displacement_correlation
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.viz import plot_grid, exporter

from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.integration.setup_mixed_dimensional_grids import set_bc_mech_tension
from test.integration.fracture_propagation_utils import propagate_and_update, \
    check_equivalent_buckets, propagate_simple


#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    #------------------------------------------------------------------------------#

    def test_sif_convergence_cartesian_2d(self):
        _, error = cartesian_2d([40, 20], .05, 1)
        assert max(error) < .08

    def test_sif_convergence_simplex_2d(self):
        _, error = simplex_2d(20, .05, 1, np.pi / 2)
        assert max(error) < .05
        _, error = simplex_2d(20, .05, 1, np.pi / 3)
        assert max(error) < .05

    def test_sif_convergence_cartesian_3d(self):
        nc = [20, 3, 20]
        _, error = cartesian_3d(nc, .05, 1, 0.03)
        assert max(error) < .04

    def test_sif_convergence_simplex_3d(self):
        _, error = simplex_3d(20, .05, 1, np.pi / 2, 0.03)
        assert max(error) < .03
        _, error = simplex_3d(20, .05, 1, np.pi / 3, 0.03)
        assert max(error) < .02


def cartesian_2d(n_cells, a, sigma):
    # Make grid bucket and assign data
    dim_h = 2
    y_1 = 0.5
    x_1 = a

    f = np.array([[0, x_1],
                  [0.5, y_1]])
    gb = meshing.cart_grid([f], n_cells, **{'physdims': [.5, 1]})
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h)

    # Analytical solution and error evaluation.
    beta = np.pi / 2
    K = analytical_sifs(a, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1))
    return sifs, e


def simplex_2d(mesh_size, a, sigma, beta):
    # Make grid bucket and assign data
    dim_h = 2
    y_1 = 0.5 + a * np.cos(beta)
    x_1 = a * np.sin(beta)
    f = np.array([[0, x_1],
                  [0.5, y_1]])
    box = {'xmin': 0, 'ymin': 0, 'xmax': .5, 'ymax': 1}
    mesh_kwargs = {}
    h = 1 / mesh_size
    mesh_kwargs = {'mesh_size_frac': h, 'mesh_size_min': 1 / 2 * h}

    gb = meshing.simplex_grid([f], box, **mesh_kwargs)
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1))
    return sifs, e


def cartesian_3d(n_cells, a, sigma, t):
    # Make grid bucket and assign data
    dim_h = 3
    f = np.array([[0, a, a, 0],
                  [0, 0, 2 * t, 2 * t],
                  [0.5, 0.5, 0.5, 0.5]])
    gb = meshing.cart_grid([f], n_cells, **{'physdims': [.5, 2 * t, 1]})
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h)

    # Analytical solution and error evaluation.
    beta = np.pi / 2
    K = analytical_sifs(a, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1))
    return sifs, e


def simplex_3d(mesh_size, a, sigma, beta, t):
    # Make grid bucket and assign data
    dim_h = 3
    z_0 = 0.5
    x_0 = 0
    z_1 = 0.5 + a * np.cos(beta)
    x_1 = a * np.sin(beta)

    f = np.array([[x_0, x_1, x_1, x_0],
                  [0, 0, 2 * t, 2 * t],
                  [z_0, z_1, z_1, z_0]])
    box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
           'xmax': .5, 'ymax': 2 * t, 'zmax': 1}
    mesh_kwargs = {}
    h = 1 / mesh_size
    mesh_kwargs = {'mesh_size_frac': h, 'mesh_size_min': 1 / 2 * h}

    gb = meshing.simplex_grid([f], box, **mesh_kwargs)
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1))
    return sifs, e


def analytical_sifs(a, beta, dim, sigma=1):
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


def assign_parameters(gb):
    gb.add_node_props(['param'])
    for g, d in gb:
        param = Parameters(g)
        d['param'] = param
    set_bc_mech_tension(gb)


def solve(gb, dim_h):
    """
    Discretize and solve mechanical problem, and evaluate stress intensity
    factors by displacement correlation method.
    """
    discr = FracturedMpsa(given_traction=True)
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)

    lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h)
    u = sps.linalg.spsolve(lhs_2, rhs_2)

    # SIF evaluation by displacemnt correlation
    critical_sifs = [.005, .1, .1]
    _, sifs = displacement_correlation(gb, u, critical_sifs)
    return sifs


if __name__ == '__main__':
    # Fracture length
    a = .05
    # Boundary traction (normal direction on top and bottom)
    sigma = 1
    # Fracture angle
    beta = np.pi / 2
    # Domain thickness (3d)
    t = 0.03
    nx = 20
#    BasicsTest().test_sif_convergence_cartesian_2d([2*nx, nx], a, sigma)
#    BasicsTest().test_sif_convergence_simplex_2d(1/nx, a, sigma, beta)
#    nc = [nx, np.ceil(2 * t * nx), nx]
#    BasicsTest().test_sif_convergence_cartesian_3d(nc, a, sigma, t)
#    BasicsTest().test_sif_convergence_simplex_3d(1/nx, a, sigma, beta, t)
#
#    # Test for other angle
#    beta = np.pi/3
#    BasicsTest().test_sif_convergence_simplex_2d(1/nx, a, sigma, beta)
    unittest.main()