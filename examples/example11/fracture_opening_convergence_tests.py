#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In these convergence tests, we set up a displacement field, evaluate stress 
intensity factors (by the displacement correlation method) and compare against 
the analytical solution. The test is performed in two and three dimensions for 
cartesian and simplex grids. 
The domain contains a single immersed through-the-thickness fracture. For the 
simplex grid, inclination axis is described by the angle beta. A normal 
traction of sigma is applied on two sides. Thanks to symmetry, only half of the 
original domain needs to be simulated. 
The test case is described in more detail as case (i) in section 6.1 of 
Nejati et al., On the use of quarter-point tetrahedral finite elements in 
linear elastic fracture mechanics, 2015. 
 
A simplified test version is found in the integration test 
test_displacement_correlation_sif_computation.py. There, the tests are run for 
only a single (rather coarse) mesh. 
""" 
 
import scipy.sparse as sps 
import numpy as np 
import matplotlib.pyplot as plt 
import porepy as pp 
 
from test.integration.setup_mixed_dimensional_grids import set_bc_mech_tension 
 
 
# -----cartesian_2d-----------------------------------------------------------# 
def cartesian_2d(n_cells, a, sigma, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 2 
    y_0 = 0.5 
    x_0 = .5 - a 
    y_1 = 0.5 
    x_1 = .5 + a 
 
    f = np.array([[x_0, x_1], 
                  [y_0, y_1]]) 
    gb = pp.meshing.cart_grid([f], n_cells, **{'physdims': [1, 1]}) 
    assign_parameters(gb) 
    print('lower-dim cartesian cells ', gb.grids_of_dimension(1)[0].num_cells) 
    # Discretize, solve and evaluate sifs. 
 
    # Analytical solution and error evaluation. 
    return gb 
 
 
# -----simplex_2d-------------------------------------------------------------# 
 
def simplex_2d(mesh_size, a, sigma, beta, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 2 
 
    y_0 = l / 2 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
    y_1 = l / 2 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, x_1], 
                  [y_0, y_1]]) 
    box = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': l} 
    mesh_kwargs = {} 
    h = 1 / mesh_size 
    mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': h, 
                                'h_min': 1 / 2 * h} 
 
    gb = pp.meshing.simplex_grid([f], box, **mesh_kwargs) 
#     assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    return gb 
 
 
# -----cartesian_3d-----------------------------------------------------------# 
def fracture_3d(beta, t): 
    #    y_0 = 0.5 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
#    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, x_1, x_1, x_0], 
                  [0, 0, 2 * t, 2 * t], 
                  [0.5, 0.5, 0.5, 0.5]]) 
    return [f] 
 
def cartesian_3d(n_cells, a, sigma, t, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 3 
    beta = np.pi/2 
    gb = pp.meshing.cart_grid(fracture_3d(beta, t), n_cells, **{'physdims': [1, 2 * t, 1]}) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
 
    # Analytical solution and error evaluation. 
    return sifs, e 
 
 
# -----simplex_3d-------------------------------------------------------------# 
 
def simplex_3d(mesh_size, a, sigma, beta, t, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 3 
 
    box = {'xmin': 0, 'ymin': 0, 'zmin': 0, 
           'xmax': 1, 'ymax': 2 * t, 'zmax': 1} 
    mesh_kwargs = {} 
    h = 1 / mesh_size 
    mesh_kwargs = {'h_ideal': h, 'h_min': 1 / 2 * h} 
 
    gb = pp.meshing.simplex_grid(fracture_3d(beta, t), box, **mesh_kwargs) 
    assign_parameters(gb, t) 
 
    # Discretize, solve and evaluate sifs. 
 
    # Analytical solution and error evaluation. 
 
    return gb 
 
 
 
# -----analytical-------------------------------------------------------------# 
 
def analytical_displacements(gb, a, sigma=1): 
    """ 
    Analytical stress intensity factors for the through-the-thickness crack 
    problem in question. 
    """ 
    dim_h = gb.dim_max() 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
 
    normal_ind = 0 
    last_ind = dim_h - 1 # pull in y direction for 2d, z in 3d 
    normal_coordinate = g_h.face_centers[normal_ind, g_h.frac_pairs] - .5 
    cons = 2*(1-d_h['Poisson']**2)/d_h['Young'] 
    u = cons * sigma * np.sqrt(np.power(a, 2) - np.power(normal_coordinate, 2)) 
    n_f = g_h.num_cells * dim_h 
    frac_ind_1 = np.arange(n_f + last_ind, n_f + g_h.frac_pairs.shape[1] * dim_h, dim_h) 
    frac_ind_2 = np.arange(last_ind + n_f + g_h.frac_pairs.shape[1] * dim_h, 
                           last_ind+ n_f + g_h.frac_pairs.size * dim_h,  dim_h) 
    frac_ind = np.array([frac_ind_1, frac_ind_2]) 
    return u, frac_ind 
 
 
# -----utility functions------------------------------------------------------# 
 
def assign_parameters(gb, t=None): 
    """ 
    Utility function to assign the parameters to the node data of the grid 
    bucket. 
    """ 
    gb.add_node_props(['param']) 
    for g, d in gb: 
        param = pp.Parameters(g) 
        d['param'] = param 
    set_bc_mech_tension(gb, t=t, l=l) 
 
 
def solve(gb, dim_h, kw={}): 
    """ 
    Discretize and solve mechanical problem, and evaluate stress intensity 
    factors by displacement correlation method. 
    """ 
    discr = pp.FracturedMpsa(given_traction=True) 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
 
    lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h) 
    u = sps.linalg.spsolve(lhs_2, rhs_2) 
    u_analytical, dof_ind = analytical_displacements(gb, a) 
    aperture = np.absolute(np.diff(u[dof_ind], axis=0)) 
    aperture_a = np.atleast_2d(np.sum(u_analytical, axis=0)) 
    return aperture, aperture_a 
 
 
def run_multiple_and_plot(nc, function): 
    errors = [] 
    errors_max = [] 
    apertures = [] 
    apertures_a = [] 
    fracture_cells = [] 
    for item in nc: 
        gb = function(item) 
        aperture, aperture_a = solve(gb, gb.dim_max()) 
 
        assert np.all(aperture > 0) 
 
        e = np.mean(np.absolute(aperture_a - aperture)/aperture_a) 
        errors.append(e) 
        e = np.max(np.absolute(aperture_a - aperture)/aperture_a) 
        errors_max.append(e) 
        apertures.append(aperture) 
        apertures_a.append(aperture_a) 
        fracture_cells.append(gb.grids_of_dimension(gb.dim_min())[0].num_cells) 
#        all_errors = np.absolute(aperture_a - aperture))/np.max(aperture_a) 
    errors = np.array(errors) 
    errors_max = np.array(errors_max) 
#    apertures = np.array(apertures) 
#    apertures_a = np.array(apertures_a) 
    plt.figure() 
    plt_info() 
    plt.title('Mean aperture error') 
 
    a = plt.loglog(fracture_cells, errors) 
 
    plt.figure() 
    plt.title('Maximum aperture error') 
    plt_info() 
    a = plt.loglog(fracture_cells, errors_max) 
 
#    plt.show() 
#    a.set_xticlabels(['2', '4', '6', '8']) 
    return errors, errors_max, apertures, apertures_a 
 
def plt_info(): 
    plt.xlabel('Fracture cells') 
plt.ylabel('E') 
 
 
def run_multiple_rm(nc, a, rms, simplex=False, beta=np.pi/2): 
    dim_h = 2 
    y_0 = 0.5 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, x_1], 
                  [y_0, y_1]]) 
    if simplex: 
        box = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1} 
        mesh_kwargs = {} 
        h = 1 / nc 
        mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': h, 
                                    'h_min': 1 / 2 * h} 
 
        gb = pp.meshing.simplex_grid([f], box, **mesh_kwargs) 
    else: 
        gb = pp.meshing.cart_grid([f], [2*nc, nc], **{'physdims': [1, 1]}) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    discr = pp.FracturedMpsa(given_traction=True) 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
 
    lhs, rhs = discr.matrix_rhs(g_h, d_h) 
    u = sps.linalg.spsolve(lhs, rhs) 
 
    # Analytical solution and error evaluation. 
    errors, sifs = evaluate_sifs_rm(a, beta, dim_h, gb, u, rms) 
 
#    plt.show() 
#    a.set_xticlabels(['2', '4', '6', '8']) 
    return errors, sifs, gb, u 
 
 
 
 
if __name__ == '__main__': 
    l = 20 
    plt.close('all') 
    a = .05 
    sigma = 1 
    beta = np.pi / 2 
    t = 0.085 
    n_cells = np.array([40, 80, 160]) 
    n_cells = np.array([20, 40]) 
#    n_cells = np.array([8, 16, 32, 64]) 
    nx = 70 
    n_cells = np.array([10, 20, 30]) 
    def cartesian_2d_of_nc(nc): 
        return cartesian_2d([2*nc, nc], a, sigma)#, {'rm': r}) 
    def simplex_2d_of_nc(nc): 
        return simplex_2d(nc, a, sigma, beta)#, {'rm': r}) 
    errors, max_errors, apertures, apertures_a = run_multiple_and_plot(n_cells, simplex_2d_of_nc) 
#    errors, max_errors, apertures, apertures_a = run_multiple_and_plot(n_cells, cartesian_2d_of_nc) 
#    errors, sifs = run_multiple_and_plot(n_cells, cartesian_2d_of_nc) 
#    errors, sifs = run_multiple_and_plot(n_cells, simplex_2d_of_nc) 
 
    # -----Cartesian rm -----------------------------------------------------# 
#    nx = 50 
#    rm = np.array([2, 4, 6, 8]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 50 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 80 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # ----- Simplex rm ------------------------------------------------------# 
#    nx = 50 
#    rm = np.array([2, 4, 6, 8]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 80 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
#    nx = 80 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
#    nx = 120 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_2d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # -----Cartesian nc -----------------------------------------------------# 
 
#    nc = np.array([20, 40, 80, 160]) 
#    errors, sifs = run_multiple_and_plot(nc, cartesian_2d_of_nc) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # ----- Simplex nc ------------------------------------------------------# 
# 
 
#    u_analytical, dof_ind = analytical_displacements(gb, a) 
#    e = np.absolute(u_analytical - u[dof_ind]) 
#    nx = 6 
 
##    _, s, s_n = displacement_correlation(gb, u, critical_sifs) 
#    errors, sifs, gb, u =run_multiple_rm_3d(nx, a, rm, t, simplex=True, beta=np.pi/2) 
#    gb, u, u_a, e, dof_ind = simplex_3d(nx, a, sigma,beta, t) 
 
#    print(u[dof_ind],'\n ') 
#    print(e) 
#    aperture = -np.diff(u[dof_ind], axis=0) 
#    aperture_a = np.sum(u_analytical, axis=0) 
#    e = np.absolute(aperture_a - aperture) 
    def cartesian_3d_of_nc(nc): 
        return cartesian_3d([nc, 3, nc], a, sigma, t)#, {'rm': r}) 
    def simplex_3d_of_nc(nc): 
        return simplex_3d(nc, a, sigma, beta, t)#, {'rm': r}) 
    n_cells = np.array([10, 20, 30]) 
#    errors, max_errors, apertures, apertures_a = run_multiple_and_plot(n_cells, simplex_3d_of_nc) 
 
#    nx = np.array([6, 10]) 
#    run_multiple_and_plot(nx, simplex_3d_of_nc) 
#    errors, sifs = run_multiple_and_plot(n_cells, cartesian_3d_of_nc) 
#    errors, sifs = run_multiple_and_plot(n_cells, simplex_3d_of_nc) 
 
    # -----Cartesian rm -----------------------------------------------------# 
#    nx = 30 
#    rm = np.array([2, 4, 6, 8]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 50 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 80 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, cartesian_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # ----- Simplex rm ------------------------------------------------------# 
#    nx = 50 
#    rm = np.array([2, 4, 6, 8]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 50 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
# 
#    nx = 80 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
#    nx = 120 
#    rm = np.array([1, 2, 3]) - .5 
#    errors, sifs = run_multiple_and_plot(rm, simplex_3d_of_rm) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # -----Cartesian nc -----------------------------------------------------# 
 
#    nc = np.array([20, 40, 80, 160]) 
#    errors, sifs = run_multiple_and_plot(nc, cartesian_3d_of_nc) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    # ----- Simplex nc ------------------------------------------------------# 
 
#    nc = np.array([20, 40, 80, 120]) 
#    errors, sifs = run_multiple_and_plot(nc, simplex_3d_of_nc) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
