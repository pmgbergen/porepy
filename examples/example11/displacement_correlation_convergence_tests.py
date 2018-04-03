#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In these convergence tests, we set up a displacement field, evaluate stress 
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
test_propagate.displacement_correlation_sif_computation.py. There, the tests are run for 
only a single (rather coarse) mesh. 
""" 
 
import scipy.sparse as sps 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
 
from porepy.numerics.fv.mpsa import FracturedMpsa 
from porepy.fracs import propagate_fracture as propagate 
from porepy.fracs import meshing 
from porepy.params.data import Parameters 
from porepy.utils import comp_geom as cg 
 
from porepy.viz.plot_grid import plot_grid 
from test.integration.setup_mixed_dimensional_grids import set_bc_mech_tension 
from examples.example11.fracture_opening_convergence_tests_sneddon import save 
 
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
    gb = meshing.cart_grid([f], n_cells, **{'physdims': [1, 1]}) 
    assign_parameters(gb) 
    print('\n\nlower-dim cartesian cells ', gb.grids_of_dimension(1)[0].num_cells) 
    print('Higher-dim cartesian cells ', gb.grids_of_dimension(2)[0].num_cells) 
    # Discretize, solve and evaluate sifs. 
    sifs = solve(gb, dim_h, kw) 
 
    # Analytical solution and error evaluation. 
    beta = np.pi / 2 
    K = analytical_sifs(a, beta, dim_h) 
    print(K - sifs.T) 
    e = np.mean(np.absolute(K - sifs.T)/K, axis=0) 
    print(e) 
    return sifs, e, gb 
 
 
# -----simplex_2d-------------------------------------------------------------# 
 
def simplex_2d(mesh_size, a, sigma, beta, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 2 
    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = a * np.sin(beta) 
    f = np.array([[0, x_1], 
                  [0.5, y_1]]) 
    box = {'xmin': 0, 'ymin': 0, 'xmax': .5, 'ymax': 1} 
    y_0 = 0.5 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, x_1], 
                  [y_0, y_1]]) 
    box = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1} 
    mesh_kwargs = {} 
    h = 1 / mesh_size 
    h_ideal = np.array([h, h, h, h, h * a * 2]) 
    mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': h_ideal, 
                                'h_min': 1 / 200 * h} 
    print(h_ideal) 
    gb = meshing.simplex_grid([f], box, **mesh_kwargs) 
#    plot_grid(gb) 
#    plt.figure() 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    sifs = solve(gb, dim_h, kw) 
 
    # Analytical solution and error evaluation. 
    K = analytical_sifs(a, beta, dim_h) 
    e = np.mean(np.absolute(K - sifs.T)/K, axis=0)#e = np.absolute(K - np.mean(sifs, axis=1))/K 
    return sifs, e, gb 
 
 
# -----cartesian_3d-----------------------------------------------------------# 
 
def cartesian_3d(n_cells, a, sigma, t, kw={}): 
    # Make grid bucket and assign data 
    dim_h = 3 
    y_0 = 0.5 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, a, a, 0], 
                  [0, 0, 2 * t, 2 * t], 
                  [0.5, 0.5, 0.5, 0.5]]) 
    gb = meshing.cart_grid([f], n_cells, **{'physdims': [1, 2 * t, 1]}) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    sifs = solve(gb, dim_h, kw) 
 
    # Analytical solution and error evaluation. 
    beta = np.pi / 2 
    K = analytical_sifs(a, beta, dim_h) 
    e = np.mean(np.absolute(K - sifs.T)/K, axis=0) 
    return sifs, e 
 
 
# -----simplex_3d-------------------------------------------------------------# 
 
def simplex_3d(mesh_size, a, sigma, beta, t, kw={}): 
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
    mesh_kwargs = {'h_ideal': h, 'h_min': 1 / 2 * h} 
 
    gb = meshing.simplex_grid([f], box, **mesh_kwargs) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    sifs = solve(gb, dim_h, kw) 
 
    # Analytical solution and error evaluation. 
    K = analytical_sifs(a, beta, dim_h) 
    e = np.absolute(K - np.mean(sifs, axis=1))/K 
    return sifs, e 
 
 
# -----analytical-------------------------------------------------------------# 
 
def analytical_sifs(a, beta, dim): 
    """ 
    Analytical stress intensity factors for the through-the-thickness crack 
    problem in question. 
    """ 
    K_I = sigma * np.sqrt(np.pi * a) * np.power(np.sin(beta), 2) 
    K_II = sigma * np.sqrt(np.pi * a) * np.cos(beta) * np.sin(beta) 
    K = np.array([K_I,  K_II]) 
    if dim == 3: 
        K = np.append(K, 1e-10) 
    return K 
 
 
# -----utility functions------------------------------------------------------# 
 
def assign_parameters(gb): 
    """ 
    Utility function to assign the parameters to the node data of the grid 
    bucket. 
    """ 
    gb.add_node_props(['param']) 
    for g, d in gb: 
        param = Parameters(g) 
        d['param'] = param 
        d['max_memory'] = 5e9 / 5 
    set_bc_mech_tension(gb, t=t, top_tension=sigma) 
 
 
def solve(gb, dim_h, kw={}): 
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
    critical_sifs = [1, .1, .1]     # Arbitrary, unused critical values 
    _, sifs, sifs_normal = propagate.displacement_correlation(gb, u, critical_sifs, **kw) 
    return sifs 
 
 
def run_multiple_and_plot(array, function): 
    errors = [] 
    sifs = [] 
    fracture_cells = [] 
    for item in array: 
        s, e, gb = function(item) 
        errors.append(e) 
        sifs.append(s) 
        fracture_cells.append(gb.grids_of_dimension(gb.dim_min())[0].num_cells) 
    errors = np.array(errors) 
    sifs= np.array(sifs) 
    plt.figure() 
    plt.loglog(fracture_cells, errors[:, 0]) 
    plt.xlabel('Fracture cells') 
    plt.ylabel('E') 
    v = errors[1, 0] *.9 
    plt.loglog(fracture_cells[:2], [v, v/2]) 
    plt.loglog(fracture_cells[:2], [v, v/4]) 
    plt.legend(['All points', 'First order', 'Second order']) 
#    plt.show() 
#    a.set_xticlabels(['2', '4', '6', '8']) 
    return errors, sifs 
 
 
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
        h_ideal = np.array([h, h, h, h, h * k_frac]) 
        h_ideal = np.array([h, h, h, h, h_frac]) 
        mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': h_ideal, 
                                    'h_min': 1 / 20000 * h} 
 
        gb = meshing.simplex_grid([f], box, **mesh_kwargs) 
    else: 
        gb = meshing.cart_grid([f], [2*nc, nc], **{'physdims': [1, 1]}) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    discr = FracturedMpsa(given_traction=True) 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
 
    lhs, rhs = discr.matrix_rhs(g_h, d_h) 
    u = sps.linalg.spsolve(lhs, rhs) 
 
    # Analytical solution and error evaluation. 
    errors, _, sifs, _ = evaluate_sifs_rm(a, beta, dim_h, gb, u, rms) 
 
 
#    plt.show() 
#    a.set_xticlabels(['2', '4', '6', '8']) 
    return errors, sifs, gb, u 
 
 
def run_multiple_rm_3d(nc, a, rms, t, simplex=False, beta=np.pi/2): 
    dim_h = 3 
    y_0 = 0.5 - a * np.cos(beta) 
    x_0 = 0.5 - a * np.sin(beta) 
 
    y_1 = 0.5 + a * np.cos(beta) 
    x_1 = 0.5 + a * np.sin(beta) 
    f = np.array([[x_0, x_1, x_1, x_0], 
                  [0, 0, 2 * t, 2 * t], 
                  [0.5, 0.5, 0.5, 0.5]]) 
 
    if simplex: 
        box = {'xmin': 0, 'ymin': 0, 'zmin': 0, 'xmax': 1, 'ymax': 2 * t, 'zmax': 1} 
        mesh_kwargs = {} 
        h = 1 / nc 
        mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': h, 
                                    'h_min': 1 / 200 * h} 
 
        gb = meshing.simplex_grid([f], box, **mesh_kwargs) 
    else: 
        gb = meshing.cart_grid([f], [2*nc, nc/5, nc], **{'physdims': [1, 2 * t, 1]}) 
    assign_parameters(gb) 
 
    # Discretize, solve and evaluate sifs. 
    discr = FracturedMpsa(given_traction=True) 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
    print('number of cells h', g_h.num_cells) 
    lhs, rhs = discr.matrix_rhs(g_h, d_h) 
    u = sps.linalg.spsolve(lhs, rhs) 
 
    # Analytical solution and error evaluation. 
#    errors, _, sifs , _= evaluate_sifs_rm(a, beta, dim_h, gb, u, rms) 
    errors, sifs, rms, signed = evaluate_sifs_rm_3d(a, beta, dim_h, gb, u) 
#    plt.show() 
#    a.set_xticlabels(['2', '4', '6', '8']) 
    return errors, sifs, gb, u, rms, signed 
 
def evaluate_sifs_rm(a, beta, dim_h, gb, u, rms): 
    K = analytical_sifs(a, beta, dim_h)[:, np.newaxis] 
    errors = [] 
    errors_n = [] 
    sifs = [] 
    sifs_n = [] 
    fracture_cells = [] 
    g_l = gb.grids_of_dimension(dim_h-1)[0] 
    if g_l.dim < 2: 
        rms= np.unique(np.round(g_l.cell_centers[0]-.5+np.sin(beta) * a, 10))/np.sin(beta) 
 
        rms = rms[rms<a] 
        rms = np.sort(rms) 
    else: 
        rms = rms * g_l.face_areas[0] 
    nc = np.min([rms.size, 5]) 
    for rc in rms: 
        critical_sifs = [1, .1, .1]     # Arbitrary, unused critical values 
        rmin = rc#*gb.grids_of_dimension(dim_h-1)[0].cell_volumes[0] 
        kw = {'rm': rmin} 
#        print('in', rmin) 
        _, s, s_n = propagate.displacement_correlation(gb, u, critical_sifs, **kw) 
        print('rm', rc, 'signed', ((K - s)/K)[0]) 
#        print(s) 
        e = np.mean(np.absolute(K - s)/K, axis=1) 
        errors.append(e) 
        e_n = np.mean(np.absolute(K - s_n)/K, axis=1) 
        errors_n.append(e_n[0]) 
        sifs.append(s) 
        sifs_n.append(s_n) 
        fracture_cells.append(gb.grids_of_dimension(gb.dim_min())[0].num_cells) 
    errors = np.array(errors) 
    errors_n = np.array(errors_n) 
    sifs= np.array(sifs) 
#    plt.figure() 
    save(gb, u, np.zeros(g_l.num_cells), file_name, folder_name, export) 
 
    ind = np.argmin(errors[:, 0])#, axis=0)[0] 
    if np.isclose(beta, np.pi / 2): 
        xaxis = np.arange(nc) + .5 
        xaxis = rms[:nc]/g_l.cell_volumes[0] 
        plt.semilogy(xaxis, errors[:nc, 0]) 
    else: 
        plt.semilogy(rms/a, errors) 
#    plt.semilogy(rms, errors_n[:, 0]) 
    if np.isclose(beta, np.pi / 2): 
        ma = np.max(errors[:, 0])#[:, 0]) 
    mi = np.min(errors)#[:, 0]) 
    ra = ma - mi 
    plt.xlabel('$r_m$ / $L_n$') 
    plt.ylabel('$e_t$') 
 
#    v = errors[1] *.9 
#    plt.loglog(fracture_cells[:2], [v, v/2]) 
#    plt.loglog(fracture_cells[:2], [v, v/4]) 
#    plt.legend(['All points', 'First order', 'Second order']) 
#    plt.text(.02, mi + ra*0.3, 'Minimal index: ' + str(ind+1) + ' yields error of ' 
#             + "{:.5f}".format(errors[ind, 0])) 
#    plt.text(.02, mi + ra*0.2, 'Number of fracture cells: ' + str(g_l.num_cells)) 
#    rmmin = rms[ind] 
#    if g_l.dim == 2: 
#        cellno = rmmin/g_l.face_areas[0] 
#    else: 
#        cellno = rmmin/g_l.cell_volumes[0] 
# 
#    plt.text(.02, mi + ra*0.1, 'Mode I Rm: ' + "{:.5f}".format(rmmin)  + 
#             ' corresponds to ' +"{:.5f}".format(cellno)+ ' times h and $\eta$/c = ' 
#             + "{:.5f}".format(rmmin/a)) 
#    print('\n\nOptimal rm ', rmmin / a) 
    if np.isclose(beta, np.pi / 2): 
        return errors, errors_n, sifs, sifs_n 
    ind = np.argmin(errors[:, 1]) 
    rmmin = rms[ind] 
    if g_l.dim == 2: 
        cellno = rmmin/g_l.face_areas[0] 
    else: 
        cellno = rmmin/g_l.cell_volumes[0] 
 
    plt.text(.02, mi + ra*0.05, 'Mode II Rm: ' + "{:.5f}".format(rmmin)  + 
             ' corresponds to ' +"{:.4f}".format(cellno) + ' times h and r/a = ' 
             + "{:.4f}".format(.5 * rmmin/a)) 
    return errors, errors_n, sifs, sifs_n 
 
def evaluate_sifs_rm_3d(a, beta, dim_h, gb, u): 
    K = analytical_sifs(a, beta, dim_h)[:, np.newaxis] 
    g_l = gb.grids_of_dimension(dim_h-1)[0] 
    g_h = gb.grids_of_dimension(dim_h)[0] 
    d_h = gb.node_props(g_h) 
 
    f_c = gb.edge_prop((g_h, g_l), 'face_cells')[0] 
    cells_l = np.arange(g_l.num_cells) 
    mock_face_l = g_l.tags['tip_faces'].nonzero()[0][0] 
    faces_l = np.tile(mock_face_l, cells_l.size) 
    start = np.array([.5 - a, 0, .5]) 
    end = np.array([.5 - a, 2 * t, .5]) 
    rms, _ = cg.dist_points_segments(g_l.cell_centers, start, end) 
    d_u = propagate.relative_displacements(u, f_c, g_l, cells_l, faces_l, g_h) 
 
    E = d_h.get('Young') 
    poisson = d_h.get('Poisson') 
    mu = E/(2*(1 + poisson)) 
    kappa = 3 - 4 * poisson 
 
    s, s_n \ 
        = propagate.sif_from_delta_u(d_u, rms, rms, mu, kappa) 
 
    nc = np.min([rms.size, 5]) 
    nc = g_l.num_cells 
    errors = (np.absolute(K - s)/K)[0] 
    plt.figure() 
    save(gb, u, errors, file_name, folder_name, export, sort_error=False) 
#    print('e', errors) 
    signed = ((K - s)/K)[0] 
#    print('signed', signed) 
#    print('rm', rms.T) 
#    print('cc', g_l.cell_centers) 
#    print('cv', g_l.cell_volumes) 
#    ind = np.argmin(errors[:, 0])#, axis=0)[0] 
    if np.isclose(beta, np.pi / 2): 
        xaxis = rms / a 
        aa = plt.gca() 
        aa.set_yscale('log') 
        aa.scatter(xaxis, errors) 
        aa.set_ylabel('$e_t$') 
        aa.set_xlabel('$r_m$ / $a$') 
        plt.scatter(xaxis, errors[:nc]) 
    else: 
        plt.semilogy(rms/a, errors) 
        plt.xlabel('$r_m$ / $L_n$') 
        plt.ylabel('$e_t$') 
    plot_vs_normalized(g_l, errors, rms[:, 0], 5) 
    return errors, s, rms, signed 
 
def plot_vs_normalized(g_l, errors, rm, cut_off=100, g_h=None, face_cells=None): 
    sort_ind = np.argsort(rm) 
    errors = errors[sort_ind] 
    rm = rm[sort_ind] 
    if face_cells is None: 
        h = np.mean(g_l.face_areas[g_l.tags['tip_faces']]) 
    else: 
        faces = face_cells[g_l.tags['tip_faces']].nonzero()[1] 
        cells = g_h.cell_faces[faces].nonzero()[1] 
        h4 = [] 
        cn = g_h.cell_nodes() 
        for i in range(len(faces)): 
#            h4.append(cg.dist_point_pointset(g_h.cell_centers[:, cells[i]], 
#                                             g_h.face_centers[:, faces[i]])) 
            nodes = cn[:, cells[i]].nonzero()[0] 
            h4.append(np.max(cg.dist_point_pointset(g_h.nodes[:, nodes[0]], 
                                             g_h.nodes[:, nodes[0:]]))) 
#        h5 = np.array(h4)[:, 0] 
        h = np.mean(h4) 
#        h = np.mean(h5) * 4 
    xaxis = rm / h 
    ind = np.sum(xaxis < cut_off) 
    plt.figure() 
    aa = plt.gca() 
    aa.set_yscale('log') 
    aa.scatter(xaxis[:ind], errors[:ind]) 
    aa.set_ylabel('$e_t$') 
    aa.set_xlabel('$r_m$ / $L_n$') 
#    plt.scatter(xaxis, errors) 
 
 
def pickle_dump(fn): 
    d = {} 
    d['errors'] = errors 
    d['gb'] = gb 
    d['sifs'] = sifs 
    d['rm'] = rm 
    d['rms'] = rms 
    d['signed'] = signed 
    pickle.dump( d, open( fn, "wb" ) ) 
 
 
if __name__ == '__main__': 
#    plt.clf() 
    export = 1 == 1 
    file_name = 'displacements' 
    folder_name = 'displacement_correlation/simplex_2d' 
    a = .05 
    sigma = 1e-3 
    beta = np.pi / 2 
    t = 0.1 
#    n_cells = np.array([40, 80, 160]) 
    n_cells = np.array([20, 40, 80]) 
    nx = 100 
    rm = np.array([35, 40, 45]) - .5 
    rm = np.array([50, 60, 70]) 
    rm = np.array([10, 12, 15]) 
    rm = np.array([10, 11, 12]) - .5 
#    rm = np.array([16, 17, 18]) - .5 
#    rm = np.array([16]) - .5 
    rm = np.array([2, 3, 4, 5]) - .5 
#    nx = 140 
#    rm = np.array([6, 7]) - .5 
    rm = np.array([2]) - .5 
    r = .1/10 
    def cartesian_2d_of_nc(nc): 
        return cartesian_2d([2*nc, nc], a, sigma)#, {'rm': r}) 
    def simplex_2d_of_nc(nc): 
        return simplex_2d(nc, a, sigma, beta)#, {'rm': r}) 
    def cartesian_2d_of_rm(r): 
        l_c = 1/(2*nx) 
        r *= l_c 
        return cartesian_2d([nx*2, nx], a, sigma, {'rm': r}) 
    # eventuelt neste linje: 
#    beta = np.pi / 4 
    def simplex_2d_of_rm(r): 
        l_c = 1/(5*nx) 
        r *= l_c 
        return simplex_2d(nx, a, sigma, beta, {'rm': r}) 
#    n_cells = np.array([20, 40, 80, 160]) 
#    errors, sifs = run_multiple_and_plot(n_cells, cartesian_2d_of_nc) 
#        n_cells = np.array([20, 40, 80]) 
#    errors, sifs = run_multiple_and_plot(n_cells, simplex_2d_of_nc) 
 
    # -----Cartesian rm -----------------------------------------------------# 
    kk = 3 
    nx = 80#90 / kk#30 #3, 4, 5,45 
    k_frac = 3 / kk * a 
#    h_frac = a / #15 
    rm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]) - .5 
#    def simplex_2d_of_rm(r): 
#        l_c = 1/(15*nx) 
#        r *= l_c 
#        return simplex_2d(nx, a, sigma, beta, {'rm': r}) 
 
 
#    nfrac = [8, 12, 16, 20, 31, 40] 
#    for i in nfrac: 
#        h_frac = a / i 
#        errors, sifs, gb, u =run_multiple_rm(nx, a, rm, simplex=True, beta=beta) 
#    plt.legend(['a/$L_n = $' + str(i) for i in nfrac]) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
    n_frac = 15 
    h_frac = 2* a / n_frac 
    nxs = np.array([4, 8, 10, 14, 20, 24, 30, 40, 60, 90, 120]) + 1 
    nxs = np.array([90, 120, 150, 200, 250]) / 5 
    nxs = np.array([30, 50, 70, 90]) 
    nxs = [50] 
#    nxs = [8] 
#    h_frac = a / 20 
#    nxs = [30, 40, 60] 
#    nxs = [121] 
#    nxs = np.arange(25, 33, 1) 
 
    for nx in nxs: 
        h_frac = 1 / (30*nx) 
        print('\nnx', nx) 
        errors, sifs, gb, u = run_multiple_rm(nx, a, rm, simplex=True, beta=beta) 
    plt.legend(['nx = ' + str(i) for i in nxs]) 
#    errors, sifs = evaluate_sifs_rm(a, beta, 2, gb, u, rm) 
#    errors, sifs, gb, u =run_multiple_rm(nx, a, rm, simplex=False, beta=np.pi/4) 
#    print(sifs[:,0],'\n E:', errors[:,0]) 
 
 
    #  ---------------- 3 D 
#    nxs = [10, 15, 20] 
#    nxs = [6] 
#    rm = np.array([.5, 1, 1.5, 2, 2.5]) 
#    for nx in nxs: 
#        h_frac = 1 / (2*nx) 
#        print('\nnx', nx) 
#        errors, sifs, gb, u, rms, signed = run_multiple_rm_3d(nx, a, rm, t, 
#                                                              simplex=True, 
#                                                              beta=beta) 
#    plt.legend(['nx = ' + str(i) for i in nxs]) 
 
 
 
##    _, s, s_n = propagate.displacement_correlation(gb, u, critical_sifs) 
#    errors, sifs, gb, u =run_multiple_rm_3d(nx, a, rm, t, simplex=True, beta=np.pi/2) 
#    e, e_n, sifs, sifs_n = evaluate_sifs_rm(a, beta, 3, gb, u, rm) 
#    print(sifs[0, 0],'\n ') 
#    print(sifs_n) 
 
 
 
    def cartesian_3d_of_nc(nc): 
        return cartesian_3d([nc, 3, nc], a, sigma, t)#, {'rm': r}) 
    def simplex_3d_of_nc(nc): 
        return simplex_3d(nc, a, sigma, beta, t)#, {'rm': r}) 
    def cartesian_3d_of_rm(r): 
        l_c = 1/(2*nx) 
        r *= l_c 
        return cartesian_3d([nx*2, nx], a, sigma, t, {'rm': r}) 
    def simplex_3d_of_rm(r): 
        l_c = 1/(1.5*nx) 
        r *= l_c 
        return simplex_3d(nx, a, sigma, beta, t, {'rm': r}) 
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
    nx = 50 
    rm = np.array([1, 2, 3, 4, 5]) - .5 
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
