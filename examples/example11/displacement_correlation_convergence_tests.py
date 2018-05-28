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

import porepy as pp

from test.integration.setup_mixed_dimensional_grids import \
    set_bc_mech_tension, set_bc_mech_tension_sneddon
from examples.example11.fracture_opening_convergence_tests_sneddon import save
from examples.example11.fracture_opening_convergence_tests_sneddon import \
    simplex_2d as simplex_2d_gb, simplex_3d as simplex_3d_gb


# -----simplex_2d-------------------------------------------------------------#

def simplex_2d(nx, a, sigma, beta, kw={}):
    # Make grid bucket and assign data
    dim_h = 2
    h = 1 / nx

    gb = simplex_2d_gb(h, length, height, a, beta, folder_name, from_gmsh,
                       fracture=True)
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, kw)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, beta, dim_h)
    e = np.mean(np.absolute(K - sifs.T)/K, axis=0)
    return sifs, e, gb


# -----simplex_3d-------------------------------------------------------------#

def simplex_3d(nx, a, beta, t, kw={}):
    # Make grid bucket and assign data
    dim_h = 3
    h = 1 / nx
    gb = simplex_3d_gb(h, length, height, a, t, beta, folder_name,
                       from_gmsh=False, fracture=True, penny=False, kw={})
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    sifs = solve(gb, dim_h, kw)

    # Analytical solution and error evaluation.
    K = analytical_sifs(a, beta, dim_h)
    e = np.absolute(K - np.mean(sifs, axis=1))/K
    return sifs, e


# -----analytical-------------------------------------------------------------#

def analytical_sifs(a, beta, dim, sigma):
    """
    Analytical stress intensity factors for the through-the-thickness crack
    problem in question.
    """
    K_I = sigma * np.sqrt(np.pi * a) * np.power(np.sin(beta), 2)
    K_II = sigma * np.sqrt(np.pi * a) * np.cos(beta) * np.sin(beta)
    K = np.array([K_I, K_II])
    if dim == 3:
        K = np.append(K, 1e-10)
    if penny:
        K *= 2 / np.pi
    return K


# -----utility functions------------------------------------------------------#

def assign_parameters(gb):
    """
    Utility function to assign the parameters to the node data of the grid
    bucket.
    """
    gb.add_node_props(['param'])
    for g, d in gb:
        param = pp.Parameters(g)
        d['param'] = param
        d['max_memory'] = 5e9 / 5

    if sneddon:
        set_bc_mech_tension_sneddon(gb, p0=sigma, height=height, length=length,
                                    beta=beta, aperture=a, t=t, penny=penny,
                                    fix_faces=False)
    else:
        set_bc_mech_tension(gb, t=t, top_tension=sigma)


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

    # SIF evaluation by displacemnt correlation
    critical_sifs = [1, .1, .1]     # Arbitrary, unused critical values
    _, sifs = pp.displacement_correlation.faces_to_open(gb, u, critical_sifs,
                                                        **kw)
    return sifs[0]


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
    sifs = np.array(sifs)
    plt.figure()
    plt.loglog(fracture_cells, errors[:, 0])
    plt.xlabel('Fracture cells')
    plt.ylabel('E')
    v = errors[1, 0] * .9
    plt.loglog(fracture_cells[:2], [v, v/2])
    plt.loglog(fracture_cells[:2], [v, v/4])
    plt.legend(['All points', 'First order', 'Second order'])
#    plt.show()
#    a.set_xticlabels(['2', '4', '6', '8'])
    return errors, sifs


def run_multiple_rm(nc, a, simplex=False, beta=np.pi/2):
    dim_h = 2
    if simplex:
        h = 1 / nc
        gb = simplex_2d_gb(h, length, height, a, beta, folder_name, from_gmsh,
                           fracture=True)
    else:
        raise NotImplementedError
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    discr = pp.FracturedMpsa(given_traction=True)
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)

    lhs, rhs = discr.matrix_rhs(g_h, d_h)
    u = sps.linalg.spsolve(lhs, rhs)

    # Analytical solution and error evaluation.
    errors, sifs = evaluate_sifs_rm(a, beta, dim_h, gb, u)

#    plt.show()
#    a.set_xticlabels(['2', '4', '6', '8'])
    return errors, sifs, gb, u


def run_multiple_rm_3d(nc, a, t, simplex=False, beta=np.pi/2):
    dim_h = 3

    if simplex:
        h = 1 / nc
        gb = simplex_3d_gb(h, length, height, a, t, beta, file_name='simplex_3d',
                           from_gmsh=from_gmsh, fracture=True, penny=penny,
                           n_points=n_fracture_points)
    else:
        print('not implemented')
    assign_parameters(gb)

    # Discretize, solve and evaluate sifs.
    discr = pp.FracturedMpsa(given_traction=True)
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)
    print('number of cells h', g_h.num_cells, ' and l ',
          gb.grids_of_dimension(dim_h - 1)[0].num_cells)
    lhs, rhs = discr.matrix_rhs(g_h, d_h)
    u = sps.linalg.spsolve(lhs, rhs)

    # Analytical solution and error evaluation.
#    errors, sifs = evaluate_sifs_rm(a, beta, dim_h, gb, u, rms)
    errors, sifs, rms, signed = evaluate_sifs_rm_3d(a, beta, gb, u)
#    plt.show()
#    a.set_xticlabels(['2', '4', '6', '8'])
    return errors, sifs, gb, u, rms, signed


def evaluate_sifs_rm(a, beta, dim_h, gb, u, p0):
    K = analytical_sifs(a, beta, dim_h, sigma=p0)[:, np.newaxis]
    errors = []
    sifs = []
    fracture_cells = []
    g_l = gb.grids_of_dimension(dim_h-1)[0]
    if g_l.dim < 2:
        rms = np.unique(np.round(g_l.cell_centers[0]-.5+np.sin(beta) * a, 9)) \
                        / np.sin(beta)

        rms = rms[rms < a]
        rms = np.sort(rms)
    else:
        rms = rms * g_l.face_areas[0]
    nc = np.min([rms.size, 10])
    for rc in rms:
        critical_sifs = [1, .1, .1]     # Arbitrary, unused critical values
        rmin = rc  # *gb.grids_of_dimension(dim_h-1)[0].cell_volumes[0]
        kw = {'rm': rmin}
        _, s = pp.displacement_correlation.faces_to_open(gb, u, critical_sifs,
                                                         **kw)
        s = s[0]
        e = np.mean(np.absolute(K - s)/K, axis=1)
        errors.append(e)
        sifs.append(s)
        fracture_cells.append(gb.grids_of_dimension(gb.dim_min())[0].num_cells)
    errors = np.array(errors)
    print('\nrm', rms[0], 'signed', ((K - sifs[0])/K)[0])
    print('e', errors[0, 0])
    sifs = np.array(sifs)
    save(gb, u, np.zeros(g_l.num_cells), file_name, folder_name, export)

    ind = np.argmin(errors[:, 0])  # , axis=0)[0]
#    if np.isclose(beta, np.pi / 2):
#        xaxis = np.arange(nc) + .5
#        xaxis = rms[:nc]/g_l.cell_volumes[0]
##        plt.semilogy(xaxis, errors[:nc, 0])
#        plt.scatter(xaxis, np.log(errors[:nc, 0]))
#    else:
#        plt.semilogy(rms/a, errors)
#    plt.semilogy(rms, errors_n[:, 0])
    if np.isclose(beta, np.pi / 2):
        ma = np.max(errors[:, 0])  # [:, 0])
    mi = np.min(errors)
    ra = ma - mi
    plt.xlabel('$r_m$ / $L_n$')
    plt.ylabel('$e_t$')

    if np.isclose(beta, np.pi / 2):
        return errors, sifs
    ind = np.argmin(errors[:, 1])
    rmmin = rms[ind]
    if g_l.dim == 2:
        cellno = rmmin/g_l.face_areas[0]
    else:
        cellno = rmmin/g_l.cell_volumes[0]

    plt.text(.02, mi + ra * 0.05, 'Mode II Rm: ' + "{:.5f}".format(rmmin) +
             ' corresponds to ' + "{:.4f}".format(cellno) +
             ' times h and r/a = ' + "{:.4f}".format(.5 * rmmin/a))
    return errors, sifs


def evaluate_sifs_rm_3d(a, beta, gb, u, sigma, move_tip=False):
    dim_h = 3
    K = analytical_sifs(a, beta, dim_h, sigma)[:, np.newaxis]
    g_l = gb.grids_of_dimension(dim_h-1)[0]
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)

    f_c = gb.edge_props((g_h, g_l), 'face_cells')
    cells_l = np.arange(g_l.num_cells)
    tip_face_l = g_l.tags['tip_faces'].nonzero()[0][0]
    faces_l = np.tile(tip_face_l, cells_l.size)
#    start = np.array([.5 - a, 0, .5])
#    end = np.array([.5 - a, 2 * t, .5])
#    rms, _ = pp.cg.dist_points_segments(g_l.cell_centers, start, end)
    tip_face_coo = g_l.face_centers[:, tip_face_l]
    if move_tip:
        center = np.array([length / 2, t, height / 2])
        radius_vector = tip_face_coo - center
        tip_face_coo = center + radius_vector / np.cos(np.pi / n_fracture_points)
    rms = pp.cg.dist_point_pointset(tip_face_coo,
                                    g_l.cell_centers)
    d_u = pp.displacement_correlation.relative_displacements(u, f_c, g_l,
                                                             cells_l, faces_l,
                                                             g_h)

    E = d_h.get('Young')
    poisson = d_h.get('Poisson')
    mu = E / (2 * (1 + poisson))
    kappa = 3 - 4 * poisson

    s = pp.displacement_correlation.sif_from_delta_u(d_u, rms, mu, kappa)

    nc = np.min([rms.size, 5])
    nc = g_l.num_cells
    errors = (np.absolute(K - s) / K)[0]
    plt.figure()
    save(gb, u, errors, file_name, folder_name, export, sort_error=False)
#    print('e', errors)
    signed = ((K - s)/K)[0]
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
#    plot_vs_normalized(g_l, errors, rms[:, 0], 5)
    return errors, s, rms, signed


def plot_vs_normalized(g_l, errors, rm, cut_off=100, g_h=None, fc=None):
    sort_ind = np.argsort(rm)
    errors = errors[sort_ind]
    rm = rm[sort_ind]
    if fc is None:
        h = np.mean(g_l.face_areas[g_l.tags['tip_faces']])
    else:
        faces = fc[g_l.tags['tip_faces']].nonzero()[1]
        cells = g_h.cell_faces[faces].nonzero()[1]
        h4 = []
        cn = g_h.cell_nodes()
        for i in range(len(faces)):
#            h4.append(cg.dist_point_pointset(g_h.cell_centers[:, cells[i]],
#                                             g_h.face_centers[:, faces[i]]))
            nodes = cn[:, cells[i]].nonzero()[0]
            hh = np.max(pp.cg.dist_point_pointset(g_h.nodes[:, nodes[0]],
                                                  g_h.nodes[:, nodes[0:]]))
            h4.append(hh)
        h = np.mean(h4)
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
    pickle.dump(d, open(fn, "wb"))


if __name__ == '__main__':
    plt.close('all')
    export = 1 == 10
    file_name = 'displacements_coarse'
    folder_name = 'displacement_correlation/penny/simplex_3d'
    length = 1
    height = 1
    sneddon = True
    a = .05
    p0 = 1e-5
    beta = np.pi / 2
    t = 0.1
    from_gmsh = False
    penny = True
    n_fracture_points = 12 * 2
    if penny:
        t = .5


#    nxs = np.array([10, 20, 40, 80, 160, 200]) - 1
#    nxs = np.array([20, 40, 80, 160, 200]) - 1
#    nxs = np.array([38, 40, 42, 58, 60, 62, 138, 140, 142]) - 1
#    nxs = np.array([38, 39, 40, 41, 42, 138, 139, 140, 141, 142])

    nxs = np.array([38, 39, 40, 41, 42, 78, 79, 80, 81, 82])#, 158, 159, 160, 161, 162])
    nxs = np.array([8])
    nxs = np.array([8, 9, 10, 11, 12, 28, 29, 30, 31, 32])
    nxs = np.array([8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 38, 39, 40, 41, 42])
    all_errors = []
    nxs = np.array([38, 39, 40, 41, 42])
    if not penny:
        for nx in nxs:
            errors, sifs, gb, u \
                = run_multiple_rm(nx, a, simplex=True, beta=beta)
            all_errors.append(errors[:, 0])

        all_errors = np.array(all_errors)
        nc = 15
        fig = plt.figure()
        ax = plt.gca()
        bin_size = 5
        colors = ['r', 'g', 'b']

        for i in range(nxs.size // bin_size):
            er = all_errors[(bin_size*i):(bin_size*(i+1))]
            ind1 = np.inf
            for e in er:
                ind1 = np.minimum(ind1, e.size)
            ind1 = int(ind1)
            for j, e in enumerate(er):
                e = e[:ind1]
                er[j] = e
            averages = np.mean(er)
            xaxis = np.arange(nc) + .5
            xaxis = xaxis[:averages.size]#rms[:nc]/g_l.cell_volumes[0]
            ax.plot(xaxis, averages[:nc], color=colors[i])

        nxs1 = nxs[bin_size//2:nxs.size:bin_size]
        plt.legend(['nx = ' + str(i) for i in nxs1])
        markers = ['d', 'x', '+', 'o', 'v']
        for i in range(nxs.size):
            c = colors[i // bin_size]
            m = markers[i % bin_size]
            er = all_errors[i]
            xaxis = np.arange(nc) + .5
            xaxis = np.arange(nc) + .5
            xaxis = xaxis[:er.size]
            ax.scatter(xaxis, er[:nc], color=c, marker=m)
        ax.set_yscale('log')
        ax.set_xlabel('$r_m$ / $L_n$')  # = cell number from tip - 1 / 2')
        ax.set_ylabel('$e_t$')
    nx = 6
    # ------------------ 3 D -------------------------------------------------#
    nxs = [11]
#    nx = 10
    rm = np.array([.5, 1, 1.5, 2, 2.5])
    for nx in nxs:
        h_frac = 1 / (2*nx)
    errors, sifs, gb, u, rms, signed = run_multiple_rm_3d(nx, a, t,
                                                          simplex=True,
                                                          beta=beta, sigma=p0)
#    plt.legend(['nx = ' + str(i) for i in nxs])
#
##    _, s = pp.displacement_correlation.faces_to_open(gb, u, critical_sifs)
##    e, sifs = evaluate_sifs_rm(a, beta, 3, gb, u, rm)
##    print(sifs[0, 0],'\n ')
##    print(sifs_n)
    errors, sifs, rms, signed = evaluate_sifs_rm_3d(a, beta, gb, u,
                                                    move_tip=True, sigma=p0)
#
    if rms.ndim < 2:
        rms = rms[:, np.newaxis]
    ind = np.argsort(rms[:, 0])
    close_sifs = sifs[0, ind[:10]]
    close_errors = errors[ind[:10]]
    K = analytical_sifs(a, beta, 3)
    print('\nAnalytical ', K[0])
    print(close_sifs)
    print('errors\n', close_errors)
    save(gb, u, errors, file_name, folder_name, export=True, sort_error=True,
         sifs=sifs)
#    errors, sifs = run_multiple_and_plot(rm, simplex_3d_of_rm)
#    print(sifs[:,0],'\n E:', errors[:,0])
    g_h = gb.grids_of_dimension(gb.dim_max())[0]
    g_l = gb.grids_of_dimension(gb.dim_min())[0]
#    tip_face_l = g_l.tags['tip_faces'].nonzero()[0][0]
#    print('Evaluated for face ', tip_face_l, ' at ',
#          g_l.face_centers[:, tip_face_l])
#    indicator = np.zeros(g_h.num_cells)
#    for c in range(g_h.num_cells):
#        f = g_h.cell_faces[:, c].nonzero()[0]
#        mean_face = np.max(g_h.face_areas[f])
#        ideal_volume = np.power(mean_face, 2) / 4 * np.sqrt(3)
#        indicator[c] =  g_h.cell_volumes[c] / ideal_volume
#    print('minimum face ratio', np.min(indicator))
