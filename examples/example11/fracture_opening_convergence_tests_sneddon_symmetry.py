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
test_displacement_correlation_sif_computation.py. There, the tests are run for
only a single (rather coarse) mesh.
"""

import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

import porepy as pp

from test.integration.setup_mixed_dimensional_grids import \
    set_bc_mech_tension_sneddon, set_bc_mech_tension
from displacement_correlation_convergence_tests import evaluate_sifs_rm_3d
from displacement_correlation_convergence_tests_robin import analytical_sifs

from porepy.numerics.linalg.linsolve import Factory as LSFactory



def fracture_3d(length, height, a, beta, fracture):
    z_0 = height / 2 - a * np.cos(beta)
    x_0 = 0

    z_1 = height / 2 + a * np.cos(beta)
    x_1 = a * np.sin(beta)#/ 2 + a * np.sin(beta)
    f = np.array([[x_0, x_1, x_1, x_0],
                  [0, 0, 2 * t, 2 * t],
                  [z_0, z_1, z_1, z_0]])
    return [f]

def fracture_penny(a, center, beta, t, n_points):
    """
    Penny-shaped fracture at the center of the domain.
    """
    major_axis = a
    minor_axis = a
    major_axis_angle = 0
    strike_angle = 0
    dip_angle = 0
    f = pp.EllipticFracture(center, major_axis, minor_axis,
                                  major_axis_angle, strike_angle, dip_angle,
                                  num_points=n_points)
    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    AA = PolyArea(f.p[0], f.p[1])
    print('Area ratio ', AA / (np.power(a, 2) * np.pi))
    return [f]

# -----simplex_3d-------------------------------------------------------------#
def simplex_3d(h, length, height, a, t, beta, file_name, from_gmsh=False,
               fracture=True, penny=False, **kw):
    # Make grid bucket and assign data

    box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
           'xmax': length, 'ymax': 2 * t, 'zmax': height}
    if symmetry:
        box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
           'xmax': length, 'ymax': t, 'zmax': height}
#    fn = folder_name #+ '/nc_equals_'# + str(int(h))
    mesh_kwargs = {'mesh_size_frac': h / 1,
                   'mesh_size_min': a * np.pi * 2.01 / n_fracture_points,
                   'mesh_size_bound': h, 'file_name': file_name,
                   'from_gmsh': from_gmsh}
    sd = []
    if penny:
        c = np.array([length / 2, t, height / 2])
        if symmetry:
            c = np.array([0, t, height / 2])
        n_points = kw.get('n_points', 31) #251
        f = fracture_penny(a, c, beta, t, n_points)
#        sd1 = fracture_penny(a, c1, beta, t, n_points)[0]
#        sd2 = fracture_penny(a, c2, beta, t, n_points)[0]
#        sd = [sd1, sd2]
    else:
        f = fracture_3d(length, height, a, beta, fracture)
#    if from_gmsh:
#        grids = pp.fracs.simplex.triangle_grid_from_gmsh(file_name)
#        gb = pp.meshing.grid_list_to_grid_bucket(grids)
#    else:
    gb = pp.meshing.simplex_grid(f, box, subdomains=sd, **mesh_kwargs)

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
    mu = d_h['shear_modulus']
    nu = d_h['Poisson']

    last_ind = dim_h - 1 # pull in y direction for 2d, z in 3d
    eta = compute_eta(g_h)

    cons = (1 - nu) / mu * p0 * a * 2
    if penny:
        cons *= 2/np.pi
    apertures = cons * np.sqrt(1 - np.power(eta / a, 2) )

    n_f = g_h.num_cells * dim_h
    frac_ind_1 = np.arange(n_f + last_ind, n_f + g_h.frac_pairs.shape[1] * dim_h, dim_h)
    frac_ind_2 = np.arange(last_ind + n_f + g_h.frac_pairs.shape[1] * dim_h,
                           last_ind + n_f + g_h.frac_pairs.size * dim_h,  dim_h)
    if not np.isclose(beta, np.pi / 2):
        x_ind_1 = np.arange(n_f, n_f + g_h.frac_pairs.shape[1] * dim_h, dim_h)
        x_ind_2 = np.arange(n_f + g_h.frac_pairs.shape[1] * dim_h,
                            n_f + g_h.frac_pairs.size * dim_h,  dim_h)
        frac_ind_1 = np.array([frac_ind_1, x_ind_1])
        frac_ind_2 = np.array([frac_ind_2, x_ind_2])
    frac_ind = np.array([frac_ind_1, frac_ind_2])

    return apertures, frac_ind, eta


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
    if not penny and False:
        set_bc_mech_tension(gb, top_tension=p0, t=t, l=length, h=height, fix_faces=False, symmetry=True)
    else:
        set_bc_mech_tension_sneddon(gb, p0=p0, height=height, length=length,
                                beta = beta, aperture=a, t=t / 2, penny=penny,
                                symmetry=symmetry)


def solve(gb, dim_h, kw={}):
    """
    Discretize and solve mechanical problem, and evaluate stress intensity
    factors by displacement correlation method.
    """
    discr = pp.FracturedMpsa(given_traction=True)
    g_h = gb.grids_of_dimension(dim_h)[0]
    d_h = gb.node_props(g_h)

    lhs, rhs = discr.matrix_rhs(g_h, d_h)
#    ls = LSFactory()
#    slv = ls.gmres(lhs)
#    u, info = slv(rhs, maxiter=10000, restart=1500, tol=1e-8)
    u = sps.linalg.spsolve(lhs, rhs)
    aperture_analytical, dof_ind, eta = analytical_displacements(gb, a, p0)
    if np.isclose(beta, np.pi / 2):
        aperture = np.absolute(np.diff(u[dof_ind], axis=0))[0]
    elif np.isclose(beta, np.pi / 4):

        u_right = u[dof_ind[0]]
        u_left = u[dof_ind[1]]
        d_u = u_right - u_left
        aperture = np.linalg.norm(d_u, axis=0)
    return aperture, aperture_analytical, u, eta



def compute_eta(g_h):
    if not penny:
        return g_h.face_centers[0, g_h.frac_pairs[0]]
#    else:
    return pp.cg.dist_point_pointset(g_h.face_centers[:, g_h.frac_pairs[0]],
                                      center)
def L2_norm(val, area=None):
    if area is None:
        area = np.ones(val.size) / val.size
    return np.sqrt(np.sum(np.multiply(area, np.square(val))))


def L2_error(v_ref, v_approx, area):
    enum = L2_norm(v_approx - v_ref, area)
    denom = L2_norm(v_ref, area)
    return enum / denom


def run_multiple_and_plot(nc, function):
    errors = []
    errors_I = []
    errors_max = []
    errors_el = []
    apertures = []
    apertures_a = []
    fracture_cells = []
    for item in nc:
        gb = function(item)
        g_h = gb.grids_of_dimension(gb.dim_max())[0]
        g_l = gb.grids_of_dimension(gb.dim_min())[0]
        print(g_h)
        print(g_l)
        assign_parameters(gb, t=t)
        aperture, aperture_a, u, eta = solve(gb, gb.dim_max())
        assert np.all(aperture > 0)
        areas = g_h.face_areas[g_h.frac_pairs[0]]
        e = np.absolute(aperture_a - aperture)/np.max(aperture_a)


        errors.append(L2_error(aperture_a, aperture, areas))#np.sqrt(np.sum(np.power(e, 2)) / np.sum(areas)))
        errors_I.append(L2_norm(aperture_a - aperture, areas) / (np.sum(areas) * np.max(aperture_a)))
        errors_max.append(np.max(e))
        i =  eta<(.9*a)
        errors_el.append(L2_error(aperture_a[i], aperture[i], areas[i]))
        apertures.append(aperture)
        apertures_a.append(aperture_a)
        fracture_cells.append(gb.grids_of_dimension(gb.dim_min())[0].num_cells)
        if plot:
            plot_against_eta(e, eta)
            print('errors', errors)
    errors = np.array(errors)
    errors_el = np.array(errors_el)
    errors_max = np.array(errors_max)
    errors_I = np.array(errors_I)
    if plot:
        plt.figure()
        plt.xlabel('Fracture cells')
        plt.ylabel('E')
        plt.title('Mean aperture error')

        plt.loglog(fracture_cells, errors)
        plt.loglog(fracture_cells, errors_max)
        plt.loglog(fracture_cells, errors_el)
        plt.loglog(fracture_cells, errors_I)
        v = errors_el[0] * .4
        i = 0
        nc1 = fracture_cells[i]
        nc2 = fracture_cells[i + 1]
        f = nc2 / nc1
        x = np.array([nc1, nc2, nc1, nc1])
        plt.loglog(x, [v, v/f, v/f, v], ls=':')
        v = errors_el[1] * .4
        i = 1
        nc1 = fracture_cells[i]
        nc2 = fracture_cells[i + 1]
        f = np.power(nc2 / nc1, 2)
        x = np.array([fracture_cells[i], fracture_cells[i+1], fracture_cells[i], fracture_cells[i]])
        plt.loglog(x, [v, v/f, v/f, v], ls=':')
        plt.legend(['L2', 'Max', 'El', 'I', 'First order', 'Second order'])


    save(gb, u, e, file_name, folder_name, export=export)
    return errors, errors_max, apertures, apertures_a, u, gb, e, eta


def evaluate_sifs_rm_3d(a, beta, gb, u, sigma, move_tip=False):
    dim_h = 3
    phi = np.array([0])
    K = analytical_sifs(a, beta, dim_h, sigma, phi)[:, np.newaxis]
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
#    plt.figure()
    save(gb, u, errors, file_name, folder_name, export, sort_error=False)
#    print('e', errors)
    signed = ((K - s)/K)[0]
#    if np.isclose(beta, np.pi / 2):
#        xaxis = rms / a
#        aa = plt.gca()
#        aa.set_yscale('log')
#        aa.scatter(xaxis, errors)
#        aa.set_ylabel('$e_t$')
#        aa.set_xlabel('$r_m$ / $a$')
#        plt.scatter(xaxis, errors[:nc])
#    else:
#        plt.semilogy(rms/a, errors)
#        plt.xlabel('$r_m$ / $L_n$')
#        plt.ylabel('$e_t$')
#    plot_vs_normalized(g_l, errors, rms[:, 0], 5)
    return errors, s, rms, signed


def plot_against_eta(e, eta):
#    plt.figure()
    plt.title('Absolute aperture error along fracture')
    aa = plt.gca()
    aa.set_yscale('log')
    aa.scatter(eta, e)
    aa.set_xlabel('$\eta$')
    aa.set_ylabel('E')


def save(gb, u, errors, file_name, folder_name, export=True, sort_error=True,
         sifs=None):
    if not export:
        return
    for g, d in gb:
        if g.dim == gb.dim_max():

            disp = u[:g.dim * g.num_cells].reshape((g.dim, g.num_cells),
                                                   order='F')
            disp = np.append(disp, np.zeros(((3-g.dim), g.num_cells)), axis=0)
            d['displacement'] = disp
            d['error'] = np.zeros(g.num_cells)
            n_digits = 10
            cell_map = pp.utils.setmembership.ismember_rows(
                    np.around(gb.grids_of_dimension(g.dim-1)[0].cell_centers,
                              n_digits),
                    np.around(g.face_centers[:, g.frac_pairs[0]], n_digits),
                    sort=False)[1]
            if sifs is not None:
                d['sifs'] = np.zeros((g.dim, g.num_cells))
        else:
            if sort_error:
                d['error'] = errors[cell_map]
            else:
                d['error'] = errors
            u_left = u[-gb.dim_max() * 2 * g.num_cells:-gb.dim_max() * g.num_cells]
            u_right = u[-gb.dim_max() * g.num_cells:]
            # only y component
            u_left = u_left[np.arange(g.dim, u_left.size, gb.dim_max())]
            u_right = u_right[np.arange(g.dim, u_right.size, gb.dim_max())]
            u_f = np.array([u_right, u_left])[:, cell_map]
            u_f = np.append(u_f, -np.diff(u_f, axis=0), axis=0)
            d['displacement'] = u_f
            if sifs is not None:
                if sort_error:
                    d['sifs'] = sifs[:, cell_map]
                else:
                    d['sifs'] = sifs
    e = pp.Exporter(gb, file_name, folder_name)
    if sifs is None:
        e.write_vtk(data=['displacement', 'error'])
    else:
        e.write_vtk(data=['displacement', 'error', 'sifs'])


if __name__ == '__main__':
#    plt.close('all')
    penny = True
    symmetry = True
    plot = False
    export = False
    file_name = 'displacement_and_aperture_error_temp1'
    folder_name = 'aperture/sneddon_symmetry'
    if not penny:
        file_name += '_notpenny'
        folder_name += '_notpenny'
    from_gmsh = 1 == 1
    # Geometry1
    height = 50
    length = 25
    if symmetry:
        if penny:
            t = 25
        center = np.array([0, t, height / 2])
    else:
        center = np.array([length / 2, length / 2, 0])
    beta = np.pi / 2  # inclination from vertical axis
    a = 2  # fracture radius
    # Driving force
    p0 = 1e-5
    n_fracture_points = 71#39
    def simplex_3d_of_h(h):
        return simplex_3d(h, length, height, a, t, beta, file_name, from_gmsh,
                          penny=penny, n_points=n_fracture_points)
    h = np.array([14, 7]) # Factor 8
    h = np.array([10, 5])
    h = np.array([18.5])
#    folder_name = 'aperture/sneddon/simplex_3d_penny'
    e, e_max, apertures, apertures_a, u, gb, errors, eta \
        = run_multiple_and_plot(h, simplex_3d_of_h)
    save(gb, u, errors, file_name, folder_name, export=True, sort_error=True)

    sif_errors, sifs, rms, signed = evaluate_sifs_rm_3d(a, beta, gb, u,
                                                        move_tip=False, sigma=p0)