"""
This is a set up of a test case from Robin Thomas' submitted paper "rowth rate
effects on multiple fracture growth in brittle materials under tension and
compression". Apertures and SIFs are compared against analytical solution.

TODO:
    Still issue with 180 degree rotation in displacement_correlation
"""
import os
import time
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
import pickle

import porepy as pp

from examples.example11.fracture_opening_convergence_tests_sneddon import \
    save, L2_error, analytical_displacements
from examples.example11.fracture_opening_convergence_tests_sneddon import \
    simplex_3d as simplex_3d_gb

# -----analytical-------------------------------------------------------------#


def analytical_sifs(a, beta, dim, sigma, phi=0):
    """
    Analytical stress intensity factors for the through-the-thickness crack
    problem in question.
    """

    nu = .23
    cons = 2 * sigma * np.sqrt(a / np.pi)
    K_I = cons * np.square(np.sin(beta)) * np.ones(phi.size)
    K_II = cons / (2 - nu) * np.sin(2 * beta) * np.cos(phi)
    K_III = cons * (1 - nu) / (2 - nu) * np.sin(2 * beta) * np.sin(phi)
    K = np.array([K_I, K_II, K_III])
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
        d['penny'] = penny

    physics = 'mechanics'
    eps = 1e-10
    last_ind = gb.dim_max() - 1

    for g, d in gb:
        param = d['param']
        # Update fields s.a. param.nfaces.
        # TODO: Add update method
        param.__init__(g)
        E = Young
        Poisson = .23

        d['Poisson'] = Poisson
        d['Young'] = E
        mu = E / (2 * (1 + Poisson))  # shear modulus
        d['shear_modulus'] = mu
        if g.dim > 1:
            lam = np.ones(g.num_cells) * E * Poisson \
                / ((1 + Poisson) * (1 - 2 * Poisson))
            mu = np.ones(g.num_cells) * mu
            constit = pp.FourthOrderTensor(g.dim, mu, lam)
            param.set_tensor(physics, constit)

        bound_faces = g.get_all_boundary_faces()
        internal_faces = g.tags['fracture_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]
            top = bound_face_centers[last_ind, :] > height - eps
            bottom = bound_face_centers[last_ind, :] < eps

            labels = ['neu' for _ in range(bound_faces.size)]
            internal_ind = np.isin(bound_faces, internal_faces).nonzero()[0]
            for i in internal_ind:
                labels[i] = 'dir'
            bc_val = np.zeros((g.dim, g.num_faces))
            if g.dim > last_ind:
#                ind_t = bound_faces[top]
#
#                ind_t = bound_faces[top]
#                normals = g.face_normals[last_ind, ind_t] \
#                    * g.cell_faces[ind_t].data
#                bc_val[last_ind, ind_t] = np.absolute(p0 * normals)
#                for i in bottom.nonzero()[0]:
#                    labels[i] = 'dir_z'
                dist = pp.cg.dist_point_pointset(np.array([10, 10, 0]),
                                                 bound_face_centers[:, bottom])
                ind = np.argmin(dist)
#                labels[bottom[ind]] = 'dir'
                internal_tension = np.zeros((g.dim, g.num_faces))
                internal_tension[last_ind, g.frac_pairs[0]] \
                    = p0 * g.face_areas[g.frac_pairs[0]] * np.sin(beta)
                internal_tension[0, g.frac_pairs[0]] \
                    = p0 * g.face_areas[g.frac_pairs[0]] * np.cos(beta)
                bcs = pp.BoundaryConditionVectorial(g, bound_faces, labels)
                param.set_bc(physics, bcs)
                param.set_bc_val(physics, bc_val.ravel('F'))
                param.set_slip_distance(internal_tension.ravel('F'))


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


def run(nc, a, t, beta=np.pi/2, sigma=1, fn=None):
    dim_h = 3
    h = length / nc
    gb = simplex_3d_gb(h, length, height, a, t, beta, file_name=fn,
                       from_gmsh=from_gmsh, fracture=True, penny=penny,
                       n_points=n_fracture_points)
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
    errors, sifs, local_errors = evaluate_sifs_from_u(a, beta, gb, u, sigma)
    return errors, sifs, gb, u, local_errors


def evaluate_sifs_from_u(a, beta, gb, u, sigma, move_tip=False, **dc_kw):
    dim_h = 3
    g_l = gb.grids_of_dimension(dim_h-1)[0]
    t1 = time.time()
    _, sifs = pp.displacement_correlation.faces_to_open(gb, u, [1, 1, 1],
                                                        **dc_kw)
    sifs = sifs[0]
    print('DC time', time.time() - t1)
    sifs[1:] = -sifs[1:]
    areas = g_l.face_areas[g_l.tags['tip_faces']]
    phi = compute_phi(g_l, beta)
    K = analytical_sifs(a, beta, 3, sigma, phi)
    ind = np.multiply(K[1], sifs[1]) < 0
    sifs[1, ind] = -sifs[1, ind]
    ind = np.multiply(K[2], sifs[2]) < 0
    sifs[2, ind] = -sifs[2, ind]

    errors = L2_error(K, sifs, areas)
    local_errors = np.zeros(g_l.num_cells)
    ind = g_l.cell_faces[g_l.tags['tip_faces']].nonzero()[1]
    local_errors[ind] = (K[0] - sifs[0]) / K[0, 0]
    return errors, sifs, local_errors


def compute_phi(g_l, beta):
    """
    Compute the polar angle in the fracture plane of g_l.
    """
    R = pp.cg.rot(-beta, [1, 0, 0])
    v = g_l.face_centers[:, g_l.tags['tip_faces']] - center[:, np.newaxis]
    v = np.dot(R, v)
    phi = np.angle(v[0] + v[1]*1j) + np.pi / 2
    return phi


def plot_all(do_plot=True):
    if not do_plot:
        return
    if fixed_ntips:
        xlabel = 'Number of boundary elements'
    else:
        xlabel = 'Number of fracture tips'
    plt.figure()
    plt.plot(v, all_sif_l2)
    plt.xlabel(xlabel)
    plt.ylabel('Aperture error')
    plt.figure()
    plt.plot(v, all_aperture_l2)
    plt.xlabel(xlabel)
    plt.ylabel('SIF error')
    plt.figure()
    plt.scatter(eta, aperture_analytical)
    plt.scatter(eta, aperture)
    plt.xlabel('$\eta$')
    plt.ylabel('Aperture')
    plt.legend(['Analytical', 'FV'])


if __name__ == '__main__':
    export = 1 == 10
    file_name = 'displacements_coarse'
    folder_name = 'displacement_correlation/robin'
    length = 20
    height = 20
    sneddon = False
    a = 1
    p0 = 1e8
    beta = np.pi / 2
    penny = True
    t = 10
    Young = 2e10
    center = np.array([length / 2, height / 2, t])

    existing_gb = 11 == 1
    from_gmsh = 1 == 1
    fixed_ntips = 1 == 19
    # ------------------ 3 D -------------------------------------------------#
    if fixed_ntips:
        # Number of boundary elements:
        v = [2, 3, 5, 8, 10]
        n_fracture_points = 20
    else:
        # Number of fracture tips:
        v = [6, 8, 12, 15, 20, 25, 30]
        v = [20]
        nx = 12

    all_aperture_l2 = []
    all_sif_l2 = []
    root = 'robin_comparison'  #'robin_unshielded'
    for i in v:
        if fixed_ntips:
            nx = i
            folder = root + '/fracture_tips_{}/boundary_size_{}' \
                .format(n_fracture_points, i)
        else:
            n_fracture_points = i
            size = int(length / nx)
            folder = root + '/boundary_size_{}/fracture_tips_{}' \
                .format(size, i)
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = folder + '/geometry'
        if not existing_gb:
            sif_l2, sifs, gb, u, sif_e_loc = run(nx, a, t, fn=file_name,
                                                 beta=beta, sigma=p0)
        else:
            d = pickle.load(open(folder + '/data', 'rb'))
            u = d['u']
            h = length / nx
            gb = simplex_3d_gb(h, length, height, a, t, beta,
                               file_name=file_name, from_gmsh=True,
                               fracture=True, penny=penny,
                               n_points=n_fracture_points)
            assign_parameters(gb)
            sif_l2, sifs, sif_e_loc = evaluate_sifs_from_u(a, beta, gb, u,
                                                          sigma=p0,
                                                          rm_factor=1)
        g_h = gb.grids_of_dimension(gb.dim_max())[0]
        g_l = gb.grids_of_dimension(gb.dim_min())[0]
        d_h = gb.node_props(g_h)
        aperture_analytical, dof_ind, eta = analytical_displacements(gb, a, beta, p0)
        if np.isclose(beta, np.pi / 2):
            aperture = np.absolute(np.diff(u[dof_ind], axis=0))[0]
        aperture_e_loc = (aperture_analytical - aperture) \
                            / np.max(aperture_analytical)
        aperture_error = np.mean(np.absolute(aperture_e_loc))
        aperture_l2 = L2_error(aperture_analytical, aperture, g_l.cell_volumes)
        all_sif_l2.append(sif_l2)
        if not existing_gb:
            d = {'u': u, 'a': aperture, 'a_a': aperture_analytical, 'eta': eta,
                 'aperture_L2': aperture_l2, 'aperture_e_loc': aperture_e_loc,
                 'sif_l2': sif_l2, 'sif_e_loc': sif_e_loc, 'sifs': sifs}
            pickle.dump(d, open(folder + '/data', 'wb'))

        all_aperture_l2.append(aperture_l2)
        save(gb, u, np.absolute(aperture_e_loc), 'aperture',
             folder_name=folder, export=True)

    plot_all(do_plot=False)
