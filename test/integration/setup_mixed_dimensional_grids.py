import warnings
import numpy as np
from porepy.fracs import meshing
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.numerics.fv import fvutils
from porepy.utils import comp_geom as cg

def grid_2d_1d(nx=[2, 2], x_start=0, x_stop=1):
    """
    Make the simplest possible fractured grid: 2d unit square with one
    horizontal fracture extending from 0 to fracture_x <= 1.

    """
    eps = 1e-10
    assert x_stop < 1 + eps and x_stop > - eps
    assert x_start < 1 + eps and x_stop > - eps

    f = np.array([[x_start, x_stop],
                  [.5, .5]])
    gb = meshing.cart_grid([f], nx, **{'physdims': [1, 1]})
    return gb


def grid_3d_2d(nx=[2, 2, 2], x_start=0, x_stop=1):
    """
    Make the simplest possible fractured grid: 2d unit square with one
    horizontal fracture extending from 0 to fracture_x <= 1.

    """
    eps = 1e-10
    assert x_stop < 1 + eps and x_stop > - eps
    assert x_start < 1 + eps and x_stop > - eps

    f = np.array([[x_start, x_stop, x_stop, x_start],
                  [0.0, 0.0, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5]])
    gb = meshing.cart_grid([f], nx, **{'physdims': [1, 1, 1]})
    return gb


def setup_flow(nx=[2, 2], x_stop=1, x_start=0):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a flow
    problem simulation (top to bottom).
    """
    dim_max = len(nx)
    if dim_max == 2:
        gb = grid_2d_1d(nx, x_start, x_stop)
    else:
        gb = grid_3d_2d(nx, x_start, x_stop)
    gb.add_node_props(['param'])
    for g, d in gb:
        param = Parameters(g)
        d['param'] = param

    set_bc_flow(gb)
    return gb


def set_bc_flow(gb):
    """
    Set bc parameters in all node data dictionaries in the bucket for flow
    problem.
    """
    a = 1e-2
    kf = 1e-3
    physics = 'flow'
    eps = 1e-1
    for g, d in gb:
        param = d['param']

        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)

        perm = tensor.SecondOrderTensor(3, np.ones(g.num_cells)
                                * np.power(kf, g.dim < gb.dim_max()))
        param.set_tensor(physics, perm)

        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 1 - eps
            bottom = bound_face_centers[1, :] < eps

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = g.face_centers[1, bc_dir]

            param.set_bc(physics, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(physics, bc_val)


def setup_mech(nx=[2, 2], x_stop=1, x_start=0):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a
    mechanical simulation.
    """
    dim_max = len(nx)
    if dim_max == 2:
        gb = grid_2d_1d(nx,  x_start, x_stop)
    else:
        gb = grid_3d_2d(nx, x_start, x_stop)
    gb.add_node_props(['param'])
    for g, d in gb:
        param = Parameters(g)
        d['param'] = param

    set_bc_mech(gb)
    return gb

def setup_mech_tension(nx=[2, 2], x_stop=1, x_start=0):
    dim_max = len(nx)
    if dim_max == 2:
        gb = grid_2d_1d(nx,  x_start, x_stop)
    else:
        gb = grid_3d_2d(nx, x_start, x_stop)
    gb.add_node_props(['param'])
    for g, d in gb:
        param = Parameters(g)
        d['param'] = param

    set_bc_mech_tension(gb)
    return gb

def tensor_plane_strain(c, E, nu):
    print('in', np.round(c.c[:,:,0], 3))
    nu_mat = np.array([[-1, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, -1, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, -1, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    cons_mat = np.zeros((9, 9))
#    for i in [0, 1, 3, 4]:
#        cons_mat[i, i] = 1


    for i in [0, 4]:
        cons_mat[i, i] = 1

    for i in [1, 3]:
        cons_mat[i, i] = 1/2

    cons_mat[1, 3] = 1/2
    cons_mat[3, 1] = 1/2
    nu_mat = nu_mat[:, :, np.newaxis]
    cons_mat = cons_mat[:, :, np.newaxis]
    constant = np.divide(E, np.multiply(1 + nu, 1 - 2 * nu))
    c.c = constant * (cons_mat + nu_mat * nu)
    print('\nout', np.round(c.c[:,:,10], 3))


def tensor_plane_stress(c, E, nu):
    print('in', np.round(c.c[:,:,0], 3))
    nu_mat = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, -1/2, 0, -1/2, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1/2, 0, -1/2, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    cons_mat = np.zeros((9, 9))

    for i in [0, 4]:
        cons_mat[i, i] = 1

    for i in [1, 3]:
        cons_mat[i, i] = 1/2

    cons_mat[1, 3] = 1/2
    cons_mat[3, 1] = 1/2
    nu_mat = nu_mat[:, :, np.newaxis]
    cons_mat = cons_mat[:, :, np.newaxis]
    constant = np.divide(E, 1 - np.power(nu, 2))
    c.c = constant * (cons_mat + nu_mat * nu)
    print('\nout', np.round(c.c[:,:,10], 3))


def set_bc_mech_tension(gb, top_tension=1, t=.05, l=1, fix_faces=True):
    """
    Minimal bcs for mechanics.
    """
#    a = 1e-2
    physics = 'mechanics'
    eps = 1e-10
    last_ind = gb.dim_max() - 1

    for g, d in gb:
        param = d['param']
        # Update fields s.a. param.nfaces.
        # TODO: Add update method
        param.__init__(g)
        E = 1
        poisson = .3
        d['Young'] = E
        d['Poisson'] = poisson # = nu
        if g.dim > 1:
            lam = np.ones(g.num_cells) * E * poisson / ((1 + poisson) * (1 - 2 * poisson))
            mu = np.ones(g.num_cells) * E / (2 * (1 + poisson))
            constit = tensor.FourthOrderTensor(g.dim, mu, lam)
#            tensor_plane_strain(constit, E* np.ones(g.num_cells),
#                                poisson * np.ones(g.num_cells))
            param.set_tensor(physics, constit)

#        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
#        param.set_aperture(aperture)
        bound_faces = g.get_all_boundary_faces()
        internal_faces = g.tags['fracture_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]


            top = bound_face_centers[last_ind, :] > l - eps
            bottom = bound_face_centers[last_ind, :] < eps
            right = bound_face_centers[0, :] > 1 - eps
            left = bound_face_centers[0, :] < eps

            labels = ['neu' for _ in range(bound_faces.size)]
            internal_ind = np.isin(bound_faces, internal_faces).nonzero()[0]
            for i in internal_ind:
                labels[i] = 'dir'
#            side_ind = right.nonzero()[0] #np.logical_or(left, right).nonzero()[0]
            topbot_ind  = np.logical_or(bottom, top).nonzero()[0]
            bc_val = np.zeros((g.dim, g.num_faces))
#            bc_dir = bound_faces[np.logical_or(top, bottom)]
            if g.dim == 3:
                back = bound_face_centers[1, :] <  eps
                front = bound_face_centers[1, :] > 2 * t - eps
                frontback_ind = np.logical_or(back, front).nonzero()[0]
                for i in frontback_ind:
                    labels[i] = 'dir_y'
            if g.dim > last_ind:
#                for i in side_ind:
#                    labels[i] = 'dir_x'
                ind_b = bound_faces[bottom]
                ind_t = bound_faces[top]
#                for i in bottom:
#                    labels[i] = 'dir_y'
#                ind = np.argmin(cg.dist_point_pointset(np.array([, eps, eps]), bound_faces))
#                labels[ind] = 'dir'
                if g.dim == 3 and fix_faces:
                    le = left.nonzero()[0]
                    ri = right.nonzero()[0]
                    dist = cg.dist_point_pointset(np.array([0, t, .5]),
                                                  bound_face_centers[:, le])
                    lock_index_0 = le[np.argmin(dist)]
                    labels[lock_index_0] = 'dir_z'
                    dist = cg.dist_point_pointset(np.array([1, t, .5]),
                                                  bound_face_centers[:, ri])
                    lock_index_1 = ri[np.argmin(dist)]
                    labels[lock_index_1] = 'dir_z'
                    print('fixed faces at ',
                          bound_face_centers[:, lock_index_0],
                          ' and ', bound_face_centers[:, lock_index_1])
                    to = top.nonzero()[0]
                    bo = bottom.nonzero()[0]
                    dist = np.absolute(bound_face_centers[0, to] - .5 + eps)
                    dist = cg.dist_point_pointset(np.array([.5, t, 1]),
                                                  bound_face_centers[:, to])

                    lock_index_0 = to[np.argmin(dist)]
                    labels[lock_index_0] = 'dir_x'
                    dist = np.absolute(bound_face_centers[0, bo] - .5 + eps)
                    dist = cg.dist_point_pointset(np.array([.5, t, 0]),
                                                  bound_face_centers[:, bo])

                    lock_index_1 = bo[np.argmin(dist)]
                    labels[lock_index_1] = 'dir_x'
                    print('fixed faces at ',
                          bound_face_centers[:, lock_index_0],
                          ' and ', bound_face_centers[:, lock_index_1])
#                labels[ind_b] = ['dir']
#                labels[ind_t] = ['dir']
                labels = np.array(labels)
#                labels[top] = 'dir'
#                labels[bottom] = 'dir'

                ind_t = bound_faces[top]

                bc_val[last_ind, ind_t] = top_tension * g.face_areas[ind_t]
                bc_val[last_ind, ind_b] = - top_tension * g.face_areas[ind_b]

#                bc_val[last_ind, bound_faces[ind]] = 0

#            top = g.face_centers[1, :] > 1 - eps
#            bottom = bound_face_centers[1, :] < eps
#            labels = np.array(['neu'] * bound_faces.size)
#            labels[np.logical_or(top, bottom)] = ['dir']

#            bc_val = np.zeros((g.dim, g.num_faces))
#            if g.dim > 1:
#                bc_val[1, top] = top_displacement
                param.set_bc(physics, bc.BoundaryConditionVectorial(g, bound_faces, labels))
                param.set_bc_val(physics, bc_val.ravel('F'))
                param.set_slip_distance(np.zeros(g.num_faces * g.dim))


def set_bc_mech_tension_sneddon(gb, p0, height, length, beta, t=.05, penny=False):
    """
    Minimal bcs for mechanics.
    """
    physics = 'mechanics'
    eps = 1e-10
    last_ind = gb.dim_max() - 1
    dim_h = gb.dim_max()
    for g, d in gb:
        param = d['param']
        # Update fields s.a. param.nfaces.
        # TODO: Add update method
        param.__init__(g)
        mu = 1 # shear modulus
        poisson = .25
        d['shear_modulus'] = mu
        d['Poisson'] = poisson
#        d['max_memory'] = 1e7#2959680000
        if g.dim > 1:
            lam = np.ones(g.num_cells) * 2 * mu * poisson /(1 - 2 * poisson)
            mu = np.ones(g.num_cells) * mu
            constit = tensor.FourthOrderTensor(3, mu, lam)
            param.set_tensor(physics, constit)

#        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
#        param.set_aperture(aperture)
        bound_faces = g.get_all_boundary_faces()
        internal_faces = g.tags['fracture_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[last_ind, :] > height - eps
            bottom = bound_face_centers[last_ind, :] < eps
            right = bound_face_centers[0, :] > length - eps
            left = bound_face_centers[0, :] < eps

            labels = ['neu' for _ in range(bound_faces.size)]
            internal_ind = np.isin(bound_faces, internal_faces).nonzero()[0]
            for i in internal_ind:
                labels[i] = 'dir'
            side_ind = right.nonzero()[0] #np.logical_or(left, right).nonzero()[0]
#            topbot_ind  = np.logical_or(bottom, top).nonzero()[0]
            topbot_ind  = bottom.nonzero()[0]
            bc_val = np.zeros((g.dim, g.num_faces))
            internal_tension = np.zeros((g.dim, g.num_faces))
#            bc_dir = bound_faces[np.logical_or(top, bottom)]
            if g.dim == 3 and not penny:
                back = bound_face_centers[1, :] <  eps
                front = bound_face_centers[1, :] > 2 * t - eps
                frontback_ind = np.logical_or(back, front).nonzero()[0]
                for i in frontback_ind:
                    labels[i] = 'dir_y'

            if g.dim > last_ind:

#                for i in side_ind:
#                    labels[i] = 'dir_x'
#                for i in topbot_ind:
#                    labels[i] = 'dir_y'
                labels = np.array(labels)
#                ind_t = bound_faces[top]
#                ind_b = bound_faces[bottom]
#                bc_val[last_ind, ind_t] = top_tension * g.face_areas[ind_t]
#                bc_val[last_ind, ind_b] = - top_tension * g.face_areas[ind_b]

#                normals = np.multiply(g.face_normals[last_ind, internal_faces], g.cell_faces[internal_faces].data)
                internal_tension[last_ind, g.frac_pairs[0]] \
                    = p0 * g.face_areas[g.frac_pairs[0]] * np.sin(beta)
                internal_tension[0, g.frac_pairs[0]] \
                    = p0 * g.face_areas[g.frac_pairs[0]] * np.cos(beta)
                neighbour_cells = g.cell_faces[g.frac_pairs[0]]
                cc = g.cell_centers[last_ind, neighbour_cells.nonzero()[1]]
                if np.isclose(beta, np.pi / 2):
                    internal_tension[:, g.frac_pairs[0]] \
                    = (2 * (cc > (height / 2)) - 1) * internal_tension[:, g.frac_pairs[0]]
                fix_sides(g, labels, bc_val, left, right, top, bottom,
                          bound_face_centers, t, bound_faces)
#            top = g.face_centers[1, :] > 1 - eps
#            bottom = bound_face_centers[1, :] < eps
#            labels = np.array(['neu'] * bound_faces.size)
#            labels[np.logical_or(top, bottom)] = ['dir']

#            bc_val = np.zeros((g.dim, g.num_faces))
#            if g.dim > 1:
#                bc_val[1, top] = top_displacement
            param.set_bc(physics, bc.BoundaryConditionVectorial(g, bound_faces, labels))
            param.set_bc_val(physics, bc_val.ravel('F'))
            param.set_slip_distance(internal_tension.ravel('F'))


#def boundary_tension_sneddon(g, p0):

def fix_sides(g, labels, bc_vals, left, right, top, bottom, bound_face_centers,
              t, bound_faces):
    le = left.nonzero()[0]
#                    ri = right.nonzero()[0]
#                    dist = cg.dist_point_pointset(np.array([0, t, .5]),
#                                                  bound_face_centers[:, le])
#                    lock_index_0 = le[np.argmin(dist)]
#                    labels[lock_index_0] = 'dir_z'
#                    dist = cg.dist_point_pointset(np.array([1, t, .5]),
#                                                  bound_face_centers[:, ri])
#                    lock_index_1 = ri[np.argmin(dist)]
#                    labels[lock_index_1] = 'dir_z'
#                    print('fixed faces at ',
#                          bound_face_centers[:, lock_index_0],
#                          ' and ', bound_face_centers[:, lock_index_1])
#                    to = top.nonzero()[0]
#                    bo = bottom.nonzero()[0]
#                    dist = np.absolute(bound_face_centers[0, to] - .5 + eps)
#                    dist = cg.dist_point_pointset(np.array([.5, t, 1]),
#                                                  bound_face_centers[:, to])
#
#                    lock_index_0 = to[np.argmin(dist)]
#                    labels[lock_index_0] = 'dir_x'
#                    dist = np.absolute(bound_face_centers[0, bo] - .5 + eps)
#                    dist = cg.dist_point_pointset(np.array([.5, t, 0]),
#                                                  bound_face_centers[:, bo])
#
#                    lock_index_1 = bo[np.argmin(dist)]
#                    labels[lock_index_1] = 'dir_x'
#                    print('fixed faces at ',
#                          bound_face_centers[:, lock_index_0],
#                          ' and ', bound_face_centers[:, lock_index_1])
    if g.dim == 2:
        last_ind = 1
        left_point = np.array([0, .5, 0])
        right_point = np.array([1, .5, 0])
        bottom_point = np.array([.5, 0, 0])
        top_point = np.array([.5, 1, 0])
        le = left.nonzero()[0]
        ri = right.nonzero()[0]
        dist = cg.dist_point_pointset(left_point,
                                      bound_face_centers[:, le])
        lock_index_l = le[np.argmin(dist)]
        labels[lock_index_l] = 'dir_z'
        dist = cg.dist_point_pointset(right_point,
                                      bound_face_centers[:, ri])
        lock_index_r = ri[np.argmin(dist)]
        labels[lock_index_r] = 'dir_z'
        print('fixed faces at ',
              bound_face_centers[:, lock_index_l],
              ' and ', bound_face_centers[:, lock_index_r])
        to = top.nonzero()[0]
        bo = bottom.nonzero()[0]
        dist = cg.dist_point_pointset(top_point,
                                      bound_face_centers[:, to])

        lock_index_t = to[np.argmin(dist)]
        labels[lock_index_t] = 'dir_x'
        dist = cg.dist_point_pointset(bottom_point,
                                      bound_face_centers[:, bo])

        lock_index_b = bo[np.argmin(dist)]
        labels[lock_index_b] = 'dir_x'
        print('fixed faces at ',
              bound_face_centers[:, lock_index_b],
              ' and ', bound_face_centers[:, lock_index_t])
        bc_vals[0, bound_faces[lock_index_l]] = 0
        bc_vals[0, bound_faces[lock_index_r]] = 0
        bc_vals[last_ind, bound_faces[lock_index_t]] = 0
        bc_vals[last_ind, bound_faces[lock_index_b]] = 0


def set_bc_mech(gb, top_displacement=.01):
    """
    Minimal bcs for mechanics.
    """
    a = 1e-2
    physics = 'mechanics'
    eps = 1e-10
    last_ind = gb.dim_max() - 1
    
    for g, d in gb:
        param = d['param']
        
        E = 1
        poisson = .3
        d['Young'] = E
        d['Poisson'] = poisson
        # Update fields s.a. param.nfaces.
        # TODO: Add update method
        param.__init__(g)
        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)
        bound_faces = g.get_all_boundary_faces()
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[last_ind, :] > 1 - eps
            bottom = bound_face_centers[last_ind, :] < eps
            left = bound_face_centers[0, :] > 1 - eps
            right = bound_face_centers[0, :] < eps

            labels = np.array(['dir'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['neu']

            bc_val = np.zeros((g.dim, g.num_faces))
            bc_dir = bound_faces[top]
            if g.dim > last_ind:
                bc_val[last_ind, bc_dir] = top_displacement


#            top = g.face_centers[1, :] > 1 - eps
#            bottom = bound_face_centers[1, :] < eps
#            labels = np.array(['neu'] * bound_faces.size)
#            labels[np.logical_or(top, bottom)] = ['dir']

#            bc_val = np.zeros((g.dim, g.num_faces))
#            if g.dim > 1:
#                bc_val[1, top] = top_displacement
            param.set_bc(physics, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(physics, bc_val.ravel('F'))
            param.set_slip_distance(np.zeros(g.num_faces * g.dim))


def update_apertures(gb, gl, faces_h):
    """
    Assign apertures to new lower-dimensional cells.
    """
    apertures_l = 0.01*np.ones(faces_h.size)
    try:
        a = np.append(gb.node_props(gl, 'param').get_aperture(), apertures_l)
        gb.node_props(gl, 'param').set_aperture(a)
    except KeyError:
        warnings.warn('apertures not updated')



