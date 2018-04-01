import warnings
import numpy as np
from porepy.fracs import meshing
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.numerics.fv import fvutils


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

def set_bc_mech_tension(gb, top_tension=1):
    """
    Minimal bcs for mechanics.
    """
    a = 1e-2
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
        d['poisson'] = poisson
        if g.dim > 1:
            lam = np.ones(g.num_cells) * E * poisson / ((1 + poisson) * (1 - 2 * poisson))
            mu = np.ones(g.num_cells) * E / (2 * (1 + poisson))
            constit = tensor.FourthOrderTensor(g.dim, mu, lam)
            param.set_tensor(physics, constit)

        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)
        bound_faces = g.get_all_boundary_faces()
        internal_faces = g.tags['fracture_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[last_ind, :] > 1 - eps
            bottom = bound_face_centers[last_ind, :] < eps
#            left = bound_face_centers[0, :] > 1 - eps
#            right = bound_face_centers[0, :] < eps

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.isin(bound_faces, internal_faces)] = ['dir']

            bc_val = np.zeros((g.dim, g.num_faces))
#            bc_dir = bound_faces[np.logical_or(top, bottom)]
            if g.dim > last_ind:
                ind_t = bound_faces[top]
                ind_b = bound_faces[bottom]
                bc_val[last_ind, ind_t] = top_tension * g.face_areas[ind_t]
                bc_val[last_ind, ind_b] = - top_tension * g.face_areas[ind_b]


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



