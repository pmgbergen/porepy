import numpy as np

from porepy.fracs import meshing
from porepy.params import bc, tensor
from porepy.params.data import Parameters


def grid_2d_1d(nx=[2, 2], fracture_x=1):
    """
    Make the simplest possible fractured grid: 2d unit square with one
    horizontal fracture extending from 0 to fracture_x <= 1.

    """
    eps = 1e-10
    assert fracture_x < 1 + eps and fracture_x > - eps

    f = np.array([[0, fracture_x],
                  [.5, .5]])
    gb = meshing.cart_grid([f], nx, **{'physdims': [1, 1]})
    return gb


def grid_3d_2d(nx=[2, 2, 2], fracture_x=1):
    """
    Make the simplest possible fractured grid: 2d unit square with one
    horizontal fracture extending from 0 to fracture_x <= 1.

    """
    eps = 1e-10
    assert fracture_x < 1 + eps and fracture_x > - eps

    f = np.array([[0.0, fracture_x, fracture_x, 0.0],
                  [0.0, 0.0, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5]])
    gb = meshing.cart_grid([f], nx, **{'physdims': [1, 1, 1]})
    return gb


def setup_flow_2d_1d(nx=[2, 2], fracture_x=1):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a flow
    problem simulation (top to bottom).
    """
    gb = grid_2d_1d(nx, fracture_x)
    kf = 1e-3
    gb.add_node_props(['param'])
    a = 1e-2
    physics = 'flow'
    for g, d in gb:
        param = Parameters(g)

        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)

        p = tensor.SecondOrder(3, np.ones(g.num_cells)
                               * np.power(kf, g.dim < gb.dim_max()))
        param.set_tensor(physics, p)

        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 1 - 0.1 / nx[1]
            bottom = bound_face_centers[1, :] < 0.1 / nx[1]

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = g.face_centers[1, bc_dir]

            param.set_bc(physics, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(physics, bc_val)

        d['param'] = param
    return gb


def setup_flow_3d_2d(nx=[2, 2, 2], fracture_x=1):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a flow
    problem simulation (top to bottom).
    """
    gb = grid_3d_2d(nx, fracture_x)
    kf = 1e-3
    gb.add_node_props(['param'])
    a = 1e-2
    for g, d in gb:
        param = Parameters(g)

        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)

        p = tensor.SecondOrder(3, np.ones(g.num_cells)
                               * np.power(kf, g.dim < gb.dim_max()))
        param.set_tensor('flow', p)

        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size > 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 1 - 0.1 / nx[1]
            bottom = bound_face_centers[1, :] < 0.1 / nx[1]

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = g.face_centers[1, bc_dir]

            param.set_bc('flow', bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val('flow', bc_val)

        d['param'] = param
    return gb


def setup_flow(nx=[2, 2], fracture_x=1, dim_max=2):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a flow
    problem simulation (top to bottom).
    """
    if dim_max == 2:
        gb = grid_2d_1d(nx, fracture_x)
    else:
        gb = grid_3d_2d(nx, fracture_x)
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
    eps = 1e-10
    for g, d in gb:
        param = d['param']

        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)

        perm = tensor.SecondOrder(3, np.ones(g.num_cells)
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


def setup_mech(nx=[2, 2], fracture_x=1, dim_max=2):
    """
    Set up a simple grid bucket with minimal parameters and BCs for a
    mechanical simulation.
    """
    if dim_max == 2:
        gb = grid_2d_1d(nx, fracture_x)
    else:
        gb = grid_3d_2d(nx, fracture_x)
    gb.add_node_props(['param'])
    for g, d in gb:
        param = Parameters(g)
        d['param'] = param

    set_bc_mech(gb)
    return gb


def set_bc_mech(gb):
    """
    Minimal bcs for mechanics.
    """
    a = 1e-2
    physics = 'mechanics'
    for g, d in gb:
        param = d['param']
        # Update fields s.a. param.nfaces.
        # TODO: Add update method
        param.__init__(g)
        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(aperture)

        bound_faces = g.get_boundary_faces()
        if bound_faces.size > 0:
            bc_val = np.zeros(g.num_faces * g.dim)
            param.set_bc(physics, bc.BoundaryCondition(g, bound_faces, 'dir'))
            param.set_bc_val(physics, bc_val)
            param.set_slip_distance(np.zeros(g.num_faces * g.dim))


def update_apertures(gb, gl, faces_h):
    """
    Assign apertures to new lower-dimensional cells.
    """
    apertures_l = 0.01*np.ones(faces_h.size)
    try:
        a = np.append(gb.node_prop(gl, 'param').get_aperture(), apertures_l)
        gb.node_prop(gl, 'param').set_aperture(a)
    except KeyError:
        warnings.warn('apertures not updated')



