import numpy as np

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

#------------------------------------------------------------------------------#

def add_data(gb, domain, direction, tol):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['param'])

    # Aavatsmark_transmissibilities only for tpfa intra-dimensional coupling

    for g, d in gb:
        d['Aavatsmark_transmissibilities'] = True

        if g.dim < 2:
            continue

        param = Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells)
        param.set_tensor("flow", tensor.SecondOrder(3, kxx))

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            if direction == 'left_right':
                left = bound_face_centers[0, :] < domain['xmin'] + tol
                right = bound_face_centers[0, :] > domain['xmax'] - tol
                bc_dir = np.logical_or(left, right)
                bc_one = right
            elif direction == 'bottom_top':
                bottom = bound_face_centers[2, :] < domain['zmin'] + tol
                top = bound_face_centers[2, :] > domain['zmax'] - tol
                bc_dir = np.logical_or(top, bottom)
                bc_one = top
            elif direction == 'back_front':
                back = bound_face_centers[1, :] < domain['ymin'] + tol
                front = bound_face_centers[1, :] > domain['ymax'] - tol
                bc_dir = np.logical_or(back, front)
                bc_one = front

            labels = np.array(['neu'] * bound_faces.size)
            labels[bc_dir] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[bc_one]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    gb.add_edge_prop('Aavatsmark_transmissibilities')
    for _, d in gb.edges_props():
        d['Aavatsmark_transmissibilities'] = True

#------------------------------------------------------------------------------#
