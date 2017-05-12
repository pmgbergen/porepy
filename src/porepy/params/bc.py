# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""

import numpy as np


class BoundaryCondition(object):
    """ Class to store information on boundary conditions.

        The BCs are specified by face number, and can have type Dirichlet or
        Neumann (Robin may be included later).
        The face-based indexing makes this natural for control volume methods.
        For vector equations, it is ismplicitly assumed that the

    """

    """ Initialize boundary conditions

        Note: By default, all boundaries

    """
    def __init__(self, g, faces=None, cond=None):

        self.num_faces = g.num_faces
        self.dim = g.dim - 1

        # Find boundary faces
        bf = g.get_boundary_faces()

        self.is_neu = np.zeros(self.num_faces, dtype=bool)
        self.is_dir = np.zeros(self.num_faces, dtype=bool)

        self.is_neu[bf] = True

        if faces is not None:
            # Validate arguments
            assert cond is not None
            if not np.all(np.in1d(faces, bf)):
                raise ValueError('Give boundary condition only on the boundary')
            if faces.size != len(cond):
                raise ValueError('One BC per face')

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == 'neu':
                    pass  # Neumann is already default
                elif s.lower() == 'dir':
                    self.is_dir[faces[l]] = True
                    self.is_neu[faces[l]] = False
                else:
                    raise ValueError('Boundary should be Dirichlet or Neumann')


def face_on_side(g, side, tol=1e-8):
    """ Find faces on specified sides of a grid.

    It is assumed that the grid forms a box in 2d or 3d.

    The faces are specified by one of two type of keywords: (xmin / left),
    (xmax / right), (ymin / south), (ymax / north), (zmin, bottom),
    (zmax / top).

    Parameters:
        g (grid): For which we want to find faces.
        side (str, or list of str): Sides for which we want to find the
            boundary faces.
        tol (double, optional): Geometric tolerance for deciding whether a face
            lays on the boundary. Defaults to 1e-8.

    Returns:
        list of lists: Outer list has one element per element in side (same
            ordering). Inner list contains global indices of faces laying on
            that side.

    """
    if isinstance(side, str):
        side = [side]

    faces = []
    for s in side:
        s = s.lower().strip()
        if s == 'west' or s == 'xmin':
            xm = g.nodes[0].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[0] - xm) <
                                              tol)))
        if s == 'east' or s == 'xmax':
            xm = g.nodes[0].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[0] - xm) <
                                              tol)))
        if s == 'south' or s == 'ymin':
            xm = g.nodes[1].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[1] - xm) <
                                              tol)))
        if s == 'north' or s == 'ymax':
            xm = g.nodes[1].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[1] - xm) <
                                              tol)))
        if s == 'bottom' or s == 'zmin':
            xm = g.nodes[2].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[2] - xm) <
                                              tol)))
        if s == 'top' or s == 'zmax':
            xm = g.nodes[2].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[2] - xm) <
                                              tol)))
    return faces

