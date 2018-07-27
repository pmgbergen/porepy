# -*- coding: utf-8 -*-
"""

@author:
"""
import numpy as np
import scipy.sparse as sps

import porepy.utils.comp_geom as cg

# ------------------------------------------------------------------------------#


def grid(g):
    """ Sanity check for the grid. General method which apply the following:
    - check if the face normals are actually normal to the faces
    - check if a bidimensional grid is planar

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.

    How to use:
        import core.grids.check as check
        check.grid(g)
    """

    if g.dim == 1:
        assert cg.is_collinear(g.nodes)
        face_normals_1d(g)

    if g.dim == 2:
        assert cg.is_planar(g.nodes)

    if g.dim != 1:
        face_normals(g)


# ------------------------------------------------------------------------------#


def face_normals(g):
    """ Check if the face normals are actually normal to the faces.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
    """

    nodes, faces, _ = sps.find(g.face_nodes)

    for f in np.arange(g.num_faces):
        loc = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f + 1])
        normal = g.face_normals[:, f]
        tangent = cg.compute_tangent(g.nodes[:, nodes[loc]])
        assert np.isclose(np.dot(normal, tangent), 0)


# ------------------------------------------------------------------------------#


def face_normals_1d(g):
    """ Check if the face normals are actually normal to the faces, 1d case.

    Args:
        g (grid): 1D grid, or a subclass, with geometry fields computed.
    """

    assert g.dim == 1
    a_normal = cg.compute_a_normal_1d(g.nodes)
    for f in np.arange(g.num_faces):
        assert np.isclose(np.dot(g.face_normals[:, f], a_normal), 0)


# ------------------------------------------------------------------------------#
