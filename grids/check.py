# -*- coding: utf-8 -*-
"""

@author:
"""
import numpy as np
import scipy.sparse as sps

import compgeom.basics as cg

#------------------------------------------------------------------------------#

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

    if g.dim == 2:
        assert cg.is_planar( g.nodes )

    face_normals(g)

#------------------------------------------------------------------------------#

def face_normals(g):
    """ Check if the face normals are actually normal to the faces.

    Args:
        g (grid): Grid, or a subclass, with geometry fields computed.
    """

    nodes, faces, _ = sps.find(g.face_nodes)

    for f in np.arange(g.num_faces):
        loc = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
        normal = g.face_normals[:, f]
        tangent = cg.compute_tangent(g.nodes[:, nodes[loc]])
        assert np.isclose(np.dot(normal, tangent), 0)

#------------------------------------------------------------------------------#
