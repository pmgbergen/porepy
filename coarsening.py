# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps

from core.grids.grid import Grid
from core.constit import second_order_tensor
from core.bc import bc

from utils import matrix_compression

from fvdiscr import tpfa

#------------------------------------------------------------------------------#

def generate_coarse_grid( g, subdiv ):
    """ generate a coarse grid clustering the cells according to some flags.

    The values computed in "compute_geometry" are not preserved and they should
    be computed out from this function.

    Note: there is no check for disconnected cells in the final grid.

    Parameters:
    g: the grid
    subdiv: a list of flags, one for each cell of the original grid

    How to use:
    subdiv = np.array([0,0,1,1,1,1,3,4,6,4,6,4])
    g = generate_coarse_grid( g, subdiv )

    """

    subdiv = np.asarray( subdiv )
    assert( subdiv.size == g.num_cells )

    # declare the storage array to build the cell_faces map
    cell_faces = np.empty(0, dtype=g.cell_faces.indptr.dtype)
    cells = np.empty(0, dtype=cell_faces.dtype)
    orient = np.empty(0, dtype=g.cell_faces.data.dtype)

    # declare the storage array to build the face_nodes map
    face_nodes = np.empty(0, dtype=g.face_nodes.indptr.dtype)
    nodes = np.empty(0, dtype=face_nodes.dtype)
    visit = np.zeros(g.num_faces, dtype=np.bool)

    # compute the face_node indexes
    num_nodes_per_face = g.face_nodes.indptr[1:] - g.face_nodes.indptr[:-1]
    face_node_ind = matrix_compression.rldecode(np.arange(g.num_faces), \
                                                num_nodes_per_face)

    cells_list = np.unique( subdiv )
    for cellId, cell in enumerate( cells_list ):
        # extract the cells of the original mesh associated to a specific label
        cells_old = np.where( subdiv == cell )[0]

        # reconstruct the cell_faces mapping
        faces_old, _, orient_old = sps.find( g.cell_faces[:, cells_old] )
        mask = np.ones( faces_old.size, dtype=np.bool )
        mask[ np.unique( faces_old, return_index=True )[1] ] = False
        # extract the indexes of the internal edges, to be discared
        index = np.array([ np.where( faces_old == f )[0] \
                                              for f in faces_old[mask]]).ravel()
        faces_new = np.delete( faces_old, index )
        cell_faces = np.r_[ cell_faces, faces_new ]
        cells = np.r_[ cells, np.repeat( cellId, faces_new.shape[0] ) ]
        orient = np.r_[ orient, np.delete( orient_old, index ) ]

        # reconstruct the face_nodes mapping
        # consider only the unvisited faces
        not_visit = ~visit[faces_new]
        if not_visit.size == 0: continue
        # mask to consider only the external faces
        mask = np.sum( [ face_node_ind == f for f in faces_new[not_visit] ], \
                        axis = 0, dtype = np.bool )
        face_nodes = np.r_[ face_nodes, face_node_ind[ mask ] ]
        nodes_new = g.face_nodes.indices[ mask ]
        nodes = np.r_[ nodes, nodes_new ]
        visit[faces_new] = True

    # Rename the faces
    cell_faces_unique = np.unique(cell_faces)
    cell_faces_id = np.arange(cell_faces_unique.size, dtype=cell_faces.dtype)
    cell_faces = np.array([cell_faces_id[np.where( cell_faces_unique == f )[0]]\
                                                   for f in cell_faces]).ravel()

    shape = (cell_faces_unique.size, cells_list.size)
    cell_faces =  sps.csc_matrix((orient, (cell_faces, cells)), shape = shape)

    # Rename the nodes
    face_nodes = np.array([cell_faces_id[np.where( cell_faces_unique == f )[0]]\
                                                   for f in face_nodes]).ravel()
    nodes_list = np.unique(nodes)
    nodes_id = np.arange(nodes_list.size, dtype=nodes.dtype)
    nodes = np.array([nodes_id[np.where( nodes_list == n )[0]] \
                                                        for n in nodes]).ravel()

    # sort the nodes
    nodes = nodes[ np.argsort(face_nodes,kind='mergesort') ]
    data = np.ones(nodes.size, dtype=g.face_nodes.data.dtype)
    indptr = np.r_[0, np.cumsum( np.bincount( face_nodes ) )]
    face_nodes =  sps.csc_matrix(( data, nodes, indptr ))

    name = g.name
    name.append( "coarse" )
    return Grid( g.dim, g.nodes[:,nodes_list], face_nodes, cell_faces, name )

#------------------------------------------------------------------------------#

def tpfa_matrix( g, perm = None ):
    if perm is None:
        perm = second_order_tensor.SecondOrderTensor(g.dim,np.ones(g.num_cells))
    bound = bc.BoundaryCondition(g, np.empty(0), '')
    trm = tpfa.tpfa(g, perm, bound)
    div = g.cell_faces.T
    return div * trm
