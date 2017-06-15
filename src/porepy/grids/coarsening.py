# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps

from porepy.grids import grid, grid_bucket
from porepy.params import second_order_tensor, bc

from porepy.utils import matrix_compression, mcolon, accumarray, setmembership

from porepy.numerics.fv import tpfa

#------------------------------------------------------------------------------#

def generate_coarse_grid(g, subdiv):
    """ Generate a coarse grid clustering the cells according to the flags
    given by subdiv. Subdiv should be long as the number of cells in the
    original grid, it contains integers (possibly not continuous) which
    represent the cells in the final mesh. If a grid bucket is given the
    coarsening is applied to the higher dimensional grid.

    The values computed in "compute_geometry" are not preserved and they should
    be computed out from this function.

    Note: there is no check for disconnected cells in the final grid.
    Note: the return is different if a grid or a grid bucket is given.

    Parameters:
        g: the grid or grid bucket
        subdiv: a list of flags, one for each cell of the original grid

    Return:
        grid: if a grid is given as input, its coarser version is returned.
        If a grid bucket is given as input, the grid is updated in place.

    How to use:
    subdiv = np.array([0,0,1,1,1,1,3,4,6,4,6,4])
    g = generate_coarse_grid(g, subdiv)

    or with a grid bucket:
    subdiv = np.array([0,0,1,1,1,1,3,4,6,4,6,4])
    generate_coarse_grid(gb, subdiv)

    """
    if isinstance(g, grid.Grid):
        return generate_coarse_grid_single(g, subdiv, False)

    if isinstance(g, grid_bucket.GridBucket):
        generate_coarse_grid_gb(g, subdiv)

#------------------------------------------------------------------------------#

def generate_coarse_grid_single(g, subdiv, face_map):
    """
    Specific function for a single grid. Use the common interface instead.
    """

    subdiv = np.asarray(subdiv)
    assert subdiv.size == g.num_cells

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

    cells_list = np.unique(subdiv)
    for cellId, cell in enumerate(cells_list):
        # extract the cells of the original mesh associated to a specific label
        cells_old = np.where(subdiv == cell)[0]

        # reconstruct the cell_faces mapping
        faces_old, _, orient_old = sps.find(g.cell_faces[:, cells_old])
        mask = np.ones(faces_old.size, dtype=np.bool)
        mask[ np.unique(faces_old, return_index=True )[1]] = False
        # extract the indexes of the internal edges, to be discared
        index = np.array([ np.where( faces_old == f )[0] \
                                for f in faces_old[mask]], dtype=np.int).ravel()
        faces_new = np.delete( faces_old, index )
        cell_faces = np.r_[ cell_faces, faces_new ]
        cells = np.r_[ cells, np.repeat( cellId, faces_new.shape[0] ) ]
        orient = np.r_[ orient, np.delete( orient_old, index ) ]

        # reconstruct the face_nodes mapping
        # consider only the unvisited faces
        not_visit = ~visit[faces_new]
        if not_visit.size == 0:
            continue
        # mask to consider only the external faces
        mask = np.atleast_1d(np.sum([face_node_ind == f \
                                     for f in faces_new[not_visit]], \
                                    axis=0, dtype=np.bool))
        face_nodes = np.r_[face_nodes, face_node_ind[mask]]
        nodes_new = g.face_nodes.indices[mask]
        nodes = np.r_[nodes, nodes_new]
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
    g_co = grid.Grid(g.dim, g.nodes[:, nodes_list], face_nodes, cell_faces, name)

    if face_map:
        return g_co, np.array([cell_faces_unique, cell_faces_id])

    return g_co

#------------------------------------------------------------------------------#

def generate_coarse_grid_gb(gb, subdiv):
    """
    Specific function for a grid bucket. Use the common interface instead.
    """

    # Extract the higher dimensional grids
    # NOTE: we assume only one high dimensional grid
    g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]

    # Construct the coarse grids
    g_co, face_map = generate_coarse_grid_single(g, subdiv, True)

    # Update all the face_cells for all the 'edges' connected to the grid
    for e, d in gb.edges_props_of_node(g):
        # The indices that need to be mapped to the new grid
        face_cells = d['face_cells'].tocsr()
        indices = face_cells.indices
        # Map indices
        mask = np.argsort(indices)
        indices = np.in1d(face_map[0, :], indices[mask]).nonzero()[0]
        # Reverse the ordering
        face_cells.indices = indices[np.argsort(mask)]
        d['face_cells'] = face_cells.tocsc()

    gb.update_nodes(g, g_co)

#------------------------------------------------------------------------------#

def tpfa_matrix(g, perm=None, faces=None):
    """
    Compute a two-point flux approximation matrix useful related to a call of
    create_partition.

    Parameters
    ----------
    g: the grid
    perm: (optional) permeability, the it is not given unitary tensor is assumed
    faces (np.array, int): Index of faces where TPFA should be applied.
            Defaults all faces in the grid.

    Returns
    -------
    out: sparse matrix
        Two-point flux approximation matrix

    """
    if isinstance(g, grid_bucket.GridBucket):
       g = g.get_grids(lambda g_: g_.dim == g.dim_max())[0]

    if perm is None:
        perm = second_order_tensor.SecondOrderTensor(g.dim,np.ones(g.num_cells))

    bound = bc.BoundaryCondition(g, np.empty(0), '')
    trm, _ = tpfa.tpfa(g, perm, bound, faces)
    div = g.cell_faces.T
    return div * trm

#------------------------------------------------------------------------------#

def generate_seeds(gb):
    # Extract the higher dimensional grid
    g_h = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
    g_h_faces, g_h_cells, _ = sps.find(g_h.cell_faces)

    # Extract the 1-codimensional grids
    gs = gb.get_grids(lambda g: g.dim == gb.dim_max()-1)

    seeds = np.empty(0, dtype=np.int)
    for g in gs:
        tips = np.where(g.has_face_tag(grid.FaceTag.TIP))[0]
        faces, cells, _ = sps.find(g.cell_faces)
        index = np.in1d(faces, tips).nonzero()[0]
        cells = np.unique(cells[index])

        face_cells = gb.graph.edge[g][g_h]['face_cells']
        interf_cells, interf_faces, _ = sps.find(face_cells)
        index = np.in1d(interf_cells, cells).nonzero()[0]

        index = np.in1d(g_h_faces, interf_faces[index]).nonzero()[0]
        seeds = np.concatenate((seeds, g_h_cells[index]))

    return seeds

#------------------------------------------------------------------------------#

def create_partition(A, cdepth=2, epsilon=0.25, seeds=None):
    """
    Create the partition based on an input matrix using the algebraic multigrid
    method coarse/fine-splittings based on direct couplings. The standard values
    for cdepth and epsilon are taken from the following reference.

    For more information see: U. Trottenberg, C. W. Oosterlee, and A. Schuller.
    Multigrid. Academic press, 2000.

    Parameters
    ----------
    A: sparse matrix used for the agglomeration
    cdepth: the greather is the more intense the aggregation will be, e.g. less
        cells if it is used combined with generate_coarse_grid
    epsilon: weight for the off-diagonal entries to define the "strong
        negatively cupling"
    seeds: (optional) to define a-priori coarse cells

    Returns
    -------
    out: agglomeration indices

    How to use
    ----------
    part = create_partition(tpfa_matrix(g))
    g = generate_coarse_grid(g, part)

    """

    if A.size == 0: return np.zeros(1)
    Nc = A.shape[0]

    # For each node, which other nodes are strongly connected to it
    ST = sps.lil_matrix((Nc,Nc),dtype=np.bool)

    # In the first instance, all cells are strongly connected to each other
    At = A.T

    for i in np.arange(Nc):
        ci, _, vals = sps.find(At[:,i])
        neg = vals < 0.
        nvals = vals[neg]
        nci = ci[neg]
        minId = np.argmin(nvals)
        ind = -nvals >= epsilon * np.abs(nvals[minId])
        ST[nci[ind], i] = True

    # Temporary field, will store connections of depth 1
    STold = ST.copy()
    for _ in np.arange(2, cdepth+1):
        for j in np.arange(Nc):
            rowj = np.array(STold.rows[j])
            row = np.hstack([STold.rows[r] for r in rowj])
            ST[j, np.concatenate((rowj, row))] = True
        STold = ST.copy()

    del STold

    ST.setdiag(False)
    lmbda = np.array([len(s) for s in ST.rows])

    # Define coarse nodes
    candidate = np.ones(Nc, dtype=np.bool)
    is_fine = np.zeros(Nc, dtype=np.bool)
    is_coarse = np.zeros(Nc, dtype=np.bool)

    # cells that are not important for any other cells are on the fine scale.
    for row_id, row in enumerate(ST.rows):
        if not row:
            is_fine[row_id] = True
            candidate[row_id] = False

    ST = ST.tocsr()
    it = 0
    while np.any(candidate):
        i = np.argmax(lmbda)
        is_coarse[i] = True
        j = ST.indices[ST.indptr[i]:ST.indptr[i+1]]
        jf = j[candidate[j]]
        is_fine[jf] = True
        candidate[np.r_[i, jf]] = False
        loop = ST.indices[ mcolon.mcolon(ST.indptr[jf], ST.indptr[jf+1]) ]
        for row in np.unique(loop):
            s = ST.indices[ST.indptr[row]:ST.indptr[row+1]]
            lmbda[row] = s[candidate[s]].size + 2*s[is_fine[s]].size
        lmbda[np.logical_not(candidate)]= -1
        it = it + 1

        # Something went wrong during aggregation
        assert it <= Nc

    del lmbda, ST

    if seeds is not None:
        is_coarse[seeds] = True
        is_fine[seeds] = False

    # If two neighbors are coarse, eliminate one of them
    c2c = np.abs(A) > 0
    c2c_rows, _, _ = sps.find(c2c)

    pairs = np.empty((2,0), dtype=np.int)
    for idx, it in enumerate(np.where(is_coarse)[0]):
        loc = slice(c2c.indptr[it], c2c.indptr[it+1])
        ind = np.setdiff1d(c2c_rows[loc], it)
        cind = ind[is_coarse[ind]]
        new_pair = np.stack((np.repeat(it, cind.size), cind))
        pairs = np.append(pairs, new_pair, axis=1)

    if pairs.size:
        pairs = setmembership.unique_columns_tol(np.sort(pairs, axis=0),
                                                 axis=1)
        for ij in pairs.T:
            mi = np.argmin(A[ij, ij])
            is_coarse[ij[mi]] = False
            is_fine[ij[mi]] = True

    coarse = np.where(is_coarse)[0]

    # Primal grid
    NC = coarse.size
    primal = sps.lil_matrix((NC,Nc),dtype=np.bool)
    for i in np.arange(NC):
        primal[i, coarse[i]] = True

    connection = sps.lil_matrix((Nc,Nc),dtype=np.double)
    for it in np.arange(Nc):
        n = np.setdiff1d(c2c_rows[c2c.indptr[it]:c2c.indptr[it+1]], it)
        connection[it, n] = np.abs(A[it, n] / At[it, it])

    connection = connection.tocsr()

    candidates_rep = np.ediff1d(connection.indptr)
    candidates_idx = np.repeat(is_coarse, candidates_rep)
    candidates = np.stack((connection.indices[candidates_idx],
                           np.repeat(np.arange(NC), candidates_rep[is_coarse])),
                           axis=-1)

    connection_idx = mcolon.mcolon(connection.indptr[coarse],
                                   connection.indptr[coarse+1])
    vals = accumarray.accum(candidates, connection.data[connection_idx],
                            size=[Nc,NC])
    del candidates_rep, candidates_idx, connection_idx

    mcind = np.argmax(vals, axis=0)
    mcval = [ vals[r,c] for c,r in enumerate(mcind) ]

    it = NC
    not_found = np.logical_not(is_coarse)
    # Process the strongest connection globally
    while np.any(not_found):
        mi = np.argmax(mcval)
        nadd = mcind[mi]

        primal[mi, nadd] = True
        not_found[nadd] = False
        vals[nadd, :] *= 0

        nc = connection.indices[connection.indptr[nadd]:connection.indptr[nadd+1]]
        af = not_found[nc]
        nc = nc[af]
        nv = mcval[mi] * connection[nadd, :]
        nv = nv.data[af]
        if len(nc) > 0:
            vals += sps.csr_matrix((nv,(nc, np.repeat(mi,len(nc)))),
                                          shape=(Nc,NC)).todense()
        mcind = np.argmax(vals, axis=0)
        mcval = [ vals[r,c] for c,r in enumerate(mcind) ]

        it = it + 1
        if it > Nc + 5: break

    coarse, fine = primal.tocsr().nonzero()
    return coarse[np.argsort(fine)]

#------------------------------------------------------------------------------#
