# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps
import scipy.stats as stats

from porepy.grids import grid, grid_bucket

from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params.bc import BoundaryCondition


from porepy.utils import matrix_compression, mcolon, accumarray, setmembership
from porepy.utils import half_space, tags

from porepy.numerics.fv import tpfa


# ------------------------------------------------------------------------------#


def coarsen(g, method, **method_kwargs):
    """ Create a coarse grid from a given grid. If a grid bucket is passed the
    procedure is applied to the higher in dimension.
    Note: the grid is modified in place.
    Note: do not call compute_geometry afterward.
    Parameters:
        g: the grid or grid bucket
        method: string which define the method to coarse. Current options:
            'by_volume' (the coarsening is based on the cell volumes) or 'by_tpfa'
            (using the algebraic multigrid method coarse/fine-splittings based on
            direct couplings)
        method_kwargs: the arguments for each method

    """

    if method.lower() == "by_volume":
        partition = create_aggregations(g, **method_kwargs)

    elif method.lower() == "by_tpfa":
        seeds = np.empty(0, dtype=np.int)
        if method_kwargs.get("if_seeds", False):
            seeds = generate_seeds(g)
        matrix = tpfa_matrix(g)
        partition = create_partition(matrix, seeds=seeds, **method_kwargs)

    else:
        raise ValueError("Undefined coarsening algorithm")

    generate_coarse_grid(g, partition)


# ------------------------------------------------------------------------------#


def generate_coarse_grid(g, subdiv):
    """ Generate a coarse grid clustering the cells according to the flags
    given by subdiv. Subdiv should be long as the number of cells in the
    original grid, it contains integers (possibly not continuous) which
    represent the cells in the final mesh. If a grid bucket is given the
    coarsening is applied to the higher dimensional grid.

    The values computed in "compute_geometry" are not preserved and they should
    be computed out from this function.

    Note: there is no check for disconnected cells in the final grid.

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
        generate_coarse_grid_single(g, subdiv, False)

    if isinstance(g, grid_bucket.GridBucket):
        generate_coarse_grid_gb(g, subdiv)


# ------------------------------------------------------------------------------#


def reorder_partition(subdiv):
    """
    Re-order the partition id in case to obtain contiguous numbers.
    Parameters:
        subdiv: array where for each cell one id
    Return:
        the subdivision written in a contiguous way
    """
    if isinstance(subdiv, dict):
        for _, partition in subdiv.items():
            old_ids = np.unique(partition)
            for new_id, old_id in enumerate(old_ids):
                partition[partition == old_id] = new_id
    else:
        old_ids = np.unique(subdiv)
        for new_id, old_id in enumerate(old_ids):
            subdiv[subdiv == old_id] = new_id

    return subdiv


# ------------------------------------------------------------------------------#


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
    face_node_ind = matrix_compression.rldecode(
        np.arange(g.num_faces), num_nodes_per_face
    )

    cells_list = np.unique(subdiv)
    cell_volumes = np.zeros(cells_list.size)
    cell_centers = np.zeros((3, cells_list.size))

    for cellId, cell in enumerate(cells_list):
        # extract the cells of the original mesh associated to a specific label
        cells_old = np.where(subdiv == cell)[0]

        # compute the volume
        cell_volumes[cellId] = np.sum(g.cell_volumes[cells_old])
        cell_centers[:, cellId] = np.average(g.cell_centers[:, cells_old], axis=1)

        # reconstruct the cell_faces mapping
        faces_old, _, orient_old = sps.find(g.cell_faces[:, cells_old])
        mask = np.ones(faces_old.size, dtype=np.bool)
        mask[np.unique(faces_old, return_index=True)[1]] = False
        # extract the indexes of the internal edges, to be discared
        index = np.array(
            [np.where(faces_old == f)[0] for f in faces_old[mask]], dtype=np.int
        ).ravel()
        faces_new = np.delete(faces_old, index)
        cell_faces = np.r_[cell_faces, faces_new]
        cells = np.r_[cells, np.repeat(cellId, faces_new.shape[0])]
        orient = np.r_[orient, np.delete(orient_old, index)]

        # reconstruct the face_nodes mapping
        # consider only the unvisited faces
        not_visit = ~visit[faces_new]
        if not_visit.size == 0 or np.all(~not_visit):
            continue
        # mask to consider only the external faces
        mask = np.atleast_1d(
            np.sum(
                [face_node_ind == f for f in faces_new[not_visit]],
                axis=0,
                dtype=np.bool,
            )
        )
        face_nodes = np.r_[face_nodes, face_node_ind[mask]]

        nodes_new = g.face_nodes.indices[mask]
        nodes = np.r_[nodes, nodes_new]
        visit[faces_new] = True

    # Rename the faces
    cell_faces_unique = np.unique(cell_faces)
    cell_faces_id = np.arange(cell_faces_unique.size, dtype=cell_faces.dtype)
    cell_faces = np.array(
        [cell_faces_id[np.where(cell_faces_unique == f)[0]] for f in cell_faces]
    ).ravel()
    shape = (cell_faces_unique.size, cells_list.size)
    cell_faces = sps.csc_matrix((orient, (cell_faces, cells)), shape=shape)

    # Rename the nodes
    face_nodes = np.array(
        [cell_faces_id[np.where(cell_faces_unique == f)[0]] for f in face_nodes]
    ).ravel()
    nodes_list = np.unique(nodes)
    nodes_id = np.arange(nodes_list.size, dtype=nodes.dtype)
    nodes = np.array([nodes_id[np.where(nodes_list == n)[0]] for n in nodes]).ravel()

    # sort the nodes
    nodes = nodes[np.argsort(face_nodes, kind="mergesort")]
    data = np.ones(nodes.size, dtype=g.face_nodes.data.dtype)
    indptr = np.r_[0, np.cumsum(np.bincount(face_nodes))]
    face_nodes = sps.csc_matrix((data, nodes, indptr))

    # store again the data in the same grid
    g.name.append("coarse")

    g.nodes = g.nodes[:, nodes_list]
    g.num_nodes = g.nodes.shape[1]

    g.face_nodes = face_nodes
    g.num_faces = g.face_nodes.shape[1]
    g.face_areas = g.face_areas[cell_faces_unique]
    g.tags = tags.extract(g.tags, cell_faces_unique, tags.standard_face_tags())
    g.face_normals = g.face_normals[:, cell_faces_unique]
    g.face_centers = g.face_centers[:, cell_faces_unique]

    g.cell_faces = cell_faces
    g.num_cells = g.cell_faces.shape[1]
    g.cell_volumes = cell_volumes
    g.cell_centers = half_space.star_shape_cell_centers(g)
    is_nan = np.isnan(g.cell_centers[0, :])
    g.cell_centers[:, is_nan] = cell_centers[:, is_nan]

    if face_map:
        return np.array([cell_faces_unique, cell_faces_id])


# ------------------------------------------------------------------------------#


def generate_coarse_grid_gb(gb, subdiv):
    """
    Specific function for a grid bucket. Use the common interface instead.
    """

    if not isinstance(subdiv, dict):
        g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
        subdiv = {g: subdiv}

    for g, partition in subdiv.items():

        # Construct the coarse grids
        face_map = generate_coarse_grid_single(g, partition, True)

        # Update all the face_cells for all the 'edges' connected to the grid
        for e, d in gb.edges_of_node(g):
            # The indices that need to be mapped to the new grid
            face_cells = d["face_cells"].tocsr()
            indices = face_cells.indices
            # Map indices
            mask = np.argsort(indices)
            indices = np.in1d(face_map[0, :], indices[mask]).nonzero()[0]
            # Reverse the ordering
            face_cells.indices = indices[np.argsort(mask)]
            d["face_cells"] = face_cells.tocsc()


# ------------------------------------------------------------------------------#


def tpfa_matrix(g, perm=None):
    """
    Compute a two-point flux approximation matrix useful related to a call of
    create_partition.

    Parameters
    ----------
    g: the grid
    perm: (optional) permeability, the it is not given unitary tensor is assumed

    Returns
    -------
    out: sparse matrix
        Two-point flux approximation matrix

    """
    if isinstance(g, grid_bucket.GridBucket):
        g = g.get_grids(lambda g_: g_.dim == g.dim_max())[0]

    if perm is None:
        perm = tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))

    solver = tpfa.Tpfa()
    param = Parameters(g)
    param.set_tensor(solver, perm)
    param.set_bc(solver, BoundaryCondition(g, np.empty(0), ""))
    return solver.matrix_rhs(g, {"param": param})[0]


# ------------------------------------------------------------------------------#


def generate_seeds(gb):
    """
    Giving the higher dimensional grid in a grid bucket, generate the seed for
    the tip of lower
    """
    seeds = np.empty(0, dtype=np.int)

    if isinstance(gb, grid.Grid):
        return seeds

    # Extract the higher dimensional grid
    g_h = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
    g_h_faces, g_h_cells, _ = sps.find(g_h.cell_faces)

    # Extract the 1-codimensional grids
    gs = gb.get_grids(lambda g: g.dim == gb.dim_max() - 1)

    for g in gs:
        tips = np.where(g.tags["tip_faces"])[0]
        faces, cells, _ = sps.find(g.cell_faces)
        index = np.in1d(faces, tips).nonzero()[0]
        cells = np.unique(cells[index])

        face_cells = gb.graph.adj[g][g_h]["face_cells"]
        interf_cells, interf_faces, _ = sps.find(face_cells)
        index = np.in1d(interf_cells, cells).nonzero()[0]

        index = np.in1d(g_h_faces, interf_faces[index]).nonzero()[0]
        seeds = np.concatenate((seeds, g_h_cells[index]))

    return seeds


# ------------------------------------------------------------------------------#


def create_aggregations(g, **kwargs):
    """ Create a cell partition based on their volumes.

    Parameter:
        g: grid or grid bucket

    Return:
        partition: partition of the cells for the coarsening algorithm

    """

    # Extract the higher dimensional grids
    if isinstance(g, grid_bucket.GridBucket):
        g = g.get_grids(lambda g_: g_.dim == g.dim_max())

    g_list = np.atleast_1d(g)
    partition = dict()

    for g in g_list:
        partition_local = -np.ones(g.num_cells, dtype=np.int)

        volumes = g.cell_volumes.copy()
        volumes_checked = volumes.copy()
        c2c = g.cell_connection_map()

        # Compute the inverse of the harminc mean
        weight = kwargs.get("weight", 1.)
        mean = weight / stats.hmean(1. / volumes)

        new_id = 1
        while np.any(partition_local < 0):
            # Consider the smallest element to be aggregated
            cell_id = np.argmin(volumes_checked)

            # If the smallest fulfil the condition, stop the loop
            if volumes[cell_id] > mean:
                break

            do_it = True
            old_cluster = np.array([cell_id])
            while do_it:
                cluster = __get_neigh(old_cluster, c2c, partition_local)
                volume = np.sum(volumes[cluster])
                if volume > mean or np.array_equal(old_cluster, cluster):
                    do_it = False
                else:
                    old_cluster = cluster

            # If one of the element in the cluster has already a partition id,
            # we uniform the ids
            partition_cluster = partition_local[cluster]
            has_coarse_id = partition_cluster > 0
            if np.any(has_coarse_id):
                # For each coarse id in the cluster, rename the coarse ids in the
                # partition_local
                for partition_id in partition_cluster[has_coarse_id]:
                    which_partition_id = partition_local == partition_id
                    partition_local[which_partition_id] = new_id
                    volumes[which_partition_id] = volume
                    volumes_checked[which_partition_id] = volume

            # Update the data for the cluster
            partition_local[cluster] = new_id
            volumes[cluster] = volume
            new_id += 1

            volumes_checked[cluster] = np.inf

        volumes_checked = volumes.copy()
        which_cell = volumes_checked < mean
        volumes_checked[np.logical_not(which_cell)] = np.inf

        while np.any(which_cell):
            cell_id = np.argmin(volumes_checked)
            part_cell = partition_local[cell_id]
            # Extract the neighbors of the current cell
            loc = slice(c2c.indptr[cell_id], c2c.indptr[cell_id + 1])
            neighbors = np.setdiff1d(c2c.indices[loc], np.asarray(cell_id))
            part_neighbors = partition_local[neighbors]
            neighbors = neighbors[part_neighbors != part_cell]
            if neighbors.size == 0:
                volumes_checked = np.inf
                which_cell = volumes_checked < mean
                continue
            smallest = np.argmin(volumes[neighbors])
            mask = partition_local == part_cell
            partition_local[mask] = partition_local[neighbors[smallest]]
            volumes[mask] = volumes[smallest] + volumes[cell_id]
            volumes_checked[mask] = volumes[smallest] + volumes[cell_id]
            which_cell = volumes_checked < mean

        # Fill up the cells which are left
        has_not_coarse_id = partition_local < 0
        partition_local[has_not_coarse_id] = new_id + np.arange(
            np.sum(has_not_coarse_id)
        )

        partition[g] = partition_local

    return partition


# ------------------------------------------------------------------------------#


def __get_neigh(cells_id, c2c, partition):
    """ Support function for create_aggregations
    """
    neighbors = np.empty(0, dtype=np.int)

    for cell_id in np.atleast_1d(cells_id):
        # Extract the neighbors of the current cell
        loc = slice(c2c.indptr[cell_id], c2c.indptr[cell_id + 1])
        neighbors = np.hstack((neighbors, c2c.indices[loc]))

    neighbors = np.unique(neighbors)
    partition_neighbors = partition[neighbors]

    # Check if some neighbor has already a coarse id
    return np.sort(neighbors[partition_neighbors < 0])


# ------------------------------------------------------------------------------#


def create_partition(A, seeds=None, **kwargs):
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

    cdepth = int(kwargs.get("cdepth", 2))
    epsilon = kwargs.get("epsilon", 0.25)

    if A.size == 0:
        return np.zeros(1)
    Nc = A.shape[0]

    # For each node, which other nodes are strongly connected to it
    ST = sps.lil_matrix((Nc, Nc), dtype=np.bool)

    # In the first instance, all cells are strongly connected to each other
    At = A.T

    for i in np.arange(Nc):
        loc = slice(At.indptr[i], At.indptr[i + 1])
        ci, vals = At.indices[loc], At.data[loc]
        neg = vals < 0.
        nvals = vals[neg]
        nci = ci[neg]
        minId = np.argmin(nvals)
        ind = -nvals >= epsilon * np.abs(nvals[minId])
        ST[nci[ind], i] = True

    # Temporary field, will store connections of depth 1
    for _ in np.arange(2, cdepth + 1):
        STold = ST.copy()
        for j in np.arange(Nc):
            rowj = np.array(STold.rows[j])
            if rowj.size == 0:
                continue
            row = np.hstack([STold.rows[r] for r in rowj])
            ST[j, np.concatenate((rowj, row))] = True

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
        j = ST.indices[ST.indptr[i] : ST.indptr[i + 1]]
        jf = j[candidate[j]]
        is_fine[jf] = True
        candidate[np.r_[i, jf]] = False
        loop = ST.indices[mcolon.mcolon(ST.indptr[jf], ST.indptr[jf + 1])]
        for row in np.unique(loop):
            s = ST.indices[ST.indptr[row] : ST.indptr[row + 1]]
            lmbda[row] = s[candidate[s]].size + 2 * s[is_fine[s]].size
        lmbda[np.logical_not(candidate)] = -1
        it = it + 1

        # Something went wrong during aggregation
        assert it <= Nc

    del lmbda, ST

    if seeds is not None:
        is_coarse[seeds] = True
        is_fine[seeds] = False

    # If two neighbors are coarse, eliminate one of them without touching the
    # seeds
    c2c = np.abs(A) > 0
    c2c_rows, _, _ = sps.find(c2c)

    pairs = np.empty((0, 2), dtype=np.int)
    for idx, it in enumerate(np.where(is_coarse)[0]):
        loc = slice(c2c.indptr[it], c2c.indptr[it + 1])
        ind = np.setdiff1d(c2c_rows[loc], it)
        cind = ind[is_coarse[ind]]
        new_pair = np.stack((np.repeat(it, cind.size), cind), axis=-1)
        pairs = np.append(pairs, new_pair, axis=0)

    # Remove one of the neighbors cells
    if pairs.size:
        pairs = setmembership.unique_rows(np.sort(pairs, axis=1))[0]
        for ij in pairs:
            A_val = np.array(A[ij, ij]).ravel()
            ids = ij[np.argsort(A_val)]
            ids = np.setdiff1d(ids, seeds, assume_unique=True)
            if ids.size:
                is_coarse[ids[0]] = False
                is_fine[ids[0]] = True

    coarse = np.where(is_coarse)[0]

    # Primal grid
    NC = coarse.size
    primal = sps.lil_matrix((NC, Nc), dtype=np.bool)
    primal[np.arange(NC), coarse[np.arange(NC)]] = True

    connection = sps.lil_matrix((Nc, Nc), dtype=np.double)
    for it in np.arange(Nc):
        n = np.setdiff1d(c2c_rows[c2c.indptr[it] : c2c.indptr[it + 1]], it)
        loc = slice(A.indptr[it], A.indptr[it + 1])
        A_idx, A_row = A.indices[loc], A.data[loc]
        mask = A_idx != it
        connection[it, n] = np.abs(A_row[mask] / A_row[np.logical_not(mask)])

    connection = connection.tocsr()

    candidates_rep = np.ediff1d(connection.indptr)
    candidates_idx = np.repeat(is_coarse, candidates_rep)
    candidates = np.stack(
        (
            connection.indices[candidates_idx],
            np.repeat(np.arange(NC), candidates_rep[is_coarse]),
        ),
        axis=-1,
    )

    connection_idx = mcolon.mcolon(
        connection.indptr[coarse], connection.indptr[coarse + 1]
    )
    vals = sps.csr_matrix(
        accumarray.accum(candidates, connection.data[connection_idx], size=[Nc, NC])
    )
    del candidates_rep, candidates_idx, connection_idx

    it = NC
    not_found = np.logical_not(is_coarse)
    # Process the strongest connection globally
    while np.any(not_found):

        np.argmax(vals.data)
        vals.argmax(axis=0)
        mcind = np.atleast_1d(np.squeeze(np.asarray(vals.argmax(axis=0))))
        mcval = -np.inf * np.ones(mcind.size)
        for c, r in enumerate(mcind):
            loc = slice(vals.indptr[r], vals.indptr[r + 1])
            vals_idx, vals_data = vals.indices[loc], vals.data[loc]
            mask = vals_idx == c
            if vals_idx.size == 0 or not np.any(mask):
                continue
            mcval[c] = vals_data[mask]

        mi = np.argmax(mcval)
        nadd = mcind[mi]

        primal[mi, nadd] = True
        it = it + 1
        if it > Nc + 5:
            break

        not_found[nadd] = False
        vals.data[vals.indptr[nadd] : vals.indptr[nadd + 1]] = 0

        loc = slice(connection.indptr[nadd], connection.indptr[nadd + 1])
        nc = connection.indices[loc]
        af = not_found[nc]
        nc = nc[af]
        nv = mcval[mi] * connection[nadd, :]
        nv = nv.data[af]
        if len(nc) > 0:
            vals += sps.csr_matrix((nv, (nc, np.repeat(mi, len(nc)))), shape=(Nc, NC))

    coarse, fine = primal.tocsr().nonzero()
    return coarse[np.argsort(fine)]


# ------------------------------------------------------------------------------#
