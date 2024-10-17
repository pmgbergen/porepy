"""This module contains methods to coarsen grids.

The main function is :func:`~porepy.grids.coarsening.coarsen`
(see there for more information).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids import grid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.utils import accumarray, grid_utils, mcolon, setmembership, tags


def coarsen(
    g: Union[pp.Grid, pp.MixedDimensionalGrid], method: str, **method_kwargs
) -> None:
    """Create a coarse grid from a given grid.

    If a md-grid is passed, the procedure is applied to the grid of highest dimension.

    Notes:

        - The passed grid is modified.
        - Do not call :meth:`~porepy.grids.grid.Grid.compute_geometry` afterwards.

    Parameters:
        g: The grid or mixed-dimensional grid to be coarsened.
        method: A string defining the coarsening method.

            The available options are:

            - ``'by_volume'``: The coarsening is based on the cell volume.
            - ``'by_tpfa'``: Uses the AMG method's coarse/fine-splittings based on
              direct couplings

        method_kwargs: Arguments for each ``method``.

            For the ``'by_volume'``- method, see :func:`create_aggregations` for an
            overview on admissible keyword arguments.

            For the ``'by_tpfa'``- method, see :func:`create_partition`.
            Additionally, a keyword argument ``if_seeds`` of boolean type is supported.
            If ``True``, :func:`generate_seeds` is called to create seeds.

    Raises:
        ValueError: If ``method`` is not among the supported options.

    """
    msg = "This functionality is deprecated and will be removed in a future version"
    warnings.warn(msg, DeprecationWarning)

    if method.lower() == "by_volume":
        partition = create_aggregations(g, **method_kwargs)

    elif method.lower() == "by_tpfa":
        seeds = np.empty(0, dtype=int)
        if method_kwargs.get("if_seeds", False):
            seeds = generate_seeds(g)
        matrix = _tpfa_matrix(g)
        partition = create_partition(
            matrix, g, seeds=seeds, **method_kwargs
        )  # type: ignore

    else:
        raise ValueError(f"Undefined method `{method}` for coarsening algorithm.")

    generate_coarse_grid(g, partition)  # type: ignore


def generate_coarse_grid(
    g: Union[pp.Grid, pp.MixedDimensionalGrid],
    subdiv: Union[np.ndarray, dict[pp.Grid, tuple[Any, np.ndarray]]],
) -> None:
    """Generates a coarse grid by clustering the cells according to the flags
    given by ``subdiv``.

    ``subdiv`` must be as long as the number of cells in the original grid, it contains
    integers (possibly not continuous) which represent the cell IDs in the final,
    coarser mesh. I.e. it is a cell map from finer to coarser.

    If ``g`` is a mixed-dimensional grid, the coarsening is applied to the higher
    dimensional grid.

    Warning:
        This method effectively overwrites every grid property computed by
        :meth:`~porepy.grids.grid.Grid.compute_geometry`. Do not call that method after
        calling this one.

    Note:
        There is no check for disconnected cells in the final grid.

    Example:

        >>> g = ...  # some grid with 12 cells
        >>> subdiv = np.array([0,0,1,1,2,2,3,3,4,4,5,5])  # coarser grid with 6 cells
        >>> generate_coarse_grid(g, subdiv)

    Parameters:
        g: A grid or mixed-dimensional grid.
        subdiv: If ``g`` is a single grid, a single array-like object as in above
            example suffices.

            If ``g`` is a mixed-dimensional grid, a dictionary containing per grid (key)
            a 2-tuple, where the second entry is the partition map as seen above.
            This special structure is passed by :func:`coarsen`.

    """
    msg = "This functionality is deprecated and will be removed in a future version"
    warnings.warn(msg, DeprecationWarning)

    if isinstance(g, grid.Grid):
        if isinstance(subdiv, dict):
            # If the subdiv is a dictionary with g as a key (this can happen if we are
            # forwarded here from coarsen), the input must be simplified.
            subdiv = subdiv[g][1]
        _generate_coarse_grid_single(g, subdiv, False)

    if isinstance(g, pp.MixedDimensionalGrid):
        _generate_coarse_grid_mdg(g, subdiv)


def reorder_partition(
    subdiv: Union[np.ndarray, dict[Any, tuple[Any, np.ndarray]]],
) -> Union[np.ndarray, dict[Any, tuple[Any, np.ndarray]]]:
    """Re-order the partition IDs in order to obtain contiguous numbers.

    Parameters:
        subdiv: A subdivision/partition as an array containing an ID for each cell, or
            a dictionary containing the previous in a 2-tuple for any key.

    Return:
        The subdivision stored in a contiguous way.

    """
    if isinstance(subdiv, dict):
        for _, (_, partition) in subdiv.items():
            old_ids = np.unique(partition)
            for new_id, old_id in enumerate(old_ids):
                partition[partition == old_id] = new_id
    else:
        old_ids = np.unique(subdiv)
        for new_id, old_id in enumerate(old_ids):
            subdiv[subdiv == old_id] = new_id

    return subdiv


def generate_seeds(mdg: Union[pp.Grid, pp.MixedDimensionalGrid]) -> np.ndarray:
    """Generates a priory seeds (cells) for coarsening a mixed-dimensional grid based
    on topological information about the highest-dimensional grid.

    Parameters:
        mdg: A grid or mixed-dimensional grid.

    Returns:
        If ``mdg`` is a single grid, this function returns an empty array.

        If ``mdg`` is a mixed-dimensional grid, this function returns an initial seed
        for the coarsening based on the mortar projections between the grid of highest
        dimension and grids of co-dimension 1.

    """
    seeds = np.empty(0, dtype=int)

    if isinstance(mdg, grid.Grid):
        return seeds

    # Extract the higher dimensional grid
    g_h = mdg.subdomains(dim=mdg.dim_max())[0]
    g_h_faces, g_h_cells, _ = sparse_array_to_row_col_data(g_h.cell_faces)

    # Extract the 1-codimensional grids
    gs = mdg.subdomains(dim=mdg.dim_max() - 1)

    for g in gs:
        tips = np.where(g.tags["tip_faces"])[0]
        faces, cells, _ = sparse_array_to_row_col_data(g.cell_faces)
        index = np.isin(faces, tips).nonzero()[0]
        cells = np.unique(cells[index])

        # recover the mapping between the secondary and the primary grid
        mg = mdg.subdomain_pair_to_interface((g_h, g))
        primary_to_mortar = mg.primary_to_mortar_int()
        secondary_to_mortar = mg.secondary_to_mortar_int()
        # this is the old face_cells mapping
        face_cells = secondary_to_mortar.T * primary_to_mortar

        interf_cells, interf_faces, _ = sparse_array_to_row_col_data(face_cells)
        index = np.isin(interf_cells, cells).nonzero()[0]
        index = np.isin(g_h_faces, interf_faces[index]).nonzero()[0]
        seeds = np.concatenate((seeds, g_h_cells[index]))

    return seeds


def create_aggregations(
    grid: Union[pp.Grid, pp.MixedDimensionalGrid], **kwargs
) -> dict[pp.Grid, tuple[pp.Grid, np.ndarray]]:
    """Creates a cell partition based on their volumes.

    Parameters:
        grid: A single grid or mixed-dimensional grid.
        **kwargs: Following keyword arguments are supported:

            - ``'weight'``: A float serving as weight for the mean of the cell volumes.
              Defaults to 1.

    Returns:
        A dictionary containing a partition per grid.

    """
    # Extract the higher dimensional grids and store in a list
    if isinstance(grid, pp.MixedDimensionalGrid):
        grid_list: list[pp.Grid] = grid.subdomains(dim=grid.dim_max())
    elif isinstance(grid, pp.Grid):
        grid_list = [grid]
    else:
        raise ValueError("Only subdomains and mixed-dimensional grids supported.")

    partition: dict[pp.Grid, tuple[pp.Grid, np.ndarray]] = {}

    for g in grid_list:
        partition_local = -np.ones(g.num_cells, dtype=int)

        volumes = g.cell_volumes.copy()
        volumes_checked = volumes.copy()
        c2c = g.cell_connection_map()

        # Compute the inverse of the harmonic mean
        weight = kwargs.get("weight", 1.0)
        mean = weight * np.mean(volumes)

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
                volumes_checked[:] = np.inf
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

        partition[g] = (g.copy(), partition_local)

    return partition


def create_partition(
    A: sps.spmatrix,
    g: Union[pp.Grid, pp.MixedDimensionalGrid],
    seeds: Optional[np.ndarray] = None,
    **kwargs,
) -> dict[pp.Grid, tuple[pp.Grid, np.ndarray]]:
    """Create the partition based on an input matrix using the AMG
    method's coarse/fine-splittings based on direct couplings.

    The standard values for ``cdepth`` and ``epsilon`` are taken from the reference
    below.

    Example:

        >>> part = create_partition(tpfa_matrix(g))
        >>> g = generate_coarse_grid(g, part)

    References:
        U. Trottenberg, C. W. Oosterlee, and A. Schuller (200):
        Multigrid, Academic press.

    Parameters:
        A: A sparse matrix used for the agglomeration.
        g: A single grid or mixed-dimensional grid.
        seeds: ``default=None``

            A-priory defined cells of the coarser grid.
        **kwargs: The following keyword arguments are supported:

            - ``'cdepth'``: A number to define the strength of the aggregation, i.e. a
              a greater number results in lesser cells. Defaults to 2.
            - ``'epsilon'``: A float representing the weight for the off-diagonal
              entries to define the *strong negative coupling*. Defaults to 0.25.

    Returns:
        A dictionary containing the a 2-tuple per grid. The 2-tuple contains the grid
        with the highest dimension and the map from finer to coarser grid containing as
        an array of agglomeration indices.

        If ``g`` is a single grid, the grid of highest dimension is ``g`` itself.

    """

    cdepth = int(kwargs.get("cdepth", 2))
    epsilon = kwargs.get("epsilon", 0.25)

    # NOTE: Extract the higher dimensional grids, we suppose it is unique
    if isinstance(g, pp.MixedDimensionalGrid):
        g_high = g.subdomains(dim=g.dim_max())[0]
    else:
        g_high = g

    if A.size == 0:
        return {g_high: (g_high.copy(), np.zeros(1))}

    Nc = A.shape[0]

    # For each node, which other nodes are strongly connected to it
    ST = sps.lil_matrix((Nc, Nc), dtype=bool)

    # In the first instance, all cells are strongly connected to each other
    At = A.T

    for i in np.arange(Nc):
        loc = slice(At.indptr[i], At.indptr[i + 1])
        ci, vals = At.indices[loc], At.data[loc]
        neg = vals < 0.0
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

    ST.setdiag(False)
    lmbda = np.array([len(s) for s in ST.rows])

    # Define coarse nodes
    candidate = np.ones(Nc, dtype=bool)
    is_fine = np.zeros(Nc, dtype=bool)
    is_coarse = np.zeros(Nc, dtype=bool)

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
    c2c_rows, _, _ = sparse_array_to_row_col_data(c2c.transpose())

    pairs = np.empty((0, 2), dtype=int)
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
            A_val = A[ij, ij].A.ravel()
            ids = ij[np.argsort(A_val)]
            if seeds is not None:
                ids = np.setdiff1d(ids, seeds, assume_unique=True)
            if ids.size:
                is_coarse[ids[0]] = False
                is_fine[ids[0]] = True

    coarse = np.where(is_coarse)[0]

    # Primal grid
    NC = coarse.size
    primal = sps.lil_matrix((NC, Nc), dtype=bool)
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
    partition = {g_high: (g_high.copy(), coarse[np.argsort(fine)])}
    return partition


def _generate_coarse_grid_single(
    g: pp.Grid, subdiv: np.ndarray, face_map: bool
) -> Union[np.ndarray, None]:
    """Auxiliary function to create a coarsening for a given single grid.

    Parameters:
        g: A single-dimensional grid.
        subdiv: A coarsening map containing cell IDs of the coarser grid for each ID
            in the finer grid
        face_map: A bool indicating if the face map for the coarser grid should be
            returned

    Returns:
        If ``face_map`` is True, returns the cell-to-face map of the coarser grid.

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
    visit = np.zeros(g.num_faces, dtype=bool)

    # compute the face_node indexes
    num_nodes_per_face = g.face_nodes.indptr[1:] - g.face_nodes.indptr[:-1]
    face_node_ind = pp.matrix_operations.rldecode(
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
        faces_old, _, orient_old = sparse_array_to_row_col_data(
            g.cell_faces[:, cells_old]
        )
        mask = np.ones(faces_old.size, dtype=bool)
        mask[np.unique(faces_old, return_index=True)[1]] = False
        # extract the indexes of the internal edges, to be discared
        index = np.array(
            [np.where(faces_old == f)[0] for f in faces_old[mask]], dtype=int
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
                dtype=bool,
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

    g.name = "Coarsened grid"
    # store again the data in the same grid
    g.history.append("coarse")

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
    g.cell_centers = grid_utils.star_shape_cell_centers(g, as_nan=True)
    is_nan = np.isnan(g.cell_centers[0, :])
    g.cell_centers[:, is_nan] = cell_centers[:, is_nan]

    if face_map:
        return np.array([cell_faces_unique, cell_faces_id])
    else:
        # Explicitly return None to make mypy happy.
        return None


def _generate_coarse_grid_mdg(
    mdg: pp.MixedDimensionalGrid,
    subdiv: Union[np.ndarray, dict[pp.Grid, tuple[pp.Grid, np.ndarray]]],
):
    """Auxiliary function to create a coarsening for a given mixed-dimensional grid.

    Parameters:
        mdg: A mixed-dimensional grid.
        subdiv: A subdivision containing the coarsening map for each grid in a 2-tuple.

    """

    if not isinstance(subdiv, dict):
        g = mdg.subdomains(dim=mdg.dim_max())[0]
        subdiv = {g: subdiv}  # type: ignore

    for g, (_, partition) in subdiv.items():
        # Construct the coarse grids
        face_map = _generate_coarse_grid_single(g, partition, True)
        assert face_map is not None  # make mypy happy

        # Update all the primary_to_mortar_int for all the 'edges' connected to the grid
        # We update also all the face_cells
        for intf in mdg.subdomain_to_interfaces(g):
            # The indices that need to be mapped to the new grid

            data = mdg.interface_data(intf)

            projections = [
                intf.primary_to_mortar_int().tocsr(),
                intf.primary_to_mortar_avg().tocsr(),
            ]

            for ind, mat in enumerate(projections):
                indices = mat.indices

                # Map indices
                mask = np.argsort(indices)
                indices = np.isin(face_map[0, :], indices[mask]).nonzero()[0]
                # Reverse the ordering
                indices = indices[np.argsort(mask)]

                # Create the new matrix
                shape = (mat.shape[0], g.num_faces)
                projections[ind] = sps.csr_matrix(
                    (mat.data, indices, mat.indptr), shape=shape
                )

            # Update mortar projection
            intf._primary_to_mortar_int = projections[0].tocsc()
            intf._primary_to_mortar_avg = projections[1].tocsc()

            # Also update all other projections to primary
            intf._set_projections(secondary=False)

            # update also the face_cells map
            face_cells = data["face_cells"].tocsr()
            indices = face_cells.indices

            # map indices
            mask = np.argsort(indices)
            indices = np.isin(face_map[0, :], indices[mask]).nonzero()[0]
            face_cells.indices = indices[np.argsort(mask)]

            # update the map
            data["face_cells"] = face_cells.tocsc()


def _tpfa_matrix(
    g: Union[pp.Grid, pp.MixedDimensionalGrid],
    perm: Optional[pp.SecondOrderTensor] = None,
) -> sps.spmatrix:
    """Compute a two-point flux approximation for a given grid

    This is a helper method for :func:`create_partition`.

    Parameters:
        g: A single grid or mixed-dimensional grid.
        perm: ``default=None``

            The permeability as a tensor. If not given, defaults to the unit tensor.

    Returns:
        The TPFA matrix for given grid and permeability.

    """
    if isinstance(g, pp.MixedDimensionalGrid):
        g = g.subdomains(dim=g.dim_max())[0]

    if perm is None:
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))

    solver = pp.Tpfa("flow")
    specified_parameters = {
        "second_order_tensor": perm,
        "bc": pp.BoundaryCondition(g, np.empty(0), ""),
    }
    data = pp.initialize_default_data(g, {}, "flow", specified_parameters)
    solver.discretize(g, data)
    flux, _ = solver.assemble_matrix_rhs(g, data)
    return flux


def __get_neigh(
    cells_id: np.ndarray, c2c: sps.spmatrix, partition: np.ndarray
) -> np.ndarray:
    """An auxiliary function for :func:`create_aggregations` to get neighbouring cells.

    Parameters:
        cells_id: An array containing cell IDs.
        c2c: A sparse map between cells sharing a face.
        partition: A partition map containing indices of the coarser grid for each cell
            in the finer grid.

    Returns:
        Neighbouring cell IDs in the ``partition``.

    """
    neighbors = np.empty(0, dtype=int)

    for cell_id in np.atleast_1d(cells_id):
        # Extract the neighbors of the current cell
        loc = slice(c2c.indptr[cell_id], c2c.indptr[cell_id + 1])
        neighbors = np.hstack((neighbors, c2c.indices[loc]))

    neighbors = np.unique(neighbors)
    partition_neighbors = partition[neighbors]

    # Check if some neighbor has already a coarse id
    return np.sort(neighbors[partition_neighbors < 0])
