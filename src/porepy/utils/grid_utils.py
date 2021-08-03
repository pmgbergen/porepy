from typing import List, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["gridding", "grids"]


@pp.time_logger(sections=module_sections)
def merge_grids(grids: List[pp.Grid]) -> pp.Grid:
    """Merge the given list of grids into one grid. Quantities in the grids
    are appendded by the grid ordering of the input list. First all
    quantities related to grids[0], then all quantities related to grids[1], ...

    Paramemeters:
    grids (List [pp.Grid]): A list of porepy grids.

    Returns:
    pp.Grid: The merged grid
    """
    if len(grids) < 1:
        raise ValueError("Need at least one grid for merging")

    grid = grids[0].copy()
    if hasattr(grids[0], "cell_facetag"):
        grid.cell_facetag = grids[0].cell_facetag

    if len(grids) == 1:
        return grid

    for sg in grids[1:]:
        if sg.dim != grids[0].dim:
            raise ValueError(
                """All grids must have the same dimension. found grid of dimension {},
                expected dimension {}""".format(
                    sg.dim, grids[0].dim
                )
            )
        grid.num_cells += sg.num_cells
        grid.num_faces += sg.num_faces
        grid.num_nodes += sg.num_nodes
        grid.nodes = np.hstack((grid.nodes, sg.nodes))

        grid.face_nodes = pp.utils.sparse_mat.stack_diag(grid.face_nodes, sg.face_nodes)
        grid.cell_faces = pp.utils.sparse_mat.stack_diag(grid.cell_faces, sg.cell_faces)

        grid.face_areas = np.hstack((grid.face_areas, sg.face_areas))
        grid.face_centers = np.hstack((grid.face_centers, sg.face_centers))
        grid.face_normals = np.hstack((grid.face_normals, sg.face_normals))
        grid.cell_volumes = np.hstack((grid.cell_volumes, sg.cell_volumes))
        grid.cell_centers = np.hstack((grid.cell_centers, sg.cell_centers))

        if hasattr(grid, "cell_facetag"):
            grid.cell_facetag = np.hstack((grid.cell_facetag, sg.cell_facetag))

        for key in grid.tags.keys():
            grid.tags[key] = np.hstack((grid.tags[key], sg.tags[key]))

    return grid


def merge_grids_of_equal_dim(gb: pp.GridBucket) -> pp.GridBucket:
    """Merges all grids that have the same dimension in the GridBucket. Thus
    the returned GridBucket only has one grid per dimension. See also
    pp.utils.grid_utils.merge_grids(grids).

    Parameters:
    gb (pp.GridBucket): A Grid bucket with possible many grids per dimension

    Returns:
    mergedGb (pp.GridBucket): A grid bucket with the merged grids of gb, and
        updated projections and mortar grids.
    """
    dimMax = gb.dim_max()

    # First merge all grid nodes.
    mergedGrids: List[List[pp.Grid]] = []
    gridsOfDim = np.empty(dimMax + 1, dtype=list)
    gridIdx = np.empty(dimMax + 1, dtype=list)
    numCells = np.empty(dimMax + 1, dtype=list)
    for i in range(dimMax + 1):
        gridIdx[i] = []
        numCells[i] = []
        gridsOfDim[i] = gb.grids_of_dimension(i)
        if len(gridsOfDim[i]) == 0:
            mergedGrids.append([])
            continue

        mergedGrids.append(merge_grids(gridsOfDim[i]))
        # Store the node number of each merged node.
        # This is used to obtain the correct merged
        # mortar projections
        for grid in gridsOfDim[i]:
            d = gb.node_props(grid)
            gridIdx[i].append(d["node_number"])
            numCells[i].append(grid.num_cells)

    # Initiate mortar grids.
    mortarsOfDim = np.empty(dimMax + 1, dtype=list)
    for i in range(len(mortarsOfDim)):
        mortarsOfDim[i] = []

    # Collect all mortar grids in a list of list. list[i] contains
    # all mortar grids of dimension i.
    for e, d in gb.edges():
        mortar_grids = []
        mg = d["mortar_grid"]
        for sg in mg.side_grids.values():
            mortar_grids.append(sg)
        mortarsOfDim[mg.dim].append(merge_grids(mortar_grids))

    # Initialize mortar projections.
    primary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)
    secondary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)

    # Loop over all dimensions and initiate the mapping size
    for i in range(dimMax):
        primary2mortar[i] = np.empty(
            (len(mortarsOfDim[i]), len(gridsOfDim[i + 1])), dtype=object
        )
        secondary2mortar[i] = np.empty(
            (len(mortarsOfDim[i]), len(gridsOfDim[i])), dtype=object
        )

        # Add an empty grid for mortar row. This is to let the block matrices
        # mergedSecondary2Mortar and mergedPrimary2Mortar know the correct dimension
        # if there is an empty mapping. It should be sufficient to add zeros to
        # one of the mortar grids.
        for j in range(len(gridsOfDim[i + 1])):
            if len(mortarsOfDim[i]) == 0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridFaces = gridsOfDim[i + 1][j].num_faces
            primary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridFaces))

        for j in range(len(gridsOfDim[i])):
            if len(mortarsOfDim[i]) == 0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridCells = gridsOfDim[i][j].num_cells
            secondary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridCells))

    # Collect the mortar projections
    mortarPos = np.zeros(dimMax + 1, dtype=int)
    for e, d in gb.edges():
        mg = d["mortar_grid"]
        gs, gm = gb.nodes_of_edge(e)
        ds = gb.node_props(gs)
        dm = gb.node_props(gm)
        assert gs.dim == mg.dim and gm.dim == mg.dim + 1

        secondaryPos = np.argwhere(
            np.array(gridIdx[mg.dim]) == ds["node_number"]
        ).ravel()
        primaryPos = np.argwhere(
            np.array(gridIdx[mg.dim + 1]) == dm["node_number"]
        ).ravel()

        assert secondaryPos.size == 1 and primaryPos.size == 1

        secondary2mortar[mg.dim][
            mortarPos[mg.dim], secondaryPos
        ] = mg.secondary_to_mortar_int()
        primary2mortar[mg.dim][
            mortarPos[mg.dim], primaryPos
        ] = mg.primary_to_mortar_int()
        mortarPos[mg.dim] += 1

    # Finally, merge the mortar grids and projections
    mergedMortars: List[List[pp.MortarGrid]] = []
    mergedSecondary2Mortar: List[Union[List, sps.spmatrix]] = []
    mergedPrimary2Mortar: List[Union[List, sps.spmatrix]] = []
    for dim in range(dimMax + 1):
        if len(mortarsOfDim[dim]) == 0:
            mergedMortars.append([])
            mergedSecondary2Mortar.append([])
            mergedPrimary2Mortar.append([])
        else:
            mergedMortars.append(merge_grids(mortarsOfDim[dim]))
            mergedSecondary2Mortar.append(sps.bmat(secondary2mortar[dim], format="csc"))
            mergedPrimary2Mortar.append(sps.bmat(primary2mortar[dim], format="csc"))

    # Initiate the new grid bucket and add the merged grids.
    mergedGb = pp.GridBucket()
    mergedGb.add_nodes([g for g in mergedGrids if g != []])
    for dim in range(dimMax + 1):
        cell_idx = []
        for i in range(len(gridIdx[dim])):
            cell_idx.append(gridIdx[dim][i] * np.ones(numCells[dim][i], dtype=int))
        if len(gridIdx[dim]) > 0:
            data = mergedGb.node_props(mergedGb.grids_of_dimension(dim)[0])
            data["cell_2_frac"] = np.hstack(cell_idx)

    for dim in range(dimMax):
        mg = mergedMortars[dim]
        if mg == list([]):
            continue
        gm = mergedGrids[dim + 1]
        gs = mergedGrids[dim]

        mergedGb.add_edge((gm, gs), np.empty(0))
        mg = pp.MortarGrid(gs.dim, {pp.grids.mortar_grid.MortarSides.NONE_SIDE: mg})
        mg._primary_to_mortar_int = mergedPrimary2Mortar[dim]
        mg._secondary_to_mortar_int = mergedSecondary2Mortar[dim]

        d = mergedGb.edge_props((gm, gs))
        d["mortar_grid"] = mg

    mergedGb.assign_node_ordering()
    # for g, _ in mergedGb:
    #     g.tags: Dict = {}
    #     g.initiate_face_tags()
    #     g.update_boundary_face_tag()

    #     # Add tag for the boundary nodes
    #     g.initiate_node_tags()
    #     g.update_boundary_node_tag()

    return mergedGb


@pp.time_logger(sections=module_sections)
def switch_sign_if_inwards_normal(
    g: pp.Grid, nd: int, faces: np.ndarray
) -> sps.spmatrix:
    """Construct a matrix that changes sign of quantities on faces with a
    normal that points into the grid.

    Parameters:
        g (pp.Grid): Grid.
        nd (int): Number of quantities per face; this will for instance be the
            number of components in a face-vector.
        faces (np.array-like of ints): Index for which faces to be considered. Should only
            contain boundary faces.

    Returns:
        sps.dia_matrix: Diagonal matrix which switches the sign of faces if the
            normal vector of the face points into the grid g. Faces not considered
            will have a 0 diagonal term. If nd > 1, the first nd rows are associated
            with the first face, then nd elements of the second face etc.

    """

    faces = np.asarray(faces)

    # Find out whether the boundary faces have outwards pointing normal vectors
    # Negative sign implies that the normal vector points inwards.
    sgn, _ = g.signs_and_cells_of_boundary_faces(faces)

    # Create vector with the sign in the places of faces under consideration,
    # zeros otherwise
    sgn_mat = np.zeros(g.num_faces)
    sgn_mat[faces] = sgn
    # Duplicate the numbers, the operator is intended for vector quantities
    sgn_mat = np.tile(sgn_mat, (nd, 1)).ravel(order="F")

    # Create the diagonal matrix.
    return sps.dia_matrix((sgn_mat, 0), shape=(sgn_mat.size, sgn_mat.size))
