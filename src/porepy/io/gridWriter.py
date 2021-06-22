import os
import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["io"]


@pp.time_logger(sections=module_sections)
def dump_grid_to_file(g: pp.Grid, fn: str) -> None:
    """
    Dump a PorePy grid to a file. The file format of the dumped file
    can be read by pp.read_grid_from_file. It is also compatible
    with a MRST (https://www.sintef.no/projectweb/mrst/) or an
    OPM unstructured grid ( https://opm-project.org/ ).

    Parameters:
    g  (Grid): The grid will be written to file
    fn (String): The file name. This will be passed to open() using 'w'
    
    Returns:
    None
    """
    # Test if directory in file name exists and create if not
    dirpath = os.path.dirname(fn)
    os.makedirs(dirpath, exist_ok=True)

    # Open file and start writing
    with open(fn, "w") as outfile:
        # First line of file contains grid info:
        num_face_nodes = g.face_nodes.indices.size
        num_cell_faces = g.cell_faces.indices.size
        outfile.write(
            "{:d} {:d} {:d} {:d} {:d} {:d} {:d} 0\n".format(
                g.dim,
                g.num_cells,
                g.num_faces,
                g.num_nodes,
                num_face_nodes,
                num_cell_faces,
                hasattr(g, "cell_facetag"),
            )
        )
        # Second line contains the cartesian indexes of the grid.
        # If no cartesian indexing, fill with zeros.
        if hasattr(g, "cartdims"):
            cartdims = np.asarray(g.cartdims)
        else:
            cartdims = np.zeros(g.dim, dtype=np.int)
        outfile.write(" ".join(map(str, cartdims)) + "\n")
        ## Nodes:
        # Third line is the node coordinates (x1, y1, z1, x2, y2, z2, ...)
        outfile.write(" ".join(map(str, g.nodes[: g.dim].ravel("F"))) + "\n")
        # line 4 and 5, the sparse csc representation of the face-nodes map.
        outfile.write(" ".join(map(str, g.face_nodes.indptr)) + "\n")
        outfile.write(" ".join(map(str, g.face_nodes.indices)) + "\n")

        ## Faces:
        # line 6: neighbourship of faces. This is a [num_faces, 2] array.
        # Each line represent a face. neighbourship[f, 0] is the inner cell
        # (normal points out of cell) connected to face f. neighbourship[f, 1]
        # is the outer cell (normal points into cell). If if is a boundary face,
        # either the inner or the outer cell contains the value -1. The neighbouship
        # is included to be compatible with OPM and MRST.
        neighbourship = get_neighbourship(g)
        outfile.write(" ".join(map(str, neighbourship.ravel("C"))) + "\n")
        # Line 7-9: face geometry
        outfile.write(" ".join(map(str, g.face_areas)) + "\n")
        outfile.write(" ".join(map(str, g.face_centers[: g.dim].ravel("F"))) + "\n")
        outfile.write(" ".join(map(str, g.face_normals[: g.dim].ravel("F"))) + "\n")

        ## Cells
        # Line 10 write the csc representation of the cell-faces map. If g has a
        # cell_facetag append this to the indices.
        outfile.write(" ".join(map(str, g.cell_faces.indptr)) + "\n")
        if hasattr(g, "cell_facetag"):
            cell_face_tag = np.ravel(np.vstack((g.cell_faces.indices, g.cell_facetag)),'F')
            outfile.write(" ".join(map(str, cell_face_tag)) + "\n")
        else:
            outfile.write(" ".join(map(str, g.cell_faces.indices)) + "\n")
        # Line 11 - 12: cell geometry
        outfile.write(" ".join(map(str, g.cell_volumes)) + "\n")
        outfile.write(" ".join(map(str, g.cell_centers[: g.dim].ravel("F"))) + "\n")

        # Finaly, if the grid is given an index we write that to file.
        if hasattr(g, "idx"):
            outfile.write("{:d} \n".format(g.idx))


def dump_mortar_grid_to_file(gb, e, d, fn, max_1_grid_per_dim=False, dfn=False):
    """
    Dump a PorePy mortar grid to a file. The file format of the dumped file
    can be read by pp.read_grid_from_file. It is also compatible
    with a MRST (https://www.sintef.no/projectweb/mrst/) or an
    OPM unstructured grid ( https://opm-project.org/ ).
    
    Both the mortar grid and the mappings to the Primary and Secondary will
    be dumpt to file. The mappings will have the surfix "_mapping"

    Parameters:
    gb  (GridBucket): The grid bucket associated with the mortar grid.
    e (Tuple(Grid, Grid)): The edge in gb associated with the mortar grid.
    d (Dict): The dictionary of the edge
    fn (String): The file name. The file name will be appended with an index. 
        This will be passed to open() using 'w'
    max_1_grid_per_dim (bool): OPTIONAL, defaults to False. If True, the
        grid dimension will be used to index the file name. If False,
        the edge_number will be used.
    dfn (bool) OPTIONAL, defaults to False. If the gb is a DFN, set this value to
        True

    Returns:
    None
    """

    # Get the index of the grid and append it to file
    if max_1_grid_per_dim:
        grid_id = str(d['mortar_grid'].dim)
    else:
        grid_id = str(d["edge_number"])
    grid_name = append_id_(fn , "mortar_" + grid_id)
    mg = d['mortar_grid']
    mg.idx = d['edge_number']
    # We need to obtain the actuall mortar grids from the side grids
    mortar_grids = []
    for sg in mg.side_grids.values():
        mortar_grids.append(sg)
    # We store both sides of the grids as one grid.
    mortar_grid = merge_grids(mortar_grids)
    mortar_grid.idx = mg.idx
    dim = mortar_grid.dim

    # We need to set the maximum dimension = maximum dimension of the gb,
    # in order to export the coorect number of coordinates of vector-valued
    # geometry variables in the grid. If dfn, add 1 to the dimension.
    mortar_grid.dim = gb.dim_max() + dfn

    # Dump the mortar grids (or side grids in pp context).
    dump_grid_to_file(mortar_grid, grid_name)
    mortar_grid.dim = dim

    # Now, dump the mortar projections to the primary and secondary
    gl, gh = gb.nodes_of_edge(e)

    dh = gb.node_props(gh)
    dl = gb.node_props(gl)

    name = append_id_(
        fn , "mapping_" + grid_id
    )

    gh_to_mg = mg.mortar_to_primary_int()
    gl_to_mg = mg.mortar_to_secondary_int()

    dump_mortar_projections_to_file(gh, mg, gh_to_mg, name)
    dump_mortar_projections_to_file(gl, mg, gl_to_mg, name, "a")


def merge_grids(grids):
    grid = grids[0].copy()
    if hasattr(grids[0], "cell_facetag"):
        grid.cell_facetag = grids[0].cell_facetag

    for sg in grids[1:]:
        grid.num_cells += sg.num_cells
        grid.num_faces += sg.num_faces
        grid.num_nodes += sg.num_nodes
        grid.nodes = np.hstack((grid.nodes, sg.nodes))

        grid.face_nodes = pp.utils.sparse_mat.stack_diag(
            grid.face_nodes, sg.face_nodes
        )
        grid.cell_faces = pp.utils.sparse_mat.stack_diag(
            grid.cell_faces, sg.cell_faces
        )

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
        

def dump_mortar_projections_to_file(g, mg, proj, fn, mode="w"):
    """
    Dump a PorePy grid to a file that can be read by as an unstructured
    opm grid.

    Parameters:
    g  (Grid): The grid will be written to file
    fn (String): The file name. This will be passed to open() using 'w'
    
    Returns:
    None
    """
    if not np.allclose(proj.data, 1):
        raise NotImplemented("Can not store non-matching grids, yet.")

    if not(proj.getformat() == "csc"):
        proj = proj.tocsc()

    # Test if directory in file name exists and create if not
    dirpath = os.path.dirname(fn)
    os.makedirs(dirpath, exist_ok=True)

    # Open file and start writing
    with open(fn, mode) as outfile:
        # Write grid info
        outfile.write(" ".join(map(str, proj.indptr)) + "\n")
        outfile.write(" ".join(map(str, proj.indices)) + "\n")
    

def dump_grid_bucket_to_file(gb, fn, dfn=False):
    """
    Dump all grids and mortar grids in a GridBucket to a file.

    Parameters:
    gb  (GridBucket): Each grid of the grid bucket will be written to file.
    fn (String): The file name. This name will be passed to open() using 'w'
        with a surfix giving the grid number.
    dfn (bool): OPTIONAL. Set to true if grid bucket is a DFN network
    Returns:
    None
    """
    # If only 1 grid per dimension use the grid dimension as an index. Otherwise
    # we use the graph node number.
    max_1_grid_per_dim = True
    for dim in range(gb.dim_max()):
        if len(gb.grids_of_dimension(dim)) > 1:
            max_1_grid_per_dim = False
            break

    for g, d in gb:
        if max_1_grid_per_dim:
            grid_name = append_id_(fn, str(g.dim))
        else:
            grid_name = append_id_(fn, d["node_number"])
        g.idx = d["node_number"]
        dim = g.dim
        g.dim = gb.dim_max() + dfn
        dump_grid_to_file(g, grid_name)
        g.dim = dim

    for e, d in gb.edges():
        dump_mortar_grid_to_file(gb, e, d, fn, max_1_grid_per_dim, dfn)


def append_id_(filename, idx):
  return "{0}_{2}{1}".format(*os.path.splitext(filename) + (idx,))

def get_neighbourship(g):
    fi, ci, sgn = sps.find(g.cell_faces)
    # Boundary faces are faces with only 1 cell
    _, IA, counts = np.unique(fi, return_inverse=True, return_counts=True)
    bnd_f = np.where(counts[IA] == 1)[0]

    # We add a connection between all boundary faces and the auxillary cell -1
    fi_padded = np.insert(fi, bnd_f, fi[bnd_f])
    ci_padded = np.insert(ci, bnd_f, -1)
    sgn_padded = np.insert(sgn, bnd_f, -sgn[bnd_f])

    # The face index of the neighbourship is given implicitly from 0 to num_faces.
    # Therefore, sort the faces.
    idx = np.argsort(fi_padded)
    # The neighbourship is a num_faces x 2 array
    neighbourship = np.reshape(ci_padded[idx], [-1, 2], order="C")
    # neigh_sgn tells us if the normal vector points into or out of each cell
    neigh_sgn = np.reshape(sgn_padded[idx], [-1, 2], order="C")
    # Check that we have one inner cell and one outer cell
    if not np.all(np.sum(neigh_sgn, axis=1) == 0):
        raise AssertionError("Could not find cell neighbourship.")

    # The order of the cells should be given implicitly in neighbourship; first row are
    # inner cells second row are outer cells
    swap = neigh_sgn[:, 0] == -1
    temp = np.zeros(neighbourship.shape[0], dtype=np.int)
    temp[swap] = neighbourship[swap, 0]
    neighbourship[swap, 0] = neighbourship[swap, 1]
    neighbourship[swap, 1] = temp[swap]

    # Return
    return neighbourship


def merge_grids_of_equal_dim(gb):
    dimMax = gb.dim_max()
    
    mergedGrids = []
    gridsOfDim = np.empty(dimMax + 1, dtype=list)
    gridIdx = np.empty(dimMax + 1, dtype=list)
    numCells = np.empty(dimMax + 1, dtype=list)
    for i in range(dimMax + 1):
        gridIdx[i] = []
        numCells[i] = []
        gridsOfDim[i] = gb.grids_of_dimension(i)
        if len(gridsOfDim[i])==0:
            mergedGrids.append([])
            continue

        mergedGrids.append(merge_grids(gridsOfDim[i]))
        for grid in gridsOfDim[i]:
            d = gb.node_props(grid)
            gridIdx[i].append(d['node_number'])
            numCells[i].append(grid.num_cells)
        
    mortarsOfDim = np.empty(dimMax + 1, dtype=list)
    for i in range(len(mortarsOfDim)):
        mortarsOfDim[i] = []
        

    for e, d in gb.edges():
        mortar_grids = []
        mg = d['mortar_grid']
        for sg in mg.side_grids.values():
            mortar_grids.append(sg)
        mortarsOfDim[mg.dim].append(merge_grids(mortar_grids))

    primary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)
    secondary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)

    for i in range(dimMax):
        primary2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i+1])),dtype=np.object)
        secondary2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i])), dtype=np.object)
        
        # Add an empty grid for mortar row. This is to let the block matrices
        # mergedSecondary2Mortar and mergedPrimary2Mortar know the correct dimension
        # if there is an empty mapping. It should be sufficient to add zeros to
        # one of the mortar grids.
        for j in range(len(gridsOfDim[i+1])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridFaces = gridsOfDim[i+1][j].num_faces
            primary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridFaces))

        for j in range(len(gridsOfDim[i])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridCells = gridsOfDim[i][j].num_cells
            secondary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridCells))
                       

    mortarPos = np.zeros(dimMax + 1, dtype=np.int)
    for e, d in gb.edges():
        mg = d['mortar_grid']
        gs, gm = gb.nodes_of_edge(e)
        ds = gb.node_props(gs)
        dm = gb.node_props(gm)
        assert gs.dim==mg.dim and gm.dim==mg.dim + 1

        secondaryPos = np.argwhere(np.array(gridIdx[mg.dim]) == ds['node_number']).ravel()
        primaryPos = np.argwhere(np.array(gridIdx[mg.dim + 1]) == dm['node_number']).ravel()

        assert (secondaryPos.size==1 and primaryPos.size==1)

        
        secondary2mortar[mg.dim][mortarPos[mg.dim], secondaryPos] = mg.secondary_to_mortar_int()
        primary2mortar[mg.dim][mortarPos[mg.dim], primaryPos] = mg.primary_to_mortar_int()
        mortarPos[mg.dim] += 1

    mergedMortars = []
    mergedSecondary2Mortar = []
    mergedPrimary2Mortar = []
    for dim in range(dimMax + 1):
        if len(mortarsOfDim[dim])==0:
            mergedMortars.append([])
            mergedSecondary2Mortar.append([])
            mergedPrimary2Mortar.append([])
        else:
            mergedMortars.append(merge_grids(mortarsOfDim[dim]))
            mergedSecondary2Mortar.append(sps.bmat(secondary2mortar[dim], format="csc"))
            mergedPrimary2Mortar.append(sps.bmat(primary2mortar[dim], format="csc"))

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
        if (mg == list([])):
            continue
        gm = mergedGrids[dim + 1]
        gs = mergedGrids[dim]
        
        mergedGb.add_edge((gm, gs), np.empty(0))
        mg = pp.MortarGrid(gs.dim, {pp.grids.mortar_grid.MortarSides.NONE_SIDE: mg})
        mg._primary_to_mortar_int = mergedPrimary2Mortar[dim]
        mg._secondary_to_mortar_int = mergedSecondary2Mortar[dim]

        d = mergedGb.edge_props((gm, gs))
        d['mortar_grid'] = mg

    mergedGb.assign_node_ordering()
    # for g, _ in mergedGb:
    #     g.tags: Dict = {}
    #     g.initiate_face_tags()
    #     g.update_boundary_face_tag()

    #     # Add tag for the boundary nodes
    #     g.initiate_node_tags()
    #     g.update_boundary_node_tag()

    return mergedGb

def purge0d_faces_and_nodes(gb):
    for g, d in gb:
        if g.dim!=0:
            continue
        _purge_face_and_nodes_from_grid(g)

    for e, d in gb.edges():
        mg = d['mortar_grid']
        if mg.dim!=0:
            continue
        for sg in mg.side_grids.values():
            _purge_face_and_nodes_from_grid(sg)


def addCellFaceTag(gb):

    tol = 1e-10
    if isinstance(gb, pp.GridBucket):
        for g, _ in gb:
            if g.dim < 2:
                continue
            addCellFaceTagGrid(g, tol)
    else:
        addCellFaceTagGrid(gb, tol)

def addCellFaceTagGrid(g, tol=1e-10):

    g.cell_facetag = np.zeros(g.cell_faces.indptr[-1], dtype=int)
    if g.dim == 3:
        faces_per_cell = 6
    elif g.dim == 2:
        faces_per_cell = 4
    elif g.dim < 2:
        return
    else:
        raise ValueError("Invalid grid dimension (must be 0,1,2, or 3): {}".format(g.dim))

    if g.dim <= 2:
        R = pp.map_geometry.project_plane_matrix(g.nodes, reference=np.array([0, 1, 0]))
        cell_centers, _, face_centers, _, _, _ = pp.map_geometry.map_grid(g, R=R)
    else:
        cell_centers = g.cell_centers
        face_centers = g.face_centers
    if not ("CartGrid" in g.name or "TensorGrid" in g.name):
        raise ValueError("Can only enforce face ordering for CartGrid or TensorGrid")

    for k in range(g.num_cells):
        cc = cell_centers[:, k]
        for i in range(faces_per_cell):
            face = g.cell_faces.indices[k * faces_per_cell + i]
            fc = face_centers[:, face]

            diff = cc - fc
            num_tags = 0
            if diff[0] > tol:
                g.cell_facetag[k * faces_per_cell + i] = 0
                num_tags += 1
            if diff[0] < - tol:
                g.cell_facetag[k * faces_per_cell + i] = 1
                num_tags += 1
            if diff[1] > tol:
                g.cell_facetag[k * faces_per_cell + i] = 2
                num_tags += 1
            if diff[1] < - tol:
                g.cell_facetag[k * faces_per_cell + i] = 3
                num_tags += 1
            if g.dim == 3 and diff[2] > tol:
                g.cell_facetag[k * faces_per_cell + i] = 4
                num_tags += 1
            if g.dim == 3 and diff[2] < - tol:
                g.cell_facetag[k * faces_per_cell + i] = 5
                num_tags += 1

            if num_tags != 1:
                raise AttributeError(
                    "Could not find W, E, S, or N face of cell {}".format(k)
                )


def enforce_opm_face_ordering(gb):
    if isinstance(gb, pp.GridBucket):
        for g, _ in gb:
            enforce_opm_face_ordering_grid(g)
    else:
        enforce_opm_face_ordering_grid(gb)


def enforce_opm_face_ordering_grid(g):
    # OPM faces ordered counterclockwise starting at West face
    if g.dim==2:
        opm_sort = [0,2,1,3]
    elif g.dim==3:
        opm_sort = [0, 1, 2, 3, 4, 5]
    else:
        raise ValueError(
            "Can not assign face ordering for grids of dimension {}".format(g.dim)
        )

    if not hasattr(g, "cell_facetag"):
        raise ValueError("Can only order grids with cell_facetag")

    # Get ordering of OPM faces in a cell
    _, IC = np.unique(opm_sort, return_inverse=True)
    cell_facetag = g.cell_facetag.reshape((-1, len(opm_sort)))

    old2new = np.empty(g.num_faces, dtype=int)
    for k in range(g.num_cells):
        # Get pp ordering of faces
        IA = np.argsort(cell_facetag[k,:])
        # Get faces of cell
        cell_face_pos = pp.utils.mcolon.mcolon(
            g.cell_faces.indptr[k], g.cell_faces.indptr[k + 1]
        )
        faces = g.cell_faces.indices[cell_face_pos]
        sgn = g.cell_faces.data[cell_face_pos]
        # for face in faces[[2,3]]:
        #     nodePos = g.face_nodes.indptr[face]
        #     nodes = g.face_nodes.indices[nodePos:nodePos+2].copy()
        #     g.face_nodes.indices[nodePos] = nodes[1]
        #     g.face_nodes.indices[nodePos + 1] = nodes[0]

        # change the orderign to the OPM ordering
        old2new[faces] = faces[IA][IC]
        g.cell_faces.indices[cell_face_pos] = faces[IA][IC]
        g.cell_faces.data[cell_face_pos] = sgn[IA][IC]
        g.cell_facetag[cell_face_pos] = opm_sort


def circumcenterCellCenters(gb):
    if len(gb.grids_of_dimension(3))>0:
        raise NotImplementedError('Have not implemented circumcenterCellCenters for dimension 3')

    for g in gb.grids_of_dimension(2):
        nodes = g.cell_nodes().indices
        assert np.all(np.diff(g.cell_nodes().indptr)==3)

        coords = g.nodes[:, nodes]

        ax = coords[0, ::3]
        ay = coords[1, ::3]
        bx = coords[0, 1::3]
        by = coords[1, 1::3]
        cx = coords[0, 2::3]
        cy = coords[1, 2::3]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        uz = np.zeros(g.num_cells)
        g.cell_centers = np.vstack((ux, uy, uz))
#        pp.plot_grid(g, alpha=0,info='c')

    
def _findCellsXY(g):
    y0 = g.cell_centers[1, 0]
    nx = 1
    while np.is_close(g.cell_centers(1, nx), y0):
        nx += 1

    ny = g.num_cells / nx

    return nx, ny

def _purge_face_and_nodes_from_grid(g):
    g.num_nodes = 0
    g.num_faces = 0

    g.nodes = np.zeros((3, 0))
    g.face_nodes = sps.csc_matrix((g.num_nodes, 0))
    g.cell_faces = sps.csc_matrix((0, g.num_cells))

    g.face_areas =  np.zeros((0))
    g.face_normals = np.zeros((3, 0))
    g.face_centers = np.zeros((3, 0))

if __name__ == "__main__":
    import porepy as pp
    import sys
    if len(sys.argv) == 6:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        dx = float(sys.argv[3])
        dy = float(sys.argv[4])
        file_name = sys.argv[5]

        Lx = nx * dx
        Ly = ny * dy
        f1 = np.array([[Lx / 2, Ly * 3 / 4], [Lx, Ly * 3 / 4]]).T
        f2 = np.array([[0, Ly * 2 / 4], [Lx / 2, Ly * 2 / 4]]).T
        f3 = np.array([[Lx / 2, Ly * 1 / 4], [Lx, Ly * 1 / 4]]).T
        gb = pp.meshing.cart_grid([], [nx, ny], physdims=[Lx,Ly])
        fracPts = np.hstack((f1, f2, f3))
        fracEdgs = np.array([[0, 1], [2, 3], [4, 5]]).T
        domain = {'xmin': 0, 'ymin': 0, 'xmax': Lx, 'ymax': Ly}
        fracture_network = pp.FractureNetwork2d(fracPts, fracEdgs, domain)
        gb = fracture_network.mesh({})
        g = gb.grids_of_dimension(2)[0]
        pp.plot_grid(g)
        # p = np.array([[0, 0], [Lx, 0], [0, Ly]]).T
        # t = np.array([[0], [1], [2]])
        # g = pp.TriangleGrid(p, t)
        if "CartGrid" in g.name:
            g.cell_facetag = np.zeros(g.cell_faces.indptr[-1], dtype=int)
            for k in range(nx * ny):
                for i in range(4):
                    g.cell_facetag[k * 4 + i] = i

    elif len(sys.argv) == 8:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        nz = int(sys.argv[3])
        dx = float(sys.argv[4])
        dy = float(sys.argv[5])
        dz = float(sys.argv[6])

        file_name = sys.argv[7]
        g = pp.CartGrid([nx, ny, nz], [nx * dx, ny * dy, nz * dz])
        for k in range(nx * ny * nz):
            for i in range(6):
                g.cell_facetag[k * 6 + i] = i

    else:
        print("Wrong number of arguments.")
        print("For 3D use:")
        print("ioWriter.py nx ny nz dx dy dz file_name")
        print("For 2D use:")
        print("ioWriter.py nx ny dx dy file_name")
        raise ValueError("Wrong number of arguments")
    g.compute_geometry()
    #g.cell_facetag = np.zeros(g.cell_faces.indptr[-1])

    dump_grid_to_file(g, file_name)
