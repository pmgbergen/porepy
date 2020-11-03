import os
import numpy as np
import scipy.sparse as sps

import porepy as pp

def dumpGridToFile(g, fn):
    """
    Dump a PorePy grid to a file that can be read by as an unstructured
    opm grid.

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
        # Write grid info
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
        # test if grid has cartdims (i.e. Cartesian indexing)
        if hasattr(g, "cartdims"):
            cartdims = np.asarray(g.cartdims)
        else:
            cartdims = np.zeros(g.dim, dtype=np.int)
        outfile.write(" ".join(map(str, cartdims)) + "\n")
        # Write nodes
        outfile.write(" ".join(map(str, g.nodes[: g.dim].ravel("F"))) + "\n")
        outfile.write(" ".join(map(str, g.face_nodes.indptr)) + "\n")
        outfile.write(" ".join(map(str, g.face_nodes.indices)) + "\n")

        # Write faces
        neighbourship = get_neighbourship(g)
        outfile.write(" ".join(map(str, neighbourship.ravel("C"))) + "\n")
        outfile.write(" ".join(map(str, g.face_areas)) + "\n")
        outfile.write(" ".join(map(str, g.face_centers[: g.dim].ravel("F"))) + "\n")
        outfile.write(" ".join(map(str, g.face_normals[: g.dim].ravel("F"))) + "\n")

        # Write cells
        outfile.write(" ".join(map(str, g.cell_faces.indptr)) + "\n")
        if hasattr(g, "cell_facetag"):
            cell_face_tag = np.ravel(np.vstack((g.cell_faces.indices, g.cell_facetag)),'F')
            outfile.write(" ".join(map(str, cell_face_tag)) + "\n")
        else:
            outfile.write(" ".join(map(str, g.cell_faces.indices)) + "\n")
        outfile.write(" ".join(map(str, g.cell_volumes)) + "\n")
        outfile.write(" ".join(map(str, g.cell_centers[: g.dim].ravel("F"))) + "\n")

        if hasattr(g, "idx"):
            outfile.write("{:d} \n".format(g.idx))


def dumpMortarGridToFile(gb, e, d, fn):

    grid_name = append_id(fn , "mortar_" + str(d["edge_number"]))

    mg = d['mortar_grid']
    mg.idx = d['edge_number']
    mortar_grids = []
    for sg in mg.side_grids.values():
        mortar_grids.append(sg)

    mortar_grid = merge_grids(mortar_grids)
    mortar_grid.idx = mg.idx
    dim = mortar_grid.dim
    mortar_grid.dim = gb.dim_max()
    dumpGridToFile(mortar_grid, grid_name)
    mortar_grid.dim = dim

    # Now, dump the mortar projections
    gl, gh = gb.nodes_of_edge(e)

    dh = gb.node_props(gh)
    dl = gb.node_props(gl)

    name = append_id(
        fn , "mapping_" + str(d["edge_number"])
    )

    gh_to_mg = mg.mortar_to_master_int()
    gl_to_mg = mg.mortar_to_slave_int()

    dumpMortarProjectionsToFile(gh, mg, gh_to_mg, name)
    dumpMortarProjectionsToFile(gl, mg, gl_to_mg, name, "a")


def merge_grids(grids):
    grid = grids[0].copy()
    if hasattr(grids[0], "cell_facetag"):
        grid.cell_facetag = grids[0].cell_facetag

    for sg in grids[1:]:
        grid.num_cells += sg.num_cells
        grid.num_faces += sg.num_faces
        grid.num_nodes += sg.num_nodes
        grid.nodes = np.hstack((grid.nodes, sg.nodes))
        grid.face_nodes = sps.block_diag(
            (grid.face_nodes, sg.face_nodes), dtype=np.bool, format='csc'
        )
        grid.cell_faces = sps.block_diag(
            (grid.cell_faces, sg.cell_faces), dtype=np.int, format='csc'
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
        

def dumpMortarProjectionsToFile(g, mg, proj, fn, mode="w"):
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
    

def dumpGridBucketToFile(gb, fn):
    """
    Dump a PorePy grid to a file that can be read by as an unstructured
    opm grid.

    Parameters:
    gb  (GridBucket): Each grid of the grid bucket will be written to file.
    fn (String): The file name. This name will will be passed to open() using 'w'
        with a surfix giving the grid number.
    
    Returns:
    None
    """
    for g, d in gb:
        grid_name = append_id(fn, d["node_number"])
        g.idx = d["node_number"]
        dim = g.dim
        g.dim = gb.dim_max()
        dumpGridToFile(g, grid_name)
        g.dim = dim

    for e, d in gb.edges():
        dumpMortarGridToFile(gb, e, d, fn)        


def append_id(filename, idx):
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


def mergeGridsOfEqualDim(gb):
    dimMax = gb.dim_max()
    
    mergedGrids = []
    gridsOfDim = np.empty(dimMax + 1, dtype=list)
    gridIdx = np.empty(dimMax + 1, dtype=list)
    for i in range(dimMax + 1):
        gridIdx[i] =[]
        gridsOfDim[i] = gb.grids_of_dimension(i)
        if len(gridsOfDim[i])==0:
            mergedGrids.append([])
            continue

        mergedGrids.append(merge_grids(gridsOfDim[i]))
        for grid in gridsOfDim[i]:
            d = gb.node_props(grid)
            gridIdx[i].append(d['node_number'])
        
    mortarsOfDim = np.empty(dimMax + 1, dtype=list)
    for i in range(len(mortarsOfDim)):
        mortarsOfDim[i] = []
        

    for e, d in gb.edges():
        mortar_grids = []
        mg = d['mortar_grid']
        for sg in mg.side_grids.values():
            mortar_grids.append(sg)
        mortarsOfDim[mg.dim].append(merge_grids(mortar_grids))

    master2mortar = np.empty(dimMax + 1, dtype=np.ndarray)
    slave2mortar = np.empty(dimMax + 1, dtype=np.ndarray)

    for i in range(dimMax):
        master2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i+1])),dtype=np.object)
        slave2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i])), dtype=np.object)
        
        # Add an empty grid for mortar row. This is to let the block matrices
        # mergedSlave2Mortar and mergedMaster2Mortar know the correct dimension
        # if there is an empty mapping. It should be sufficient to add zeros to
        # one of the mortar grids.
        for j in range(len(gridsOfDim[i+1])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridFaces = gridsOfDim[i+1][j].num_faces
            master2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridFaces))

        for j in range(len(gridsOfDim[i])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridCells = gridsOfDim[i][j].num_cells
            slave2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridCells))
                       

    mortarPos = np.zeros(dimMax + 1, dtype=np.int)
    for e, d in gb.edges():
        mg = d['mortar_grid']
        gs, gm = gb.nodes_of_edge(e)
        ds = gb.node_props(gs)
        dm = gb.node_props(gm)
        assert gs.dim==mg.dim and gm.dim==mg.dim + 1

        slavePos = np.argwhere(np.array(gridIdx[mg.dim]) == ds['node_number']).ravel()
        masterPos = np.argwhere(np.array(gridIdx[mg.dim + 1]) == dm['node_number']).ravel()

        assert (slavePos.size==1 and masterPos.size==1)

        
        slave2mortar[mg.dim][mortarPos[mg.dim], slavePos] = mg.slave_to_mortar_int()
        master2mortar[mg.dim][mortarPos[mg.dim], masterPos] = mg.master_to_mortar_int()
        mortarPos[mg.dim] += 1

    mergedMortars = []
    mergedSlave2Mortar = []
    mergedMaster2Mortar = []
    for dim in range(dimMax + 1):
        if len(mortarsOfDim[dim])==0:
            mergedMortars.append([])
            mergedSlave2Mortar.append([])
            mergedMaster2Mortar.append([])
        else:
            mergedMortars.append(merge_grids(mortarsOfDim[dim]))
            mergedSlave2Mortar.append(sps.bmat(slave2mortar[dim], format="csc"))
            mergedMaster2Mortar.append(sps.bmat(master2mortar[dim], format="csc"))

    mergedGb = pp.GridBucket()
    mergedGb.add_nodes([g for g in mergedGrids if g != []])

    for dim in range(dimMax):
        mg = mergedMortars[dim]
        if (mg == list([])):
            continue
        gm = mergedGrids[dim + 1]
        gs = mergedGrids[dim]
        
        mergedGb.add_edge((gm, gs), np.empty(0))
        mg = pp.MortarGrid(gs.dim, {'0': mg}, sps.csc_matrix(0))
        mg._master_to_mortar_int = mergedMaster2Mortar[dim]
        mg._slave_to_mortar_int = mergedSlave2Mortar[dim]

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

def purge0dFacesAndNodes(gb):
    for g, d in gb:
        if g.dim!=0:
            continue
        _purgefaceAndNodesFromGrid(g)

    for e, d in gb.edges():
        mg = d['mortar_grid']
        if mg.dim!=0:
            continue
        for sg in mg.side_grids.values():
            _purgefaceAndNodesFromGrid(sg)

def addCellFaceTag(gb):

    if isinstance(gb, pp.GridBucket):
        for g in gb.grids_of_dimension(3):
            if "CartGrid" in g.name:
                raise NotImplementedError('Have not implemented addCellFaceTag for dimension 3')
        grid_list = gb.grids_of_dimension(2)

    else:
        if gb.dim != 2:
            raise NotImplementedError("Have only implemented callFaceTag for dimension 2")
        grid_list = [gb]

    tol = 1e-10
    for g in grid_list:
        if "CartGrid" in g.name:
            g.cell_facetag = np.zeros(g.cell_faces.indptr[-1], dtype=int)
            for k in range(g.num_cells):
                cc = g.cell_centers[:, k]
                for i in range(4):
                    face = g.cell_faces.indices[k * 4 + i]
                    fc = g.face_centers[:, face]

                    diff = cc - fc

                    num_tags = 0
                    if diff[0] > tol:
                        g.cell_facetag[k * 4 + i] = 0
                        num_tags += 1
                    if diff[0] < - tol:
                        g.cell_facetag[k * 4 + i] = 1
                        num_tags += 1
                    if diff[1] > tol:
                        g.cell_facetag[k * 4 + i] = 2
                        num_tags += 1
                    if diff[1] < - tol:
                        g.cell_facetag[k * 4 + i] = 3
                        num_tags += 1

                    if num_tags!=1:
                        raise AttributeError(
                            "Could not find W, E, S, or N face of cell {}".format(k)
                        )

def enforce_opm_face_ordering(gb):
    if isinstance(gb, pp.GridBucket):
        for g in gb.grids_of_dimension(3):
            if "CartGrid" in g.name:
                raise NotImplementedError('Have not implemented addCellFaceTag for dimension 3')
        grid_list = gb.grids_of_dimension(2)

    else:
        if gb.dim != 2:
            raise NotImplementedError("Have only implemented callFaceTag for dimension 2")
        grid_list = [gb]

    # OPM faces ordered counterclockwise starting at West face
    opm_sort = [0,2,1,3]
    for g in grid_list:
        if not "CartGrid" in g.name:
            raise ValueError("Can only enforce face ordering for CartGrid")
        if not hasattr(g, "cell_facetag"):
            raise ValueError("Can only order grids with cell_facetag")

        # Get ordering of OPM faces in a cell
        _, IC = np.unique(opm_sort, return_inverse=True)
        cell_facetag = g.cell_facetag.reshape((-1, 4))

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

def _purgefaceAndNodesFromGrid(g):
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

    dumpGridToFile(g, file_name)
