import warnings
import numpy as np
from scipy import sparse as sps
from itertools import islice
import csv

from porepy.grids import grid, grid_bucket
from porepy.grids.gmsh import gmsh_interface
from porepy.fracs import meshing, split_grid, simplex
from porepy.fracs.fractures import Fracture, FractureNetwork, EllipticFracture
from porepy.utils.setmembership import unique_columns_tol
from porepy.utils.sort_points import sort_point_pairs
import porepy.utils.comp_geom as cg

# ------------------------------------------------------------------------------#


def dfm_3d_from_csv(file_name, tol=1e-4, elliptic_fractures=False, **mesh_kwargs):
    """
    Create the grid bucket from a set of 3d fractures stored in a csv file and
    domain. In the csv file, we assume the following structure
    - first line describes the domain as a rectangle with
      X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
    - the other lines descibe the N fractures as a list of points
      P0_X, P0_Y, P0_Z, ...,PN_X, PN_Y, PN_Z

    Parameters:
        file_name: name of the file
        tol: (optional) tolerance for the methods
        mesh_kwargs: kwargs for the gridding, see meshing.simplex_grid

    Return:
        gb: the grid bucket
    """
    if elliptic_fractures:
        frac_list, network, domain = elliptic_network_3d_from_csv(file_name)
    else:
        frac_list, network, domain = network_3d_from_csv(file_name)

    gb = meshing.simplex_grid(domain=domain, network=network, **mesh_kwargs)
    return gb, domain


# ------------------------------------------------------------------------------#


def network_3d_from_csv(file_name, has_domain=True, tol=1e-4):
    """
    Create the fracture network from a set of 3d fractures stored in a csv file and
    domain. In the csv file, we assume the following structure
    - first line (optional) describes the domain as a rectangle with
      X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
    - the other lines descibe the N fractures as a list of points
      P0_X, P0_Y, P0_Z, ...,PN_X, PN_Y, PN_Z

    Lines that start with a # are ignored.

    Parameters:
        file_name: name of the file
        has_domain: if the first line in the csv file specify the domain
        tol: (optional) tolerance for the methods

    Return:
        frac_list: the list of fractures
        network: the fracture network
        domain: (optional, returned if has_domain==True) the domain
    """

    # The first line of the csv file defines the bounding box for the domain

    frac_list = []
    # Extract the data from the csv file
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first
        if has_domain:
            domain = np.asarray(next(spam_reader), dtype=np.float)
            assert domain.size == 6
            domain = {
                "xmin": domain[0],
                "xmax": domain[3],
                "ymin": domain[1],
                "ymax": domain[4],
                "zmin": domain[2],
                "zmax": domain[5],
            }

        for row in spam_reader:
            # If the line starts with a '#', we consider this a comment
            if row[0][0] == "#":
                continue

            # Read the points
            pts = np.asarray(row, dtype=np.float)
            assert pts.size % 3 == 0

            # Skip empty lines. Useful if the file ends with a blank line.
            if pts.size == 0:
                continue

            frac_list.append(Fracture(pts.reshape((3, -1), order="F")))

    # Create the network
    network = FractureNetwork(frac_list, tol=tol)

    if has_domain:
        return frac_list, network, domain
    else:
        return frac_list, network


# ------------------------------------------------------------------------------#


def elliptic_network_3d_from_csv(file_name, has_domain=True, tol=1e-4, degrees=False):

    """
    Create the fracture network from a set of 3d fractures stored in a csv file and
    domain. In the csv file, we assume the following structure
    - first line (optional) describes the domain as a rectangle with
      X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
    - the other lines descibe the N fractures as a elliptic fractures:
      center_x, center_y, center_z, major_axis, minor_axis, major_axis_angle,
                 strike_angle, dip_angle, num_points.
     See EllipticFracture for information about the parameters

    Lines that start with a # are ignored.

    Parameters:
        file_name: name of the file
        has_domain: if the first line in the csv file specify the domain
        tol: (optional) tolerance for the methods

    Return:
        frac_list: the list of fractures
        network: the fracture network
        domain: (optional, returned if has_domain==True) the domain
    """

    # The first line of the csv file defines the bounding box for the domain

    frac_list = []
    # Extract the data from the csv file
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first
        if has_domain:
            domain = np.asarray(next(spam_reader), dtype=np.float)
            assert domain.size == 6
            domain = {
                "xmin": domain[0],
                "xmax": domain[3],
                "ymin": domain[1],
                "ymax": domain[4],
                "zmin": domain[2],
                "zmax": domain[5],
            }

        for row in spam_reader:
            # If the line starts with a '#', we consider this a comment
            if row[0][0] == "#":
                continue

            # Read the data
            data = np.asarray(row, dtype=np.float)
            assert data.size % 9 == 0

            # Skip empty lines. Useful if the file ends with a blank line.
            if data.size == 0:
                continue
            centers = data[0:3]
            maj_ax = data[3]
            min_ax = data[4]
            maj_ax_ang = data[5] * (1 - degrees + degrees * np.pi / 180)
            strike_ang = data[6] * (1 - degrees + degrees * np.pi / 180)
            dip_ang = data[7] * (1 - degrees + degrees * np.pi / 180)
            num_points = data[8]

            frac_list.append(
                EllipticFracture(
                    centers, maj_ax, min_ax, maj_ax_ang, strike_ang, dip_ang, num_points
                )
            )
    # Create the network
    network = FractureNetwork(frac_list, tol=tol)

    if has_domain:
        return frac_list, network, domain
    else:
        return frac_list, network


# ------------------------------------------------------------------------------#


def dfm_2d_from_csv(
    f_name, mesh_kwargs, domain=None, return_domain=False, tol=1e-8, **kwargs
):
    """
    Create the grid bucket from a set of fractures stored in a csv file and a
    domain. In the csv file, we assume the following structure:
    FID, START_X, START_Y, END_X, END_Y

    Where FID is the fracture id, START_X and START_Y are the abscissa and
    coordinate of the starting point, and END_X and END_Y are the abscissa and
    coordinate of the ending point.
    Note: the delimiter can be different.

    Parameters:
        f_name: the file name in CSV format
        mesh_kwargs: list of additional arguments for the meshing
        domain: rectangular domain, if not given the bounding-box is computed
        kwargs: list of argument for the numpy function genfromtxt

    Returns:
        gb: grid bucket associated to the configuration.
        domain: if the domain is not given as input parameter, the bounding box
        is returned.

    """
    pts, edges = lines_from_csv(f_name, tol=tol, **kwargs)
    f_set = np.array([pts[:, e] for e in edges.T])

    # Define the domain as bounding-box if not defined
    if domain is None:
        overlap = kwargs.get("domain_overlap", 0)
        domain = cg.bounding_box(pts, overlap)

    if return_domain:
        return meshing.simplex_grid(f_set, domain, **mesh_kwargs), domain
    else:
        return meshing.simplex_grid(f_set, domain, **mesh_kwargs)


# ------------------------------------------------------------------------------#


def lines_from_csv(f_name, tagcols=None, tol=1e-8, max_num_fracs=None, **kwargs):
    """ Read csv file with fractures to obtain fracture description.

    Create the grid bucket from a set of fractures stored in a csv file and a
    domain. In the csv file, we assume the following structure:
    FID, START_X, START_Y, END_X, END_Y

    Where FID is the fracture id, START_X and START_Y are the abscissa and
    coordinate of the starting point, and END_X and END_Y are the abscissa and
    coordinate of the ending point.

    To change the delimiter from the default comma, use kwargs passed to
    np.genfromtxt.

    The csv file is assumed to have a header of 1 line. To change this number,
    use kwargs skip_header.

    Parameters:
        f_name (str): Path to csv file
        tagcols (array-like, int. Optional): Column index where fracture tags
            are stored. 0-offset. Defaults to no columns.
        tol (double, optional): Tolerance for merging points with almost equal
            coordinates.
        max_num_fracs (int, optional): Maximum number of fractures included,
            counting from the start of the file. Defaults to inclusion of all
            fractures.
        **kwargs: keyword arguments passed on to np.genfromtxt.

    Returns:
        np.ndarray (2 x num_pts): Point coordinates used in the fracture
            description.
        np.ndarray (2+numtags x num_fracs): Fractures, described by their start
            and endpoints (first and second row). If tags are assigned to the
            fractures, these are stored in rows 2,...

    """
    npargs = {}
    # EK: Should these really be explicit keyword arguments?
    npargs["delimiter"] = kwargs.get("delimiter", ",")
    npargs["skip_header"] = kwargs.get("skip_header", 1)

    # Extract the data from the csv file
    data = np.genfromtxt(f_name, **npargs)
    if data.size == 0:
        return np.empty((2, 0)), np.empty((2, 0), dtype=np.int)
    data = np.atleast_2d(data)
    if max_num_fracs is not None:
        data = data[:max_num_fracs]

    num_fracs = data.shape[0] if data.size > 0 else 0
    num_data = data.shape[1] if data.size > 0 else 0

    pt_cols = np.arange(1, num_data)
    if tagcols is not None:
        pt_cols = np.setdiff1d(pt_cols, tagcols)

    pts = data[:, pt_cols].reshape((-1, 2)).T

    # Let the edges correspond to the ordering of the fractures
    edges = np.vstack((np.arange(0, 2 * num_fracs, 2), np.arange(1, 2 * num_fracs, 2)))
    if tagcols is not None:
        edges = np.vstack((edges, data[:, tagcols].T))

    pts, _, old_2_new = unique_columns_tol(pts, tol=tol)
    edges[:2] = old_2_new[edges[:2]]

    to_remove = np.where(edges[0, :] == edges[1, :])[0]
    edges = np.delete(edges, to_remove, axis=1)

    assert np.all(np.diff(edges[:2], axis=0) != 0)

    return pts, edges.astype(np.int)


# ------------ End of CSV-based functions. Start of gmsh related --------------#


def dfm_from_gmsh(file_name, dim, network=None, **kwargs):

    verbose = kwargs.get("verbose", 1)

    # run gmsh to create .msh file if
    if file_name[-4:] == ".msh":
        out_file = file_name
    else:
        if file_name[-4:] == ".geo":
            file_name = file_name[:-4]
        in_file = file_name + ".geo"
        out_file = file_name + ".msh"

        gmsh_opts = kwargs.get("gmsh_opts", {})
        gmsh_verbose = kwargs.get("gmsh_verbose", verbose)
        gmsh_opts["-v"] = gmsh_verbose
        gmsh_status = gmsh_interface.run_gmsh(in_file, out_file, dims=dim, **gmsh_opts)
        if verbose:
            print("Gmsh finished with status " + str(gmsh_status))

    if dim == 2:
        raise NotImplementedError()
    elif dim == 3:
        assert (
            network is not None
        ), """Need access to the network used to
            produce the .geo file"""
        grids = simplex.tetrahedral_grid_from_gmsh(out_file, network, **kwargs)
        return meshing.grid_list_to_grid_bucket(grids, **kwargs)


# ------------ End of gmsh-based functions, start of fab related --------------#


def dfn_3d_from_fab(
    file_name, file_inters=None, conforming=True, tol=None, vtk_name=None, **kwargs
):
    """ Read the fractures and (possible) intersection from files.
    The fractures are in a .fab file, as specified by FracMan.
    The intersection are specified in the function intersection_dfn_3d.

    Parameters:
        file_name (str): Path to .fab file.
        file_intersections (str): Optional path to intersection file.
        conforming (boolean): If True, the mesh will be conforming along 1d
            intersections.
        vtk_name (str): Gives the possibility to export the network in a vtu
            file. Consider the suffix of the file as ".vtu".
        **kwargs: Parameters passed to gmsh.

    Returns:
        gb (GridBucket): The grid bucket.

    """
    network = network_3d_from_fab(file_name, return_all=False, tol=tol)

    if vtk_name is not None:
        network.to_vtk(vtk_name)

    if file_inters is None:
        return meshing.dfn(network, conforming, **kwargs)
    else:
        inters = intersection_dfn_3d(file_inters, fractures)
        return meshing.dfn(network, conforming, inters, **kwargs)


# ------------------------------------------------------------------------------#


def network_3d_from_fab(f_name, return_all=False, tol=None):
    """ Read fractures from a .fab file, as specified by FracMan.

    The filter is based on the .fab-files available at the time of writing, and
    may not cover all options available.

    Parameters:
        f_name (str): Path to .fab file.

    Returns:
        network: the network of fractures
        tess_fracs (optional returned if return_all==True, list of np.ndarray):
            Each list element contains fracture
            cut by the domain boundary, represented by vertexes as a nd x n_pt
            array.
        tess_sgn (optional returned if return_all==True, np.ndarray):
            For each element in tess_frac, a +-1 defining
            which boundary the fracture is on.

    The function also reads in various other information of unknown usefulness,
    see implementation for details. This information is currently not returned.

    """

    def read_keyword(line):
        # Read a single keyword, on the form  key = val
        words = line.split("=")
        assert len(words) == 2
        key = words[0].strip()
        val = words[1].strip()
        return key, val

    def read_section(f, section_name):
        # Read a section of the file, surrounded by a BEGIN / END wrapping
        d = {}
        for line in f:
            if line.strip() == "END " + section_name.upper().strip():
                return d
            k, v = read_keyword(line)
            d[k] = v

    def read_fractures(f, is_tess=False):
        # Read the fracture
        fracs = []
        fracture_ids = []
        trans = []
        nd = 3
        for line in f:
            if not is_tess and line.strip() == "END FRACTURE":
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            elif is_tess and line.strip() == "END TESSFRACTURE":
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            if is_tess:
                ids, num_vert = line.split()
            else:
                ids, num_vert, t = line.split()[:3]

                trans.append(float(t))

            ids = int(ids)
            num_vert = int(num_vert)
            vert = np.zeros((num_vert, nd))
            for i in range(num_vert):
                data = f.readline().split()
                vert[i] = np.asarray(data[1:])

            # Transpose to nd x n_pt format
            vert = vert.T

            # Read line containing normal vector, but disregard result
            data = f.readline().split()
            if is_tess:
                trans.append(int(data[1]))
            fracs.append(vert)
            fracture_ids.append(ids)

    with open(f_name, "r") as f:
        for line in f:
            if line.strip() == "BEGIN FORMAT":
                # Read the format section, but disregard the information for
                # now
                formats = read_section(f, "FORMAT")
            elif line.strip() == "BEGIN PROPERTIES":
                # Read in properties section, but disregard information
                props = read_section(f, "PROPERTIES")
            elif line.strip() == "BEGIN SETS":
                # Read set section, but disregard information.
                sets = read_section(f, "SETS")
            elif line.strip() == "BEGIN FRACTURE":
                # Read fractures
                fracs, frac_ids, trans = read_fractures(f, is_tess=False)
            elif line.strip() == "BEGIN TESSFRACTURE":
                # Read tess_fractures
                tess_fracs, tess_frac_ids, tess_sgn = read_fractures(f, is_tess=True)
            elif line.strip()[:5] == "BEGIN":
                # Check for keywords not yet implemented.
                raise ValueError("Unknown section type " + line)

    fractures = [Fracture(f) for f in fracs]
    if tol is not None:
        network = FractureNetwork(fractures, tol=tol)
    else:
        network = FractureNetwork(fractures)

    if return_all:
        return network, tess_fracs, tess_sgn
    else:
        return network


# ------------------------------------------------------------------------------#


def intersection_dfn_3d(file_name, fractures):
    """ Read the fracture intersections from file.
    NOTE: We assume that the fracture id in the file starts from 1. The file
    format is as follow:
    x_0 y_0 x_1 y_1 frac_id0 frac_id1

    Parameters:
        file_name (str): Path to file.
        fractures (either Fractures, or a FractureNetwork).

    Returns:
        intersections (list of lists): Each item corresponds to an
            intersection between two fractures. In each sublist, the first two
            indices gives fracture ids (refering to order in fractures). The third
            item is a numpy array representing intersection coordinates.
    """

    class DummyFracture(object):
        def __init__(self, index):
            self.index = index

    inter_from_file = np.atleast_2d(np.loadtxt(file_name))
    intersections = np.empty((inter_from_file.shape[0], 3), dtype=np.object)

    for line, intersection in zip(inter_from_file, intersections):
        intersection[0] = DummyFracture(int(line[6]) - 1)
        intersection[1] = DummyFracture(int(line[7]) - 1)
        intersection[2] = np.array(line[:6]).reshape((3, -1), order="F")
    return intersections


# ------------------------------------------------------------------------------#


def read_dfn_grid(folder, num_fractures, case_id, **kwargs):

    # TODO: tag tip faces

    offset_name = kwargs.get("offset_name", 1)
    folder += "/"
    g_2d = np.empty(num_fractures, dtype=np.object)
    gb = grid_bucket.GridBucket()

    global_node_id = 0
    for f_id in np.arange(num_fractures):
        post = "_F" + str(f_id + offset_name) + "_" + str(case_id) + ".txt"
        nodes_2d, face_nodes_2d, cell_faces_2d = _dfn_grid_2d(folder, post, **kwargs)
        g_2d[f_id] = grid.Grid(
            2,
            nodes_2d,
            face_nodes_2d,
            cell_faces_2d,
            "fracture_" + str(f_id) + "_" + str(case_id),
        )

        bnd_faces = g_2d[f_id].get_all_boundary_faces()
        g_2d[f_id].tags["domain_boundary_faces"][bnd_faces] = True

        g_2d[f_id].global_point_ind = np.arange(g_2d[f_id].num_nodes) + global_node_id
        global_node_id += g_2d[f_id].num_nodes

    gb.add_nodes(g_2d)

    for f_id in np.arange(num_fractures):
        post = "_F" + str(f_id + offset_name) + "_" + str(case_id) + ".txt"

        face_name = kwargs.get("face_name", "Faces")
        face_file_name = folder + face_name + post

        with open(face_file_name, "r") as f:
            skip_lines = int(f.readline().split()[0]) + 1
            lines = np.array(f.readlines()[skip_lines:])
            conn = np.atleast_2d(
                [np.fromstring(l, dtype=np.int, sep=" ") for l in lines]
            )
            # Consider only the new intersections
            conn = conn[conn[:, 2] > f_id, :]

            for g_id in np.unique(conn[:, 2]):
                other_f_id = g_id - 1
                mask = conn[:, 2] == g_id

                nodes_id = _nodes_faces_2d(g_2d[f_id], conn[mask, 0])
                nodes_1d, face_nodes_1d, cell_faces_1d = _dfn_grid_1d(
                    g_2d[f_id], nodes_id
                )
                g_1d = grid.Grid(
                    1,
                    nodes_1d,
                    face_nodes_1d,
                    cell_faces_1d,
                    "intersection_"
                    + str(f_id)
                    + "_"
                    + str(g_id - 1)
                    + "_"
                    + str(case_id),
                )

                global_point_ind = g_2d[f_id].global_point_ind[nodes_id]
                g_1d.global_point_ind = global_point_ind

                nodes_id = _nodes_faces_2d(g_2d[other_f_id], conn[mask, 1])
                for g, _ in gb:  # TODO: better access
                    if g is g_2d[other_f_id]:
                        g.global_point_ind[nodes_id] = global_point_ind
                        break

                gb.add_nodes(g_1d)

                shape = (g_1d.num_cells, g_2d[f_id].num_faces)
                data = np.ones(g_1d.num_cells, dtype=np.bool)
                face_cells = sps.csc_matrix(
                    (data, (np.arange(g_1d.num_cells), conn[mask, 0])), shape
                )
                gb.add_edge([g_2d[f_id], g_1d], face_cells)

                shape = (g_1d.num_cells, g_2d[other_f_id].num_faces)
                face_cells = sps.csc_matrix(
                    (data, (np.arange(g_1d.num_cells), conn[mask, 1])), shape
                )
                gb.add_edge([g_2d[other_f_id], g_1d], face_cells)

    gb.compute_geometry()
    # Split the grids.
    split_grid.split_fractures(gb, offset=0.1)
    return gb


# ------------------------------------------------------------------------------#


def _dfn_grid_2d(folder, post, **kwargs):

    cell_name = kwargs.get("cell_name", "Cells")
    cell_file_name = folder + cell_name + post

    # Read the cells and construct the cell_faces matrix
    with open(cell_file_name, "r") as f:
        num_cells, num_faces_cells = map(int, f.readline().split())
        cell_faces_data = np.empty(num_faces_cells, dtype=np.int)
        cell_faces_indptr = np.zeros(num_cells + 1, dtype=np.int)
        cell_faces_indices = np.empty(num_faces_cells, dtype=np.int)

        lines = list(islice(f, num_cells))
        pos = 0
        for cell_id, line in enumerate(lines):
            data = np.fromstring(line, dtype=np.int, sep=" ")
            cell_faces_indptr[cell_id + 1] = pos + data.size
            index = slice(pos, pos + data.size)
            cell_faces_indices[index] = np.abs(data)
            cell_faces_data[index] = 2 * (data > 0) - 1
            pos += data.size

        cell_faces = sps.csc_matrix(
            (cell_faces_data, cell_faces_indices, cell_faces_indptr)
        )
        del cell_faces_data, cell_faces_indptr, cell_faces_indices

    face_name = kwargs.get("face_name", "Faces")
    face_file_name = folder + face_name + post

    # Read the faces and construct the face_nodes matrix
    with open(face_file_name, "r") as f:
        num_faces, num_nodes_faces = map(int, f.readline().split())
        face_nodes_indices = np.empty(num_nodes_faces, dtype=np.int)

        lines = list(islice(f, num_faces))
        pos = 0
        for face_id, line in enumerate(lines):
            data = np.fromstring(line, dtype=np.int, sep=" ")
            index = slice(pos, pos + data.size)
            face_nodes_indices[index] = data
            pos += data.size

        face_nodes_indptr = np.hstack((np.arange(0, 2 * num_faces, 2), 2 * num_faces))
        face_nodes = sps.csc_matrix(
            (
                np.ones(num_nodes_faces, dtype=np.bool),
                face_nodes_indices,
                face_nodes_indptr,
            )
        )
        del face_nodes_indices, face_nodes_indptr

    node_name = kwargs.get("node_name", "Vertexes")
    node_file_name = folder + node_name + post

    with open(node_file_name, "r") as f:
        num_nodes = int(f.readline())

        nodes = np.empty((3, num_nodes))
        lines = list(islice(f, num_nodes))
        for node_id, line in enumerate(lines):
            nodes[:, node_id] = np.fromstring(line, dtype=np.float, sep=" ")

    return nodes, face_nodes, cell_faces


# ------------------------------------------------------------------------------#


def _nodes_faces_2d(grid_2d, faces_2d_id):

    nodes_id = np.empty((2, faces_2d_id.size), dtype=np.int)
    for cell_id, face_2d_id in enumerate(faces_2d_id):
        index = slice(
            grid_2d.face_nodes.indptr[face_2d_id],
            grid_2d.face_nodes.indptr[face_2d_id + 1],
        )
        nodes_id[:, cell_id] = grid_2d.face_nodes.indices[index]

    nodes_id = sort_point_pairs(nodes_id, is_circular=False)
    return np.hstack((nodes_id[0, :], nodes_id[1, -1]))


# ------------------------------------------------------------------------------#


def _dfn_grid_1d(grid_2d, nodes_id):

    num_faces = nodes_id.size
    num_cells = num_faces - 1

    cell_faces_indptr = 2 * np.arange(num_faces)
    cell_faces_indices = np.vstack(
        (np.arange(num_cells), np.arange(1, num_cells + 1))
    ).ravel("F")
    cell_faces_data = np.ones(2 * num_cells)
    cell_faces_data[::2] *= -1

    cell_faces = sps.csc_matrix(
        (cell_faces_data, cell_faces_indices, cell_faces_indptr)
    )
    del cell_faces_data, cell_faces_indptr, cell_faces_indices

    face_nodes = sps.csc_matrix(
        (
            np.ones(num_faces + 1, dtype=np.bool),
            np.arange(num_faces + 1),
            np.arange(num_faces + 1),
        )
    )

    return grid_2d.nodes[:, nodes_id], face_nodes, cell_faces


# ------------------------------------------------------------------------------#
