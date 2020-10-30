import csv

import numpy as np

import porepy as pp


def network_3d_from_csv(file_name, has_domain=True, tol=1e-4, **kwargs):
    """
    Create the fracture network from a set of 3d fractures stored in a csv file and
    domain. In the csv file, we assume the following structure
    - first line (optional) describes the domain as a rectangle with
      X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
    - the other lines descibe the N fractures as a list of points
      P0_X, P0_Y, P0_Z, ...,PN_X, PN_Y, PN_Z

    Lines that start with a # are ignored.

    Parameters:
        file_name (str): path to the csv file
        has_domain (boolean): if the first line in the csv file specify the domain
        tol: (double, optional) geometric tolerance used in the computations.
            Defaults to 1e-4.
        **kwargs: Keyword arguments passed on to Fracture and FractureNetwork3d

    Return:
        network: the fracture network

    """

    # The first line of the csv file defines the bounding box for the domain

    frac_list = []
    # Extract the data from the csv file
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first
        if has_domain:
            domain = np.asarray(next(spam_reader), dtype=np.float)
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
            if not pts.size % 3 == 0:
                raise ValueError("Points are always 3d")

            # Skip empty lines. Useful if the file ends with a blank line.
            if pts.size == 0:
                continue

            check_convexity = kwargs.get("check_convexity", True)

            frac_list.append(
                pp.Fracture(
                    pts.reshape((3, -1), order="F"), check_convexity=check_convexity
                )
            )

    # Create the network
    if has_domain:
        return pp.FractureNetwork3d(frac_list, tol=tol, domain=domain)
    else:
        return pp.FractureNetwork3d(frac_list, tol=tol)


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
        FractureNetwork3d: the fracture network

    """

    # The first line of the csv file defines the bounding box for the domain

    frac_list = []
    # Extract the data from the csv file
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first
        if has_domain:
            domain = np.asarray(next(spam_reader), dtype=np.float)
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
            if not data.size % 9 == 0:
                raise ValueError("Data has to have size 9")

            # Skip empty lines. Useful if the file ends with a blank line.
            if data.size == 0:
                continue
            centers = data[0:3]
            maj_ax = data[3]
            min_ax = data[4]
            maj_ax_ang = data[5] * (1 - degrees + degrees * np.pi / 180)
            strike_ang = data[6] * (1 - degrees + degrees * np.pi / 180)
            dip_ang = data[7] * (1 - degrees + degrees * np.pi / 180)
            num_points = int(data[8])

            frac_list.append(
                pp.EllipticFracture(
                    centers, maj_ax, min_ax, maj_ax_ang, strike_ang, dip_ang, num_points
                )
            )
    # Create the network
    if has_domain:
        return pp.FractureNetwork3d(frac_list, tol=tol, domain=domain)
    else:
        return pp.FractureNetwork3d(frac_list, tol=tol)


def network_2d_from_csv(
    f_name,
    tagcols=None,
    tol=1e-8,
    max_num_fracs=None,
    polyline=False,
    return_frac_id=False,
    domain=None,
    **kwargs,
):
    """Read csv file with fractures to obtain fracture description.

    Create the grid bucket from a set of fractures stored in a csv file and a
    domain. In the csv file, we assume one of the two following structures:

        a) FID, START_X, START_Y, END_X, END_Y
        b) FID, PT_X, PT_Y

    Format a) is used to describe fractures consisting of a straight line.
    FID is the fracture id, START_X and START_Y are the abscissa and
    coordinate of the starting point, and END_X and END_Y are the abscissa and
    coordinate of the ending point.

    Format b) can be used to describe polyline fractures: Each row in the file
    represents a separate points, points with the same FID will be assigned to
    the same fracture *in the order specified in the file*.

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
        FractureNetwork2d: Network representation of the fractures

    Raises:
        ValueError: If a fracture of a single point is specified.

    """
    npargs = {}
    # EK: Should these really be explicit keyword arguments?
    npargs["delimiter"] = kwargs.get("delimiter", ",")
    npargs["skip_header"] = kwargs.get("skip_header", 1)

    # Extract the data from the csv file
    data = np.genfromtxt(f_name, **npargs)
    # Shortcut if no data is loaded
    if data.size == 0:
        # we still consider the possibility that a domain is given
        return pp.FractureNetwork2d(domain=domain, tol=tol)
    data = np.atleast_2d(data)

    # Consider subset of fractures if asked for
    if max_num_fracs is not None:
        if max_num_fracs == 0:
            return pp.FractureNetwork2d(tol=tol)
        else:
            data = data[:max_num_fracs]

    num_fracs = data.shape[0] if data.size > 0 else 0
    num_data = data.shape[1] if data.size > 0 else 0

    pt_cols = np.arange(1, num_data)
    if tagcols is not None:
        pt_cols = np.setdiff1d(pt_cols, tagcols)

    pts = data[:, pt_cols].reshape((-1, 2)).T

    if polyline:
        frac_id = data[:, 0]
        fracs = np.unique(frac_id)

        edges = np.empty((2, 0))
        edges_frac_id = np.empty(0)
        pt_ind = np.arange(frac_id.size)

        for fi in fracs:
            ind = np.argwhere(frac_id == fi).ravel()
            if ind.size < 2:
                raise ValueError("A fracture should consist of more than one line")
            if ind.size == 2:
                edges_loc = np.array([[pt_ind[ind[0]]], [pt_ind[ind[1]]]])
            else:
                start = pt_ind[ind[0] : ind[-1]]
                end = pt_ind[ind[1] : ind[-1] + 1]
                edges_loc = np.vstack((start, end))

            edges = np.hstack((edges, edges_loc))
            edges_frac_id = np.hstack((edges_frac_id, [fi] * edges_loc.shape[1]))

        edges = edges.astype(np.int)
        edges_frac_id = edges_frac_id.astype(np.int)

    else:
        # Let the edges correspond to the ordering of the fractures
        edges = np.vstack(
            (np.arange(0, 2 * num_fracs, 2), np.arange(1, 2 * num_fracs, 2))
        )
        # Fracture id is the first column of data
        edges_frac_id = data[:, 0]
        if tagcols is not None:
            edges = np.vstack((edges, data[:, tagcols].T))

    if domain is None:
        overlap = kwargs.get("domain_overlap", 0)
        domain = pp.bounding_box.from_points(pts, overlap)

    pts, _, old_2_new = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)

    edges[:2] = old_2_new[edges[:2].astype(np.int)]

    to_remove = np.where(edges[0, :] == edges[1, :])[0]
    edges = np.delete(edges, to_remove, axis=1).astype(np.int)

    if not np.all(np.diff(edges[:2], axis=0) != 0):
        raise ValueError

    network = pp.FractureNetwork2d(pts, edges, domain, tol=tol)

    if return_frac_id:
        edges_frac_id = np.delete(edges_frac_id, to_remove)
        return network, edges_frac_id.astype(np.int)
    else:
        return network


# ------------ End of CSV-based functions. Start of gmsh related --------------#


def dfm_from_gmsh(file_name: str, dim: int, **kwargs):
    """Generate a GridBucket from a gmsh file.

    If the provided file is input for gmsh (.geo, not .msh), gmsh will be called
    to generate the mesh before the GridBucket is constructed.

    Parameters:
        file_name (str): Name of gmsh in and out file. Should have extsion .geo
            or .msh. In the former case, gmsh will be called upon to generate the
            mesh before the mixed-dimensional mesh is constructed.
        dim (int): Dimension of the problem. Should be 2 or 3.

    Returns:
        GridBucket: Mixed-dimensional grid as contained in the gmsh file.

    """

    # run gmsh to create .msh file if
    if file_name[-4:] == ".msh":
        out_file = file_name
    else:
        if file_name[-4:] == ".geo":
            file_name = file_name[:-4]
        in_file = file_name + ".geo"
        out_file = file_name + ".msh"

        pp.grids.gmsh.gmsh_interface.run_gmsh(
            in_file,
            out_file,
            dim=dim,
        )

    if dim == 2:
        grids = pp.fracs.simplex.triangle_grid_from_gmsh(out_file, **kwargs)
    elif dim == 3:
        grids = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
            file_name=out_file, **kwargs
        )
    else:
        raise ValueError(f"Unknown dimension, dim: {dim}")
    return pp.meshing.grid_list_to_grid_bucket(grids, **kwargs)


# ------------ End of gmsh-based functions, start of fab related --------------#


def dfm_3d_from_fab(
    file_name, tol=1e-4, domain=None, return_domain=False, **mesh_kwargs
):
    """
    Create the grid bucket from a set of 3d fractures stored in a fab file and
    domain.

    Parameters:
        file_name: name of the file
        tol: (optional) tolerance for the methods
        domain: (optional) the domain, otherwise a bounding box is considered
        mesh_kwargs: kwargs for the gridding, see meshing.simplex_grid

    Return:
        gb: the grid bucket
    """

    network = network_3d_from_fab(file_name, return_all=False, tol=tol)

    # Define the domain as bounding-box if not defined
    if domain is None:
        domain = network.bounding_box()

    # TODO: No reference to simplex_grid in pp.fracs.meshing.py
    gb = pp.meshing.simplex_grid(domain=domain, network=network, **mesh_kwargs)

    if return_domain:
        return gb, domain
    else:
        return gb


# ------------------------------------------------------------------------------#


def network_3d_from_fab(f_name, return_all=False, tol=None):
    """Read fractures from a .fab file, as specified by FracMan.

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
                _ = read_section(f, "FORMAT")
            elif line.strip() == "BEGIN PROPERTIES":
                # Read in properties section, but disregard information
                _ = read_section(f, "PROPERTIES")
            elif line.strip() == "BEGIN SETS":
                # Read set section, but disregard information.
                _ = read_section(f, "SETS")
            elif line.strip() == "BEGIN FRACTURE":
                # Read fractures
                fracs, _, _ = read_fractures(f, is_tess=False)
            elif line.strip() == "BEGIN TESSFRACTURE":
                # Read tess_fractures
                tess_fracs, _, tess_sgn = read_fractures(f, is_tess=True)
            elif line.strip() == "BEGIN ROCKBLOCK":
                # Not considered block
                pass
            elif line.strip()[:5] == "BEGIN":
                # Check for keywords not yet implemented.
                raise ValueError("Unknown section type " + line)

    fractures = [pp.Fracture(f) for f in fracs]
    if tol is not None:
        network = pp.FractureNetwork3d(fractures, tol=tol)
    else:
        network = pp.FractureNetwork3d(fractures)

    if return_all:
        return network, tess_fracs, tess_sgn
    else:
        return network


# ------------------------------------------------------------------------------#
