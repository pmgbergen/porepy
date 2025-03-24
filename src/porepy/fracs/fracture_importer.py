from __future__ import annotations

import csv
from typing import Optional, Union

import gmsh
import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.utils import pts_edges_to_linefractures


def network_3d_from_csv(
    file_name: str, has_domain: bool = True, tol: float = 1e-4, **kwargs
) -> FractureNetwork3d:
    """Create the fracture network from a set of 3d fractures stored in a CSV file.

    In the CSV file, we assume the following structure:

        The first line (optional) describes the domain as a cuboid with ``X_MIN,
        Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX``.

        The other lines descibe the `N` fractures as a list of points ``P0_X, P0_Y,
        P0_Z, ..., PN_X, PN_Y, PN_Z``.

    Lines starting with ``#`` will be ignored.

    Parameters:
        file_name: Path to the CSV file.
        has_domain: ``default=True``

            Whether the first line in the CSV file specifies the domain.
        tol: ``default=1e-4``

            Geometric tolerance used in the computations.
        **kwargs: Keyword arguments passed to
            :class:`~porepy.fracs.plane_fracture.PlaneFracture` and
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

    Returns:
        Three-dimensional fracture network object.

    """

    # The first line of the csv file defines the bounding box for the domain.
    frac_list = []
    # Extract the data from the csv file.
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")
        # Read the domain first.
        if has_domain:
            read_domain = False

            while not read_domain:
                line = next(spam_reader)
                if line[0][0] == "#":
                    continue
                else:
                    data = np.asarray(line, dtype=float)
                    bbox = {
                        "xmin": data[0],
                        "xmax": data[3],
                        "ymin": data[1],
                        "ymax": data[4],
                        "zmin": data[2],
                        "zmax": data[5],
                    }
                    domain = pp.Domain(bbox)
                    read_domain = True

        for row in spam_reader:
            # If the line starts with a '#', we consider this a comment.
            if len(row) == 0 or row[0][0] == "#":
                continue

            # Read the points
            pts = np.asarray(row, dtype=float)
            if not pts.size % 3 == 0:
                raise ValueError("Points are always 3d")

            # Skip empty lines. Useful if the file ends with a blank line.
            if pts.size == 0:
                continue

            check_convexity = kwargs.get("check_convexity", True)

            frac_list.append(
                pp.PlaneFracture(
                    pts.reshape((3, -1), order="F"), check_convexity=check_convexity
                )
            )

    # Create the network
    if has_domain:
        fn = pp.create_fracture_network(frac_list, domain, tol=tol)
        assert isinstance(fn, FractureNetwork3d)  # needed to please mypy
        return fn
    else:
        fn = pp.create_fracture_network(frac_list, tol=tol)
        assert isinstance(fn, FractureNetwork3d)  # needed to please mypy
        return fn


def elliptic_network_3d_from_csv(
    file_name: str, has_domain: bool = True, tol: float = 1e-4, degrees: bool = False
) -> pp.fracture_network:
    """Create fracture network from a set of elliptic fractures stored in a CSV file.

    In the CSV file, we assume the following structure:

        The first line (optional) describes the domain as a cuboid with ``X_MIN,
        Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX``.

        The other lines describe the ``N`` fractures as a elliptic fractures with
        ``center_x, center_y, center_z, major_axis, minor_axis, major_axis_angle,
        strike_angle, dip_angle, num_points``.

    See :meth:`~porepy.fracs.plane_fracture.create_elliptic_fracture` for
    information about the parameters

    Lines that start with a # are ignored.

    Parameters:
        file_name: Path to the CSV file.
        has_domain: ``default=True``

            Indicates whether the first line in the CSV file specifies the domain.
        tol: ``default=1e-4``

            Geometric tolerance used in the computations.
        degrees: ``default=False``

            Indicates whether the angles are given in degrees or radians.

    Returns:
        3D fracture network object with elliptic fractures.

    """

    # The first line of the csv file defines the bounding box for the domain.
    frac_list = []
    # Extract the data from the csv file.
    with open(file_name, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first.
        if has_domain:
            bbox_as_array = np.asarray(next(spam_reader), dtype=float)
            bbox = {
                "xmin": bbox_as_array[0],
                "xmax": bbox_as_array[3],
                "ymin": bbox_as_array[1],
                "ymax": bbox_as_array[4],
                "zmin": bbox_as_array[2],
                "zmax": bbox_as_array[5],
            }
            domain = pp.Domain(bbox)

        for row in spam_reader:
            # If the line starts with a '#', we consider this a comment.
            if row[0][0] == "#":
                continue

            # Read the data.
            data = np.asarray(row, dtype=float)
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
                pp.create_elliptic_fracture(
                    centers, maj_ax, min_ax, maj_ax_ang, strike_ang, dip_ang, num_points
                )
            )
    # Create the network.
    if has_domain:
        return pp.create_fracture_network(frac_list, domain, tol=tol)
    else:
        return pp.create_fracture_network(frac_list, tol=tol)


def network_2d_from_csv(
    f_name: str,
    tagcols: Optional[ArrayLike] = None,
    tol: float = 1e-8,
    max_num_fracs: Optional[int] = None,
    polyline: bool = False,
    return_frac_id: bool = False,
    domain: Optional[pp.Domain] = None,
    **kwargs,
) -> Union[FractureNetwork2d, tuple[FractureNetwork2d, np.ndarray]]:
    """Create 2D fracture network from a CSV file.

    In the CSV file, we assume one of the two following structures:

    1. ``FID, START_X, START_Y, END_X, END_Y``.
    2. ``FID, PT_X, PT_Y``.

    Format 1 is used to describe fractures consisting of a straight line. ``FID`` is
    the fracture id, ``START_X`` and ``START_Y`` are the abscissa and coordinate of
    the starting point, and ``END_X`` and ``END_Y`` are the abscissa and coordinate
    of the ending point.

    Format 2 can be used to describe polyline fractures, where each row in the file
    represents separate points, points with the same ``FID`` will be assigned to the
    same fracture *in the order specified in the file*.

    To change the delimiter from the default comma, use kwargs passed to
    :obj:`numpy.genfromtxt`.

    The CSV file is assumed to have a header of 1 line. To change this number,
    use kwargs ``skip_header``.

    Parameters:
        f_name: Path to the CSV file.
        tagcols: ``dtype=np.int32, default=None``

            Column index where fracture tags are stored. 0-offset. Defaults to no
            columns.
        tol: ``default=1e-8``

            Tolerance for merging points with almost equal coordinates.
        max_num_fracs: ``default=None``

            Maximum number of fractures included, counting from the start of the
            file. Defaults to inclusion of all fractures.
        polyline: ``default=False``

            Indicates if the fractures are given as a polyline, i.e., via format 2.
        return_frac_id: ``default=False``

            Indicates whether to return the fracture IDs or not.
        domain:  ``default=None``

            Domain specification. If not given, the domain will be set as the
            bounding box of the set of fractures.
        **kwargs: keyword arguments passed to :obj:`numpy.genfromtxt`.

    Raises:
        ValueError:
            If a fracture of a single point is specified.

    Returns:
        The 2D fracture network. If ``return_frac_id=True``,
        also the fracture IDs are returned as a numpy array of ``shape=(num_fracs,)``.

    """
    npargs = {}
    # EK: Should these really be explicit keyword arguments?
    npargs["delimiter"] = kwargs.get("delimiter", ",")
    npargs["skip_header"] = kwargs.get("skip_header", 1)

    # Extract the data from the csv file
    data = np.genfromtxt(f_name, **npargs)
    # Shortcut if no data is loaded
    if data.size == 0:
        # We still consider the possibility that a domain is given.
        if return_frac_id:
            return FractureNetwork2d(domain=domain, tol=tol), np.empty(0)
        else:
            return FractureNetwork2d(domain=domain, tol=tol)
    data = np.atleast_2d(data)

    # Consider subset of fractures if asked for.
    if max_num_fracs is not None:
        if max_num_fracs == 0:
            if return_frac_id:
                return FractureNetwork2d(tol=tol), np.empty(0)
            else:
                return FractureNetwork2d(tol=tol)
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

        edges = edges.astype(int)
        edges_frac_id = edges_frac_id.astype(int)

    else:
        # Let the edges correspond to the ordering of the fractures
        edges = np.vstack(
            (np.arange(0, 2 * num_fracs, 2), np.arange(1, 2 * num_fracs, 2))
        )
        # Fracture id is the first column of data
        edges_frac_id = data[:, 0]
        if tagcols is not None:
            edges = np.vstack((edges, data[:, tagcols].T))  # type: ignore

    if domain is None:
        overlap = kwargs.get("domain_overlap", 0)
        bbox = pp.domain.bounding_box_of_point_cloud(pts, overlap)
        domain = pp.Domain(bbox)

    pts, _, old_2_new = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)

    edges[:2] = old_2_new[edges[:2].astype(int)]

    to_remove = np.where(edges[0, :] == edges[1, :])[0]
    edges = np.delete(edges, to_remove, axis=1).astype(int)

    if not np.all(np.diff(edges[:2], axis=0) != 0):
        raise ValueError

    fractures = pts_edges_to_linefractures(pts, edges)
    network = pp.create_fracture_network(fractures, domain, tol=tol)
    assert isinstance(network, FractureNetwork2d)  # for mypy

    if return_frac_id:
        edges_frac_id = np.delete(edges_frac_id, to_remove)
        return network, edges_frac_id.astype(int)
    else:
        return network


def dfm_from_gmsh(file_name: str, dim: int, **kwargs) -> pp.MixedDimensionalGrid:
    """Generate a mixed-dimensional grid from a gmsh file.

    If the provided extension of the input file for gmsh is ``.geo`` (not ``.msh``),
    gmsh will be called to generate the mesh before the mixed-dimensional grid is
    constructed.

    Parameters:
        file_name:
            Name of gmsh *in* and *out* file. Should have extension ``.geo`` or
            ``.msh``. In the former case, gmsh will be called upon to generate the
            mesh before the mixed-dimensional mesh is constructed.
        dim:
            Dimension of the problem. Should be 2 or 3.
        **kwargs:
            Optional keyword arguments.

            See :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh` for
            ``dim=2``,
            and :meth:`~porepy.fracs.fracture_network_3d.FractureNetwork3d.mesh` for
            ``dim=3``.

    Returns:
        Mixed-dimensional grid as contained in the gmsh file.

    """

    # run gmsh to create .msh file if
    if file_name[-4:] == ".msh":
        out_file = file_name
    else:
        if file_name[-4:] == ".geo":
            file_name = file_name[:-4]
        in_file = file_name + ".geo"
        out_file = file_name + ".msh"

        # initialize gmsh
        gmsh.initialize()
        # Reduce verbosity
        gmsh.option.setNumber("General.Verbosity", 3)
        # read the specified file.
        gmsh.merge(in_file)

        # Generate mesh, write
        gmsh.model.mesh.generate(dim=dim)
        gmsh.write(out_file)

        # Wipe Gmsh's memory
        gmsh.finalize()

    if dim == 2:
        subdomains = pp.fracs.simplex.triangle_grid_from_gmsh(out_file, **kwargs)
    elif dim == 3:
        subdomains = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
            file_name=out_file, **kwargs
        )
    else:
        raise ValueError(f"Unknown dimension, dim: {dim}")
    return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)


def dfm_3d_from_fab(
    file_name: str,
    tol: float = 1e-4,
    domain: Optional[pp.Domain] = None,
    return_domain: bool = False,
    **mesh_kwargs,
) -> Union[pp.MixedDimensionalGrid, tuple[pp.MixedDimensionalGrid, pp.Domain]]:
    """Create the mdg from a set of 3d fractures stored in a fab file and domain.

    Parameters:
        file_name: Name of the file.
        tol: ``default=1e-4``

            Tolerance for the methods.
        domain: ``default=None``

            Domain specification. If not given, the bounding box is considered.
        return_domain:  ``default=False``

            Whether to return the domain.
        mesh_kwargs: ``kwargs`` for the gridding, see e.g.,
            :meth:`~porepy.fracs.simplex.tetrahedral_grid_from_gmsh`.

    Returns:
        The resulting mixed-dimensional grid, and if :attr:`return_domain` is ``True``,
        also the domain.

    """

    network = network_3d_from_fab(file_name, return_all=False, tol=tol)
    assert isinstance(network, FractureNetwork3d)

    # Compute the domain if not given
    if domain is None:
        domain = pp.Domain(network.bounding_box())

    network.domain = domain
    mdg = network.mesh(mesh_kwargs)

    if return_domain:
        return mdg, domain
    else:
        return mdg


def network_3d_from_fab(
    f_name: str, return_all: bool = False, tol: Optional[float] = None
) -> Union[FractureNetwork3d, tuple[FractureNetwork3d, list[np.ndarray], np.ndarray]]:
    """Create 3D fracture network from a ``.fab`` file, as specified by FracMan.

    The filter is based on the ``.fab``-files available at the time of writing and
    may not cover all options available.

    Note:
        The function also reads various other information of unknown usefulness,
        see implementation for details. This information is currently not returned.

    Parameters:
        f_name: Path to the ``.fab`` file.
        return_all: ``default=False``

            Whether to return additional information (see the Returns section).
        tol: ``default=None``

            Tolerance passed on instantiation of the returned
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

    Returns:
        3D fracture network, and if ``return_all=True`` also

        - A list of numpy arrays of ``shape=(nd, num_points)``, where each
          item of the list contains the fractures cut by the domain boundary,
          represented by their ``num_points`` vertexes.
        - A numpy array, where for each element in the list of numpy arrays from
          above, a :math:`\\pm 1` is associated, establishing which boundary the
          fracture is on.

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

    fractures = [pp.PlaneFracture(f) for f in fracs]
    if tol is not None:
        network = pp.create_fracture_network(fractures, tol=tol)
    else:
        network = pp.create_fracture_network(fractures)
    assert isinstance(network, FractureNetwork3d)

    if return_all:
        return network, tess_fracs, tess_sgn
    else:
        return network
