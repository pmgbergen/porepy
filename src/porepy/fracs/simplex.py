"""
Module for creating simplex grids with fractures.
"""
import warnings
import time
import sys
import numpy as np
from meshio import gmsh_io
import logging

from porepy.grids import constants
from porepy.grids.gmsh import gmsh_interface, mesh_2_grid
from porepy.fracs import fractures, tools
import porepy.utils.comp_geom as cg
from porepy.utils.setmembership import unique_columns_tol, ismember_rows


logger = logging.getLogger(__name__)


def tetrahedral_grid(fracs=None, box=None, network=None, subdomains=[], **kwargs):
    """
    Create grids for a domain with possibly intersecting fractures in 3d.
    The function can be call through the wrapper function meshing.simplex_grid.

    Based on the specified fractures, the method computes fracture
    intersections if necessary, creates a gmsh input file, runs gmsh and reads
    the result, and then constructs grids in 3d (the whole domain), 2d (one for
    each individual fracture), 1d (along fracture intersections), and 0d
    (meeting between intersections).

    The fractures can be specified is terms of the keyword 'fracs' (either as
    numpy arrays or Fractures, see below), or as a ready-made FractureNetwork
    by the keyword 'network'. For fracs, the boundary of the domain must be
    specified as well, by 'box'. For a ready network, the boundary will be
    imposed if provided. For a network will use pre-computed intersection and
    decomposition if these are available (attributes 'intersections' and
    'decomposition').

    Parameters
    ----------
    fracs (list, optional): List of either pre-defined fractures, or
        np.ndarrays, (each 3xn) of fracture vertexes.
    box (dictionary, optional). Domain specification. Should have keywords
        xmin, xmax, ymin, ymax, zmin, zmax.
    network (fractures.FractureNetwork, optional): A FractureNetwork
        containing fractures.
    subdomain (list, optional): List of planes partitioning the 3d domain
        into subdomains. The planes are defined in the same way as fracs.

    The fractures should be specified either by a combination of fracs and
    box, or by network (possibly combined with box). See above.

    **kwargs: To be explored.

    Returns
    -------
    list (length 4): For each dimension (3 -> 0), a list of all grids in
        that dimension.

    Examples
    --------
    frac1 = np.array([[1,1,4,4], [1,4,4,1], [2,2,2,2]])
    frac2 = np.array([[2,2,2,2], [1,1,4,4], [1,4,4,1]])
    fracs = [frac1, frac2]
    domain = {'xmin': 0, 'ymin': 0, 'zmin': 0,
              'xmax': 5, 'ymax': 5, 'zmax': 5,}
    path_to_gmsh = .... # Set the sytem path to gmsh
    gb = tetrahedral_grid(fracs, domain, gmsh_path = path_to_gmsh)
    """

    # Verbosity level
    verbose = kwargs.get("verbose", 0)

    # File name for communication with gmsh
    file_name = kwargs.pop("file_name", "gmsh_frac_file")

    if network is None:

        frac_list = []
        if fracs is not None:
            for f in fracs:
                # Input can be either numpy arrays or predifined fractures. As a
                # guide, we treat f as a fracture if it has an attribute p which is
                # a numpy array.
                # If f turns out not to be a fracture, strange errors will result
                # as the further program tries to access non-existing methods.
                # The correct treatment here would be several
                # isinstance-statements, but that became less than elegant. To
                # revisit.
                if hasattr(f, "p") and isinstance(f.p, np.ndarray):
                    frac_list.append(f)
                else:
                    # Convert the fractures from numpy representation to our 3D
                    # fracture data structure.
                    frac_list.append(fractures.Fracture(f))

        # Combine the fractures into a network
        network = fractures.FractureNetwork(
            frac_list, verbose=verbose, tol=kwargs.get("tol", 1e-4)
        )
    # Add any subdomain boundaries:
    network.add_subdomain_boundaries(subdomains)
    # Impose external boundary. If box is None, a domain size somewhat larger
    # than the network will be assigned.
    if box is not None or not network.bounding_box_imposed:
        network.impose_external_boundary(box)
    # Find intersections and split them, preparing the way for dumping the
    # network to gmsh
    if not network.has_checked_intersections:
        network.find_intersections()
    else:
        logger.info("Use existing intersections")

    # If fields mesh_size_frac and mesh_size_min are provided, try to estimate mesh sizes.
    mesh_size_frac = kwargs.get("mesh_size_frac", None)
    mesh_size_min = kwargs.get("mesh_size_min", None)
    mesh_size_bound = kwargs.get("mesh_size_bound", None)
    if mesh_size_frac is not None and mesh_size_min is not None:
        if mesh_size_bound is None:
            mesh_size_bound = mesh_size_frac
        network.insert_auxiliary_points(mesh_size_frac, mesh_size_min, mesh_size_bound)
        # In this case we need to recompute intersection decomposition anyhow.
        network.split_intersections()

    if not hasattr(network, "decomposition"):
        network.split_intersections()
    else:
        logger.info("Use existing decomposition")

    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

    network.to_gmsh(in_file, **kwargs)

    gmsh_opts = kwargs.get("gmsh_opts", {})
    gmsh_verbose = kwargs.get("gmsh_verbose", verbose)
    gmsh_opts["-v"] = gmsh_verbose
    gmsh_status = gmsh_interface.run_gmsh(in_file, out_file, dims=3, **gmsh_opts)

    return tetrahedral_grid_from_gmsh(file_name, network, **kwargs)


def triangle_grid_embedded(
    network,
    find_isect=True,
    f_name="dfn_network",
    mesh_size_frac=None,
    mesh_size_min=None,
    mesh_size_bound=None,
    **kwargs
):
    """ Create triangular (2D) grid of a domain embedded in 3D space, without
    meshing the 3D volume.

    The resulting grid can be used in a DFN model. The grid will be fully
    conforming along intersections between fractures.

    This function produces a set of grids for fractures and lower-dimensional
    objects, but it does nothing to merge the grids. To create a GridBucket,
    use the function fracs.meshing.dfn instead, with the option
    conforming=True.

    To set the mesh size, use parameters mesh_size_frac and mesh_size_min, to represent the
    ideal and minimal mesh size sent to gmsh. For more details, see the gmsh
    manual on how to set mesh sizes.

    Parameters:
        network (FractureNetwork): To be meshed.
        find_isect (boolean, optional): If True (default), the network will
            search for intersections among fractures. Set False if
            network.find_intersections() already has been called.
        f_name (str, optional): Filename for communication with gmsh.
            The config file for gmsh will be f_name.geo, with the grid output
            to f_name.msh. Defaults to dfn_network.
        mesh_size_frac (double, optional): Target mesh size sent to gmsh. If not
            provided, gmsh will do its best to decide on the mesh size.
        mesh_size_min (double, optional): Minimal mesh size sent to gmsh. If not
            provided, gmsh will do its best to decide on the mesh size.
        **kwargs: Arguments sent to gmsh etc.

    Returns:
        list (length 3): For each dimension (2 -> 0), a list of all grids in
            that dimension.

    """

    verbose = 1

    if find_isect:
        network.find_intersections()

    # If fields mesh_size_frac and mesh_size_min are provided, try to estimate mesh sizes.
    if mesh_size_frac is not None and mesh_size_min is not None:
        if mesh_size_bound is None:
            mesh_size_bound = mesh_size_frac
        network.insert_auxiliary_points(mesh_size_frac, mesh_size_min, mesh_size_bound)
        # In this case we need to recompute intersection decomposition anyhow.
        network.split_intersections()

    if not hasattr(network, "decomposition"):
        network.split_intersections()
    else:
        logger.info("Use existing decomposition")

    pts, cells, cell_info, phys_names = _run_gmsh(
        f_name, network, in_3d=False, **kwargs
    )
    g_2d = mesh_2_grid.create_2d_grids(
        pts,
        cells,
        is_embedded=True,
        phys_names=phys_names,
        cell_info=cell_info,
        network=network,
    )
    g_1d, _ = mesh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)

    grids = [g_2d, g_1d, g_0d]

    if verbose > 0:
        logger.info("\n")
        for g_set in grids:
            if len(g_set) > 0:
                s = (
                    "Created "
                    + str(len(g_set))
                    + " "
                    + str(g_set[0].dim)
                    + "-d grids with "
                )
                num = 0
                for g in g_set:
                    num += g.num_cells
                s += str(num) + " cells"
                logger.info(s)
        logger.info("\n")

    return grids


def _run_gmsh(file_name, network, **kwargs):

    verbose = kwargs.get("verbose", 1)
    if file_name[-4:] == ".geo" or file_name[-4:] == ".msh":
        file_name = file_name[:-4]

    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

    if not hasattr(network, "decomposition"):
        network.split_intersections()
    else:
        logger.info("Use existing decomposition")

    network.to_gmsh(in_file, **kwargs)

    gmsh_opts = kwargs.get("gmsh_opts", {})
    gmsh_verbose = kwargs.get("gmsh_verbose", verbose)
    gmsh_opts["-v"] = gmsh_verbose
    gmsh_status = gmsh_interface.run_gmsh(in_file, out_file, dims=3, **gmsh_opts)

    if verbose > 0:
        if gmsh_status == 0:
            logger.info("Gmsh processed file successfully")
        else:
            logger.error("Gmsh failed with status " + str(gmsh_status))
            sys.exit()

    pts, cells, _, cell_info, phys_names = gmsh_io.read(out_file)

    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v[0]: k for k, v in phys_names.items()}

    return pts, cells, cell_info, phys_names


def triangle_grid(fracs, domain, **kwargs):
    """
    Generate a gmsh grid in a 2D domain with fractures.

    The function uses modified versions of pygmsh and mesh_io,
    both downloaded from github.

    To be added:
    Functionality for tuning gmsh, including grid size, refinements, etc.

    Parameters
    ----------
    fracs: (dictionary) Two fields: points (2 x num_points) np.ndarray,
        edges (2 x num_lines) connections between points, defines fractures.
    box: (dictionary) keys xmin, xmax, ymin, ymax, [together bounding box
        for the domain]
    **kwargs: To be explored.

    Returns
    -------
    list (length 3): For each dimension (2 -> 0), a list of all grids in
        that dimension.

    Examples
    p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
    lines = np.array([[0, 2], [1, 3]])
    char_h = 0.5 * np.ones(p.shape[1])
    tags = np.array([1, 3])
    fracs = {'points': p, 'edges': lines}
    box = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2}
    g = triangle_grid(fracs, box)

    """
    logger.info("Create 2d mesh")

    # Verbosity level
    verbose = kwargs.get("verbose", 1)

    # File name for communication with gmsh
    file_name = kwargs.get("file_name", "gmsh_frac_file")
    kwargs.pop("file_name", str())

    tol = kwargs.get("tol", 1e-4)

    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

    # Pick out fracture points, and their connections
    frac_pts = fracs["points"]
    frac_con = fracs["edges"]

    # Unified description of points and lines for domain, and fractures
    pts_all, lines, domain_pts = __merge_domain_fracs_2d(domain, frac_pts, frac_con)

    # Snap to underlying grid before comparing points
    pts_all = cg.snap_to_grid(pts_all, tol)

    assert np.all(np.diff(lines[:2], axis=0) != 0)

    # Ensure unique description of points
    pts_all, _, old_2_new = unique_columns_tol(pts_all, tol=tol)
    lines[:2] = old_2_new[lines[:2]]
    to_remove = np.where(lines[0, :] == lines[1, :])[0]
    lines = np.delete(lines, to_remove, axis=1)

    # In some cases the fractures and boundaries impose the same constraint
    # twice, although it is not clear why. Avoid this by uniquifying the lines.
    # This may disturb the line tags in lines[2], but we should not be
    # dependent on those.
    li = np.sort(lines[:2], axis=0)
    li_unique, new_2_old, old_2_new = unique_columns_tol(li)
    lines = lines[:, new_2_old]

    assert np.all(np.diff(lines[:2], axis=0) != 0)

    # We split all fracture intersections so that the new lines do not
    # intersect, except possible at the end points
    logger.info("Remove edge crossings")
    tm = time.time()
    pts_split, lines_split = cg.remove_edge_crossings(pts_all, lines, tol=tol)
    logger.info("Done. Elapsed time " + str(time.time() - tm))

    # Ensure unique description of points
    pts_split = cg.snap_to_grid(pts_split, tol)
    pts_split, _, old_2_new = unique_columns_tol(pts_split, tol=tol)
    lines_split[:2] = old_2_new[lines_split[:2]]
    to_remove = np.where(lines[0, :] == lines[1, :])[0]
    lines = np.delete(lines, to_remove, axis=1)

    # Remove lines with the same start and end-point.
    # This can be caused by L-intersections, or possibly also if the two
    # endpoints are considered equal under tolerance tol.
    remove_line_ind = np.where(np.diff(lines_split[:2], axis=0)[0] == 0)[0]
    lines_split = np.delete(lines_split, remove_line_ind, axis=1)

    # TODO: This operation may leave points that are not referenced by any
    # lines. We should probably delete these.

    # We find the end points that are shared by more than one intersection
    intersections = __find_intersection_points(lines_split)

    # Gridding size

    if "mesh_size_frac" in kwargs.keys():
        # Tag points at the domain corners
        logger.info("Determine mesh size")
        tm = time.time()
        boundary_pt_ind = ismember_rows(pts_split, domain_pts, sort=False)[0]
        mesh_size, pts_split, lines_split = tools.determine_mesh_size(
            pts_split, boundary_pt_ind, lines_split, **kwargs
        )
        logger.info("Done. Elapsed time " + str(time.time() - tm))
    else:
        mesh_size = None

    # gmsh options

    meshing_algorithm = kwargs.get("meshing_algorithm")

    # Create a writer of gmsh .geo-files
    gw = gmsh_interface.GmshWriter(
        pts_split,
        lines_split,
        domain=domain,
        mesh_size=mesh_size,
        intersection_points=intersections,
        meshing_algorithm=meshing_algorithm,
    )
    gw.write_geo(in_file)

    triangle_grid_run_gmsh(file_name, **kwargs)
    return triangle_grid_from_gmsh(file_name, **kwargs)


# ------------------------------------------------------------------------------#


def triangle_grid_run_gmsh(file_name, **kwargs):

    if file_name.endswith(".geo"):
        file_name = file_name[:-4]
    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

    # Verbosity level
    verbose = kwargs.get("verbose", 1)
    gmsh_verbose = kwargs.get("gmsh_verbose", verbose)
    gmsh_opts = {"-v": gmsh_verbose}

    logger.info("Run gmsh")
    tm = time.time()
    # Run gmsh
    gmsh_status = gmsh_interface.run_gmsh(in_file, out_file, dims=2, **gmsh_opts)

    if gmsh_status == 0:
        logger.info("Gmsh processed file successfully")
        logger.info("Elapsed time " + str(time.time() - tm))
    else:
        logger.error("Gmsh failed with status " + str(gmsh_status))


# ------------------------------------------------------------------------------#


def triangle_grid_from_gmsh(file_name, **kwargs):

    start_time = time.time()

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    out_file = file_name + ".msh"

    # Verbosity level
    verbose = kwargs.get("verbose", 1)

    pts, cells, _, cell_info, phys_names = gmsh_io.read(out_file)

    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names.
    # As of meshio 1.10, the value of the physical name is defined as a numpy
    # array, with the first item being the tag, the second the dimension.
    phys_names = {v[0]: k for k, v in phys_names.items()}

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()

    # Create grids from gmsh mesh.
    logger.info("Create grids of various dimensions")
    g_2d = mesh_2_grid.create_2d_grids(pts, cells, is_embedded=False)
    g_1d, _ = mesh_2_grid.create_1d_grids(
        pts, cells, phys_names, cell_info, line_tag=const.PHYSICAL_NAME_FRACTURES
    )
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)
    grids = [g_2d, g_1d, g_0d]

    logger.info(
        "Grid creation completed. Elapsed time " + str(time.time() - start_time)
    )

    for g_set in grids:
        if len(g_set) > 0:
            s = (
                "Created "
                + str(len(g_set))
                + " "
                + str(g_set[0].dim)
                + "-d grids with "
            )
            num = 0
            for g in g_set:
                num += g.num_cells
            s += str(num) + " cells"
            logger.info(s)

    return grids


# ------------------------------------------------------------------------------#


def tetrahedral_grid_from_gmsh(file_name, network, **kwargs):

    start_time = time.time()
    # Verbosity level
    verbose = kwargs.get("verbose", 1)

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    file_name = file_name + ".msh"

    pts, cells, _, cell_info, phys_names = gmsh_io.read(file_name)

    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v[0]: k for k, v in phys_names.items()}

    # Call upon helper functions to create grids in various dimensions.
    # The constructors require somewhat different information, reflecting the
    # different nature of the grids.
    g_3d = mesh_2_grid.create_3d_grids(pts, cells)
    g_2d = mesh_2_grid.create_2d_grids(
        pts,
        cells,
        is_embedded=True,
        phys_names=phys_names,
        cell_info=cell_info,
        network=network,
    )
    g_1d, _ = mesh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)

    grids = [g_3d, g_2d, g_1d, g_0d]

    if verbose > 0:
        print("\n")
        print("Grid creation completed. Elapsed time " + str(time.time() - start_time))
        print("\n")
        for g_set in grids:
            if len(g_set) > 0:
                s = (
                    "Created "
                    + str(len(g_set))
                    + " "
                    + str(g_set[0].dim)
                    + "-d grids with "
                )
                num = 0
                for g in g_set:
                    num += g.num_cells
                s += str(num) + " cells"
                print(s)
        print("\n")

    return grids


# -----------------------------------------------------------------------------#


def __merge_domain_fracs_2d(dom, frac_p, frac_l):
    """
    Merge fractures, domain boundaries and lines for compartments.
    The unified description is ready for feeding into meshing tools such as
    gmsh

    Parameters:
    dom: dictionary defining domain. fields xmin, xmax, ymin, ymax
    frac_p: np.ndarray. Points used in fracture definition. 2 x num_points.
    frac_l: np.ndarray. Connection between fracture points. 2 x num_fracs

    returns:
    p: np.ndarary. Merged list of points for fractures, compartments and domain
        boundaries.
    l: np.ndarray. Merged list of line connections (first two rows), tag
        identifying which type of line this is (third row), and a running index
        for all lines (fourth row)
    """

    # Use constants set outside. If we ever
    const = constants.GmshConstants()

    if isinstance(dom, dict):
        # First create lines that define the domain
        x_min = dom["xmin"]
        x_max = dom["xmax"]
        y_min = dom["ymin"]
        y_max = dom["ymax"]
        dom_p = np.array([[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]])
        dom_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T
    else:
        dom_p = dom
        tmp = np.arange(dom_p.shape[1])
        dom_lines = np.vstack((tmp, (tmp + 1) % dom_p.shape[1]))

    num_dom_lines = dom_lines.shape[1]  # Should be 4

    # The  lines will have all fracture-related tags set to zero.
    # The plan is to ignore these tags for the boundary and compartments,
    # so it should not matter
    dom_tags = const.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l, const.FRACTURE_TAG * np.ones(frac_l.shape[1])))

    # Merge the point arrays, compartment points first
    p = np.hstack((dom_p, frac_p))

    # Adjust index of fracture points to account for the compartment points
    frac_l[:2] += dom_p.shape[1]

    l = np.hstack((dom_l, frac_l)).astype("int")

    # Add a second tag as an identifier of each line.
    l = np.vstack((l, np.arange(l.shape[1])))

    return p, l, dom_p


def __find_intersection_points(lines):
    const = constants.GmshConstants()
    frac_id = np.ravel(lines[:2, lines[2] == const.FRACTURE_TAG])
    _, ia, count = np.unique(frac_id, True, False, True)
    return frac_id[ia[count > 1]]
