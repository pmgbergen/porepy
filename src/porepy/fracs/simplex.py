"""
Module for creating simplex grids with fractures.
"""
import time
import sys
import numpy as np
import meshio
import logging

import porepy as pp

from porepy.grids import constants
from porepy.grids.gmsh import gmsh_interface, mesh_2_grid
from porepy.fracs import fractures, tools
from porepy.utils.setmembership import unique_columns_tol, ismember_rows


logger = logging.getLogger(__name__)


def triangle_grid_embedded(network, f_name, **kwargs):
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
        f_name (str, optional): Filename for communication with gmsh.
            The config file for gmsh will be f_name.geo, with the grid output
            to f_name.msh. Defaults to dfn_network.
        **kwargs: Arguments sent to gmsh etc.

    Returns:
        list (length 3): For each dimension (2 -> 0), a list of all grids in
            that dimension.

    """

    out_file = _run_gmsh(f_name, in_3d=False, **kwargs)

    # The interface of meshio changed between versions 1 and 2. We make no
    # assumption on which version is installed here.
    if int(meshio.__version__[0]) < 2:
        pts, cells, _, cell_info, phys_names = meshio.gmsh_io.read(out_file)
        # Invert phys_names dictionary to map from physical tags to corresponding
        # physical names
        phys_names = {v[0]: k for k, v in phys_names.items()}
    else:
        mesh = meshio.read(out_file)

        pts = mesh.points
        cells = mesh.cells
        cell_info = mesh.cell_data
        phys_names = {v[0]: k for k, v in mesh.field_data.items()}

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


def _run_gmsh(file_name, **kwargs):

    verbose = kwargs.get("verbose", 1)
    if file_name[-4:] == ".geo" or file_name[-4:] == ".msh":
        file_name = file_name[:-4]

    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

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
    return out_file


def triangle_grid(points, edges, domain, constraints=None, **kwargs):
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
    constraints (np.array, optional): Index of edges that only act as constraints
        in meshing, but do not generate lower-dimensional grids.
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

    # Unified description of points and lines for domain, and fractures
    pts_all, lines, domain_pts = _merge_domain_fracs_2d(
        domain, points, edges, constraints
    )
    assert np.all(np.diff(lines[:2], axis=0) != 0)

    tol = kwargs.get("tol", 1e-4)
    pts_split, lines_split = _segment_2d_split(pts_all, lines, tol)

    # We find the end points that are shared by more than one intersection
    intersections = _find_intersection_points(lines_split)

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

    # File name for communication with gmsh
    file_name = kwargs.get("file_name", "gmsh_frac_file")
    kwargs.pop("file_name", str())

    in_file = file_name + ".geo"

    gw.write_geo(in_file)

    _run_gmsh(file_name, **kwargs)
    return triangle_grid_from_gmsh(file_name, **kwargs)

def line_grid_embedded(points, edges, domain, **kwargs):

    logger.info("Create 1d mesh embedded")

    # Unified description of points and lines for domain, and fractures
    constraints = np.empty(0, dtype=np.int)
    pts_all, lines, domain_pts = _merge_domain_fracs_2d(
        domain, points, edges, constraints
    )

    tol = kwargs.get("tol", 1e-4)
    pts_split, lines_split = _segment_2d_split(pts_all, lines, tol)

    # We find the end points that are shared by more than one intersection
    intersections = _find_intersection_points(lines_split)

    # Gridding size
    if "mesh_size_frac" in kwargs.keys():
        # Tag points at the domain corners
        logger.info("Determine mesh size")
        tm = time.time()
        mesh_size, pts_split, lines_split = tools.determine_mesh_size(
            pts_split, None, lines_split, **kwargs
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
        mesh_size=mesh_size,
        intersection_points=intersections,
        meshing_algorithm=meshing_algorithm,
        nd=1
    )

    # File name for communication with gmsh
    file_name = kwargs.get("file_name", "gmsh_frac_file")
    kwargs.pop("file_name", str())

    in_file = file_name + ".geo"

    gw.write_geo(in_file)

    _run_gmsh(file_name, **kwargs)

    return line_grid_from_gmsh(file_name, **kwargs)

def triangle_grid_from_gmsh(file_name, constraints=None, **kwargs):
    """ Generate a list of grids dimensions {2, 1, 0}, starting from a gmsh mesh.

    Parameters:
        file_name (str): Path to file of gmsh.msh specification.
        constraints (np.array, optional): Index of fracture lines that are
            constraints in the meshing, but should not have a lower-dimensional
            mesh. Defaults to empty.

    Returns:
        list of list of grids: grids in 2d, 1d and 0d. If no grids exist in a
            specified dimension, the inner list will be empty.

    """

    if constraints is None:
        constraints = np.empty(0, dtype=np.int)

    start_time = time.time()

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    out_file = file_name + ".msh"

    mesh = meshio.read(out_file)

    pts = mesh.points
    cells = mesh.cells
    cell_info = mesh.cell_data
    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v[0]: k for k, v in mesh.field_data.items()}

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()

    # Create grids from gmsh mesh.
    logger.info("Create grids of various dimensions")
    g_2d = mesh_2_grid.create_2d_grids(
        pts, cells, is_embedded=False, phys_names=phys_names, cell_info=cell_info
    )
    g_1d, _ = mesh_2_grid.create_1d_grids(
        pts,
        cells,
        phys_names,
        cell_info,
        line_tag=const.PHYSICAL_NAME_FRACTURES,
        constraints=constraints,
        **kwargs,
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

def line_grid_from_gmsh(file_name, constraints=None, **kwargs):
    """ Generate a list of grids dimensions {1, 0}, starting from a gmsh mesh.

    Parameters:
        file_name (str): Path to file of gmsh.msh specification.
        constraints (np.array, optional): Index of fracture lines that are
            constraints in the meshing, but should not have a lower-dimensional
            mesh. Defaults to empty.

    Returns:
        list of list of grids: grids in 1d and 0d. If no grids exist in a
            specified dimension, the inner list will be empty.

    """

    if constraints is None:
        constraints = np.empty(0, dtype=np.int)

    start_time = time.time()

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    out_file = file_name + ".msh"

    mesh = meshio.read(out_file)

    pts = mesh.points
    cells = mesh.cells
    cell_info = mesh.cell_data
    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v[0]: k for k, v in mesh.field_data.items()}

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()

    # Create grids from gmsh mesh.
    logger.info("Create grids of various dimensions")
    g_1d, _ = mesh_2_grid.create_1d_grids(
        pts,
        cells,
        phys_names,
        cell_info,
        line_tag=const.PHYSICAL_NAME_FRACTURES,
        constraints=constraints,
        **kwargs,
    )
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)
    grids = [g_1d, g_0d]

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

def tetrahedral_grid_from_gmsh(network, file_name, **kwargs):
    """ Generate a list of grids of dimensions {3, 2, 1, 0}, starting from a gmsh
    mesh.

    Parameters:
        network (pp.FractureNetwork3d): The network used to generate the gmsh
            input file.
        file_name (str): Path to file of gmsh.msh specification.

    Returns:
        list of list of grids: grids in 2d, 1d and 0d. If no grids exist in a
            specified dimension, the inner list will be empty.

    """

    start_time = time.time()
    # Verbosity level
    verbose = kwargs.get("verbose", 1)

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    file_name = file_name + ".msh"

    mesh = meshio.read(file_name)

    pts = mesh.points
    cells = mesh.cells
    cell_info = mesh.cell_data
    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v[0]: k for k, v in mesh.field_data.items()}

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


### Helper methods below


def _merge_domain_fracs_2d(dom, frac_p, frac_l, constraints):
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
    if frac_p is None:
        frac_p = np.zeros((2, 0))
        frac_l = np.zeros((2, 0))

    # Use constants set outside. If we ever
    const = constants.GmshConstants()

    if dom is None:
        dom_lines = np.empty((2, 0))
    elif isinstance(dom, dict):
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

    num_dom_lines = dom_lines.shape[1]  # Should be 4 for the dictionary case

    # The  lines will have all fracture-related tags set to zero.
    # The plan is to ignore these tags for the boundary and compartments,
    # so it should not matter
    dom_tags = const.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l, const.FRACTURE_TAG * np.ones(frac_l.shape[1])))
    is_constraint = np.in1d(np.arange(frac_l.shape[1]), constraints)
    frac_l[-1][is_constraint] = const.AUXILIARY_TAG

    # Merge the point arrays, compartment points first
    p = np.hstack((frac_p, dom_p))

    # Adjust index of fracture points to account for the compartment points
    dom_l[:2] += frac_p.shape[1]

    l = np.hstack((frac_l, dom_l)).astype(np.int)

    # Add a second tag as an identifier of each line.
    l = np.vstack((l, np.arange(l.shape[1])))

    return p, l, dom_p


def _find_intersection_points(lines):
    const = constants.GmshConstants()

    frac_id = np.ravel(lines[:2, lines[2] == const.FRACTURE_TAG])
    _, frac_ia, frac_count = np.unique(frac_id, True, False, True)

    # In the case we have auxiliary points remove do not create a 0d point in
    # case one intersects a single fracture. In the case of multiple fractures intersection
    # with an auxiliary point do consider the 0d.
    aux_id = lines[2] == const.AUXILIARY_TAG
    if np.any(aux_id):
        aux_id = np.ravel(lines[:2, aux_id])
        _, aux_ia, aux_count = np.unique(aux_id, True, False, True)

        # probably it can be done more efficiently but currently we rarely use the
        # auxiliary points in 2d
        for a in aux_id[aux_ia[aux_count > 1]]:
            # if a match is found decrease the frac_count only by one, this prevent
            # the multiple fracture case to be handle wrongly
            frac_count[frac_id[frac_ia] == a] -= 1

    return frac_id[frac_ia[frac_count > 1]]

def _segment_2d_split(pts_all, lines, tol):

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
    _, new_2_old, old_2_new = unique_columns_tol(li, tol=tol)
    lines = lines[:, new_2_old]

    assert np.all(np.diff(lines[:2], axis=0) != 0)

    # We split all fracture intersections so that the new lines do not
    # intersect, except possible at the end points
    logger.info("Remove edge crossings")
    tm = time.time()

    pts_split, lines_split = pp.intersections.split_intersecting_segments_2d(
        pts_all, lines, tol=tol
    )
    logger.info("Done. Elapsed time " + str(time.time() - tm))

    # Ensure unique description of points
    pts_split, _, old_2_new = unique_columns_tol(pts_split, tol=tol)
    lines_split[:2] = old_2_new[lines_split[:2]]
    to_remove = np.where(lines[0, :] == lines[1, :])[0]
    lines = np.delete(lines, to_remove, axis=1)

    # Remove lines with the same start and end-point.
    # This can be caused by L-intersections, or possibly also if the two
    # endpoints are considered equal under tolerance tol.
    remove_line_ind = np.where(np.diff(lines_split[:2], axis=0)[0] == 0)[0]
    lines_split = np.delete(lines_split, remove_line_ind, axis=1)

    return pts_split, lines_split

