"""
Module for creating simplex grids with fractures.
"""
import logging
import time
from typing import Any, Dict, Tuple

import meshio
import numpy as np

import porepy as pp

from . import msh_2_grid
from .gmsh_interface import PhysicalNames

logger = logging.getLogger(__name__)
module_sections = ["gridding"]


@pp.time_logger(sections=module_sections)
def triangle_grid_embedded(file_name):
    """Create triangular (2D) grid of a domain embedded in 3D space, without
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
        file_name (str, optional): Filename for communication with gmsh.
            The config file for gmsh will be f_name.geo, with the grid output
            to f_name.msh. Defaults to dfn_network.

    Returns:
        list (length 3): For each dimension (2 -> 0), a list of all grids in
            that dimension.

    """

    if file_name[-4:] == ".geo" or file_name[-4:] == ".msh":
        file_name = file_name[:-4]

    out_file = file_name + ".msh"

    pts, cells, cell_info, phys_names = _read_gmsh_file(out_file)

    g_2d = msh_2_grid.create_2d_grids(
        pts,
        cells,
        is_embedded=True,
        phys_names=phys_names,
        cell_info=cell_info,
    )
    g_1d, _ = msh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = msh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)

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


@pp.time_logger(sections=module_sections)
def triangle_grid_from_gmsh(file_name, constraints=None, **kwargs):
    """Generate a list of grids dimensions {2, 1, 0}, starting from a gmsh mesh.

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
        constraints = np.empty(0, dtype=int)

    start_time = time.time()

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    out_file = file_name + ".msh"

    pts, cells, cell_info, phys_names = _read_gmsh_file(out_file)

    # Create grids from gmsh mesh.
    logger.info("Create grids of various dimensions")
    g_2d = msh_2_grid.create_2d_grids(
        pts, cells, is_embedded=False, phys_names=phys_names, cell_info=cell_info
    )
    g_1d, _ = msh_2_grid.create_1d_grids(
        pts,
        cells,
        phys_names,
        cell_info,
        line_tag=PhysicalNames.FRACTURE.value,
        constraints=constraints,
        **kwargs,
    )
    g_0d = msh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)
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


@pp.time_logger(sections=module_sections)
def line_grid_from_gmsh(file_name, constraints=None, **kwargs):
    """Generate a list of grids dimensions {1, 0}, starting from a gmsh mesh.

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
        constraints = np.empty(0, dtype=int)

    start_time = time.time()

    if file_name.endswith(".msh"):
        file_name = file_name[:-4]
    out_file = file_name + ".msh"

    pts, cells, cell_info, phys_names = _read_gmsh_file(out_file)

    # Create grids from gmsh mesh.
    logger.info("Create grids of various dimensions")
    g_1d, _ = msh_2_grid.create_1d_grids(
        pts,
        cells,
        phys_names,
        cell_info,
        line_tag=PhysicalNames.FRACTURE.value,
        constraints=constraints,
        **kwargs,
    )
    g_0d = msh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)
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


@pp.time_logger(sections=module_sections)
def tetrahedral_grid_from_gmsh(file_name, constraints=None, **kwargs):
    """Generate a list of grids of dimensions {3, 2, 1, 0}, starting from a gmsh
    mesh.

    Parameters:
        file_name (str): Path to file of gmsh.msh specification.
        TODO: Line tag is unused. Maybe surface_tag replaces it?? Fix docs.
            This documentation is copied from mesh_2_grid.create_2d_grids().
        constraints (np.array, optional): Array with lists of lines that should not
            become grids. The array items should match the INDEX in line_tag, see above.

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

    pts, cells, cell_info, phys_names = _read_gmsh_file(file_name)

    # Call upon helper functions to create grids in various dimensions.
    # The constructors require somewhat different information, reflecting the
    # different nature of the grids.
    g_3d = msh_2_grid.create_3d_grids(pts, cells)
    g_2d = msh_2_grid.create_2d_grids(
        pts,
        cells,
        is_embedded=True,
        phys_names=phys_names,
        cell_info=cell_info,
        constraints=constraints,
    )

    g_1d, _ = msh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = msh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)

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


@pp.time_logger(sections=module_sections)
def _read_gmsh_file(
    file_name: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[int, str]]:
    """
    Read a gmsh .msh file, and convert the result to a format that is compatible with
    the porepy functionality for mesh processing.

    Args:
        file_name (str): Name of the file to be processed.

    Returns:
        pts (np.ndarray, npt x dim): Coordinates of all vertexes in the grid.
        cells (dict): Mapping between cells of different shapes, and the index of
            vertexes for the cells.
        cell_info (dict): Mapping between cells of different shapes, and the tags of
            each cell.
        phys_names (TYPE): Mapping from gmsh tags to the physical names assigned in the
            gmsh .geo file.

    """
    mesh = meshio.read(file_name)

    pts = mesh.points
    cells = mesh.cells
    cell_info = mesh.cell_data
    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    # The first
    phys_names = {v[0]: k for k, v in mesh.field_data.items()}

    # The API of meshio is constantly changing; this if-else is needed to take care
    # of different generations of meshio.
    if isinstance(cells, list):
        # This is tested for meshio v4
        # The cells are stored as a list of namedtuples, consisting of an attribute
        # 'type' which is the type of the cell (line, triangle etc.), and an attribute
        # 'data', which gives the vertex numbers for each cell.
        # We will convert this into a simpler dictionary, that maps the type to an
        # array of vertexes.
        # Moreover, the cell_info, which gives the tags of each cell, is stored in a
        # list, with ordering equal to that of the cells. This list will again be split
        # into a dictionary with the cell type as keys, and tags as values.

        # Initialize the dictionaries to be constructed
        tmp_cells: Dict[str, Any] = {}
        tmp_info: Dict[str, Any] = {}
        keys = set([cb.type for cb in cells])
        for key in keys:
            tmp_cells[key] = []
            tmp_info[key] = []

        # Loop simulatneously over cells and the cell info; process data
        for cb, info in zip(cells, cell_info["gmsh:physical"]):
            tmp_cells[cb.type].append(cb.data)
            tmp_info[cb.type].append(info)

        # Repack the data
        for k, v in tmp_cells.items():
            tmp_cells[k] = np.vstack([part for part in v])
        for k, v in tmp_info.items():
            tmp_info[k] = np.hstack([part for part in v])

        cells = tmp_cells
        cell_info = tmp_info
    else:
        # Meshio v2. May also work for v3?
        # This data format is much closer to what is used in the further processing in
        # porepy. The only thing we need to do is to dump a subdictionary level in
        # cell_info.
        for k, v in cell_info.items():
            cell_info[k] = v["gmsh:physical"]

    return pts, cells, cell_info, phys_names
