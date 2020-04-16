"""
Module for creating simplex grids with fractures.
"""
import time
import numpy as np
import meshio
import logging


from porepy.grids import constants
from porepy.grids.gmsh import mesh_2_grid


logger = logging.getLogger(__name__)


def triangle_grid_embedded(network, file_name, **kwargs):
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
        file_name (str, optional): Filename for communication with gmsh.
            The config file for gmsh will be f_name.geo, with the grid output
            to f_name.msh. Defaults to dfn_network.
        **kwargs: Arguments sent to gmsh etc.

    Returns:
        list (length 3): For each dimension (2 -> 0), a list of all grids in
            that dimension.

    """

    if file_name[-4:] == ".geo" or file_name[-4:] == ".msh":
        file_name = file_name[:-4]

    out_file = file_name + ".msh"

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
    g_0d = mesh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)

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
    g_0d = mesh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)
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


def tetrahedral_grid_from_gmsh(network, file_name, constraints=None, **kwargs):
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
        constraints=constraints,
    )

    g_1d, _ = mesh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = mesh_2_grid.create_0d_grids(pts, cells, phys_names, cell_info)

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
