"""Helper methods to compare grids.
"""

import numpy as np

import porepy as pp

from . import arrays


def compare_grids(g1: pp.Grid, g2: pp.Grid) -> bool:
    """Compare two grids.

    Two grids are considered equal if the topology and geometry is the same, that is:
        - Same dimension
        - Same number of cells, faces, nodes
        - Same face-node and cell-face relations
        - Same coordinates (nodes, or for 0d grids cell centers).
    Two grids that are identical, but have different ordering of cells, faces, and/or
    nodes are considered different.

    Parameters:
        g1: The first grid to compare.
        g2: The second grid to compare.

    Returns:
        bool: True if the grids are identical, False otherwise.

    """
    if g1.dim != g2.dim:
        return False

    if (g1.num_cells, g1.num_faces, g1.num_nodes) != (
        g2.num_cells,
        g2.num_faces,
        g2.num_nodes,
    ):
        return False

    if not arrays.compare_matrices(g1.face_nodes, g2.face_nodes, 0.1):
        return False

    if not arrays.compare_matrices(g1.cell_faces, g2.cell_faces, 0.1):
        return False

    if g1.dim > 0:
        coord = g1.nodes - g2.nodes
    else:
        coord = g1.cell_centers - g2.cell_centers
    dist = np.sum(coord**2, axis=0)
    if dist.max() > 1e-16:
        return False

    # No need to test other geometric quantities; these are processed from those already
    # checked, thus the grids are identical.
    return True


def compare_mortar_grids(mg1: pp.MortarGrid, mg2: pp.MortarGrid) -> bool:
    """Compare two mortar grids.

    Two mortar grids are considered equal if:
        - They have the same dimension.
        - They have the same number of cells.
        - The grids on each side are identical (as defined by compare_grids()).

    Parameters:
        mg1: The first mortar grid to compare.
        mg2: The second mortar grid to compare.

    Returns:
        bool: True if the grids are identical, False otherwise.

    """
    if mg1.dim != mg2.dim:
        return False

    if mg1.num_cells != mg2.num_cells:
        return False

    for key, g1 in mg1.side_grids.items():
        if key not in mg2.side_grids:
            return False
        g2 = mg2.side_grids[key]
        if not compare_grids(g1, g2):
            return False

    return True


def compare_md_grids(mdg1: pp.MixedDimensionalGrid, mdg2: pp.MixedDimensionalGrid):
    """Compare two mixed-dimensional grids.

    Two mixed-dimensional grids are considered equal if they have the same subdomains
    and interfaces (as defined by compare_grids() and compare_mortar_grids(),
    respectively).

    Parameters:
        mdg1: The first mixed-dimensional grid to compare.
        mdg2: The second mixed-dimensional grid to compare.

    Returns:
        bool: True if the grids are identical, False otherwise.

    """
    for dim in range(4):
        subdomains_1 = mdg1.subdomains(dim=dim)
        subdomains_2 = mdg2.subdomains(dim=dim)
        # Two mdgs are considered equal only if the grids are returned in the same
        # order.
        if len(subdomains_1) != len(subdomains_2):
            return False
        for sd1, sd2 in zip(subdomains_1, subdomains_2):
            if not compare_grids(sd1, sd2):
                return False
        interfaces_1 = mdg1.interfaces(dim=dim)
        interfaces_2 = mdg2.interfaces(dim=dim)
        if len(interfaces_1) != len(interfaces_2):
            return False
        for if1, if2 in zip(interfaces_1, interfaces_2):
            if not compare_mortar_grids(if1, if2):
                return False
    return True
