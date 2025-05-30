"""Helper methods to compare grids.

Content:
    - compare_grids: Compare two grids for equality of topology and geometry.
    - compare_mortar_grids: Compare two mortar grids for equality by comparing their
        cell numbers, as well as the geometry and topology of the side grids.
    - compare_md_grids: Compare two mixed-dimensional grids for equality by comparing
        their subdomains and interfaces (grids and mortar grids).
    - polytop_grid_2d: Create a hard-coded 2D polytopal grid with 4 cells on the unit
        square.
    - polytop_grid_3d: Create a hard-coded 3D polytopal grid with 5 cells on the unit
        cube.
"""

import numpy as np
import scipy.sparse as sps

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


def polytop_grid_2d() -> pp.Grid:
    """Hard coded specification of a 2d polytopal grid with 4 cells on the unit square.

    This grid was originally created by coarsening a 2d triangular grid, using the now
    deleted coarsening module. This function is used as a substitute, primarily to
    enable testing on 2d polytop grids.

    """
    nodes = np.array(
        [
            [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    cf_indptr = np.array([0, 3, 8, 12, 16], dtype=np.int32)
    cf_indices = np.array(
        [0, 2, 4, 1, 2, 6, 8, 10, 7, 8, 9, 11, 3, 4, 5, 7], dtype=np.int32
    )
    cf_data = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])

    cf = sps.csc_matrix((cf_data, cf_indices, cf_indptr), shape=(12, 4))

    fn_indptr = np.array(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], dtype=np.int32
    )
    fn_indices = np.array(
        [0, 1, 0, 3, 0, 4, 1, 2, 1, 4, 2, 5, 3, 6, 4, 5, 4, 7, 5, 8, 6, 7, 7, 8],
        dtype=np.int32,
    )
    fn_data = np.ones(fn_indices.size, dtype=bool)
    fn = sps.csc_matrix((fn_data, fn_indices, fn_indptr), shape=(9, 12))
    # Create the grid
    g = pp.Grid(2, nodes, fn, cf, "polytopal grid 2d")
    g.compute_geometry()
    return g


def polytop_grid_3d() -> pp.Grid:
    """Hard coded specification of a 3d polytopal grid with 5 cells on the unit cube.

    This grid was originally created by coarsening a 3d tetrahedral grid, using the now
    deleted coarsening module. This function is used as a substitute, primarily to
    enable testing on 3d polytop grids.

    """
    nodes = np.array(
        [
            [
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
                0.0,
                0.33333333,
                0.66666667,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                0.66666667,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ]
    )
    cf_indptr = np.array([0, 18, 32, 48, 54, 76], dtype=np.int32)
    cf_indices = np.array(
        [
            0,
            1,
            3,
            4,
            6,
            7,
            16,
            17,
            19,
            20,
            23,
            26,
            36,
            37,
            39,
            42,
            44,
            47,
            1,
            2,
            4,
            5,
            18,
            19,
            21,
            22,
            38,
            40,
            41,
            43,
            45,
            46,
            7,
            8,
            10,
            11,
            24,
            25,
            28,
            29,
            42,
            43,
            45,
            46,
            48,
            49,
            51,
            52,
            9,
            10,
            26,
            27,
            44,
            50,
            12,
            13,
            14,
            15,
            30,
            31,
            32,
            33,
            34,
            35,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
        ],
        dtype=np.int32,
    )
    cf_data = np.array(
        [
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    cf = sps.csc_matrix((cf_data, cf_indices, cf_indptr), shape=(59, 5))

    fn_indptr = np.array(
        [
            0,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            100,
            104,
            108,
            112,
            116,
            120,
            124,
            128,
            132,
            136,
            140,
            144,
            148,
            152,
            156,
            160,
            164,
            168,
            172,
            176,
            180,
            184,
            188,
            192,
            196,
            200,
            204,
            208,
            212,
            216,
            220,
            224,
            228,
            232,
            236,
        ],
        dtype=np.int32,
    )
    fn_indices = np.array(
        [
            0,
            4,
            16,
            12,
            2,
            6,
            18,
            14,
            3,
            7,
            19,
            15,
            4,
            8,
            20,
            16,
            5,
            9,
            21,
            17,
            7,
            11,
            23,
            19,
            12,
            16,
            28,
            24,
            13,
            17,
            29,
            25,
            15,
            19,
            31,
            27,
            16,
            20,
            32,
            28,
            17,
            21,
            33,
            29,
            19,
            23,
            35,
            31,
            24,
            28,
            40,
            36,
            27,
            31,
            43,
            39,
            28,
            32,
            44,
            40,
            31,
            35,
            47,
            43,
            0,
            12,
            13,
            1,
            1,
            13,
            14,
            2,
            2,
            14,
            15,
            3,
            5,
            17,
            18,
            6,
            8,
            20,
            21,
            9,
            9,
            21,
            22,
            10,
            10,
            22,
            23,
            11,
            12,
            24,
            25,
            13,
            13,
            25,
            26,
            14,
            14,
            26,
            27,
            15,
            16,
            28,
            29,
            17,
            20,
            32,
            33,
            21,
            21,
            33,
            34,
            22,
            22,
            34,
            35,
            23,
            24,
            36,
            37,
            25,
            25,
            37,
            38,
            26,
            26,
            38,
            39,
            27,
            32,
            44,
            45,
            33,
            33,
            45,
            46,
            34,
            34,
            46,
            47,
            35,
            0,
            1,
            5,
            4,
            1,
            2,
            6,
            5,
            2,
            3,
            7,
            6,
            4,
            5,
            9,
            8,
            5,
            6,
            10,
            9,
            6,
            7,
            11,
            10,
            13,
            14,
            18,
            17,
            14,
            15,
            19,
            18,
            16,
            17,
            21,
            20,
            17,
            18,
            22,
            21,
            18,
            19,
            23,
            22,
            24,
            25,
            29,
            28,
            25,
            26,
            30,
            29,
            26,
            27,
            31,
            30,
            28,
            29,
            33,
            32,
            29,
            30,
            34,
            33,
            30,
            31,
            35,
            34,
            36,
            37,
            41,
            40,
            37,
            38,
            42,
            41,
            38,
            39,
            43,
            42,
            40,
            41,
            45,
            44,
            41,
            42,
            46,
            45,
            42,
            43,
            47,
            46,
        ],
        dtype=np.int32,
    )
    fn_data = np.ones(fn_indices.size, dtype=bool)
    fn = sps.csc_matrix((fn_data, fn_indices, fn_indptr), shape=(48, 59))
    # Create the grid
    g = pp.Grid(3, nodes, fn, cf, "polytopal grid 3d")
    g.compute_geometry()
    return g
