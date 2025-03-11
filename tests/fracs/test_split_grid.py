"""Tests for the module split_grid in the fracs package."""

import numpy as np
import scipy.sparse as sps

import porepy as pp


def test_duplicate_nodes():
    # Test of the method to split nodes. The function is useful for debugging, but also
    # does some testing of known properties of the split nodes.
    nn = 23
    nf = 59
    nc = 29

    # Hard code grid topology
    fn_0_9 = np.array([1, 21, 0, 1, 1, 2, 2, 4, 1, 4, 2, 5, 4, 5, 6, 4, 4, 7, 6, 7])
    fn_10_19 = np.array(
        [7, 8, 7, 9, 9, 8, 9, 10, 8, 10, 6, 10, 10, 11, 6, 11, 5, 11, 11, 12]
    )
    fn_20_29 = np.array(
        [5, 12, 3, 12, 12, 13, 3, 13, 13, 14, 13, 16, 13, 15, 15, 16, 15, 22, 17, 22]
    )
    fn_30_39 = np.array(
        [16, 22, 16, 17, 18, 17, 16, 18, 18, 14, 14, 2, 2, 18, 18, 19, 2, 19, 0, 19]
    )
    fn_40_49 = np.array(
        [19, 20, 20, 0, 20, 21, 0, 21, 0, 2, 0, 2, 2, 3, 2, 3, 1, 2, 14, 16]
    )
    fn_50_58 = np.array([14, 16, 14, 3, 14, 3, 3, 5, 3, 5, 5, 6, 6, 5, 6, 8, 6, 8])

    fn_rows = np.hstack((fn_0_9, fn_10_19, fn_20_29, fn_30_39, fn_40_49, fn_50_58))

    fn_cols = np.tile(np.arange(nf), (2, 1)).ravel("f")
    fn_data = np.ones(nf * 2, dtype=bool)
    face_nodes = sps.coo_matrix((fn_data, (fn_rows, fn_cols)), shape=(nn, nf)).tocsc()

    cf_0_6 = np.array(
        [0, 1, 43, 1, 2, 44, 3, 4, 48, 3, 6, 5, 46, 5, 53, 6, 7, 55, 7, 8, 9]
    )
    cf_7_12 = np.array(
        [57, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 58, 15, 16, 17, 17, 56, 18]
    )
    cf_13_16 = np.array([18, 19, 20, 20, 21, 54, 22, 23, 21, 23, 24, 52])
    cf_17_19 = np.array([24, 25, 50, 25, 26, 27, 27, 28, 30])
    cf_20_25 = np.array(
        [29, 30, 31, 31, 32, 33, 33, 49, 34, 34, 35, 36, 47, 51, 35, 36, 37, 38]
    )
    cf_26_28 = np.array([38, 39, 45, 39, 40, 41, 41, 42, 43])

    cf_rows = np.hstack((cf_0_6, cf_7_12, cf_13_16, cf_17_19, cf_20_25, cf_26_28))
    cf_cols = np.tile(np.arange(nc), (3, 1)).ravel("f")
    # Will not need +- sign on cell_faces for this test
    cf_data = np.ones(3 * nc)
    cell_faces = sps.coo_matrix((cf_data, (cf_rows, cf_cols)), shape=(nf, nc)).tocsc()

    # Random node coordinates, these are not important for the test
    nodes = np.random.rand(3, nn)

    g = pp.Grid(2, nodes, face_nodes, cell_faces, "mock_grid")
    split_nodes = np.array([2, 3, 5, 6, 14])

    _ = pp.fracs.split_grid.duplicate_nodes(g, split_nodes, offset=0)

    cn = g.cell_nodes().tocsc()

    # Test: What used to be node 2 should be split into three nodes (2, 3, and 4).
    # Each of these should be shared by all cells in the clusters.
    found = {2: False, 3: False, 4: False}

    # Known clusters
    clusters = [[1], [2, 3, 4], [23, 24, 25, 26]]

    for cells in clusters:
        node_set = cn[:, cells[0]].indices
        for c in cells[1:]:
            node_set = np.intersect1d(node_set, cn[:, c].indices)

        if len(cells) == 1:
            ind = np.argwhere([np.isin(n, list(found.keys())) for n in node_set])[0]
            node_set = node_set[ind]

        assert node_set.size == 1
        assert node_set[0] in found
        assert not found[node_set[0]]

        found[node_set[0]] = True
