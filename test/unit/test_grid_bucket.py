import numpy as np
import scipy.sparse as sps
import unittest
import warnings

from porepy.grids.grid_bucket import GridBucket
from porepy.fracs import meshing

class TestGridBucket(unittest.TestCase):

    def test_cell_global2loc_1_grid(self):
        gb = meshing.cart_grid([], [2,2])
        gb.cell_global2loc()
        R = sps.eye(4)
        for g, d in gb:
            assert np.sum(d['cell_global2loc'] != R) == 0

    def test_cell_global2loc_1_frac(self):
        f = np.array([[0,1],[1,1]])
        gb = meshing.cart_grid([f], [2,2])
        gb.cell_global2loc()
        glob = np.arange(5)
        # test grids
        for g, d in gb:
            if g.dim == 2:
                loc = np.array([0,1,2,3])
            elif g.dim == 1:
                loc = np.array([4])
            else:
                assert False
            R = d['cell_global2loc']
            assert np.all(R*glob==loc)
        # test mortars
        glob = np.array([0, 1])
        for _, d in gb.edges_props():
            loc = np.array([0, 1])
            R = d['cell_global2loc']
            assert np.all(R*glob==loc)

    def test_cell_global2loc_2_fracs(self):
        f1 = np.array([[0,1],[1,1]])
        f2 = np.array([[1,2],[1,1]])        
        f3 = np.array([[1,1],[0,1]])
        f4 = np.array([[1,1],[1,2]])

        gb = meshing.cart_grid([f1, f2, f3, f4], [2,2])
        gb.cell_global2loc()
        glob = np.arange(9)
        # test grids
        for g, d in gb:
            if g.dim == 2:
                loc = np.array([0,1,2,3])
            elif g.dim == 1:
                i = d['node_number']
                loc = np.arange(4 + (i-1), 4 + i)
            else:
                loc = np.array([8])
            R = d['cell_global2loc']
            assert np.all(R*glob==loc)

        # test mortars
        glob = np.arange(12)
        start = 0
        end = 0
        for e, d in gb.edges_props():
            i = d['edge_number']
            end += d['mortar_grid'].num_cells
            loc = np.arange(start, end)
            start = end
            R = d['cell_global2loc']
            assert np.all(R*glob==loc)
