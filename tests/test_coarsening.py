import numpy as np
import scipy as sps
import unittest

from core.grids import structured, simplex
from gridding.coarsening import *

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_coarse_grid_2d( self ):
        g = structured.CartGrid([3, 2])
        g = generate_coarse_grid( g, [5, 2, 2, 5, 2, 2] )

        assert g.num_cells == 2
        assert g.num_faces == 12
        assert g.num_nodes == 11

        pt = np.tile(np.array([2,1,0]), (g.nodes.shape[1],1) ).T
        find = np.isclose( pt, g.nodes ).all( axis = 0 )
        assert find.any() == False

        faces_cell0, _, orient_cell0 = sps.find( g.cell_faces[:,0] )
        assert np.array_equal( faces_cell0, [1, 2, 4, 5, 7, 8, 10, 11] )
        assert np.array_equal( orient_cell0, [-1, 1, -1, 1, -1, -1, 1, 1] )

        faces_cell1, _, orient_cell1 = sps.find( g.cell_faces[:,1] )
        assert np.array_equal( faces_cell1, [0, 1, 3, 4, 6, 9] )
        assert np.array_equal( orient_cell1, [-1, 1, -1, 1, -1, 1] )

        known = np.array( [ [0, 4], [1, 5], [3, 6], [4, 7], [5, 8], [6, 10],
                            [0, 1], [1, 2], [2, 3], [7, 8], [8, 9], [9, 10] ] )

        for f in np.arange( g.num_faces ):
            assert np.array_equal( sps.find( g.face_nodes[:,f] )[0], known[f,:] )

#------------------------------------------------------------------------------#

    def test_coarse_grid_3d( self ):
        g = structured.CartGrid([2, 2, 2])
        g = generate_coarse_grid( g, [0, 0, 0, 0, 1, 1, 2, 2] )

        assert g.num_cells == 3
        assert g.num_faces == 30
        assert g.num_nodes == 27

        faces_cell0, _, orient_cell0 = sps.find( g.cell_faces[:,0] )
        known = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25]
        assert np.array_equal( faces_cell0, known )
        known = [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
        assert np.array_equal( orient_cell0, known )

        faces_cell1, _, orient_cell1 = sps.find( g.cell_faces[:,1] )
        known = [4, 5, 12, 13, 14, 15, 22, 23, 26, 27]
        assert np.array_equal( faces_cell1, known )
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal( orient_cell1, known )

        faces_cell2, _, orient_cell2 = sps.find( g.cell_faces[:,2] )
        known = [6, 7, 14, 15, 16, 17, 24, 25, 28, 29]
        assert np.array_equal( faces_cell2, known )
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal( orient_cell2, known )

        known = np.array( [ [0, 3, 9, 12], [2, 5, 11, 14], [3, 6, 12, 15],
                            [5, 8, 14, 17], [9, 12, 18, 21], [11, 14, 20, 23],
                            [12, 15, 21, 24], [14, 17, 23, 26], [0, 1, 9, 10],
                            [1, 2, 10, 11], [6, 7, 15, 16], [7, 8, 16, 17],
                            [9, 10, 18, 19], [10, 11, 19, 20], [12, 13, 21, 22],
                            [13, 14, 22, 23], [15, 16, 24, 25], [16, 17, 25, 26],
                            [0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7],
                            [4, 5, 7, 8], [9, 10, 12, 13], [10, 11, 13, 14],
                            [12, 13, 15, 16], [13, 14, 16, 17], [18, 19, 21, 22],
                            [19, 20, 22, 23], [21, 22, 24, 25],
                            [22, 23, 25, 26] ] )

        for f in np.arange( g.num_faces ):
            assert np.array_equal( sps.find( g.face_nodes[:,f] )[0], known[f,:] )

