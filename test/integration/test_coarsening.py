import numpy as np
import scipy.sparse as sps
import unittest

from porepy.grids import structured, simplex
from porepy.grids import coarsening as co
from porepy.fracs import meshing

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_coarse_grid_2d( self ):
        g = structured.CartGrid([3, 2])
        g = co.generate_coarse_grid( g, [5, 2, 2, 5, 2, 2] )

        assert g.num_cells == 2
        assert g.num_faces == 12
        assert g.num_nodes == 11

        pt = np.tile(np.array([2,1,0]), (g.nodes.shape[1], 1) ).T
        find = np.isclose( pt, g.nodes ).all( axis = 0 )
        assert find.any() == False

        faces_cell0, _, orient_cell0 = sps.find( g.cell_faces[:, 0] )
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

    def test_coarse_grid_3d(self):
        g = structured.CartGrid([2, 2, 2])
        g = co.generate_coarse_grid( g, [0, 0, 0, 0, 1, 1, 2, 2] )

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

#------------------------------------------------------------------------------#

    def test_coarse_grid_2d_1d(self):
        part = np.array([0, 0, 1, 1, 2, 0, 3, 1])
        f = np.array([[2, 2], [0, 2]])

        gb = meshing.cart_grid([f], [4, 2])
        co.generate_coarse_grid(gb, part)

        # Test
        known = np.array([1, 5, 18, 19])

        for _, d in gb.edges_props():
            faces = sps.find(d['face_cells'])[1]
            assert np.array_equal(faces, known)

#------------------------------------------------------------------------------#

    def test_coarse_grid_2d_1d_cross(self):
        part = np.zeros(36)
        part[[0, 1, 2, 6, 7]] = 1
        part[[8, 14, 13]] = 2
        part[[12, 18, 19]] = 3
        part[[24, 30, 31, 32]] = 4
        part[[21, 22, 23, 27, 28, 29, 33, 34, 35]] = 5
        part[[9]] = 6
        part[[15, 16, 17]] = 7
        part[[9, 10]] = 8
        part[[20, 26, 25]] = 9
        part[[3, 4, 5, 11]] = 10
        f1 = np.array([[3, 3], [1, 5]])
        f2 = np.array([[1, 5], [3, 3]])

        gb = meshing.cart_grid([f1, f2], [6, 6])
        co.generate_coarse_grid(gb, part)

        # Test
        known = np.array([[2, 5], [5, 10, 14, 18, 52, 53, 54, 55],
                          [2, 5], [37, 38, 39, 40, 56, 57, 58, 59]])

        for i, e_d in enumerate(gb.edges_props()):
            faces = sps.find(e_d[1]['face_cells'])[1]
            assert np.array_equal(faces, known[i])

#------------------------------------------------------------------------------#

    def test_coarse_grid_3d_2d(self):
        f = np.array([[2, 2, 2, 2],
                      [0, 2, 2, 0],
                      [0, 0, 2, 2]])
        gb = meshing.cart_grid([f], [4, 2, 2])

        g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
        part = np.zeros(g.num_cells)
        part[g.cell_centers[0, :] < 2] = 1
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([1, 3, 0, 2, 1, 3, 0, 2])
        known = np.array([1, 4, 7, 10, 44, 45, 46, 47])

        for _, d in gb.edges_props():
            indices, faces, _ = sps.find(d['face_cells'])
            assert np.array_equal(indices, known_indices)
            assert np.array_equal(faces, known)

#------------------------------------------------------------------------------#

    def test_coarse_grid_3d_2d_cross(self):
        f1 = np.array([[3, 3, 3, 3],
                       [1, 5, 5, 1],
                       [1, 1, 5, 5]])
        f2 = np.array([[1, 5, 5, 1],
                       [1, 1, 5, 5],
                       [3, 3, 3, 3]])
        gb = meshing.cart_grid([f1, f2], [6, 6, 6])

        g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
        part = np.zeros(g.num_cells)
        p1, p2 = g.cell_centers[0, :] < 3, g.cell_centers[2, :] < 3
        part[np.logical_and(p1, p2)] = 1
        part[np.logical_and(p1, ~p2)] = 2
        part[np.logical_and(~p1, p2)] = 3
        part[np.logical_and(~p1, ~p2)] = 4

        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([[3, 2, 1, 0, 3, 2, 1, 0],
        [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 3, 7, 11, 15, 2,
         6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12],
        [3, 2, 1, 0, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
         6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        known = np.array([[2, 7, 12, 17, 40, 41, 42, 43],
        [22, 25, 28, 31, 40, 43, 46, 49, 58, 61, 64, 67, 76, 79, 82, 85, 288,
         289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
         303],
         [2, 7, 12, 17, 40, 41, 42, 43],
         [223, 224, 225, 226, 229, 230, 231, 232, 235, 236, 237, 238, 241, 242,
          243, 244, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315,
          316, 317, 318, 319]])

        for i, e_d in enumerate(gb.edges_props()):
            indices, faces, _ = sps.find(e_d[1]['face_cells'])
            assert np.array_equal(indices, known_indices[i])
            assert np.array_equal(faces, known[i])

#------------------------------------------------------------------------------#

    def test_create_partition_2d_cart(self):
        g = structured.CartGrid([5, 5])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = np.array([0,0,0,1,1,0,0,2,1,1,3,2,2,2,1,3,3,2,4,4,3,3,4,4,4])
        assert np.array_equal(part, known)

#------------------------------------------------------------------------------#

    def test_create_partition_2d_tri(self):
        g = simplex.StructuredTriangleGrid([3,2])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = np.array([1,1,1,0,0,1,0,2,2,0,2,2])
        known_map = np.array([4,3,7,5,11,8,1,2,10,6,12,9])-1
        assert np.array_equal(part, known[known_map])

#------------------------------------------------------------------------------#

    def test_create_partition_2d_cart_cdepth4(self):
        g = structured.CartGrid([10, 10])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g), cdepth=4)
        known = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                          1,1,2,1,1,1,1,1,1,1,1,2,2,3,1,1,1,1,1,1,1,2,2,3,3,1,1,
                          1,1,1,2,2,2,3,3,3,1,1,1,1,2,2,2,3,3,3,3,1,1,2,2,2,2,3,
                          3,3,3,3,2,2,2,2,2,3,3,3,3,3,2,2,2,2,2])-1
        assert np.array_equal(part, known)

#------------------------------------------------------------------------------#

    def test_create_partition_3d_cart(self):
        g = structured.CartGrid([4,4,4])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = np.array([1,1,1,1,2,4,1,3,2,2,3,3,2,2,3,3,5,4,1,6,4,4,4,3,2,4,7,
                          3,8,8,3,3,5,5,6,6,5,4,7,6,8,7,7,7,8,8,7,9,5,5,6,6,5,5,
                          6,6,8,8,7,9,8,8,9,9])-1
        assert np.array_equal(part, known)

#------------------------------------------------------------------------------#
