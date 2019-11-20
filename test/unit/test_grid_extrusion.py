"""
Unit tests for grid extrusion functionality
"""

import numpy as np
import unittest
import porepy as pp
import scipy.sparse as sps


class TestGridExtrusion0d(unittest.TestCase):
    
    def setUp(self, x=0, y=0):
        c = np.array([x, y, 0]).reshape((-1, 1))
        self.g = pp.PointGrid(c)
    
    def known_array(self, z, x=0, y=0):
        num_p = z.size
        return np.vstack((x * np.ones(num_p),
                          y * np.ones(num_p),
                          z))
        
    def make_grid_check(self, z, x=0, y=0):
        gn = pp.grid_extrusion._extrude_0d(self.g, z)
        coord_known = self.known_array(z, x, y)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
    
    def test_one_layer(self):
        z = np.array([0, 1])
        self.make_grid_check(z)

    def test_not_origin(self):
        # Point grid is away from origin
        self.setUp(x=1, y=2)
        z = np.array([0, 1])
        self.make_grid_check(z, x=1, y=2)
        
    def test_two_layers(self):
        z = np.array([0, 1, 3])
        self.make_grid_check(z)        
        
    def test_ignore_original_z(self):
        # Assign non-zero z-coordinate of the point grid. This should be ignored
        self.g.nodes[2, 0] = 3
        z = np.array([0, 1])
        gn = pp.grid_extrusion._extrude_0d(self.g, z)
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        
    def test_negative_z(self):
        z = np.array([0, -1])
        self.make_grid_check(z)
        
class TestGridExtrusion1d(unittest.TestCase):
    
    def setUp(self):
        self.x = np.array([0, 1, 3])
        self.y = np.array([1, 3, 7])
        self.g = pp.TensorGrid(self.x)
        self.g.nodes = np.vstack((self.x, self.y, np.zeros(3)))
        
    def known_array(self, z_coord):
        x_tmp = np.array([])
        y_tmp = np.empty_like(x_tmp)
        z_tmp = np.empty_like(x_tmp)
        for z in z_coord:
            x_tmp = np.hstack((x_tmp, self.x))
            y_tmp = np.hstack((y_tmp, self.y))
            z_tmp = np.hstack((z_tmp, z * np.ones(self.x.size)))
        return np.vstack((x_tmp, y_tmp, z_tmp))
        
    def test_one_layer(self):
        z = np.array([0, 1])
        gn = pp.grid_extrusion._extrude_1d(self.g, z)
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        
    def test_two_layers(self):
        z = np.array([0, 1, 3])
        gn = pp.grid_extrusion._extrude_1d(self.g, z)
        
        self.assertTrue(gn.num_nodes == 9)
        
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))

    def test_negative_z(self):
        z = np.array([0, -1, -3])
        gn = pp.grid_extrusion._extrude_1d(self.g, z)
        
        self.assertTrue(gn.num_nodes == 9)
        
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        
class TestGridExtrusion2d(unittest.TestCase):

    def test_single_cell(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction
        
        fn_2d = sps.csc_matrix(np.array([[1, 0, 1],
                                          [1, 1, 0],
                                          [0, 1, 1]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))
        
        nodes_2d = np.array([[0, 2, 1],
                             [0, 0, 1],
                             [0, 0, 0]])
        
        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, 'tmp')
        
        z = np.array([0, 1, 3])
        nz = z.size
        
        g_3d = pp.grid_extrusion._extrude_2d(g_2d, z)
    
        self.assertTrue(g_3d.num_nodes == 9)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)
        
        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))
        
        fn_3d = np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0],
                          [0, 1, 1, 0, 1, 1, 0, 0, 0],
                          [1, 0, 1, 1, 0, 1, 0, 0, 0],  # Done with first vertical layer
                          [0, 0, 0, 1, 1, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 1, 0, 1, 1],  
                          [0, 0, 0, 1, 0, 1, 1, 0, 1],  # Second vertical layer
                          [1, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1]
                          ]).T
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))
        
        cf_3d = np.array([[-1, 0],
                          [1, 0],
                          [-1, 0],
                          [0, -1],
                          [0, 1],
                          [0, -1],
                          [-1, 0],
                          [1, -1],
                          [0, 1]
                          ])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))
        
        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1/3, 1/3], [0.5, 2]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))
        
        # Face centers
        fc = np.array([[1, 0, 0.5], 
                       [1.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [1, 0, 2], 
                       [1.5, 0.5, 2],
                       [0.5, 0.5, 2],
                       [1, 1/3, 0],
                       [1, 1/3, 1],
                       [1, 1/3, 3]
                       ]).T
        self.assertTrue(np.allclose(g_3d.face_centers, fc))
        
        # Face areas
        fa = np.array([2, np.sqrt(2), np.sqrt(2), 4, 2* np.sqrt(2), 2* np.sqrt(2), 1, 1, 1])
        self.assertTrue(np.allclose(g_3d.face_areas, fa))
        
        # Normal vectors
        normals = np.array([[0, 2, 0],
                            [1, 1, 0],
                            [1, -1, 0],
                            [0, 4, 0],
                            [2, 2, 0],
                            [2, -2, 0],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            ]).T
        self.assertTrue(np.allclose(g_3d.face_normals, normals))

    def test_single_cell_reverted_faces(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction
        
        fn_2d = sps.csc_matrix(np.array([[1, 0, 1],
                                          [0, 1, 1],
                                          [1, 1, 0]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))
        
        nodes_2d = np.array([[0, 2, 1],
                             [0, 0, 1],
                             [0, 0, 0]])
        
        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, 'tmp')
        
        z = np.array([0, 1])
        nz = z.size
        
        g_3d = pp.grid_extrusion._extrude_2d(g_2d, z)
    
        self.assertTrue(g_3d.num_nodes == 6)
        self.assertTrue(g_3d.num_cells == 1)
        self.assertTrue(g_3d.num_faces == 5)
        
        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))
        
        fn_3d = np.array([[1, 0, 1, 1, 0, 1 ],
                          [0, 1, 1, 0, 1, 1],
                          [1, 1, 0, 1, 1, 0],
                          [1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1],
                          ]).T
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))
        
        cf_3d = np.array([[-1],
                          [1],
                          [-1],
                          [-1],
                          [1],
                          ])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))
        
        # Geometric quantities
        # Cell center
        cc = np.array([[1], [1/3], [0.5]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))
        
        # Face centers
        fc = np.array([[0.5, 0.5, 0.5],
                       [1.5, 0.5, 0.5],
                       [1, 0, 0.5], 
                       [1, 1/3, 0],
                       [1, 1/3, 1],
                       ]).T
        self.assertTrue(np.allclose(g_3d.face_centers, fc))
        
        # Face areas
        fa = np.array([np.sqrt(2), np.sqrt(2), 2, 1, 1])
        self.assertTrue(np.allclose(g_3d.face_areas, fa))
        
        # Normal vectors
        normals = np.array([[1, -1, 0],
                            [1, 1, 0],
                            [0, 2, 0],
                            [0, 0, 1],
                            [0, 0, 1],
                            ]).T
        self.assertTrue(np.allclose(g_3d.face_normals, normals))
        
    def test_single_cell_negative_z(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction
        
        fn_2d = sps.csc_matrix(np.array([[1, 0, 1],
                                          [1, 1, 0],
                                          [0, 1, 1]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))
        
        nodes_2d = np.array([[0, 2, 1],
                             [0, 0, 1],
                             [0, 0, 0]])
        
        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, 'tmp')
        
        z = np.array([0, -1, -3])
        nz = z.size
        
        g_3d = pp.grid_extrusion._extrude_2d(g_2d, z)
    
        self.assertTrue(g_3d.num_nodes == 9)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)
        
        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))
        
        fn_3d = np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0],
                          [0, 1, 1, 0, 1, 1, 0, 0, 0],
                          [1, 0, 1, 1, 0, 1, 0, 0, 0],  # Done with first vertical layer
                          [0, 0, 0, 1, 1, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 1, 0, 1, 1],  
                          [0, 0, 0, 1, 0, 1, 1, 0, 1],  # Second vertical layer
                          [1, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1]
                          ]).T
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))
        
        cf_3d = np.array([[-1, 0],
                          [1, 0],
                          [-1, 0],
                          [0, -1],
                          [0, 1],
                          [0, -1],
                          [-1, 0],
                          [1, -1],
                          [0, 1]
                          ])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))
        
        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1/3, 1/3], [-0.5, -2]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))
        
        # Face centers
        fc = np.array([[1, 0, -0.5], 
                       [1.5, 0.5, -0.5],
                       [0.5, 0.5, -0.5],
                       [1, 0, -2], 
                       [1.5, 0.5, -2],
                       [0.5, 0.5, -2],
                       [1, 1/3, 0],
                       [1, 1/3, -1],
                       [1, 1/3, -3]
                       ]).T
        self.assertTrue(np.allclose(g_3d.face_centers, fc))
        
        # Face areas
        fa = np.array([2, np.sqrt(2), np.sqrt(2), 4, 2* np.sqrt(2), 2* np.sqrt(2), 1, 1, 1])
        self.assertTrue(np.allclose(g_3d.face_areas, fa))
        
        # Normal vectors
        normals = np.array([[0, 2, 0],
                            [1, 1, 0],
                            [1, -1, 0],
                            [0, 4, 0],
                            [2, 2, 0],
                            [2, -2, 0],
                            [0, 0, -1],
                            [0, 0, -1],
                            [0, 0, -1],
                            ]).T
        self.assertTrue(np.allclose(g_3d.face_normals, normals))        
        
    def test_two_cells(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction
        
        fn_2d = sps.csc_matrix(np.array([[1, 0, 0, 1, 0],
                                         [1, 1, 0, 0, 1],
                                         [0, 1, 1, 0, 0],
                                         [0, 0, 1, 1, 1]]))

        cf_2d = sps.csc_matrix(np.array([[1, 0],
                                         [0, 1],
                                         [0, 1],
                                         [1, 0],
                                         [-1, 1]]))
        
        nodes_2d = np.array([[0, 1, 1, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 0]])
        
        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, 'tmp')
        
        z = np.array([0, 1])
        nz = z.size
        
        g_3d = pp.grid_extrusion._extrude_2d(g_2d, z)
    
        self.assertTrue(g_3d.num_nodes == 8)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)
        
        nodes = np.tile(nodes_2d, nz)
        nodes[2, :4] = z[0]
        nodes[2, 4:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))
        
        fn_3d = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                          [0, 1, 1, 0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0, 0, 1, 1],  # Done with first vertical layer
                          [1, 0, 0, 1, 1, 0, 0, 1],
                          [0, 1, 0, 1, 0, 1, 0, 1],  
                          [1, 1, 0, 1, 0, 0, 0, 0],  # Second vertical layer
                          [0, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 1, 1]
                          ]).T
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))
        
        cf_3d = np.array([[1, 0],
                          [0, 1],
                          [0, 1],
                          [1, 0],
                          [-1, 1],
                          [-1, 0],
                          [0, -1],
                          [1, 0],
                          [0, 1]
                          ])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))
        
    def test_two_cells_one_quad(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction
        
        fn_2d = sps.csc_matrix(np.array([[1, 0, 0, 0, 1, 0],
                                         [1, 1, 0, 0, 0, 1],
                                         [0, 1, 1, 0, 0, 0],
                                         [0, 0, 1, 1, 0, 0],
                                         [0, 0, 0, 1, 1, 1],
                                         ]))

        cf_2d = sps.csc_matrix(np.array([[1, 0],
                                         [0, -1],
                                         [0, 1],
                                         [0, 1],
                                         [-1, 0],
                                         [1, -1]]))
        
        nodes_2d = np.array([[0, 1, 2, 1, 0],
                             [0, 0, 1, 2, 1],
                             [0, 0, 0, 0, 0]])
        
        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, 'tmp')
        
        z = np.array([0, 1])
        nz = z.size
        
        g_3d = pp.grid_extrusion._extrude_2d(g_2d, z)
    
        self.assertTrue(g_3d.num_nodes == 10)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 10)
        
        nodes = np.tile(nodes_2d, nz)
        nodes[2, :5] = z[0]
        nodes[2, 5:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))
        
        fn_ind = np.array([0, 1, 6, 5,
                            1, 2, 7, 6,
                            2, 3, 8, 7,
                            3, 4, 9, 8,
                            4, 0, 5, 9,
                            1, 4, 9, 6,
                            0, 1, 4,
                            1, 2, 3, 4,
                            5, 6, 9,
                            6, 7, 8, 9
                                ])
        fn_ind_ptr = np.array([0, 4, 8, 12, 16, 20, 24, 27, 31, 34, fn_ind.size])
    
        fn_3d = sps.csc_matrix((np.ones(fn_ind.size), fn_ind, fn_ind_ptr), shape=(10, 10))
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d.toarray()))
        
        cf_3d = np.array([[1, 0],
                         [0, -1],
                         [0, 1],
                         [0, 1],
                         [-1, 0],
                         [1, -1],
                          [-1, 0],
                          [0, -1],
                          [1, 0],
                          [0, 1]
                          ])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))        
        
        
if __name__ == '__main__':
    TestGridExtrusion2d().test_single_cell_negative_z()
    unittest.main()