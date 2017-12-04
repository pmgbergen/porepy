import numpy as np
import unittest

from porepy.numerics.fv import mpsa
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import bc
from porepy.grids.grid import FaceTag


class BasicsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        f = np.array([[1, 1, 4], [3, 4, 1], [2, 2, 4]])
        box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
               'xmax': 5, 'ymax':  5, 'zmax': 5}

        self.gb3d = meshing.simplex_grid([f], box, h_min= 5, h_ideal=5)
        unittest.TestCase.__init__(self, *args, **kwargs)

    #------------------------------------------------------------------------------#

    def test_zero_force(self):
        """
        test that nothing moves if nothing is touched
        """
        g = self.gb3d.grids_of_dimension(3)[0]

        data = {'param': Parameters(g)}
        bound = bc.BoundaryCondition(g, g.get_boundary_faces(), 'dir') 

        data['param'].set_bc('mechanics', bound)

        solver = mpsa.FracturedMpsa()
        
        A, b = solver.matrix_rhs(g, data)
        u = np.linalg.solve(A.A, b)
        assert np.all(np.abs(u) < 1e-10)

    def test_unit_slip(self):
        """
        test if fractures are glued together it behaves as no fracture
        """
        frac = np.array([[1,1,1], [1,2,1], [2,2,1], [2,1,1]]).T
        physdims = np.array([3,3,2])

        g = meshing.cart_grid([frac], [3,3,2], physdims=physdims).grids_of_dimension(3)[0]

        data = {'param': Parameters(g)}
        bound = bc.BoundaryCondition(g, g.get_boundary_faces(), 'dir') 

        bc_val = np.zeros((g.dim, g.num_faces))
        frac_bnd = g.has_face_tag(FaceTag.FRACTURE)
        bc_val[:, frac_bnd] = np.ones((g.dim, np.sum(frac_bnd)))

        data['param'].set_bc('mechanics', bound)
        data['param'].set_bc_val('mechanics', bc_val.ravel('F'))

        solver = mpsa.FracturedMpsa()        

        A, b = solver.matrix_rhs(g, data)
        u = np.linalg.solve(A.A, b)
        
        u_f = solver.extract_frac_u(g, u)
        u_c = solver.extract_u(g, u)
        u_c = u_c.reshape((3, -1), order='F')

        frac_faces = g.frac_pairs
        frac_left = frac_faces[0]
        frac_right = frac_faces[1]

        cell_left = np.ravel(np.argwhere(g.cell_faces[frac_left, : ])[:, 1])
        cell_right = np.ravel(np.argwhere(g.cell_faces[frac_right, : ])[:, 1])

        # we have u_lhs - u_rhs = 1 so u_lhs should be positive
        assert np.all(u_c[:, cell_left] > 0)
        assert np.all(u_c[:, cell_right] < 0)
        
        u_left = u_f[:round(u_f.size/2)]
        u_right = u_f[round(u_f.size/2):]
        assert np.all(np.abs(u_left - u_right - 1) < 1e-10)

        # fracture displacement should be symetric since everything else is symetric
        assert np.allclose(u_left, 0.5)
        assert np.allclose(u_right, -0.5)


    def test_non_zero_bc(self):
        """
        test domain cut in two. We place 1 dirichlet on top. zero dirichlet on
        bottom and 0 neumann on sides. Further we place 1 displacement on
        fracture. this should give us displacement 1 on top cells and 0 on 
        bottom cells and zero traction on all faces
        """
        frac = np.array([[1,1,1], [1,2,1], [2,2,1], [2,1,1]]).T
        physdims = np.array([3,3,2])

        g = meshing.cart_grid([frac], [3,3,2], physdims=physdims).grids_of_dimension(3)[0]
        data = {'param': Parameters(g)}


        # Define boundary conditions 
        bc_val = np.zeros((g.dim, g.num_faces))
        frac_bnd = g.has_face_tag(FaceTag.FRACTURE)
        dom_bnd = g.has_face_tag(FaceTag.DOMAIN_BOUNDARY)        
        bc_val[0, frac_bnd] = np.ones(np.sum(frac_bnd))
        bc_val[:, dom_bnd] = g.face_centers[:, dom_bnd]

        bound = bc.BoundaryCondition(g, g.get_boundary_faces(), 'dir')

        data['param'].set_bc('mechanics', bound)
        data['param'].set_bc_val('mechanics', bc_val.ravel('F'))

        solver = mpsa.FracturedMpsa()        

        A, b = solver.matrix_rhs(g, data)
        u = np.linalg.solve(A.A, b)
        
        u_f = solver.extract_frac_u(g, u)
        u_c = solver.extract_u(g, u)
        u_c = u_c.reshape((3, -1), order='F')

        # we have u_lhs - u_rhs = 1 so u_lhs should be positive
        u_left = u_f[:round(u_f.size/2)]
        u_right = u_f[round(u_f.size/2):]

        true_diff = np.atleast_2d(np.array([1,0,0])).T
        u_left = u_left.reshape((3, -1), order='F')
        u_right = u_right.reshape((3, -1), order='F')
        assert np.all(np.abs(u_left - u_right - true_diff) < 1e-10)
     
        # should have a positive displacement for all cells
        assert np.all(u_c > 0)

    def test_domain_cut_in_two(self):
        """
        test if fractures are glued together it behaves as no fracture
        """
        frac = np.array([[0,0,1], [0,3,1], [3,3,1], [3,0,1]]).T
        g = meshing.cart_grid([frac], [3,3,2]).grids_of_dimension(3)[0]
        data = {'param': Parameters(g)}

        tol = 1e-6
        bc_val = np.zeros((g.dim, g.num_faces))
        frac_bnd = g.has_face_tag(FaceTag.FRACTURE)
        top = g.face_centers[2] > 2 - tol
        bot = g.face_centers[2] < tol

        dir_bound = top | bot | frac_bnd
        
        bound = bc.BoundaryCondition(g, dir_bound, 'dir') 

        bc_val[:, frac_bnd] = np.ones(np.sum(frac_bnd))
        bc_val[:, top] = np.ones((g.dim, np.sum(top)))



        data['param'].set_bc('mechanics', bound)
        data['param'].set_bc_val('mechanics', bc_val.ravel('F'))

        solver = mpsa.FracturedMpsa()        

        A, b = solver.matrix_rhs(g, data)
        u = np.linalg.solve(A.A, b)
        
        u_f = solver.extract_frac_u(g, u)
        u_c = solver.extract_u(g, u)
        u_c = u_c.reshape((3, -1), order='F')
        T = solver.traction(g, data, u)

        top_cells = g.cell_centers[2] > 1

        u_left = u_f[:round(u_f.size/2)]
        u_right = u_f[round(u_f.size/2):]



        assert np.allclose(u_left, 1)
        assert np.allclose(u_right, 0)        
        assert np.allclose(u_c[:, top_cells], 1)
        assert np.allclose(u_c[:, ~top_cells], 0)        
        assert np.allclose(T, 0)
