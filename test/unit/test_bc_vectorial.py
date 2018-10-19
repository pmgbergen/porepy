import unittest
import numpy as np

import math

import porepy as pp

"""
Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs_nd
for handling boundary conditions expressed in a vectorial form
"""


class testBoundaryConditionsVectorial(unittest.TestCase):
    def test_default_basis_2d(self):
        g = pp.StructuredTriangleGrid([1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        basis_known = np.array(
            [
                [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            ]
        )
        self.assertTrue(np.allclose(bc.basis, basis_known))

    def test_default_basis_3d(self):
        g = pp.StructuredTetrahedralGrid([1, 1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        basis_known = np.squeeze(
            np.array([[[[1, 0, 0]] * 18], [[[0, 1, 0]] * 18], [[[0, 0, 1]] * 18]])
        )

        self.assertTrue(np.allclose(bc.basis, basis_known))

    def test_2d(self):

        """
        The domain consists of a 3x3 mesh.
        Bottom faces are dirichlet
        Left and right faces are rolling along y (dir_x and neu_y)
        Top mid face is rolling along x (neu_x and dir_y)
        Top left and right faces are neumann
        """

        g = pp.CartGrid([3, 3])
        g.compute_geometry()
        nd = g.dim

        boundary_faces = np.array([0, 3, 4, 7, 8, 11, 12, 13, 14, 22])
        boundary_faces_type = ["dir_x"] * 6 + ["dir"] * 3 + ["dir_y"] * 1

        bound = pp.BoundaryConditionVectorial(g, boundary_faces, boundary_faces_type)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        # Obtain the face number for each coordinate
        fno = subcell_topology.fno_unique
        subfno = subcell_topology.subfno_unique
        subfno_nd = np.tile(subfno, (nd, 1)) * nd + np.atleast_2d(np.arange(0, nd)).T

        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

        # expand the indices
        # Define right hand side for Neumann boundary conditions
        # First row indices in rhs matrix
        # Pick out the subface indices
        subfno_neu = bound_exclusion.exclude_robin_dirichlet_nd(
            subfno_nd.ravel("C")
        ).ravel("F")
        # Pick out the Neumann boundary

        is_neu_nd = (
            bound_exclusion.exclude_robin_dirichlet_nd(bound.is_neu[:, fno].ravel("C"))
            .ravel("F")
            .astype(np.bool)
        )

        neu_ind = np.argsort(subfno_neu)
        neu_ind = neu_ind[is_neu_nd[neu_ind]]

        self.assertTrue(
            np.alltrue(
                neu_ind
                == [
                    30,
                    31,
                    36,
                    37,
                    38,
                    39,
                    44,
                    45,
                    46,
                    47,
                    52,
                    53,
                    24,
                    66,
                    25,
                    67,
                    26,
                    27,
                    28,
                    68,
                    29,
                    69,
                ]
            )
        )

        subfno_dir = bound_exclusion.exclude_neumann_robin_nd(
            subfno_nd.ravel("C")
        ).ravel("F")
        is_dir_nd = (
            bound_exclusion.exclude_neumann_robin_nd(bound.is_dir[:, fno].ravel("C"))
            .ravel("F")
            .astype(np.bool)
        )

        dir_ind = np.argsort(subfno_dir)
        dir_ind = dir_ind[is_dir_nd[dir_ind]]

        self.assertTrue(
            np.alltrue(
                dir_ind
                == [
                    0,
                    1,
                    6,
                    7,
                    8,
                    9,
                    14,
                    15,
                    16,
                    17,
                    22,
                    23,
                    24,
                    54,
                    25,
                    55,
                    26,
                    56,
                    27,
                    57,
                    28,
                    58,
                    29,
                    59,
                    72,
                    73,
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
