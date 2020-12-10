import math
import unittest

import numpy as np

import porepy as pp

"""
Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs
for handling boundary conditions expressed in a vectorial form
"""


class testBoundaryConditionsVectorial(unittest.TestCase):
    def test_default_basis_2d(self):
        g = pp.StructuredTriangleGrid([1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        basis_known = np.array(
            [
                [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            ]
        )

        self.assertTrue(np.allclose(bc.basis, basis_known))

    def test_default_basis_3d(self):
        g = pp.StructuredTetrahedralGrid([1, 1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        basis_known = np.array(
            [
                [[1] * 18, [0] * 18, [0] * 18],
                [[0] * 18, [1] * 18, [0] * 18],
                [[0] * 18, [0] * 18, [1] * 18],
            ]
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

        dir_x = np.array([0, 3, 4, 7, 8, 11])
        dir_y = np.array([22])
        dir_both = np.array([12, 13, 14])

        bound = pp.BoundaryConditionVectorial(g)

        bound.is_dir[0, dir_x] = True
        bound.is_neu[0, dir_x] = False
        bound.is_dir[1, dir_y] = True
        bound.is_neu[1, dir_y] = False
        bound.is_dir[:, dir_both] = True
        bound.is_neu[:, dir_both] = False

        subcell_topology = pp.fvutils.SubcellTopology(g)
        # Move the boundary conditions to sub-faces
        bound.is_dir = bound.is_dir[:, subcell_topology.fno_unique]
        bound.is_rob = bound.is_rob[:, subcell_topology.fno_unique]
        bound.is_neu = bound.is_neu[:, subcell_topology.fno_unique]
        bound.robin_weight = bound.robin_weight[:, :, subcell_topology.fno_unique]
        bound.basis = bound.basis[:, :, subcell_topology.fno_unique]

        # Obtain the face number for each coordinate
        subfno = subcell_topology.subfno_unique
        subfno_nd = np.tile(subfno, (nd, 1)) * nd + np.atleast_2d(np.arange(0, nd)).T

        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

        # expand the indices
        # Define right hand side for Neumann boundary conditions
        # First row indices in rhs matrix
        # Pick out the subface indices
        subfno_neu = bound_exclusion.exclude_robin_dirichlet(
            subfno_nd.ravel("C")
        ).ravel("F")
        # Pick out the Neumann boundary

        is_neu_nd = (
            bound_exclusion.exclude_robin_dirichlet(bound.is_neu.ravel("C"))
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

        subfno_dir = bound_exclusion.exclude_neumann_robin(subfno_nd.ravel("C")).ravel(
            "F"
        )
        is_dir_nd = (
            bound_exclusion.exclude_neumann_robin(bound.is_dir.ravel("C"))
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
