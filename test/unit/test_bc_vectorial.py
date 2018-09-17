import unittest
import numpy as np

import math

from porepy.grids import structured
from porepy.params.bc import BoundaryConditionVectorial
from porepy.numerics.fv import fvutils

"""
Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs_nd
for handling boundary conditions expressed in a vectorial form
"""


class testBoundaryConditionsVectorial(unittest.TestCase):
    def test_2d(self):

        """
        The domain consists of a 3x3 mesh.
        Bottom faces are dirichlet
        Left and right faces are rolling along y (dir_x and neu_y)
        Top mid face is rolling along x (neu_x and dir_y)
        Top left and right faces are neumann
        """

        g = structured.CartGrid([3, 3])
        g.compute_geometry()
        nd = g.dim

        boundary_faces = np.array([0, 3, 4, 7, 8, 11, 12, 13, 14, 22])
        boundary_faces_type = ["dir_x"] * 6 + ["dir"] * 3 + ["dir_y"] * 1

        bound = BoundaryConditionVectorial(g, boundary_faces, boundary_faces_type)

        subcell_topology = fvutils.SubcellTopology(g)
        fno = subcell_topology.fno_unique
        bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

        # Neumann
        is_neu_x = bound_exclusion.exclude_dirichlet_x(
            bound.is_neu[0, fno].astype("int64")
        )
        neu_ind_single_x = np.argwhere(is_neu_x).ravel("F")

        is_neu_y = bound_exclusion.exclude_dirichlet_y(
            bound.is_neu[1, fno].astype("int64")
        )
        neu_ind_single_y = np.argwhere(is_neu_y).ravel("F")
        neu_ind_single_y += is_neu_x.size

        # We also need to account for all half faces, that is, do not exclude
        # Dirichlet and Neumann boundaries.
        neu_ind_single_all_x = np.argwhere(bound.is_neu[0, fno].astype("int")).ravel(
            "F"
        )
        neu_ind_single_all_y = np.argwhere(bound.is_neu[1, fno].astype("int")).ravel(
            "F"
        )

        # expand the indices
        # this procedure replaces the method 'expand_ind' in the above
        # method 'create_bound_rhs'

        # 1 - stack and sort indices

        is_bnd_neu_x = nd * neu_ind_single_all_x
        is_bnd_neu_y = nd * neu_ind_single_all_y + 1

        is_bnd_neu = np.sort(np.append(is_bnd_neu_x, [is_bnd_neu_y]))

        # 2 - find the indices corresponding to the boundary components
        # having Neumann condtion

        ind_is_bnd_neu_x = np.argwhere(np.isin(is_bnd_neu, is_bnd_neu_x)).ravel("F")
        ind_is_bnd_neu_y = np.argwhere(np.isin(is_bnd_neu, is_bnd_neu_y)).ravel("F")

        neu_ind_sz = ind_is_bnd_neu_x.size + ind_is_bnd_neu_y.size

        # 3 - create the expanded neu_ind array

        neu_ind = np.zeros(neu_ind_sz, dtype="int")

        neu_ind[ind_is_bnd_neu_x] = neu_ind_single_x
        neu_ind[ind_is_bnd_neu_y] = neu_ind_single_y

        assert np.alltrue(
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

        # Dirichlet, same procedure
        is_dir_x = bound_exclusion.exclude_neumann_x(
            bound.is_dir[0, fno].astype("int64")
        )
        dir_ind_single_x = np.argwhere(is_dir_x).ravel("F")

        is_dir_y = bound_exclusion.exclude_neumann_y(
            bound.is_dir[1, fno].astype("int64")
        )
        dir_ind_single_y = np.argwhere(is_dir_y).ravel("F")
        dir_ind_single_y += is_dir_x.size

        dir_ind_single_all_x = np.argwhere(bound.is_dir[0, fno].astype("int")).ravel(
            "F"
        )
        dir_ind_single_all_y = np.argwhere(bound.is_dir[1, fno].astype("int")).ravel(
            "F"
        )

        # expand indices

        is_bnd_dir_x = nd * dir_ind_single_all_x
        is_bnd_dir_y = nd * dir_ind_single_all_y + 1

        is_bnd_dir = np.sort(np.append(is_bnd_dir_x, [is_bnd_dir_y]))

        ind_is_bnd_dir_x = np.argwhere(np.isin(is_bnd_dir, is_bnd_dir_x)).ravel("F")
        ind_is_bnd_dir_y = np.argwhere(np.isin(is_bnd_dir, is_bnd_dir_y)).ravel("F")

        dir_ind_sz = ind_is_bnd_dir_x.size + ind_is_bnd_dir_y.size

        dir_ind = np.zeros(dir_ind_sz, dtype="int")

        dir_ind[ind_is_bnd_dir_x] = dir_ind_single_x
        dir_ind[ind_is_bnd_dir_y] = dir_ind_single_y

        assert np.alltrue(
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


if __name__ == "__main__":
    unittest.main()
