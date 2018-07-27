import numpy as np
import unittest

from porepy.numerics.mechanics import StaticModel
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import bc


class BasicsTest(unittest.TestCase):
    def test_zero_force(self):
        """
        if nothing is touched nothing should happen
        """
        f = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]).T
        g = meshing.cart_grid([f], [4, 4, 2]).grids_of_dimension(3)[0]
        data = {"param": Parameters(g)}

        bound = bc.BoundaryCondition(g, g.get_all_boundary_faces(), "dir")
        data["param"].set_bc("mechanics", bound)

        solver = StaticModel(g, data)
        d = solver.solve()
        solver.traction("T")
        assert np.all(d == 0)
        assert np.all(data["T"] == 0)

    def test_unit_slip(self):
        """
        Unit slip of fractures
        """

        f = np.array([[0, 0, 1], [0, 2, 1], [2, 2, 1], [2, 0, 1]]).T
        g = meshing.cart_grid([f], [2, 2, 2]).grids_of_dimension(3)[0]
        data = {"param": Parameters(g)}

        bound = bc.BoundaryCondition(g, g.get_all_boundary_faces(), "dir")
        data["param"].set_bc("mechanics", bound)

        slip = np.ones(g.dim * g.num_faces)
        data["param"].set_slip_distance(slip)

        solver = StaticModel(g, data)
        solver.solve()

        solver.frac_displacement("d_f")
        solver.displacement("d_c")

        solver.save(["d_c"])

        # test cell displacent around fractures
        d_c = data["d_c"]
        d_c = d_c.reshape((3, -1), order="F")
        frac_faces = g.frac_pairs
        frac_left = frac_faces[0]
        frac_right = frac_faces[1]

        cell_left = np.ravel(np.argwhere(g.cell_faces[frac_left, :])[:, 1])
        cell_right = np.ravel(np.argwhere(g.cell_faces[frac_right, :])[:, 1])

        # Test traction
        solver.traction("T")
        T_left = data["T"][:, frac_left]
        T_right = data["T"][:, frac_right]

        assert np.allclose(T_left, T_right)

        # we have u_lhs - u_rhs = 1 so u_lhs should be positive
        assert np.all(d_c[:, cell_left] > 0)
        assert np.all(d_c[:, cell_right] < 0)

        # Test fracture displacement
        u_left = data["d_f"][:, : int(round(data["d_f"].shape[1] / 2))]
        u_right = data["d_f"][:, int(round(data["d_f"].shape[1] / 2)) :]
        assert np.all(np.abs(u_left - u_right - 1) < 1e-10)
