import unittest
from tests import test_utils

import numpy as np

import porepy as pp


def setup_2d_1d(nx, simplex_grid=False):
    if not simplex_grid:
        mesh_args = nx
    else:
        mesh_size = 0.08
        mesh_args = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    end_x = [0.2, 0.8]
    end_y = [0.8, 0.2]
    mdg, _ = pp.md_grids_2d.two_intersecting(mesh_args, end_x, end_y, simplex_grid)
    aperture = 0.01 / np.max(nx)
    for sd, data in mdg.subdomains(return_data=True):
        a = np.power(aperture, mdg.dim_max() - sd.dim) * np.ones(sd.num_cells)
        kxx = np.ones(sd.num_cells) * a
        perm = pp.SecondOrderTensor(kxx)
        specified_parameters = {"second_order_tensor": perm}
        if sd.dim == 2:
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                sd, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(sd.num_faces)
            bc_val[bound_faces] = sd.face_centers[1, bound_faces]
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

        pp.initialize_default_data(sd, data, "flow", specified_parameters)

    for intf, data in mdg.interfaces(return_data=True):
        kn = 1.0 / aperture
        data[pp.PARAMETERS] = pp.Parameters(
            intf, ["flow"], [{"normal_diffusivity": kn}]
        )
        data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
    return mdg


def check_pressures(mdg):
    """
    Check that the pressures are not too far from an approximate
    analytical solution. Note that the solution depends
    on the grid quality. Also sensitive to the way in which
    the tpfa half transmissibilities are computed.
    """
    for sd, data in mdg.subdomains(return_data=True):
        pressure = data[pp.STATE]["pressure"]
        pressure_analytic = sd.cell_centers[1]
        p_diff = pressure - pressure_analytic
        if np.max(np.abs(p_diff)) >= 2e-2:
            return False
    return True


class BasicsTest(unittest.TestCase):
    def test_uniform_flow_cart_2d_1d_cartesian(self):
        # Structured Cartesian grid
        mdg = setup_2d_1d(np.array([10, 10]))

        key = "flow"
        tpfa = pp.Tpfa(key)
        assembler = test_utils.setup_flow_assembler(mdg, tpfa, key)
        test_utils.solve_and_distribute_pressure(mdg, assembler)

        self.assertTrue(check_pressures(mdg))

    def test_uniform_flow_cart_2d_1d_simplex(self):
        # Unstructured simplex grid
        mdg = setup_2d_1d(np.array([10, 10]), simplex_grid=True)

        key = "flow"
        tpfa = pp.Tpfa(key)
        assembler = test_utils.setup_flow_assembler(mdg, tpfa, key)
        test_utils.solve_and_distribute_pressure(mdg, assembler)
        self.assertTrue(check_pressures(mdg))


if __name__ == "__main__":
    unittest.main()
