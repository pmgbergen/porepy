import unittest
from test import test_utils

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
    gb, _ = pp.grid_buckets_2d.two_intersecting(mesh_args, end_x, end_y, simplex_grid)
    aperture = 0.01 / np.max(nx)
    for g, d in gb:
        a = np.power(aperture, gb.dim_max() - g.dim) * np.ones(g.num_cells)
        kxx = np.ones(g.num_cells) * a
        perm = pp.SecondOrderTensor(kxx)
        specified_parameters = {"second_order_tensor": perm}
        if g.dim == 2:
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

        pp.initialize_default_data(g, d, "flow", specified_parameters)

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        mg = d["mortar_grid"]
        kn = 1.0 / aperture
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
    return gb


def check_pressures(gb):
    """
    Check that the pressures are not too far from an approximate
    analytical solution. Note that the solution depends
    on the grid quality. Also sensitive to the way in which
    the tpfa half transmissibilities are computed.
    """
    for g, d in gb:
        pressure = d[pp.STATE]["pressure"]
        pressure_analytic = g.cell_centers[1]
        p_diff = pressure - pressure_analytic
        if np.max(np.abs(p_diff)) >= 2e-2:
            return False
    return True


class BasicsTest(unittest.TestCase):
    def test_uniform_flow_cart_2d_1d_cartesian(self):
        # Structured Cartesian grid
        gb = setup_2d_1d(np.array([10, 10]))

        key = "flow"
        tpfa = pp.Tpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, tpfa, key)
        test_utils.solve_and_distribute_pressure(gb, assembler)

        self.assertTrue(check_pressures(gb))

    def test_uniform_flow_cart_2d_1d_simplex(self):
        # Unstructured simplex grid
        gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)

        key = "flow"
        tpfa = pp.Tpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, tpfa, key)
        test_utils.solve_and_distribute_pressure(gb, assembler)
        self.assertTrue(check_pressures(gb))


if __name__ == "__main__":
    unittest.main()
