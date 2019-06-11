import numpy as np
import unittest

import porepy as pp
from test import test_utils


def setup_2d_1d(nx, simplex_grid=False):
    if not simplex_grid:
        frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
        frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
        fracs = [frac1, frac2]

        gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1])
    else:
        p = np.array([[0.2, 0.8, 0.5, 0.5], [0.5, 0.5, 0.8, 0.2]])
        e = np.array([[0, 2], [1, 3]])
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        network = pp.FractureNetwork2d(p, e, domain)
        mesh_size = 0.08
        mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

        gb = network.mesh(mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()
    aperture = 0.01 / np.max(nx)
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(gb.dim_max(), kxx)
        a = np.power(aperture, gb.dim_max() - g.dim) * np.ones(g.num_cells)
        specified_parameters = {"aperture": a, "second_order_tensor": perm}
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
