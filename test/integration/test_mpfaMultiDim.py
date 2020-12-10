import unittest
from test import test_utils

import numpy as np

import porepy as pp


def setup_cart_2d(nx):
    gb, _ = pp.grid_buckets_2d.two_intersecting(
        nx, [0.2, 0.8], [0.8, 0.2], simplex=False
    )
    kw = "flow"
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
        # Initialize data and matrix dictionaries in d
        pp.initialize_default_data(g, d, kw, specified_parameters)

    for e, d in gb.edges():
        # Compute normal permeability
        gl, _ = gb.nodes_of_edge(e)
        kn = 1.0 / aperture
        data = {"normal_diffusivity": kn}
        # Add parameters
        d[pp.PARAMETERS] = pp.Parameters(keywords=[kw], dictionaries=[data])
        # Initialize matrix dictionary
        d[pp.DISCRETIZATION_MATRICES] = {kw: {}}
    return gb


class TestMpfaMultiDim(unittest.TestCase):
    def test_uniform_flow_cart_2d(self):
        # Structured Cartesian grid
        gb = setup_cart_2d(np.array([10, 10]))

        key = "flow"
        mpfa = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, mpfa, key)
        test_utils.solve_and_distribute_pressure(gb, assembler)
        for g, d in gb:
            pressure = d[pp.STATE]["pressure"]
            pressure_analytic = g.cell_centers[1]
            p_diff = pressure - pressure_analytic
            self.assertTrue(np.max(np.abs(p_diff)) < 0.05)


if __name__ == "__main__":
    unittest.main()
