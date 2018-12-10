import numpy as np
import unittest

import porepy as pp
from test import test_utils


def setup_cart_2d(nx):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1])
    gb.compute_geometry()
    gb.assign_node_ordering()
    kw = "flow"
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(gb.dim_max(), kxx)
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim) * np.ones(g.num_cells)
        specified_parameters = {"aperture": a, "second_order_tensor": perm}

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

        d_l = gb.node_props(gl)
        kn = 1.0 / np.mean(d_l[pp.PARAMETERS][kw]["aperture"])
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
        tpfa = pp.Tpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, tpfa, key)
        test_utils.solve_and_distribute_pressure(gb, assembler)
        for g, d in gb:
            pressure = d["pressure"]
            pressure_analytic = g.cell_centers[1]
            p_diff = pressure - pressure_analytic
            self.assertTrue(np.max(np.abs(p_diff)) < 0.05)


if __name__ == "__main__":
    unittest.main()
