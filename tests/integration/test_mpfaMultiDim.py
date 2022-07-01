import unittest
from tests import test_utils

import numpy as np

import porepy as pp


def setup_cart_2d(nx):
    mdg, _ = pp.md_grids_2d.two_intersecting(
        nx, [0.2, 0.8], [0.8, 0.2], simplex=False
    )
    kw = "flow"
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
        # Initialize data and matrix dictionaries in d
        pp.initialize_default_data(sd, data, kw, specified_parameters)

    for intf, data in mdg.interfaces(return_data=True):
        # Compute normal permeability
        kn = 1.0 / aperture
        specified_data = {"normal_diffusivity": kn}
        # Add parameters
        data[pp.PARAMETERS] = pp.Parameters(
            keywords=[kw], dictionaries=[specified_data]
        )
        # Initialize matrix dictionary
        data[pp.DISCRETIZATION_MATRICES] = {kw: {}}
    return mdg


class TestMpfaMultiDim(unittest.TestCase):
    def test_uniform_flow_cart_2d(self):
        # Structured Cartesian grid
        mdg = setup_cart_2d(np.array([10, 10]))

        key = "flow"
        mpfa = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(mdg, mpfa, key)
        test_utils.solve_and_distribute_pressure(mdg, assembler)
        for sd, data in mdg.subdomains(return_data=True):
            pressure = data[pp.STATE]["pressure"]
            pressure_analytic = sd.cell_centers[1]
            p_diff = pressure - pressure_analytic
            self.assertTrue(np.max(np.abs(p_diff)) < 0.05)


if __name__ == "__main__":
    unittest.main()
