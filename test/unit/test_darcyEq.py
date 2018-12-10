import numpy as np
import unittest
import porepy as pp


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def disabled_test_elliptic_data_default_values(self):
        """
        test that the elliptic data initialize the correct data.
        """
        p = np.random.rand(3, 10)
        g = pp.TetrahedralGrid(p)
        param = pp.Parameters(g)
        elliptic_data = dict()
        EllipticDataAssigner(g, elliptic_data)
        elliptic_param = elliptic_data[pp.PARAMETERS]

        self.check_parameters(elliptic_param, param)

    def disabled_test_elliptic_data_given_values(self):
        """
        test that the elliptic data initialize the correct data.
        """
        p = np.random.rand(3, 10)
        g = pp.TetrahedralGrid(p)
        kw = "flow"
        # Set values

        bc_val = np.pi * np.ones(g.num_faces)
        dir_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_cond = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
        porosity = 1 / np.pi * np.ones(g.num_cells)
        apperture = 0.5 * np.ones(g.num_cells)
        kxx = 2 * np.ones(g.num_cells)
        kyy = 3 * np.ones(g.num_cells)
        K = pp.SecondOrderTensor(g.dim, kxx, kyy)
        source = 42 * np.ones(g.num_cells)
        # Assign to parameter
        parameter_dictionary = {
            "bc": bc_cond,
            "bc_values": bc_val,
            "porosity": porosity,
            "aperture": apperture,
            "second_order_tensor": K,
            "source": source,
        }
        # Define EllipticData class

        class Data(EllipticDataAssigner):
            def __init__(self, g, data):
                EllipticDataAssigner.__init__(self, g, data, keyword=kw)

            def bc(self):
                return bc_cond

            def bc_val(self):
                return bc_val

            def porosity(self):
                return porosity

            def aperture(self):
                return apperture

            def permeability(self):
                return K

            def source(self):
                return source

        elliptic_data = dict()
        Data(g, elliptic_data)
        elliptic_param = elliptic_data[pp.PARAMETERS][kw]

        self.check_parameters(elliptic_param, parameter_dictionary)

    # ------------------------------------------------------------------------------#

    def check_parameters(self, param_c, param_t):
        bc_c = param_c["bc"]
        bc_t = param_t["bc"]
        k_c = param_c["second_order_tensor"].values
        k_t = param_t["second_order_tensor"].values

        self.assertTrue(np.alltrue(bc_c.is_dir == bc_t.is_dir))
        self.assertTrue(np.alltrue(param_c["bc_values"] == param_t["bc_values"]))
        self.assertTrue(np.alltrue(param_c["porosity"] == param_t["porosity"]))
        self.assertTrue(np.alltrue(param_c["aperture"] == param_t["aperture"]))
        self.assertTrue(np.alltrue(k_c == k_t))
        self.assertTrue(np.alltrue(param_c["source"] == param_t["source"]))


if __name__ == "__main__":
    unittest.main()
