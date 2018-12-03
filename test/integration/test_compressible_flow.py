""" Disabled test of the obsolete SlightlyCompressible class.
"""
import unittest
import numpy as np

import porepy as pp


class TestCompressibleFlow(unittest.TestCase):
    def disabled_test_3d_Fracture_problem(self):
        problem = UnitSquareInjectionMultiDim()
        problem.solve()
        problem.pressure()
        for g, d in problem.grid():
            if g.dim == 3:
                pT = d["flow"]
        p_refT = _reference_solution_multi_grid()
        self.assertTrue(np.allclose(pT, p_refT))


###############################################################################


class MatrixDomain(pp.SlightlyCompressibleDataAssigner):
    def __init__(self, g, d):
        pp.SlightlyCompressibleDataAssigner.__init__(self, g, d)


class FractureDomain(pp.SlightlyCompressibleDataAssigner):
    def __init__(self, g, d):
        pp.SlightlyCompressibleDataAssigner.__init__(self, g, d)
        aperture = np.power(0.001, 3 - g.dim) * np.ones(g.num_cells)
        self.data()[pp.PARAMETERS]["flow"]["aperture"] = aperture

    def permeability(self):
        kxx = 1000 * np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(3, kxx)


class IntersectionDomain(FractureDomain):
    def __init__(self, g, d):
        FractureDomain.__init__(self, g, d)

    def source(self, t):
        if not self.grid().num_cells == 1:
            raise AssertionError("0D grid should only have 1 cell")
        f = 0.4 * self.grid().cell_volumes  # m**3/s
        return f * (t < 0.05)


def set_sub_problems(gb):
    for g, d in gb:
        if g.dim == 3:
            d["flow_data"] = MatrixDomain(g, d)
        elif g.dim == 2 or g.dim == 1:
            d["flow_data"] = FractureDomain(g, d)
        elif g.dim == 0:
            d["flow_data"] = IntersectionDomain(g, d)
        else:
            raise ValueError("Unkown grid-dimension %d" % g.dim)
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        d_l = gb.node_props(gl)
        mortar_grid = d["mortar_grid"]
        #        d_l["param"].get_aperture())
        kn = 1000 / np.mean(d_l[pp.PARAMETERS]["flow"]["aperture"])
        data = {"normal_diffusivity": kn}
        d[pp.PARAMETERS] = pp.Parameters(mortar_grid, ["flow"], [data])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}


class UnitSquareInjectionMultiDim(pp.SlightlyCompressibleModel):
    def __init__(self):

        f_1 = np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
        f_2 = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_3 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        f_set = [f_1, f_2, f_3]

        g = pp.meshing.cart_grid(f_set, [4, 4, 4], physdims=[1, 1, 1])
        g.compute_geometry()
        g.assign_node_ordering()
        set_sub_problems(g)
        self.g = g
        # Initialize base class
        pp.SlightlyCompressibleModel.__init__(self, self.g, "flow")

    # --------grid function--------

    def grid(self):
        return self.g

    # --------Time stepping------------

    def time_step(self):
        return 0.005

    def end_time(self):
        return 0.05


###############################################################################


def _reference_solution_multi_grid():
    pT = np.array(
        [
            0.00983112,
            0.01637886,
            0.01637886,
            0.00983112,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.00983112,
            0.01637886,
            0.01637886,
            0.00983112,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.02264049,
            0.03234641,
            0.03234641,
            0.02264049,
            0.02264049,
            0.03234641,
            0.03234641,
            0.02264049,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.02264049,
            0.03234641,
            0.03234641,
            0.02264049,
            0.02264049,
            0.03234641,
            0.03234641,
            0.02264049,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.00983112,
            0.01637886,
            0.01637886,
            0.00983112,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.01637886,
            0.02264049,
            0.02264049,
            0.01637886,
            0.00983112,
            0.01637886,
            0.01637886,
            0.00983112,
        ]
    )
    return pT


if __name__ == "__main__":
    unittest.main()
