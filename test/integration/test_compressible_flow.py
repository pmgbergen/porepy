import unittest
import numpy as np

from porepy.numerics.compressible import problems, solvers
from porepy.grids import structured
from porepy.fracs import meshing
from porepy.params import tensor


class TestCompressibleFlow(unittest.TestCase):
    def test_single_grid_problem(self):
        problem = UnitSquareInjection()
        solution = problem.solve()
        p = solution['pressure']
        p0 = p[10]
        pT = p[-1]
        p_ref0, p_refT = _reference_solution_single_grid()
        assert np.allclose(p0, p_ref0)
        assert np.allclose(pT, p_refT)

    def test_3d_Fracture_problem(self):
        problem = UnitSquareInjectionMultiDim()
        solution = problem.solve()
        p = solution['pressure'][-1]
        problem.time_disc().split(problem.grid(), 'pressure', p)
        for g, d in problem.grid():
            if g.dim == 3:
                pT = d['pressure']
        p_refT = _reference_solution_multi_grid()
        assert np.allclose(pT, p_refT)


###############################################################################


class UnitSquareInjection(problems.SlightlyCompressible):

    def __init__(self):
        nx = 11
        ny = 11

        physdims = [1, 1]

        g = structured.CartGrid([nx, ny], physdims=physdims)
        g.compute_geometry()
        self.g = g
        # Initialize base class
        problems.SlightlyCompressible.__init__(self)

    #--------grid function--------

    def grid(self):
        return self.g

    #--------Inn/outflow terms---------------

    def source(self, t):
        f = np.zeros(self.g.num_cells)
        source_cell = round(self.g.num_cells / 2)
        f[source_cell] = 20 * self.g.cell_volumes[source_cell]  # m**3/s
        return f * (t < .05)

    #--------Time stepping------------
    def time_step(self):
        return .005

    def end_time(self):
        return 0.1


class MatrixDomain(problems.SubProblem):
    def __init__(self, g):
        problems.SubProblem.__init__(self, g)


class FractureDomain(problems.SubProblem):
    def __init__(self, g):
        problems.SubProblem.__init__(self, g)
        aperture = np.power(0.001, 3 - g.dim)
        self.data()['param'].set_aperture(aperture)

    def permeability(self):
        kxx = 1000 * np.ones(self.g.num_cells)
        return tensor.SecondOrder(3, kxx)


class IntersectionDomain(FractureDomain):
    def __init__(self, g):
        FractureDomain.__init__(self, g)

    def source(self, t):
        assert self.g.num_cells == 1, '0D grid should only have 1 cell'
        f = .4 * self.g.cell_volumes  # m**3/s
        return f * (t < .05)


class UnitSquareInjectionMultiDim(problems.SlightlyCompressibleMultiDim):

    def __init__(self):

        f_1 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])
        f_2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_3 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f_set = [f_1, f_2, f_3]

        g = meshing.cart_grid(f_set, [4, 4, 4], physdims=[1, 1, 1])
        g.compute_geometry()
        g.assign_node_ordering()

        self.g = g
        # Initialize base class
        problems.SlightlyCompressibleMultiDim.__init__(self)

    #--------grid function--------

    def grid(self):
        return self.g

    def set_sub_problems(self):
        self.grid().add_node_props(['problem'])
        for g, d in self.grid():
            if g.dim == 3:
                d['problem'] = MatrixDomain(g)
            elif g.dim == 2 or g.dim == 1:
                d['problem'] = FractureDomain(g)
            elif g.dim == 0:
                d['problem'] = IntersectionDomain(g)
            else:
                raise ValueError('Unkown grid-dimension %d' % g.dim)
    #--------Time stepping------------

    def time_step(self):
        return .005

    def end_time(self):
        return 0.05

###############################################################################


def _reference_solution_single_grid():
    pT = np.array([0.00621076,  0.0064875,  0.00695763,  0.00747579,  0.00787619,
                   0.00802654,  0.00787619,  0.00747579,  0.00695763,  0.0064875,
                   0.00621076,  0.0064875,  0.00679082,  0.00730681,  0.00787666,
                   0.00831785,  0.00848373,  0.00831785,  0.00787666,  0.00730681,
                   0.00679082,  0.0064875,  0.00695763,  0.00730681,  0.00790215,
                   0.00856158,  0.00907363,  0.00926651,  0.00907363,  0.00856158,
                   0.00790215,  0.00730681,  0.00695763,  0.00747579,  0.00787666,
                   0.00856158,  0.00932247,  0.00991504,  0.01013867,  0.00991504,
                   0.00932247,  0.00856158,  0.00787666,  0.00747579,  0.00787619,
                   0.00831785,  0.00907363,  0.00991504,  0.01057172,  0.01081989,
                   0.01057172,  0.00991504,  0.00907363,  0.00831785,  0.00787619,
                   0.00802654,  0.00848373,  0.00926651,  0.01013867,  0.01081989,
                   0.01107748,  0.01081989,  0.01013867,  0.00926651,  0.00848373,
                   0.00802654,  0.00787619,  0.00831785,  0.00907363,  0.00991504,
                   0.01057172,  0.01081989,  0.01057172,  0.00991504,  0.00907363,
                   0.00831785,  0.00787619,  0.00747579,  0.00787666,  0.00856158,
                   0.00932247,  0.00991504,  0.01013867,  0.00991504,  0.00932247,
                   0.00856158,  0.00787666,  0.00747579,  0.00695763,  0.00730681,
                   0.00790215,  0.00856158,  0.00907363,  0.00926651,  0.00907363,
                   0.00856158,  0.00790215,  0.00730681,  0.00695763,  0.0064875,
                   0.00679082,  0.00730681,  0.00787666,  0.00831785,  0.00848373,
                   0.00831785,  0.00787666,  0.00730681,  0.00679082,  0.0064875,
                   0.00621076,  0.0064875,  0.00695763,  0.00747579,  0.00787619,
                   0.00802654,  0.00787619,  0.00747579,  0.00695763,  0.0064875,
                   0.00621076])
    p0 = np.array([0.00163541,  0.00196744,  0.00257963,  0.00333943,  0.00400819,
                   0.00428807,  0.00400819,  0.00333943,  0.00257963,  0.00196744,
                   0.00163541,  0.00196744,  0.00243047,  0.00331231,  0.00446589,
                   0.00555531,  0.00605294,  0.00555531,  0.00446589,  0.00331231,
                   0.00243047,  0.00196744,  0.00257963,  0.00331231,  0.00477516,
                   0.00684537,  0.00903211,  0.010208,  0.00903211,  0.00684537,
                   0.00477516,  0.00331231,  0.00257963,  0.00333943,  0.00446589,
                   0.00684537,  0.01056736,  0.01517516,  0.01844586,  0.01517516,
                   0.01056736,  0.00684537,  0.00446589,  0.00333943,  0.00400819,
                   0.00555531,  0.00903211,  0.01517516,  0.02468009,  0.0353479,
                   0.02468009,  0.01517516,  0.00903211,  0.00555531,  0.00400819,
                   0.00428807,  0.00605294,  0.010208,  0.01844586,  0.0353479,
                   0.07602835,  0.0353479,  0.01844586,  0.010208,  0.00605294,
                   0.00428807,  0.00400819,  0.00555531,  0.00903211,  0.01517516,
                   0.02468009,  0.0353479,  0.02468009,  0.01517516,  0.00903211,
                   0.00555531,  0.00400819,  0.00333943,  0.00446589,  0.00684537,
                   0.01056736,  0.01517516,  0.01844586,  0.01517516,  0.01056736,
                   0.00684537,  0.00446589,  0.00333943,  0.00257963,  0.00331231,
                   0.00477516,  0.00684537,  0.00903211,  0.010208,  0.00903211,
                   0.00684537,  0.00477516,  0.00331231,  0.00257963,  0.00196744,
                   0.00243047,  0.00331231,  0.00446589,  0.00555531,  0.00605294,
                   0.00555531,  0.00446589,  0.00331231,  0.00243047,  0.00196744,
                   0.00163541,  0.00196744,  0.00257963,  0.00333943,  0.00400819,
                   0.00428807,  0.00400819,  0.00333943,  0.00257963,  0.00196744,
                   0.00163541])
    return p0, pT


def _reference_solution_multi_grid():
    pT = np.array([0.00983112,  0.01637886,  0.01637886,  0.00983112,  0.01637886,
                   0.02264049,  0.02264049,  0.01637886,  0.01637886,  0.02264049,
                   0.02264049,  0.01637886,  0.00983112,  0.01637886,  0.01637886,
                   0.00983112,  0.01637886,  0.02264049,  0.02264049,  0.01637886,
                   0.02264049,  0.03234641,  0.03234641,  0.02264049,  0.02264049,
                   0.03234641,  0.03234641,  0.02264049,  0.01637886,  0.02264049,
                   0.02264049,  0.01637886,  0.01637886,  0.02264049,  0.02264049,
                   0.01637886,  0.02264049,  0.03234641,  0.03234641,  0.02264049,
                   0.02264049,  0.03234641,  0.03234641,  0.02264049,  0.01637886,
                   0.02264049,  0.02264049,  0.01637886,  0.00983112,  0.01637886,
                   0.01637886,  0.00983112,  0.01637886,  0.02264049,  0.02264049,
                   0.01637886,  0.01637886,  0.02264049,  0.02264049,  0.01637886,
                   0.00983112,  0.01637886,  0.01637886,  0.00983112])
    return pT
