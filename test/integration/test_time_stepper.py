import unittest
import numpy as np

from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics.time_stepper import Implicit, Explicit, BDF2
from porepy.numerics.time_stepper import CrankNicolson
from porepy.grids import structured
from porepy.fracs import meshing
from porepy.params import tensor


class TestBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        f_1 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])
        f_2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_3 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f_set = [f_1, f_2, f_3]

        self.gb = meshing.cart_grid(f_set, [4, 4, 4], physdims=[1, 1, 1])
        self.gb.add_node_props(["transport_data"])
        for g, d in self.gb:
            d["transport_data"] = ParabolicDataAssigner(g, d)

    def test_implicit_solver(self):
        """Inject 1 in cell 0. Test that rhs and pressure solution
        is correct"""
        problem = UnitSquareInjectionMultiDim(self.gb)
        problem.update(0.0)
        solver = Implicit(problem)
        solver.solve()

        assert np.sum(np.abs(solver.rhs) > 1e-6) == 1
        assert np.sum(np.abs(solver.rhs - 1) < 1e-6) == 1
        assert np.sum(np.abs(solver.p) > 1e-6) == 1
        assert np.sum(np.abs(solver.p - 1) < 1e-6) == 1

    def test_BDF2_solver(self):
        """Inject 1 in cell 0. Test that rhs and pressure solution
        is correct"""
        problem = UnitSquareInjectionTwoSteps(self.gb)
        problem.update(0.0)
        solver = BDF2(problem)
        solver.update(solver.dt)
        solver.reassemble()
        solver.step()
        # The first step should be an implicit step
        assert np.sum(np.abs(solver.rhs) > 1e-6) == 1
        assert np.sum(np.abs(solver.rhs - 1.0) < 1e-6) == 1
        assert np.sum(np.abs(solver.p) > 1e-6) == 1
        assert np.sum(np.abs(solver.p - 1.0 / 2.0) < 1e-6) == 1
        assert np.allclose(solver.p0, 0)
        assert np.allclose(solver.p_1, 0)

        solver.update(2 * solver.dt)
        solver.reassemble()
        solver.step()
        # The second step should be a full bdf2 step
        assert np.sum(np.abs(solver.rhs) > 1e-6) == 1
        assert np.sum(np.abs(solver.rhs - 2.0) < 1e-6) == 1
        assert np.sum(np.abs(solver.p) > 1e-6) == 1
        assert np.sum(np.abs(solver.p - 1) < 1e-6) == 1
        assert np.sum(np.abs(solver.p0 - 0.5) < 1e-6) == 1
        assert np.allclose(solver.p_1, 0.0)

    def test_explicit_solver(self):
        """Inject 1 in cell 0. Test that rhs and pressure solution
        is correct"""
        problem = UnitSquareInjectionMultiDim(self.gb)
        problem.update(0.0)
        solver = Explicit(problem)
        solver.solve()
        assert np.sum(np.abs(solver.rhs) > 1e-6) == 0
        assert np.sum(np.abs(solver.p) > 1e-6) == 0

    def test_CrankNicolson_solver(self):
        """Inject 1 in cell 0. Test that rhs and pressure solution
        is correct"""
        problem = UnitSquareInjectionMultiDim(self.gb)
        problem.update(0.0)
        solver = CrankNicolson(problem)
        solver.solve()

        assert np.sum(np.abs(solver.rhs) > 1e-6) == 1
        assert np.sum(np.abs(solver.rhs - 0.5) < 1e-6) == 1
        assert np.sum(np.abs(solver.p) > 1e-6) == 1
        assert np.sum(np.abs(solver.p - 0.5) < 1e-6) == 1


###############################################################################


class UnitSquareInjectionMultiDim(ParabolicModel):
    def __init__(self, gb):
        # Initialize base class
        ParabolicModel.__init__(self, gb)

    def space_disc(self):
        return self.source_disc()

    # --------Time stepping------------

    def update(self, t):
        for g, d in self.grid():
            source = np.zeros(g.num_cells)
            if g.dim == 0 and t > self.time_step() - 1e-5:
                source[0] = 1.0
            d["param"].set_source("transport", source)


###############################################################################
class UnitSquareInjectionTwoSteps(UnitSquareInjectionMultiDim):
    def __init__(self, gb):
        UnitSquareInjectionMultiDim.__init__(self, gb)

    def time_step(self):
        return 0.5
