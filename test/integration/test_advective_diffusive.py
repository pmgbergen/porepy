import numpy as np
import unittest

from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics import elliptic
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import tensor, bc
from porepy.params.units import *
from porepy.grids.grid import FaceTag
from porepy.numerics.fv import fvutils


class BasicsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        f = np.array([[1, 1, 4, 4], [1, 4, 4, 1], [2, 2, 2, 2]])
        box = {'xmin': 0, 'ymin': 0, 'zmin': 0,
               'xmax': 5, 'ymax':  5, 'zmax': 5}
        mesh_size = 1.0
        mesh_kwargs = {'mode': 'constant',
                       'value': mesh_size, 'bound_value': mesh_size}
        self.gb3d = meshing.simplex_grid([f], box, mesh_size=mesh_kwargs)
        unittest.TestCase.__init__(self, *args, **kwargs)

    #------------------------------------------------------------------------------#

    def test_src_2d(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        gb = meshing.cart_grid([], [10, 10])

        for sub_g, d in gb:
            if sub_g.dim == 2:
                d['transport_data'] = InjectionDomain(sub_g, d)
            else:
                d['transport_data'] = MatrixDomain(sub_g, d)

        problem = SourceProblem(gb, physics='transport')
        problem.solve()
        dE = change_in_energy(problem)
        assert np.abs(dE - 10) < 1e-6

    def test_src_3d(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        delete_node_data(self.gb3d)

        for sub_g, d in self.gb3d:
            if sub_g.dim == 2:
                d['transport_data'] = InjectionDomain(sub_g, d)
            else:
                d['transport_data'] = MatrixDomain(sub_g, d)

        problem = SourceProblem(self.gb3d)
        problem.solve()
        dE = change_in_energy(problem)
        assert np.abs(dE - 10) < 1e-6

    def test_src_advective(self):
        delete_node_data(self.gb3d)

        for g, d in self.gb3d:
            if g.dim == 2:
                d['transport_data'] = InjectionDomain(g, d)
            else:
                d['transport_data'] = MatrixDomain(g, d)
        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveProblem(self.gb3d)
        problem.solve()
        dE = change_in_energy(problem)
        assert np.abs(dE - 10) < 1e-6

    def test_src_advective_diffusive(self):
        delete_node_data(self.gb3d)
        for g, d in self.gb3d:
            if g.dim == 2:
                d['transport_data'] = InjectionDomain(g, d)
            else:
                d['transport_data'] = MatrixDomain(g, d)
        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveDiffusiveProblem(self.gb3d)
        problem.solve()
        dE = change_in_energy(problem)
        assert np.abs(dE - 10) < 1e-6

    def test_constant_temp(self):
        delete_node_data(self.gb3d)
        for g, d in self.gb3d:
            if g.dim == 2:
                d['transport_data'] = InjectionDomain(g, d)
            else:
                d['transport_data'] = MatrixDomain(g, d)
        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveDiffusiveDirBound(self.gb3d)
        problem.solve()
        for _, d in self.gb3d:
            T_list = d['transport']
            const_temp = [np.allclose(T, 10) for T in T_list]
            assert np.all(const_temp)


class SourceProblem(ParabolicModel):
    def __init__(self, g, physics='transport'):
        ParabolicModel.__init__(self, g, physics=physics)

    def space_disc(self):
        return self.source_disc()

    def time_step(self):
        return 0.5


class SourceAdvectiveProblem(ParabolicModel):
    def __init__(self, g, physics='transport'):
        ParabolicModel.__init__(self, g, physics=physics)

    def space_disc(self):
        return self.source_disc(), self.advective_disc()

    def time_step(self):
        return 0.5


class SourceAdvectiveDiffusiveProblem(ParabolicModel):
    def __init__(self, g, physics='transport'):
        ParabolicModel.__init__(self, g, physics=physics)

    def space_disc(self):
        return self.source_disc(), self.advective_disc(), self.diffusive_disc()

    def time_step(self):
        return 0.5


class SourceAdvectiveDiffusiveDirBound(SourceAdvectiveDiffusiveProblem):
    def __init__(self, g, physics='transport'):
        SourceAdvectiveDiffusiveProblem.__init__(self, g, physics=physics)

    def bc(self):
        dir_faces = self.grid().has_face_tag(FaceTag.DOMAIN_BOUNDARY)
        dir_faces = np.ravel(np.argwhere(dir_faces))
        bc_cond = bc.BoundaryCondition(
            self.grid(), dir_faces, ['dir'] * dir_faces.size)
        return bc_cond

    def bc_val(self, t):
        dir_faces = self.grid().has_face_tag(FaceTag.DOMAIN_BOUNDARY)
        dir_faces = np.ravel(np.argwhere(dir_faces))
        val = np.zeros(self.grid().num_faces)
        val[dir_faces] = 10 * PASCAL
        return val


def source(g, t):
    tol = 1e-4
    value = np.zeros(g.num_cells)
    cell_coord = np.atleast_2d(np.mean(g.cell_centers, axis=1)).T
    cell = np.argmin(
        np.sum(np.abs(g.cell_centers - cell_coord), axis=0))

    value[cell] = 1.0 * KILOGRAM / SECOND
    return value


class MatrixDomain(ParabolicDataAssigner):
    def __init__(self, g, d, physics='transport'):
        ParabolicDataAssigner.__init__(self, g, d, physics)

    def initial_condition(self):
        return 10 * np.ones(self.grid().num_cells)


class InjectionDomain(MatrixDomain):
    def __init__(self, g, d, physics='transport'):
        MatrixDomain.__init__(self, g, d, physics)

    def source(self, t):
        flux = source(self.grid(), t)
        T_in = 10
        return flux * T_in

    def aperture(self):
        return 0.1 * np.ones(self.grid().num_cells)


def change_in_energy(problem):
    dE = 0
    problem.advective_disc().split(
        problem.grid(), 'T', problem._solver.p)
    for g, d in problem.grid():
        p = d['T']
        p0 = d['transport_data'].initial_condition()
        V = g.cell_volumes * d['param'].get_aperture()
        dE += np.sum((p - p0) * V)

    return dE


def solve_elliptic_problem(gb):
    for g, d in gb:
        if g.dim == 2:
            d['param'].set_source(
                'flow', source(g, 0.0))

        dir_bound = np.argwhere(g.has_face_tag(FaceTag.DOMAIN_BOUNDARY))
        bc_cond = bc.BoundaryCondition(
            g, dir_bound, ['dir'] * dir_bound.size)
        d['param'].set_bc('flow', bc_cond)

    gb.add_edge_prop('param')
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        d['param'] = Parameters(g_h)
    flux = elliptic.EllipticModel(gb)
    p = flux.solve()
    flux.split('p')
    fvutils.compute_discharges(gb)


def delete_node_data(gb):
    for _, d in gb:
        d.clear()

    gb.assign_node_ordering()
