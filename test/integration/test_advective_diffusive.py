""" Disabled test of the obsolete Parabolic module.
"""
import numpy as np
import unittest

import porepy as pp


class BasicsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        f = np.array([[1, 1, 4, 4], [1, 4, 4, 1], [2, 2, 2, 2]])
        box = {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 5, "ymax": 5, "zmax": 5}
        mesh_size = 1.0
        mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
        self.gb3d = pp.meshing.simplex_grid([f], box, **mesh_kwargs)
        unittest.TestCase.__init__(self, *args, **kwargs)

    # ------------------------------------------------------------------------------#

    def disabled_test_src_2d(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        gb = pp.meshing.cart_grid([], [10, 10])

        for sub_g, d in gb:
            if sub_g.dim == 2:
                d["transport_data"] = InjectionDomain(sub_g, d)
            else:
                d["transport_data"] = MatrixDomain(sub_g, d)
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        for e, d in gb.edges():
            gl, _ = gb.nodes_of_edge(e)
            kn = 1.0 * gl.num_cells
            data = {"normal_diffusivity": kn}
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["transport"], [data]
            )
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}
        problem = SourceProblem(gb, keyword="transport")
        problem.solve()
        dE = change_in_energy(problem)
        self.assertTrue(np.abs(dE - 10) < 1e-6)

    def disabled_test_src_3d(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        delete_data(self.gb3d)

        for sub_g, d in self.gb3d:
            if sub_g.dim == 2:
                d["transport_data"] = InjectionDomain(sub_g, d)
            else:
                d["transport_data"] = MatrixDomain(sub_g, d)
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        for e, d in self.gb3d.edges():
            gl, _ = self.gb3d.nodes_of_edge(e)
            kn = 1.0 * gl.num_cells
            data = {"normal_diffusivity": kn}
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["transport"], [data]
            )
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        problem = SourceProblem(self.gb3d)
        problem.solve()
        dE = change_in_energy(problem)

        self.assertTrue(np.abs(dE - 10) < 1e-6)

    def disabled_test_src_advective(self):
        delete_data(self.gb3d)
        for g, d in self.gb3d:
            if g.dim == 2:
                d["transport_data"] = InjectionDomain(g, d)
            else:
                d["transport_data"] = MatrixDomain(g, d)
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}
        for e, d in self.gb3d.edges():
            gl, _ = self.gb3d.nodes_of_edge(e)
            kn = 1.0 * gl.num_cells
            data = {"normal_diffusivity": kn}
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["transport"], [data]
            )
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveProblem(self.gb3d, "transport")
        problem.solve()
        dE = change_in_energy(problem)
        self.assertTrue(np.abs(dE - 10) < 1e-6)

    def disabled_test_src_advective_diffusive(self):
        delete_data(self.gb3d)
        for g, d in self.gb3d:
            if g.dim == 2:
                d["transport_data"] = InjectionDomain(g, d)
            else:
                d["transport_data"] = MatrixDomain(g, d)
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}
        for e, d in self.gb3d.edges():
            gl, _ = self.gb3d.nodes_of_edge(e)
            kn = 1.0 * gl.num_cells
            data = {"normal_diffusivity": kn}
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["transport"], [data]
            )
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveDiffusiveProblem(self.gb3d)
        problem.solve()
        dE = change_in_energy(problem)
        self.assertTrue(np.abs(dE - 10) < 1e-6)

    def disabled_test_constant_temp(self):
        delete_data(self.gb3d)
        for g, d in self.gb3d:
            if g.dim == 2:
                d["transport_data"] = InjectionDomain(g, d)
            else:
                d["transport_data"] = MatrixDomain(g, d)
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}
        for e, d in self.gb3d.edges():
            gl, _ = self.gb3d.nodes_of_edge(e)
            kn = 1.0 * gl.num_cells
            data = {"normal_diffusivity": kn}
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["transport"], [data]
            )
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"transport": {}}

        solve_elliptic_problem(self.gb3d)
        problem = SourceAdvectiveDiffusiveDirBound(self.gb3d)
        problem.solve()
        for _, d in self.gb3d:
            T_list = d["transport"]
            const_temp = [np.allclose(T, 10) for T in T_list]
            self.assertTrue(np.all(const_temp))


class SourceProblem(pp.ParabolicModel):
    def __init__(self, g, keyword="transport"):
        pp.ParabolicModel.__init__(self, g, keyword=keyword)

    def space_disc(self):
        return [self.source_disc()]

    def time_step(self):
        return 0.5


class SourceAdvectiveProblem(pp.ParabolicModel):
    def __init__(self, g, keyword="transport"):
        pp.ParabolicModel.__init__(self, g, keyword=keyword)

    def space_disc(self):
        return [self.source_disc(), self.advective_disc()]

    def time_step(self):
        return 0.5


class SourceAdvectiveDiffusiveProblem(pp.ParabolicModel):
    def __init__(self, g, keyword="transport"):
        pp.ParabolicModel.__init__(self, g, keyword=keyword)

    def space_disc(self):
        return [self.source_disc(), self.advective_disc(), self.diffusive_disc()]

    def time_step(self):
        return 0.5


class SourceAdvectiveDiffusiveDirBound(SourceAdvectiveDiffusiveProblem):
    def __init__(self, g, keyword="transport"):
        SourceAdvectiveDiffusiveProblem.__init__(self, g, keyword=keyword)

    def bc(self):
        dir_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        bc_cond = pp.BoundaryCondition(self.grid(), dir_faces, ["dir"] * dir_faces.size)
        return bc_cond

    def bc_val(self, _):
        dir_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        val = np.zeros(self.grid().num_faces)
        val[dir_faces] = 10 * pp.PASCAL
        return val


def source(g, _):
    tol = 1e-4
    value = np.zeros(g.num_cells)
    cell_coord = np.atleast_2d(np.mean(g.cell_centers, axis=1)).T
    cell = np.argmin(np.sum(np.abs(g.cell_centers - cell_coord), axis=0))

    value[cell] = 1.0 * pp.KILOGRAM / pp.SECOND
    return value


class MatrixDomain(pp.ParabolicDataAssigner):
    def __init__(self, g, d, keyword="transport"):
        pp.ParabolicDataAssigner.__init__(self, g, d, keyword)

    def initial_condition(self):
        return 10 * np.ones(self.grid().num_cells)


class InjectionDomain(MatrixDomain):
    def __init__(self, g, d, keyword="transport"):
        MatrixDomain.__init__(self, g, d, keyword)

    def source(self, t):
        flux = source(self.grid(), t)
        T_in = 10
        return flux * T_in

    def aperture(self):
        return 0.1 * np.ones(self.grid().num_cells)


def change_in_energy(problem):
    dE = 0
    problem.split()
    for g, d in problem.grid():
        p = d["transport"]
        p0 = d["transport_data"].initial_condition()
        V = g.cell_volumes * d[pp.keywords.PARAMETERS]["transport"]["aperture"]
        dE += np.sum((p - p0) * V)

    return dE


def solve_elliptic_problem(gb):
    for g, d in gb:
        # Get a default dictinoary
        if g.dim == 2:
            d[pp.keywords.PARAMETERS]["flow"]["source"] = source(g, 0.0)

        dir_bound = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_cond = pp.BoundaryCondition(g, dir_bound, ["dir"] * dir_bound.size)
        d[pp.keywords.PARAMETERS]["flow"]["bc"] = bc_cond
        if pp.keywords.DISCRETIZATION_MATRICES in d:
            if "flow" not in d[pp.keywords.DISCRETIZATION_MATRICES]:
                d[pp.keywords.DISCRETIZATION_MATRICES]["flow"] = {}
        else:
            d[pp.keywords.DISCRETIZATION_MATRICES] = {"flow": {}}

    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        data = {"normal_diffusivity": 1.0 * g_l.num_cells}
        if pp.keywords.PARAMETERS in d:
            d[pp.keywords.PARAMETERS]["flow"] = data
        else:
            d[pp.keywords.PARAMETERS] = pp.Parameters(
                d["mortar_grid"], ["flow"], [data]
            )
        d[pp.keywords.DISCRETIZATION_MATRICES] = {"flow": {}}
    flux = pp.EllipticModel(gb, keyword="flow")
    p = flux.solve()
    flux.split("flux_field")
    pp.fvutils.compute_discharges(gb, p_name="flow", lam_name="flux_field")
    move_discharge(gb, "flow", "transport")


def move_discharge(gb, from_name, to_name):
    for g, d in gb:
        p = d[pp.keywords.PARAMETERS]
        p[to_name]["discharge"] = p[from_name]["discharge"]


def delete_data(gb):
    for _, d in gb:
        d.clear()
    for _, d in gb.edges():
        mortar_grid = d["mortar_grid"]
        d.clear()
        d["mortar_grid"] = mortar_grid
    gb.assign_node_ordering()


if __name__ == "__main__":
    unittest.main()
