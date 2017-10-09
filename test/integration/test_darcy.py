import numpy as np
import unittest

from porepy.numerics import darcyEq
from porepy.grids.structured import CartGrid
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import tensor, bc


class BasicsTest(unittest.TestCase):

    #------------------------------------------------------------------------------#

    def test_mono_equals_multi(self):
        """
        test that the mono_dimensional darcy solver gives the same answer as
        the grid bucket darcy
        """
        g = CartGrid([10, 10])
        g.compute_geometry()
        gb = meshing.cart_grid([], [10, 10])
        param_g = Parameters(g)

        def bc_val(g):
            left = g.face_centers[0] < 1e-6
            right = g.face_centers[0] > 10 - 1e-6

            bc_val = np.zeros(g.num_faces)
            bc_val[left] = -1
            bc_val[right] = 1
            return bc_val

        param_g.set_bc_val('flow', bc_val(g))

        gb.add_node_props(['param'])
        for sub_g, d in gb:
            d['param'] = Parameters(sub_g)
            d['param'].set_bc_val('flow', bc_val(g))

        problem_mono = darcyEq.Darcy(g, {'param': param_g})
        problem_mult = darcyEq.Darcy(gb)

        p_mono = problem_mono.solve()
        p_mult = problem_mult.solve()

        assert np.allclose(p_mono, p_mult)
#------------------------------------------------------------------------------#

    def test_darcy_uniform_flow_cart(self):
        gb = setup_2d_1d([10, 10])
        problem = darcyEq.Darcy(gb)
        p = problem.solve()
        problem.split('pressure')

        for g, d in gb:
            pressure = d['pressure']
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.03
#------------------------------------------------------------------------------#

    def test_darcy_uniform_flow_simplex(self):
        # Unstructured simplex grid
        gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)
        problem = darcyEq.Darcy(gb)
        p = problem.solve()
        problem.split('pressure')

        for g, d in gb:
            pressure = d['pressure']
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.03


def setup_2d_1d(nx, simplex_grid=False):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    if not simplex_grid:
        gb = meshing.cart_grid(fracs, nx, physdims=[1, 1])
    else:
        mesh_kwargs = {}
        mesh_size = .3
        mesh_kwargs['mesh_size'] = {'mode': 'constant',
                                    'value': mesh_size, 'bound_value': 2 * mesh_size}
        domain = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}
        gb = meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()
    gb.add_node_props(['param'])
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(gb.dim_max(), kxx)
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = Parameters(g)
        param.set_tensor('flow', perm)
        param.set_aperture(a)
        if g.dim == 2:
            bound_faces = g.get_boundary_faces()
            bound = bc.BoundaryCondition(g, bound_faces.ravel('F'),
                                         ['dir'] * bound_faces.size)
            bc_val = g.face_centers[1]
            param.set_bc('flow', bound)
            param.set_bc_val('flow', bc_val)
        d['param'] = param

    return gb
