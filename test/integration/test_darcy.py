import numpy as np
import unittest

import porepy as pp

class BasicsTest(unittest.TestCase):

    #------------------------------------------------------------------------------#

    def test_mono_equals_multi(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        g = pp.CartGrid([10, 10])
        g.compute_geometry()
        gb = pp.meshing.cart_grid([], [10, 10])
        param_g = pp.Parameters(g)

        def bc_val(g):
            left = g.face_centers[0] < 1e-6
            right = g.face_centers[0] > 10 - 1e-6

            bc_val = np.zeros(g.num_faces)
            bc_val[left] = -1
            bc_val[right] = 1
            return bc_val

        def bc_labels(g):
            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]
            left = bound_face_centers[0] < 1e-6
            right = bound_face_centers[0] > 10 - 1e-6

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(right, left)] = 'dir'
            bc_labels = pp.BoundaryCondition(g, bound_faces, labels)

            return bc_labels

        param_g.set_bc_val('flow', bc_val(g))
        param_g.set_bc('flow', bc_labels(g))

        gb.add_node_props(['param'])
        for sub_g, d in gb:
            d['param'] = pp.Parameters(sub_g)
            d['param'].set_bc_val('flow', bc_val(g))
            d['param'].set_bc('flow', bc_labels(sub_g))

        for e, d in gb.edges():
            gl, _ = gb.nodes_of_edge(e)
            d_l = gb.node_props(gl)
            d['kn'] = 1. / np.mean(d_l['param'].get_aperture())
        

        problem_mono = pp.EllipticModel(g, {'param': param_g})
        problem_mult = pp.EllipticModel(gb)

        p_mono = problem_mono.solve()
        p_mult = problem_mult.solve()

        assert np.allclose(p_mono, p_mult)

#------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_cart(self):
        gb = setup_2d_1d([10, 10])
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split('pressure')

        for g, d in gb:
            pressure = d['pressure']
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 2e-2
#------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_simplex(self):
        """
        Unstructured simplex grid. Note that the solution depends
        on the grid quality. Also sensitive to the way in which
        the tpfa half transmissibilities are computed.
        """
        gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split('pressure')

        for g, d in gb:
            pressure = d['pressure']
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.033

    def test_elliptic_dirich_neumann_source_sink_cart(self):
        gb = setup_3d(np.array([4, 4, 4]), simplex_grid=False)
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split('pressure')

        for g, d in gb:
            if g.dim == 3:
                p_ref = elliptic_dirich_neumann_source_sink_cart_ref_3d()
                assert np.allclose(d['pressure'], p_ref)
            if g.dim == 0:
                p_ref = [-10788.06883149]
                assert np.allclose(d['pressure'], p_ref)
        return gb

    if __name__ == '__main__':
        unittest.main()


def setup_3d(nx, simplex_grid=False):
    f1 = np.array(
        [[0.2, 0.2, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5]])
    f2 = np.array(
        [[0.2, 0.8, 0.8, 0.2],  [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    f3 = np.array(
        [[0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
    fracs = [f1, f2, f3]
    if not simplex_grid:
        gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1, 1])
    else:
        mesh_size = .3
        mesh_kwargs = {'mesh_size_frac': mesh_size,
                       'mesh_size_bound': 2 * mesh_size,
                       'mesh_size_min': mesh_size / 20}
        domain = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}
        gb = pp.meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.add_node_props(['param'])
    for g, d in gb:
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = pp.Parameters(g)
        param.set_aperture(a)

        # BoundaryCondition
        left = g.face_centers[0] < 1e-6
        top = g.face_centers[2] > 1 - 1e-6
        dir_faces = np.argwhere(left)
        bc_cond = pp.BoundaryCondition(g, dir_faces, ['dir'] * dir_faces.size)
        bc_val = np.zeros(g.num_faces)
        bc_val[dir_faces] = 3
        bc_val[top] = 2.4
        param.set_bc('flow', bc_cond)
        param.set_bc_val('flow', bc_val)

        # Source and sink
        src = np.zeros(g.num_cells)
        src[0] = np.pi
        src[-1] = -np.pi
        param.set_source('flow', src)
        d['param'] = param

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        d_l = gb.node_props(gl)
        d['kn'] = 1. / np.mean(d_l['param'].get_aperture())

        
    return gb


def setup_2d_1d(nx, simplex_grid=False):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    if not simplex_grid:
        gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1])
    else:
        mesh_kwargs = {}
        mesh_size = .08
        mesh_kwargs = {'mesh_size_frac': mesh_size,
                       'mesh_size_bound': 2 * mesh_size,
                       'mesh_size_min': mesh_size / 20}
        domain = {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}
        gb = pp.meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()
    gb.add_node_props(['param'])
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx)
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = pp.Parameters(g)
        param.set_tensor('flow', perm)
        param.set_aperture(a)
        if g.dim == 2:
            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            bound = pp.BoundaryCondition(g, bound_faces.ravel('F'),
                                         ['dir'] * bound_faces.size)
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            param.set_bc('flow', bound)
            param.set_bc_val('flow', bc_val)
        d['param'] = param

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        d_l = gb.node_props(gl)
        d['kn'] = 1. / np.mean(d_l['param'].get_aperture())
        

    return gb


def elliptic_dirich_neumann_source_sink_cart_ref_3d():
    p_ref = np.array([0.54570555, -11.33848749, -19.44484907, -23.13293673,
                      -2.03828237, -12.73249228, -20.96189563, -24.14626244,
                      -2.81412045, -14.03104316, -22.2699576, -25.06029852,
                      -2.82350853, -13.6157183, -21.47553858, -24.92564315,
                      -2.46107297, -13.72231418, -22.34607642, -25.80769869,
                      -3.22878706, -15.29275276, -26.21591677, -27.42991885,
                      -3.99188865, -18.72292714, -29.82101211, -28.89933092,
                      -3.68770392, -16.13278292, -25.09083527, -28.24109233,
                      -4.36104216, -17.17318124, -26.53960339, -30.32186276,
                      -4.81751268, -18.63731792, -30.45264082, -32.08038545,
                      -5.489682, -22.06836116, -34.23287575, -33.94433277,
                      -5.17804343, -19.54672997, -29.78375042, -34.04856,
                      -7.71448607, -22.60562853, -32.40425572, -36.8597635,
                      -8.00575965, -23.53059111, -34.01202746, -38.25317203,
                      -8.37196805, -24.79222197, -35.8194776, -40.46051172,
                      -8.34414468, -24.57071193, -35.99975111, -44.22506448])
    return p_ref

if __name__ == '__main__':
    unittest.main()
