import numpy as np

from porepy.numerics.fv.mpfa import MpfaMixedDim
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import tensor, bc


def setup_cart_2d(nx):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    gb = meshing.cart_grid(fracs, nx, physdims=[1, 1])
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


def test_uniform_flow_cart_2d():
    # Structured Cartesian grid
    gb = setup_cart_2d(np.array([10, 10]))

    # Python inverter is most efficient for small problems
    flux_discr = MpfaMixedDim('flow')
    A, rhs = flux_discr.matrix_rhs(gb)
    p = np.linalg.solve(A.A, rhs)

    flux_discr.split(gb, 'pressure', p)
    for g, d in gb:
        pressure = d['pressure']
        pressure_analytic = g.cell_centers[1]
        p_diff = pressure - pressure_analytic
        assert np.max(np.abs(p_diff)) < 0.05
