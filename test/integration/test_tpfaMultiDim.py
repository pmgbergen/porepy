import numpy as np

from porepy.numerics.fv.tpfa import TpfaMultiDim
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import tensor, bc


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
                            'value': mesh_size, 'bound_value': 2*mesh_size}
        domain = {'xmin': 0, 'ymin': 0, 'xmax':1, 'ymax':1}
        gb = meshing.simplex_grid(fracs, domain,**mesh_kwargs)
        
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

def check_pressures(gb):
    """
    Check that the pressures are not too far from an approximate 
    analytical solution. Note that the solution depends
    on the grid quality. Also sensitive to the way in which
    the tpfa half transmissibilities are computed. 
    """
    for g, d in gb:
        pressure = d['pressure']
        pressure_analytic = g.cell_centers[1]
        p_diff = pressure - pressure_analytic
        assert np.max(np.abs(p_diff)) < 0.033

def test_uniform_flow_cart_2d_1d_cartesian():
    # Structured Cartesian grid
    gb = setup_2d_1d(np.array([10, 10]))

    # Python inverter is most efficient for small problems
    flux_discr = TpfaMultiDim('flow')
    A, rhs = flux_discr.matrix_rhs(gb)
    p = np.linalg.solve(A.A, rhs)

    flux_discr.solver.split(gb, 'pressure', p)

    check_pressures(gb)

def test_uniform_flow_cart_2d_1d_simplex():
    # Unstructured simplex grid
    gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)

    # Python inverter is most efficient for small problems
    flux_discr = TpfaMultiDim('flow')
    A, rhs = flux_discr.matrix_rhs(gb)
    p = np.linalg.solve(A.A, rhs)

    flux_discr.solver.split(gb, 'pressure', p)

    check_pressures(gb)
