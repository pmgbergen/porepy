import numpy as np
import single_data
from porepy.numerics import elliptic, parabolic

tol = 1e-8

gb, domain = single_data.create_grid(True)

# select the permeability depending on the selected test case
data_problem = {'domain': domain, 'tol': tol,
                'aperture': 1e-2,
                'km_low': 1e-13, 'km_high': 1e-12, 'kf': 1e-10,
                'phi_low': 0.2, 'phi_high': 0.25, 'phi_f': 0.4}

gb.add_node_props(['is_tangential', 'problem', 'frac_num', 'low_zones'])
for g, d in gb:
    d['problem'] = single_data.DarcyModelData(g, d, **data_problem)
    d['is_tangential'] = True
    d['low_zones'] = d['problem'].low_zones()

    if g.dim == 2:
        d['frac_num'] = g.frac_num*np.ones(g.num_cells)
    else:
        d['frac_num'] = -1*np.ones(g.num_cells)

# Assign coupling permeability, the aperture is read from the lower dimensional grid
gb.add_edge_prop('kn')
for e, d in gb.edges_props():
    g_l = gb.sorted_nodes_of_edge(e)[0]
    aperture = gb.node_prop(g_l, 'param').get_aperture()
    d['kn'] = data_problem['kf'] / aperture / 1e-3 # viscosity

problem = elliptic.DualEllipticModel(gb)
problem.solve()
problem.split()

problem.pressure('pressure_head')
problem.discharge('discharge')
problem.project_discharge('P0u')
problem.save(['pressure_head', 'P0u', 'frac_num', 'low_zones'])

physics = "transport"
for g, d in gb:
    d[physics+'_data'] = single_data.AdvectiveDataAssigner(g, d, **data_problem)

T = 1e8
advective = single_data.AdvectiveProblem(gb, physics, time_step=T/2., end_time=T)
advective.solve("transport")

