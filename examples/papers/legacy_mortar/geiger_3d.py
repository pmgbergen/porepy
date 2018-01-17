
test_case = 2
assert test_case == 1 or test_case == 2


import numpy as np
from porepy.fracs import importer

# geometrical tolerance
tol = 1e-8

# define the mesh size
h = 0.15
grid_kwargs = {}
grid_kwargs['mesh_size'] = {'mode': 'constant', 'value': h, 'bound_value': h, 'tol': tol}

# first line in the file is the domain boundary, the others the fractures
file_dfm = 'geiger_3d.csv'

# import the dfm and generete the grids
gb, domain = importer.dfm_3d_from_csv(file_dfm, tol, **grid_kwargs)
gb.compute_geometry()



# select the permeability depending on the selected test case
if test_case == 1:
    kf = 1e4
else:
    kf = 1e-4
data_problem = {'domain': domain, 'tol': tol, 'aperture': 1e-4, 'km_low': 1e-1, 'km': 1, 'kf': kf}



from geiger_3d_data import DarcyModelData

gb.add_node_props(['is_tangential', 'problem', 'frac_num', 'low_zones'])
for g, d in gb:
    d['problem'] = DarcyModelData(g, d, **data_problem)
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
    d['kn'] = data_problem['kf'] / aperture


# Create the problem and solve it. Export the pressure and projected velocity for visualization.


from porepy.numerics import elliptic

problem = elliptic.DualEllipticModel(gb)
problem.solve()
problem.split()

problem.pressure('pressure')
problem.discharge('discharge')
problem.project_discharge('P0u')
problem.save(['pressure', 'P0u', 'frac_num', 'low_zones'])

