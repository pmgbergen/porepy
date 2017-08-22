from porepy.viz import exporter
from porepy.fracs import importer

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'weighted',
                            'value': 500,
                            'bound_value': 500}

gb = importer.from_csv('network_2d.csv', mesh_kwargs)
gb.compute_geometry()
gb.assign_node_ordering()

exporter.export_vtk(gb, 'network_2d', folder='network_2d')
