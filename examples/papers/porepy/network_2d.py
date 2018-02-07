from porepy.viz import exporter
from porepy.fracs import importer

from porepy.grids import coarsening as co

mesh_kwargs = {}
mesh_kwargs = {'mesh_mode': 'weighted', 'h_ideal': 500, 'h_min': 100}

name = 'network_2d_second'
gb = importer.from_csv(name + '.csv', mesh_kwargs)
gb.compute_geometry()
co.coarsen(gb, 'by_volume')
gb.assign_node_ordering()

exporter.export_vtk(gb, name + '_coarse', folder=name)
