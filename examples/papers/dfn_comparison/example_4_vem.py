from porepy.viz.exporter import Exporter
import scipy.sparse as sps

import example_4_create_grid

#------------------------------------------------------------------------------#

folder_export = 'viz/'
file_export = 'vem'

# consider gmsh 2.11
conforming = True
gb = example_4_create_grid.create(conforming)

save = Exporter(gb, file_export, folder_export)
save.write_vtk()

