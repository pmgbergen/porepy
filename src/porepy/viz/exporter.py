import sys, os
import numpy as np
import scipy.sparse as sps

import vtk
import vtk.util.numpy_support as ns

from porepy.grids import grid, grid_bucket
from porepy.utils import sort_points

#------------------------------------------------------------------------------#

def export_vtk(g, name, data=None, binary=True, time_step=None, folder=None):
    """ Interface function to export in VTK the grid and additional data.

    In 2d the cells are represented as polygon, while in 3d as polyhedra.
    VTK module need to be installed.
    In 3d the geometry of the mesh needs to be computed.

    To work with python3, the package vtk should be installed in version 7 or
    higher.

    Parameters:
    gb: the grid bucket
    name: the file name without extension.
    data: if g is a single grid then data is a dictionary (see example below)
          if g is a grid bucket then list of names for optional data,
          they are the keys in the grid bucket (see example below).
    binary: export in binary format, default is True.
    time_step: (optional) in a time dependent problem defines the full name of
        the file.
    folder: (optional) the folder to save the file. If the folder does not exist
        it will be created.

    How to use:
    if you need to export a single grid:
    export_vtk(g, "polyhedron", { "cell": np.arange( g.num_cells ) })
    export_vtk(g, "conc", {"conc": conc}, time_step=i)

    if you need to export the grid bucket
    export_vtk(gb, "grid_bucket", ["cell", "pressure"])
    export_vtk(gb, file_name, ["conc"], time_step=i_export, folder="simu")

    """
    if isinstance(g, grid.Grid):
        export_vtk_single(g, make_folder(folder, name), data, binary, time_step)

    if isinstance(g, grid_bucket.GridBucket):
        export_vtk_gb(g, name, data, binary, time_step, folder)

#------------------------------------------------------------------------------#

def export_pvd(g, name, time, folder=None):
    """ Interface function to export in PVD file the time loop informations.
    The user should open only this file in paraview.

    Parameters:
    gb: the grid or grid bucket
    name: the file name without extension. We assume that the VTU associated
        files have the same name.
    time: vector of time.
    folder: (optional) the folder to save the file. We assume that the VTU
        associated files are in the same folder. If the folder does not exist
        it will be created.

    How to use:
    if you need to export a single grid:
    export_pvd(g, "conc", time, folder="simu")

    if you need to export the grid bucket
    export_pvd(gb, file_name, step_to_export*deltaT, folder="simu")

    """
    o_file = open( make_folder(folder, name) + ".pvd",'w')
    b = 'LittleEndian' if sys.byteorder == 'little' else 'BigEndian'
    c = ' compressor="vtkZLibDataCompressor"'
    header = '<?xml version="1.0"?>\n'+ \
             '<VTKFile type="Collection" version="0.1" ' + \
                                              'byte_order="%s"%s>\n' % (b,c) + \
             '<Collection>\n'
    o_file.write(header)
    fm = '\t<DataSet group="" part="" timestep="%f" file="%s"/>\n'

    time_step = np.arange(time.size)

    if isinstance(g, grid.Grid):
        [o_file.write( fm % (time[t], make_file_name(name, t))) \
                                                             for t in time_step]

    if isinstance(g, grid_bucket.GridBucket):
        [o_file.write( fm % (time[t], make_file_name(name, t, d['node_number'])) ) \
                                               for t in time_step for _, d in g]

    o_file.write('</Collection>\n'+'</VTKFile>')
    o_file.close()

#------------------------------------------------------------------------------#
#                            SUPPORT FUNCTIONALITY
#------------------------------------------------------------------------------#

def export_vtk_single(g, name, data, binary, time_step):
    assert isinstance(g, grid.Grid)
    export_vtk_grid(g, make_file_name(name, time_step), data, binary)

#------------------------------------------------------------------------------#

def export_vtk_gb(gb, name, data, binary, time_step, folder):
    assert isinstance(gb, grid_bucket.GridBucket)
    assert isinstance(data, list) or data is None
    gb.assign_node_ordering()

    gb.add_node_prop('file_name')
    gb.add_node_prop('grid_dim')

    data = list() if data is None else data
    data.append('grid_dim')

    for g, d in gb:
        d['file_name'] = make_file_name(name, time_step, d['node_number'])
        d['grid_dim'] = np.tile(g.dim, g.num_cells)

    [export_vtk_grid(g, make_folder(folder, d['file_name']),
                                        gb.node_props_of_keys(g, data), binary)\
                                                                 for g, d in gb]
    export_pvd_gb(make_folder(folder, name)+".pvd", gb)

#------------------------------------------------------------------------------#

def export_pvd_gb(filename, gb):
    o_file = open(filename,'w')
    b = 'LittleEndian' if sys.byteorder == 'little' else 'BigEndian'
    c = ' compressor="vtkZLibDataCompressor"'
    header = '<?xml version="1.0"?>\n'+ \
             '<VTKFile type="Collection" version="0.1" ' + \
                                              'byte_order="%s"%s>\n' % (b,c) + \
             '<Collection>\n'
    o_file.write(header)
    fm = '\t<DataSet group="" part="" file="%s"/>\n'
    [o_file.write( fm % d['file_name'] ) for g, d in gb if g.dim!=0]
    o_file.write('</Collection>\n'+'</VTKFile>')
    o_file.close()

#------------------------------------------------------------------------------#

def export_vtk_grid(g, name, data, binary):
    if g.dim == 0: return
    if g.dim == 1: gVTK = export_vtk_1d( g )
    if g.dim == 2: gVTK = export_vtk_2d( g )
    if g.dim == 3: gVTK = export_vtk_3d( g )
    write_vtk( gVTK, name, data, binary )

#------------------------------------------------------------------------------#

def export_vtk_1d( g ):
    cell_nodes = g.cell_nodes()
    nodes, cells, _  = sps.find( cell_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( g.num_cells ):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
        ptsId = nodes[loc]
        fsVTK = vtk.vtkIdList()
        [ fsVTK.InsertNextId( p ) for p in ptsId ]

        gVTK.InsertNextCell( vtk.VTK_LINE, fsVTK )

    ptsVTK = vtk.vtkPoints()
    if g.nodes.shape[0] == 1:
        [ ptsVTK.InsertNextPoint( node, 0., 0. ) for node in g.nodes.T ]
    if g.nodes.shape[0] == 2:
        [ ptsVTK.InsertNextPoint( node[0], node[1], 0. ) for node in g.nodes.T ]
    else:
        [ ptsVTK.InsertNextPoint( *node ) for node in g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def export_vtk_2d( g ):
    faces_cells, _, _ = sps.find( g.cell_faces )
    nodes_faces, _, _ = sps.find( g.face_nodes )

    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( g.num_cells ):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        ptsId = np.array( [ nodes_faces[ g.face_nodes.indptr[f]: \
                                         g.face_nodes.indptr[f+1] ]
                          for f in faces_cells[loc] ] ).T
        ptsId = sort_points.sort_point_pairs( ptsId )[0,:]

        fsVTK = vtk.vtkIdList()
        [ fsVTK.InsertNextId( p ) for p in ptsId ]

        gVTK.InsertNextCell( vtk.VTK_POLYGON, fsVTK )

    ptsVTK = vtk.vtkPoints()
    if g.nodes.shape[0] == 2:
        [ ptsVTK.InsertNextPoint( node[0], node[1], 0. ) for node in g.nodes.T ]
    else:
        [ ptsVTK.InsertNextPoint( *node ) for node in g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def export_vtk_3d( g ):
    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( g.num_cells ):
        loc_c = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        fs = faces_cells[loc_c]
        fsVTK = vtk.vtkIdList()
        fsVTK.InsertNextId( fs.shape[0] ) # Number faces that make up the cell
        for f in fs:
            loc_f = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
            ptsId = nodes_faces[loc_f]
            mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                                 g.face_centers[:, f], \
                                                 g.face_normals[:, f] )
            fsVTK.InsertNextId( ptsId.shape[0] ) # Number of points in face
            [ fsVTK.InsertNextId( p ) for p in ptsId[mask] ]

        gVTK.InsertNextCell( vtk.VTK_POLYHEDRON, fsVTK )

    ptsVTK = vtk.vtkPoints()
    [ ptsVTK.InsertNextPoint( *node ) for node in g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def write_vtk( gVTK, name, data = None, binary = True ):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData( gVTK )
    writer.SetFileName( name )

    if data is not None:
        for name, data in data.items():
            dataVTK = ns.numpy_to_vtk( data.ravel(order='F'), deep = True, \
                                       array_type = vtk.VTK_DOUBLE )
            dataVTK.SetName( str( name ) )
            dataVTK.SetNumberOfComponents( 1 if data.ndim == 1 else 3 )
            gVTK.GetCellData().AddArray( dataVTK )

    if not binary: writer.SetDataModeToAscii()
    writer.Update()

#------------------------------------------------------------------------------#

def make_folder(folder, name=None):
    if folder is None: return name

    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, name)

#------------------------------------------------------------------------------#

def make_file_name(name, time_step=None, node_number=None):

    extension = ".vtu"
    padding = 6
    if node_number is None: # normal grid
        if time_step is None:
            return name + extension
        else:
            time = str(time_step).zfill(padding)
            return name + "_" + time + extension
    else: # part of a grid bucket
        grid = str(node_number).zfill(padding)
        if time_step is None:
            return name + "_" + grid + extension
        else:
            time = str(time_step).zfill(padding)
            return name + "_" + grid + "_" + time + extension

#------------------------------------------------------------------------------#
