import numpy as np
import scipy.sparse as sps

import vtk
import vtk.util.numpy_support as ns

from compgeom import sort_points

#------------------------------------------------------------------------------#

def export( _g, _name, _data = None, _binary = True ):
    """ export in VTK the grid and additional data.

    In 2d the cells are represented as polygon, while in 3d as polyhedra.
    VTK module need to be installed.
    In 3d the geometry of the mesh needs to be computed.

    Parameters:
    _g: the grid
    _name: the file name with extension ".vtu".
    _data: optional data, it is a dictionary: key the name of the field.
    _binary: export in binary format, default is True.

    How to use:
    export( g, "polyhedron.vtu", { "cell": np.arange( g.num_cells ) } )

    """

    if _g.dim == 1: raise NotImplementedError
    if _g.dim == 2: gVTK = export_2d( _g )
    if _g.dim == 3: gVTK = export_3d( _g )
    writeVTK( gVTK, _name, _data, _binary )

#------------------------------------------------------------------------------#

def export_2d( _g ):

    faces_cells, cells, _ = sps.find( _g.cell_faces )
    nodes_faces, faces, _ = sps.find( _g.face_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( _g.num_cells ):
        fs = faces_cells[ cells == c ]
        ptsId = np.array( [ nodes_faces[ faces == f ] for f in fs ] ).T
        ptsId = sort_points.sort_point_pairs( ptsId )[0,:]

        fsVTK = vtk.vtkIdList()
        [ fsVTK.InsertNextId( p ) for p in ptsId ]

        gVTK.InsertNextCell( vtk.VTK_POLYGON, fsVTK )

    ptsVTK = vtk.vtkPoints()
    if _g.nodes.shape[0] == 2:
        [ ptsVTK.InsertNextPoint( *node, 0. ) for node in _g.nodes.T ]
    else:
        [ ptsVTK.InsertNextPoint( *node ) for node in _g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def export_3d( _g ):

    faces_cells, cells, _ = sps.find( _g.cell_faces )
    nodes_faces, faces, _ = sps.find( _g.face_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( _g.num_cells ):
        fs = faces_cells[ cells == c ]
        fsVTK = vtk.vtkIdList()
        fsVTK.InsertNextId( fs.shape[0] ) # Number faces that make up the cell
        for f in fs:
            ptsId = nodes_faces[ faces == f ]
            mask = sort_points.sort_point_face( _g.nodes[:, ptsId], _g.face_centers[:, f] )
            fsVTK.InsertNextId( ptsId.shape[0] ) # Number of points in face
            [ fsVTK.InsertNextId( p ) for p in ptsId[mask] ]

        gVTK.InsertNextCell( vtk.VTK_POLYHEDRON, fsVTK )

    ptsVTK = vtk.vtkPoints()
    [ ptsVTK.InsertNextPoint( *node ) for node in _g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def writeVTK( _gVTK, _name, _data = None, _binary = True ):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData( _gVTK )
    writer.SetFileName( _name )

    if _data is not None:
        for name, data in _data.items():
            dataVTK = ns.numpy_to_vtk( data, deep = True, array_type = vtk.VTK_DOUBLE )
            dataVTK.SetName( str( name ) )
            _gVTK.GetCellData().AddArray( dataVTK )

    if not _binary: writer.SetDataModeToAscii()
    writer.Update()

#------------------------------------------------------------------------------#
