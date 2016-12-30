import numpy as np
import scipy.sparse as sps

import vtk
import vtk.util.numpy_support as ns

from compgeom import sort_points

#------------------------------------------------------------------------------#

def export( g, name, data = None, binary = True ):
    """ export in VTK the grid and additional data.

    In 2d the cells are represented as polygon, while in 3d as polyhedra.
    VTK module need to be installed.
    In 3d the geometry of the mesh needs to be computed.

    Parameters:
    g: the grid
    name: the file name with extension ".vtu".
    data: optional data, it is a dictionary: key the name of the field.
    binary: export in binary format, default is True.

    How to use:
    export( g, "polyhedron.vtu", { "cell": np.arange( g.num_cells ) } )

    """

    if g.dim == 1: raise NotImplementedError
    if g.dim == 2: gVTK = export_2d( g )
    if g.dim == 3: gVTK = export_3d( g )
    writeVTK( gVTK, name, data, binary )

#------------------------------------------------------------------------------#

def export_2d( g ):

    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( g.num_cells ):
        fs = faces_cells[ cells == c ]
        ptsId = np.array( [ nodes_faces[ faces == f ] for f in fs ] ).T
        print( ptsId )
        ptsId = sort_points.sort_point_pairs( ptsId )[0,:]

        fsVTK = vtk.vtkIdList()
        [ fsVTK.InsertNextId( p ) for p in ptsId ]

        gVTK.InsertNextCell( vtk.VTK_POLYGON, fsVTK )

    ptsVTK = vtk.vtkPoints()
    if g.nodes.shape[0] == 2:
        [ ptsVTK.InsertNextPoint( *node, 0. ) for node in g.nodes.T ]
    else:
        [ ptsVTK.InsertNextPoint( *node ) for node in g.nodes.T ]
    gVTK.SetPoints( ptsVTK )

    return gVTK

#------------------------------------------------------------------------------#

def export_3d( g ):

    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )
    gVTK = vtk.vtkUnstructuredGrid()

    for c in np.arange( g.num_cells ):
        fs = faces_cells[ cells == c ]
        fsVTK = vtk.vtkIdList()
        fsVTK.InsertNextId( fs.shape[0] ) # Number faces that make up the cell
        for f in fs:
            ptsId = nodes_faces[ faces == f ]
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

def writeVTK( gVTK, name, data = None, binary = True ):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData( gVTK )
    writer.SetFileName( name )

    if data is not None:
        for name, data in data.items():
            dataVTK = ns.numpy_to_vtk( data, deep = True, array_type = vtk.VTK_DOUBLE )
            dataVTK.SetName( str( name ) )
            gVTK.GetCellData().AddArray( dataVTK )

    if not binary: writer.SetDataModeToAscii()
    writer.Update()

#------------------------------------------------------------------------------#
