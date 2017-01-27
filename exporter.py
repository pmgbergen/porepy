import numpy as np
import scipy.sparse as sps

import vtk
import vtk.util.numpy_support as ns

from compgeom import sort_points

#------------------------------------------------------------------------------#

def export_vtk( g, name, data = None, binary = True ):
    """ export in VTK the grid and additional data.

    In 2d the cells are represented as polygon, while in 3d as polyhedra.
    VTK module need to be installed.
    In 3d the geometry of the mesh needs to be computed.

    To work with python3, the package vtk should be installed in version 7 or
    higher.

    Parameters:
    g: the grid
    name: the file name with extension ".vtu".
    data: optional data, it is a dictionary: key the name of the field.
    binary: export in binary format, default is True.

    How to use:
    export_vtk( g, "polyhedron.vtu", { "cell": np.arange( g.num_cells ) } )

    """

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
        [ ptsVTK.InsertNextPoint( *node, 0. ) for node in g.nodes.T ]
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
        [ ptsVTK.InsertNextPoint( *node, 0. ) for node in g.nodes.T ]
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
