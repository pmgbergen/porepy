import numpy as np
from scipy import sparse as sps
from utils.half_space import half_space_int

def split_grid(g, frac_tag):
    """
    Split faces and nodes to create an internal boundary
    
    The tagged faces are split in two along with connected nodes (except
    tips).

    To be added:
    3D and intersection of fractures

    Parameters
    ----------
    g - A valid grid

    frac_tag - ndarray
        A boolean array that is true for face indices that will be split

    Returns
    -------
    g - A valid grid deformation where with internal boundaries. 


    Examples
    >>> import numpy as np
    >>> from core.grids import structured
    >>> from viz import plot_grid
    >>> import matplotlib.pyplot as plt
    >>> from gridding.fractured.split_grid import split_grid 
    >>> # Set up a Cartesian grid
    >>> n = 10
    >>> g = structured.CartGrid([n, n])
    >>> g.compute_geometry()
    >>> # Define fracture
    >>> frac_tag = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2, 
    >>>            g.face_centers[0,:]>n/4), g.face_centers[0,:]<3*n/4)
    >>> g = split_grid(g, frac_tag)
    >>> frac_nodes = np.arange((np.sum(frac_tag)-1))+g.num_nodes -(np.sum(frac_tag)-1)
    >>> g.nodes[1,frac_nodes] -=.5
    >>> plot_grid.plot_grid(g)
    >>> plt.show()
    """

    # Create convenient mappings
    frac_tag_id    = np.argwhere(frac_tag)
    frac_nodes     = g.face_nodes[:,frac_tag]
    cell_frac      = g.cell_faces[frac_tag,:]
    node_mult      = np.ravel(np.sum(g.face_nodes[:,frac_tag],axis=1))
    assert all(node_mult <= 2), "Not yet support for intersecting fractures"
    tip_nodes      = node_mult==1 # tip nodes
    int_nodes      = node_mult==2 # internal nodes
    node_coord     = g.nodes[:,int_nodes]

    # add duplicate nodes at the end of array
    g.nodes = np.concatenate((g.nodes,node_coord),axis=1)
    g.num_nodes += node_coord.shape[1]
    
    # faces connected to fracture nodes
    face_fracnodes = g.face_nodes[int_nodes,:]
    face_fracnodes[:,frac_tag] = False #don't consider tagged faces
    face_fracnodes_id = np.argwhere(face_fracnodes)

    # group faces into left side and right side of fracture
    n  = g.face_normals[:,frac_tag_id[0]]
    x0 = g.face_centers[:,frac_tag_id[0]]
    left = half_space_int(n,x0,g.face_centers[:,face_fracnodes_id[:,1]])

    # faces on right side should not be connected to remove connection to left side of fracture
    col  = face_fracnodes_id[~left,1]
    row  = face_fracnodes_id[~left,0]
    face_fracnodes_right = sps.csc_matrix(([True]*row.size,(row,col)),(sum(int_nodes),g.face_nodes.shape[1]))
    g.face_nodes[int_nodes,:] = face_fracnodes_right

    # Connect new nodes to left side of fracture
    col  = face_fracnodes_id[left,1]
    row  = face_fracnodes_id[left,0]
    face_fracnodes_left = sps.csc_matrix(([True]*row.size,(row,col)),(sum(int_nodes),g.face_nodes.shape[1]))
    g.face_nodes = sps.vstack((g.face_nodes, face_fracnodes_left))

    # create new faces along fracture
    tip_nodes_id = np.argwhere(tip_nodes)
    frac_int_nodes_id = np.argwhere(frac_nodes[int_nodes,:])
    frac_tip_nodes_id = np.argwhere(frac_nodes[tip_nodes,:])
    row  = frac_int_nodes_id[:,0] + g.num_nodes - node_coord.shape[1]
    row  = np.append(row, tip_nodes_id[frac_tip_nodes_id[:,0]])
    col  = frac_int_nodes_id[:,1]
    col  = np.append(col, frac_tip_nodes_id[:,1])
    face_fracnodes_right = sps.csc_matrix(([True]*row.size,(row,col)),(g.num_nodes,frac_tag_id.size))
    g.face_nodes = sps.hstack((g.face_nodes,face_fracnodes_right))
    g.num_faces+=frac_tag_id.size

    # update face info
    g.face_normals = np.hstack((g.face_normals, g.face_normals[:,frac_tag]))
    g.face_areas = np.append(g.face_areas, g.face_areas[frac_tag])
    g.face_centers = np.hstack((g.face_centers, g.face_centers[:,frac_tag]))

    # Set new cell conectivity
    cell_frac_id = np.argwhere(cell_frac)
    left_cell = half_space_int(n,x0,g.cell_centers[:,cell_frac_id[:,1]])

    col  = cell_frac_id[left_cell,1]
    row  = cell_frac_id[left_cell,0]
    data = np.ravel(g.cell_faces[np.ravel(frac_tag_id[row]), col])
    cell_frac_left = sps.csc_matrix((data,(row,col)),(frac_tag_id.size,g.cell_faces.shape[1]))

    col  = cell_frac_id[~left_cell,1]
    row  = cell_frac_id[~left_cell,0]
    data = np.ravel(g.cell_faces[np.ravel(frac_tag_id[row]), col])
    cell_frac_right = sps.csc_matrix((data,(row,col)),(frac_tag_id.size,g.cell_faces.shape[1]))
    g.cell_faces[frac_tag,:] = cell_frac_right

    g.cell_faces = sps.vstack((g.cell_faces,cell_frac_left))

    return g
