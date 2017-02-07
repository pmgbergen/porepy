import numpy as np
from scipy import sparse as sps
from utils.half_space import half_space_int

import numpy as np
from scipy import sparse as sps
from utils.half_space import half_space_int

def split_fracture(g, frac_tag,tip_nodes_id=None,offset=0):
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
    tip_nodes_id - ndarray
        Defaults to None. If None, it tries to locate the fracture tips
        based on the number of tagged faces connecting each node. Fracture
        tips are then tagged as nodes only connected to one tagged face.
        If None, tip_nodes_id should be the indices of the tip nodes of 
        the fractures. The nodes in the tip_nodes_id will not be split.
    offset - float
        Defaults to 0. The fracture nodes are moved a distance 0.5*offest
        to each side of the fractures. WARNING: this is for visualization
        purposes only. E.g, the face centers are not moved.
    Returns
    -------
    g - A valid grid deformation where with internal boundaries. 


    Examples
    >>> import numpy as np
    >>> from core.grids import structured
    >>> from viz import plot_grid
    >>> import matplotlib.pyplot as plt
    >>> from gridding.fractured.split_grid import split_fracture
    >>> # Set up a Cartesian grid
    >>> n = 10
    >>> g = structured.CartGrid([n, n])
    >>> g.compute_geometry()
    >>> # Define fracture
    >>> frac_tag = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2, 
    >>>            g.face_centers[0,:]>n/4), g.face_centers[0,:]<3*n/4)
    >>> g = split_fracture(g, frac_tag)
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
    assert all(node_mult <= 2), "Not support for intersecting fractures"
    if tip_nodes_id == None:
        tip_nodes_id   = np.ravel(np.argwhere(node_mult==1)) # tip nodes
        int_nodes      = node_mult==2                        # internal nodes
    else:
        int_nodes               = node_mult>0 # all internal nodes
        int_nodes[tip_nodes_id] = False       # Remove given tip nodes
    int_nodes_id = np.ravel(np.argwhere(int_nodes))
    
    # faces connected to fracture nodes
    face_fracnodes             = g.face_nodes[int_nodes,:]
    face_fracnodes[:,frac_tag] = False #don't consider tagged faces
    face_fracnodes_id          = np.argwhere(face_fracnodes)

    # group faces into left side and right side of fracture
    n    = g.face_normals[:,frac_tag_id[0]]
    x0   = g.face_centers[:,frac_tag_id[0]]
    left = half_space_int(n,x0,g.face_centers[:,face_fracnodes_id[:,1]])

    # add duplicate nodes at the end of array
    node_coord            = g.nodes[:,int_nodes]
    node_coord           -= .5*offset*n
    g.nodes[:,int_nodes] += .5*offset*n 
    g.nodes      = np.concatenate((g.nodes,node_coord),axis=1)
    g.num_nodes += node_coord.shape[1]

    # faces on left side of fracture should not be connected to
    # nodes on right side of fracture. The right nodes keeps the
    # other face-mappings.
    leftface_nodes_id      = face_fracnodes_id[left,:]
    leftface_nodes_id[:,0] = int_nodes_id[leftface_nodes_id[:,0]]
    g.face_nodes[leftface_nodes_id[:,0],leftface_nodes_id[:,1]] = False

    # Connect new nodes to left side of fracture
    col  = face_fracnodes_id[left,1]
    row  = face_fracnodes_id[left,0]
    face_fracnodes_left = sps.csc_matrix(([True]*row.size,(row,col)),
                                         (sum(int_nodes),g.face_nodes.shape[1]))
    g.face_nodes = sps.vstack((g.face_nodes, face_fracnodes_left),format='csc')

    # create new faces along fracture and connect them to left
    # nodes
    frac_int_nodes_id = np.argwhere(frac_nodes[int_nodes,:])
    frac_tip_nodes_id = np.argwhere(frac_nodes[tip_nodes_id,:])
    row  = frac_int_nodes_id[:,0] + g.num_nodes - node_coord.shape[1]
    row  = np.append(row, tip_nodes_id[frac_tip_nodes_id[:,0]])
    col  = frac_int_nodes_id[:,1]
    col  = np.append(col, frac_tip_nodes_id[:,1])
    face_fracnodes_right = sps.csc_matrix(([True]*row.size,(row,col)),
                                          (g.num_nodes,frac_tag_id.size))
    g.face_nodes = sps.hstack((g.face_nodes,face_fracnodes_right),format='csc')

    # update face info
    g.num_faces   +=frac_tag_id.size
    g.face_normals = np.hstack((g.face_normals, g.face_normals[:,frac_tag]))
    g.face_areas   = np.append(g.face_areas, g.face_areas[frac_tag])
    g.face_centers = np.hstack((g.face_centers, g.face_centers[:,frac_tag]))

    # Set new cell conectivity
    # Cells on right side does not change. We first add the new left-faces
    # to the left cells
    cell_frac_id = np.argwhere(cell_frac)
    left_cell    = half_space_int(n,x0,g.cell_centers[:,cell_frac_id[:,1]])

    col            = cell_frac_id[left_cell,1]
    row            = cell_frac_id[left_cell,0]
    data           = np.ravel(g.cell_faces[np.ravel(frac_tag_id[row]), col])
    cell_frac_left = sps.csc_matrix((data,(row,col)),(frac_tag_id.size,g.cell_faces.shape[1]))
    # We remove the right faces of the left cells.
    col  = cell_frac_id[~left_cell,1]
    row  = cell_frac_id[~left_cell,0]
    data = np.ravel(g.cell_faces[np.ravel(frac_tag_id[row]), col])
    cell_frac_right = sps.csc_matrix((data,(row,col)),
                                     (frac_tag_id.size,g.cell_faces.shape[1]))
    g.cell_faces[frac_tag,:] = cell_frac_right
    g.cell_faces    = sps.vstack((g.cell_faces,cell_frac_left),format='csc')

    return g


def split_grid(g,f,offset=0):
    for i in range(f.num):
        g = split_fracture(g,f.tag[i,:],tip_nodes_id=f.tips[i,:],offset=offset)
        if i<f.num:
            new_faces = np.zeros((f.tag.shape[0],sum(f.tag[i,:])),dtype='bool')
            f.tag=np.hstack((f.tag, new_faces))
    return g


class Fracture:
    def __init__(self,g):
        self.num = 0
        self.tag = np.zeros((0,g.num_faces),dtype='bool')
        self.tips = np.zeros((0,2),dtype=np.int8)
    
    def add_tag(self,tag):
        self.tag = np.vstack((self.tag,tag))
        self.num+=1

    def set_tips(self, g):
        self.tips = np.zeros((self.num,2),dtype=np.int8)
        for i in range(self.num):
            node_mult = np.ravel(np.sum(g.face_nodes[:,self.tag[i,:]],axis=1))
            self.tips[i,:] = np.ravel(np.argwhere(node_mult==1)) # tip nodes
