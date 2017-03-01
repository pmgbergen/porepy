import numpy as np
from scipy import sparse as sps
from utils.half_space import half_space_int
from core.grids.grid import Grid, FaceTag
from utils.graph import Graph



class Fracture:
    """ 
    Fracture class. Contains the information about the fractures
    that is needed to split the grid. This class is so far created
    as a convenience during the implementation of the splitting, 
    but this is probably not the best way to do this.
    """
    def __init__(self,g):
        self.num  = 0
        self.tag  = np.zeros((0,g.num_faces),dtype='bool')
        self.tips = np.zeros((0,2),dtype=np.int8)
        self.n    = np.zeros((3,0))
        self.x0   = np.zeros((3,0))

    def add_tag(self,g,tag):
        self.tag = np.vstack((self.tag,tag))
        self.n   = np.hstack((self.n, g.face_normals[:,tag][:,0,None]))
        self.x0  = np.hstack((self.x0,g.face_centers[:,tag][:,0,None]))
        self.num+=1
        self.set_tips(g)
        
    def set_tips(self, g):
        self.tips = np.zeros((self.num,2),dtype=np.int8)
        for i in range(self.num):
            node_mult = np.ravel(np.sum(g.face_nodes[:,self.tag[i,:]],axis=1))
            self.tips[i,:] = np.ravel(np.argwhere(node_mult==1)) # tip

def create_fracture_grid(g,f):
    """
    Create a lower dimensional grid of the fractures. The main purpose
    of this function is to be able to call the g.get_boundary_faces()
    function. Only works for 1D fractures in 2D
    """
    assert g.dim == 2,'only support for 2D'
    h_list = []
    for i in range(f.num):
        nodes = np.ravel(np.sum(g.face_nodes[:,f.tag[i,:]],axis=1))>0
        node_coords = g.nodes[:,nodes]
        cell_nodes  = g.face_nodes[nodes,:]
        cell_nodes  = cell_nodes[:,f.tag[i,:]]
        num_nodes   = cell_nodes.shape[0]
        face_nodes  = sps.csc_matrix(([True]*num_nodes,
                                      (np.arange(num_nodes),np.arange(num_nodes))),
                                     (num_nodes,num_nodes),dtype='bool')
        [fi, ci, val] = sps.find(cell_nodes)
        data = val.astype(int)
        data[::2] = data[::2]*-1
        cell_faces = sps.csc_matrix((data,(fi,ci)),(face_nodes.shape[1],cell_nodes.shape[1]))
        dim = g.dim - 1
        h = Grid(dim, node_coords, face_nodes, cell_faces,'Lower_dim_grid')
        h.nodes_nodes = np.ravel(np.argwhere(nodes))
        # Add mapping from parent to lower dim grids
        col = np.ravel(np.argwhere(f.tag[i,:]))
        row = np.arange(col.size)
        dat = np.array([True]*col.size)
        h.child_parent = sps.csc_matrix((dat,(row,col)),(h.num_cells,g.num_faces))
        h_list.append(h)
    return h_list

def tag_nodes(g,f):
    """
    Add fields g.node_info and g.frac_nodes to the grid. The g.node_info
    contains information about the fracture nodes. 
    g.node_info: 
        g.node_info[:,0] is an index map to the nodes that lie on the 
            fracutres (excluding free tips).
        g.node_info[:,1] is the number of nodes that should be 
            added to split the grid.
    g.frac_nodes is a sparse matrix that tells which fracture each
        node lies on.
    """
    node_info = np.zeros((0,3),dtype='int32')
    # find tips and nodes of each fracture
    h_list = create_fracture_grid(g,f)
    
    for i, h in enumerate(h_list):
        tip_nodes = np.argwhere(h.face_nodes[:,h.get_boundary_faces()])[:,0]
        node_type = np.ones(h.num_nodes,dtype='int32')
        node_type[tip_nodes] = 0
        new_info = np.vstack((h.nodes_nodes,node_type,[i]*h.num_nodes)).T
        node_info = np.vstack((node_info,new_info))
        

    # Find shared nodes 
    nodes,ix, iv, node_mult = np.unique(node_info[:,0],return_index=True,
                                        return_inverse=True, return_counts=True)
    node_info[:,1] += (node_mult[iv] - 1).astype('int32')
    node_short_info = node_info[ix,:] 
    # row0:node number row1: number of nodes that should be added
    g.node_info = node_short_info[node_short_info[:,1]>0,:2]
    row = node_info[:,0]
    col = node_info[:,2]
    data = [True]*row.size
    # create mapping from nodes to the fracture numbers.
    g.frac_nodes = sps.csc_matrix((data,(row,col)),(g.nodes.shape[1],f.num))

    return h_list
    
def split_faces(g,h_list, f):
    """
    Split faces of the grid along each fractures. This function will
    add an extra face to each fracture face. Note that the original
    and new fracture face will share the same nodes. However, the 
    cell_faces connectivity is updated such that the fractures are
    be internal boundaries (cells on left side of fractures are not
    connected to cells on right side of fracture and vise versa).
    """
    for i in range(f.num):
        frac = f.tag[i,:]
        # Create convenientmappings
        frac_id    = np.argwhere(frac)
        frac_nodes = g.face_nodes[:,frac]
        child_frac = h_list[i].child_parent[:,frac]
        cell_frac  = g.cell_faces[frac,:]

        # duplicate faces along tagged faces.
        g.face_nodes   = sps.hstack((g.face_nodes,frac_nodes))
        for j, h in enumerate(h_list):
            if j==i:
                h.child_parent = sps.hstack((h.child_parent,
                                             child_frac))
                print(h.child_parent)
            else:
                emty = sps.csc_matrix(h.child_parent[:,frac].shape)
                h.child_parent = sps.hstack((h.child_parent,
                                             emty))

        
        # update face info
        g.num_faces   += np.sum(frac)
        g.face_normals = np.hstack((g.face_normals, -g.face_normals[:,frac]))
        g.face_areas   = np.append(g.face_areas, g.face_areas[frac])
        g.face_centers = np.hstack((g.face_centers, g.face_centers[:,frac]))
        g.add_face_tag(frac, FaceTag.FRACTURE | FaceTag.BOUNDARY)
        g.face_tags = np.append(g.face_tags, g.face_tags[frac])

        # Set new cell conectivity
        # Cells on right side does not change. We first add the new left-faces
        # to the left cells
        cell_frac_id   = np.argwhere(cell_frac)
        left_cell      = half_space_int(f.n[:,i,None],f.x0[:,i,None],
                                        g.cell_centers[:,cell_frac_id[:,1]])

        col            = cell_frac_id[left_cell,1]
        row            = cell_frac_id[left_cell,0]
        data           = np.ravel(g.cell_faces[np.ravel(frac_id[row]), col])
        cell_frac_left = sps.csc_matrix((data,(row,col)),
                                        (frac_id.size,g.cell_faces.shape[1]))

        # We remove the right faces of the left cells.
        col  = cell_frac_id[~left_cell,1]
        row  = cell_frac_id[~left_cell,0]
        data = np.ravel(g.cell_faces[np.ravel(frac_id[row]), col])
        cell_frac_right = sps.csc_matrix((data,(row,col)),
                                         (frac_id.size,g.cell_faces.shape[1]))

        g.cell_faces[frac,:] = cell_frac_right
        g.cell_faces    = sps.vstack((g.cell_faces,cell_frac_left),format='csc')

        if i<f.num:
            new_faces = np.zeros((f.tag.shape[0],sum(f.tag[i,:])),dtype='bool')
            new_faces[i,:] = True
            f.tag=np.hstack((f.tag, new_faces))
    return g


def split_nodes(g,f,graph,offset=0):
    """
    splits the nodes of a grid given a fracture and a colored graph. 
    Parameters
    ----------
    g - A grid. All fracture faces should first be duplicated
    f - a Fracture object. 
    graph - a Graph object. All the nodes in the graph should be colored
            acording to the fracture regions (Graph.color_nodes())
    offset - float
             Optional, defaults to 0. This gives the offset from the 
             fracture to the new nodes. Note that this is only for
             visualization, e.g., g.face_centers is not updated. 
    """
    assert g.dim == 2 , 'Only support for 2D'
    # Create convenient mappings
    all_tag        = np.sum(f.tag,axis=0,dtype='bool')
    node_mult      = np.ravel(np.sum(g.face_nodes[:,all_tag],axis=1))
    tip_nodes_id   = np.ravel(np.argwhere(node_mult==1)) # tip nodes
    int_nodes      = node_mult>=3                        # internal nodes
    # We check for >=3 since it is assumed that the faces are already split
    int_nodes_id   = np.ravel(np.argwhere(int_nodes))

    row = np.array([],dtype=np.int32)
    col = np.array([],dtype=np.int32)
    node_count = 0
    # Iterate over each internal node and split it according to the graph.
    # For each cell attached to the node, we check wich color the cell has.
    # All cells with the same color is then attached to a new copy of the
    # node.
    for i, node in enumerate(int_nodes_id):
        # Find cells connected to node
        (_,cells,_) = sps.find(g.cell_nodes()[node,:])
        colors = graph.color[g.child_cell_ind[cells]]
        colors,ix,_ = np.unique(colors,return_inverse=True,return_counts=True)
        new_nodes = np.repeat(g.nodes[:,node,None],colors.size,axis=1)
        for j in range(colors.size):
            # Find faces of each cell that are attached to node
            faces,_,_ = sps.find(g.cell_faces[:,cells[ix==j]])
            faces = np.unique(faces)
            con_to_node = np.ravel(g.face_nodes[node,faces].todense())
            faces = faces[con_to_node]
            col = np.append(col,faces)
            row = np.append(row,[node_count + j]*faces.size)
            # Change position of nodes
            n = np.mean(g.face_normals[:,faces],axis=1)
            new_nodes[:,j] +=n*offset
        g.nodes = np.hstack((g.nodes, new_nodes))
        node_count += colors.size
        
    # Add new nodes to face-node map
    new_face_nodes = sps.csc_matrix(([True]*row.size,(row,col)),(node_count,g.num_faces))
    g.face_nodes = sps.vstack((g.face_nodes, new_face_nodes),format='csc')
    # Remove old nodes
    g = remove_nodes(g, int_nodes_id)

    # Update the number of nodes
    g.num_nodes = g.num_nodes + node_count - int_nodes_id.size
    return True

def split_fractures(g,f,offset=0):
    """
    Wrapper function to split all fractures. Will split faces and nodes
    to create an internal boundary.
    
    The tagged faces are split in two along with connected nodes (except
    tips).

    To be added:
    3D fractures

    Parameters
    ----------
    g - A valid grid

    frac_tag - Fracture class
        an object of Fracture class
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
    >>> import gridding.fractured.split_grid
    >>> # Set up a Cartesian grid
    >>> n = 10
    >>> g = structured.CartGrid([n, n])
    >>> g.compute_geometry()
    >>> # Define fracture
    >>> frac_tag1 = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2, 
    >>>            g.face_centers[0,:]>n/4), g.face_centers[0,:]<3*n/4)
    >>> frac_tag2 = np.logical_and(np.logical_and(g.face_centers[0,:]==n/2,
    >>>                            g.face_centers[1,:]>=n/4),
    >>>                    g.face_centers[1,:]<3*n/4)
    >>> f = split_grid.Fracture(h)
    >>> f.add_tag(g,frac_tag1)
    >>> f.add_tag(g,frac_tag2)
    >>> split_grid.split_fractures(g,f,offset=0.25
    >>> plot_grid.plot_grid(g)
    >>> plt.show()
    """
    h_list = tag_nodes(g,f)
    split_faces(g,h_list,f)
    cell_nodes = g.cell_nodes()
    # Extract cell around fracture
    all_tag = np.sum(f.tag,axis=0,dtype='bool')
    (_,cells,_) = sps.find(g.cell_nodes()[g.node_info[:,0],:])
    (g_frac, unique_faces, unique_nodes) = extract_subgrid(g, cells, sort=True)
    # create mapping from child to parrent grid
    g.child_cell_ind = np.array([-1]*g.num_cells,dtype=np.int)
    g.child_cell_ind[g_frac.parent_cell_ind] = np.arange(g_frac.num_cells)
    # Create a graph of cells around fractures. Each cell is considered a node,
    # and two cells are connected by an edge in the graph if they share a face.
    graph = Graph(g_frac.cell_connection_map())
    graph.color_nodes()
    # Split the nodes along fractures
    split_nodes(g,f,graph,offset=offset)
    g.cell_faces.eliminate_zeros()
    return g, h_list

def remove_nodes(g, rem):
    """
    Remove nodes from grid.
    g - a valid grid definition
    rem - a ndarray of indecies of nodes to be removed
    """
    all_rows = np.arange(g.face_nodes.shape[0])
    rows_to_keep = np.where(np.logical_not(np.in1d(all_rows, rem)))[0]
    g.face_nodes = g.face_nodes[rows_to_keep,:]
    g.nodes = g.nodes[:,rows_to_keep]
    return g


def extract_subgrid(g, c, sort=True):
    """
    Extract a subgrid based on cell indices.

    For simplicity the cell indices will be sorted before the subgrid is
    extracted.

    If the parent grid has geometry attributes (cell centers etc.) these are
    copied to the child.

    No checks are done on whether the cells form a connected area. The method
    should work in theory for non-connected cells, the user will then have to
    decide what to do with the resulting grid. This option has however not been
    tested.

    Parameters:
        g (core.grids.Grid): Grid object, parent
        c (np.array, dtype=int): Indices of cells to be extracted

    Returns:
        core.grids.Grid: Extracted subgrid. Will share (note, *not* copy)
	    geometric fileds with the parent grid. Also has an additional
	    field parent_cell_ind giving correspondance between parent and
	    child cells.
        np.ndarray, dtype=int: Index of the extracted faces, ordered so that
            element i is the global index of face i in the subgrid.
        np.ndarray, dtype=int: Index of the extracted nodes, ordered so that
            element i is the global index of node i in the subgrid.

    """
    if sort:
        c = np.sort(c)

    # Local cell-face and face-node maps.
    cf_sub, unique_faces = __extract_submatrix(g.cell_faces, c)
    fn_sub, unique_nodes = __extract_submatrix(g.face_nodes, unique_faces)

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append('Extract subgrid')

    # Construct new grid.
    h = Grid(g.dim, g.nodes[:, unique_nodes], fn_sub, cf_sub, name)

    # Copy geometric information if any
    if hasattr(g, 'cell_centers'):
        h.cell_centers = g.cell_centers[:, c]
    if hasattr(g, 'cell_volumes'):
        h.cell_volumes = g.cell_volumes[c]
    if hasattr(g, 'face_centers'):
        h.face_centers = g.face_centers[:, unique_faces]
    if hasattr(g, 'face_normals'):
        h.face_normals = g.face_normals[:, unique_faces]
    if hasattr(g, 'face_areas'):
        h.face_areas = g.face_areas[unique_faces]

    h.parent_cell_ind = c

    return h, unique_faces, unique_nodes


def __extract_submatrix(mat, ind):
    """ From a matrix, extract the column specified by ind. All zero columns
    are stripped from the sub-matrix. Mappings from global to local row numbers
    are also returned.
    """
    sub_mat = mat[:, ind]
    cols = sub_mat.indptr
    rows = sub_mat.indices
    data = sub_mat.data
    unique_rows, rows_sub = np.unique(sub_mat.indices,
                                      return_inverse=True)
    return sps.csc_matrix((data, rows_sub, cols)), unique_rows


