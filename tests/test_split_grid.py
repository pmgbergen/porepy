import numpy as np
from scipy import sparse as sps

from gridding.fractured import split_grid
from core.grids import structured

def test_split_fracture():
    """ Check that no error messages are created in the process of creating a
    split_fracture.
    """
    n = 10
    g = structured.CartGrid([n,n])
    g.compute_geometry()
    frac_tag = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2,
                                             g.face_centers[0,:]>n/4),
                              g.face_centers[0,:]<3*n/4)
    h = g.copy()
    f = split_grid.Fracture(g)
    f.add_tag(g,frac_tag)
    split_grid.split_fractures(h,f)
    # Check that correct number of faces are added
    assert h.num_faces == g.num_faces + sum(frac_tag)
    assert h.num_faces == h.face_nodes.shape[1]
    assert h.cell_faces.shape[0] == h.num_faces
    assert h.cell_faces.shape[1] == g.num_cells
    assert h.num_cells == g.num_cells
    # Check that correct number of nodes are added
    assert h.num_nodes == g.num_nodes + sum(frac_tag)-1
    #assert h.num_nodes == h.face_nodes.shape[0]
    # check that no faces are hanging
    b = np.abs(h.cell_faces).sum(axis=1).A.ravel(1)==0
    assert np.any(b) == False
    # check that internal boundaries are added
    bndr_g = g.get_boundary_faces()
    bndr_h = h.get_boundary_faces()
    assert bndr_g.size ==  bndr_h.size - 2*np.sum(frac_tag)


def test_intersecting_fracture():
    n = 10
    g = structured.CartGrid([n,n])
    g.compute_geometry()
    frac_tag1 = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2,
                                              g.face_centers[0,:]>n/4),
                               g.face_centers[0,:]<3*n/4)

    frac_tag2 = np.logical_and(np.logical_and(g.face_centers[0,:]==n/2,
                                          g.face_centers[1,:]>n/4),
                           g.face_centers[1,:]<3*n/4)
    f = split_grid.Fracture(g)
    f.add_tag(g,frac_tag1)
    f.add_tag(g,frac_tag2)
    h = g.copy()
    split_grid.split_fractures(h,f)
    # check that no nodes are hanging
    # Check that correct number of faces are added
    assert h.num_faces == g.num_faces + np.sum(f.tag)/2
    assert h.num_faces == h.face_nodes.shape[1]
    assert h.cell_faces.shape[0] == h.num_faces
    assert h.cell_faces.shape[1] == g.num_cells
    assert h.num_cells == g.num_cells
    # check that correct number of nodes are added
    print(h.num_nodes)
    print(g.num_nodes)
    print(h.face_nodes.shape)
    assert h.num_nodes == h.face_nodes.shape[0]
    assert h.num_nodes == 128
    # check that no faces are hanging
    b = np.abs(h.cell_faces).sum(axis=1).A.ravel(1)==0
    assert np.any(b) == False
    # check that internal boundaries are added
    bndr_g = g.get_boundary_faces()
    bndr_h = h.get_boundary_faces()
    assert bndr_g.size ==  bndr_h.size - np.sum(f.tag)

def test_simple_case():
    n = 4
    g = structured.CartGrid([n,n])
    g.compute_geometry()

    frac_tag1 = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2,
                                          g.face_centers[0,:]>1),
                               g.face_centers[0,:]<n-1)
    frac_tag2 = np.logical_and(np.logical_and(g.face_centers[0,:]==n/2,
                                              g.face_centers[1,:]>1),
                               g.face_centers[1,:]<n-1)
    h = g.copy()
    
    f = split_grid.Fracture(h)
    f.add_tag(h,frac_tag1)

    h,h_list = split_grid.split_fractures(h,f,offset=.25)
    # Check cell_faces
    cf_col = np.arange(16)
    cf_col = np.repeat(cf_col[:,None],4,axis=1)
    cf_col = np.ravel(cf_col,'c')
    cf_row = np.array([0,1,20,24,1,2,21,25,2,3,22,26,3,4,23,27,5,6,24,28,6,7,25,40,7,8,26,41,
                       8,9,27,31,10,11,28,32,11,12,29,33,12,13,30,34,13,14,31,35,15,16,32,36,16,17,33,37,
                       17,18,34,38,18,19,35,39])

    cf_dat = np.array([-1,1]*32)

    cell_faces=sps.csc_matrix((cf_dat,(cf_row,cf_col)),(42,16))

    assert ((h.cell_faces!=cell_faces).nnz==0)

    # Check cell_nodes
    cn_col = cf_col
    cn_row = np.array([0,1,5,6,1,2,6,7,2,3,7,8,3,4,8,9,5,6,10,11,6,7,11,24,
                       7,8,24,12,8,9,12,13,10,11,14,15,11,25,15,16,25,12,16,17,
                       12,13,17,18,14,15,19,20,15,16,20,21,16,17,21,22,17,18,22,23])
    cn_dat = np.array([True]*cn_col.size)
    cell_nodes = sps.csc_matrix((cn_dat,(cn_row,cn_col)),(26,16))
    assert((h.cell_nodes()!= cell_nodes).nnz==0)


def test_1D_grid():
    n = 6
    g = structured.CartGrid([n,n])
    g.compute_geometry()
    frac_tag1 = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2,
                                              g.face_centers[0,:]>1),
                               g.face_centers[0,:]<n-1)
    frac_tag2 = np.logical_and(np.logical_and(g.face_centers[0,:]==n/2,
                                              g.face_centers[1,:]>n/2),
                               g.face_centers[1,:]<n-1)
    h = g.copy()

    f = split_grid.Fracture(h)
    f.add_tag(h,frac_tag1)
    f.add_tag(h,frac_tag2)
    h,h_list = split_grid.split_fractures(h,f,offset=.25)

    nodes1 = np.array([[1,2,3,4,5],[3,3,3,3,3],[0,0,0,0,0]])
    nodes2 = np.array([[3,3,3],[3,4,5],[0,0,0]])
    assert np.all(h_list[0].nodes==nodes1)
    assert np.all(h_list[1].nodes==nodes2)


    bdr_node = np.array([[0,4],[0,2]])
    for i,k in enumerate(h_list):
        bdr = np.ravel(np.argwhere(k.face_nodes[:,k.get_boundary_faces()])[:,0])
        print(bdr)
        assert np.all(bdr == bdr_node[i])



if __name__ == '__main__':
    test_split_fracture()
    test_intersecting_fracture()
