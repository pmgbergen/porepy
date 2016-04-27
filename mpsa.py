import numpy as np
import scipy.sparse as sps

from fvdiscr import subcellMapping
from fvdiscr import fvutils
from utils import matrix_compression
from core.grids import structured
from core.constit import fourth_order_tensor
from core.bc import bc


def mpsa(g, constit, bound, faces=None, eta=0):

    nd = g.dim

    # Define subcell topology
    nno, cno, fno, subfno, subhfno = subcellMapping.create_mapping(g)
    num_subhfno = subhfno.size

    csym, casym = _split_stiffness_matrix(constit)

    ncsym, ncasym, cell_node_blocks, sub_cell_index = _tensor_vector_prod(g,
                                                                    csym,
                                                                    casym,
                                                                    cno,
                                                                    fno,
                                                                    nno,
                                                                    subhfno)
    num_sub_cells = cell_node_blocks[0].size
    num_subfno = np.max(subfno) + 1

    # Normal vectors, used for computing pressure gradient terms in
    # Biot's equations. These are mappings from cells to their faces,
    # and are most easily computed prior to elimination of subfaces (below)
    ind_face = np.argsort(np.tile(subhfno, nd))
    hook_normal = sps.coo_matrix((np.ones(num_subhfno * nd),
                                  (np.arange(num_subhfno*nd), ind_face)),
                                 shape=(nd*num_subhfno, ind_face.size)).tocsr()

    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for displacement
    # continuity
    d_cont_grad = fvutils.compute_dist_face_cell(g, cno, fno, nno, subhfno,
                                                 eta)

    # Make subface indices unique, that is, pair the indices from the two
    # adjacent cells
    _, unique_sub_fno = np.unique(subfno, return_index=True)

    # The final expression of Hook's law will involve deformation gradients
    # on one side of the faces only; eliminate the other one.
    hook_sym_grad, hook_asym_grad = __unique_hooks_law(ncsym, ncasym,
                                                       unique_sub_fno, nd)

    # Hook's law, as it comes out of the normal-vector * stiffness matrix is
    # sorted with x-component balances first, then y-, etc. Sort this to a
    # face-wise ordering
    comp2face_ind = np.argsort(np.tile(subfno[unique_sub_fno], nd),
                               kind='mergesort')
    comp2face = sps.coo_matrix((np.ones(comp2face_ind.size),
                                (np.arange(comp2face_ind.size),
                                 comp2face_ind)),
                               shape=(comp2face_ind.size, comp2face_ind.size))
    hook = comp2face * (hook_sym_grad + hook_asym_grad)
    #hook = (hook_sym_grad + hook_asym_grad)
    # For force balance, displacements and stresses on the two sides of the
    # matrices must be paired
    # Operator to create the pairing
    sgn = g.cellFaces[fno, cno].A
    pair_over_subfaces = sps.coo_matrix((sgn[0], (subfno, subhfno)))
    # vector version, to be used on stresses
    pair_over_subfaces_nd = sps.kron(sps.eye(nd), pair_over_subfaces)

    num_eqs_per_component = ncsym.shape[0] / nd

    # Pair displacements
    d_cont_grad = pair_over_subfaces * d_cont_grad
    # ... and stresses
    ncsym = pair_over_subfaces_nd * ncsym
    ncasym = pair_over_subfaces_nd * ncasym

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    d_cont_cell = sps.coo_matrix((sgn[0], (subfno, cno))).tocsr()
    d_cont_cell = sps.kron(sps.eye(nd), d_cont_cell)
    # Zero contribution to stress continuity
    hook_cell = sps.coo_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),
                               shape=(num_subfno * nd,
                                      (np.max(cno) + 1) * nd)).tocsr()

    # Face-wise gradient operator. Used for the term grad_p in Biot's equations
    rows = __expand_indices_nd(cno, nd)
    cols = np.arange(num_subhfno * nd)
    vals = np.tile(sgn, nd)
    div_gradp = sps.coo_matrix((vals[0], (rows, cols)),
                               shape=((np.max(cno) + 1) * nd,
                                      num_subhfno * nd)).tocsr()

    # Reduce topology to one field per subface
    nno = nno[unique_sub_fno]
    fno = fno[unique_sub_fno]
    cno = cno[unique_sub_fno]
    subfno = subfno[unique_sub_fno]
    nsubfno = subfno.max() + 1

    hfi = __expand_indices_nd(subfno, nd)
    hf = __expand_indices_nd(fno, nd)
    hf2f = sps.coo_matrix((np.ones(hf.size), (hf, hfi)),
                          shape=(hf.max()+1, hfi.max()+1)).tocsr()

    # Update signs
    sgn = g.cellFaces[fno, cno].A.ravel(1)

    # Obtain mappings to exclude boundary faces
    exclude_neumann, exclude_dirichlet = _exclude_boundary_mappings(fno,
                                                                    nsubfno,
                                                                    bound)
    exclude_neumann_nd = sps.kron(sps.eye(nd), exclude_neumann)
    exclude_dirichlet_nd = sps.kron(sps.eye(nd), exclude_dirichlet)

    d_cont_grad = sps.kron(sps.eye(nd), d_cont_grad)
    d_cont_grad = exclude_neumann_nd * d_cont_grad
    d_cont_cell = exclude_neumann_nd * d_cont_cell

    ncsym = exclude_dirichlet_nd * ncsym

    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, nno, exclude_dirichlet,
        exclude_neumann, nd)

    rep_ci_single_blk = np.tile(np.arange(num_sub_cells),
                                (nd, 1)).reshape(-1, order='F')

    d_cont_grad_map = np.argsort(np.tile(rep_ci_single_blk, nd),
                                 kind='mergesort')
    d_cont_grad = d_cont_grad[:, d_cont_grad_map]

    d_cont_cell_map = np.argsort(np.tile(np.arange(cno.max()+1), nd),
                                 kind='mergesort')
    d_cont_cell = d_cont_cell[:, d_cont_cell_map]

    grad_eqs = sps.vstack([ncsym, d_cont_grad])
    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    igrad = cols2blk_diag * fvutils.invert_diagonal_blocks(grad,
                                                           size_of_blocks) \
            * rows2blk_diag

    rhs_cells = -sps.vstack([hook_cell, d_cont_cell])

    stress = hf2f * hook * igrad * rhs_cells


    return stress


def _split_stiffness_matrix(constit):
    """
    Split the stiffness matrix into symmetric and asymetric part

    Parameters
    ----------
    constit stiffness tensor

    Returns
    -------
    csym part of stiffness tensor that enters the local calculation
    casym part of stiffness matrix not included in local calculation
    """
    dim = np.sqrt(constit.c.shape[0])

    # We do not know how constit is used outside the discretization,
    # so create deep copies to avoid overwriting
    csym = 0 * constit.copy()
    casym = constit.copy()

    # The splitting is hard coded based on the ordering of elements in the
    # stiffness matrix
    if dim == 2:
        csym[0, 0, :] = casym[0, 0, :]
        csym[1, 1, :] = casym[1, 1, :]
        csym[2, 2, :] = casym[2, 2, :]
        csym[3, 0, :] = casym[3, 0, :]
        csym[0, 3, :] = casym[0, 3, :]
        csym[3, 3, :] = casym[3, 3, :]
    else:  # dim == 3
        csym[0, 0, :] = casym[0, 0, :]
        csym[1, 1, :] = casym[1, 1, :]
        csym[2, 2, :] = casym[2, 2, :]
        csym[3, 3, :] = casym[3, 3, :]
        csym[4, 4, :] = casym[4, 4, :]
        csym[4, 4, :] = casym[4, 4, :]
        csym[5, 5, :] = casym[5, 5, :]
        csym[6, 6, :] = casym[6, 6, :]
        csym[7, 7, :] = casym[7, 7, :]

        csym[4, 0, :] = casym[4, 0, :]
        csym[8, 0, :] = casym[8, 0, :]
        csym[0, 4, :] = casym[0, 4, :]
        csym[8, 4, :] = casym[8, 4, :]
        csym[0, 8, :] = casym[0, 8, :]
        csym[4, 8, :] = casym[4, 8, :]

    casym -= csym
    return csym, casym


def _tensor_vector_prod(g, sym_tensor, asym_tensor, cno, fno, nno, subhfno):
    # Stack cells and nodes, and remove duplicate rows. Since subcell_mapping
    # defines cno and nno (and others) working cell-wise, this will
    # correspond to a unique rows (Matlab-style) from what I understand.
    # This also means that the pairs in cell_node_blocks uniquely defines
    # subcells, and can be used to index gradients etc.
    cell_node_blocks, blocksz = matrix_compression.rlencode(np.vstack((cno,
                                                                       nno)))

    nd = g.dim

    # Duplicates in [cno, nno] corresponds to different faces meeting at the
    # same node. There should be exactly nd of these. This test will fail
    # for pyramids in 3D
    assert np.all(blocksz == nd)

    # Define row and column indices to be used for normal vector matrix
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a vector) and
    # is adjusted according to block sizes
    rn, cn = np.meshgrid(subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    cn += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces, and store in a matrix
    num_nodes = np.diff(g.faceNodes.indptr)
    normals = g.faceNormals[:, fno] / num_nodes[fno]
    normals_mat = sps.coo_matrix((normals.ravel(1), (rn.ravel(1),
                                                     cn.ravel(1)))).tocsr()

    # Then row and columns for stiffness matrix. There are nd^2 elements in
    # the gradient operator, and so the structure is somewhat different from
    # the normal vectors
    rc, cc = np.meshgrid(subhfno, np.arange(nd**2))
    sum_blocksz = np.cumsum(blocksz**2)
    cc += matrix_compression.rldecode(sum_blocksz - blocksz[0]**2, blocksz)

    # Getting the right elements out of the constitutive laws was a bit
    # tricky, but the following code turned out to do the trick
    sym_tensor_swp = np.swapaxes(sym_tensor, 2, 0)
    asym_tensor_swp = np.swapaxes(asym_tensor, 2, 0)

    # The first dimension in csym and casym represent the contribution from
    # all dimensions to the stress in one dimension (in 2D, csym[0:2,:,
    # :] together gives stress in the x-direction etc.
    # Define index vector to access the right rows
    rind = np.arange(nd)

    # Empty matrices to initialize matrix-tensor products. Will be expanded
    # as we move on
    zr = np.zeros(0)
    ncsym = sps.coo_matrix((zr, (zr,
                                 zr)), shape=(0, cc.max() + 1)).tocsr()
    ncasym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()

    # For the asymmetric part of the tensor, we will apply volume averaging.
    # Associate a volume with each sub-cell, and a node-volume as the sum of
    # all surrounding sub-cells
    num_cell_nodes = g.num_cell_nodes()
    cell_vol = g.cellVolumes / num_cell_nodes
    node_vol = np.bincount(nno, weights=cell_vol[cno]) / g.dim

    num_elem = cell_node_blocks.shape[1]
    map_mat = sps.coo_matrix((np.ones(num_elem),
                                  (np.arange(num_elem), cell_node_blocks[1])))
    weight_mat = sps.coo_matrix((cell_vol[cell_node_blocks[0]] / node_vol[
        cell_node_blocks[1]], (cell_node_blocks[1], np.arange(num_elem))))
    average = map_mat * weight_mat

    for iter1 in range(nd):
        # Pick out part of Hook's law associated with this dimension
        #        sym_dim = sym_tensor[rind, :, :]  # not sure if I should use 'view(
        #       # )' here
        # sym_dim = np.reshape(sym_dim, (g.Nc * nd, nd**2))
        sym_dim = np.hstack(sym_tensor_swp[:, :, rind]).transpose()
        asym_dim = np.hstack(asym_tensor_swp[:, :, rind]).transpose()
        # asym_dim = asym_tensor[rind, :,:]
        # asym_dim = np.reshape(asym_dim, (g.Nc * nd, nd ** 2))

        # Distribute (relevant parts of) Hook's law on subcells
        # This will be nd rows, thus cell ci is associated with indices
        # ci*nd+np.arange(nd)
        # sub_cell_base = nd * cell_node_blocks[1]
        # dim_inds = np.arange(nd)
        # dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
        # sub_cell_ind = sub_cell_base + dim_inds
        sub_cell_ind = __expand_indices_nd(cell_node_blocks[0], nd)
        sym_vals = sym_dim[sub_cell_ind, :]
        asym_vals = asym_dim[sub_cell_ind, :]

        csym_mat = sps.coo_matrix((sym_vals.ravel(0),
                                   (rc.ravel(1), cc.ravel(1)))).tocsr()
        casym_mat = sps.coo_matrix((asym_vals.ravel(0),
                                   (rc.ravel(1), cc.ravel(1)))).tocsr()

        for iter2 in range(nd):
            casym_mat[iter2::nd] = average * casym_mat[iter2::nd]

        ncsym = sps.vstack((ncsym, normals_mat * csym_mat))
        ncasym = sps.vstack((ncasym, normals_mat * casym_mat))

        # Increase index vector, so that we get rows contributing to forces
        # in the next dimension
        rind += nd

    grad_ind = cc[::, 0::nd]

    return ncsym, ncasym, cell_node_blocks, grad_ind


def _exclude_boundary_mappings(fno, nsubfno, bnd):
    """
    Define mappings to exclude boundary faces with dirichlet and neumann
    conditions

    Parameters
    ----------
    fno
    nsubfno

    Returns
    -------
    exclude_neumann: Matrix, mapping from all faces to those having flux
                     continuity
    exclude_dirichlet: Matrix, mapping from all faces to those having pressure
                       continuity
    """
    # Define mappings to exclude boundary values
    j = np.argwhere([not it for it in bnd.isNeu[fno]])
    i = np.arange(j.size)
    exclude_neumann = sps.coo_matrix((np.ones(i.size), (i, j.ravel(0))),
                                     shape=(i.size, nsubfno)).tocsr()
    j = np.argwhere([not it for it in bnd.isDir[fno]])
    i = np.arange(j.size)
    exclude_dirichlet = sps.coo_matrix((np.ones(i.size), (i, j.ravel(0))),
                                       shape=(i.size, nsubfno)).tocsr()
    return exclude_neumann, exclude_dirichlet


def _block_diagonal_structure(sub_cell_index, cell_node_blocks, nno,
                              exclude_dirichlet, exclude_neumann, nd):
    """
    Define matrices to turn linear system into block-diagonal form

    Parameters
    ----------
    sub_cell_index
    cell_node_blocks: pairs of cell and node pairs, which defines sub-cells
    nno node numbers associated with balance equations
    exclude_dirichlet mapping to remove rows associated with flux boundary
    exclude_neumann mapping to remove rows associated with pressure boundary

    Returns
    -------
    rows2blk_diag transform rows of linear system to block-diagonal form
    cols2blk_diag transform columns of linear system to block-diagonal form
    size_of_blocks number of equations in each block
    """

    # Stack node numbers of equations on top of each other, and sort them to
    # get block-structure. First eliminate node numbers at the boundary, where
    # the equations are either of flux or pressure continuity (not both)
    nno_stress = exclude_dirichlet * nno
    nno_displacement = exclude_neumann * nno
    node_occ = np.hstack((np.tile(nno_stress, nd),
                          np.tile(nno_displacement, nd)))
    sorted_ind = np.argsort(node_occ, kind='mergesort')
    rows2blk_diag = sps.coo_matrix((np.ones(sorted_ind.size),
                                    (np.arange(sorted_ind.size),
                                     sorted_ind))).tocsr()
    # Size of block systems
    sorted_nodes_rows = node_occ[sorted_ind]
    size_of_blocks = np.bincount(sorted_nodes_rows.astype('int64'))

    # cell_node_blocks[1] contains the node numbers associated with each
    # sub-cell gradient (and so column of the local linear systems). A sort
    # of these will give a block-diagonal structure
    sorted_nodes_cols = np.argsort(cell_node_blocks[1], kind='mergesort')
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel(1)
    cols2blk_diag = sps.coo_matrix((np.ones(sub_cell_index.size),
                                    (subcind_nodes,
                                     np.arange(sub_cell_index.size))
                                    )).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def __expand_indices_nd(ind, nd, direction=1):
    dim_inds = np.arange(nd)
    dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
    new_ind = nd * ind + dim_inds
    new_ind = new_ind.ravel(direction)
    return new_ind


def __unique_hooks_law(csym, casym, unique_sub_fno, nd):
    ind_all = __expand_indices_nd(unique_sub_fno, nd, 0)
    num_eqs = csym.shape[0] / nd
    ind_single = np.tile(unique_sub_fno, (nd, 1))
    increments = np.arange(nd) * num_eqs
    ind_all2 = np.reshape(ind_single + increments[:, np.newaxis], -1)
    ind_all = np.argsort(np.tile(unique_sub_fno, nd), kind='mergesort')
    t = csym
    a = casym
    hook_sym = csym[ind_all2, ::]
    hook_asym = casym[ind_all2, ::]
    return hook_sym, hook_asym

if __name__ == '__main__':
    # Method used for debuging
    nx = np.array([2, 2])
    g = structured.CartGrid(nx)
    g.nodes[0, 4] = 1.
    g.computeGeometry()

    lmbda = np.ones(g.Nc)
    mu = lmbda
    perm = fourth_order_tensor.FourthOrderTensor(g.dim, mu, lmbda)

    bound = bc.BoundaryCondition(g)
    mpsa(g, perm, bound)

