import numpy as np
import scipy.sparse as sps

from fvdiscr import fvutils
from utils import matrix_compression
from core.grids import structured
from core.constit import fourth_order_tensor
from core.bc import bc


def mpsa(g, constit, bound, faces=None, eta=0, inverter='numba'):

    nd = g.dim

    # Define subcell topology
    subcell_topology = fvutils.SubcellTopology(g)
    # Obtain mappings to exclude boundary faces
    bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

    hook, igrad, rhs_cells, _, _ = __mpsa_elasticity(g, constit,
                                                   subcell_topology,
                                                  bound_exclusion, eta,
                                                  inverter)

    # Output should be on face-level (not sub-face)
    hf2f = _map_hf_2_f(subcell_topology.fno_unique,
                       subcell_topology.subfno_unique, nd)

    # Stress discretization
    stress = hf2f * hook * igrad * rhs_cells

    # Right hand side for boundary discretization
    rhs_bound = _create_bound_rhs(bound, bound_exclusion, subcell_topology, g)
    # Discretization of boundary values
    bound_stress = hf2f * hook * igrad * rhs_bound

    return stress, bound_stress

# @profile
def __mpsa_elasticity(g, constit, subcell_topology, bound_exclusion, eta,
                      inverter):

    nd = g.dim

    # Compute product between normal vectors and stiffness matrices
    ncsym, ncasym, cell_node_blocks, \
        sub_cell_index = _tensor_vector_prod(g, constit, subcell_topology)

    ind_f = np.argsort(np.tile(subcell_topology.subhfno, nd), kind='mergesort')
    hook_normal = sps.coo_matrix((np.ones(ind_f.size),
                                  (np.arange(ind_f.size), ind_f)),
                                 shape=(ind_f.size, ind_f.size)) * (ncsym
                                                                    + ncasym)

    # The final expression of Hook's law will involve deformation gradients
    # on one side of the faces only; eliminate the other one.
    # Note that this must be done before we can pair forces from the two
    # sides of the faces.
    hook = __unique_hooks_law(ncsym, ncasym, subcell_topology, nd)

    ncsym = subcell_topology.pair_over_subfaces_nd(ncsym)
    ncsym = bound_exclusion.exclude_dirichlet_nd(ncsym)
    num_subfno = subcell_topology.subfno.max() + 1
    hook_cell = sps.coo_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),
                               shape=(num_subfno * nd,
                                      (np.max(subcell_topology.cno) + 1) *
                                      nd)).tocsr()
    hook_cell = bound_exclusion.exclude_dirichlet_nd(hook_cell)

    # Book keeping
    num_sub_cells = cell_node_blocks[0].size

    d_cont_grad, d_cont_cell = __get_displacement_submatrices(g,
                                                              subcell_topology,
                                                              eta,
                                                              num_sub_cells,
                                                              bound_exclusion)

    grad_eqs = sps.vstack([ncsym, d_cont_grad])

    igrad = inverse_gradient(grad_eqs, sub_cell_index, cell_node_blocks,
                             subcell_topology.nno_unique, bound_exclusion,
                             nd, inverter)

    # Right hand side for cell center variables
    rhs_cells = -sps.vstack([hook_cell, d_cont_cell])
    return hook, igrad, rhs_cells, cell_node_blocks, hook_normal


def biot(g, constit, bound, faces=None, eta=0, inverter='numba'):
    nd = g.dim

    # Define subcell topology
    subcell_topology = fvutils.SubcellTopology(g)
    # Obtain mappings to exclude boundary faces
    bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

    num_subhfno = subcell_topology.subhfno.size

    num_nodes = np.diff(g.face_nodes.indptr)
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

    def build_rhs_normals_single_dimension(dim):
        val = g.face_normals[dim, subcell_topology.fno] \
              * sgn / num_nodes[subcell_topology.fno]
        mat = sps.coo_matrix((val.squeeze(), (subcell_topology.subfno,
                                              subcell_topology.cno)),
                             shape=(subcell_topology.num_subfno,
                                    subcell_topology.num_cno))
        return mat

    rhs_normals = build_rhs_normals_single_dimension(0)
    for iter1 in range(1, nd):
        this_dim = build_rhs_normals_single_dimension(iter1)
        rhs_normals = sps.vstack([rhs_normals, this_dim])

    rhs_normals = bound_exclusion.exclude_dirichlet_nd(rhs_normals)

    num_dir_subface = (bound_exclusion.exclude_neu.shape[1] -
                       bound_exclusion.exclude_neu.shape[0]) * nd
    rhs_normals_displ_var = sps.coo_matrix((nd * subcell_topology.num_subfno
                                            - num_dir_subface,
                                            subcell_topology.num_cno))

    # Why minus?
    rhs_normals = -sps.vstack([rhs_normals, rhs_normals_displ_var])

    # Call core part of MPSA
    hook, igrad, rhs_cells, \
        cell_node_blocks, hook_normal = __mpsa_elasticity(g, constit,
                                                   subcell_topology,
                                             bound_exclusion, eta, inverter)

    # Output should be on face-level (not sub-face)
    hf2f = _map_hf_2_f(subcell_topology.fno_unique,
                       subcell_topology.subfno_unique, nd)

    # Stress discretization
    stress = hf2f * hook * igrad * rhs_cells

    # Right hand side for boundary discretization
    rhs_bound = _create_bound_rhs(bound, bound_exclusion, subcell_topology, g)
    # Discretization of boundary values
    bound_stress = hf2f * hook * igrad * rhs_bound

    # Face-wise gradient operator. Used for the term grad_p in Biot's
    # equations.
    rows = __expand_indices_nd(subcell_topology.cno, nd)
    cols = np.arange(num_subhfno * nd)
    vals = np.tile(sgn, (nd, 1)).ravel('F')
    div_gradp = sps.coo_matrix((vals, (rows, cols)),
                               shape=(subcell_topology.num_cno * nd,
                                      num_subhfno * nd)).tocsr()
    # Normal vectors, used for computing pressure gradient terms in
    # Biot's equations. These are mappings from cells to their faces,
    # and are most easily computed prior to elimination of subfaces (below)
    # ind_face = np.argsort(np.tile(subcell_topology.subhfno, nd))
    # hook_normal = sps.coo_matrix((np.ones(num_subhfno * nd),
    #                               (np.arange(num_subhfno*nd), ind_face)),
    #                              shape=(nd*num_subhfno, ind_face.size)).tocsr()

    grad_p = div_gradp * hook_normal * igrad * rhs_normals
    # assert np.allclose(grad_p.sum(axis=0), np.zeros(g.num_cells))

    num_cell_nodes = g.num_cell_nodes()
    cell_vol = g.cell_volumes / num_cell_nodes

    if nd == 2:
        trace = np.array([0, 3])
    elif nd == 3:
        trace = np.array([0, 4, 8])
    row, col = np.meshgrid(np.arange(cell_node_blocks.shape[1]), trace)
    incr = np.cumsum(nd**2 * np.ones(cell_node_blocks.shape[1])) - nd**2
    col += incr.astype('int32')
    val = np.tile(cell_vol[cell_node_blocks[0]], (nd, 1))
    vector_2_scalar = sps.coo_matrix((val.ravel('F'),
                                      (row.ravel('F'),
                                       col.ravel('F')))).tocsr()
    div_op = sps.coo_matrix((np.ones(cell_node_blocks.shape[1]),
                             (cell_node_blocks[0], np.arange(
                                 cell_node_blocks.shape[1])))).tocsr()
    div = div_op * vector_2_scalar

    div_d = div * igrad * rhs_cells
    stabilization = div * igrad * rhs_normals

    return stress, bound_stress, grad_p, div_d, stabilization


def __get_displacement_submatrices(g, subcell_topology, eta, num_sub_cells,
                                   bound_exclusion):
    nd = g.dim
    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for displacement
    # continuity
    d_cont_grad = fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # For force balance, displacements and stresses on the two sides of the
    # matrices must be paired
    d_cont_grad = sps.kron(sps.eye(nd), d_cont_grad)

    # Contribution from cell center potentials to local systems
    d_cont_cell = __cell_variable_contribution(g, subcell_topology)

    # Expand equations for displacement balance, and eliminate rows
    # associated with neumann boundary conditions
    d_cont_grad = bound_exclusion.exclude_neumann_nd(d_cont_grad)
    d_cont_cell = bound_exclusion.exclude_neumann_nd(d_cont_cell)

    # The column ordering of the displacement equilibrium equations are
    # formed as a Kronecker product of scalar equations. Bring them to the
    # same form as that applied in the force balance equations
    d_cont_grad, d_cont_cell = __rearange_columns_displacement_eqs(
        d_cont_grad, d_cont_cell, num_sub_cells, nd,
        subcell_topology.cno_unique)
    return d_cont_grad, d_cont_cell


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
    # so create deep copies to avoid overwriting. Not really sure if this is
    #  necessary
    csym = 0 * constit.copy()
    casym = constit.copy()

    # The splitting is hard coded based on the ordering of elements in the
    # stiffness matrix
    if dim == 2:
        csym[0, 0] = casym[0, 0]
        csym[1, 1] = casym[1, 1]
        csym[2, 2] = casym[2, 2]
        csym[3, 0] = casym[3, 0]
        csym[0, 3] = casym[0, 3]
        csym[3, 3] = casym[3, 3]
    else:  # dim == 3
        csym[0, 0] = casym[0, 0]
        csym[1, 1] = casym[1, 1]
        csym[2, 2] = casym[2, 2]
        csym[3, 3] = casym[3, 3]
        csym[4, 4] = casym[4, 4]
        csym[5, 5] = casym[5, 5]
        csym[6, 6] = casym[6, 6]
        csym[7, 7] = casym[7, 7]
        csym[8, 8] = casym[8, 8]

        csym[4, 0] = casym[4, 0]
        csym[8, 0] = casym[8, 0]
        csym[0, 4] = casym[0, 4]
        csym[8, 4] = casym[8, 4]
        csym[0, 8] = casym[0, 8]
        csym[4, 8] = casym[4, 8]
    # The asymmetric part is whatever is not in the symmetric part
    casym -= csym
    return csym, casym


def _tensor_vector_prod(g, constit, subcell_topology):
    # Stack cells and nodes, and remove duplicate rows. Since subcell_mapping
    # defines cno and nno (and others) working cell-wise, this will
    # correspond to a unique rows (Matlab-style) from what I understand.
    # This also means that the pairs in cell_node_blocks uniquely defines
    # subcells, and can be used to index gradients etc.
    cell_node_blocks, blocksz = matrix_compression.rlencode(np.vstack((
        subcell_topology.cno, subcell_topology.nno)))

    nd = g.dim

    # Duplicates in [cno, nno] corresponds to different faces meeting at the
    # same node. There should be exactly nd of these. This test will fail
    # for pyramids in 3D
    assert np.all(blocksz == nd)

    # Define row and column indices to be used for normal vector matrix
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a vector) and
    # is adjusted according to block sizes
    rn, cn = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    cn += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces, and store in a matrix
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[
        subcell_topology.fno]
    normals_mat = sps.coo_matrix((normals.ravel(1), (rn.ravel('F'),
                                                     cn.ravel('F')))).tocsr()

    # Then row and columns for stiffness matrix. There are nd^2 elements in
    # the gradient operator, and so the structure is somewhat different from
    # the normal vectors
    rc, cc = np.meshgrid(subcell_topology.subhfno, np.arange(nd**2))
    sum_blocksz = np.cumsum(blocksz**2)
    cc += matrix_compression.rldecode(sum_blocksz - blocksz[0]**2, blocksz)

    # Splitt stiffness matrix into symmetric and anti-symmatric part
    sym_tensor, asym_tensor = _split_stiffness_matrix(constit)

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
    cell_vol = g.cell_volumes / num_cell_nodes
    node_vol = np.bincount(subcell_topology.nno, weights=cell_vol[
        subcell_topology.cno]) / g.dim

    num_elem = cell_node_blocks.shape[1]
    map_mat = sps.coo_matrix((np.ones(num_elem),
                             (np.arange(num_elem), cell_node_blocks[1])))
    weight_mat = sps.coo_matrix((cell_vol[cell_node_blocks[0]] / node_vol[
        cell_node_blocks[1]], (cell_node_blocks[1], np.arange(num_elem))))
    # Operator for carying out the average
    average = sps.kron(map_mat * weight_mat, sps.identity(nd)).tocsr()

    for iter1 in range(nd):
        # Pick out part of Hook's law associated with this dimension
        # The code here looks nasty, it should be possible to get the right
        # format of the submatrices in a simpler way, but I couldn't do it.
        sym_dim = np.hstack(sym_tensor_swp[:, :, rind]).transpose()
        asym_dim = np.hstack(asym_tensor_swp[:, :, rind]).transpose()

        # Distribute (relevant parts of) Hook's law on subcells
        # This will be nd rows, thus cell ci is associated with indices
        # ci*nd+np.arange(nd)
        sub_cell_ind = __expand_indices_nd(cell_node_blocks[0], nd)
        sym_vals = sym_dim[sub_cell_ind]
        asym_vals = asym_dim[sub_cell_ind]

        # Represent this part of the stiffness matrix in matrix form
        csym_mat = sps.coo_matrix((sym_vals.ravel('C'),
                                   (rc.ravel('F'), cc.ravel('F')))).tocsr()
        casym_mat = sps.coo_matrix((asym_vals.ravel(0),
                                   (rc.ravel('F'), cc.ravel('F')))).tocsr()

        # Compute average around vertexes
        casym_mat = average * casym_mat

        # Compute products of normal vectors and stiffness tensors,
        # and stack dimensions vertically
        ncsym = sps.vstack((ncsym, normals_mat * csym_mat))
        ncasym = sps.vstack((ncasym, normals_mat * casym_mat))

        # Increase index vector, so that we get rows contributing to forces
        # in the next dimension
        rind += nd

    grad_ind = cc[:, ::nd]

    return ncsym, ncasym, cell_node_blocks, grad_ind


def inverse_gradient(grad_eqs, sub_cell_index, cell_node_blocks,
                     nno_unique, bound_exclusion, nd, inverter):

    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, nno_unique, bound_exclusion, nd)

    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    # Compute inverse gradient operator, and map back again
    igrad = cols2blk_diag * fvutils.invert_diagonal_blocks(grad,
                                                           size_of_blocks,
                                                           method=inverter) \
            * rows2blk_diag
    return igrad


def _block_diagonal_structure(sub_cell_index, cell_node_blocks, nno,
                              bound_exclusion, nd):
    """
    Define matrices to turn linear system into block-diagonal form.

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
    nno_stress = bound_exclusion.exclude_dirichlet(nno)
    nno_displacement = bound_exclusion.exclude_neumann(nno)
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


def _create_bound_rhs(bound, bound_exclusion, subcell_topology, g):
    """
    Define rhs matrix to get basis functions for incorporates boundary
    conditions

    Parameters
    ----------
    bound
    bound_exclusion
    fno
    sgn : +-1, defining here and there of the faces
    g : grid
    num_stress : number of equations for flux continuity
    num_displ: number of equations for pressure continuity

    Returns
    -------
    rhs_bound: Matrix that can be multiplied with inverse block matrix to get
               basis functions for boundary values
    """
    nd = g.dim
    num_stress = bound_exclusion.exclude_dir.shape[0] * nd
    num_displ = bound_exclusion.exclude_neu.shape[0] * nd
    fno = subcell_topology.fno_unique
    subfno = subcell_topology.subfno_unique
    sgn = g.cell_faces[subcell_topology.fno_unique,
                       subcell_topology.cno_unique].A.ravel(1)

    num_neu = sum(bound.is_neu[fno]) * nd
    num_dir = sum(bound.is_dir[fno]) * nd
    num_bound = num_neu + num_dir

    # Convenience method for duplicating a list, with a certain increment
    def expand_ind(ind, dim, increment):
        # Duplicate rows
        ind_nd = np.tile(ind, (dim, 1))
        # Add same increment to each row (0*incr, 1*incr etc.)
        ind_incr = ind_nd + increment * np.array([np.arange(dim)]).transpose()
        # Back to row vector
        ind_new = ind_incr.reshape(-1, order='F')
        return ind_new

    # Define right hand side for Neumann boundary conditions
    # First row indices in rhs matrix
    is_neu = bound_exclusion.exclude_dirichlet(bound.is_neu[fno].astype(
        'int64'))
    neu_ind_single = np.argwhere(is_neu).ravel('F')
    # There are is_neu.size Neumann conditions per dimension
    neu_ind = expand_ind(neu_ind_single, nd, is_neu.size)

    # Some care is needed to compute coefficients in Neumann matrix: sgn is
    # already defined according to the subcell topology [fno], while areas
    # must be drawn from the grid structure, and thus go through fno

    # The signs of the faces should be expanded exactly the same way as the
    # row indices, but with zero increment
    neu_sgn = expand_ind(sgn[neu_ind_single], nd, 0)
    fno_ext = np.tile(fno, nd)
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel('F')
    # No need to expand areas, this is done implicitly via neu_ind
    neu_area = g.face_areas[fno_ext[neu_ind]] / num_face_nodes[fno_ext[
        neu_ind]]
    # Coefficients in the matrix
    neu_val = neu_sgn * neu_area

    # The columns will be 0:neu_ind.size
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix((neu_val.ravel('F'),
                                   (neu_ind, np.arange(neu_ind.size))),
                                  shape=(num_stress, num_bound)).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_stress, num_bound)).tocsr()

    # Dirichlet boundary conditions, procedure is similar to that for Neumann
    is_dir = bound_exclusion.exclude_neumann(bound.is_dir[fno].astype(
        'int64'))
    dir_ind_single = np.argwhere(is_dir).ravel('F')
    dir_ind = expand_ind(dir_ind_single, nd, is_dir.size)
    # The coefficients in the matrix should be duplicated the same way as
    # the row indices, but with no increment
    dir_val = expand_ind(sgn[dir_ind_single], nd, 0)
    # Column numbering starts right after the last Neumann column
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix((dir_val, (dir_ind, num_neu +
                                             np.arange(dir_ind.size))),
                                  shape=(num_displ, num_bound)).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_displ, num_bound)).tocsr()

    num_subfno = np.max(subfno) + 1

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices
    is_bnd = np.hstack((neu_ind_single, dir_ind_single))
    bnd_ind = __expand_indices_nd(is_bnd, nd)
    bnd_2_all_hf = sps.coo_matrix((np.ones(num_bound),
                                   (np.arange(num_bound), bnd_ind)),
                                  shape=(num_bound, num_subfno * nd))
    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.
    hf_2_f = _map_hf_2_f(fno, subfno, nd).transpose()

    rhs_bound = -sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f
    return rhs_bound


def _map_hf_2_f(fno, subfno, nd):
    """
    Create mapping from half-faces to faces
    Parameters
    ----------
    fno face numbering in sub-cell topology based on unique subfno
    subfno sub-face numbering
    nd dimension

    Returns
    -------

    """

    hfi = __expand_indices_nd(subfno, nd)
    hf = __expand_indices_nd(fno, nd)
    hf2f = sps.coo_matrix((np.ones(hf.size), (hf, hfi)),
                          shape=(hf.max() + 1, hfi.max() + 1)).tocsr()
    return hf2f


def __expand_indices_nd(ind, nd, direction=1):
    """
    Expand indices from scalar to vector form.

    Examples:
    >>> i = np.array([0, 1, 3])
    >>> __expand_indices_nd(i, 2)
    (array([0, 1, 2, 3, 6, 7]))

    >>> __expand_indices_nd(i, 3, 0)
    (array([0, 3, 9, 1, 4, 10, 2, 5, 11])

    Parameters
    ----------
    ind
    nd
    direction

    Returns
    -------

    """
    dim_inds = np.arange(nd)
    dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
    new_ind = nd * ind + dim_inds
    new_ind = new_ind.ravel(direction)
    return new_ind


def __unique_hooks_law(csym, casym, subcell_topology, nd):
    """
    Go from products of normal vectors with stiffness matrices (symmetric
    and asymmetric), covering both sides of faces, to a discrete Hook's law,
    that, when multiplied with sub-cell gradients, will give face stresses

    Parameters
    ----------
    csym
    casym
    unique_sub_fno
    subfno
    nd

    Returns
    -------
    hook (sps.csr) nd * (nsubfno, ncells)
    """
    # unique_sub_fno covers scalar equations only. Extend indices to cover
    # multiple dimensions
    num_eqs = csym.shape[0] / nd
    ind_single = np.tile(subcell_topology.unique_subfno, (nd, 1))
    increments = np.arange(nd) * num_eqs
    ind_all = np.reshape(ind_single + increments[:, np.newaxis], -1)

    # Unique part of symmetric and asymmetric products
    hook_sym = csym[ind_all, ::]
    hook_asym = casym[ind_all, ::]

    # Hook's law, as it comes out of the normal-vector * stiffness matrix is
    # sorted with x-component balances first, then y-, etc. Sort this to a
    # face-wise ordering
    comp2face_ind = np.argsort(np.tile(subcell_topology.subfno_unique, nd),
                               kind='mergesort')
    comp2face = sps.coo_matrix((np.ones(comp2face_ind.size),
                                (np.arange(comp2face_ind.size),
                                 comp2face_ind)),
                               shape=(comp2face_ind.size, comp2face_ind.size))
    hook = comp2face * (hook_sym + hook_asym)

    return hook


def __cell_variable_contribution(g, subcell_topology):
    """
    Construct contribution from cell center variables to local systems.
    For stress equations, these are zero, while for cell centers it is +- 1
    Parameters
    ----------
    g
    fno
    cno
    subfno

    Returns
    -------

    """
    nd = g.dim
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    d_cont_cell = sps.coo_matrix((sgn[0], (subcell_topology.subfno,
                                           subcell_topology.cno))).tocsr()
    d_cont_cell = sps.kron(sps.eye(nd), d_cont_cell)
    # Zero contribution to stress continuity

    return d_cont_cell


def __rearange_columns_displacement_eqs(d_cont_grad, d_cont_cell,
                                        num_sub_cells, nd, cno):
    """ Transform columns of displacement balance from increasing cell
    ordering (first x-variables of all cells, then y) to increasing
    variables (first all variables of the first cells, then...)

    Parameters
    ----------
    d_cont_grad
    d_cont_cell
    num_sub_cells
    nd
    cno

    Returns
    -------

    """
    # Repeat sub-cell indices nd times. Fortran ordering (column major)
    # gives same ordering of indices as used for the scalar equation (where
    # there are nd gradient variables for each sub-cell), and thus the
    # format of each block in d_cont_grad
    rep_ci_single_blk = np.tile(np.arange(num_sub_cells),
                                (nd, 1)).reshape(-1, order='F')
    # Then repeat the single-block indices nd times (corresponding to the
    # way d_cont_grad is constructed by Kronecker product), and find the
    # sorting indices
    d_cont_grad_map = np.argsort(np.tile(rep_ci_single_blk, nd),
                                 kind='mergesort')
    # Use sorting indices to bring d_cont_grad to the same order as that
    # used for the columns in the stress continuity equations
    d_cont_grad = d_cont_grad[:, d_cont_grad_map]

    # For the cell displacement variables, we only need a single expansion (
    # corresponding to the second step for the gradient unknowns)
    d_cont_cell_map = np.argsort(np.tile(np.arange(cno.max() + 1), nd),
                                 kind='mergesort')
    d_cont_cell = d_cont_cell[:, d_cont_cell_map]
    return d_cont_grad, d_cont_cell


if __name__ == '__main__':
    # Method used for debuging
    nx = np.array([2, 2])
    g = structured.CartGrid(nx)
    g.nodes[0, 4] = 1.5
    g.compute_geometry()

    lmbda = np.ones(g.num_cells)
    mu = lmbda
    perm = fourth_order_tensor.FourthOrderTensor(g.dim, mu, lmbda)

    bnd = bc.BoundaryCondition(g)
    biot(g, perm, bnd)

