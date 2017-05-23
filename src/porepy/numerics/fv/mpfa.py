"""
Implementation of the multi-point flux approximation O-method.

"""
from __future__ import division
import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils, tpfa
from porepy.grids import structured
from porepy.params import second_order_tensor, bc
from porepy.utils import matrix_compression
from porepy.utils import comp_geom as cg


def mpfa(g, k, bnd, faces=None, eta=0, inverter='numba'):
    """
    Discretize the scalar elliptic equation by the multi-point flux
    approximation method.

    The method computes fluxes over faces in terms of pressures in adjacent
    cells (defined as all cells sharing at least one vertex with the face).
    This corresponds to the MPFA-O method, see

    Aavatsmark (2002): An introduction to the MPFA-O method on
            quadrilateral grids, Comp. Geosci. for details.


    Implementation needs:
        1) The local linear systems should be scaled with the permeability and
        the local grid size, so that we avoid rounding errors accumulating
        under grid refinement / convergence tests.
        2) It should be possible to do a partial update of the discretization
        stensil (say, if we introduce an internal boundary, or modify the
        permeability field).
        3) For large grids, the current implementation will run into memory
        issues, due to the construction of a block diagonal matrix. This can be
        overcome by splitting the discretization into several partial updates.
        4) It probably makes sense to create a wrapper class to store the
        discretization, interface to linear solvers etc.
    Right now, there are concrete plans for 2) - 4).

    Parameters:
        g (core.grids.grid): grid to be discretized
        k (core.constit.second_order_tensor) permeability tensor
        bnd (core.bc.bc) class for boundary values
        faces (np.ndarray) faces to be considered. Intended for partial
            discretization, may change in the future
        eta Location of pressure continuity point. Should be 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.

    Returns:
        scipy.sparse.csr_matrix (shape num_faces, num_cells): flux
            discretization, in the form of mapping from cell pressures to face
            fluxes.
        scipy.sparse.csr_matrix (shape num_faces, num_faces): discretization of
            boundary conditions. Interpreted as fluxes induced by the boundary
            condition (both Dirichlet and Neumann). For Neumann, this will be
            the prescribed flux over the boundary face, and possibly fluxes
            over faces having nodes on the boundary. For Dirichlet, the values
            will be fluxes induced by the prescribed pressure. Incorporation as
            a right hand side in linear system by multiplication with
            divergence operator.

    Example:
        # Set up a Cartesian grid
        g = structured.CartGrid([5, 5])
        k = second_order_tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))
        g.compute_geometry()

        # Dirirchlet boundary conditions
        bound_faces = g.get_boundary_faces().ravel()
        bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

        # Discretization
        flux, bound_flux = mpfa(g, k, bnd)

        # Source in the middle of the domain
        q = np.zeros(g.num_cells)
        q[12] = 1

        # Divergence operator for the grid
        div = fvutils.scalar_divergence(g)

        # Discretization matrix
        A = div * flux

        # Assign boundary values to all faces on the bounary
        bound_vals = np.zeros(g.num_faces)
        bound_vals[bound_faces] = np.arange(bound_faces.size)

        # Assemble the right hand side and solve
        rhs = q + div * bound_flux * bound_vals
        x = sps.linalg.spsolve(A, rhs)
        f = flux * x - bound_flux * bound_vals

    """

    """
    Method properties and implementation details.

    The pressure is discretized as a linear function on sub-cells (see
    reference paper). In this implementation, the pressure is represented by
    its cell center value and the sub-cell gradients (this is in contrast to
    most papers, which use auxiliary pressures on the faces; the current
    formulation is equivalent, but somewhat easier to implement).

    The method will give continuous fluxes over the faces, and pressure
    continuity for certain points (controlled by the parameter eta). This can
    be expressed as a linear system on the form

        (i)   A * grad_p            = 0
        (ii)  B * grad_p + C * p_cc = 0
        (iii) 0            D * p_cc = I

    Here, the first equation represents flux continuity, and involves only the
    pressure gradients (grad_p). The second equation gives pressure continuity
    over cell faces, thus B will contain distances between cell centers and the
    face continuity points, while C consists of +- 1 (depending on which side
    the cell is relative to the face normal vector). The third equation
    enforces the pressure to be unity in one cell at a time. Thus (i)-(iii) can
    be inverted to express the pressure gradients as in terms of the cell
    center variables, that is, we can compute the basis functions on the
    sub-cells. Because of the method construction (again see reference paper),
    the basis function of a cell c will be non-zero on all sub-cells sharing
    a vertex with c. Finally, the fluxes as functions of cell center values are
    computed by insertion into Darcy's law (which is essentially half of A from
    (i), that is, only consider contribution from one side of the face.

    Boundary values can be incorporated with appropriate modifications -
    Neumann conditions will have a non-zero right hand side for (i), while
    Dirichlet gives a right hand side for (ii).
    """

    # The method reduces to the more efficient TPFA in one dimension, so that
    # method may be called. In 0D, there is no internal discretization to be
    # done.
    if g.dim == 1:
        return tpfa.tpfa(g, k, bnd)
    elif g.dim == 0:
        return [0], [0]

    # The grid coordinates are always three-dimensional, even if the grid is
    # really 2D. This means that there is not a 1-1 relation between the number
    # of coordinates of a point / vector and the real dimension. This again
    # violates some assumptions tacitly made in the discretization (in
    # particular that the number of faces of a cell that meets in a vertex
    # equals the grid dimension, and that this can be used to construct an
    # index of local variables in the discretization). These issues should be
    # possible to overcome, but for the moment, we simply force 2D grids to be
    # proper 2D.

    if g.dim == 2:
        # Rotate the grid into the xy plane and delete third dimension. First
        # make a copy to avoid alterations to the input grid
        g = g.copy()
        cell_centers, face_normals, face_centers, R, _, nodes = cg.map_grid(
            g)
        g.cell_centers, g.face_normals, g.face_centers, g.nodes = cell_centers, face_normals, face_centers, nodes

        # Rotate the permeability tensor and delete last dimension
        k = k.copy()
        k.perm = np.tensordot(R.T, np.tensordot(R, k.perm, (1, 0)), (0, 1))
        k.perm = np.delete(k.perm, (2), axis=0)
        # k.perm[0:2, 0:2, :]  # , R.T)
        k.perm = np.delete(k.perm, (2), axis=1)

    # Define subcell topology, that is, the local numbering of faces, subfaces,
    # sub-cells and nodes. This numbering is used throughout the
    # discretization.
    subcell_topology = fvutils.SubcellTopology(g)

    # Obtain normal_vector * k, pairings of cells and nodes (which together
    # uniquely define sub-cells, and thus index for gradients.
    nk_grad, cell_node_blocks, \
        sub_cell_index = _tensor_vector_prod(g, k, subcell_topology)

    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for pressure continuity
    pr_cont_grad = fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # Darcy's law
    darcy = -nk_grad[subcell_topology.unique_subfno]

    # Pair fluxes over subfaces, that is, enforce conservation
    nk_grad = subcell_topology.pair_over_subfaces(nk_grad)

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1 (Depending on whether the cell is on the
    # positive or negative side of the face.
    # The .A suffix is necessary to get a numpy array, instead of a scipy
    # matrix.
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A
    pr_cont_cell = sps.coo_matrix((sgn[0], (subcell_topology.subfno,
                                            subcell_topology.cno))).tocsr()
    # The cell centers give zero contribution to flux continuity
    nk_cell = sps.coo_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),
                             shape=(subcell_topology.num_subfno,
                                    subcell_topology.num_cno)
                             ).tocsr()
    del sgn

    # Mapping from sub-faces to faces
    hf2f = sps.coo_matrix((np.ones(subcell_topology.unique_subfno.size),
                           (subcell_topology.fno_unique,
                            subcell_topology.subfno_unique)))

    # Update signs
    sgn_unique = g.cell_faces[subcell_topology.fno_unique,
                              subcell_topology.cno_unique].A.ravel('F')

    # The boundary faces will have either a Dirichlet or Neumann condition, but
    # not both (Robin is not implemented).
    # Obtain mappings to exclude boundary faces.
    bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bnd, g.dim)

    # No flux conditions for Dirichlet boundary faces
    nk_grad = bound_exclusion.exclude_dirichlet(nk_grad)
    nk_cell = bound_exclusion.exclude_dirichlet(nk_cell)
    # No pressure condition for Neumann boundary faces
    pr_cont_grad = bound_exclusion.exclude_neumann(pr_cont_grad)
    pr_cont_cell = bound_exclusion.exclude_neumann(pr_cont_cell)

    # So far, the local numbering has been based on the numbering scheme
    # implemented in SubcellTopology (which treats one cell at a time). For
    # efficient inversion (below), it is desirable to get the system over to a
    # block-diagonal structure, with one block centered around each vertex.
    # Obtain the necessary mappings.
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, subcell_topology.nno_unique,
        bound_exclusion)

    del cell_node_blocks, sub_cell_index

    # System of equations for the subcell gradient variables. On block diagonal
    # form.
    grad_eqs = sps.vstack([nk_grad, pr_cont_grad])

    num_nk_cell = nk_cell.shape[0]
    num_pr_cont_grad = pr_cont_grad.shape[0]
    del nk_grad, pr_cont_grad

    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    del grad_eqs
    darcy_igrad = darcy * cols2blk_diag * fvutils.invert_diagonal_blocks(grad,
                                                                         size_of_blocks,
                                                                         method=inverter) \
        * rows2blk_diag

    del grad, cols2blk_diag, rows2blk_diag, darcy

    flux = hf2f * darcy_igrad * (-sps.vstack([nk_cell, pr_cont_cell]))

    del nk_cell, pr_cont_cell
    ####
    # Boundary conditions
    rhs_bound = _create_bound_rhs(bnd, bound_exclusion,
                                  subcell_topology, sgn_unique, g,
                                  num_nk_cell, num_pr_cont_grad)
    # Discretization of boundary values
    bound_flux = hf2f * darcy_igrad * rhs_bound

    return flux, bound_flux


#----------------------------------------------------------------------------#
#
# The functions below are helper functions, which are not really necessary to
# understand in detail to use the method. They also tend to be less well
# documented.
#
#----------------------------------------------------------------------------#

def _tensor_vector_prod(g, k, subcell_topology):
    """
    Compute product of normal vectors and tensors on a sub-cell level.

    This is essentially defining Darcy's law for each sub-face in terms of
    sub-cell gradients. Thus, we also implicitly define the global ordering
    of sub-cell gradient variables (via the interpretation of the columns in
    nk).

    NOTE: In the local numbering below, in particular in the variables i and j,
    it is tacitly assumed that g.dim == g.nodes.shape[0] ==
    g.face_normals.shape[0] etc. See implementation note in main method.

    Parameters:
        g (core.grids.grid): Discretization grid
        k (core.constit.second_order_tensor): The permeability tensor
        subcell_topology (fvutils.SubcellTopology): Wrapper class containing
            subcell numbering.

    Returns:
        nk: sub-face wise product of normal vector and permeability tensor.
        cell_node_blocks pairings of node and cell indices, which together
            define a sub-cell.
        sub_cell_ind: index of all subcells

    """

    # Stack cell and nodes, and remove duplicate rows. Since subcell_mapping
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

    # Define row and column indices to be used for normal_vectors * perm.
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a gradient) and
    # is adjusted according to block sizes
    _, j = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    j += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[
        subcell_topology.fno]

    # Represent normals and permeability on matrix form
    ind_ptr = np.hstack((np.arange(0, j.size, nd), j.size))
    normals_mat = sps.csr_matrix((normals.ravel('F'), j.ravel('F'), ind_ptr))
    k_mat = sps.csr_matrix((k.perm[::, ::, cell_node_blocks[0]].ravel('F'),
                            j.ravel('F'), ind_ptr))

    nk = normals_mat * k_mat

    # Unique sub-cell indexes are pulled from column indices, we only need
    # every nd column (since nd faces of the cell meet at each vertex)
    sub_cell_ind = j[::, 0::nd]
    return nk, cell_node_blocks, sub_cell_ind


def _block_diagonal_structure(sub_cell_index, cell_node_blocks, nno,
                              bound_exclusion):
    """ Define matrices to turn linear system into block-diagonal form

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
    nno_flux = bound_exclusion.exclude_dirichlet(nno)
    nno_pressure = bound_exclusion.exclude_neumann(nno)
    node_occ = np.hstack((nno_flux, nno_pressure))
    sorted_ind = np.argsort(node_occ)
    sorted_nodes_rows = node_occ[sorted_ind]
    # Size of block systems
    size_of_blocks = np.bincount(sorted_nodes_rows.astype('int64'))
    rows2blk_diag = sps.coo_matrix((np.ones(sorted_nodes_rows.size),
                                    (np.arange(sorted_ind.size),
                                     sorted_ind))).tocsr()

    # cell_node_blocks[1] contains the node numbers associated with each
    # sub-cell gradient (and so column of the local linear systems). A sort
    # of these will give a block-diagonal structure
    sorted_nodes_cols = np.argsort(cell_node_blocks[1])
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel('F')
    cols2blk_diag = sps.coo_matrix((np.ones(sub_cell_index.size),
                                    (subcind_nodes,
                                     np.arange(sub_cell_index.size))
                                    )).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def _create_bound_rhs(bnd, bound_exclusion,
                      subcell_topology, sgn, g, num_flux, num_pr):
    """
    Define rhs matrix to get basis functions for incorporates boundary
    conditions

    Parameters
    ----------
    bnd
    exclude_dirichlet
    exclude_neumann
    fno
    sgn : +-1, defining here and there of the faces
    g : grid
    num_flux : number of equations for flux continuity
    num_pr: number of equations for pressure continuity

    Returns
    -------
    rhs_bound: Matrix that can be multiplied with inverse block matrix to get
               basis functions for boundary values
    """

    fno = subcell_topology.fno_unique
    num_neu = np.sum(bnd.is_neu[fno])
    num_dir = np.sum(bnd.is_dir[fno])
    num_bound = num_neu + num_dir

    # Neumann boundary conditions
    # Find Neumann faces, exclude Dirichlet faces (since these are excluded
    # from the right hand side linear system), and do necessary formating.
    neu_ind = np.argwhere(bound_exclusion.exclude_dirichlet(
        bnd.is_neu[fno].astype('int64'))).ravel('F')
    # We also need to map the respective Neumann and Dirichlet half-faces to
    # the global half-face numbering (also interior faces). The latter should
    # not have Dirichlet and Neumann excluded (respectively), and thus we need
    # new fields
    neu_ind_all = np.argwhere(bnd.is_neu[fno].astype('int')).ravel('F')
    dir_ind_all = np.argwhere(bnd.is_dir[fno].astype('int')).ravel('F')
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel(order='F')

    # For the Neumann boundary conditions, we define the value as seen from
    # the outside fo the domain. E.g. innflow is defined to be positive. We
    # therefore set the matrix indices to 1. We also have to scale it with
    # the number of nodes per face because the flux of face is the sum of its
    # half-faces.
    scaled_sgn = 1 / num_face_nodes[fno[neu_ind_all]]
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix((scaled_sgn,
                                   (neu_ind, np.arange(neu_ind.size))),
                                  shape=(num_flux, num_bound))
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_flux, num_bound))

    # Dirichlet boundary conditions
    dir_ind = np.argwhere(bound_exclusion.exclude_neumann(
                          bnd.is_dir[fno].astype('int64'))).ravel('F')
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix((sgn[dir_ind_all], (dir_ind, num_neu +
                                                      np.arange(dir_ind.size))),
                                  shape=(num_pr, num_bound))
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_pr, num_bound))

    # Number of elements in neu_ind and neu_ind_all are equal, we can test with
    # any of them. Same with dir.
    if neu_ind.size > 0 and dir_ind.size > 0:
        neu_dir_ind = np.hstack([neu_ind_all, dir_ind_all]).ravel('F')
    elif neu_ind.size > 0:
        neu_dir_ind = neu_ind_all
    elif dir_ind.size > 0:
        neu_dir_ind = dir_ind_all
    else:
        raise ValueError("Boundary values should be either Dirichlet or "
                         "Neumann")

    num_subfno = subcell_topology.num_subfno_unique

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices
    bnd_2_all_hf = sps.coo_matrix((np.ones(num_bound),
                                   (np.arange(num_bound), neu_dir_ind)),
                                  shape=(num_bound, num_subfno))
    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.
    hf_2_f = sps.coo_matrix((np.ones(subcell_topology.subfno_unique.size),
                             (subcell_topology.subfno_unique,
                              subcell_topology.fno_unique)),
                            shape=(num_subfno, g.num_faces))
    rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f
    return rhs_bound


def _neu_face_sgn(g, neu_ind):
    neu_sgn = (g.cell_faces[neu_ind, :]).data
    assert neu_sgn.size == neu_ind.size, \
        'A normal sign is only well defined for a boundary face'
    sort_id = np.argsort(g.cell_faces[neu_ind, :].indices)
    return neu_sgn[sort_id]
