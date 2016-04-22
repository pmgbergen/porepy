# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:34:33 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps

from fvdiscr import subcellMapping
from fvdiscr import fvutils
from utils import matrix_compression
from core.grids import structured
from core.constit import second_order_tensor
from core.bc import bc


def mpfa(g, k, bnd, faces=None, eta=0):
    """
    MPFA discretization

    Parameters
    ----------
    g (core.grids.grid) to be discretized
    k (core.constit.second_order_tensor) permeability tensor
    bnd (core.bc.bc) class for boundary values
    faces (np.ndarray) faces to be considered. Intended for partial
        discretization, may change in the future
    eta Location of continuity point. Should be 1/3 for simplex grids,
        0 otherwise

    Returns
    -------
    flux (np.ndarray, shape (num_faces, num_cells), flux discretization,
        in the form of mapping from cell pressures to face fluxes
    bound_flux (np.ndaray, shape (num_faces, num_faces) discretization of
        boundary conditions. Interpreted as fluxes induced by the boundary
        condition (both Dirichlet and Neumann). For Neumann, this will be
        the prescribed flux over the boundary face, and possibly fluxes over
        faces having nodes on the boundary. For Dirichlet, the values will
        be fluxes induced by the prescribed pressure. Incorporation as a
        right hand side in linear system by multiplication with divergence
        operator
    """

    # Define subcell topology
    nno, cno, fno, subfno, subhfno = subcellMapping.create_mapping(g)

    # TODO: Scaling should be done here, but first in Matlab

    # Obtain normal_vector * k, pairings of cells and nodes (which together
    # uniquely define sub-cells, and thus index for gradients.
    nk_grad, cell_node_blocks, sub_cell_index = _tensor_vector_prod(g, k, cno,
                                                                    fno, nno,
                                                                    subhfno)

    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for pressure continuity
    pr_cont_grad = fvutils.compute_dist_face_cell(g, cno, fno, nno, subhfno,
                                                  eta)

    # Make subface indices unique, that is, pair the indices from the two
    # adjacent cells
    _, unique_sub_fno = np.unique(subfno, return_index=True)

    # Operator to create the pairing
    sgn = g.cellFaces[fno, cno].A
    pair_over_subfaces = sps.coo_matrix((sgn[0], (subfno, subhfno)))

    # Darcy's law
    darcy = -nk_grad[unique_sub_fno, ::]

    # Pair over subfaces    
    nk_grad = pair_over_subfaces * nk_grad
    pr_cont_grad = pair_over_subfaces * pr_cont_grad

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    pr_cont_cell = sps.coo_matrix((sgn[0], (subfno, cno))).tocsr()
    # Zero contribution to flux continuity
    nk_cell = sps.coo_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),
                             shape=(np.max(subfno) + 1, np.max(cno) + 1)
                             ).tocsr()

    # Reduce topology to one field per subface
    nno = nno[unique_sub_fno]
    fno = fno[unique_sub_fno]
    cno = cno[unique_sub_fno]
    subfno = subfno[unique_sub_fno]
    nsubfno = subfno.max() + 1

    # Mapping from sub-faces to faces
    hf2f = sps.coo_matrix((np.ones(unique_sub_fno.size), (fno, subfno)))

    # Update signs
    sgn = g.cellFaces[fno, cno].A.ravel(1)

    # Obtain mappings to exclude boundary faces
    exclude_neumann, exclude_dirichlet = _exclude_boundary_mappings(fno,
                                                                    nsubfno,
                                                                    bnd)

    # No flux conditions for Dirichlet boundary faces
    nk_grad = exclude_dirichlet * nk_grad
    nk_cell = exclude_dirichlet * nk_cell
    # No pressure condition for Neumann boundary faces
    pr_cont_grad = exclude_neumann * pr_cont_grad
    pr_cont_cell = exclude_neumann * pr_cont_cell

    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, nno, exclude_dirichlet,
        exclude_neumann)

    grad_eqs = sps.vstack([nk_grad, pr_cont_grad])
    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    igrad = cols2blk_diag * fvutils.invert_diagonal_blocks(grad,
                                                           size_of_blocks) \
                          * rows2blk_diag

    rhs_cells = -sps.vstack([nk_cell, pr_cont_cell])

    flux = hf2f * darcy * igrad * rhs_cells

    ####
    # Boundary conditions
    rhs_bound = _create_bound_rhs(bnd, exclude_dirichlet, exclude_neumann,
                                  fno, subfno, sgn, g, nk_cell.shape[0],
                                  pr_cont_grad.shape[0])
    # Discretization of boundary values
    bound_flux = hf2f * darcy * igrad * rhs_bound

    return flux, bound_flux


def _tensor_vector_prod(g, k, cno, fno, nno, subhfno):
    """
    Compute product of normal vectors and tensors on a sub-cell level.

    This is essentially defining Darcy's law for each sub-face in terms of
    sub-cell gradients. Thus, we also implicitly define the global ordering
    of sub-cell gradient variables (via the interpretation of the columns in
    nk).

    Parameters
    ----------
    g
    k
    cno
    fno
    nno
    subhfno

    Returns
    -------
    nk sub-face wise product of normal vector and permeability tensor.
    cell_node_blocks pairings of node and cell indices, which together define
              a sub-cell
    sub_cell_ind index of all subcells
    """

    # Stack cell and nodes, and remove duplicate rows. Since subcell_mapping
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

    # Define row and column indices to be used for normal_vectors * perm.
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a gradient) and
    # is adjusted according to block sizes
    i, j = np.meshgrid(subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    j += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces
    num_nodes = np.diff(g.faceNodes.indptr)
    normals = g.faceNormals[:, fno] / num_nodes[fno]

    # Represent normals and permeability on matrix form
    normals_mat = sps.coo_matrix((normals.ravel(1), (i.ravel(1),
                                                     j.ravel(1)))).tocsr()
    k_mat = sps.coo_matrix((k.perm[::, ::, cell_node_blocks[0]].ravel(1),
                            (i.ravel(1), j.ravel(1)))).tocsr()

    nk = normals_mat * k_mat

    # Unique sub-cell indexes are pulled from column indices, we only need
    # every nd column (since nd faces of the cell meet at each vertex)
    sub_cell_ind = j[::, 0::nd]
    return nk, cell_node_blocks, sub_cell_ind


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
                              exclude_dirichlet, exclude_neumann):
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
    nno_flux = exclude_dirichlet * nno
    nno_pressure = exclude_neumann * nno
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
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel(1)
    cols2blk_diag = sps.coo_matrix((np.ones(sub_cell_index.size),
                                    (subcind_nodes,
                                     np.arange(sub_cell_index.size))
                                    )).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def _create_bound_rhs(bnd, exclude_dirichlet, exclude_neumann, fno, subfno,
                      sgn, g, num_flux, num_pr):
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
    num_neu = sum(bnd.isNeu[fno])
    num_dir = sum(bnd.isDir[fno])
    num_bound = num_neu + num_dir

    # Neumann boundary conditions
    neu_ind = np.argwhere(exclude_dirichlet *
                          bnd.isNeu[fno].astype('int64')).ravel(1)
    num_face_nodes = g.faceNodes.sum(axis=0).A.ravel(1)
    # sgn is already defined according to fno, while g.faceAreas is raw data,
    # and therefore needs a combined mapping
    signed_bound_areas = sgn[neu_ind] * g.faceAreas[fno[neu_ind]]\
                        /num_face_nodes[fno[neu_ind]]
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix((signed_bound_areas.ravel(1),
                                   (neu_ind, np.arange(neu_ind.size))),
                                  shape=(num_flux, num_bound))
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_flux, num_bound))

    # Dirichlet boundary conditions
    dir_ind = np.argwhere(exclude_neumann *
                          bnd.isDir[fno].astype('int64')).ravel(1)
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix((sgn[dir_ind], (dir_ind, num_neu +
                                                  np.arange(dir_ind.size))),
                                  shape=(num_pr, num_bound))
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_pr, num_bound))

    if neu_ind.size > 0 and dir_ind.size > 0:
        neu_dir_ind = sps.hstack([neu_ind, dir_ind]).A.ravel(1)
    elif neu_ind.size > 0:
        neu_dir_ind = neu_ind
    elif dir_ind.size > 0:
        neu_dir_ind = dir_ind
    else:
        raise ValueError("Boundary values should be either Dirichlet or "
                         "Neumann")

    num_subfno = np.max(subfno) + 1

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices
    bnd_2_all_hf = sps.coo_matrix((np.ones(num_bound),
                                      (np.arange(num_bound), neu_dir_ind)),
                                     shape=(num_bound, num_subfno))
    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.
    hf_2_f = sps.coo_matrix((np.ones(subfno.size), (subfno, fno)),
                                shape=(num_subfno, g.Nf))

    rhs_bound = -sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f
    return rhs_bound


if __name__ == '__main__':
    # Method used for debuging
    nx = np.array([100, 200])
    g = structured.CartGrid(nx)
    g.computeGeometry()

    kxx = np.ones(g.Nc)
    perm = second_order_tensor.SecondOrderTensor(g.dim, kxx)

    bound = bc.BoundaryCondition(g)
    mpfa(g, perm, bound)
