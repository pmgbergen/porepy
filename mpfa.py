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


def mpfa(g, k, bound, faces=None, eta=0):
    
    Nc = g.Nc
    Nd = g.dim

    # Define subcell topology
    nno, cno, fno, subfno, subhfno = subcellMapping.create_mapping(g)

    # Scaling should be done here, but first in Matlab

    """
    Obtain normal_vector * k, pairings of cells and nodes (which together uniquely
    define sub-cells, and thus index for gradients, and
    """

    nk_grad, cell_node_blocks, sub_cell_index = _tensor_vector_prod(g, k, cno, fno, nno,
                                                                    subhfno)

    # Distance from cell centers to face centers, this will be the contribution from
    # gradient unknown to equatinos for pressure continuity
    pr_cont_grad = fvutils.compute_dist_face_cell(g, cno, fno, nno, subhfno, eta)

    # Make subfacen indices unique, that is, pair the indices from the two adjacent cells
    _, unique_sub_fno = np.unique(subfno, return_index=True)

    # Operator to create the pairing
    sgn = g.cellFaces[fno, cno].A
    pair_over_subfaces = sps.coo_matrix((sgn[0], (subfno, subhfno)))

    # Darcy's law
    darcy = -nk_grad[unique_sub_fno, ::]
    # Mapping from sub-faces to faces
    hf2f = sps.coo_matrix((np.ones(unique_sub_fno.size), (fno[unique_sub_fno],
                          subfno[unique_sub_fno])))

    # Pair over subfaces    
    nk_grad = pair_over_subfaces * nk_grad
    pr_cont_grad = pair_over_subfaces * pr_cont_grad

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    pr_cont_cell = sps.coo_matrix((sgn[0], (subfno, cno)))
    # Zero contribution to flux continuity
    nk_cell = sps.coo_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),
                             shape=(np.max(subfno)+1, np.max(cno)+1))
    
    # Reduce topology to one field per subface
    nno = nno[unique_sub_fno]
    fno = fno[unique_sub_fno]
    cno = cno[unique_sub_fno]
    subfno = subfno[unique_sub_fno]
    nsubfno = subfno.max()+1
    
    sgn = g.cellFaces[fno, cno].A.ravel(1)

    # Obtain mappings to exclude boundary faces
    exclude_neumann, exclude_dirichlet = _exclude_boundary_mappings(fno, nsubfno, bound)

    # No flux conditions for Dirichlet boundary faces
    nk_grad = exclude_dirichlet * nk_grad
    nk_cell = exclude_dirichlet * nk_cell
    # No pressure condition for Neumann boundary faces
    pr_cont_grad = exclude_neumann * pr_cont_grad
    pr_cont_cell = exclude_neumann * pr_cont_cell
    
    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
                sub_cell_index, cell_node_blocks, nno, exclude_dirichlet, exclude_neumann)

    grad_eqs = sps.vstack([nk_grad, pr_cont_grad])
    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    igrad = cols2blk_diag * fvutils.invert_diagonal_blocks(grad, size_of_blocks) \
            * rows2blk_diag

    rhs_cells = -sps.vstack([nk_cell, pr_cont_cell])

    flux = hf2f * darcy * igrad * rhs_cells

    ####
    # Boundary conditions
    rhs_bound = _create_bound_rhs(bound, exclude_dirichlet, exclude_neumann, fno, sgn, g,
                                  nk_cell.shape[0], pr_cont_grad.shape[0])
    # Discretization of boundary values
    bound_flux = hf2f * darcy * igrad * rhs_bound

    return flux, bound_flux


def _tensor_vector_prod(g, k, cno, fno, nno, subhfno):
    """
    Compute product of normal vectors and tensors on a sub-cell level.
    Also some topological information
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
    a pairings of node and cell indices, which together define a sub-cell
    j index of all subcells
    """
    a, blocksz = matrix_compression.rlencode(np.vstack((cno, nno)))

    nd = g.dim

    assert np.all(blocksz == nd)
    i, j = np.meshgrid(subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    j += matrix_compression.rldecode(sum_blocksz-blocksz[0], blocksz)
  
    num_nodes = np.diff(g.faceNodes.indptr)

    normals = g.faceNormals[:, fno] / num_nodes[fno]
    normals_mat = sps.coo_matrix((normals.ravel(1), (i.ravel(1), j.ravel(1))))
    k_mat = sps.coo_matrix((k.perm[::, ::, a[0]].ravel(1), (i.ravel(1), j.ravel(1))))

    nk = normals_mat.multiply(k_mat)
    j = j[::, 0::2]
    return nk, a, j


def _exclude_boundary_mappings(fno, nsubfno, bnd):
    """
    Define mappings to exclude boundary faces with dirichlet and neumann conditions

    Parameters
    ----------
    fno
    nsubfno

    Returns
    -------
    exclude_neumann: Matrix, mapping from all faces to those having flux continuity
    exclude_dirichlet: Matrix, mapping from all faces to those having pressure
    """
    # Define mappings to exclude boundary values
    j = np.argwhere([not it for it in bnd.isNeu[fno]])
    i = np.arange(j.size)
    exclude_neumann = sps.coo_matrix((np.ones(i.size), (i, j.ravel(0))),
                                     shape=(i.size, nsubfno))
    j = np.argwhere([not it for it in bnd.isDir[fno]])
    i = np.arange(j.size)
    exclude_dirichlet = sps.coo_matrix((np.ones(i.size), (i, j.ravel(0))),
                                       shape=(i.size, nsubfno))
    return exclude_neumann, exclude_dirichlet


def _block_diagonal_structure(sub_cell_index, cell_node_blocks, nno, exclude_dirichlet,
                              exclude_neumann):
    nno_flux = exclude_dirichlet * nno
    nno_pressure = exclude_neumann * nno
    node_occ = np.hstack((nno_flux, nno_pressure))
    sorted_ind = np.argsort(node_occ)
    sorted_nodes_rows = node_occ[sorted_ind]
    size_of_blocks = np.bincount(sorted_nodes_rows.astype('int64'))
    rows2blk_diag = sps.coo_matrix((np.ones(sorted_nodes_rows.size),
                                   (np.arange(sorted_ind.size), sorted_ind)))

    sorted_nodes_cols = np.argsort(cell_node_blocks[1])
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel(1) # Direction here is uncertain
    cols2blk_diag = sps.coo_matrix((np.ones(sub_cell_index.size),
                                   (subcind_nodes, np.arange(sub_cell_index.size))))
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def _create_bound_rhs(bnd, exclude_dirichlet, exclude_neumann, fno, sgn, g,
                      num_flux, num_pr):
    """
    Define rhs matrix to get basis functions that incorporates boundary conditions

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
    rhs_bound: Matrix that can be multiplied with inverse block matrix to get basis
            functions for boundary values
    """
    num_neu = sum(bnd.isNeu[fno])
    num_dir = sum(bnd.isDir[fno])
    num_bound = num_neu + num_dir

    # Neumann boundary conditions
    neu_ind = np.argwhere(exclude_dirichlet * bnd.isNeu[fno].astype('int64')).ravel(1)
    num_face_nodes = g.faceNodes.sum(axis=0).A.ravel(1)
    # sgn is already defined according to fno, while g.faceAreas is raw data,
    # and therefore needs a combined mapping
    signed_bound_areas = sgn[neu_ind] * g.faceAreas[fno[neu_ind]] / \
                       num_face_nodes[fno[neu_ind]]
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix((signed_bound_areas.ravel(1), (neu_ind, np.arange(neu_ind.size))),
                                  shape=(num_flux, num_bound))
    else:
        # Special handling when no elements are found. Not sure if this is necessary,
        # or if it is me being stupid
        neu_cell = sps.coo_matrix((num_flux, num_bound))

    # Dirichlet boundary conditions
    dir_ind = np.argwhere(exclude_neumann * bnd.isDir[fno].astype('int64')).ravel(1)
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix((sgn[dir_ind], (dir_ind, num_neu + np.arange(dir_ind.size))),
                                  shape=(num_pr, num_bound))
    else:
        dir_cell = sps.coo_matrix((num_pr, num_bound))

    rhs_bound = sps.vstack([neu_cell, dir_cell])
    return rhs_bound


if __name__ == '__main__':
    nx = np.array([2, 1])
    g = structured.CartGrid(nx)
    g.computeGeometry()
    
    kxx = np.ones(g.Nc)
    perm = second_order_tensor.SecondOrderTensor(g.dim, kxx)
    
    bound = bc.BoundaryCondition(g)
    mpfa(g, perm, bound)
