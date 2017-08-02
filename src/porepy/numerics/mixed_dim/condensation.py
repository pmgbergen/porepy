"""
The static condensation or Schur complement proceedure 
It can e.g. be used to improve condition numbers when solving linear
systems by removing the 0d fracture intersection cells.
"""
import numpy as np
import scipy.sparse as sps

from porepy.grids import grid_bucket
from porepy.params.data import Parameters


def solve_static_condensation(A, rhs, gb, dim=0, condensation_inverter=sps.linalg.inv,
                              system_inverter=sps.linalg.spsolve):
    """
    A call to this function uses a static condensation to solve a linear
    problem without the degrees of freedom related to grids of dimension dim.

    Input:
        A (sps.csr_matrix): Original matrix and right hand side of the problem to be
            solved.
        rhs (np.array): Original matrix and right hand side of the problem to be
            solved.
        dim: The dimension one wishes to get rid of. No tests for dim>0.
        condensation_inverter: The inverter of the (small) system solved 
            to perform the static condensation.
        system_inverter: Inverter for solving the problem after static 
            condensation has been performed.
    Returns:
        x: The solution vector corresponding to the initial system, i.e.,
            with all dofs.
        x_reduced: The solution vector for the reduced system, i.e.,
            corresponding to the master dofs only.
        original_to_kept_dofs: Mapping from the full to the reduced set of 
            degrees of freedom.
        eliminated_dofs: Mapping from the full to the removed set of degrees of
            freedom (i.e., which of the initial correspond to grids of dimension
            dim).
    """
    to_be_eliminated = dofs_of_dimension(gb, A, dim)

    
    a_reduced, rhs_reduced, condensation_matrix, original_to_kept_dofs, a_ss_inv \
        = eliminate_dofs(A, rhs, to_be_eliminated, condensation_inverter)

    eliminated_dofs = np.nonzero(to_be_eliminated)[0]
    x_reduced = system_inverter(a_reduced, rhs_reduced)
    x = np.zeros(A.shape[0])
    x[original_to_kept_dofs] = x_reduced
    x[to_be_eliminated] = condensation_matrix * x_reduced + a_ss_inv*rhs[to_be_eliminated]

    return x, x_reduced, original_to_kept_dofs, eliminated_dofs


def dofs_of_dimension(gb, A, dim=0):
    """
    Extracts the global dof numbers corresponding to a given dimension.
    Returns a boolean mask extracting the dofs to be eliminated.
    """
    original_ndof = A.shape[1]  # bytt ut med info fra gb
    dofs = np.empty(gb.size(), dtype=int)
    for _, d in gb:
        dofs[d['node_number']] = d['dof']
    dofs = np.r_[0, np.cumsum(dofs)]

    to_be_eliminated = np.zeros(original_ndof, dtype=bool)

    for g, d in gb:
        i = d['node_number']
        if g.dim == dim:
            to_be_eliminated[slice(dofs[i], dofs[i + 1])] = \
                np.ones(d['dof'], dtype=bool)

    return to_be_eliminated


def eliminate_dofs(A, rhs, to_be_eliminated, inverter=sps.linalg.inv):
    """
    Splits the system matrix A into four blocks according to which dofs
    are to be eliminated (the "slaves"). The right hand side is split 
    into two parts. For the computation of the blocks, the diagonal 
    part corresponding to the slaves has to be inverted, hence the option
    to choose inverter. This system will usually be quite small.

    Input:
    A, rhs: original matrix and right hand side of the problem to be
        solved.
        to_be_eliminated: boolean mask specifying which degrees of freedom 
        should be eliminated from the system.

    Returns:
        A_reduced (scipy.sparse.csr_matrix): The system matrix for the reduced system, i.e.,
            corresponding to the master dofs only. It is computed as the "master" part of 
            A minus the slave-master contribution: 
                A_reduced = A_mm - A_ms inv(A_ss) A_sm 
        rhs_reduced: The right hand side for the reduced system.
        Condensation_matrix: The matrix used for back-computation of the
            unknowns of the slaves once the reduced system has been solved.
        to_be_kept: Indices of the masters.

    """
    to_be_kept = np.invert(to_be_eliminated)
    # Get indexes of the masters:
    to_be_kept = np.nonzero(to_be_kept)[0]
    # and slaves:
    to_be_eliminated = np.nonzero(to_be_eliminated)[0]

    # Masters and slaves:
    A_mm = A[to_be_kept, :]
    A_mm = A_mm[:, to_be_kept]

    A_ms = A[:, to_be_eliminated]
    A_ms = A_ms[to_be_kept, :]

    A_sm = A[:, to_be_kept]
    A_sm = A_sm[to_be_eliminated, :]

    A_ss = A[:, to_be_eliminated]
    A_ss = A_ss[to_be_eliminated, :]
    
    A_ss_inv = inverter(A_ss)
    A_ms_A_ss_inv = A_ms * A_ss_inv
    
    # Needed for broadcasting
    if A_ss.size == 1:
        A_ms_A_ss_inv = A_ms_A_ss_inv[:, np.newaxis]

    sparse_product = sps.csr_matrix(A_ms_A_ss_inv * A_sm)

    A_reduced = A_mm - sparse_product
    rhs_reduced = rhs[to_be_kept][:, np.newaxis] - \
        A_ms_A_ss_inv * rhs[to_be_eliminated, np.newaxis]
    condensation_matrix = sps.csc_matrix(- A_ss_inv * A_sm)
    
    return A_reduced, rhs_reduced, condensation_matrix, to_be_kept, A_ss_inv


def new_coupling_fluxes(gb, node, neighbours): 
    """ 
    Adds new coupling_flux data fields to the new gb edges arising through  
    the removal of one node. 
    Idea: set up a condensation for the local system of old coupling_fluxes 
    for the removed nodes and its n_neighbours neighbour nodes/grids. 
    """ 
     
    neighbours = gb.sort_multiple_nodes(neighbours)
    # sorted from "lowest" to "highest", we want the opposite 
    neighbours = neighbours[::-1] 
    n_cells_l = node.num_cells  
    n_neighbours = len(neighbours) 

    # Initialize coupling matrix (see coupler.py, matrix_rhs)
    all_cc = np.empty((n_neighbours+1, n_neighbours+1), dtype=np.object) 
    pos_i = 0
    for ni in neighbours: 
        pos_j = 0 
        for nj in neighbours: 
            all_cc[pos_i, pos_j] = sps.coo_matrix((ni.num_cells, nj.num_cells)) 
            pos_j += 1 
         
        all_cc[pos_i, n_neighbours] = sps.coo_matrix((ni.num_cells, node.num_cells)) 
        all_cc[n_neighbours, pos_i] =sps.coo_matrix((node.num_cells, ni.num_cells)) 
        pos_i +=1
        
    all_cc[n_neighbours, n_neighbours] = sps.coo_matrix((node.num_cells,
                                                         node.num_cells)) 
    dofs = np.zeros(n_neighbours) 
    
    # Assemble original system: 
    for i in range(n_neighbours): 
        cc = gb.edge_prop((neighbours[i], node), 'coupling_discretization') 
        idx = np.ix_([i, n_neighbours], [i, n_neighbours]) 
        all_cc[idx] += cc[0]
        dofs[i] = cc[0][0][0].shape[0] 
    global_idx = np.r_[0, np.cumsum(dofs)].astype(int)
    all_cc = sps.bmat(all_cc, 'csr')
        
    # Eliminate "node"
    n_dof = all_cc.shape[0] 
    to_be_eliminated = np.zeros(n_dof, dtype=bool) 
    to_be_eliminated[range(n_dof-n_cells_l, n_dof)] = True 
    all_cc, _,_,_,_ = eliminate_dofs(all_cc, np.zeros(n_dof), to_be_eliminated)
        
    # Extract the new coupling fluxes from the eliminated system and map to faces of 
    # the first grid
    for i, n_0 in enumerate(neighbours):
        id_0 = slice(global_idx[i], global_idx[i+1])
        # Get the internal contribution (grids that have an internal hole after
        # the elimination), to be added to the node n_0. This contribution
        # is found at the off-diagonal part of the diagonal blocks
        cc_00 = all_cc.tocsr()[id_0, :].tocsc()[:,id_0]

        # Keep only one connection, the one from the "first/higher" cell(s) to the
        # "second/lower". Fluxes from higher to lower, so entries should be
        # positive (coming from off-diagonal, they are now negative)
        c_f = -np.triu(cc_00.todense(), k=1)

        # Check whether there is an internal hole in the grid. If so, add connections
        # between the cells on either side
        if not np.allclose(c_f, 0, 1e-10, 1e-12):
            cell_cells = sps.csr_matrix(c_f>0)
            # The fluxes c_f*p go from cells_1 to cells_2:
            # c_1, c_2, _ = sparse.find(cell_cells)
            gb.add_edge([n_0, n_0], cell_cells)
            d_edge = gb.edge_props([n_0, n_0])
            d_edge['coupling_flux'] = sps.csr_matrix(c_f)
            d_edge['param'] = Parameters(n_0)
            
        # Get the contribution between different grids
        for j in range(i+1, n_neighbours): 
            n_1 = neighbours[j]
            id_1 = slice(global_idx[j], global_idx[j+1])
            cc_01 = all_cc.tocsr()[id_0, :].tocsc()[:,id_1]
            gb.add_edge_prop('coupling_flux', [[n_0,n_1]], [-cc_01])
