# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""
import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import second_order_tensor
from porepy.numerics.mixed_dim.solver import Solver


class Tpfa(Solver):

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data, faces=None):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a two-point flux approximation.
        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise. If not given a identity permeability
            is assumed and a warning arised.
        f : array (self.g.num_cells)
            Scalar source term defined cell-wise. If not given a zero source
            term is assumed and a warning arised.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse csr (g_num_cells, g_num_cells)
            Discretization matrix.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        k, bnd, bc_val, a, f = data.get('k'), data.get(
            'bc'), data.get('bc_val'), data.get('apertures'), data.get('f')
        if k is None:
            kxx = np.ones(g.num_cells)
            k = second_order_tensor.SecondOrderTensor(g.dim, kxx)
            warnings.warn('Permeability not assigned, assumed identity')

        trm, bound_flux = tpfa(g, k, bnd, faces, apertures=a)
        div = g.cell_faces.T
        M = div * trm

        return M, self.rhs(g, bound_flux, bc_val, f)

#------------------------------------------------------------------------------#

    def rhs(self, g, bound_flux, bc_val, f):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the TPFA method. See self.matrix_rhs for a detailed
        description.
        """
        if f is None:
            f = np.zeros(g.num_cells)
            warnings.warn('Scalar source not assigned, assumed null')

        div = g.cell_faces.T
        return div * bound_flux * bc_val + f * g.cell_volumes


#------------------------------------------------------------------------------#

def tpfa(g, k, bnd, faces=None, apertures=None):
    """  Discretize the second order elliptic equation using two-point flux

    The method computes fluxes over faces in terms of pressures in adjacent
    cells (defined as the two cells sharing the face).

    Parameters
        g (core.grids.grid): grid to be discretized
        k (core.constit.second_order_tensor): permeability tensor.
        bc (core.bc.bc): class for boundary values
        faces (np.ndarray) faces to be considered. Intended for partial
            discretization, may change in the future
        apertures (np.ndarray) apertures of the cells for scaling of the face
        normals.
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

    """
    if g.dim == 0:
        return sps.csr_matrix([0]), 0
    if faces is None:
        is_not_active = np.zeros(g.num_faces, dtype=np.bool)
    else:
        is_active = np.zeros(g.num_faces, dtype=np.bool)
        is_active[faces] = True

        is_not_active = np.logical_not(is_active)

    fi, ci, sgn = sps.find(g.cell_faces)

    # Normal vectors and permeability for each face (here and there side)
    if apertures is None:
        n = g.face_normals[:, fi]
    else:
        n = g.face_normals[:, fi] * apertures[ci]
    n *= sgn
    perm = k.perm[::, ::, ci]

    # Distance from face center to cell center
    fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

    # Transpose normal vectors to match the shape of K

    nk = perm * n
    nk = nk.sum(axis=0)
    nk *= fc_cc
    t_face = nk.sum(axis=0)
    # print(t_face, 'tface')
    dist_face_cell = np.power(fc_cc, 2).sum(axis=0)

    t_face = np.divide(t_face, dist_face_cell)

    # Return harmonic average
    t = 1 / np.bincount(fi, weights=1 / t_face)

    # Move Neumann faces to Neumann transmissibility
    bndr_ind = g.get_boundary_faces()
    t_b = np.zeros(g.num_faces)
    t_b[bnd.is_dir] = t[bnd.is_dir]
    t_b[bnd.is_neu] = 1
    t_b = t_b[bndr_ind]
    t[np.logical_or(bnd.is_neu, is_not_active)] = 0

    # Create flux matrix
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci)))

    # Create boundary flux matrix
    bndr_sgn = (g.cell_faces[bndr_ind, :]).data
    sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
    bndr_sgn = bndr_sgn[sort_id]
    bound_flux = sps.coo_matrix((t_b * bndr_sgn, (bndr_ind, bndr_ind)),
                                (g.num_faces, g.num_faces))
    return flux, bound_flux
