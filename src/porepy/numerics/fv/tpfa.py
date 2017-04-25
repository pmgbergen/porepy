# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps

from porepy.params import bc, second_order_tensor


def tpfa(g, k, bc, faces=None):
    """  Discretize the second order elliptic equation using two-point flux

    The method computes fluxes over faces in terms of pressures in adjacent
    cells (defined as the two cells sharing the face).

    Parameters
        g (core.grids.grid): grid to be discretized
        k (core.constit.second_order_tensor): permeability tensor.
        bc (core.bc.bc): class for boundary values
        faces (np.ndarray) faces to be considered. Intended for partial
            discretization, may change in the future
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
    if faces is None:
        is_not_active = np.zeros(g.num_faces, dtype=np.bool)
    else:
        is_active = np.zeros(g.num_faces, dtype=np.bool)
        is_active[faces] = True
        is_not_active = np.logical_not(is_active)

    fi, ci, sgn = sps.find(g.cell_faces)

    # Normal vectors and permeability for each face (here and there side)
    n = g.face_normals[:, fi]
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
    t_b[bc.is_dir] = t[bc.is_dir]
    t_b[bc.is_neu] = 1
    t_b = t_b[bndr_ind]
    t[np.logical_or(bc.is_neu, is_not_active)] = 0

    # Create flux matrix
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci)))

    # Create boundary flux matrix
    bndr_sgn = (g.cell_faces[bndr_ind, :]).data
    sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
    bndr_sgn = bndr_sgn[sort_id]
    bound_flux = sps.coo_matrix((t_b * bndr_sgn, (bndr_ind, bndr_ind)),
                                (g.num_faces, g.num_faces))
    return flux, bound_flux
