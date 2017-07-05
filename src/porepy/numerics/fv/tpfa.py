# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""
import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import tensor
from porepy.numerics.mixed_dim.solver import Solver
from porepy.numerics.fv import fvutils


class Tpfa(Solver):
    """ Discretize elliptic equations by a two-point flux approximation.

    Attributes:

    physics : str
        Which physics is the solver intended flow. Will determine which data
        will be accessed (e.g. flow specific, or conductivity / heat-related).
        See Data class for more details. Defaults to flow.

    """

    def __init__(self, physics='flow'):
        self.physics = physics

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

    def matrix_rhs(self, g, data, faces=None, discretize=True):
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
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): Whether to discetize prior to matrix
            assembly. If False, data should already contain discretization.
            Defaults to True.

        Return
        ------
        matrix: sparse csr (g_num_cells, g_num_cells)
            Discretization matrix.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        div = fvutils.scalar_divergence(g)
        self.discretize(g, data)
        flux = data['flux']
        M = div * flux

        bound_flux = data['bound_flux']
        param = data['param']
        bc_val = param.get_bc_val(self)
        sources = param.get_source(self)

        return M, self.rhs(g, bound_flux, bc_val, sources)

#------------------------------------------------------------------------------#

    def rhs(self, g, bound_flux, bc_val, f):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the TPFA method. See self.matrix_rhs for a detaild
        description.
        """
        if f is None:
            f = np.zeros(g.num_cells)
            warnings.warn('Scalar source not assigned, assumed null')
        div = g.cell_faces.T

        return -div * bound_flux * bc_val + f

#------------------------------------------------------------------------------#

    def discretize(self, g, data):
        """
        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise. If not given a identity permeability
            is assumed and a warning arised.
        f : array (self.g.num_cells)
            Scalar source term defined cell-wise. Given as net inn/out-flow, i.e.
            should already have been multiplied with the cell sizes. Positive
            values are considered innflow. If not given a zero source
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

        """
        param = data['param']
        k = param.get_tensor(self)
        bnd = param.get_bc(self)
        bc_val = param.get_bc_val(self)
        a = param.get_aperture()
        sources = param.get_source(self)

        trm, bound_flux = tpfa(g, k, bnd, faces=None, apertures=a)
        data['flux'] = trm
        data['bound_flux'] = bound_flux

#------------------------------------------------------------------------------#


def tpfa(g, k, bnd, faces=None, apertures=None):
    """  Discretize the second order elliptic equation using two-point flux

    The method computes fluxes over faces in terms of pressures in adjacent
    cells (defined as the two cells sharing the face).

    Parameters
        g (porepy.grids.grid.Grid): grid to be discretized
        k (porepy.params.tensor.SecondOrder) permeability tensor
        bnd (porepy.params.bc.BoundarCondition) class for boundary conditions
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
    return flux,  bound_flux
