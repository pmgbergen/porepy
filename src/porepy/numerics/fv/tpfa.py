# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""
import copy
import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import tensor

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

from porepy.numerics.fv import fvutils
from porepy.grids.grid import Grid

#------------------------------------------------------------------------------

class TpfaMixedDim(SolverMixedDim):
    def __init__(self, physics='flow'):
        self.physics = physics

        self.discr = Tpfa(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = TpfaCoupling(self.discr)

        self.solver = Coupler(self.discr, self.coupling_conditions)

#------------------------------------------------------------------------------

class TpfaDFN(SolverMixedDim):

    def __init__(self, dim_max, physics='flow'):
        # NOTE: There is no flow along the intersections of the fractures.

        self.physics = physics
        self.dim_max = dim_max

        self.discr = Tpfa(self.physics)
        self.coupling_conditions = TpfaCouplingDFN(self.discr)

        kwargs = {"discr_ndof": self.discr.ndof,
                  "discr_fct": self.__matrix_rhs__}
        self.solver = Coupler(coupling = self.coupling_conditions, **kwargs)
        SolverMixedDim.__init__(self)

    def __matrix_rhs__(self, g, data):
        # The highest dimensional problem compute the matrix and rhs, the lower
        # dimensional problem and empty matrix. For the latter, the size of the
        # matrix is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.matrix_rhs(g, data)
        else:
            ndof = self.discr.ndof(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)

#------------------------------------------------------------------------------

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

        To set a source see the source.Integral discretization class

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
        if discretize:
            self.discretize(g, data)
        flux = data['flux']
        M = div * flux

        bound_flux = data['bound_flux']
        param = data['param']
        bc_val = param.get_bc_val(self)

        return M, self.rhs(g, bound_flux, bc_val)

#------------------------------------------------------------------------------#

    def rhs(self, g, bound_flux, bc_val):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the TPFA method. See self.matrix_rhs for a detaild
        description.
        """
        div = g.cell_faces.T
        return -div * bound_flux * bc_val

#------------------------------------------------------------------------------#

    def discretize(self, g, data, faces=None):
        """
        Discretize the second order elliptic equation using two-point flux

        The method computes fluxes over faces in terms of pressures in adjacent
        cells (defined as the two cells sharing the face).

        The name of data in the input dictionary (data) are:
        param : Parameter(Class). Contains the following parameters:
            tensor : second_order_tensor
                Permeability defined cell-wise. If not given a identity permeability
                is assumed and a warning arised.
            bc : boundary conditions (optional)
            bc_val : dictionary (optional)
                Values of the boundary conditions. The dictionary has at most the
                following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
                conditions, respectively.
            apertures : (np.ndarray) (optional) apertures of the cells for scaling of
                the face normals.

        Hidden option (intended as "advanced" option that one should normally not
        care about):
            Half transmissibility calculation according to Ivar Aavatsmark, see
            folk.uib.no/fciia/elliptisk.pdf. Activated by adding the entry
            Aavatsmark_transmissibilities: True   to the data dictionary.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        """
        param = data['param']
        k = param.get_tensor(self)
        bnd = param.get_bc(self)
        aperture = param.get_aperture()

        if g.dim == 0:
            data['flux'] = sps.csr_matrix([0])
            data['bound_flux'] = 0
            return None
        if faces is None:
            is_not_active = np.zeros(g.num_faces, dtype=np.bool)
        else:
            is_active = np.zeros(g.num_faces, dtype=np.bool)
            is_active[faces] = True

            is_not_active = np.logical_not(is_active)

        fi, ci, sgn = sps.find(g.cell_faces)

        # Normal vectors and permeability for each face (here and there side)
        if aperture is None:
            n = g.face_normals[:, fi]
        else:
            n = g.face_normals[:, fi] * aperture[ci]
        n *= sgn
        perm = k.perm[::, ::, ci]

        # Distance from face center to cell center
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

        # Transpose normal vectors to match the shape of K and multiply the two
        nk = perm * n
        nk = nk.sum(axis=1)

        if data.get('Aavatsmark_transmissibilities', False):
            # These work better in some cases (possibly if the problem is grid
            # quality rather than anisotropy?). To be explored (with care) or
            # ignored.
            dist_face_cell = np.linalg.norm(fc_cc, 2, axis=0)
            t_face = np.linalg.norm(nk, 2, axis=0)
        else:
            nk *= fc_cc
            t_face = nk.sum(axis=0)
            dist_face_cell = np.power(fc_cc, 2).sum(axis=0)


        t_face = np.divide(t_face, dist_face_cell)

        # Return harmonic average
        t = 1 / np.bincount(fi, weights=1 / t_face)

        # Save values for use in recovery of boundary face pressures
        t_full = t.copy()
        sgn_full = np.bincount(fi, sgn)

        # Move Neumann faces to Neumann transmissibility
        bndr_ind = g.get_boundary_faces()
        t_b = np.zeros(g.num_faces)
        t_b[bnd.is_dir] = -t[bnd.is_dir]
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

        data['flux'] = flux
        data['bound_flux'] = bound_flux

        # Next, construct operator to reconstruct pressure on boundaries
        # Fields for data storage
        v_cell = np.zeros(fi.size)
        v_face = np.zeros(g.num_faces)
        # On Dirichlet faces, simply recover boundary condition
        v_face[bnd.is_dir] = 1
        # On Neumann faces, the, use half-transmissibilities
        v_face[bnd.is_neu] = -1/t_full[bnd.is_neu]
        v_cell[bnd.is_neu[fi]] = 1

        bound_pressure_cell = sps.coo_matrix((v_cell, (fi, ci)),
                                             (g.num_faces, g.num_cells))
        bound_pressure_face = sps.dia_matrix((v_face, 0),
                                             (g.num_faces, g.num_faces))
        data['bound_pressure_cell'] = bound_pressure_cell
        data['bound_pressure_face'] = bound_pressure_face

#------------------------------------------------------------------------------

class TpfaCoupling(AbstractCoupling):

    def __init__(self, solver):
        self.solver = solver
        self.discr_ndof = solver.ndof

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l
        using the two-point flux approximation.

        Parameters:
            g_h and g_l: grid structures of the higher and lower dimensional
                subdomains, respectively.
            data_h and data_l: the corresponding data dictionaries. Assumed
                to contain both permeability values ('perm') and apertures
                ('apertures') for each of the cells in the grids.

        Two hidden options (intended as "advanced" options that one should
        normally not care about):
            Half transmissibility calculation according to Ivar Aavatsmark, see
            folk.uib.no/fciia/elliptisk.pdf. Activated by adding the entry
            'Aavatsmark_transmissibilities': True   to the edge data.

            Aperture correction. The face centre is moved half an aperture
            away from the fracture for the matrix side transmissibility
            calculation. Activated by adding the entry
            'aperture_correction': True   to the edge data.

        Returns:
            cc: Discretization matrices for the coupling terms assembled
                in a csc.sparse matrix.
        """

        a_l = data_l['param'].get_aperture()
        a_h = data_h['param'].get_aperture()

        # Obtain the cells and face signs of the higher dimensional grid
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        # Full cell-face map on the higher dimension
        faces, cells_h, sgn_h = sps.find(g_h.cell_faces)
        # Obtain indices to uniquify faces
        # Is this necessary, considering we do
        _, ind = np.unique(faces, return_index=True)
        sgn_h = sgn_h[ind]
        cells_h = cells_h[ind]
        # Restrict to faces on the faces
        cells_h, sgn_h = cells_h[faces_h], sgn_h[faces_h]

        # Mortar data structure.
        mg = data_edge['mortar_grid']
        mortar_size = mg.num_cells

        # Matrices for reconstruction of face pressures.
        # Contribution from cell center values
        bound_pressure_cc_h = data_h['bound_pressure_cell']
        # Contribution from boundary value
        bound_pressure_face_h = data_h['bound_pressure_face']

        # Discretization of boundary conditions
        bound_flux_h = data_h['bound_flux']

        div_h = fvutils.scalar_divergence(g_h)

        # Projection from mortar grid to upper dimension
        Pi_h_mg = mg.high_to_mortar
        # Projection from mortar grid to lower dimension
        Pi_mg_l = mg.low_to_mortar

        dof, cc = self.create_block_matrix([g_h, g_l, mg])
        # Create the block matrix for the contributions

        k = 2.*data_edge['kn']
        assert k.size == mortar_size
        # The aperture is a diagonal matrix in the lower dimensional grid,
        # then mapped to the mortar grid.
        kappa = sps.diags(k)

        face_areas_h = sps.diags(g_h.face_areas)

        mortar_div = np.sign((Pi_h_mg * div_h.T).A.sum(axis=1))
        mortar_div_mat = sps.diags(mortar_div)

        # Contribution from mortar variable to conservation in the higher domain
        # TODO: Check scaling
        # Acts as a boundary condition, treat with standard boundary discretization
        cc[0, 2] = div_h *  bound_flux_h * face_areas_h *  Pi_h_mg.T
        # Acts as a source term.
        cc[1, 2] = -Pi_mg_l.T * sps.diags(mg.cell_volumes)

        # Governing equation for the inter-dimensional flux law.
        # Equation on the form
        #   \lambda = \kappa (p_h - p_l), with

        # Pressure is an intensive property
        Pi_h_mg_pressure = Pi_h_mg.copy()
        row_sum = Pi_h_mg_pressure.sum(axis=1).A.ravel()
        Pi_h_mg_pressure = sps.diags(1./row_sum) * Pi_h_mg_pressure

        # The trace of the pressure from the higher dimension is composed of the cell center pressure,
        # and a contribution from the boundary flux, represented by the mortar flux
        # Cell center contribution, mapped to the mortar grid
        cc[2, 0] = kappa * Pi_h_mg_pressure * bound_pressure_cc_h
        # Contribution from mortar
        # Should we have Pi_h_mg_pressure here?
        cc[2, 2] += kappa * Pi_h_mg_pressure * bound_pressure_face_h * face_areas_h * Pi_h_mg.T

        # Contribution from the lower dimensional pressure
        cc[2, 1] = -kappa * Pi_mg_l

        # Contribution from the \lambda term, moved to the right hand side
        cc[2, 2] -= sps.diags(1./mg.cell_volumes)
#        data_edge['coupling_flux'] = sps.hstack([cells2faces * cc[0, 0],
#                                                 cells2faces * cc[0, 1]])
#        data_edge['coupling_discretization'] = cc

        return cc

#------------------------------------------------------------------------------#

class TpfaCouplingDFN(AbstractCoupling):

    def __init__(self, solver):
        self.solver = solver

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l
        using the two-point flux approximation.

        Parameters:
            g_h and g_l: grid structures of the higher and lower dimensional
                subdomains, respectively.
            data_h and data_l: the corresponding data dictionaries. Assumed
                to contain both permeability values ('perm') and apertures
                ('apertures') for each of the cells in the grids.

        Returns:
            cc: Discretization matrices for the coupling terms assembled
                in a csc.sparse matrix.
        """

        k_h = data_h['param'].get_tensor(self.solver)
        a_h = data_h['param'].get_aperture()

        dof = np.array([self.solver.ndof(g_h), self.solver.ndof(g_l)])

        # Obtain the cells and face signs of the higher dimensional grid
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, cells_h, sgn_h = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sgn_h = sgn_h[ind]
        cells_h = cells_h[ind]

        cells_h, sgn_h = cells_h[faces_h], sgn_h[faces_h]

        # The procedure for obtaining the face transmissibilities of the higher
        # grid is analougous to the one used in the discretize function of the
        # Tpfa class.
        n = g_h.face_normals[:, faces_h]
        n *= sgn_h
        perm_h = k_h.perm[:, :, cells_h]
        fc_cc_h = g_h.face_centers[::, faces_h] - g_h.cell_centers[::, cells_h]

        nk_h = perm_h * n
        nk_h = nk_h.sum(axis=1)
        if data_edge.get('Aavatsmark_transmissibilities', False):
            dist_face_cell_h = np.linalg.norm(fc_cc_h, 2, axis=0)
            t_face_h = np.linalg.norm(nk_h, 2, axis=0)
        else:
            nk_h *= fc_cc_h
            t_face_h = nk_h.sum(axis=0)
            dist_face_cell_h = np.power(fc_cc_h, 2).sum(axis=0)

        # Account for the apertures
        t_face_h = np.multiply(t_face_h, a_h[cells_h])
        t = np.divide(t_face_h, dist_face_cell_h)

        # Create the block matrix for the contributions
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof]
                      ).reshape((2, 2))

        # Compute the off-diagonal terms
        dataIJ, I, J = -t, cells_l, cells_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        # Compute the diagonal terms
        dataIJ, I, J = t, cells_h, cells_h
        cc[0, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))
        I, J = cells_l, cells_l
        cc[1, 1] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[1]))

        # Save the flux discretization for back-computation of fluxes
        cells2faces = sps.csr_matrix((sgn_h, (faces_h, cells_h)),
                                     (g_h.num_faces, g_h.num_cells))

        data_edge['coupling_flux'] = sps.hstack([cells2faces * cc[0, 0],
                                                 cells2faces * cc[0, 1]])
        data_edge['coupling_discretization'] = cc

        return cc

#------------------------------------------------------------------------------#
