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
from porepy.numerics.mixed_dim.solver import Solver
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling
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
        if discretize:
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
            source : array (self.g.num_cells)
                Scalar source term defined cell-wise. Given as net inn/out-flow, i.e.
                should already have been multiplied with the cell sizes. Positive
                values are considered innflow. If not given a zero source
                term is assumed and a warning arised.
            bc : boundary conditions (optional)
            bc_val : dictionary (optional)
                Values of the boundary conditions. The dictionary has at most the
                following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
                conditions, respectively.
            apertures : (np.ndarray) (optional) apertures of the cells for scaling of
                the face normals.

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


#------------------------------------------------------------------------------

class TpfaMultiDim():
    def __init__(self, physics='flow'):
        self.physics = physics
        discr = Tpfa(self.physics)
        coupling_conditions = TpfaCoupling(discr)
        self.solver = Coupler(discr, coupling_conditions)

    def matrix_rhs(self, gb):
        return self.solver.matrix_rhs(gb)

    def split(self, gb, names, var):
        return self.solver.split(gb, names, var)

    def compute_discharges(self, gb):
        """
        Computes discharges over all faces in the entire grid bucket given
        pressures for all nodes, provided as node properties.

        Parameter:
            gb: grid bucket with the following data fields for all nodes/grids:
                    'flux': Internal discretization of fluxes.
                    'bound_flux': Discretization of boundary fluxes.
                    'p': Pressure values for each cell of the grid.
                    'bc_val': Boundary condition values.
                and the following edge property field for all connected grids:
                    'coupling_flux': Discretization of the coupling fluxes.
        Returns:
            gb, the same grid bucket with the added field 'discharge' added to all
            node data fields. Note that the fluxes between grids will be added doubly,
            both to the data corresponding to the higher dimensional grid and as a
            edge property.
        """
        gb.add_node_props(['discharge'])

        for gr, da in gb:
            if gr.dim > 0:
                f, _, s = sps.find(gr.cell_faces)
                _, ind = np.unique(f, return_index=True)
                s = s[ind]
                da['discharge'] = (da['flux'] * da['p']
                                   + da['bound_flux'] * da['bc_val'])

        gb.add_edge_prop('discharge')
        for e, data in gb.edges_props():
            g1, g2 = gb.sorted_nodes_of_edge(e)
            if data['face_cells'] is not None:
                coupling_flux = gb.edge_prop(e, 'coupling_flux')[0]
                pressures = gb.nodes_prop([g2, g1], 'p')
                coupling_contribution = coupling_flux * \
                    np.concatenate(pressures)
                flux2 = coupling_contribution + gb.node_prop(g2, 'discharge')
                data2 = gb.node_props(g2)
                data2['discharge'] = copy.deepcopy(flux2)
                data['discharge'] = copy.deepcopy(flux2)

        return gb


#------------------------------------------------------------------------------


class TpfaCoupling(AbstractCoupling):

    def __init__(self, solver):
        self.solver = solver

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l
        using the two-printoint flux approximation.

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

        k_l = data_l['param'].get_tensor(self.solver)
        k_h = data_h['param'].get_tensor(self.solver)
        a_l = data_l['param'].get_aperture()
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
        # grid is analougous to the one used in numerics.fv.tpfa.py, see that file
        # for explanations
        n = g_h.face_normals[:, faces_h]
        n *= sgn_h
        perm_h = k_h.perm[:, :, cells_h]

        fc_cc_h = g_h.face_centers[::, faces_h] - g_h.cell_centers[::, cells_h]
        nk_h = perm_h * n

        nk_h = nk_h.sum(axis=0)
        nk_h *= fc_cc_h
        t_face_h = nk_h.sum(axis=0)

        # Account for the apertures
        t_face_h = t_face_h * a_h[cells_h]
        dist_face_cell_h = np.power(fc_cc_h, 2).sum(axis=0)
        t_face_h = np.divide(t_face_h, dist_face_cell_h)

        # For the lower dimension some simplifications can be made, due to the
        # alignment of the face normals and (normal) permeabilities of the
        # cells. First, the normal component of the permeability of the lower
        # dimensional cells must be found. While not provided in g_l, the
        # normal of these faces is the same as that of the corresponding higher
        # dimensional face, up to a sign.
        n1 = n[np.newaxis, :, :]
        n2 = n[:, np.newaxis, :]
        n1n2 = n1 * n2

        normal_perm = np.einsum(
            'ij...,ij...', n1n2, k_l.perm[:, :, cells_l])
        # The area has been multiplied in twice, not once as above, through n1
        # and n2
        normal_perm = np.divide(normal_perm, g_h.face_areas[faces_h])

        # Account for aperture contribution to face area
        t_face_l = a_h[cells_h] * normal_perm

        # And use it for face-center cell-center distance
        t_face_l = np.divide(
            t_face_l, 0.5 * np.divide(a_l[cells_l], a_h[cells_h]))

        # Assemble face transmissibilities for the two dimensions and compute
        # harmonic average
        t_face = np.array([t_face_h, t_face_l])
        t = t_face.prod(axis=0) / t_face.sum(axis=0)

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

        return cc

#------------------------------------------------------------------------------#
