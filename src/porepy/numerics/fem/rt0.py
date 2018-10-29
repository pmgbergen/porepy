# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg as linalg
import logging

import porepy as pp

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

# Module-wide logger
logger = logging.getLogger(__name__)


class RT0(pp.numerics.vem.dual_elliptic.DualElliptic):
    def __init__(self, keyword):
        super(RT0, self).__init__(keyword)

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using RT0-P0 method.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse csr (g.num_faces+g_num_cells, g.num_faces+g_num_cells)
            Saddle point matrix obtained from the discretization.
        rhs: array (g.num_faces+g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        if not self._key() + "RT0_mass" in data.keys():
            self.discretize(g, data)

        M = self.assemble_matrix(g, data)

        M, bc_weight = self.assemble_neumann(g, data, M, bc_weight=True)

        return M, self.assemble_rhs(g, data, bc_weight)

    def assemble_matrix(self, g, data):
        """ Assemble VEM matrix from an existing discretization.
        """
        if not self._key() + "RT0_mass" in data.keys():
            self.discretize(g, data)

        mass = data[self._key() + "RT0_mass"]
        div = data[self._key() + "RT0_div"]
        return sps.bmat([[mass, div.T], [div, None]], format="csr")

    def discretize(self, g, data):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using RT0-P0 method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to compute the infinity norm of the matrix and use it as a
            weight to impose the boundary conditions. Default True.

        Additional return:
        weight: if bc_weight is True return the weight computed.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            mass = sps.dia_matrix(([1], 0), (g.num_faces, g.num_faces))
            data[self._key() + "RT0_mass"] = mass
            data[self._key() + "RT0_div"] = sps.csr_matrix((g.num_faces, g.num_cells))
            return
        # Retrieve the permeability, boundary conditions, and aperture
        # The aperture is needed in the hybrid-dimensional case, otherwise is
        # assumed unitary
        param = data["param"]
        k = param.get_tensor(self)
        a = param.get_aperture()

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.cg.map_grid(g)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.perm = np.delete(k.perm, (remove_dim), axis=0)
                k.perm = np.delete(k.perm, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        nodes, _, _ = sps.find(g.face_nodes)

        size_HB = g.dim * (g.dim + 1)
        HB = np.zeros((size_HB, size_HB))
        for it in np.arange(0, size_HB, g.dim):
            HB += np.diagflat(np.ones(size_HB - it), it)
        HB += HB.T
        HB /= g.dim * g.dim * (g.dim + 1) * (g.dim + 2)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            face_nodes_loc = [
                nodes[g.face_nodes.indptr[f] : g.face_nodes.indptr[f + 1]]
                for f in faces_loc
            ]
            nodes_loc = np.unique(face_nodes_loc)

            opposite_node = np.array(
                [np.setdiff1d(nodes_loc, f, assume_unique=True) for f in face_nodes_loc]
            ).flatten()

            coord_loc = node_coords[:, opposite_node]

            # Compute the H_div-mass local matrix
            A = self.massHdiv(
                a[c] * k.perm[0 : g.dim, 0 : g.dim, c],
                g.cell_volumes[c],
                coord_loc,
                sign[loc],
                g.dim,
                HB,
            )

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.tile(faces_loc, (faces_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        mass = sps.coo_matrix((dataIJ, (I, J)))
        div = -g.cell_faces.T
        data[self._key() + "RT0_mass"] = mass
        data[self._key() + "RT0_div"] = div

    def assemble_neumann(self, g, data, M, bc_weight=None):
        """ Impose Neumann boundary discretization on an already assembled
        system matrix.

        """
        # Obtain the RT0 mass matrix
        mass = data[self._key() + "RT0_mass"]
        # Use implementation in superclass
        return self._assemble_neumann_common(g, data, M, mass, bc_weight=bc_weight)

    def massHdiv(self, K, c_volume, coord, sign, dim, HB):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_volume : scalar
            Cell volume.
        sign : array (num_faces_of_cell)
            +1 or -1 if the normal is inward or outward to the cell.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name
        inv_K = np.linalg.inv(K)
        inv_K = linalg.block_diag(*([inv_K] * (dim + 1))) / c_volume

        coord = coord[0:dim, :]
        N = coord.flatten("F").reshape((-1, 1)) * np.ones((1, dim + 1)) - np.tile(
            coord, (dim + 1, 1)
        )

        C = np.diagflat(sign)

        return np.dot(C.T, np.dot(N.T, np.dot(HB, np.dot(inv_K, np.dot(N, C)))))


# ------------------------------------------------------------------------------#
