# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""
import numpy as np
import scipy.sparse as sps
import logging

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)


class MVEM(pp.numerics.vem.dual_elliptic.DualElliptic):
    """
    @ALL: I have kept the inheritance from the general Solver for now, or else
    the Parameter class start making trouble. It still may be useful to have a
    parent class for all discretizations, mainly to guide the implementation of
    new methods. Opinions?

    """

    def __init__(self, keyword):
        super(MVEM, self).__init__(keyword, "MVEM")

    def discretize(self, g, data):
        """ Discretize a second order elliptic equation using a dual virtual element
        method.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to rotate 2d and 1d grids

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined
                cell-wise. This is the effective permeability, including any
                aperture scalings etc.

        matrix_dictionary will be updated with the following entries:
            mass: sps.csc_matrix (g.num_faces, g.num_faces) The mass matrix.
            div: sps.csc_matrix (g.num_cells, g.num_faces) The divergence matrix.

        Optional parameter:
        --------------------
        is_tangential: Whether the lower-dimensional permeability tensor has been
            rotated to the fracture plane. Defaults to False. Stored in the data
            dictionary.
        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # Get dictionary for discretization matrix storage
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            mass = sps.dia_matrix(([1], 0), (g.num_faces, g.num_faces))
            matrix_dictionary["mass"] = mass
            matrix_dictionary["div"] = sps.csr_matrix((g.num_faces, g.num_cells))
            return

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability
        k = parameter_dictionary["second_order_tensor"]

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        c_centers, f_normals, f_centers, R, dim, _ = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # In the virtual cell approach the cell diameters should involve the
        # apertures, however to keep consistency with the hybrid-dimensional
        # approach and with the related hypotheses we avoid.
        diams = g.cell_diameters()
        # Weight for the stabilization term
        weight = np.power(diams, 2 - g.dim)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            # Compute the H_div-mass local matrix
            A = self.massHdiv(
                k.values[0 : g.dim, 0 : g.dim, c],
                c_centers[:, c],
                g.cell_volumes[c],
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                sign[loc],
                diams[c],
                weight[c],
            )[0]

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

        matrix_dictionary["mass"] = mass
        matrix_dictionary["div"] = div

    def project_flux(self, g, u, data):
        """  Project the velocity computed with a dual vem solver to obtain a
        piecewise constant vector field, one triplet for each cell.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to rotate 2d and 1d grids

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        u : array (g.num_faces) Velocity at each face.

        Return
        ------
        P0u : ndarray (3, g.num_faces) Velocity at each cell.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        if g.dim == 0:
            return np.zeros(3).reshape((3, 1))

        # The velocity field already has permeability effects incorporated,
        # thus we assign a unit permeability to be passed to MVEM.massHdiv
        k = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        c_centers, f_normals, f_centers, R, dim, _ = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )

        # In the virtual cell approach the cell diameters should involve the
        # apertures, however to keep consistency with the hybrid-dimensional
        # approach and with the related hypotheses we avoid.
        diams = g.cell_diameters()

        P0u = np.zeros((3, g.num_cells))

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            Pi_s = self.massHdiv(
                k.values[0 : g.dim, 0 : g.dim, c],
                c_centers[:, c],
                g.cell_volumes[c],
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                sign[loc],
                diams[c],
            )[1]

            # extract the velocity for the current cell
            P0u[dim, c] = np.dot(Pi_s, u[faces_loc]) / diams[c]
            P0u[:, c] = np.dot(R.T, P0u[:, c])

        return P0u

    @staticmethod
    def massHdiv(K, c_center, c_volume, f_centers, normals, sign, diam, weight=0):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_center : array (g.dim)
            Cell center.
        c_volume : scalar
            Cell volume.
        f_centers : ndarray (g.dim, num_faces_of_cell)
            Center of the cell faces.
        normals : ndarray (g.dim, num_faces_of_cell)
            Normal of the cell faces weighted by the face areas.
        sign : array (num_faces_of_cell)
            +1 or -1 if the normal is inward or outward to the cell.
        diam : scalar
            Diameter of the cell.
        weight : scalar
            weight for the stabilization term. Optional, default = 0.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        dim = K.shape[0]
        mono = np.array(
            [lambda pt, i=i: (pt[i] - c_center[i]) / diam for i in np.arange(dim)]
        )
        grad = np.eye(dim) / diam

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T)) * c_volume

        # local matrix F
        F = np.array(
            [s * m(f) for m in mono for s, f in zip(sign, f_centers.T)]
        ).reshape((dim, -1))

        assert np.allclose(G, np.dot(F, D)), "G " + str(G) + " F*D " + str(np.dot(F, D))

        # local matrix Pi_s
        Pi_s = np.linalg.solve(G, F)
        I_Pi = np.eye(f_centers.shape[1]) - np.dot(D, Pi_s)

        # local Hdiv-mass matrix
        w = weight * np.linalg.norm(np.linalg.inv(K), np.inf)
        A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

        return A, Pi_s

    @staticmethod
    def check_conservation(g, u):
        """
        Return the local conservation of mass in the cells.
        Parameters
        ----------
        g: grid, or a subclass.
        u : array (g.num_faces) velocity at each face.
        """
        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        conservation = np.empty(g.num_cells)
        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            conservation[c] = np.sum(u[faces[loc]] * sign[loc])

        return conservation
