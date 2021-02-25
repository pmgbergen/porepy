# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""
import logging
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["numerics", "discretization", "assembly"]


class MVEM(pp.numerics.vem.dual_elliptic.DualElliptic):
    """
    @ALL: I have kept the inheritance from the general Solver for now, or else
    the Parameter class start making trouble. It still may be useful to have a
    parent class for all discretizations, mainly to guide the implementation of
    new methods. Opinions?

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword: str) -> None:
        super(MVEM, self).__init__(keyword, "MVEM")

    @pp.time_logger(sections=module_sections)
    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Discretize a second order elliptic equation using a dual virtual element
        method.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d and 1d grids

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
            matrix_dictionary[self.mass_matrix_key] = mass
            matrix_dictionary[self.div_matrix_key] = sps.csr_matrix(
                (g.num_faces, g.num_cells)
            )
            matrix_dictionary[self.vector_proj_key] = sps.csr_matrix((3, g.num_cells))
            return

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability
        k = parameter_dictionary["second_order_tensor"]
        # Identity tensor for vector source computation
        identity = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))

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
        size_A = np.sum(np.square(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1]))
        rows_A = np.empty(size_A, dtype=int)
        cols_A = np.empty(size_A, dtype=int)
        data_A = np.empty(size_A)
        idx_A = 0

        # Allocate the data to store matrix P entries
        size_P = 3 * np.sum(g.cell_faces.indptr[1:] - g.cell_faces.indptr[:-1])
        rows_P = np.empty(size_P, dtype=int)
        cols_P = np.empty(size_P, dtype=int)
        data_P = np.empty(size_P)
        idx_P = 0
        idx_row_P = 0

        # define the function to compute the inverse of the permeability matrix
        if g.dim == 1:
            inv_matrix = self._inv_matrix_1d
        elif g.dim == 2:
            inv_matrix = self._inv_matrix_2d
        elif g.dim == 3:
            inv_matrix = self._inv_matrix_3d

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            # Compute the H_div-mass local matrix
            A = self.massHdiv(
                k.values[0 : g.dim, 0 : g.dim, c],
                inv_matrix(k.values[0 : g.dim, 0 : g.dim, c]),
                c_centers[:, c],
                g.cell_volumes[c],
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                sign[loc],
                diams[c],
                weight[c],
            )[0]

            # Compute the flux reconstruction matrix
            P = np.zeros((3, faces_loc.size))
            P[dim, :] = self.massHdiv(
                identity.values[0 : g.dim, 0 : g.dim, c],
                identity.values[0 : g.dim, 0 : g.dim, c],
                c_centers[:, c],
                g.cell_volumes[c],
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                sign[loc],
                diams[c],
            )[1]
            P = np.dot(R.T, P) / diams[c]

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.concatenate(faces_loc.size * [[faces_loc]])
            loc_idx = slice(idx_A, idx_A + A.size)
            rows_A[loc_idx] = cols.T.ravel()
            cols_A[loc_idx] = cols.ravel()
            data_A[loc_idx] = A.ravel()
            idx_A += A.size

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx_P, idx_P + P.size)
            cols_P[loc_idx] = np.concatenate(3 * [[faces_loc]]).ravel()
            rows_P[loc_idx] = np.repeat(np.arange(3), faces_loc.size) + idx_row_P
            data_P[loc_idx] = P.ravel()
            idx_P += P.size
            idx_row_P += 3

        # Construct the global matrices
        mass = sps.coo_matrix((data_A, (rows_A, cols_A)))
        div = -g.cell_faces.T
        proj = sps.coo_matrix((data_P, (rows_P, cols_P)))

        matrix_dictionary[self.mass_matrix_key] = mass
        matrix_dictionary[self.div_matrix_key] = div
        matrix_dictionary[self.vector_proj_key] = proj

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def massHdiv(
        K: np.ndarray,
        inv_K: np.ndarray,
        c_center: np.ndarray,
        c_volume: float,
        f_centers: np.ndarray,
        normals: np.ndarray,
        sign: np.ndarray,
        diam: float,
        weight: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the local mass Hdiv matrix using the mixed vem approach.

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
        w = weight * np.linalg.norm(inv_K, np.inf)
        A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

        return A, Pi_s

    @staticmethod
    @pp.time_logger(sections=module_sections)
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
