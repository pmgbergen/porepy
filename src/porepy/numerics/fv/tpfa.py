# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:09:29 2016

@author: keile
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class Tpfa(pp.FVElliptic):
    """Discretize elliptic equations by a two-point flux approximation.

    Attributes:

    keyword : str
        Which keyword is the solver intended flow. Will determine which data
        will be accessed (e.g. flow specific, or conductivity / heat-related).
        See Data class for more details. Defaults to flow.

    """

    def __init__(self, keyword):
        super(Tpfa, self).__init__(keyword)

    def discretize(self, g, data):
        """
        Discretize the second order elliptic equation using two-point flux approximation.

        The method computes fluxes over faces in terms of pressures in adjacent
        cells (defined as the two cells sharing the face).

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            second_order_tensor : (SecondOrderTensor) Permeability defined
                cell-wise. This is the effective permeability, any scaling by
                for intance apertures should be incorporated before calling
                this function.
            bc : (BoundaryCondition) boundary conditions
            ambient_dimension: (int) Optional. Ambient dimension, used in the
                discretization of vector source terms. Defaults to the dimension of the
                grid.

        matrix_dictionary will be updated with the following entries:
            flux: sps.csc_matrix (g.num_faces, g.num_cells)
                flux discretization, cell center contribution
            bound_flux: sps.csc_matrix (g.num_faces, g.num_faces)
                flux discretization, face contribution
            bound_pressure_cell: sps.csc_matrix (g.num_faces, g.num_cells)
                Operator for reconstructing the pressure trace. Cell center contribution
            bound_pressure_face: sps.csc_matrix (g.num_faces, g.num_faces)
                Operator for reconstructing the pressure trace. Face contribution
            vector_source: sps.csc_matrix (g.num_faces)
                discretization of flux due to vector source, e.g. gravity. Face contribution.
                Active only if vector_source = True, and only for 1D.

        Hidden option (intended as "advanced" option that one should normally not
        care about):
            Half transmissibility calculation according to Ivar Aavatsmark, see
            folk.uib.no/fciia/elliptisk.pdf. Activated by adding the entry
            Aavatsmark_transmissibilities: True   to the data dictionary.

        Parameters
        ----------
        g (pp.Grid): grid, or a subclass, with geometry fields computed.
        data (dict): For entries, see above.
        """
        # Get the dictionaries for storage of data and discretization matrices
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Ambient dimension of the grid
        vector_source_dim: int = parameter_dictionary.get("ambient_dimension", g.dim)

        if g.dim == 0:
            # Short cut for 0d grids
            matrix_dictionary[self.flux_matrix_key] = sps.csr_matrix((0, g.num_cells))
            matrix_dictionary[self.bound_flux_matrix_key] = sps.csr_matrix((0, 0))
            matrix_dictionary[self.bound_pressure_cell_matrix_key] = sps.csr_matrix(
                (0, g.num_cells)
            )
            matrix_dictionary[self.bound_pressure_face_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            matrix_dictionary[self.vector_source_matrix_key] = sps.csr_matrix(
                (0, g.num_cells * max(vector_source_dim, 1))
            )
            matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ] = sps.csr_matrix((0, g.num_cells * max(vector_source_dim, 1)))
            return None

        # Extract parameters
        k = parameter_dictionary["second_order_tensor"]
        bnd = parameter_dictionary["bc"]

        fi_g, ci_g, sgn_g = sps.find(g.cell_faces)

        # fi_g and ci_g now defines the geometric (grid) mapping from subfaces to cells.
        # The cell with index ci_g[i] has the face with index fi_g[i].
        # In addition to the geometric mappings, we need to add connections between
        # cells and faces over the periodic boundary.
        # The periodic boundary is defined by a mapping from left faces to right
        # faces:
        if hasattr(g, "periodic_face_map"):
            fi_left = g.periodic_face_map[0]
            fi_right = g.periodic_face_map[1]
        else:
            fi_left = np.array([], dtype=int)
            fi_right = np.array([], dtype=int)
        # We find the left(right)_face -> left(right)_cell mapping
        left_sfi, ci_left, left_sgn = sps.find(g.cell_faces[fi_left])
        right_sfi, ci_right, right_sgn = sps.find(g.cell_faces[fi_right])

        # Sort subface indices to not loose left to right periodic mapping
        # I.e., fi_left[i] maps to fi_right[i]
        I_left = np.argsort(left_sfi)
        I_right = np.argsort(right_sfi)
        if not (
            np.array_equal(left_sfi[I_left], np.arange(fi_left.size))
            and np.array_equal(right_sfi[I_right], np.arange(fi_right.size))
        ):
            raise RuntimeError("Could not find correct periodic boundary mapping")
        ci_left = ci_left[I_left]
        ci_right = ci_right[I_right]
        # Now, ci_left gives the cell indices of the left cells, and ci_right gives
        # the indices of the right cells. Further, fi_left gives the face indices of the
        # left faces that is periodic with the faces with indices fi_right. This means
        # that ci_left[i] is connected to ci_right[i] over the face fi_left (face of
        # ci_left[i]) and fi_right[i] (face of ci_right[i]).
        #
        # Next, we add connection between the left cells and right faces (and vice versa).
        # The flux over the periodic boundary face is defined equivalently to the
        # flux over an internal face: flux_left = T_left * (p_left - p_right).
        # The term T_left * p_left is already included in fi_g and ci_g, but we need
        # to add the second term T_left * (-p_right). Equivalently for flux_right.
        # f_mat and c_mat defines the indices of these entries in the flux matrix.
        fi_periodic = np.hstack((fi_g, fi_left, fi_right))
        ci_periodic = np.hstack((ci_g, ci_right, ci_left))
        sgn_periodic = np.hstack((sgn_g, -left_sgn, -right_sgn))

        # When calculating the subface transmissibilities, left cells should be mapped
        # to left faces, while right cells should be mapped to right faces.
        fi = np.hstack((fi_g, fi_right, fi_left))
        ci = np.hstack((ci_g, ci_right, ci_left))
        sgn = np.hstack((sgn_g, right_sgn, left_sgn))

        # Normal vectors and permeability for each face (here and there side)
        n = g.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn
        perm = k.values[::, ::, ci]

        # Distance from face center to cell center
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

        # Transpose normal vectors to match the shape of K and multiply the two
        nk = perm * n
        nk = nk.sum(axis=1)

        if data.get("Aavatsmark_transmissibilities", False):
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

        # Return harmonic average. Note that we here use fi_mat to count indices.
        t = 1 / np.bincount(fi_periodic, weights=1 / t_face)

        # Save values for use in recovery of boundary face pressures
        t_full = t.copy()

        # For primal-like discretizations like the TPFA, internal boundaries
        # are handled by assigning Neumann conditions.
        is_dir = np.logical_and(bnd.is_dir, np.logical_not(bnd.is_internal))
        is_neu = np.logical_or(bnd.is_neu, bnd.is_internal)

        # Move Neumann faces to Neumann transmissibility
        bndr_ind = g.get_all_boundary_faces()
        t_b = np.zeros(g.num_faces)
        t_b[is_dir] = -t[is_dir]
        t_b[is_neu] = 1
        t_b = t_b[bndr_ind]
        t[is_neu] = 0

        # Create flux matrix
        flux = sps.coo_matrix(
            (t[fi_periodic] * sgn_periodic, (fi_periodic, ci_periodic))
        ).tocsr()

        # Create boundary flux matrix
        bndr_sgn = (g.cell_faces[bndr_ind, :]).data
        sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
        bndr_sgn = bndr_sgn[sort_id]
        bound_flux = sps.coo_matrix(
            (t_b * bndr_sgn, (bndr_ind, bndr_ind)), (g.num_faces, g.num_faces)
        ).tocsr()

        # Store the matrix in the right dictionary:
        matrix_dictionary[self.flux_matrix_key] = flux
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux

        # Next, construct operator to reconstruct pressure on boundaries
        # Fields for data storage
        v_cell = np.zeros(fi.size)
        v_face = np.zeros(g.num_faces)
        # On Dirichlet faces, simply recover boundary condition
        v_face[bnd.is_dir] = 1
        # On Neumann faces, the, use half-transmissibilities
        v_face[bnd.is_neu] = -1 / t_full[bnd.is_neu]
        v_cell[bnd.is_neu[fi]] = 1

        bound_pressure_cell = sps.coo_matrix(
            (v_cell, (fi, ci)), (g.num_faces, g.num_cells)
        ).tocsr()
        bound_pressure_face = sps.dia_matrix(
            (v_face, 0), (g.num_faces, g.num_faces)
        ).tocsr()
        matrix_dictionary[self.bound_pressure_cell_matrix_key] = bound_pressure_cell
        matrix_dictionary[self.bound_pressure_face_matrix_key] = bound_pressure_face

        # Discretization of vector source
        # e.g. gravity in Darcy's law
        # Use harmonic average of cell transmissibilities

        # The discretization involves the transmissibilities, multiplied with the
        # distance between cell and face centers, and with the sgn adjustment (or else)
        # the vector source will point in the wrong direction in certain cases.
        # See Starnoni et al 2020, WRR for details.
        vals = (t[fi_periodic] * fc_cc * sgn_periodic)[:vector_source_dim].ravel("f")

        # Rows and cols are given by fi / ci, expanded to account for the vector source
        # having multiple dimensions.
        rows = np.tile(fi_periodic, (vector_source_dim, 1)).ravel("f")
        cols = pp.fvutils.expand_indices_nd(ci_periodic, vector_source_dim)

        vector_source = sps.coo_matrix((vals, (rows, cols))).tocsr()

        matrix_dictionary[self.vector_source_matrix_key] = vector_source

        # Gravity contribution to pressure reconstruction
        # The pressure difference is computed as the dot product between the
        # vector source and the distance vector from cell to face centers.
        vals = np.zeros((vector_source_dim, fi.size))
        vals[:, bnd.is_neu[fi]] = fc_cc[:vector_source_dim, bnd.is_neu[fi]]
        bound_pressure_vector_source = sps.coo_matrix(
            (vals.ravel("f"), (rows, cols))
        ).tocsr()
        matrix_dictionary[
            self.bound_pressure_vector_source_matrix_key
        ] = bound_pressure_vector_source
