"""The module contains an implementation of the finite volume two-point flux
approximation scheme. The implementation resides in the class Tpfa.

"""
from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


class Tpfa(pp.FVElliptic):
    """Discretize elliptic equations by a two-point flux approximation.

    Attributes:

    keyword : str
        Which keyword is the solver intended flow. Will determine which data
        will be accessed (e.g. flow specific, or conductivity / heat-related).

    """

    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
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
                for instance apertures should be incorporated before calling
                this function.
            bc : (BoundaryCondition) boundary conditions
            ambient_dimension: (int) Optional. Ambient dimension, used in the
                discretization of vector source terms. Defaults to the dimension of the
                grid.

        matrix_dictionary will be updated with the following entries:
            flux: sps.csc_matrix (sd.num_faces, sd.num_cells)
                flux discretization, cell center contribution
            bound_flux: sps.csc_matrix (sd.num_faces, sd.num_faces)
                flux discretization, face contribution
            bound_pressure_cell: sps.csc_matrix (sd.num_faces, sd.num_cells)
                Operator for reconstructing the pressure trace. Cell center contribution
            bound_pressure_face: sps.csc_matrix (sd.num_faces, sd.num_faces)
                Operator for reconstructing the pressure trace. Face contribution
            vector_source: sps.csc_matrix (sd.num_faces)
                discretization of flux due to vector source, e.g. gravity. Face
                contribution. Active only if vector_source = True, and only for 1D.

        Hidden option (intended as "advanced" option that one should normally not
        care about):
            Half transmissibility calculation according to Ivar Aavatsmark, see
            folk.uib.no/fciia/elliptisk.pdf. Activated by adding the entry
            Aavatsmark_transmissibilities: True   to the data dictionary.

        Parameters:
            sd: Grid with geometry fields computed.
            data: For entries, see above.

        """
        # Get the dictionaries for storage of data and discretization matrices
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Ambient dimension of the grid
        vector_source_dim: int = parameter_dictionary.get("ambient_dimension", sd.dim)

        if sd.dim == 0:
            # Shortcut for 0d grids
            matrix_dictionary[self.flux_matrix_key] = sps.csr_matrix((0, sd.num_cells))
            matrix_dictionary[self.bound_flux_matrix_key] = sps.csr_matrix((0, 0))
            matrix_dictionary[self.bound_pressure_cell_matrix_key] = sps.csr_matrix(
                (0, sd.num_cells)
            )
            matrix_dictionary[self.bound_pressure_face_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            matrix_dictionary[self.vector_source_matrix_key] = sps.csr_matrix(
                (0, sd.num_cells * max(vector_source_dim, 1))
            )
            matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ] = sps.csr_matrix((0, sd.num_cells * max(vector_source_dim, 1)))
            return None

        # Extract parameters
        k = parameter_dictionary["second_order_tensor"]
        bnd = parameter_dictionary["bc"]

        fi_g, ci_g, sgn_g = sparse_array_to_row_col_data(sd.cell_faces)

        # fi_g and ci_g now defines the geometric (grid) mapping from subfaces to cells.
        # The cell with index ci_g[i] has the face with index fi_g[i].
        # In addition to the geometric mappings, we need to add connections between
        # cells and faces over the periodic boundary.
        # The periodic boundary is defined by a mapping from left faces to right
        # faces:
        if hasattr(sd, "periodic_face_map"):
            fi_left = sd.periodic_face_map[0]
            fi_right = sd.periodic_face_map[1]
        else:
            fi_left = np.array([], dtype=int)
            fi_right = np.array([], dtype=int)
        # We find the left(right)_face -> left(right)_cell mapping
        left_sfi, ci_left, left_sgn = sparse_array_to_row_col_data(
            sd.cell_faces[fi_left]
        )
        right_sfi, ci_right, right_sgn = sparse_array_to_row_col_data(
            sd.cell_faces[fi_right]
        )

        # Sort subface indices to not lose left to right periodic mapping
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
        # Next, we add connection between the left cells and right faces (and vice
        # versa). The flux over the periodic boundary face is defined equivalently to
        # the flux over an internal face: flux_left = T_left * (p_left - p_right). The
        # term T_left * p_left is already included in fi_g and ci_g, but we need to add
        # the second term T_left * (-p_right). Equivalently for flux_right. f_mat and
        # c_mat defines the indices of these entries in the flux matrix.
        fi_periodic = np.hstack((fi_g, fi_left, fi_right))
        ci_periodic = np.hstack((ci_g, ci_right, ci_left))
        sgn_periodic = np.hstack((sgn_g, -left_sgn, -right_sgn))

        # When calculating the subface transmissibilities, left cells should be mapped
        # to left faces, while right cells should be mapped to right faces.
        fi = np.hstack((fi_g, fi_right, fi_left))
        ci = np.hstack((ci_g, ci_right, ci_left))
        sgn = np.hstack((sgn_g, right_sgn, left_sgn))

        # Normal vectors and permeability for each face (here and there side)
        n = sd.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn
        perm = k.values[::, ::, ci]

        # Distance from face center to cell center
        fc_cc = sd.face_centers[::, fi] - sd.cell_centers[::, ci]

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
        bndr_ind = sd.get_all_boundary_faces()
        t_b = np.zeros(sd.num_faces)
        t_b[is_dir] = -t[is_dir]
        t_b[is_neu] = 1
        t_b = t_b[bndr_ind]
        t[is_neu] = 0

        # Create flux matrix
        flux = sps.coo_matrix(
            (t[fi_periodic] * sgn_periodic, (fi_periodic, ci_periodic))
        ).tocsr()

        # Create boundary flux matrix
        bndr_sgn = (sd.cell_faces[bndr_ind, :]).data
        sort_id = np.argsort(sd.cell_faces[bndr_ind, :].indices)
        bndr_sgn = bndr_sgn[sort_id]
        bound_flux = sps.coo_matrix(
            (t_b * bndr_sgn, (bndr_ind, bndr_ind)), (sd.num_faces, sd.num_faces)
        ).tocsr()

        # Store the matrix in the right dictionary:
        matrix_dictionary[self.flux_matrix_key] = flux
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux

        # Next, construct operator to reconstruct pressure on boundaries
        # Fields for data storage
        v_cell = np.zeros(fi.size)
        v_face = np.zeros(sd.num_faces)
        # On Dirichlet faces, simply recover boundary condition
        v_face[bnd.is_dir] = 1
        # On Neumann faces, use half-transmissibilities
        v_face[bnd.is_neu] = -1 / t_full[bnd.is_neu]
        v_cell[bnd.is_neu[fi]] = 1

        bound_pressure_cell = sps.coo_matrix(
            (v_cell, (fi, ci)), (sd.num_faces, sd.num_cells)
        ).tocsr()
        bound_pressure_face = sps.dia_matrix(
            (v_face, 0), (sd.num_faces, sd.num_faces)
        ).tocsr()
        matrix_dictionary[self.bound_pressure_cell_matrix_key] = bound_pressure_cell
        matrix_dictionary[self.bound_pressure_face_matrix_key] = bound_pressure_face

        # Discretization of vector source, e.g. gravity in Darcy's law.
        # Use harmonic average of cell transmissibilities

        # The discretization involves the transmissibilities, multiplied with the
        # distance between cell and face centers, and with the sgn adjustment (or else
        # the vector source will point in the wrong direction in certain cases). See
        # Starnoni et al. 2020, WRR for details.
        vals = (t[fi_periodic] * fc_cc * sgn_periodic)[:vector_source_dim].ravel("f")

        # Rows and cols are given by fi / ci, expanded to account for the vector source
        # having multiple dimensions.
        rows = np.tile(fi_periodic, (vector_source_dim, 1)).ravel("F")
        cols = pp.fvutils.expand_indices_nd(ci_periodic, vector_source_dim)

        vector_source = sps.coo_matrix((vals, (rows, cols))).tocsr()

        matrix_dictionary[self.vector_source_matrix_key] = vector_source

        # Gravity contribution to pressure reconstruction.
        # The pressure difference is computed as the dot product between the vector
        # source and the distance vector from cell to face centers.
        vals = np.zeros((vector_source_dim, fi.size))
        vals[:, bnd.is_neu[fi]] = fc_cc[:vector_source_dim, bnd.is_neu[fi]]
        bound_pressure_vector_source = sps.coo_matrix(
            (vals.ravel("f"), (rows, cols))
        ).tocsr()
        matrix_dictionary[
            self.bound_pressure_vector_source_matrix_key
        ] = bound_pressure_vector_source


class DifferentiableTpfa:
    """Helper class for constructing differentiable operators for the TPFA scheme.

    This is really a container class for methods that are needed to construct the
    discretization. The actual discretization is considered a constitutive law, and can
    be found in that part of the code.

    """

    def external_boundary_filters(
        self,
        mdg: pp.MixedDimensionalGrid,
        subdomains: Sequence[pp.Grid],
        boundary_grids: Sequence[pp.BoundaryGrid],
        name: str,
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        """Filters for Dirichlet and Neumann boundary conditions.

        Parameters:
            mdg: Mixed-dimensional grid.
            subdomains: List of grids.
            boundary_grids: List of boundary grids.
            name: Name of the operator.

        Returns:
            Tuple of Dirichlet and Neumann boundary filters.

        """
        dir_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_dir"), domains=boundary_grids
        )
        neu_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_neu"), domains=boundary_grids
        )
        proj = pp.ad.BoundaryProjection(mdg, subdomains, dim=1).boundary_to_subdomain
        return proj @ dir_filter, proj @ neu_filter

    def internal_boundary_filter(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """Construct a dense array that filters out everything but internal boundaries.

        Parameters:
            subdomains: List of grids.

        Returns:
            DenseArray representation of the internal boundary filter.

        """
        is_internal = []
        for sd in subdomains:
            is_internal.append(sd.tags["fracture_faces"])
        if len(is_internal) == 0:
            fracture_faces = np.array([], dtype=int)
        else:
            fracture_faces = np.hstack(is_internal)

        return pp.wrap_as_dense_ad_array(
            fracture_faces, name="internal_boundary_filter"
        )

    def tip_filter(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """Construct a dense array that filters out everything but faces on immersed
        tips.

        Parameters:
            subdomains: List of grids.

        Returns:
            DenseArray representation of the internal boundary filter.

        """
        is_tip = []
        for sd in subdomains:
            # Find faces that are tagged as tip faces, but exclude those that are also
            # tagged as domain boundary faces.
            is_tip.append(
                np.logical_and(
                    sd.tags["tip_faces"],
                    np.logical_not(sd.tags["domain_boundary_faces"]),
                )
            )
        if len(is_tip) == 0:
            tip_faces = np.array([], dtype=int)
        else:
            tip_faces = np.hstack(is_tip)

        return pp.wrap_as_dense_ad_array(tip_faces, name="tip_filter")

    def _block_diagonal_grid_property_matrix(
        self,
        domains: list[pp.Grid],
        grid_property_getter: Callable[[pp.Grid], Any],
    ) -> sps.spmatrix:
        """Construct mapping matrix for the connectivity between two grids entities.

        The mapping matrix is a block diagonal matrix where each block contains 1 where
        the two entities are connected and 0 otherwise.

        Parameters:
            domains: List of grids.
            grid_property_getter: Function that returns the property of the grid that
                should be used for the mapping.

        Returns:
            Mapping matrix.
        """
        if len(domains) == 0:
            return sps.csr_matrix((0, 0))

        blocks = []
        for g in domains:
            mat_loc = grid_property_getter(g)
            blocks.append(mat_loc)

        block_matrix = pp.matrix_operations.optimized_compressed_storage(
            sps.block_diag(blocks)
        )
        return block_matrix

    def half_face_map(
        self,
        subdomains: list[pp.Grid],
        from_entity: Literal["cells", "faces", "half_faces"] = "half_faces",
        to_entity: Literal["cells", "faces", "half_faces"] = "half_faces",
        dimensions: tuple[int, int] = (1, 1),
        with_sign: bool = False,
    ) -> sps.spmatrix:
        """Mapping between half-faces and cells or faces.

        Parameters:
            subdomains: List of grids.
            from_entity: Entity to map from.
            to_entity: Entity to map to.
            dimensions: Dimensions of the to and from entities. EK: Verify order
            with_sign: Whether to include sign in the mapping.

        Returns:
            spmatrix ``(num_{from_entity} * dimensions[0],
                        num_{to_entity} * dimensions[1])``:
                Mapping matrix.

        """

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            fi, ci, sgn = sps.find(g.cell_faces)

            indices = []
            sizes = []
            for name in [to_entity, from_entity]:
                if name == "cells":
                    indices.append(ci)
                    sizes.append(g.num_cells)
                elif name == "faces":
                    indices.append(fi)
                    sizes.append(g.num_faces)
                elif name == "half_faces":
                    indices.append(np.arange(fi.size))
                    sizes.append(fi.size)
                else:
                    raise ValueError(f"Unknown entity {name}.")

            # If the dimensions of the from and to entities are unequal, we need to
            # repeat the indices of the lowest dimension to match the highest dimension.
            # Find the ratio between the dimensions to know how many times to repeat.
            # Ceil is used to treat the lower of the two dimensions as 1.
            repeat_row_inds = np.ceil(dimensions[1] / dimensions[0]).astype(int)
            repeat_col_inds = np.ceil(dimensions[0] / dimensions[1]).astype(int)

            # Check that the dimensions are compatible. One of the dimensions must be a
            # multiple of the other.
            assert (
                dimensions[0] % dimensions[1] == 0 or dimensions[1] % dimensions[0] == 0
            )
            # Expand the indices, with the following logic (for rows): First repeat the
            # row indices to match the size of columns (the from entity). Then expand
            # these indices to account for the dimenison of the to entity
            # (dimensions[0]). The latter assigns rows[indices[0]:indices[0] +
            # dimensions[0]] to the first face etc. The logic for the columns is the
            # same.
            rows = pp.fvutils.expand_indices_nd(
                np.repeat(indices[0], repeat_row_inds), dimensions[0]
            )
            cols = pp.fvutils.expand_indices_nd(
                np.repeat(indices[1], repeat_col_inds), dimensions[1]
            )

            if with_sign:
                # The sign must be repeated as many times as the row and column indices
                # were repeated and expanded. This will equal the maximum of the two
                # dimensions.
                vals = np.repeat(sgn, max(dimensions))
            else:
                vals = np.ones(cols.size)
            mat = sps.csr_matrix(
                (vals, (rows, cols)),
                shape=(
                    sizes[0] * dimensions[0],
                    sizes[1] * dimensions[1],
                ),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )

    def _cell_face_vectors(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Distance between face centers and cell centers.

        Parameters:
            subdomains: List of grids.

        Returns:
            Distance matrix. See documentation of method half_face_geometry_matrices
                for details.

        """

        vec_dim = 3

        def get_c_f_vec_matrix(g: pp.Grid) -> sps.csr_matrix:
            """Construct matrix of vectors connecting cell centers and face centers.

            Parameters:
                g: Grid.

            Returns:
                spmatrix ``(num_half_faces, num_half_faces * vec_dim)``:
                    Matrix of vectors connecting cell centers and face centers.

            """

            # Find the cell and face indices for each half-face, identified by the
            # elements in the cell_faces, that is, the divergence matrix.
            fi, ci, _ = sps.find(g.cell_faces)
            num_hf = fi.size

            # Construct vectors from cell centers to face centers.
            fc_cc = g.face_centers[:, fi] - g.cell_centers[:, ci]

            # Each row contains vec_dim entries and corresponds to one half-face.
            row_inds = np.repeat(np.arange(num_hf), vec_dim)
            # There are num_hf * vec_dim columns, each vec_dim-long block corresponding
            # to one half-face.
            col_inds = pp.fvutils.expand_indices_nd(np.arange(num_hf), vec_dim)
            # Fortran order to get the first vec_dim entries to correspond to the first
            # half-face, etc.
            vals = fc_cc.ravel("F")
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)), shape=(num_hf, num_hf * vec_dim)
            )
            return mat

        dist_vec = self._block_diagonal_grid_property_matrix(
            subdomains,
            get_c_f_vec_matrix,
        )

        return dist_vec

    def _normal_vectors(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Normal vectors on half-faces, repeated for each dimension.

        Parameters:
            subdomains: List of grids.

        Returns:
            Matrix of normal vectors. See documentation of function
                half_face_geometry_matrices for details.

        """
        vector_dim = 3

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            """Construct normal vector matrix. Each vector is repeated vector_dim times.

            Half-face i corresponds to rows
                vector_dim * i:vector_dim * (i+1)
            and contains n_0^i, n_1^i, n_2^i, with subscripts indicating x, y and z
            components of the normal vector. The column indices makes sure we hit the
            right permeability entries. The permeability being a tensor_dim * num_cells
            vector, we expand the cell indices to tensor_dim indices.

            Parameters:
                g: Grid.

            Returns:
                spmatrix ``(num_half_faces * vector_dim, num_cells * tensor_dim)``:
                    Normal vector matrix.

            """
            # Bookkeeping
            tensor_dim = vector_dim**2
            fi, ci, sgn = sps.find(g.cell_faces)
            num_hf = fi.size
            n = g.face_normals
            # There are vector_dim * num_hf rows (each half-face has vector_dim rows
            # associated with it, to accommodate one vector per half face). Each row
            # again has vector_dim entries to fit one normal vector (this is achieved by
            # the call to repeat).
            row_inds = np.repeat(np.arange(num_hf * vector_dim), vector_dim)
            # The columns must be expanded to tensor_dim to hit the right permeability
            # entries (ex: cell 0 has permeability entries 0, 1, 2, 3, 4, 5, 6, 7, 8,
            # so the indices belonging to cell 1 start at 9, etc.). This is achieved by
            # the call to expand_indices_nd.
            col_inds = pp.fvutils.expand_indices_nd(ci, tensor_dim)
            # Make vector_dim copies of each normal vector, ravelling in Fortran order
            # to get the right order of elements.
            repeat_fi = np.repeat(fi, vector_dim)
            vals = n[:, repeat_fi].ravel("F")
            # Construct the matrix.
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)),
                shape=(num_hf * vector_dim, g.num_cells * tensor_dim),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )

    def _cell_face_distances(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Scalar distance between face centers and cell centers for each half face.

        Parameters:
            subdomains: List of grids.

        Returns:
            Array of distances. See documentation of function
                half_face_geometry_matrices for details.

        """
        if len(subdomains) == 0:
            return np.array([])

        vals = []
        for g in subdomains:
            fi, ci, _ = sps.find(g.cell_faces)
            fc_cc = g.face_centers[:, fi] - g.cell_centers[:, ci]
            vals.append(np.power(fc_cc, 2).sum(axis=0))
        return np.hstack(vals)

    def half_face_geometry_matrices(
        self,
        subdomains: list[pp.Grid],
    ) -> tuple[np.ndarray, sps.spmatrix, np.ndarray]:
        """Get matrices describing the geometry in terms of half-faces.

        Parameters:
            subdomains: List of grids.

        Returns:
            Tuple:
                n ``shape=(3 x num_half_faces,9 x num_half_faces)``:
                    Normal vector on half-faces.
                d_vec: ``shape=(num_half_faces, 3 x num_half_faces)``:
                    Vectors between cell centers and face centers.
                dist, ``shape=(num_half_faces,)``:
                    Distance between cell centers and face centers.

                See documentation in the implementation for details.

        """
        # Half-face transmissibilities are computed as
        # t_hf = d_vec @ n @ k_hf / dist
        # k_hf: Permeability on half-faces shape=(9 x num_half_faces,)
        # n: Normal vector on half-faces shape=(3 x num_half_faces,9 x num_half_faces)
        # d_vec: Vectors cell centers and face centers shape=(num_half_faces,
        # 3 x num_half_faces)
        # dist: Distance between cell centers and face centers shape=(num_half_faces,)

        # Normal vectors, each row contains a normal vector. Matrix shape =
        # (3 * n_hf, 9 * n_hf). Multiplying k_c by this ammounts to half-face-wise
        # inner product between the normal vector and the diffusivity tensor, with each
        # resulting vector having shape (3,), hence the 3 in the first dimension.
        n = self._normal_vectors(subdomains)

        # Face-cell vectors, each row contains a distance vector for a half-face.
        # Matrix shape = (n_hf, 3 * n_hf). Multiplying (n @ k_c) by this matrix ammounts
        # to half-face-wise inner products as above.
        d_vec = self._cell_face_vectors(subdomains)

        # Face-cell distances, scalar for each half-face
        dist = self._cell_face_distances(subdomains)

        return n, d_vec, dist

    def face_pairing_from_cell_array(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Mapping from cell to face pairs.

        Intended usage: Given a vector of cell values of the potential u, multiply with
        this matrix to get u_l - u_r on faces. Includes the "half" pairing on
        boundaries.

        Parameters:
            subdomains: List of grids.

        Returns:
            Mapping matrix.

        """
        # Construct difference operator to get p_l - p_r on faces. First map p to half-
        # faces, then to faces with the signed matrix.
        c_to_hf_mat = self.half_face_map(
            subdomains, to_entity="half_faces", from_entity="cells"
        )
        hf_to_f_mat = self.half_face_map(subdomains, to_entity="faces", with_sign=True)
        return hf_to_f_mat @ c_to_hf_mat

    def boundary_sign(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Get operator defining the sign of boundary faces.

        Parameters:
            subdomains: List of grids.

        Returns:
            Array of signs.

        """
        bnd_sgn: list[np.ndarray] = []
        for sd in subdomains:
            # Signs on all half faces. fi will contain the indices of all internal faces
            # twice (one for each side).
            fi, _, sgn = sps.find(sd.cell_faces)
            # Obtain a map to uniquify the face indices. This will also sort the indices.
            _, fi_ind = np.unique(fi, return_index=True)
            # Get the unique signs, ordered according to the unique (sorted) face
            # indices.
            sgn_unique = sgn[fi_ind]
            # Interior faces are those that are not domain boundary faces or fracture
            # faces.
            is_int = np.logical_not(
                np.logical_or(
                    sd.tags["domain_boundary_faces"], sd.tags["fracture_faces"]
                )
            )
            # Set signs on interior faces to zero.
            sgn_unique[is_int] = 0
            bnd_sgn.append(sgn_unique)

        return pp.wrap_as_dense_ad_array(np.hstack(bnd_sgn), name="boundary_sign")

    def nd_to_3d(
        self,
        subdomains: list[pp.Grid],
        nd: int,
        entity: Literal["cells", "faces", "nodes"] = "cells",
    ) -> sps.spmatrix:
        """Expand a vector from nd to 3d.

        Intended usage: Expand vector source, defined as a nd vector (because mpfa
        discretization is performed in nd), to a 3d vector (because tpfa, specifically
        this version, is implemented in 3d).

        Parameters:
            subdomains: List of grids.
            nd: Dimension of the vector source.
            entity: Entity to expand.

        Returns:
            Mapping matrix.

        """

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            num = getattr(g, f"num_{entity}")
            rows = np.concatenate([np.arange(i, num * 3, 3) for i in range(nd)])
            cols = np.concatenate([np.arange(i, num * nd, nd) for i in range(nd)])
            vals = np.ones(cols.size)
            mat = sps.csr_matrix(
                (vals, (rows, cols)),
                shape=(num * 3, num * nd),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )
