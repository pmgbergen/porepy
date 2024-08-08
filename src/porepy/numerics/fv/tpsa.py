import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils
from porepy.grids.grid import Grid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.params.tensor import FourthOrderTensor
import porepy as pp
import warnings


class Tpsa:

    def __init__(self, keyword: str) -> None:

        self.keyword: str = keyword
        """Keyword used to identify the parameter dictionary."""

        self.stress_displacement_matrix_key: str = "stress"
        """Keyword used to identify the discretization matrix for the stress generated
        by the cell center displacements. Defaults to 'stress'.
        """
        self.stress_rotation_matrix_key: str = "stress_rotation"
        """Keyword used to identify the discretization matrix for the rotational stress
        generated by the cell center rotational stress. Defaults to 'stress_rotation'."""

        self.stress_total_pressure_matrix_key: str = "stress_total_pressure"
        """Keyword used to identify the discretization matrix for the solid mass
        conservation generated by the cell center solid pressure. Defaults to 
        'stress_total_pressure'."""

        self.rotation_displacement_matrix_key: str = "rotation_displacement"
        """Keyword used to identify the discretization matrix for the rotation 
        generated by the cell center displacements. Defaults to
        'rotation_displacement'."""

        self.rotation_diffusion_matrix_key: str = "rotation_diffusion"
        """Keyword used to identify the discretization matrix for the rotational
        diffusion generated by the cell center rotational stress. Defaults to
        'rotation_diffusion'."""

        self.mass_total_pressure_matrix_key = "solid_mass_total_pressure"
        """Keyword used to identify the discretization matrix for the solid mass
        conservation generated by the cell center solid pressure. Defaults to
        'solid_mass_total_pressure'."""

        self.mass_displacement_matrix_key = "solid_mass_displacement"
        """Keyword used to identify the discretization matrix for the solid mass
        conservation generated by the cell center displacements. Defaults to
        'solid_mass_displacement'."""

        # Boundary conditions
        self.bound_stress_matrix_key: str = "bound_stress"
        self.bound_mass_displacement_matrix_key = "bound_mass_displacement"
        self.bound_rotation_diffusion_matrix_key = "bound_rotation_diffusion"
        self.bound_rotation_displacement_matrix_key = "bound_rotation_displacement"

    def discretize(self, sd: Grid, data: dict) -> None:
        """Discretize linear elasticity equation using a two-point stress approximation
        (TPSA).

        Optionally, the discretization can include microrotations, in the form of a
        Cosserat material.

        The method constructs a set of discretization matrices for the balance of linear
        and angular momentum, as well as conservation of solid mass.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in ``data[pp.PARAMETERS][self.keyword]``.
            matrix_dictionary, for storage of discretization matrices.
                Stored in ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``

        parameter_dictionary contains the entries:
            - fourth_order_tensor: ``class:~porepy.params.tensor.FourthOrderTensor``
                Stiffness tensor defined cell-wise. Note that the discretization will
                act directly on the lame parameters ``FourthOrderTensor.mu``,
                ``FourthOrderTensor.lmbda``. That is, anisotropy encoded into the
                stiffness tensor will not be considered.

            - bc: ``class:~porepy.params.bc.BoundaryConditionVectorial``
                Boundary conditions

            - cosserat_parameter (optional): np.ndarray giving the Cosserat parameter,
                which can be considered a parameter for diffusion of microrotations.
                Should have length equal to the number of cells. If not provided, the
                Cosserat parameter is set to zero.

        TOOD: Complete documentation.

        matrix_dictionary will be updated with the following entries:
            - ``stress: sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_cells)``
                stress discretization, cell center contribution
            - ``bound_flux: sps.csc_matrix (sd.dim * sd.num_faces, sd.dim *
                sd.num_faces)`` stress discretization, face contribution
            - ``bound_displacement_cell: sps.csc_matrix (sd.dim * sd.num_faces,
                                                         sd.dim * sd.num_cells)``
                Operator for reconstructing the displacement trace. Cell center
                contribution.
            - ``bound_displacement_face: sps.csc_matrix (sd.dim * sd.num_faces,
                                                         sd.dim * sd.num_faces)``
                Operator for reconstructing the displacement trace. Face contribution.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.

            data: For entries, see above.

        """

        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Bookkeeping
        nf = sd.num_faces
        nc = sd.num_cells
        nd = sd.dim

        # Fetch parameters for the mechanical behavior
        stiffness: FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]
        # The Cosserat parameter, if present. If this is None, the Cosserat parameter is
        # considered to be zero. In practice, we will set all Cosserat discretization
        # matrices to zero with no explicit computations
        cosserat_values: np.ndarray | None = parameter_dictionary.get(
            "cosserat_parameter", None
        )

        # Boundary condition object. Use the keyword 'bc' here to be compatible with the
        # implementation in mpsa.py, although symmetry with the boundary conditions for
        # rotation seems to call for a keyword like 'bc_disp'.
        bnd_disp: pp.BoundaryConditionVectorial = parameter_dictionary["bc"]

        # Boundary conditions for the rotation variable. This should only be used if
        # the Cosserat parameter is non-zero. Since the rotation variable is scalar if
        # nd == 2 and vector if nd == 3, the type of boundary condition depends on the
        # dimension.
        bnd_rot: pp.BoundaryCondition | pp.BoundaryConditionVectorial = (
            parameter_dictionary.get("bc_rot", None)
        )

        # Check that the type of boundary condition is consistent with the dimension.
        # This is a bit awkward, since it requires an if-else on the client side, but
        # the alternative is to always use a vectorial boundary condition and make a
        # hack to interpret the vectorial condition as a scalar one for 2d problems.
        # Note that, if the Cosserat parameter is zero, all of this is irrelevant.
        if nd == 2:
            if isinstance(bnd_rot, pp.BoundaryConditionVectorial):
                raise ValueError(
                    "Boundary conditions for rotations should be scalar if nd == 2"
                )
        elif nd == 3:
            if isinstance(bnd_rot, pp.BoundaryCondition):
                raise ValueError(
                    "Boundary conditions for rotations should be vectorial if nd == 3"
                )

        # Sanity check: If the Cosserat parameter is None, the boundary conditions for
        # the rotation variable are not relevant.
        if bnd_rot is not None and cosserat_values is None:
            # TODO: Should this be a warning, an error, or just ignored? The latter
            # allows for a more unified implementation of the pure elasticity and
            # Cosserat cases on the client side.
            warnings.warn(
                "Boundary conditions for rotations are only relevant if the Cosserat "
                "parameter is non-zero."
            )

        # The discretization matrices give generalized fluxes across faces in terms of
        # variables in the centers of adjacent cells. The discretization is based on a
        # two-point scheme, thus we need a mapping between cells and faces. The below
        # code generates a triplet of (face index, cell index, sign), where the sign
        # indicates the orientation of the face normal. Internal faces will occur twice,
        # with two different cell indices and opposite signs. Boundary faces will occur
        # only once. In the documentation below, we will refer to this ordering as the
        # **face-wise ordering**.
        fi, ci, sgn = sparse_array_to_row_col_data(sd.cell_faces)

        # Map the stiffness tensor to the face-wise ordering
        mu = stiffness.mu[ci]
        lmbda = stiffness.lmbda[ci]
        if cosserat_values is not None:
            cosserat_parameter = cosserat_values[ci]

        # For vector quantities, we need fi repeated nd times, do this once and for all
        # here. Also repeat the face areas and signs
        fi_nd = np.repeat(fi, nd)
        face_areas_fi_nd = sd.face_areas[fi_nd]
        sgn_nd = np.repeat(sgn, nd)

        # Expand face and cell indices to construct nd discretization matrices
        fi_expanded = fvutils.expand_indices_nd(fi, nd)
        ci_expanded = fvutils.expand_indices_nd(ci, nd)

        # Data structures for boundary conditions. Only homogeneous Dirichlet conditions
        # treated so far.
        dir_displacement = bnd_disp.is_dir.ravel("f")
        dir_scalar = bnd_disp.is_dir[0]

        # Normal vectors in the face-wise ordering
        n_fi = sd.face_normals[:, fi]
        # Switch signs where relevant
        n_fi *= sgn

        # Get a vector from cell center to face center and project to the direction of
        # the face normal. Divide by the face area to get a unit vector in the normal
        # direction.
        fc_cc = (
            n_fi
            * (sd.face_centers[::, fi] - sd.cell_centers[::, ci])
            / sd.face_areas[fi]
        )
        # Get the length of the projected vector; take the absolute value to avoid
        # negative distances.
        dist_fc_cc = np.abs(np.sum(fc_cc, axis=0))

        # For each face, the summed length of the vectors from the face center to the
        # cell centers on either side of the face. For boundary faces, there is only one
        # contribution.
        dist_cc_cc = np.bincount(fi, weights=dist_fc_cc, minlength=nc)

        # Helper function to compute the harmonic mean of a field over faces.
        def facewise_harmonic_mean(field: np.ndarray) -> np.ndarray:
            # Note that the implementation of np.bincount means the returned array is
            # ordered linearly in the face index, not in the face-wise ordering.
            return 1 / np.bincount(fi, weights=1 / field, minlength=nf)

        # Harmonic average of the shear modulus divided by the distance between the face
        # center and the cell center.
        shear_modulus_by_face_cell_distance = mu / dist_fc_cc
        t_shear = 2 * sd.face_areas * facewise_harmonic_mean(
            shear_modulus_by_face_cell_distance
        )

        # Arithmetic average of the shear modulus.
        arithmetic_average_shear_modulus = np.bincount(
            fi, weights=shear_modulus_by_face_cell_distance, minlength=nf
        )

        # Construct an averaging operator from cell centers to faces. The averaging
        # weights are the cell-wise shear modulus, divided by the distance between the
        # face center and the cell center (projected to the face normal direction). The
        # scalar cell-to-face averaging is not used directly, but will form the basis
        # for its nd equivalent (which is needed in several places), and the (scalar)
        # complement which is used in a few places.
        cell_to_face_average = (
            sps.dia_matrix(
                (1 / np.bincount(fi, shear_modulus_by_face_cell_distance), 0),
                shape=(nf, nf),
            )
            @ sps.coo_matrix(
                ((shear_modulus_by_face_cell_distance, (fi, ci))), shape=(nf, nc)
            ).tocsr()
        )


        # For Dirichlet conditions, set the averaging map to zero (as is the correct
        # discretization). TODO: Treat Neumann, and possibly Robin, conditions.
        is_dir = bnd_disp.is_dir.ravel("f")
        is_neu = bnd_disp.is_neu.ravel("f")
        is_rob = bnd_disp.is_rob.ravel("f")

        # On Dirichlet faces, the
        dir_indices = np.where(is_dir)[0]
        neu_indices = np.where(is_neu)[0]

        # dir_indices = np.where(dir_scalar)[0]
        # r, _, _ = sps.find(cell_to_face_average)
        # hit = np.in1d(r, dir_indices)
        # cell_to_face_average.data[hit] = 0

        # For vector quantities we need a nd version of the averaging operator
        cell_to_face_average_nd = sps.kron(cell_to_face_average, sps.eye(nd)).tocsr()
        # Drop zero elements to avoid mismatch between the data and the indices obtained
        # from below call to sps.find (which will only find non-zero elements).
        cell_to_face_average_nd.eliminate_zeros()
        # On rows corresponding to Dirichlet boundary conditions, set the averaging
        # operator to zero. On such faces, the displacement is known, and the average
        # displacement is not needed.
        rows, _, _ = sps.find(cell_to_face_average_nd)
        dir_rows = np.in1d(rows, dir_indices)
        cell_to_face_average_nd.data[dir_rows] = 0
        # On Neumann faces, the interior cell should be given unit weight in the average
        # map. This should be the case anyhow, since any boundary face should have a
        # single cell, but we include a check for safeguarding.
        neu_rows = np.in1d(rows, neu_indices)
        cell_to_face_average_nd.data[neu_rows] = 1
        # TODO: Robin conditions will introduce a non-zero value in the average map.

        # Complement average map, defined as 1 - the average map. Note that this will
        # have a zero value for all boundary faces (since the average map has a
        # one-sided weight of 1 here). This is corrected below.
        cell_to_face_average_complement = sps.csr_matrix(
            (
                1 - cell_to_face_average.data,
                cell_to_face_average.indices,
                cell_to_face_average.indptr,
            ),
            shape=cell_to_face_average.shape,
        )
        # Find all rows in the complement average map that correspond to boundary faces,
        # and set the data to 1. Note the contrast with the nd version, where the data
        # of the averaging map is set to zero on Dirichlet faces (thus the complement
        # map has 0 on Neumann faces), and thus a kind of filtering on the boundary
        # condition is imposed by the averaging map. Since the boundary condition is
        # given for the (vector) displacement variable, it is not possible to do a
        # similar filtering applied to a scalar variable (total pressure and, in 2d,
        # rotation). Instead, we let the complement averaging map be non-zero on all
        # boundary faces, and do the filtering explicitly (again, the original scalar
        # averaging map is not used in the discretization).
        c2f_rows, *_ = sps.find(cell_to_face_average)
        c2f_rows_is_bound = np.in1d(c2f_rows, sd.get_all_boundary_faces())
        cell_to_face_average_complement.data[c2f_rows_is_bound] = 1

        # .. as is the nd version. The data is taken from cell_to_face_average_nd, thus
        # the correct treatment of boundary conditions is already included.
        cell_to_face_average_complement_nd = sps.csr_matrix(
            (
                1 - cell_to_face_average_nd.data,
                cell_to_face_average_nd.indices,
                cell_to_face_average_nd.indptr,
            ),
            shape=cell_to_face_average_nd.shape,
        )

        dir_filter = sps.dia_matrix((dir_scalar.astype(int), 0), shape=(nf, nf))
        dir_filter_nd = sps.dia_matrix(
            (is_dir.astype(int), 0), shape=(nf * nd, nf * nd)
        )
        dir_nopass_filter = sps.dia_matrix(
            (1 - dir_scalar.astype(int), 0), shape=(nf, nf)
        )
        dir_nopass_filter_nd = sps.dia_matrix(
            (np.logical_or(is_neu, is_rob).astype(int), 0), shape=(nf * nd, nf * nd)
        )

        neu_nopass_filter_nd = sps.dia_matrix(
            (np.logical_or(is_dir, is_rob).astype(int), 0), shape=(nf * nd, nf * nd)
        )

        neu_pass_filter_nd = sps.dia_matrix(
            (is_neu.astype(int), 0), shape=(nf * nd, nf * nd)
        )


        # Finally we are ready to construct the discretization matrices.

        def vector_laplace_matrices(
            trm: np.ndarray, bnd: pp.BoundaryConditionVectorial
        ) -> tuple[sps.spmatrix, sps.spmatrix]:
            # The linear stress due to cell center displacements is computed from the
            # harmonic average of the shear modulus, scaled by the face areas. The
            # transmissibility is the same for each dimension, implying that the
            # material is in a sense isotropic.

            # Get the types of boundary conditions
            dir_faces = bnd.is_dir
            neu_faces = bnd.is_neu
            # TODO: Robin

            # Expand the discretization to vectorial form
            trm_nd = np.tile(trm, (nd, 1))

            # Data structure for the discretization of the boundary conditions
            trm_bnd = np.zeros((nd, nf))
            # On Dirichlet faces, the coefficient of the boundary condition is the
            # same as weight of the nearby cell, but with the opposite sign.
            trm_bnd[dir_faces] = -trm_nd[dir_faces]
            # On Neumann faces, the coefficient of the discretization itself is
            # zero, as the 'flux' through the boundary face is given by the boundary
            # condition.
            trm_nd[neu_faces] = 0
            # The boundary condition should simply be imposed. Put a -1 to counteract
            # the minus sign in the construction of the discretization matrix.
            trm_bnd[neu_faces] = -1

            # Discretization of the vector Laplacian. Regarding indexing,
            # the ravel gives a vector-sized array in linear ordering, which is
            # shuffled to the (vector version of the) face-wise ordering.
            # TODO: Do we need to shuffle sgn_nd?
            discr = -sps.coo_matrix(
                (
                    trm_nd.ravel("F")[fi_expanded] * sgn_nd,
                    (fi_expanded, ci_expanded),
                ),
                shape=(nf * nd, nc * nd),
            ).tocsr()

            # Boundary condition.
            bound_discr = -sps.coo_matrix(
                (
                    trm_bnd.ravel("F")[fi_expanded] * sgn_nd,
                    (fi_expanded, fi_expanded),
                ),
                shape=(nf * nd, nf * nd),
            ).tocsr()
            return discr, bound_discr

        # Discretize the stress-displacement relation
        stress, bound_stress = vector_laplace_matrices(t_shear, bnd_disp)

        # Face normals (note: in the usual ordering, not the face-wise ordering used in
        # the variable n_fi)
        n = sd.face_normals

        # The stress generated by the total pressure is computed using the complement of
        # the average map (this is just how the algebra works out), scaled with the
        # normal vector. The latter also gives the correct scaling with the face area.
        # The effect of boundary conditions are already included in
        # cell_to_face_average_complement.
        stress_total_pressure = (neu_nopass_filter_nd @ 
            sps.csc_matrix(
                (
                    n[:nd].ravel("F"),
                    np.arange(0, nd * nf),
                    np.arange(0, nd * nf + 1, nd),
                ),
                shape=(nd * nf, nf),
            )
            @ cell_to_face_average_complement
        )

        # The solid mass conservation equation is discretized by taking the average
        # displacement over the faces (not using the complement, again, this is just how
        # it is), and scaling with the normal vector. To that end, construct a sparse
        # matrix that has one normal vector per row.
        normal_vector_nd = sps.csr_matrix(
            (n[:nd].ravel("F"), np.arange(nd * nf), np.arange(0, nd * nf + 1, nd)),
            shape=(nf, nd * nf),
        )
        # The impact on the solid mass flux from the displacement is then the matrix of
        # normal vectors multiplied with the average displacement over the faces.
        # This matrix will be empty on Dirichlet faces due to the filtering in
        # cell_to_face_average_nd.
        mass_displacement = normal_vector_nd @ cell_to_face_average_nd



        # While there is no spatial operator that that relates the total pressure to the
        # conservation of solid mass in the continuous equation, the TPSA discretization
        # naturally leads to a stabilization term, as computed below. This acts on
        # differences in the total pressure, and is scaled with the face area.
        #
        # It is not fully clear what to do with this term on the boundary: There is no
        # boundary condition for the total pressure, this is in a sense inherited from
        # the displacement. The discretization scheme must however be adjusted, so that
        # it is zero on Dirichlet faces. The question is, what to do with rolling
        # boundary conditions, where a mixture of Dirichlet and Neumann conditions are
        # applied? For now, we pick the condition in the direction which is closest to
        # the normal vector of the face. While this should work nicely for domains where
        # the grid is aligned with the coordinate axis, it is more of a question mark
        # how this will work for rotated domains.
        # TODO: IMPLEMENT THIS
        mass_total_pressure = -dir_nopass_filter @ (
            sps.dia_matrix(
                (sd.face_areas / (2 * arithmetic_average_shear_modulus), 0),
                shape=(nf, nf),
            )
            @ sd.cell_faces
        )

        # Take the harmonic average of the Cosserat parameter.
        # TODO: For zero Cosserat parameters, this involves a division by zero. This
        # gives no actual problem, but filtering would have been more elegant.
        if cosserat_values is not None:
            t_cosserat = sd.face_areas * facewise_harmonic_mean(
                cosserat_parameter / dist_fc_cc
            )

        # The relations involving rotations are more cumbersome, as a rotation in 2d has
        # a single degree of freedom, while a 3d rotation has 3 degrees of freedom. This
        # necessitates (or at least is most easily realized) by a split into a 2d and a
        # 3d code. In the below if-else, we construct the matrices Rn_hat and Rn_bar
        # (see the TPSA paper for details) and use this to discretize stress generated by
        # cell center rotations. Moreover, we discretize the diffusion of rotations
        # generated by cell center displacements, which is different in 2d and 3d.
        if nd == 3:
            # In this case, \hat{R}_k^n = \bar{R}_k^n is the 3x3 projection matrix as
            # given in the TPSA paper reads
            #
            #    R^n = [[0, -n2, n0], [n2, 0, -n0], [-n1, n0, 0]]
            #
            # However, for efficient implementation we will use the function, which
            # in turns out, requires a transpose in the inner array. Quite likely this
            # could have been achieved by a different order of raveling (see below), but
            # this also worked.
            #
            # For reference, it is possible to use the following code to construct R_hat
            #
            # Rn_data = np.array([[z, -n[2], n[1]], [n[2], z, -n[0]], [-n[1], n[0], z]])
            # Rn_hat = sps.block_diag([Rn_data[:, :, i] for i in range(Rn.shape[2])])
            #
            # but this is much slower due to the block_diag construction.

            z = np.zeros(nf)
            Rn_data = np.array([[z, n[2], -n[1]], [-n[2], z, n[0]], [n[1], -n[0], z]])

            Rn_hat = pp.matrix_operations.csr_matrix_from_blocks(
                Rn_data.ravel("F"), nd, nf
            )
            Rn_bar = Rn_hat

            # Discretization of the stress generated by cell center rotations.
            stress_rotation = -neu_nopass_filter_nd @ Rn_hat @ cell_to_face_average_complement_nd

            if cosserat_values is not None:
                rotation_diffusion, bound_rotation_diffusion = vector_laplace_matrices(
                    t_cosserat, bnd_rot
                )

            else:
                # If the Cosserat parameter is zero, the diffusion operator is zero.
                rotation_diffusion = sps.csr_matrix((nf * nd, nc * nd))
                bound_rotation_diffusion = sps.csr_matrix((nf * nd, nf * nd))

        elif nd == 2:
            # In this case, \hat{R}_k^n and \bar{R}_k^n differ, and read, respectively
            #   \hat{R}_k^n = [[n2], [-n1]],
            #   \bar{R}_k^n = [-n2, n1].

            # Vector of normal vectors
            normal_vector_data = np.array([n[1], -n[0]])

            # Mapping from average displacements over faces to rotations on the face.
            # minus sign from definition of Rn_bar
            Rn_bar = sps.csr_matrix(
                (
                    -normal_vector_data.ravel("F"),
                    np.arange(nd * nf),
                    np.arange(0, nd * nf + 1, nd),
                ),
                shape=(nf, nd * nf),
            )
            # Mapping from average rotations over faces to stresses
            Rn_hat = sps.csc_matrix(
                (
                    normal_vector_data.ravel("F"),
                    np.arange(nf * nd),
                    np.arange(0, nd * nf + 1, nd),
                ),
                shape=(nd * nf, nf),
            )
            # # Discretization of the stress generated by cell center rotations.
            stress_rotation = -neu_nopass_filter_nd @ Rn_hat @ cell_to_face_average_complement

            # Diffusion operator on the rotation if relevant.
            if cosserat_values is not None:
                # In 2d, the rotation is a scalar variable and we can treat this by
                # essentially, tpfa.

                bndr_ind = sd.get_all_boundary_faces()
                t_cosserat_bnd = np.zeros(nf)
                t_cosserat_bnd[bnd_rot.is_dir] = -t_cosserat[bnd_rot.is_dir]
                # The boundary condition should simply be imposed. Put a -1 to
                # counteract the minus sign in the construction of the discretization
                # matrix.
                t_cosserat_bnd[bnd_rot.is_neu] = -1
                # t_cosserat_bnd = t_cosserat_bnd[bnd_rot.is_neu]
                t_cosserat[bnd_rot.is_neu] = 0

                # TODO: Why minus sign here, but not in tpfa?
                rotation_diffusion = -sps.coo_matrix(
                    (t_cosserat[fi] * sgn, (fi, ci)),
                    shape=(nf, nc),
                ).tocsr()

                bound_rotation_diffusion = -sps.coo_matrix(
                    (t_cosserat_bnd[fi] * sgn, (fi, fi)), shape=(nf, nf)
                ).tocsr()

            else:
                rotation_diffusion = sps.csr_matrix((nf, nc))
                bound_rotation_diffusion = sps.csr_matrix((nf, nf))

        # The rotation generated by the cell center displacements is computed from the
        # average displacement over the faces, multiplied by Rn_bar. This construction
        # is common for both 2d and 3d.
        rotation_displacement = -Rn_bar @ cell_to_face_average_nd

        # The boundary condition for the rotation equation's dependency on the
        # cell center displacements.
        
        mu_face = sps.dia_matrix((np.repeat(np.bincount(fi, weights=dist_fc_cc / (2 * mu)), nd), 0), shape=(nf * nd, nf * nd))
        # TODO: Implement Dirichlet conditions
        bound_rotation_displacement = Rn_bar @ (neu_pass_filter_nd @ mu_face - dir_filter_nd)
        
        # Boundary condition. There should be no contribution from Dofs which are
        # assigned a Dirichlet condition, so filter out these variables (this is likely
        # not fully consistent for domains with boundaries not aligned with the
        # coordinate axes, and with rolling boundary conditions, but EK does not know
        # what to do there).
        bound_mass_displacement = normal_vector_nd @ (neu_pass_filter_nd @ mu_face + dir_filter_nd)

        ## Store the computed fields

        # Discretization matrices
        matrix_dictionary[self.stress_displacement_matrix_key] = stress
        matrix_dictionary[self.stress_rotation_matrix_key] = stress_rotation
        matrix_dictionary[self.stress_total_pressure_matrix_key] = stress_total_pressure
        matrix_dictionary[self.rotation_displacement_matrix_key] = rotation_displacement
        matrix_dictionary[self.rotation_diffusion_matrix_key] = rotation_diffusion
        matrix_dictionary[self.mass_total_pressure_matrix_key] = mass_total_pressure
        matrix_dictionary[self.mass_displacement_matrix_key] = mass_displacement

        # Boundary conditions (NB: Only Dirichlet implemented for now)
        matrix_dictionary[self.bound_stress_matrix_key] = bound_stress
        matrix_dictionary[self.bound_mass_displacement_matrix_key] = (
            bound_mass_displacement
        )
        matrix_dictionary[self.bound_rotation_diffusion_matrix_key] = (
            bound_rotation_diffusion
        )
        matrix_dictionary[self.bound_rotation_displacement_matrix_key] = (
            bound_rotation_displacement
        )
