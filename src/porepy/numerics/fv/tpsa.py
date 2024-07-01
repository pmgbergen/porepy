import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils
from porepy.grids.grid import Grid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.params.tensor import FourthOrderTensor
import porepy as pp


class Tpsa:

    def __init__(self, keyword: str):

        self.keyword: str = keyword

        self.stress_displacement_matrix_key: str = "stress"
        self.stress_rotation_matrix_key: str = "stress_rotation"
        self.stress_solid_pressure_matrix_key: str = "pressure_stress"

        self.rotation_displacement_matrix_key: str = "rotation_displacement"
        self.rotation_diffusion_matrix_key: str = "rotation_diffusion"

        self.mass_solid_pressure_matrix_key = "pressure_pressure"
        # TODO: Need a name for the equaiton for the (solid) pressure. Volumetric
        # compression, but in a shorter form?
        self.mass_displacement_matrix_key = "displacement_pressure"

        # Boundary conditions
        self.bound_stress_matrix_key: str = "bound_stress"
        self.bound_mass_displacement_matrix_key = "bound_mass_displacement"
        self.bound_rotation_diffusion_matrix_key = "bound_rotation_diffusion"
        self.bound_rotation_displacement_matrix_key = "bound_rotation_displacement"

    def discretize(self, sd: Grid, data: dict) -> None:

        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        nf = sd.num_faces
        nc = sd.num_cells
        nd = sd.dim

        stiffness: FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]
        cosserat_values = parameter_dictionary["cosserat_parameter"]

        bnd = parameter_dictionary["bc"]

        fi, ci, sgn = sparse_array_to_row_col_data(sd.cell_faces)

        fi_nd = np.repeat(fi, nd)
        face_areas_fi_nd = sd.face_areas[fi_nd]

        mu = stiffness.mu[ci]
        lmbda = stiffness.lmbda[ci]
        cosserat_parameter = cosserat_values[ci]

        # Data structures for boundary conditions. Only homogeneous Dirichlet conditions
        # treated so far.
        dir_displacement = bnd.is_dir.ravel("f")
        dir_scalar = bnd.is_dir[0]

        # Normal vectors and permeability for each face (here and there side)
        n = sd.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn

        # Vector from face center to cell center
        fc_cc = (
            n * (sd.face_centers[::, fi] - sd.cell_centers[::, ci]) / sd.face_areas[fi]
        )
        dist_fc_cc = np.abs(np.sum(fc_cc, axis=0))

        # Distance between neighboring cells
        dist_cc_cc = np.bincount(fi, weights=dist_fc_cc, minlength=nc)

        def facewise_harmonic_mean(field):
            return 1 / np.bincount(fi, weights=1 / field, minlength=nf)

        ## Harmonic average of the shear modulus
        #
        shear_modulus_by_face_cell_distance = mu / dist_fc_cc
        t_shear = facewise_harmonic_mean(shear_modulus_by_face_cell_distance)
        # Take the harmonic average of the Cosserat parameter
        t_cosserat = facewise_harmonic_mean(cosserat_parameter / dist_fc_cc)

        # Arithmetic average of the shear modulus.
        arithmetic_average_shear_modulus = np.bincount(
            fi, weights=shear_modulus_by_face_cell_distance, minlength=nf
        )

        face_area_diag_matrix = sps.dia_matrix((sd.face_areas, 0), shape=(nf, nf))

        # The vector difference operator over a face is simply a Kronecker product of
        # the standard cell-face map.
        cell_to_face_difference = sps.kron(sd.cell_faces, sps.eye(nd)).tocsr()
        # The weighted average
        cell_to_face_average = (
            sps.dia_matrix(
                (1 / np.bincount(fi, shear_modulus_by_face_cell_distance), 0),
                shape=(nf, nf),
            )
            @ sps.coo_matrix(
                ((shear_modulus_by_face_cell_distance, (fi, ci))),
                shape=(nf, nc),
            ).tocsr()
        )

        # For Dirichlet conditions, set the averaging map to zero (as is the correct
        # discretization). TODO: Treat Neumann, and possibly Robin, conditions.
        dir_indices = np.where(dir_scalar)[0]
        r, _, _ = sps.find(cell_to_face_average)
        hit = np.in1d(r, dir_indices)
        cell_to_face_average.data[hit] = 0

        cell_to_face_average_nd = sps.kron(
            cell_to_face_average,
            sps.eye(nd),
        ).tocsr()

        # Complement average map, defined as 1 - the average map
        # The copies may not be needed here.
        indptr = cell_to_face_average.indptr.copy()
        indices = cell_to_face_average.indices.copy()
        vals = 1 - cell_to_face_average.data.copy()
        cell_to_face_average_complement = sps.csr_matrix(
            (vals, indices, indptr), shape=cell_to_face_average.shape
        )
        cell_to_face_average_complement_nd = sps.kron(
            cell_to_face_average_complement,
            sps.eye(nd),
        ).tocsr()

        dir_filter = sps.dia_matrix((dir_scalar.astype(int), 0), shape=(nf, nf))
        dir_filter_nd = sps.dia_matrix(
            (np.repeat(dir_scalar.astype(int), nd), 0), shape=(nf * nd, nf * nd)
        )
        dir_nopass_filter = sps.dia_matrix(
            (1 - dir_scalar.astype(int), 0), shape=(nf, nf)
        )
        dir_nopass_filter_nd = sps.dia_matrix(
            (1 - np.repeat(dir_scalar.astype(int), nd), 0), shape=(nf * nd, nf * nd)
        )

        # Stress due to displacements
        row = fvutils.expand_indices_nd(fi, nd)
        col = fvutils.expand_indices_nd(ci, nd)
        # Stress is scaled by the face area.
        stress = -(
            
            sps.coo_matrix(  # Note minus sign
                (2 * face_areas_fi_nd * t_shear[fi_nd] * np.repeat(sgn, nd), (row, col)),
                shape=(nf * nd, nc * nd),
            ).tocsr()
        )

        bound_stress = (
            -dir_filter_nd
            @ sps.coo_matrix(
                (2 * face_areas_fi_nd * t_shear[fi_nd] * np.repeat(sgn, nd), (row, row)),
                shape=(nf * nd, nf * nd),
            ).tocsr()
        )

        n = sd.face_normals

        stress_solid_pressure = (
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

        normal_vector_fat_matrix = sps.csr_matrix(
            (
                n[:nd].ravel("F"),
                np.arange(nd * nf),
                np.arange(0, nd * nf + 1, nd),
            ),
            shape=(nf, nd * nf),
        )

        mass_displacement = normal_vector_fat_matrix @ cell_to_face_average_nd
        bound_mass_displacement = dir_filter @ normal_vector_fat_matrix

        mass_solid_pressure = -dir_nopass_filter @ (
            sps.dia_matrix(
                (sd.face_areas / (2 * arithmetic_average_shear_modulus), 0),
                shape=(nf, nf),
            )
            @ sd.cell_faces
        )

        # Operator R_k^n
        if nd == 3:
            # In this case, \hat{R}_k^n = \bar{R}_k^n is the 3x3 projection matrix
            #
            #  R^n = [[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]]
            z = np.zeros(nf)
            Rn_data = np.array([[z, n[2], -n[1]], [-n[2], z, n[0]], [n[1], -n[0], z]])

            indices = np.repeat(np.arange(0, nd * nf), nd)
            indptr = np.arange(0, nd**2 * nf + 1, nd)
            #Rn_hat = sps.block_diag([Rn_data[:, :, i] for i in range(nf)], format="csr")
            Rn_hat = pp.matrix_operations.csr_matrix_from_blocks(Rn_data.ravel('F'), nd, nf)
            Rn_bar = Rn_hat

            stress_rotation = -Rn_hat @ cell_to_face_average_complement_nd

            rotation_diffusion = -(
                sps.coo_matrix(  # Note minus sign
                    (face_areas_fi_nd * t_cosserat[fi_nd] * np.repeat(sgn, nd), (row, col)),
                    shape=(nf * nd, nc * nd),
                ).tocsr()
            )
            bound_rotation_diffusion = -dir_filter_nd @ (
                 sps.coo_matrix(  # Note minus sign
                    (face_areas_fi_nd * t_cosserat[fi_nd] * np.repeat(sgn, nd), (row, row)),
                    shape=(nf * nd, nf * nd),
                ).tocsr()
            )

        elif nd == 2:
            # In this case, \hat{R}_k^n and \bar{R}_k^n differs, and read, respectively
            # \hat{R}_k^n = [[n2], [-n1]], \bar{R}_k^n = [-n2, n1].
            # We may need a ravel of sorts of the data
            normal_vector_data = np.array([n[1], -n[0]])

            # Mapping from average displacements over faces to rotations on the face
            Rn_bar = sps.csr_matrix(
                (
                    -normal_vector_data.ravel(
                        "F"
                    ),  # minus sign from definition of Rn_bar
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
            # TODO: sd.cell_faces must be replaced by weighting
            stress_rotation = -Rn_hat @ cell_to_face_average_complement

            rotation_diffusion = -(
                sps.coo_matrix(  # Note minus sign
                    (sd.face_areas[fi] * t_cosserat[fi] * sgn, (fi, ci)),
                    shape=(nf, nc),
                ).tocsr()
            )
            bound_rotation_diffusion = (
                -dir_filter
                @ sps.coo_matrix(
                    (sd.face_areas[fi] * t_cosserat[fi] * sgn, (fi, fi)),
                    shape=(nf, nf),
                ).tocsr()
            )
        rotation_displacement = -Rn_bar @ cell_to_face_average_nd

        v = np.random.rand(nc * nd)
        d = sps.dia_matrix((v, 0), shape=(nc * nd, nc * nd))

        tmp = cell_to_face_average_nd @ v
        tmp = cell_to_face_average_complement_nd @ tmp

        if nd == 2:
            bound_rotation_displacement = -dir_filter @ Rn_bar
        else:  # 3D
            bound_rotation_displacement = -dir_filter_nd @ Rn_bar

        ## Store the computed fields

        # Discretization matrices
        matrix_dictionary[self.stress_displacement_matrix_key] = stress
        matrix_dictionary[self.stress_rotation_matrix_key] = stress_rotation
        matrix_dictionary[self.stress_solid_pressure_matrix_key] = stress_solid_pressure
        matrix_dictionary[self.rotation_displacement_matrix_key] = rotation_displacement
        matrix_dictionary[self.rotation_diffusion_matrix_key] = rotation_diffusion
        matrix_dictionary[self.mass_solid_pressure_matrix_key] = mass_solid_pressure
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