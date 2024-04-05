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

        self.bound_stress_matrix_key: str = "bound_stress"

        self.stress_displacement_matrix_key: str = "stress"
        self.stress_rotation_matrix_key: str = "stress_rotation"
        self.stress_volumetric_strain_matrix_key: str = "pressure_stress"

        self.rotation_displacement_matrix_key: str = "rotation_displacement"
        self.rotation_diffusion_matrix_key: str = "rotation_diffusion"

        self.mass_volumetric_strain_matrix_key = "pressure_pressure"
        # TODO: Need a name for the equaiton for the (solid) pressure. Volumetric
        # compression, but in a shorter form?
        self.mass_displacement_matrix_key = "displacement_pressure"

    def discretize(self, sd: Grid, data: dict) -> None:
        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        stiffness: FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]

        bnd = parameter_dictionary["bc"]

        fi, ci, sgn = sparse_array_to_row_col_data(sd.cell_faces)

        mu = stiffness.mu[ci]
        lmbda = stiffness.lmbda[ci]

        cosserat_parameter = np.ones(sd.num_cells)[ci]

        # Normal vectors and permeability for each face (here and there side)
        n = sd.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn

        # Vector from face center to cell center
        fc_cc = sd.face_centers[::, fi] - sd.cell_centers[::, ci]
        dist_fc_cc = np.sqrt(np.sum(fc_cc**2, axis=0))

        # Distance between neighboring cells
        dist_cc_cc = np.bincount(fi, weights=dist_fc_cc, minlength=sd.num_cells)

        ## Harmonic average of the shear modulus
        #
        shear_modulus_by_face_cell_distance = mu / dist_fc_cc
        t_shear = (
            1
            / np.bincount(
                fi,
                weights=1 / shear_modulus_by_face_cell_distance,
                minlength=sd.num_faces,
            )
        )
        cosserat_parameter_by_face_cell_distance = cosserat_parameter / dist_fc_cc
        # Take the harmonic average of the Cosserat parameter
        t_cosserat = (1 / np.bincount(fi, weights=1 / cosserat_parameter_by_face_cell_distance, minlength=sd.num_faces))

        # Arithmetic average of the shear modulus. Note slight misnomer here due to the
        # factor 1 - distance (not distance).
        arithmetic_average_shear_modulus = (
            np.bincount(
                fi, weights=shear_modulus_by_face_cell_distance, minlength=sd.num_cells
            )
            * dist_cc_cc
        )

        # The vector difference operator over a face is simply a Kronecker product of
        # the standard cell-face map.
        cell_to_face_difference = sps.kron(sd.cell_faces, sps.eye(sd.dim)).tocsr()
        # The weighted average
        cell_to_face_average = (
            sps.dia_matrix(
                (1 / np.bincount(fi, shear_modulus_by_face_cell_distance), 0),
                shape=(sd.num_faces, sd.num_faces),
            )
            @ sps.coo_matrix(
                ((shear_modulus_by_face_cell_distance, (fi, ci))),
                shape=(sd.num_faces, sd.num_cells),
            ).tocsr()
        )
        cell_to_face_average_nd = sps.kron(
            cell_to_face_average,
            sps.eye(sd.dim),
        ).tocsr()

        # Complement average map, defined as 1 - the average map
        # The copies may not be needed here.
        indptr = cell_to_face_average.indptr.copy()
        indices = cell_to_face_average.indices.copy()
        data = 1 - cell_to_face_average.data.copy()
        cell_to_face_average_complement = sps.csr_matrix(
            (data, indices, indptr), shape=cell_to_face_average.shape
        )
        cell_to_face_average_complement_nd = sps.kron(
            cell_to_face_average_complement,
            sps.eye(sd.dim),
        ).tocsr()

        # Stress due to displacements
        row = fvutils.expand_indices_nd(fi, sd.dim)
        col = fvutils.expand_indices_nd(ci, sd.dim)
        # Stress is scaled by the face area.
        stress = -(
            sps.dia_matrix(
                (np.repeat(sd.face_areas, sd.dim), 0),
                shape=(sd.num_faces * sd.dim, sd.num_faces * sd.dim),
            )
            @ sps.coo_matrix(  # Note minus sign
                (2 * t_shear[np.repeat(fi, sd.dim)] * np.repeat(sgn, sd.dim), (row, col)),
                shape=(sd.num_faces * sd.dim, sd.num_cells * sd.dim),
            ).tocsr()
        )

        n = sd.face_normals

        # Indices and index pointers to make a block diagonal matrix
        indptr_csc = np.arange(0, sd.dim * sd.num_cells + 1, sd.dim)
        indices = np.arange(0, sd.dim * sd.num_faces)

        stress_volumetric_strain = (
            sps.csc_matrix(
                (
                    n[: sd.dim].ravel("F"),
                    np.arange(0, sd.dim * sd.num_faces),
                    np.arange(0, sd.dim * sd.num_faces + 1, sd.dim),
                ),
                shape=(sd.dim * sd.num_faces, sd.num_faces),
            )
            @ cell_to_face_average_complement
        )
        mass_displacement = (
            sps.csr_matrix(
                (
                    n[: sd.dim].ravel("F"),
                    np.arange(sd.dim * sd.num_faces),
                    np.arange(0, sd.dim * sd.num_faces + 1, sd.dim),
                ),
                shape=(sd.num_faces, sd.dim * sd.num_faces),
            )
            @ cell_to_face_average_nd
        )

        mass_volumetric_strain = -(
            sps.dia_matrix(
                (sd.face_areas / (2 * arithmetic_average_shear_modulus), 0),
                shape=(sd.num_faces, sd.num_faces),
            )
            @ sd.cell_faces
        )

        # Operator R_k^n
        if sd.dim == 3:
            # In this case, \hat{R}_k^n = \bar{R}_k^n is a 3x3 projection matrix that reads
            #
            #  R^n = [[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]]
            z = np.zeros(sd.num_faces)
            Rn_data = np.array([[z, -n[2], n[1]], [n[2], z, -n[0]], [-n[1], n[0], z]])

            indices = np.repeat(np.arange(0, sd.dim * sd.num_faces), sd.dim)
            indptr = np.arange(0, sd.dim**2 * sd.num_faces + 1, sd.dim)
            Rn_hat = sps.csr_matrix(
                (Rn_data.ravel("C"), indices, indptr),
                shape=(sd.dim * sd.num_faces, sd.dim * sd.num_faces),
            )

            Rn_bar = Rn_check

            stress_rotation = -Rn_check @ cell_to_face_average_complement_nd

            rotation_diffusion = -(
                sps.dia_matrix(
                    (np.repeat(sd.face_areas, sd.dim), 0),
                    shape=(sd.num_faces * sd.dim, sd.num_faces * sd.dim),
                )
                @ sps.coo_matrix(  # Note minus sign
                    (t_cosserat[np.repeat(fi, sd.dim)] * np.repeat(sgn, sd.dim), (row, col)),
                    shape=(sd.num_faces * sd.dim, sd.num_cells * sd.dim),
                    ).tocsr()
            )

        elif sd.dim == 2:
            # In this case, \hat{R}_k^n and \bar{R}_k^n differs, and read, respectively
            # \hat{R}_k^n = [[n2], [-n1]], \bar{R}_k^n = [-n2, n1].
            # We may need a ravel of sorts of the data
            data = np.array([n[1], -n[0]])

            # Mapping from average displacements over faces to rotations on the face
            Rn_bar = sps.csr_matrix(
                (
                    -data.ravel("F"),  # minus sign from definition of Rn_bar
                    np.arange(sd.dim * sd.num_faces),
                    np.arange(0, sd.dim * sd.num_faces + 1, sd.dim),
                ),
                shape=(sd.num_faces, sd.dim * sd.num_faces),
            )
            # Mapping from average rotations over faces to stresses
            Rn_hat = sps.csc_matrix(
                (
                    data.ravel("F"),
                    np.arange(sd.num_faces * sd.dim),
                    np.arange(0, sd.dim * sd.num_faces + 1, sd.dim),
                ),
                shape=(sd.dim * sd.num_faces, sd.num_faces),
            )
            # TODO: sd.cell_faces must be replaced by weighting
            stress_rotation = -Rn_hat @ cell_to_face_average_complement

            rotation_diffusion = -(
                sps.dia_matrix((
                    sd.face_areas, 0),
                    shape=(sd.num_faces, sd.num_faces),
                )
                @ sps.coo_matrix(  # Note minus sign
                    (t_cosserat[fi] * sgn, (fi, ci)),
                    shape=(sd.num_faces, sd.num_cells),
                    ).tocsr()
            )            

        rotation_displacement = -Rn_bar @ cell_to_face_average_nd

        # TODO: Cosserat model

        # Brute force way to set the Dirichlet boundary conditions (assuming the
        # displacement is zero at the boundary): The 
        dir_displacement = bnd.is_dir.ravel('f')
        dir_scalar = bnd.is_dir[0]
        #rotation_displacement[dir_scalar] = 0
        #mass_displacement[dir_scalar] = 0



        div = pp.fvutils.scalar_divergence(sd)
        div_vec = pp.fvutils.vector_divergence(sd)

        dsr = div_vec @ stress_rotation
        drs = div @ rotation_displacement

        matrix_dictionary[self.stress_displacement_matrix_key] = stress
        matrix_dictionary[self.stress_rotation_matrix_key] = stress_rotation
        matrix_dictionary[self.stress_volumetric_strain_matrix_key] = (
            stress_volumetric_strain
        )
        matrix_dictionary[self.rotation_displacement_matrix_key] = rotation_displacement
        matrix_dictionary[self.rotation_diffusion_matrix_key] = rotation_diffusion
        matrix_dictionary[self.mass_volumetric_strain_matrix_key] = (
            mass_volumetric_strain
        )
        matrix_dictionary[self.mass_displacement_matrix_key] = mass_displacement


if True:
    import porepy as pp

    g = pp.CartGrid([2, 2], [1, 2])
    g.compute_geometry()
    mu = np.ones(g.num_cells)

    C = pp.FourthOrderTensor(mu, mu)

    dir_faces = g.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(g)
    bc.is_neu[:, dir_faces] = False
    bc.is_dir[:, dir_faces] = True

    data = {
        pp.PARAMETERS: {
            "mechanics": {
                "fourth_order_tensor": C,
                "bc": bc,
            }
        },
        pp.DISCRETIZATION_MATRICES: {"mechanics": {}},
    }

    tpsa = Tpsa("mechanics")
    tpsa.discretize(g, data)

    matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["mechanics"]

    debug = []
