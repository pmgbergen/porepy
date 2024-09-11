import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils
from porepy.grids.grid import Grid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.params.tensor import FourthOrderTensor


class Tpsa:

    def __init__(self, keyword: str):

        self.keyword: str = keyword

        self.stress_matrix_key: str = "stress"
        self.bound_stress_matrix_key: str = "bound_stress"

        self.rotation_stress_matrix_key: str = "rotation_stress"
        self.stress_rotation_matrix_key: str = "stress_rotation"
        self.stress_pressure_matrix_key: str = "pressure_stress"
        self.rotation_displacement_matrix_key: str = "rotation_displacement"
        self.pressure_pressure_matrix_key = "pressure_pressure"
        # TODO: Need a name for the equaiton for the (solid) pressure. Volumetric
        # compression, but in a shorter form?
        self.displacement_displacement_matrix_key = "displacement_displacement"
        self.displacement_pressure_matrix_key = "displacement_pressure"

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

        # Normal vectors and permeability for each face (here and there side)
        n = sd.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn

        # Vector from face center to cell center
        fc_cc = sd.face_centers[::, fi] - sd.cell_centers[::, ci]
        dist_fc_cc = np.sqrt(np.sum(fc_cc**2, axis=0))

        # Distance between neighboring cells
        dist_cc_cc = np.bincount(ci, weights=dist_fc_cc, minlength=sd.num_cells)

        ## Harmonic average of the shear modulus
        #
        shear_modulus_by_face_cell_distance = mu / dist_fc_cc
        t = 1 / np.bincount(
            fi, weights=1 / shear_modulus_by_face_cell_distance, minlength=sd.num_faces
        )

        # Arithmetic average of the shear modulus
        arithmetic_average_shear_modulus = np.bincount(
            ci, weights=shear_modulus_by_face_cell_distance, minlength=sd.num_cells
        ) / np.bincount(ci, weights=dist_fc_cc, minlength=sd.num_cells)

        # Stress due to displacements
        row = fvutils.expand_indices_nd(fi, g.dim)
        col = fvutils.expand_indices_nd(ci, g.dim)
        data = np.repeat(t, g.dim)

        stress = sps.coo_matrix(
            (t[np.repeat(fi, sd.dim)], (row, col)), shape=(sd.num_faces * sd.dim, g.num_cells * sd.dim)
        ).tocsr()

        # The vector difference operator over a face is simply a Kronecker product of
        # the standard cell-face map.
        cell_to_face_difference = sps.kron(sd.cell_faces, sps.eye(sd.dim)).tocsr()
        # The weighted average
        cell_to_face_average = sps.kron(
            sps.coo_matrix(
                ((shear_modulus_by_face_cell_distance, (fi, ci))),
                shape=(sd.num_faces, sd.num_cells),
            ).tocsr(),
            sps.eye(sd.dim),
        ).tocsr()

        n = sd.face_normals

        # Indices and index pointers to make a block diagonal matrix
        indptr_csc = np.arange(0, sd.dim * sd.num_cells + 1, sd.dim)
        indices = np.arange(0, sd.dim * sd.num_faces)

        stress_pressure = sps.csc_matrix(
            (n[:sd.dim].ravel("F"), indices, indptr_csc), shape=(sd.dim * sd.num_faces, sd.num_cells)
        )
        pressure_displacement = sps.csc_matrix(
            (n[:sd.dim].ravel("F"), indices, indptr_csc), shape=(sd.dim * sd.num_faces, sd.num_cells)
        )

        pressure_pressure = (
            sps.dia_matrix(
                (arithmetic_average_shear_modulus, 0),
                shape=(sd.num_faces, sd.num_faces),
            )
            @ sd.cell_faces
        )

        indptr_csr = np.arange(0, sd.dim * sd.num_faces + 1, sd.dim)
        indices_csr = np.arange(0, sd.num_cells)

        # Operator R_k^n
        if sd.dim == 3:
            # In this case, \hat{R}_k^n = \bar{R}_k^n is a 3x3 projection matrix that reads
            #
            #  R^n = [[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]]
            z = np.zeros(sd.num_faces)
            Rn_data = np.array([[z, -n[2], n[1]], [n[2], z, -n[0]], [-n[1], n[0], z]])

            indices = np.repeat(np.arange(0, sd.dim * sd.num_faces), sd.dim)
            indptr = np.arange(0, sd.dim**2 * sd.num_faces + 1, sd.dim)
            Rn_check = sps.csr_matrix((Rn_data.ravel("C"), indices, indptr), shape=(sd.dim * sd.num_faces, sd.dim * sd.num_faces))

            Rn_bar = Rn_check

            stress_rotation = -Rn_check @ cell_to_face_average

        elif sd.dim == 2:
            # In this case, \hat{R}_k^n and \bar{R}_k^n differs, and read, respectively
            # \hat{R}_k^n = [[n2], [-n1]], \bar{R}_k^n = [-n2, n1].
            # We may need a ravel of sorts of the data
            data = np.array([n[1], -n[0]])
            # Not sure about the arguments here, but the logic should be okay. The csr
            # and csc are random (50% chance it is correct).

            # Mapping from average displacements over faces to rotations on the face
            Rn_bar = sps.csr_matrix(
                (data.ravel('F'), np.arange(sd.dim * sd.num_faces), np.arange(0, sd.dim * sd.num_faces + 1, sd.dim)), shape=(sd.num_faces, sd.dim * sd.num_faces)
            )
            # Mapping from average rotations over faces to stresses
            Rn_check = sps.csc_matrix(
                (data.ravel('F'), np.repeat(np.arange(sd.num_faces), sd.dim), np.arange(0, sd.dim * sd.num_faces + 1, sd.dim)), shape=(sd.dim * sd.num_faces, g.num_faces)
            )

            stress_rotation = -Rn_check @ g.cell_faces
        rotation_displacement = -Rn_bar @ cell_to_face_average

        # TODO: Cosserat model

        matrix_dictionary[self.stress_matrix_key] = stress
        matrix_dictionary[self.stress_rotation_matrix_key] = stress_rotation
        matrix_dictionary[self.stress_pressure_matrix_key] = stress_pressure
        matrix_dictionary[self.rotation_displacement_matrix_key] = rotation_displacement
        matrix_dictionary[self.pressure_pressure_matrix_key] = pressure_pressure
        matrix_dictionary[self.displacement_pressure_matrix_key] = pressure_displacement


import porepy as pp

g = pp.CartGrid([3, 3, 3])
g.compute_geometry()
mu = np.ones(g.num_cells)

C = pp.FourthOrderTensor(mu, mu)

data = {pp.PARAMETERS: {'mechanics': {'fourth_order_tensor': C, 'bc': pp.BoundaryConditionVectorial(g)}},
        pp.DISCRETIZATION_MATRICES: {'mechanics': {}}}

tpsa = T("mechanics")
tpsa.discretize(g, data)

matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]['mechanics']

debug = []



