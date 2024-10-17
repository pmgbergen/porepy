"""Tests for the Tpsa discretization. Contains two sets of tests:
    1. TestTpsaTailoredGrid is a class that defines a cell of two grids and verifies the
       discretization matrices for the Tpsa discretization. Both interior cells and all
       types of boundary conditions are tested.
    2. Various other tests that probe different aspects of the Tpsa discretization.
       These are not grouped in a class, but are individual tests.

"""

from typing import Optional, Literal
import pytest
import porepy as pp
import numpy as np
import scipy.sparse as sps
from copy import deepcopy

KEYWORD = "mechanics"
"""Module level keyword which identifies the placement of parameters and discretization
matrices in the dictionary."""


def _discretize_get_matrices(grid: pp.Grid, d: dict):
    """Helper function to discretize with Tpsa and return the dictionary of
    discretization matrices.

    Parameters:
        grid: Grid to discretize.
        d: Dictionary with parameters.

    Returns:
        Dictionary of discretization matrices.

    """
    discr = pp.Tpsa(KEYWORD)
    discr.discretize(grid, d)
    return d[pp.DISCRETIZATION_MATRICES][KEYWORD]


def _set_uniform_bc(
    grid: pp.Grid,
    d: dict,
    bc_type: Literal["dir", "neu", "rob"],
    include_rot: Optional[bool] = True,
):
    """Set a uniform boundary condition on all faces of the grid.

    Parameters:
        grid: Grid to set boundary conditions on.
        d: Dictionary with parameters. The boundary conditions objects are added to
            this dictionary.
        bc_type: Type of boundary condition. One of 'dir', 'neu', 'rob'.
        include_rot: Whether to include a boundary condition for the rotation variable.

    """
    face_ind = grid.get_all_boundary_faces()
    nf = face_ind.size
    match bc_type:
        case "dir":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["dir"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["dir"])
        case "neu":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["neu"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["neu"])
        case "rob":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["rob"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["rob"])
        case _:
            raise ValueError(f"Unknown boundary condition type {bc_type}")

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_disp
    if include_rot:
        d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot


class TestTpsaTailoredGrid:
    """Define a grid of two cells, one internal face. Verify that the discretization
    matrices are constructed correctly for the internal face, as well as two boundary
    faces for Dirichlet, Neumann and Robin conditions.

    The test strategy is to compare the discretization with known values that are
    computed by hand following the description of the Tpsa discretization provided in
    the paper. This should mainly be straightforward, though there is room for confusion
    e.g. in sign conventions for normal vectors. Experience shows, however, that such
    errors are picked up by convergence tests and similar.

    Scope of difficulty for the tests (described here for future reference in case we
    feel like reassessing the test coverage): The target faces have indices 1 (the
    internal one), 0 (boundary face, normal vector along x-axis, normal vector points
    into the cell, unit area of face, unit distance between cell and face center), and 6
    (boundary face, not aligned with the axis, non-unitary area and cell-face distance,
    outward-pointing normal vector). The Lame and Cosserat parameters are heterogeneous.

    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # Define grid.
        g = pp.CartGrid([2, 1])
        g.nodes = np.array(
            [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
        ).T
        g.compute_geometry()
        g.face_centers[0, 3] = 1.5
        g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T
        self.g = g

        # Define shear modulus and Cosserat parameter.
        self.mu_0 = 1
        self.mu_1 = 2
        self.cos_0 = 1
        self.cos_1 = 3

        # The second Lame parameter is actually not tested here, since it only enters
        # through the accumulation term in tpsa. Still, it is needed to define a tensor.
        lmbda = np.array([1, 1])
        mu = np.array([self.mu_0, self.mu_1])
        C = pp.FourthOrderTensor(mu, lmbda)
        cosserat = np.array([self.cos_0, self.cos_1])

        self.data = {
            pp.PARAMETERS: {
                KEYWORD: {"fourth_order_tensor": C, "cosserat_parameter": cosserat}
            },
            pp.DISCRETIZATION_MATRICES: {KEYWORD: {}},
        }

        # We test the discretization on face 6, as this has two non-trivial components
        # of the normal vector, moreover the vector between cell and face center has
        # non-unitary values and is of non-unit length, the effect of both will show up
        # in the discretization scheme. These values will be overridden for the test of
        # the internal face.
        self.target_faces_scalar = np.array([0, 6])
        self.target_faces_vector = np.array([0, 1, 12, 13])

        # Store some grid quantities: Normal vectors of the two target boundary faces
        # and their norms.
        self.n_0 = np.array([1, 0])
        self.n_0_nrm = 1
        self.n_6 = np.array([1, 2])
        self.n_6_nrm = np.sqrt(5)

        # The distance from cell center to face center, projected onto the normal, for
        # the two target boundary faces.
        self.d_0_0 = 1
        self.d_1_6 = 3 / (2 * self.n_6_nrm)

    def _compare_matrices(
        self,
        matrices: dict,
        known_values: dict,
        target_faces_scalar: Optional[np.ndarray] = None,
        target_faces_vector: Optional[np.ndarray] = None,
    ):
        """Helper function to compare a dict of matrices with known values. Target only
        specified faces of the grid.
        """
        if target_faces_scalar is None:
            target_faces_scalar = self.target_faces_scalar
        if target_faces_vector is None:
            target_faces_vector = self.target_faces_vector

        # Check that all computed matrices have assigned known values. This will fail if
        # a new discretization matrix is added without also adding a known value.
        for mat_key in matrices:
            assert mat_key in known_values

        for key, known in known_values.items():
            computed = matrices[key].toarray()
            if computed.shape[0] == self.g.num_faces * 2:
                target_rows = target_faces_vector
            else:
                target_rows = target_faces_scalar
            assert np.allclose(computed[target_rows], known)

    def test_discretization_interior_cells(self):
        """Construct a tpsa discretization on a grid consisting of two cells, compare
        the computed coefficients with hardcoded values. This test is for the internal
        face.
        """

        # This test considers only the discretization on interior cells, but we still
        # need to give some boundary values to the discretization. Assign Dirichlet
        # conditions, more or less arbitrarily.
        _set_uniform_bc(self.g, self.data, "dir")
        matrices = _discretize_get_matrices(self.g, self.data)

        # Normal vector and its length.
        n = np.array([2, 1])
        n_nrm = np.sqrt(5)

        # Target the only interior face.
        target_faces_scalar = np.array([1])
        target_faces_vector = np.array([2, 3])

        # The distance from cell center to face center, projected onto the normal, is
        # 3 / (2 * sqrt(5)) for both cells.
        d_0_1 = 3 / (2 * n_nrm)
        d_1_1 = 3 / (2 * n_nrm)

        # Weighted sum of the shear moduli.
        mu_w = self.mu_0 / d_0_1 + self.mu_1 / d_1_1

        # The stress coefficient is twice the harmonic average of the two shear moduli.
        # Multiply by the length of the face.
        stress = 2 * (self.mu_0 * self.mu_1 / (d_0_1 * d_1_1) / mu_w) * n_nrm

        # The Cosserat parameter is the harmonic average of the two Cosserat parameters.
        # Multiply by the length of the face (sqrt(5)).
        cosserat = (
            self.cos_0
            * self.cos_1
            / (d_0_1 * d_1_1)
            / (self.cos_0 / d_0_1 + self.cos_1 / d_1_1)
            * n_nrm
        )

        # Weight of the cell-to-face averaging operator for cell 0.
        c2f_avg_0 = self.mu_0 / d_0_1 / mu_w
        c2f_avg_1 = self.mu_1 / d_1_1 / mu_w
        # Complement operators.
        c_c2f_avg_0 = 1 - c2f_avg_0
        c_c2f_avg_1 = 1 - c2f_avg_1

        known_values = {
            # The stress is negative for the face with outward pointing normal vector
            # (check: this should be opposite of Darcy, which has a minus sign).
            "stress": np.array([[-stress, 0, stress, 0], [0, -stress, 0, stress]]),
            # No boundary effects.
            "bound_stress": np.zeros((2, 14)),
            # Definition follows the description of the paper.
            "stress_rotation": -np.array(  # out minus sign consistent with paper
                [
                    [c_c2f_avg_0 * n[1], c_c2f_avg_1 * n[1]],
                    [-c_c2f_avg_0 * n[0], -c_c2f_avg_1 * n[0]],
                ]
            ),
            # Definition follows the description of the paper.
            "stress_total_pressure": np.array(
                [
                    [c_c2f_avg_0 * n[0], c_c2f_avg_1 * n[0]],
                    [c_c2f_avg_0 * n[1], c_c2f_avg_1 * n[1]],
                ],
            ),
            "rotation_displacement": -np.array(
                [
                    -c2f_avg_0 * n[1],
                    c2f_avg_0 * n[0],
                    -c2f_avg_1 * n[1],
                    c2f_avg_1 * n[0],
                ]
            ),
            "bound_rotation_displacement": np.zeros((1, 14)),
            "rotation_diffusion": np.array([-cosserat, cosserat]),
            "bound_rotation_diffusion": np.zeros((1, 7)),
            "solid_mass_displacement": np.array(
                [c2f_avg_0 * n[0], c2f_avg_0 * n[1], c2f_avg_1 * n[0], c2f_avg_1 * n[1]]
            ),
            "bound_mass_displacement": np.zeros((1, 14)),
            "solid_mass_total_pressure": np.array([-1 / (2 * mu_w), 1 / (2 * mu_w)])
            * n_nrm,
            # This is an interior face, all displacment reconstruction matrices should
            # be empty.
            "bound_displacement_cell": np.zeros((2, 4)),
            "bound_displacement_face": np.zeros((2, 14)),
            "bound_displacement_rotation_cell": np.zeros((2, 2)),
            "bound_displacement_solid_pressure_cell": np.zeros((2, 2)),
        }
        # Override default target faces
        self._compare_matrices(
            matrices, known_values, target_faces_scalar, target_faces_vector
        )

    def test_dirichlet_bcs(self):
        """Set Dirichlet boundary conditions on all faces, check that the implementation
        of the boundary conditions are correct.
        """
        _set_uniform_bc(self.g, self.data, "dir")
        matrices = _discretize_get_matrices(self.g, self.data)

        # The standard expression for Dirichlet conditions on Laplace problems.
        stress_0 = 2 * self.mu_0 / self.d_0_0 * self.n_0_nrm
        stress_6 = 2 * self.mu_1 / self.d_1_6 * self.n_6_nrm

        # The values of the cell to face average for faces 0 and 6. Also the complements
        # identified by prefix c_. On Dirichlet faces, the face value is imposed, thus
        # the cell is asigned weight 0 (hence the complement has weight 1).
        c2f_avg_0 = 0
        c2f_avg_6 = 0
        c_c2f_avg_0 = 1 - c2f_avg_0
        c_c2f_avg_6 = 1 - c2f_avg_6

        # Discretization of the boundary condition. 
        bound_stress = np.zeros((4, 14))
        bound_stress[0, 0] = -stress_0
        bound_stress[1, 1] = -stress_0
        bound_stress[2, 12] = stress_6
        bound_stress[3, 13] = stress_6

        # The boundary condition for the rotation diffusion problem is the standard
        # expression for Dirichlet conditions on Laplace problems.
        bound_rotation_diffusion = np.zeros((2, 7))
        bound_rotation_diffusion[0, 0] = -self.cos_0 / self.d_0_0 * self.n_0_nrm
        bound_rotation_diffusion[1, 6] = self.cos_1 / self.d_1_6 * self.n_6_nrm

        bound_rotation_displacement = np.zeros((2, 14))
        # From the definition of \bar{R}, we get [-n[1], n[0]]. The discretization is
        # -\bar{R}, hence the sign change.
        bound_rotation_displacement[0, 0] = c_c2f_avg_0 * self.n_0[1]
        bound_rotation_displacement[0, 1] = -c_c2f_avg_0 * self.n_0[0]
        bound_rotation_displacement[1, 12] = c_c2f_avg_6 * self.n_6[1]
        bound_rotation_displacement[1, 13] = -c_c2f_avg_6 * self.n_6[0]

        # On Dirichlet faces, the solid mass follows the displacement boundary
        # condition.
        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] = c_c2f_avg_0 * self.n_0[0]
        bound_mass_displacement[0, 1] = c_c2f_avg_0 * self.n_0[1]
        bound_mass_displacement[1, 12] = c_c2f_avg_6 * self.n_6[0]
        bound_mass_displacement[1, 13] = c_c2f_avg_6 * self.n_6[1]

        # On Dirichlet faces, the boundary displacement is recovered from the boundary
        # condition; all other reconstruction matrices are zero.
        bound_displacement_face = np.zeros((4, 14))
        bound_displacement_face[0, 0] = 1
        bound_displacement_face[1, 1] = 1
        bound_displacement_face[2, 12] = 1
        bound_displacement_face[3, 13] = 1

        known_values = {
            # Positive sign on the first two rows, since the normal vector is pointing
            # into that cell. Oposite sign on the two last rows, as the normal vector is
            # pointing out of the cell.
            "stress": np.array(
                [
                    [stress_0, 0, 0, 0],
                    [0, stress_0, 0, 0],
                    [0, 0, -stress_6, 0],
                    [0, 0, 0, -stress_6],
                ]
            ),
            "bound_stress": bound_stress,
            # Minus sign for the full expression (see paper).
            "stress_rotation": -np.array(
                [
                    [c_c2f_avg_0 * self.n_0[1], 0],
                    [-c_c2f_avg_0 * self.n_0[0], 0],
                    [0, c_c2f_avg_6 * self.n_6[1]],
                    [0, -c_c2f_avg_6 * self.n_6[0]],
                ]
            ),
            "stress_total_pressure": np.array(
                [
                    [c_c2f_avg_0 * self.n_0[0], 0],
                    [c_c2f_avg_0 * self.n_0[1], 0],
                    [0, c_c2f_avg_6 * self.n_6[0]],
                    [0, c_c2f_avg_6 * self.n_6[1]],
                ]
            ),
            # No contribution from the cell displacement to the boundary rotation.
            "rotation_displacement": np.zeros((2, 4)),
            "bound_rotation_displacement": bound_rotation_displacement,
            # Minus sign on the second face, since the normal vector is pointing out of
            # the cell.
            "rotation_diffusion": np.array(
                [
                    [self.cos_0 / self.d_0_0 * self.n_0_nrm, 0],
                    [0, -self.cos_1 / self.d_1_6 * self.n_6_nrm],
                ]
            ),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.zeros((2, 4)),
            "bound_mass_displacement": bound_mass_displacement,
            "solid_mass_total_pressure": np.zeros((2, 2)),
            # No contribution from cell center values to the boundary displacement
            "bound_displacement_cell": np.zeros((4, 4)),
            # The boundary displacement is the boundary condition
            "bound_displacement_face": bound_displacement_face,
            # Neither the rotation variable nor the solid pressure contribute to the
            # boundary displacement for Dirichlet faces
            "bound_displacement_rotation_cell": np.zeros((4, 2)),
            "bound_displacement_solid_pressure_cell": np.zeros((4, 2)),
        }

        self._compare_matrices(matrices, known_values)

    def test_neumann_bcs(self):
        """Set Neumann boundary conditions on all faces, check that the implementation
        of the boundary conditions are correct.
        """
        _set_uniform_bc(self.g, self.data, "neu")
        matrices = _discretize_get_matrices(self.g, self.data)

        # The values of the cell to face average for face for faces 0 and 6. Also the
        # complements identified by suffix c_. On Neumann faces, only the cell value is
        # used for the computation, thus it is assigned unit value (and the complement,
        # with value 0, is ignored in the construction of the analytical expressions
        # below, hence we don't define it).
        c2f_avg_0 = 1
        c2f_avg_6 = 1

        # Boundary stress: The coefficients in bound_stress should be negative those of
        # the stress matrix so that translation results in a stress-free configuration.
        bound_stress = np.zeros((4, 14))
        bound_stress[0, 0] = -1
        bound_stress[1, 1] = -1
        bound_stress[2, 12] = 1
        bound_stress[3, 13] = 1

        bound_rotation_diffusion = np.zeros((2, 7))
        bound_rotation_diffusion[0, 0] = -1
        bound_rotation_diffusion[1, 6] = 1

        # Transform the stress components to rotations via an inverse Hook's law.
        bound_rotation_displacement = np.zeros((2, 14))
        bound_rotation_displacement[0, 0] = -self.d_0_0 * self.n_0[1] / (2 * self.mu_0)
        bound_rotation_displacement[0, 1] = self.d_0_0 * self.n_0[0] / (2 * self.mu_0)
        bound_rotation_displacement[1, 12] = -self.d_1_6 * self.n_6[1] / (2 * self.mu_1)
        bound_rotation_displacement[1, 13] = self.d_1_6 * self.n_6[0] / (2 * self.mu_1)

        # Transform the stress components to displacements via an inverse Hook's law.
        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] = self.d_0_0 * self.n_0[0] / (2 * self.mu_0)
        bound_mass_displacement[0, 1] = self.d_0_0 * self.n_0[1] / (2 * self.mu_0)
        bound_mass_displacement[1, 12] = self.d_1_6 * self.n_6[0] / (2 * self.mu_1)
        bound_mass_displacement[1, 13] = self.d_1_6 * self.n_6[1] / (2 * self.mu_1)

        # The contribution from cell center displacement to the boundary displacement
        # has unit value in the cell neighboring the face.
        bound_displacement_cell = np.zeros((4, 4))
        bound_displacement_cell[0, 0] = 1
        bound_displacement_cell[1, 1] = 1
        bound_displacement_cell[2, 2] = 1
        bound_displacement_cell[3, 3] = 1

        # Prescribed stresses are converted to displacements by 'inverting' Hook's law.
        # Multiply with -1 on face 0 since this has an inward pointing normal vector.
        bound_displacement_face = np.zeros((4, 14))
        bound_displacement_face[0, 0] = -self.d_0_0 / (2 * self.mu_0 * self.n_0_nrm)
        bound_displacement_face[1, 1] = -self.d_0_0 / (2 * self.mu_0 * self.n_0_nrm)
        bound_displacement_face[2, 12] = self.d_1_6 / (2 * self.mu_1 * self.n_6_nrm)
        bound_displacement_face[3, 13] = self.d_1_6 / (2 * self.mu_1 * self.n_6_nrm)

        # The contribution from solid pressure. Multiply with -1 on face 0 since this
        # has an inward pointing normal vector.
        bound_displacement_rotation_cell = np.zeros((4, 2))
        bound_displacement_rotation_cell[0, 0] = (
            self.d_0_0 * self.n_0[1] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_rotation_cell[1, 0] = (
            -self.d_0_0 * self.n_0[0] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_rotation_cell[2, 1] = (
            -self.d_1_6 * self.n_6[1] / (2 * self.mu_1 * self.n_6_nrm)
        )
        bound_displacement_rotation_cell[3, 1] = (
            self.d_1_6 * self.n_6[0] / (2 * self.mu_1 * self.n_6_nrm)
        )

        # Contribution from solid pressure. Multiply with -1 on face 0 since this has an
        # inward pointing normal vector.
        bound_displacement_solid_pressure_cell = np.zeros((4, 2))
        bound_displacement_solid_pressure_cell[0, 0] = (
            -self.d_0_0 * self.n_0[0] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_solid_pressure_cell[1, 0] = (
            -self.d_0_0 * self.n_0[1] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_solid_pressure_cell[2, 1] = (
            self.d_1_6 * self.n_6[0] / (2 * self.mu_1 * self.n_6_nrm)
        )
        bound_displacement_solid_pressure_cell[3, 1] = (
            self.d_1_6 * self.n_6[1] / (2 * self.mu_1 * self.n_6_nrm)
        )

        known_values = {
            # The stress is prescribed, thus no contribution from the interior cells for
            # any of the stress-related matrices.
            "stress": np.zeros((4, 4)),
            "stress_rotation": np.zeros((4, 2)),
            "stress_total_pressure": np.zeros((4, 2)),
            # The boundary stress is defined above.
            "bound_stress": bound_stress,
            # The outer minus sign is part of the analytical expression. The inner minus
            # signs, and the coefficients, follow from the definition of R_k^n and the
            # coefficients of the discretization (see paper).
            "rotation_displacement": -np.array(
                [
                    [-c2f_avg_0 * self.n_0[1], c2f_avg_0 * self.n_0[0], 0, 0],
                    [0, 0, -c2f_avg_6 * self.n_6[1], c2f_avg_6 * self.n_6[0]],
                ]
            ),
            "bound_rotation_displacement": bound_rotation_displacement,
            "rotation_diffusion": np.zeros((2, 2)),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.array(
                [
                    [c2f_avg_0 * self.n_0[0], c2f_avg_0 * self.n_0[1], 0, 0],
                    [0, 0, c2f_avg_6 * self.n_6[0], c2f_avg_6 * self.n_6[1]],
                ]
            ),
            "bound_mass_displacement": bound_mass_displacement,
            "solid_mass_total_pressure": np.array(
                [
                    [self.d_0_0 / (2 * self.mu_0) * self.n_0_nrm, 0],
                    [0, -self.d_1_6 / (2 * self.mu_1) * self.n_6_nrm],
                ]
            ),
            "bound_displacement_cell": bound_displacement_cell,
            "bound_displacement_face": bound_displacement_face,
            "bound_displacement_rotation_cell": bound_displacement_rotation_cell,
            "bound_displacement_solid_pressure_cell": bound_displacement_solid_pressure_cell,
        }

        self._compare_matrices(matrices, known_values)

    def test_robin_bcs(self):
        """Set Robin boundary conditions on all faces, check that the discretization
        stencil for boundary faces, as well as the implementation of the boundary
        conditions are correct.
        """
        # Set the boundary conditions. We will do some modifications below.
        _set_uniform_bc(self.g, self.data, "rob")

        # Modify the Robin weight in the displacement boundary condition. Assign
        # different weights in the x- and y-direction for face 0, equal weights for face
        # 6. The different weights in the x- and y-directions will be reflected in the
        # analytical expressions for the boundary conditions.
        rw_0_x = 2
        rw_0_y = 1
        rw_6 = 1
        # Assign to boundary condition object.
        bc_disp = self.data[pp.PARAMETERS][KEYWORD]["bc"]
        bc_disp.robin_weight[0, 0, 0] = rw_0_x
        bc_disp.robin_weight[1, 1, 0] = rw_0_y
        bc_disp.robin_weight[0, 0, 6] = rw_6
        bc_disp.robin_weight[1, 1, 6] = rw_6

        # Robin boundaries have not yet been implemented for the rotation variable, so
        # set this to Neumann and ignore the computed values.
        bc_rot = self.data[pp.PARAMETERS][KEYWORD]["bc_rot"]
        # This will actually set Neumann conditions also on internal faces, which is
        # not meaningful, but it will not be a problem for the purpose of this test.
        bc_rot.is_neu[:] = True
        bc_rot.is_rob[:] = False

        matrices = _discretize_get_matrices(self.g, self.data)

        d_0_x_bound = rw_0_x
        d_0_y_bound = rw_0_y
        d_6_bound = rw_6

        # Shorthand for the shear modulus divided by the cell to face distance.
        mu_0_d = self.mu_0 / self.d_0_0
        mu_1_d = self.mu_1 / self.d_1_6

        # Averaging coefficient for the interior cell. The Robin condition manifest as
        # an elastic modulus divided by a distance.
        c2f_avg_0_x = 2 * mu_0_d / (2 * mu_0_d + rw_0_x)
        c2f_avg_0_y = 2 * mu_0_d / (2 * mu_0_d + rw_0_y)
        c2f_avg_6 = 2 * mu_1_d / (2 * mu_1_d + rw_6)
        # And the complement.
        c_c2f_avg_0_x = 1 - c2f_avg_0_x
        c_c2f_avg_0_y = 1 - c2f_avg_0_y
        c_c2f_avg_6 = 1 - c2f_avg_6

        # Averaging coefficients for the boundary term.
        c2f_avg_0_x_bound = rw_0_x / (2 * mu_0_d + rw_0_x)
        c2f_avg_0_y_bound = rw_0_y / (2 * mu_0_d + rw_0_y)
        c2f_avg_6_bound = rw_6 / (2 * mu_1_d + rw_6)
        # And the complement.
        c_c2f_avg_0_x_bound = 1 - c2f_avg_0_x_bound
        c_c2f_avg_0_y_bound = 1 - c2f_avg_0_y_bound
        c_c2f_avg_6_bound = 1 - c2f_avg_6_bound

        # The term delta_k^mu (see paper for description).
        delta_0_x = 1 / (2 * mu_0_d + rw_0_x)
        delta_0_y = 1 / (2 * mu_0_d + rw_0_y)
        delta_6 = 1 / (2 * mu_1_d + rw_6)

        # Stress discretization, use distances that incorporate the Robin condition.
        stress_0_x = 2 * self.n_0_nrm * (mu_0_d * rw_0_x) / (mu_0_d + rw_0_x)
        stress_0_y = 2 * self.n_0_nrm * (mu_0_d * rw_0_y) / (mu_0_d + rw_0_y)
        stress_6 = 2 * self.n_6_nrm * (mu_1_d * rw_6) / (mu_1_d + rw_6)

        # Boundary stress. The first term is the 'Neumann part', second is Dirichlet.
        # The signs of the respective parts are the same as in the Dirichlet and Neumann
        # tests.
        bound_stress = np.zeros((4, 14))
        bound_stress[0, 0] = -c_c2f_avg_0_x_bound - stress_0_x
        bound_stress[1, 1] = -c_c2f_avg_0_y_bound - stress_0_y
        bound_stress[2, 12] = c_c2f_avg_6_bound + stress_6
        bound_stress[3, 13] = c_c2f_avg_6_bound + stress_6

        # The boundary condition for the rotation diffusion problem is set to Neumann
        # conditions (see top of this method), so copy these conditions from the
        # relevant test.
        bound_rotation_diffusion = np.zeros((2, 7))
        bound_rotation_diffusion[0, 0] = -1
        bound_rotation_diffusion[1, 6] = 1

        # Rotation generated by displacement boundary condition. The first term is
        # associated with the stress part of the Robin condition, the second with the
        # displacement part.
        bound_rotation_displacement = np.zeros((2, 14))
        bound_rotation_displacement[0, 0] = self.n_0[1] * (
            -delta_0_x + c2f_avg_0_x_bound
        )
        bound_rotation_displacement[0, 1] = self.n_0[0] * (
            delta_0_y - c2f_avg_0_y_bound
        )
        bound_rotation_displacement[1, 12] = self.n_6[1] * (-delta_6 + c2f_avg_6_bound)
        bound_rotation_displacement[1, 13] = self.n_6[0] * (delta_6 - c2f_avg_6_bound)

        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] = self.n_0[0] * (delta_0_x + c2f_avg_0_x_bound)
        bound_mass_displacement[0, 1] = self.n_0[1] * (delta_0_y + c2f_avg_0_y_bound)
        bound_mass_displacement[1, 12] = self.n_6[0] * (delta_6 + c2f_avg_6_bound)
        bound_mass_displacement[1, 13] = self.n_6[1] * (delta_6 + c2f_avg_6_bound)

        # The contribution from cell center displacement to the boundary displacement.
        bound_displacement_cell = np.zeros((4, 4))
        bound_displacement_cell[0, 0] = c2f_avg_0_x
        bound_displacement_cell[1, 1] = c2f_avg_0_y
        bound_displacement_cell[2, 2] = c2f_avg_6
        bound_displacement_cell[3, 3] = c2f_avg_6

        # Prescribed stresses are converted to displacements by 'inverting' Hook's law.
        # Multiply with -1 on face 0 since this has an inward pointing normal vector.
        bound_displacement_face = np.zeros((4, 14))
        bound_displacement_face[0, 0] = -c2f_avg_0_x_bound / (
            (2 * mu_0_d + rw_0_x) * self.n_0_nrm
        )
        bound_displacement_face[1, 1] = -c2f_avg_0_y_bound / (
            (2 * mu_0_d + rw_0_y) * self.n_0_nrm
        )
        bound_displacement_face[2, 12] = c2f_avg_6_bound / (
            (2 * mu_1_d + rw_6) * self.n_6_nrm
        )
        bound_displacement_face[3, 13] = c2f_avg_6_bound / (
            (2 * mu_1_d + rw_6) * self.n_6_nrm
        )

        bound_displacement_rotation_cell = np.zeros((4, 2))
        bound_displacement_rotation_cell[0, 0] = (
            self.n_0[1] * c_c2f_avg_0_x / ((2 * mu_0_d + rw_0_x) * self.n_0_nrm)
        )
        bound_displacement_rotation_cell[1, 0] = (
            -self.n_0[0] * c_c2f_avg_0_y / ((2 * mu_0_d + rw_0_y) * self.n_0_nrm)
        )
        bound_displacement_rotation_cell[2, 1] = (
            -self.n_6[1] * c_c2f_avg_6 / ((2 * mu_1_d + rw_6) * self.n_6_nrm)
        )
        bound_displacement_rotation_cell[3, 1] = (
            self.n_6[0] * c_c2f_avg_6 / ((2 * mu_1_d + rw_6) * self.n_6_nrm)
        )

        # Contribution from solid pressure. Multiply with -1 on face 0 since this has an
        # inward pointing normal vector.
        bound_displacement_solid_pressure_cell = np.zeros((4, 2))
        bound_displacement_solid_pressure_cell[0, 0] = (
            -self.n_0[0] * c_c2f_avg_0_x / ((2 * mu_0_d + rw_0_x) * self.n_0_nrm)
        )
        bound_displacement_solid_pressure_cell[1, 0] = (
            -self.n_0[1] * c_c2f_avg_0_y / ((2 * mu_0_d + rw_0_y) * self.n_0_nrm)
        )
        bound_displacement_solid_pressure_cell[2, 1] = (
            self.n_6[0] * c_c2f_avg_6 / ((2 * mu_1_d + rw_6) * self.n_6_nrm)
        )
        bound_displacement_solid_pressure_cell[3, 1] = (
            self.n_6[1] * c_c2f_avg_6 / ((2 * mu_1_d + rw_6) * self.n_6_nrm)
        )

        known_values = {
            # The stress discretization is the same as in the Dirichlet case.
            "stress": np.array(
                [
                    [stress_0_x, 0, 0, 0],
                    [0, stress_0_y, 0, 0],
                    [0, 0, -stress_6, 0],
                    [0, 0, 0, -stress_6],
                ]
            ),
            "stress_rotation": -np.array(
                [
                    [c_c2f_avg_0_x * self.n_0[1], 0],
                    [-c_c2f_avg_0_y * self.n_0[0], 0],
                    [0, c_c2f_avg_6 * self.n_6[1]],
                    [0, -c_c2f_avg_6 * self.n_6[0]],
                ]
            ),
            "stress_total_pressure": np.array(
                [
                    [c_c2f_avg_0_x * self.n_0[0], 0],
                    [c_c2f_avg_0_y * self.n_0[1], 0],
                    [0, c_c2f_avg_6 * self.n_6[0]],
                    [0, c_c2f_avg_6 * self.n_6[1]],
                ]
            ),
            # The boundary stress is defined above.
            "bound_stress": bound_stress,
            # The outer minus sign is part of the analytical expression. The inner minus
            # signs, and the coefficients, follow from the definition of R_k^n and the
            # coefficients of the discretization (see paper).
            "rotation_displacement": -np.array(
                [
                    [-c2f_avg_0_x * self.n_0[1], c2f_avg_0_y * self.n_0[0], 0, 0],
                    [0, 0, -c2f_avg_6 * self.n_6[1], c2f_avg_6 * self.n_6[0]],
                ]
            ),
            "bound_rotation_displacement": bound_rotation_displacement,
            "rotation_diffusion": np.zeros((2, 2)),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.array(
                [
                    [c2f_avg_0_x * self.n_0[0], c2f_avg_0_y * self.n_0[1], 0, 0],
                    [0, 0, c2f_avg_6 * self.n_6[0], c2f_avg_6 * self.n_6[1]],
                ]
            ),
            "bound_mass_displacement": bound_mass_displacement,
            "solid_mass_total_pressure": np.array(
                [
                    [self.d_0_0 / (2 * self.mu_0) * self.n_0_nrm, 0],
                    [0, -self.d_1_6 / (2 * self.mu_1) * self.n_6_nrm],
                ]
            ),
            "bound_displacement_cell": bound_displacement_cell,
            "bound_displacement_face": bound_displacement_face,
            "bound_displacement_rotation_cell": bound_displacement_rotation_cell,
            "bound_displacement_solid_pressure_cell": bound_displacement_solid_pressure_cell,
        }

        self._compare_matrices(matrices, known_values)

    def test_mixed_bcs(self):
        """Set mixed boundary conditions (e.g. type A in one direction, B in a different
        direction) on all faces, check that the discretization stencil for internal
        faces, as well as the implementation of the boundary conditions, are correct.

        Note that it is not necessary to consider interaction between different types of
        boundary conditions on different faces, since a two-point stencil does not allow
        for such interactions.
        """
        # The values here are mainly copied from the respective tests for Dirichlet and
        # Neumann conditions. The only non-trivial part is the impact of total pressure
        # on the solid mass conservation equation, see comment below and in the
        # implementation.

        # Dirichlet condition in x-direction, Neumann in y-direction.
        _set_uniform_bc(self.g, self.data, "dir")
        bc_disp = self.data[pp.PARAMETERS][KEYWORD]["bc"]
        bc_disp.is_dir[1, :] = False
        bc_disp.is_neu[1, :] = True

        matrices = _discretize_get_matrices(self.g, self.data)

        # The standard expression for Dirichlet conditions on Laplace problems.
        stress_0 = 2 * self.mu_0 / self.d_0_0 * self.n_0_nrm
        stress_6 = 2 * self.mu_1 / self.d_1_6 * self.n_6_nrm

        # The values of the cell to face average for face for faces 0 and 6. Also the
        # complements identified by prefix c_. On Dirichlet faces, the face value is
        # imposed, thus the cell is assigned weight 0 (hence the complement has weight
        # 1).
        c2f_avg_0_x = 0
        c2f_avg_6_x = 0
        c_c2f_avg_0_x = 1 - c2f_avg_0_x
        c_c2f_avg_6_x = 1 - c2f_avg_6_x

        c2f_avg_0_y = 1
        c2f_avg_6_y = 1

        bound_stress = np.zeros((4, 14))
        bound_stress[0, 0] = -stress_0
        bound_stress[1, 1] = -1
        bound_stress[2, 12] = stress_6
        bound_stress[3, 13] = 1

        # The boundary condition for the rotation diffusion problem is the standard
        # expression for Dirichlet conditions on Laplace problems.
        bound_rotation_diffusion = np.zeros((2, 7))
        bound_rotation_diffusion[0, 0] = -self.cos_0 / self.d_0_0 * self.n_0_nrm
        bound_rotation_diffusion[1, 6] = self.cos_1 / self.d_1_6 * self.n_6_nrm

        bound_rotation_displacement = np.zeros((2, 14))
        # From the definition of \bar{R}, we get [-n[1], n[0]]. The discretization is
        # -\bar{R}, hence the sign change.
        bound_rotation_displacement[0, 0] = c_c2f_avg_0_x * self.n_0[1]
        bound_rotation_displacement[0, 1] = self.d_0_0 * self.n_0[0] / (2 * self.mu_0)
        bound_rotation_displacement[1, 12] = c_c2f_avg_6_x * self.n_6[1]
        bound_rotation_displacement[1, 13] = self.d_1_6 * self.n_6[0] / (2 * self.mu_1)

        # On Dirichlet faces, the solid mass follows the displacement boundary
        # condition.
        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] = c_c2f_avg_0_x * self.n_0[0]
        bound_mass_displacement[0, 1] = self.d_0_0 * self.n_0[1] / (2 * self.mu_0)
        bound_mass_displacement[1, 12] = c_c2f_avg_6_x * self.n_6[0]
        bound_mass_displacement[1, 13] = self.d_1_6 * self.n_6[1] / (2 * self.mu_1)

        # The contribution from cell center displacement to the boundary displacement
        # has unit value in the cell neighboring the face.
        bound_displacement_cell = np.zeros((4, 4))
        bound_displacement_cell[1, 1] = 1
        bound_displacement_cell[3, 3] = 1

        # On Dirichlet faces, the boundary displacement is recovered from the boundary
        # condition; all other reconstruction matrices are zero.
        bound_displacement_face = np.zeros((4, 14))
        bound_displacement_face[0, 0] = 1
        bound_displacement_face[1, 1] = -self.d_0_0 / (2 * self.mu_0 * self.n_0_nrm)
        bound_displacement_face[2, 12] = 1
        bound_displacement_face[3, 13] = self.d_1_6 / (2 * self.mu_1 * self.n_6_nrm)

        # The contribution from solid pressure. Multiply with -1 on face 0 since this
        # has an inward pointing normal vector.
        bound_displacement_rotation_cell = np.zeros((4, 2))
        bound_displacement_rotation_cell[1, 0] = (
            -self.d_0_0 * self.n_0[0] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_rotation_cell[3, 1] = (
            self.d_1_6 * self.n_6[0] / (2 * self.mu_1 * self.n_6_nrm)
        )

        # Contribution from solid pressure. Multiply with -1 on face 0 since this has an
        # inward pointing normal vector.
        bound_displacement_solid_pressure_cell = np.zeros((4, 2))
        bound_displacement_solid_pressure_cell[1, 0] = (
            -self.d_0_0 * self.n_0[1] / (2 * self.mu_0 * self.n_0_nrm)
        )
        bound_displacement_solid_pressure_cell[3, 1] = (
            self.d_1_6 * self.n_6[1] / (2 * self.mu_1 * self.n_6_nrm)
        )

        known_values = {
            # Positive sign on the first two rows, since the normal vector is pointing
            # into that cell. Oposite sign on the two last rows, as the normal vector is
            # pointing out of the cell.
            "stress": np.array(
                [
                    [stress_0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, -stress_6, 0],
                    [0, 0, 0, 0],
                ]
            ),
            "bound_stress": bound_stress,
            # Minus sign for the full expression (see paper).
            "stress_rotation": -np.array(
                [
                    [c_c2f_avg_0_x * self.n_0[1], 0],
                    [0, 0],
                    [0, c_c2f_avg_6_x * self.n_6[1]],
                    [0, 0],
                ]
            ),
            "stress_total_pressure": np.array(
                [
                    [c_c2f_avg_0_x * self.n_0[0], 0],
                    [0, 0],
                    [0, c_c2f_avg_6_x * self.n_6[0]],
                    [0, 0],
                ]
            ),
            # No contribution from the cell displacement to the boundary rotation.
            "rotation_displacement": -np.array(
                [
                    [0, c2f_avg_0_y * self.n_0[0], 0, 0],
                    [0, 0, 0, c2f_avg_6_y * self.n_6[0]],
                ]
            ),
            "bound_rotation_displacement": bound_rotation_displacement,
            # Minus sign on the second face, since the normal vector is pointing out of
            # the cell.
            "rotation_diffusion": np.array(
                [
                    [self.cos_0 / self.d_0_0 * self.n_0_nrm, 0],
                    [0, -self.cos_1 / self.d_1_6 * self.n_6_nrm],
                ]
            ),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.array(
                [
                    [0, c2f_avg_0_y * self.n_0[1], 0, 0],
                    [0, 0, 0, c2f_avg_6_y * self.n_6[1]],
                ]
            ),
            "bound_mass_displacement": bound_mass_displacement,
            # This is the only non-trivial term, since it involves only scalar
            # quantities, and the boundary condition is stated on the (vector)
            # displacement. As discussed in the implementation, we pick the boundary
            # condition in the direction which is closest to the face normal. This turns
            # out to be the x-direction for face 0, and the y-direction for face 6, and
            # these are therefore assigned the values of the Dirichlet and Neumann
            # condition, respectively.
            "solid_mass_total_pressure": np.array(
                [
                    [0, 0],
                    [0, -self.d_1_6 / (2 * self.mu_1) * self.n_6_nrm],
                ]
            ),
            # No contribution from cell center values to the boundary displacement.
            "bound_displacement_cell": bound_displacement_cell,
            # The boundary displacement is the boundary condition.
            "bound_displacement_face": bound_displacement_face,
            # Neither the rotation variable nor the solid pressure contribute to the
            # boundary displacement for Dirichlet faces.
            "bound_displacement_rotation_cell": bound_displacement_rotation_cell,
            "bound_displacement_solid_pressure_cell": bound_displacement_solid_pressure_cell,
        }

        self._compare_matrices(matrices, known_values)


def test_no_cosserat():
    """Set up a problem without Cosserat effects, check that the rotation diffusion
    matrix is zero.
    """
    g = pp.CartGrid([2, 2])
    g.compute_geometry()

    d = _set_uniform_parameters(g)
    bf = g.get_all_boundary_faces()
    d[pp.PARAMETERS][KEYWORD]["bc"] = pp.BoundaryConditionVectorial(
        g, faces=bf, cond=bf.size * ["dir"]
    )

    # Discretize, assemble matrices.
    matrices = _discretize_get_matrices(g, d)

    assert np.allclose(matrices["rotation_diffusion"].toarray(), 0)
    assert np.allclose(matrices["bound_rotation_diffusion"].toarray(), 0)


def test_cosserat_3d():
    """Set up a 3d problem, check that the rotation diffusion matrix has the right
    dimension, and that the boundary conditions are correctly implemented.
    """
    g = pp.CartGrid([2, 1, 1])
    g.compute_geometry()
    d = _set_uniform_parameters(g)

    d[pp.PARAMETERS][KEYWORD]["cosserat_parameter"] = np.ones(g.num_cells)
    bf = g.get_all_boundary_faces()
    d[pp.PARAMETERS][KEYWORD]["bc"] = pp.BoundaryConditionVectorial(
        g, faces=bf, cond=bf.size * ["dir"]
    )
    # Set Dirichlet conditions everywhere except on face 0, which has a Neumann
    # condition.
    bc_rot = pp.BoundaryConditionVectorial(g, faces=bf, cond=bf.size * ["dir"])
    bc_rot.is_dir[:, 0] = False
    bc_rot.is_neu[:, 0] = True
    d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot

    # Discretize
    matrices = _discretize_get_matrices(g, d)

    rot_mat = matrices["rotation_diffusion"]
    # Check size
    assert rot_mat.shape == (g.num_faces * 3, g.num_cells * 3)

    known_values = np.zeros((12, 6))

    # Face 0 has a Neumann condition, thus the coefficients are zero. Check coefficients
    # on the inner face.
    known_values[3, 0] = -1
    known_values[4, 1] = -1
    known_values[5, 2] = -1
    known_values[3, 3] = 1
    known_values[4, 4] = 1
    known_values[5, 5] = 1
    # Check coefficients on face 2, which has a Dirichlet condition.
    known_values[6, 3] = -2
    known_values[7, 4] = -2
    known_values[8, 5] = -2
    # Face 3 has a Dirichlet condition, but with inwards pointing normal vector, thus
    # the coefficients should have negative sign.
    known_values[9, 0] = 2
    known_values[10, 1] = 2
    known_values[11, 2] = 2

    assert np.allclose(rot_mat[:12].toarray(), known_values)

    bc_rot = matrices["bound_rotation_diffusion"]
    known_values = np.zeros((12, 33))
    # Neumann condition on face 0.
    known_values[0, 0] = -1
    known_values[1, 1] = -1
    known_values[2, 2] = -1
    # Dirichlet condition, outwards pointing normal vector on face 2.
    known_values[6, 6] = 2
    known_values[7, 7] = 2
    known_values[8, 8] = 2
    # Dirichlet condition, inwards pointing normal vector on face 3.
    known_values[9, 9] = -2
    known_values[10, 10] = -2
    known_values[11, 11] = -2

    assert np.allclose(bc_rot[:12].toarray(), known_values)


@pytest.mark.parametrize("g", [pp.CartGrid([2, 2]), pp.CartGrid([2, 2, 2])])
@pytest.mark.parametrize("driving_bc_type", ["dir", "neu"])
@pytest.mark.parametrize("tensile", [False, True])
def test_compression_tension(g: pp.Grid, driving_bc_type: str, tensile: bool):
    """Assign a compressive or tensile force, check that the resulting displacement is
    in the correct direction, and that the total pressure is negative.

    The logic of the test is as follows (assuming a compressive regime, tensile=False,
    is compressive on the south, east and (if 3d) bottom faces, the remaining faces
    having neutral conditions. Check that this results in displacement in the negative
    x-direction and positive y- (and z) direction. The total pressure should be
    negative, while EK cannot surmise the correct sign of the rotation by physical
    intuition. The grid is assumed to be Cartesian (or else the discretization is
    inconsistent, and anything can happen), in which case the face normal vectors will
    point into the domain on the south (and bottom) boundary, out of the domain on the
    east boundary.

    Parameters:
        g: Grid object. driving_bc_type: Type of boundary condition to apply. 
        tensile: If True, the boundary conditions are reversed, such that the force is
            tensile, rather than compressive.

    """
    # EK note to self: For the 3d test of compression with a unit value for the elastic
    # moduli, there was an unexpected sign in the direction of the displacement, but
    # this changed when moving to a stiffer material. This does not sound unreasonable,
    # but is hereby noted for future reference.

    g.compute_geometry()

    d = _set_uniform_parameters(g)
    if g.dim == 2:
        dir_list = [driving_bc_type, driving_bc_type, "dir", "dir"]
    else:
        dir_list = [
            driving_bc_type,
            driving_bc_type,
            "dir",
            "dir",
            driving_bc_type,
            "dir",
        ]
    bc_values = _set_bc_by_direction(g, d, *dir_list)

    # Tensile if requested.
    if tensile:
        bc_values *= -1

    # Discretize, assemble matrices.
    matrices = _discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)

    if g.dim == 2:
        n_rot_face = g.num_faces
        rot_dim = 1
    else:
        n_rot_face = g.num_faces * g.dim
        rot_dim = g.dim

    # Boundary values, map to right-hand side.
    bound_vec = np.hstack((bc_values.ravel("F"), np.zeros(n_rot_face)))
    x = _solve(flux, rhs_matrix, div, accum, bound_vec)

    if tensile:
        # Positive x-direction, negative y- (and z-) direction.
        assert np.all(x[: g.dim * g.num_cells : g.dim] > 0)
        assert np.all(x[1 : g.dim * g.num_cells : g.dim] < 0)
        if g.dim == 3:
            assert np.all(x[2 : g.dim * g.num_cells : g.dim] < 0)

        # Total pressure should be positive (expansion).
        assert np.all(x[(g.dim + rot_dim) * g.num_cells :] > 0)
    else:
        assert np.all(x[: g.dim * g.num_cells : g.dim] < 0)
        assert np.all(x[1 : g.dim * g.num_cells : g.dim] > 0)
        if g.dim == 3:
            assert np.all(x[2 : g.dim * g.num_cells : g.dim] > 0)

        assert np.all(x[(g.dim + rot_dim) * g.num_cells :] < 0)


@pytest.mark.parametrize("g", [pp.CartGrid([2, 2]), pp.CartGrid([2, 2, 2])])
def test_translation(g: pp.Grid):
    """Set boundary conditions that correspond to a translation of the grid, check that
    the interior cells follow the translation, and that the resulting system is
    stress-free.
    """
    g.compute_geometry()

    d = _set_uniform_parameters(g)

    # Set type and values of boundary conditions. No rotation.
    _set_uniform_bc(g, d, "dir", include_rot=False)
    if g.dim == 2:
        disp = np.array([1, -2])
    else:
        disp = np.array([1, -2, 3])

    bc_values = np.zeros((g.dim, g.num_faces))
    bf = g.get_boundary_faces()
    for dim in range(g.dim):
        bc_values[dim, bf] = disp[dim]

    # Discretize, assemble matrices.
    matrices = _discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)

    if g.dim == 2:
        n_rot_face = g.num_faces
    else:
        n_rot_face = g.num_faces * g.dim

    # Boundary values.
    bound_vec = np.hstack((bc_values.ravel("F"), np.zeros(n_rot_face)))

    # Check: Uniform translation should result in a stress-free state on all faces. This
    # is not the case for rotation and solid mass, which are only zero when integrated
    # over individual cells.
    disp_cells = np.zeros(g.num_cells * g.dim)
    disp_cells[:: g.dim] = disp[0]
    disp_cells[1 :: g.dim] = disp[1]
    if g.dim == 3:
        disp_cells[2 :: g.dim] = disp[2]
    v = matrices["stress"] @ disp_cells + matrices["bound_stress"] @ bc_values.ravel(
        "F"
    )
    assert np.allclose(v, 0)

    x = _solve(flux, rhs_matrix, div, accum, bound_vec)

    # Check that the displacement is as expected.
    assert np.allclose(x[: g.dim * g.num_cells : g.dim], disp[0])
    assert np.allclose(x[1 : g.dim * g.num_cells : g.dim], disp[1])
    if g.dim == 3:
        assert np.allclose(x[2 : g.dim * g.num_cells : g.dim], disp[2])

    # Both the rotation and the total pressure should be zero.
    assert np.allclose(x[g.dim * g.num_cells :], 0)


@pytest.mark.parametrize("g", [pp.CartGrid([2, 2]), pp.CartGrid([2, 2, 2])])
def test_robin_neumann_dirichlet_consistency(g: pp.Grid):
    """Test that a Robin boundary condition approaches the Dirichlet limit for large
    parameter values, and that Robin is equivalent to Neumann for a zero value.

    Parameters:
        g: Grid object.
    """
    np.random.seed(0)

    g.compute_geometry()

    d = _set_uniform_parameters(g)

    bf = g.get_all_boundary_faces()

    left = np.where(g.face_centers[0] == g.face_centers[0].min())[0]

    # Define three sets of boundary condition, with differing types on the left
    # boundary.
    bc_type = np.zeros(bf.size, dtype="object")
    bc_type[:] = "dir"
    bc_type_rob = bc_type.copy()
    bc_type_rob[left] = "rob"
    bc_type_neu = bc_type.copy()
    bc_type_neu[left] = "neu"

    bc_dir = pp.BoundaryConditionVectorial(g, faces=bf, cond=bc_type)
    bc_rob = pp.BoundaryConditionVectorial(g, faces=bf, cond=bc_type_rob)
    bc_neu = pp.BoundaryConditionVectorial(g, faces=bf, cond=bc_type_neu)

    # Random values for the boundary conditions.
    bc_values = np.zeros((g.dim, g.num_faces))
    vals = np.random.rand(g.dim, bf.size)
    bc_values_disp = np.zeros((g.dim, g.num_faces))
    bc_values_disp[:, bf] = vals
    bc_values_rot = (
        np.zeros(g.num_faces) if g.dim == 2 else np.zeros(g.dim * g.num_faces)
    )
    bc_values = np.hstack((bc_values_disp.ravel("F"), bc_values_rot))

    # Discretize, assemble matrices.
    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_dir
    matrices_dir = deepcopy(_discretize_get_matrices(g, d))
    flux_dir, rhs_matrix_dir, div, accum = _assemble_matrices(matrices_dir, g)
    x_dir = _solve(flux_dir, rhs_matrix_dir, div, accum, bc_values)

    # For future reference: EK has verified that, as the Robin weight is increased, the
    # discretization matrices of the Dirichlet and Robin cases do converge. The rates
    # seem to be faster for the matrices than for the solution, but this is not
    # unreasonable.

    # Set the Robin weight to a large value, such that the Robin condition approaches
    # the Dirichlet condition.
    high_weight = 1e15
    bc_rob.robin_weight[0, 0] = high_weight
    bc_rob.robin_weight[1, 1] = high_weight
    if g.dim == 3:
        bc_rob.robin_weight[2, 2] = high_weight

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_rob
    matrices_high = deepcopy(_discretize_get_matrices(g, d))
    flux_high, rhs_matrix_high, div, accum = _assemble_matrices(matrices_high, g)
    x_rob_high = _solve(flux_high, rhs_matrix_high, div, accum, bc_values)
    # The displacement should be close to the Dirichlet value.
    assert np.allclose(x_rob_high, x_dir, rtol=1e-2)

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_neu
    matrices_neu = deepcopy(_discretize_get_matrices(g, d))
    flux_neu, rhs_matrix_neu, div, accum = _assemble_matrices(matrices_neu, g)
    x_neu = _solve(flux_neu, rhs_matrix_neu, div, accum, bc_values)

    # Set the Robin weight to zero, such that the Robin condition approaches the Neumann
    # condition.
    bc_rob.robin_weight[0, 0] = 0
    bc_rob.robin_weight[1, 1] = 0
    if g.dim == 3:
        bc_rob.robin_weight[2, 2] = 0

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_rob
    matrices = _discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)
    x_rob_low = _solve(flux, rhs_matrix, div, accum, bc_values)

    # The displacement should be close to the Neumann value.
    assert np.allclose(x_rob_low, x_neu)


def _set_uniform_parameters(g: pp.Grid) -> dict:
    """Set up a uniform parameter dictionary for the TPSA problem."""
    e = 100 * np.ones(g.num_cells)
    C = pp.FourthOrderTensor(e, e)

    d = {
        pp.PARAMETERS: {KEYWORD: {"fourth_order_tensor": C}},
        pp.DISCRETIZATION_MATRICES: {KEYWORD: {}},
    }
    return d


def _assemble_matrices(
    matrices: dict, g: pp.Grid
) -> tuple[sps.sparray, sps.sparray, sps.sparray, sps.sparray]:
    """Helper method to assemble discretization matrices derived from a Tpsa
    discretization into global matrices.

    Parameters:
        matrices: Dictionary containing the discretization matrices.
        g: Grid object.

    Returns:
        sps.sparray: Discretization of the face terms as a block matrix. The first block
            row contains the stress terms, the second the rotation terms, and the third
            the solid mass 'flux'.
        sps.sparray: Discretization of the boundary conditions, as a map from numerical
            values for the boundary condition to stresses, rotations and solid mass
            fluxes on the boundary faces.
        sps.sparray: Divergence matrix for the face terms.
        sps.sparray: Accumulation matrix for the cell center terms.

    """
    # Deal with the different dimensions of the rotation variable.
    if g.dim == 2:
        n_rot_face = g.num_faces
        n_rot_cell = g.num_cells
        div_rot = pp.fvutils.scalar_divergence(g)
    else:
        n_rot_face = g.num_faces * g.dim
        n_rot_cell = g.num_cells * g.dim
        div_rot = pp.fvutils.vector_divergence(g)

    flux = sps.block_array(
        [
            [
                matrices["stress"],
                matrices["stress_rotation"],
                matrices["stress_total_pressure"],
            ],
            [
                matrices["rotation_displacement"],
                matrices["rotation_diffusion"],
                sps.csr_array((n_rot_face, g.num_cells)),
            ],
            [
                matrices["solid_mass_displacement"],
                sps.csr_array((g.num_faces, n_rot_cell)),
                matrices["solid_mass_total_pressure"],
            ],
        ],
    )

    rhs_matrix = sps.block_array(
        [
            [
                matrices["bound_stress"],
                sps.csr_array((g.num_faces * g.dim, n_rot_face)),
            ],
            [
                matrices["bound_rotation_displacement"],
                sps.csr_array((n_rot_face, n_rot_face)),
            ],
            [
                matrices["bound_mass_displacement"],
                sps.csr_array((g.num_faces, n_rot_face)),
            ],
        ]
    )

    div = sps.block_diag(
        [
            pp.fvutils.vector_divergence(g),
            div_rot,
            pp.fvutils.scalar_divergence(g),
        ],
        format="csr",
    )

    accum = sps.block_diag(
        [
            sps.csr_array((g.num_cells * g.dim, g.num_cells * g.dim)),
            sps.eye(n_rot_cell),
            sps.eye(g.num_cells),
        ],
        format="csr",
    )

    return flux, rhs_matrix, div, accum


def _solve(
    flux: sps.sparray,
    rhs_matrix: sps.sparray,
    div: sps.sparray,
    accum: sps.sparray,
    bound_vec: np.ndarray,
) -> np.ndarray:
    """Solve the TPSA problem.

    Parameters:
        flux: Discretization of the face terms as a block matrix.
        rhs_matrix: Discretization of the boundary conditions.
        div: Divergence matrix for the face terms.
        accum: Accumulation matrix for the cell center terms.
        bound_vec: Array of boundary condition values.

    Returns:
        np.ndarray: Array of cell center values.
    """
    b = -div @ rhs_matrix @ bound_vec

    # Assemble and solve. The minus sign on accum follows from the definition of the
    # governing equations in the paper.
    A = div @ flux - accum
    x = sps.linalg.spsolve(A, b)
    return x


def _set_bc_by_direction(
    g: pp.Grid,
    d: dict,
    type_south: Literal["dir", "neu"],
    type_east: Literal["dir", "neu"],
    type_north: Literal["dir", "neu"],
    type_west: Literal["dir", "neu"],
    type_bottom: Optional[Literal["dir", "neu"]] = None,
    type_top: Optional[Literal["dir", "neu"]] = None,
) -> np.ndarray:
    """Set the boundary conditions on the grid, based on the types of boundary
    conditions given.

    The boundary condition has value 0.1 on the south face, -0.1 on the east face, and
    0.1 on the bottom face (if 3d). The remaining faces have homogeneous conditions.
    The conditions correspond to a compressive force on the south, east and (if
    relevant) bottom faces.

    Parameters:
        g: Grid object.
        d: Dictionary of parameters.
        type_south: Type of boundary condition on the south face.
        type_east: Type of boundary condition on the east face.
        type_north: Type of boundary condition on the north face.
        type_west: Type of boundary condition on the west face.
        type_bottom: Type of boundary condition on the bottom face. Only relevant for 3d
            grids.
        type_top: Type of boundary condition on the top face. Only relevant for 3d
            grids.

    Returns:
        np.ndarray: Array of boundary condition values, as a g.dim x num_faces array.

    """

    face_ind = g.get_all_boundary_faces()
    nf = face_ind.size

    # Find the faces on the boundary in each direction.
    domain = pp.domain.domain_sides_from_grid(g)
    # Represent the array of boundary conditions as a numpy array for now, this makes it
    # easy to insert values in the correct order.
    bc_str = np.zeros(g.num_faces, dtype="object")

    directions = ['south', 'east', 'north', 'west']
    types = [type_south, type_east, type_north, type_west]
    if g.dim == 3:
        directions += ['bottom', 'top']
        types += [type_bottom, type_top]

    for direction, bc_type in zip(directions, types):
        fi = np.where(getattr(domain, direction))[0]
        match bc_type:
            case "dir":
                bc_str[fi] = "dir"
            case "neu":
                bc_str[fi] = "neu"
            case _:
                raise ValueError(f"Unknown boundary condition type {bc_type}")

    # Convert to list, which is the format expected by the BoundaryCondition object.
    bc_list = bc_str[face_ind].tolist()

    bc_disp = pp.BoundaryConditionVectorial(g, faces=face_ind, cond=bc_list)
    if g.dim == 2:
        bc_rot = pp.BoundaryCondition(g, faces=face_ind, cond=bc_list)
    else:
        bc_rot = pp.BoundaryConditionVectorial(g, faces=face_ind, cond=bc_list)

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_disp
    d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot

    bc_val = np.zeros((g.dim, g.num_faces))
    bc_val[1, np.where(getattr(domain, 'south'))[0]] = 0.1
    bc_val[0, np.where(getattr(domain, 'east'))[0]] = -0.1
    if g.dim == 3:
        bc_val[2, np.where(getattr(domain, 'bottom'))[0]] = 0.1

    return bc_val
