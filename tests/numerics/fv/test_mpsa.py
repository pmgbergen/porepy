"""Tests for the MPSA discretization scheme.

Content:
    - Test that the discretization matrices are correct for a partial update.
    - Test that the discretization reproduces expected values on 2d grids.
    - Test functionality to update the discretization.
    - Test functionality to reconstruct the displacement at the faces.
    - Test of methods internal to the discretization class.
    - Test that the solution is invariant to rotations of the coordinate system in which
        boundary conditions are specified.
    - Test of Robin boundary conditions.
    - Test of Neumann boundary conditions.
    - Test that the discretization reproduces expected values on 2d grids.

"""
from math import pi

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import sympy

import porepy as pp
from porepy.applications.test_utils import common_xpfa_tests as xpfa_tests
from porepy.applications.test_utils import reference_dense_arrays
from porepy.applications.test_utils.partial_discretization import (
    perform_partial_discretization_specified_nodes,
)
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

keyword = "mechanics"

discr = pp.Mpsa(keyword)

matrix_keys = [
    discr.stress_matrix_key,
    discr.bound_stress_matrix_key,
    discr.bound_displacement_cell_matrix_key,
    discr.bound_displacement_face_matrix_key,
]


@pytest.fixture
def discretization_matrices():
    """Return a grid and the corresponding mpsa discretization matrices."""
    g = pp.CartGrid([5, 5])
    g.compute_geometry()
    stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
    bnd = pp.BoundaryConditionVectorial(g)
    stress, bound_stress, _, _ = discr._stress_discretization(
        g, stiffness, bnd, inverter="python"
    )
    return g, stiffness, bnd, stress, bound_stress


def partial_update_parameters(stiffness, bnd):
    """Return parameter dictionary for partial update tests."""
    specified_data = {
        "fourth_order_tensor": stiffness,
        "bc": bnd,
        "inverter": "python",
    }
    return specified_data


@pytest.fixture
def setup_grids_2d():
    """Return a list of grids.

    The grids are:
    - A 2x2 Cartesian grid with no perturbations.
    - A 2x2 Cartesian grid with perturbation to the center node.
    - A 3x3 Cartesian grid with perturbations to all nodes.
    """
    grid_list = []

    # Unstructured perturbation
    nx = np.array([2, 2])
    g_cart_unpert = pp.CartGrid(nx)
    g_cart_unpert.compute_geometry()

    grid_list.append(g_cart_unpert)

    # Structured perturbation
    g_cart_spert = pp.CartGrid(nx)
    g_cart_spert.nodes[0, 4] = 1.5
    g_cart_spert.compute_geometry()
    grid_list.append(g_cart_spert)

    # Larger grid, random perturbations
    nx = np.array([3, 3])
    g_cart_rpert = pp.CartGrid(nx)
    dx = 1
    pert = 0.4
    rand = np.vstack(
        (
            np.random.rand(g_cart_rpert.dim, g_cart_rpert.num_nodes),
            np.repeat(0.0, g_cart_rpert.num_nodes),
        )
    )
    g_cart_rpert.nodes = g_cart_rpert.nodes + dx * pert * (0.5 - rand)
    # No perturbations of the z-coordinate (which is not active in this case)
    g_cart_rpert.nodes[2, :] = 0
    g_cart_rpert.compute_geometry()
    grid_list.append(g_cart_rpert)

    return grid_list


@pytest.mark.parametrize(
    "cell_id",
    [
        10,  # cell at domain boundary
        12,  # internal cell
    ],
)
def test_partial_discretization_specified_nodes(cell_id: int, discretization_matrices):
    """Test that the discretization matrices are correct for a partial update.

    Parameters:
        cell_id: The cell whose node will be used to define the partial update. The
            nodes are identified based on the provided cell_id outside of the
            discretization method.

    """
    g, stiffness, bnd, stress_full, bound_stress_full = discretization_matrices
    specified_data = partial_update_parameters(stiffness, bnd)
    data = perform_partial_discretization_specified_nodes(
        g, discr, specified_data, cell_id
    )

    partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
    partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_stress_matrix_key
    ]

    active_faces = data[pp.PARAMETERS][keyword]["active_faces"]
    active_faces_nd = pp.fvutils.expand_indices_nd(active_faces, g.dim)

    for partial, full in zip(
        [partial_stress, partial_bound],
        [stress_full, bound_stress_full],
    ):
        assert np.allclose(
            partial.todense()[active_faces_nd],
            full.todense()[active_faces_nd],
        )
        # For partial update, only the active faces should be nonzero. Force these to
        # zero and check that the rest is zero.
        pp.fvutils.remove_nonlocal_contribution(active_faces_nd, 1, partial)
        assert np.allclose(partial.data, 0)


def test_partial_discretization_one_cell_at_a_time():
    g = pp.CartGrid([3, 3])
    g.compute_geometry()

    # Assign random permeabilities, for good measure
    np.random.seed(42)
    mu = np.random.random(g.num_cells)
    lmbda = np.random.random(g.num_cells)
    stiffness = pp.FourthOrderTensor(mu=mu, lmbda=lmbda)
    bnd = pp.BoundaryConditionVectorial(g)
    specified_data = partial_update_parameters(stiffness, bnd)
    data = pp.initialize_default_data(
        g, {}, keyword, specified_parameters=specified_data
    )
    discr.discretize(g, data)

    stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
    bound_stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_stress_matrix_key
    ]

    # Initialize matrices to be filled one cell at a time
    stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
    bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))
    faces_covered = np.zeros(g.num_faces, bool)

    for cell_id in range(g.num_cells):
        data = perform_partial_discretization_specified_nodes(
            g, discr, specified_data, cell_id
        )
        # Extract current matrices
        partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        if np.any(faces_covered):
            del_faces = pp.fvutils.expand_indices_nd(np.where(faces_covered)[0], g.dim)
            pp.fvutils.remove_nonlocal_contribution(
                del_faces, 1, partial_stress, partial_bound
            )

        faces_covered[active_faces] = True

        stress += partial_stress
        bound_stress += partial_bound

    # Compare results
    for partial, full in zip([stress, bound_stress], [stress_full, bound_stress_full]):
        assert pp.test_utils.arrays.compare_matrices(partial, full, 1e-10)


class TestMpsaExactReproduction:
    """Test that the discretization reproduces the expected behavior for uniform
    strain, homogeneous conditions and other cases where the method should be exact.

    """

    def solve(
        self,
        g: pp.Grid,
        bound: pp.BoundaryConditionVectorial,
        bc_values: np.ndarray,
        constit: pp.FourthOrderTensor = None,
    ):
        """
        Compute the discretization matrices and solve linear system.

        Parameters:

        """
        if constit is None:
            constit = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        specified_data = {
            "fourth_order_tensor": constit,
            "bc": bound,
            "inverter": "python",
            "bc_values": bc_values,
        }
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        d = np.linalg.solve(A.toarray(), b)

        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        traction = stress * d + bound_stress * bc_values
        return d, traction

    # Test that the discretization reproduces the expected behavior for uniform strain,
    # homogeneous conditions and other cases where the method should be exact
    def test_uniform_strain(self, setup_grids_2d):
        g_list = setup_grids_2d

        for g in g_list:
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryConditionVectorial(
                g, bound_faces, ["dir"] * bound_faces.size
            )
            xc = g.cell_centers
            xf = g.face_centers

            gx = np.random.rand(1)
            gy = np.random.rand(1)

            dc_x = np.sum(xc * gx, axis=0)
            dc_y = np.sum(xc * gy, axis=0)
            df_x = np.sum(xf * gx, axis=0)
            df_y = np.sum(xf * gy, axis=0)

            d_bound = np.zeros((g.dim, g.num_faces))

            d_bound[0, bound.is_dir[0]] = df_x[bound.is_dir[0]]
            d_bound[1, bound.is_dir[1]] = df_y[bound.is_dir[1]]

            bc_values = d_bound.ravel("F")

            d, traction = self.solve(g, bound, bc_values)

            mu = 1
            lmbda = 1
            s_xx = (2 * mu + lmbda) * gx + lmbda * gy
            s_xy = mu * (gx + gy)
            s_yx = mu * (gx + gy)
            s_yy = (2 * mu + lmbda) * gy + lmbda * gx

            n = g.face_normals
            traction_ex_x = s_xx * n[0] + s_xy * n[1]
            traction_ex_y = s_yx * n[0] + s_yy * n[1]

            assert np.max(np.abs(d[::2] - dc_x)) < 1e-8
            assert np.max(np.abs(d[1::2] - dc_y)) < 1e-8
            assert np.max(np.abs(traction[::2] - traction_ex_x)) < 1e-8
            assert np.max(np.abs(traction[1::2] - traction_ex_y)) < 1e-8

    def test_uniform_displacement(self, setup_grids_2d):
        g_list = setup_grids_2d

        for g in g_list:
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryConditionVectorial(
                g, bound_faces, ["dir"] * bound_faces.size
            )
            d_x = np.random.rand(1)
            d_y = np.random.rand(1)
            d_bound = np.zeros((g.dim, g.num_faces))
            d_bound[0, bound.is_dir[0]] = d_x
            d_bound[1, bound.is_dir[1]] = d_y

            bc_values = d_bound.ravel("F")

            d, traction = self.solve(g, bound, bc_values)
            assert np.max(np.abs(d[::2] - d_x)) < 1e-8
            assert np.max(np.abs(d[1::2] - d_y)) < 1e-8
            assert np.max(np.abs(traction)) < 1e-8

    def test_uniform_displacement_neumann(self):
        physdims = [1, 1]
        g_size = [4, 8]
        g_list = [pp.CartGrid([n, n], physdims=physdims) for n in g_size]
        [g.compute_geometry() for g in g_list]
        for g in g_list:

            sides = pp.domain.domain_sides_from_grid(g)
            south = np.where(sides.south)[0]
            west = np.where(sides.west)[0]
            dir_faces = np.hstack((west, south))
            bound = pp.BoundaryConditionVectorial(
                g, dir_faces.ravel("F"), ["dir"] * dir_faces.size
            )
            d_x = np.random.rand(1)
            d_y = np.random.rand(1)
            d_bound = np.zeros((g.dim, g.num_faces))

            d_bound[0, bound.is_dir[0]] = d_x
            d_bound[1, bound.is_dir[1]] = d_y

            bc_values = d_bound.ravel("F")

            d, traction = self.solve(g, bound, bc_values)
            assert np.max(np.abs(d[::2] - d_x)) < 1e-8
            assert np.max(np.abs(d[1::2] - d_y)) < 1e-8
            assert np.max(np.abs(traction)) < 1e-8

    def test_conservation_of_momentum(self):
        pts = np.random.rand(3, 9)
        corners = [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
        pts = np.hstack((corners, pts))
        gt = pp.TetrahedralGrid(pts)
        gc = pp.CartGrid([3, 3, 3], physdims=[1, 1, 1])
        g_list = [gt, gc]
        [g.compute_geometry() for g in g_list]
        for g in g_list:
            sides = pp.domain.domain_sides_from_grid(g)
            south = np.where(sides.south)[0]
            west = np.where(sides.west)[0]
            dir_faces = np.hstack((west, south))
            bound = pp.BoundaryConditionVectorial(
                g, dir_faces.ravel("F"), ["dir"] * dir_faces.size
            )
            constit = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

            bndr = g.get_all_boundary_faces()
            d_x = np.random.rand(bndr.size)
            d_y = np.random.rand(bndr.size)
            d_bound = np.zeros((g.dim, g.num_faces))
            d_bound[0, bndr] = d_x
            d_bound[1, bndr] = d_y

            bc_values = d_bound.ravel("F")

            _, traction = self.solve(g, bound, bc_values, constit)
            traction_2d = traction.reshape((g.dim, -1), order="F")
            for cell in range(g.num_cells):
                fid, _, sgn = sparse_array_to_row_col_data(g.cell_faces[:, cell])
                assert np.all(np.sum(traction_2d[:, fid] * sgn, axis=1) < 1e-10)


def test_split_discretization_into_parts():
    """Test that the discretization matrices are correct if the domain is split into
    subdomains.

    This test is just a shallow wrapper around the common test function for the XPFA
    discretization.
    """
    discr = pp.Mpsa("mechanics")
    xpfa_tests.test_split_discretization_into_subproblems(discr)


class TestUpdateMpsaDiscretization(TestMpsaExactReproduction):
    """
    Class for testing updating the discretization, including the reconstruction
    of gradient displacements. Given a discretization we want
    to rediscretize parts of the domain. This will typically be a change of boundary
    conditions, fracture growth, or a change in aperture.
    """

    def discretize(
        self,
        g: pp.Grid,
        bc: pp.BoundaryConditionVectorial = None,
        constit: pp.FourthOrderTensor = None,
    ):
        """Utility function for discretization"""
        if bc is None:
            bc = pp.BoundaryConditionVectorial(g)
        if constit is None:
            constit = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        specified_data = {
            "fourth_order_tensor": constit,
            "bc": bc,
            "inverter": "python",
        }
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        return data, discr

    def compare_discretization_matrices(self, data_0, data_1):
        for key in matrix_keys:
            diff = (
                data_0[pp.DISCRETIZATION_MATRICES][keyword][key]
                - data_1[pp.DISCRETIZATION_MATRICES][keyword][key]
            )
            assert np.allclose(diff.data, 0)

    def test_no_change_input(self):
        """
        The input matrices should not be changed
        """
        g = pp.CartGrid([4, 4], physdims=(1, 1))
        g.compute_geometry()
        data, discr = self.discretize(g)

        matrices_old = [
            data[pp.DISCRETIZATION_MATRICES][keyword][key].copy() for key in matrix_keys
        ]

        # Update should not change anything
        faces = np.array([0, 4, 5, 6])
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = faces

        discr.discretize(g, data)
        for i, key in enumerate(matrix_keys):
            diff = matrices_old[i] - data[pp.DISCRETIZATION_MATRICES][keyword][key]
            assert np.allclose(diff.data, 0)

    def test_cart_2d(self):
        """When not changing the parameters, the output should equal the input."""
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        data, discr = self.discretize(g)
        matrices_old = [
            data[pp.DISCRETIZATION_MATRICES][keyword][key].copy() for key in matrix_keys
        ]

        # Update should not change anything
        faces = np.array([0, 3])
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = faces

        discr.discretize(g, data)
        for i, key in enumerate(matrix_keys):
            diff = matrices_old[i] - data[pp.DISCRETIZATION_MATRICES][keyword][key]
            assert np.allclose(diff.data, 0)

    def test_changing_bc(self):
        """
        We test that we can change the boundary condition
        """
        g = pp.StructuredTriangleGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        # Neumann conditions everywhere
        bc = pp.BoundaryConditionVectorial(g)
        data, discr = self.discretize(g, bc)

        # Now change the type of boundary condition on one face
        faces = 0
        bc.is_dir[:, faces] = True
        bc.is_neu[bc.is_dir] = False

        # Full discretization
        data_full, discr = self.discretize(g, bc)

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = np.array([faces])

        discr.discretize(g, data)
        self.compare_discretization_matrices(data, data_full)

    def test_changing_bc_by_cells(self):
        """
        Test that we can change the boundary condition by specifying the boundary cells
        """
        g = pp.StructuredTetrahedralGrid([2, 2, 2], physdims=(1, 1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        data, discr = self.discretize(g, bc)

        faces = g.face_centers[2] < 1e-10
        bc.is_rob[:, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Partial discretization should give same result as full
        cells = np.argwhere(g.cell_faces[faces, :])[:, 1].ravel()
        cells = np.unique(cells)

        # Full discretization of the new problem
        data_full, _ = self.discretize(g, bc)

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_cells"] = np.array([cells])
        discr.discretize(g, data)

        self.compare_discretization_matrices(data_full, data)

    def test_mixed_bc(self):
        """
        We test that we can change the boundary condition in given direction
        """
        g = pp.StructuredTriangleGrid([2, 2], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        data, discr = self.discretize(g, bc)

        faces = np.where(g.face_centers[0] > 1 - 1e-10)[0]

        bc.is_rob[1, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Full discretization
        data_full, discr = self.discretize(g, bc)

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = np.array([faces])
        discr.discretize(g, data)

        self.compare_discretization_matrices(data_full, data)


class MpsaReconstructBoundaryDisplacement(TestUpdateMpsaDiscretization):
    def test_cart_2d(self):
        """
        Test that mpsa gives out the correct matrices for
        reconstruction of the displacement at the faces
        """
        num_cells = np.ones(2, dtype=int)
        g = pp.CartGrid(num_cells, physdims=[2, 2])
        g.compute_geometry()

        k = pp.FourthOrderTensor(np.array([2]), np.array([1]))
        data, discr = self.discretize(g, constit=k)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]
        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        hf2f = pp.fvutils.map_hf_2_f(sd=g)
        num_subfaces = hf2f.sum(axis=1).toarray().ravel()
        scaling = sps.dia_matrix(
            (1.0 / num_subfaces, 0), shape=(hf2f.shape[0], hf2f.shape[0])
        )

        hf2f = (scaling * hf2f).toarray()
        expected = pp.test_utils.reference_dense_arrays.test_mpsa[
            "MpsaReconstructBoundaryDisplacement"
        ]["test_cart_2d"]
        assert np.all(
            np.abs(grad_bound - hf2f.dot(expected["grad_bound_known"])) < 1e-7
        )
        assert np.all(np.abs(grad_cell - hf2f.dot(expected["grad_cell_known"])) < 1e-12)

    def test_simplex_3d_dirichlet(self):
        """
        Test that we retrieve a linear solution exactly
        """
        num_cells = np.ones(3, dtype=int) * 2
        g = pp.StructuredTetrahedralGrid(num_cells, physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False

        x0 = np.array([[1, 2, 3]]).T
        u_b = g.face_centers + x0
        bc_val = u_b.ravel("F")

        data, discr = self.discretize(g, bc)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]
        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * bc_val).reshape((g.dim, -1), order="F")

        U = U.reshape((g.dim, -1), order="F")

        assert np.all(np.abs(U - g.cell_centers - x0) < 1e-10)
        assert np.all(np.abs(U_f - g.face_centers - x0) < 1e-10)

    def test_simplex_3d_boundary(self):
        """
        Even if we do not get exact solution at interiour we should be able to
        retrieve the boundary conditions
        """
        num_cells = np.ones(3, dtype=int) * 2
        g = pp.StructuredTetrahedralGrid(num_cells, physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        lam = 10 * np.random.rand(g.num_cells)
        mu = 10 * np.random.rand(g.num_cells)
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        dir_ind = g.get_all_boundary_faces()[[0, 2, 5, 8, 10, 13, 15, 21]]
        bc.is_dir[:, dir_ind] = True
        bc.is_neu[bc.is_dir] = False

        u_b = np.random.randn(g.face_centers.shape[0], g.face_centers.shape[1])

        discr, data = self.discretize(g, bc, k)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * u_b.ravel("F")).reshape(
            (g.dim, -1), order="F"
        )

        assert np.all(np.abs(U_f[:, dir_ind] - u_b[:, dir_ind]) < 1e-10)


class TestCreateBoundRhs:
    """
    Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs
    for handling boundary conditions expressed in a vectorial form
    """

    def test_neu(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g)
        self.run_test(g, basis, bc)

    def test_dir(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "dir")
        self.run_test(g, basis, bc)

    def test_mix(self):
        nx = 2
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        basis = np.random.rand(g.dim, g.dim, g.num_faces)

        sides = pp.domain.domain_sides_from_grid(g)
        west = np.where(sides.west)[0]
        east = np.where(sides.east)[0]
        south = np.where(sides.south)[0]
        north = np.where(sides.north)[0]

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_neu[:] = False

        bc.is_dir[0, west] = True
        bc.is_neu[1, west] = True

        bc.is_rob[0, east] = True
        bc.is_dir[1, east] = True

        bc.is_neu[0, south] = True
        bc.is_rob[1, south] = True

        bc.is_rob[0, north] = True
        bc.is_dir[1, north] = True

        self.run_test(g, basis, bc)

    def run_test(self, g, basis, bc):
        g.compute_geometry()
        g = true_2d(g)

        st = pp.fvutils.SubcellTopology(g)
        bc_sub = pp.fvutils.boundary_to_sub_boundary(bc, st)
        be = pp.fvutils.ExcludeBoundaries(st, bc_sub, g.dim)

        bound_rhs = pp.Mpsa("")._create_bound_rhs(bc_sub, be, st, g, True)

        bc.basis = basis
        bc_sub = pp.fvutils.boundary_to_sub_boundary(bc, st)
        be = pp.fvutils.ExcludeBoundaries(st, bc_sub, g.dim)
        bound_rhs_b = pp.Mpsa("")._create_bound_rhs(bc_sub, be, st, g, True)

        # rhs should not be affected by basis transform
        assert np.allclose(bound_rhs_b.toarray(), bound_rhs.toarray())


class TestMpsaRotation:
    """
    Rotating the basis should not change the answer. This unittest test that Mpsa
    with and without change of basis gives the same answer
    """

    def test_dir(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "dir")
        self.run_test(g, basis, bc)

    def test_rob(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "rob")
        self.run_test(g, basis, bc)

    def test_neu(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "neu")
        # Add a Dirichlet condition so the system is well defined
        bc.is_dir[:, g.get_boundary_faces()[0]] = True
        bc.is_neu[bc.is_dir] = False
        self.run_test(g, basis, bc)

    def test_flip_axis(self):
        """
        We want to test the rotation of mixed conditions, but this is not so easy to
        compare because rotating the basis will change the type of boundary condition
        that is applied in the different directions. This test overcomes that by
        flipping the x- and y-axis and the flipping the boundary conditions also.
        """
        g = pp.CartGrid([2, 2], [1, 1])
        g.compute_geometry()
        nf = g.num_faces
        basis = np.array([[[0] * nf, [1] * nf], [[1] * nf, [0] * nf]])
        bc = pp.BoundaryConditionVectorial(g)

        sides = pp.domain.domain_sides_from_grid(g)
        west = np.where(sides.west)[0]
        east = np.where(sides.east)[0]
        south = np.where(sides.south)[0]
        north = np.where(sides.north)[0]
                
        bc.is_dir[0, west] = True
        bc.is_rob[1, west] = True
        bc.is_rob[0, north] = True
        bc.is_neu[1, north] = True
        bc.is_dir[0, south] = True
        bc.is_neu[1, south] = True
        bc.is_dir[:, east] = True
        bc.is_neu[bc.is_dir + bc.is_rob] = False
        k = pp.FourthOrderTensor(
            np.random.rand(g.num_cells), np.random.rand(g.num_cells)
        )
        # Solve without rotations
        u_bound = np.random.rand(g.dim, g.num_faces)
        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        u = np.linalg.solve(A.toarray(), b)

        # Solve with rotations
        bc_b = pp.BoundaryConditionVectorial(g)
        bc_b.basis = basis
        bc_b.is_dir[1, west] = True
        bc_b.is_rob[0, west] = True
        bc_b.is_rob[1, north] = True
        bc_b.is_neu[0, north] = True
        bc_b.is_dir[1, south] = True
        bc_b.is_neu[0, south] = True
        bc_b.is_dir[:, east] = True
        bc_b.is_neu[bc_b.is_dir + bc_b.is_rob] = False

        bc_val_b = np.sum(basis * u_bound, axis=1).ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc_b,
            "inverter": "python",
            "bc_values": bc_val_b,
        }
        data_b = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data_b)
        A_b, b_b = discr.assemble_matrix_rhs(g, data_b)
        u_b = np.linalg.solve(A_b.toarray(), b_b)

        # Assert that solutions are the same
        assert np.allclose(u, u_b)

    def run_test(self, g, basis, bc):
        g.compute_geometry()
        c = pp.FourthOrderTensor(
            np.random.rand(g.num_cells), np.random.rand(g.num_cells)
        )
        # Solve without rotations
        u_bound = np.random.rand(g.dim, g.num_faces)
        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": c,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        u = np.linalg.solve(A.toarray(), b)

        # Solve with rotations
        bc.basis = basis
        data[pp.PARAMETERS][keyword]["bc"] = bc

        u_bound_b = np.sum(basis * u_bound, axis=1).ravel("F")
        data[pp.PARAMETERS][keyword]["bc_values"] = u_bound_b
        discr.discretize(g, data)
        A_b, b_b = discr.assemble_matrix_rhs(g, data)

        u_b = np.linalg.solve(A_b.toarray(), b_b)
        # Assert that solutions are the same
        assert np.allclose(u, u_b)


def true_2d(g, constit=None):
    if g.dim == 2:
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

    if constit is None:
        return g
    constit = constit.copy()
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=0)
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=1)
    return g, constit


class RobinBoundTest:
    def test_dir_rob(self):
        g = pp.CartGrid(np.array([2, 2]), physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = 1

        bnd_sides = pp.domain.domain_sides_from_grid(g)
        dir_ind = np.ravel(np.argwhere(bnd_sides.left + bnd_sides.bot + bnd_sides.top))
        neu_ind = np.ravel(np.argwhere([]))
        rob_ind = np.ravel(np.argwhere(bnd_sides.right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_dir_neu_rob(self):
        g = pp.CartGrid(np.array([2, 3]), physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bnd_sides = pp.domain.domain_sides_from_grid(g)
        dir_ind = np.ravel(np.argwhere(bnd_sides.left))
        neu_ind = np.ravel(np.argwhere(bnd_sides.top))
        rob_ind = np.ravel(np.argwhere(bnd_sides.right + bnd_sides.bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_structured_triang(self):
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bnd_sides = pp.domain.domain_sides_from_grid(g)

        dir_ind = np.ravel(np.argwhere(()))
        neu_ind = np.ravel(np.argwhere(()))
        rob_ind = bnd_sides.all_bf

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_unstruct_triang(self):
        corners = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        points = np.random.rand(2, 2)
        points = np.hstack((corners, points))
        g = pp.TriangleGrid(points)
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bnd_sides = pp.domain.domain_sides_from_grid(g)

        dir_ind = np.ravel(np.argwhere(()))
        neu_ind = np.ravel(np.argwhere(()))
        rob_ind = bnd_sides.all_bf

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_unstruct_tetrahedron(self):
        network = pp.create_fracture_network(
            [], pp.md_grids.domains.unit_cube_domain(3)
        )
        mesh_args = {"mesh_size_frac": 3, "mesh_size_min": 3}
        mdg = network.mesh(mesh_args)
        sd = mdg.subdomains(dim=3)[0]
        c = pp.FourthOrderTensor(np.ones(sd.num_cells), np.ones(sd.num_cells))
        robin_weight = 1.0

        bnd_sides = pp.domain.domain_sides_from_grid(sd)

        dir_ind = np.ravel(np.argwhere(bnd_sides.west + bnd_sides.top))
        neu_ind = np.ravel(np.argwhere(bnd_sides.bottom))
        rob_ind = np.ravel(
            np.argwhere(bnd_sides.east + bnd_sides.north + bnd_sides.south)
        )

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(sd, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0], 0 * x[2]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
            T_r = [np.dot(sigma, sd.face_normals[:, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((3, sd.num_faces))

        sgn_n, _ = sd.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = sd.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(sd.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(sd.face_centers[:, rob_ind]) * sd.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(sd, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(sd.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(sd.num_faces)).ravel("F"))

    def solve_mpsa(self, g, c, robin_weight, bnd, u_bound):
        bnd.robin_weight *= robin_weight

        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": c,
            "bc": bnd,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        u = np.linalg.solve(A.toarray(), b)
        T = matrix_dictionary[discr.stress_matrix_key] * u + matrix_dictionary[
            discr.bound_stress_matrix_key
        ] * u_bound.ravel("F")
        return u, T


class TestAsymmetricNeumann:
    @property
    def reference_sparse_arrays(self):
        """Returns dictionary of expected matrices for test cases.

        Each matrix is accessed as reference_sparse_arrays["test_name"]["matrix_name"]
        """
        return pp.test_utils.reference_sparse_arrays.test_mpsa["TestAsymmetricNeumann"]

    def test_cart_2d(self):
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        right = g.face_centers[0] > 1 - 1e-10
        top = g.face_centers[1] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        sides = pp.domain.domain_sides_from_grid(g)
        east = np.where(sides.east)[0]
        north = np.where(sides.north)[0]

        bc.is_dir[:, north] = True
        bc.is_dir[0, east] = True

        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        g, k = true_2d(g, k)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)
        _, igrad, _ = pp.Mpsa("")._create_inverse_gradient_matrix(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )

        expected_igrad = self.reference_sparse_arrays["test_cart_2d"]["igrad"]

        assert np.all(np.abs(igrad - expected_igrad).toarray() < 1e-12)

    def test_cart_3d(self):
        g = pp.CartGrid([1, 1, 1], physdims=(1, 1, 1))
        g.compute_geometry()

        bnd_sides = pp.domain.domain_sides_from_grid(g)

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, bnd_sides.west + bnd_sides.east + bnd_sides.south] = True
        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)

        _, igrad, _ = pp.Mpsa("")._create_inverse_gradient_matrix(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        expected_igrad = self.reference_sparse_arrays["test_cart_3d"]["igrad"]
        assert np.all(np.abs(igrad - expected_igrad).toarray() < 1e-12)


class TestMpsaReproduceKnownValues:
    """Test that Mpsa reproduces known values for simple cases.

    The test verifies that the computed values are as expected, by comparing with a
    hard-coded known solution. Failure to reproduce the known solution means that
    something is wrong with the implementation. For this reason, one should be very
    careful with changing anything in this class; in a sense, the test just is what it
    is.

    The test considers Cartesian and simplex grids in 2d, with both homogeneous and
    heterogeneous stiffness matrix.

    """

    def chi(self, xcoord, ycoord):
        return np.logical_and(np.greater(xcoord, 0.5), np.greater(ycoord, 0.5))

    def solve(self, heterogeneous: bool):
        x, y = sympy.symbols("x y")

        # The analytical solutions were different for the homogeneous and heterogeneous
        # cases, as were the grids.
        if heterogeneous:
            g = self.g_lines
            kappa = 1e-6
            ux = sympy.sin(2 * pi * x) * sympy.sin(2 * pi * y)
            uy = sympy.cos(pi * x) * (y - 0.5) ** 2
        else:
            g = self.g_nolines
            kappa = 1.0
            ux = sympy.sin(x) * sympy.cos(y)
            uy = sympy.sin(x) * x**2

        # Calculate the right hand side corresponding to the analytical solution.
        ux_f = sympy.lambdify((x, y), ux, "numpy")
        uy_f = sympy.lambdify((x, y), uy, "numpy")
        dux_x = sympy.diff(ux, x)
        dux_y = sympy.diff(ux, y)
        duy_x = sympy.diff(uy, x)
        duy_y = sympy.diff(uy, y)
        divu = dux_x + duy_y

        sxx = 2 * dux_x + divu
        sxy = dux_y + duy_x
        syx = duy_x + dux_y
        syy = 2 * duy_y + divu

        rhs_x = sympy.diff(sxx, x) + sympy.diff(syx, y)
        rhs_y = sympy.diff(sxy, x) + sympy.diff(syy, y)
        rhs_x_f = sympy.lambdify((x, y), rhs_x, "numpy")
        rhs_y_f = sympy.lambdify((x, y), rhs_y, "numpy")

        # Define stiffness
        char_func_cells = self.chi(g.cell_centers[0], g.cell_centers[1]) * 1.0
        mat_vec = (1 - char_func_cells) + kappa * char_func_cells
        k = pp.FourthOrderTensor(mat_vec, mat_vec)

        # Boundary conditions
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_vec = pp.BoundaryConditionVectorial(
            g, bound_faces, ["dir"] * bound_faces.size
        )
        xf = g.face_centers
        char_func_bound = self.chi(xf[0, bound_faces], xf[1, bound_faces]) * 1
        u_bound = np.zeros((g.dim, g.num_faces))
        u_bound[0, bound_faces] = ux_f(xf[0, bound_faces], xf[1, bound_faces]) / (
            (1 - char_func_bound) + kappa * char_func_bound
        )
        u_bound[1, bound_faces] = uy_f(xf[0, bound_faces], xf[1, bound_faces]) / (
            (1 - char_func_bound) + kappa * char_func_bound
        )
        bc_val = u_bound.ravel("F")
        # Right hand side - contribution from the solution
        xc = g.cell_centers
        rhs = (
            np.vstack((rhs_x_f(xc[0], xc[1]), rhs_y_f(xc[0], xc[1]))) * g.cell_volumes
        ).ravel("F")

        keyword = "mechanics"

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc_vec,
            "inverter": "python",
            "bc_values": bc_val,
            # NOTE: Set eta to zero. This is non-standard for simplex grids, but this
            # was what was used to generate the reference values.
            "mpsa_eta": 0,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        # Discretize
        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        # Right hand side contains both source and stress term
        u_num = spla.spsolve(A, b + rhs)
        stress_num = (
            matrix_dictionary[discr.stress_matrix_key] * u_num
            + matrix_dictionary[discr.bound_stress_matrix_key] * bc_val
        )
        return u_num, stress_num

    @pytest.mark.parametrize("grid_type", ["cart", "simplex"])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    def test_mpsa_computed_values(self, grid_type, heterogeneous):
        g_nolines, g_lines = xpfa_tests.create_grid_mpfa_mpsa_reproduce_known_values(
            grid_type
        )
        self.g_nolines = g_nolines
        self.g_lines = g_lines

        u, flux = self.solve(heterogeneous)

        # Fetch known values
        if heterogeneous:
            key = grid_type + "_heterogeneous"
        else:
            key = grid_type + "_homogeneous"

        known_u = reference_dense_arrays.test_mpsa["TestMpsaReproduceKnownValues"][key][
            "u"
        ]
        known_flux = reference_dense_arrays.test_mpsa["TestMpsaReproduceKnownValues"][
            key
        ]["stress"]

        # Compare computed and known values
        assert np.allclose(u, known_u)
        assert np.allclose(flux, known_flux)
