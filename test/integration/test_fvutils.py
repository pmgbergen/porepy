import numpy as np
import scipy.sparse as sps
import unittest
import porepy as pp
from porepy.numerics.fv import fvutils


class TestFvutils(unittest.TestCase):
    def test_block_matrix_inverters_full_blocks(self):
        """
        Test inverters for block matrices

        """

        a = np.vstack((np.array([[1, 3], [4, 2]]), np.zeros((3, 2))))
        b = np.vstack((np.zeros((2, 3)), np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])))
        block = sps.csr_matrix(np.hstack((a, b)))

        sz = np.array([2, 3], dtype="i8")
        iblock_python = fvutils.invert_diagonal_blocks(block, sz, method="python")
        iblock_ex = np.linalg.inv(block.toarray())

        self.assertTrue(np.allclose(iblock_ex, iblock_python.toarray()))

        # Numba may or may not be available on the system, so surround test with
        # try. This may not be the most pythonic approach, but it works.
        try:
            iblock_numba = fvutils.invert_diagonal_blocks(block, sz, method="numba")
            self.assertTrue(np.allclose(iblock_ex, iblock_numba.toarray()))
        except:
            # Try to import numba to see if it is installed
            try:
                import numba

                # If this works, something went wrong with the usage of numba.
                # Raise an error
                raise Exception("Numba installed, but block invert failed")

            # Pass; the user may purposefully not have installed numba
            except:
                # Numba is not installed; probably by choice of the user. For now
                # we consider this okay, and do not fail the test. This behavior
                # may change in the future.
                pass

        # Cython may or may not be available on the system, so surround test with
        # try. This may not be the most pythonic approach, but it works.
        try:
            iblock_cython = fvutils.invert_diagonal_blocks(block, sz, method="cython")
            self.assertTrue(np.allclose(iblock_ex, iblock_cython.toarray()))
        except:
            # Try to import Cython to see if it is installed
            try:
                import Cython

                # If this works, something went wrong with the usage of cython
                # Raise an error
                raise Exception("Cython installed, but block invert failed")

            # Pass; the user may purposefully not have installed numba
            except:
                # Cython is not installed; probably by choice of the user. For now
                # we consider this okay, and do not fail the test. This behavior
                # may change in the future.
                pass

    def test_block_matrix_invertes_sparse_blocks(self):
        """
        Invert the matrix

        A = [1 2 0 0 0
             3 0 0 0 0
             0 0 3 0 3
             0 0 0 7 0
             0 0 0 1 2]

        Contrary to test_block_matrix_inverters_full_blocks, the blocks of A
        will be sparse. This turned out to give problems
        """

        rows = np.array([0, 0, 1, 2, 2, 3, 4, 4])
        cols = np.array([0, 1, 0, 2, 4, 3, 3, 4])
        data = np.array([1, 2, 3, 3, 3, 7, 1, 2], dtype=np.float64)
        block = sps.coo_matrix((data, (rows, cols))).tocsr()
        sz = np.array([2, 3], dtype="i8")

        iblock_python = fvutils.invert_diagonal_blocks(block, sz, method="python")
        iblock_ex = np.linalg.inv(block.toarray())

        self.assertTrue(np.allclose(iblock_ex, iblock_python.toarray()))

        # Numba may or may not be available on the system, so surround test with
        # try. This may not be the most pythonic approach, but it works.
        try:
            iblock_numba = fvutils.invert_diagonal_blocks(block, sz, method="numba")
            self.assertTrue(np.allclose(iblock_ex, iblock_numba.toarray()))
        except:
            # Try to import numba to see if it is installed
            try:
                import numba

                # If this works, something went wrong with the usage of numba.
                # Raise an error
                raise Exception("Numba installed, but block invert failed")

            # Pass; the user may purposefully not have installed numba
            except:
                # Numba is not installed; probably by choice of the user. For now
                # we consider this okay, and do not fail the test. This behavior
                # may change in the future.
                pass

        # Cython may or may not be available on the system, so surround test with
        # try. This may not be the most pythonic approach, but it works.
        try:
            iblock_cython = fvutils.invert_diagonal_blocks(block, sz, method="cython")
            self.assertTrue(np.allclose(iblock_ex, iblock_cython.toarray()))
        except:
            # Try to import Cython to see if it is installed
            try:
                import Cython

                # If this works, something went wrong with the usage of cython
                # Raise an error
                raise Exception("Cython installed, but block invert failed")

            # Pass; the user may purposefully not have installed numba
            except:
                # Cython is not installed; probably by choice of the user. For now
                # we consider this okay, and do not fail the test. This behavior
                # may change in the future.
                pass

    def test_compute_darcy_flux_mono_grid(self):
        g = pp.CartGrid([1, 1])
        flux = sps.csc_matrix((4, 1))
        bound_flux = sps.csc_matrix(
            np.array([[0, 0, 0, 3], [5, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]])
        )
        vector_source = sps.csc_matrix((g.num_faces, g.num_cells * g.dim))

        bc_val = np.array([1, 2, 3, 4])
        specified_parameters = {"bc_values": bc_val}
        data = pp.initialize_default_data(g, {}, "flow", specified_parameters)
        data[pp.STATE] = {}
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        matrix_dictionary["flux"] = flux
        matrix_dictionary["bound_flux"] = bound_flux
        matrix_dictionary["vector_source"] = vector_source

        data[pp.STATE]["pressure"] = np.array([3.14])
        fvutils.compute_darcy_flux(g, data=data)

        dis = data[pp.PARAMETERS]["flow"]["darcy_flux"]

        dis_true = flux * data[pp.STATE]["pressure"] + bound_flux * bc_val
        self.assertTrue(np.allclose(dis, dis_true))


if __name__ == "__main__":
    unittest.main()
