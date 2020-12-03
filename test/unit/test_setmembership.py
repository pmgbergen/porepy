# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:43:38 2016

@author: keile
"""
import unittest

import numpy as np

from porepy.utils import setmembership


class TestUniqueRows(unittest.TestCase):
    def test_unique_rows_1(self):

        a = np.array([[1, 2], [2, 1], [2, 4], [2, 1], [2, 4]])
        ua_expected = np.array([[1, 2], [2, 1], [2, 4]])
        ia_expected = np.array([0, 1, 2])
        ic_expected = np.array([0, 1, 2, 1, 2])
        ua, ia, ic = setmembership.unique_rows(a)
        self.assertTrue(np.sum(np.abs(ua) - np.abs(ua_expected)) == 0)
        self.assertTrue(np.all(ia - ia_expected == 0))
        self.assertTrue(np.all(ic - ic_expected == 0))


class TestIsmember(unittest.TestCase):
    def test_ismember_rows_with_sort(self):
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 2, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_no_sort(self):
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]])
        ma, ia = setmembership.ismember_rows(a, b, sort=False)

        ma_known = np.array([1, 1, 0, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_unqual_sizes_1(self):
        # a larger than b
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 2, 5], [3, 3, 3, 1]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 2, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_unqual_sizes_1(self):
        # b larger than b
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 2, 5, 3, 4, 7], [3, 3, 3, 1, 9, 9, 9]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 2, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_double_occurence_a_no_b(self):
        # There are duplicate occurences in a that are not found in b
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 2, 5], [3, 3, 1]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([0, 1, 1, 0, 0], dtype=bool)
        ia_known = np.array([0, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_double_occurence_a_and_b(self):
        # There are duplicate occurences in a, and the same item is found in b
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 2, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_1d(self):
        a = np.array([0, 2, 1, 3, 0])
        b = np.array([2, 4, 3])

        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([0, 1, 0, 1, 0], dtype=bool)
        ia_known = np.array([0, 2])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_ismember_rows_1d(self):
        a = np.array([0, 2, 1, 13, 0])
        b = np.array([2, 4, 13, 0])

        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 0, 1, 1], dtype=bool)
        ia_known = np.array([3, 0, 2, 3])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))

    def test_issue_123(self):
        # Reported bug #123.
        # Referred to an implementation of ismember which is now replaced.
        a = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.2, 0.8, 0.5, 0.5, 0.5],
                [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.8, 0.2, 0.5],
            ]
        )
        b = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        ia_known = np.array([0, 1, 2, 1])

        self.assertTrue(np.allclose(ma, ma_known))
        self.assertTrue(np.allclose(ia, ia_known))


class TestUniqueColumns(unittest.TestCase):
    def test_no_common_points(self):
        p = np.array([[0, 1, 2], [0, 0, 0]])
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        self.assertTrue(np.allclose(p, p_unique))
        self.assertTrue(np.alltrue(old_2_new == np.arange(3)))
        self.assertTrue(np.alltrue(old_2_new == new_2_old))

    def test_remove_one_point(self):
        p = np.ones((2, 2))
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        self.assertTrue(np.allclose(p, np.ones((2, 1))))
        self.assertTrue(np.alltrue(old_2_new == np.zeros(2)))
        self.assertTrue(np.alltrue(new_2_old == np.zeros(1)))

    def test_remove_one_of_tree(self):
        p = np.array([[1, 1, 0], [1, 1, 0]])
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        # The sorting of the output depends on how the unique array is computed
        # (see unique_columns_tol for the various options that may be applied).
        # Do a simple sort to ensure we're safe.
        if p_unique[0, 0] == 0:
            self.assertTrue(np.alltrue(np.sort(old_2_new) == np.array([0, 1, 1])))
            self.assertTrue(np.alltrue(np.sort(new_2_old) == np.array([0, 2])))
        else:
            self.assertTrue(np.alltrue(np.sort(old_2_new) == np.array([0, 0, 1])))
            self.assertTrue(np.alltrue(np.sort(new_2_old) == np.array([0, 2])))

        p_known = np.array([[0, 1], [0, 1]])

        for i in range(p_unique.shape[1]):
            self.assertTrue(
                np.min(np.sum(np.abs(p_known - p_unique[:, i]), axis=0)) == 0
            )
        for i in range(p_known.shape[1]):
            self.assertTrue(
                np.min(np.sum(np.abs(p_known[:, i] - p_unique), axis=0)) == 0
            )


if __name__ == "__main__":
    unittest.main()
