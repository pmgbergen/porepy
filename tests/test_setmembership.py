# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:43:38 2016

@author: keile
"""
import numpy as np
import unittest

from utils import setmembership


class TestUniqueRows(unittest.TestCase):
    def test_unique_rows_1(self):

        a = np.array([[1, 2], [2, 1], [2, 4], [2, 1], [2, 4]])
        ua_expected = np.array([[1, 2], [2, 1], [2, 4]])
        ia_expected = np.array([0, 1, 2])
        ic_expected = np.array([0, 1, 2, 1, 2])
        ua, ia, ic = setmembership.unique_rows(a)
        assert np.sum(np.abs(ua) - np.abs(ua_expected)) == 0
        assert np.all(ia - ia_expected == 0)
        assert np.all(ic - ic_expected == 0)

    if __name__ == '__main__':
        unittest.main()


class TestIsmember(unittest.TestCase):

    def test_ismember_rows_with_sort(self):
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]])
        ma, ia = setmembership.ismember_rows(a, b)

        ma_known = np.array([1, 1, 1, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 2, 1])

        assert np.allclose(ma, ma_known)
        assert np.allclose(ia, ia_known)


    def test_ismember_rows_no_sort(self):
        a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        b = np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]])
        ma, ia = setmembership.ismember_rows(a, b, sort=False)

        ma_known = np.array([1, 1, 0, 1, 0], dtype=bool)
        ia_known = np.array([1, 0, 1])

        assert np.allclose(ma, ma_known)
        assert np.allclose(ia, ia_known)
        
    if __name__ == '__main__':
        unittest.main()


class TestUniqueColumns(unittest.TestCase):

    def test_no_common_points(self):
        p = np.array([[0, 1, 2], [0, 0, 0]])
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        assert np.allclose(p, p_unique)
        assert np.alltrue(old_2_new == np.arange(3))
        assert np.alltrue(old_2_new == new_2_old)
        
    def test_remove_one_point(self):
        p = np.ones((2, 2))
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        assert np.allclose(p, np.ones((2, 1)))
        assert np.alltrue(old_2_new == np.zeros(2))
        assert np.alltrue(new_2_old == np.zeros(1))
        
    def test_remove_one_of_tree(self):
        p = np.array([[1, 1, 0], [1, 1, 0]])
        p_unique, new_2_old, old_2_new = setmembership.unique_columns_tol(p)

        assert np.allclose(p_unique, np.array([[1, 0], [1, 0]]))
        assert np.alltrue(old_2_new == np.array([0, 0, 1]))
        assert np.alltrue(new_2_old == np.array([0, 2]))    

    if __name__ == '__main__':
        unittest.main()
