# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:43:38 2016

@author: keile
"""

import numpy as np
import pytest

import porepy as pp


@pytest.mark.parametrize(
    "a,b,ma_known,ia_known",
    [
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
        ),
        # unequal sizes
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5], [3, 3, 3, 1]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
        ),
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3, 4, 7], [3, 3, 3, 1, 9, 9, 9]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
        ),
        # double occurences
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 2, 5], [3, 3, 1]]),
            np.array([0, 1, 1, 0, 0], dtype=bool),
            np.array([0, 1]),
        ),
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]]),
            np.array([1, 1, 1, 1, 0], dtype=bool),
            np.array([1, 0, 2, 1]),
        ),
        # 1D
        (
            np.array([0, 2, 1, 3, 0]),
            np.array([2, 4, 3]),
            np.array([0, 1, 0, 1, 0], dtype=bool),
            np.array([0, 2]),
        ),
        (
            np.array([0, 2, 1, 13, 0]),
            np.array([2, 4, 13, 0]),
            np.array([1, 1, 0, 1, 1], dtype=bool),
            np.array([3, 0, 2, 3]),
        ),
        # test issue #123
        (
            np.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 0.2, 0.8, 0.5, 0.5, 0.5],
                    [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.8, 0.2, 0.5],
                ]
            ),
            np.array([[0, 1, 1, 0], [0, 0, 1, 1]]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
            np.array([0, 1, 2, 1]),
        ),
    ],
)
def test_ismember_rows_with_sort(a, b, ma_known, ia_known):
    ma, ia = pp.array_operations.ismember_rows(a, b)
    assert np.allclose(ma, ma_known)
    assert np.allclose(ia, ia_known)


@pytest.mark.parametrize(
    "a,b,ma_known,ia_known",
    [
        (
            np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]]),
            np.array([[3, 1, 2, 5, 3], [3, 3, 3, 1, 1]]),
            np.array([1, 1, 0, 1, 0], dtype=bool),
            np.array([1, 0, 1]),
        )
    ],
)
def test_ismember_rows_no_sort(a, b, ma_known, ia_known):
    ma, ia = pp.array_operations.ismember_rows(a, b, sort=False)

    assert np.allclose(ma, ma_known)
    assert np.allclose(ia, ia_known)
