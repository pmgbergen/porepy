"""Test the mcolon function."""

import numpy as np
import pytest

from porepy.utils import mcolon


@pytest.mark.parametrize(
    "input,expected",
    [
        ((np.array([1, 2]), np.array([3, 4])), np.array([1, 2, 2, 3])),
        ((np.array([1, 2]), np.array([3, 1])), np.array([1, 2])),
    ],
)
def test_mcolon_simple(input, expected):
    """Test for simple comparison of input (2 arrays) and expected output of mcolon."""
    c = mcolon.mcolon(input[0], input[1])
    assert np.all((c - expected) == 0)


def test_mcolon_zero_output():
    a = np.array([1, 2])
    b = np.array([1, 2])
    c = mcolon.mcolon(a, b)
    assert c.size == 0


def test_mcolon_middle_equal():
    # Motivated by GitHub issue #11
    indPtr = np.array([1, 5, 5, 6])
    select = np.array([0, 1, 2])

    c = mcolon.mcolon(indPtr[select], indPtr[select + 1])
    c_known = np.array([1, 2, 3, 4, 5])
    assert np.allclose(c, c_known)


def test_mcolon_last_equal():
    # Motivated by GitHub issue #11
    indPtr = np.array([1, 5, 5, 6])
    select = np.array([0, 1])

    c = mcolon.mcolon(indPtr[select], indPtr[select + 1])
    c_known = np.array([1, 2, 3, 4])
    assert np.allclose(c, c_known)


def test_hi_low_same_size():
    a = np.array([1, 2])
    b = np.array([2, 3, 4])
    with pytest.raises(ValueError):
        mcolon.mcolon(a, b)


def test_type_conversion():
    a = np.array([1, 3], dtype=np.int8)
    b = 1 + np.array([1, 3], dtype=np.int16)
    c = mcolon.mcolon(a, b)
    assert c.dtype == int
