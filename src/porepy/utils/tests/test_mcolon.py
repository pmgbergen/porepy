# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:51:05 2016

@author: eke001
"""

import numpy as np

from utils import mcolon


def test_mcolon_simple():
    a = np.array([1, 2])
    b = np.array([2, 3])
    c = mcolon.mcolon(a, b)
    assert np.all((c - np.array([1, 2, 2, 3])) == 0)


def test_mcolon_one_missing():
    a = np.array([1, 2])
    b = np.array([2, 1])
    c = mcolon.mcolon(a, b)
    assert np.all((c - np.array([1, 2])) == 0)
