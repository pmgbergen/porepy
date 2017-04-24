import unittest
import doctest
from compgeom import basics

test_suite = unittest.TestSuite()
test_suite.addTest(doctest.DocTestSuite(basics))

unittest.TextTestRunner().run(test_suite)
