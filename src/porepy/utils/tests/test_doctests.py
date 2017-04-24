import unittest
import doctest
from porepy_new.src.porepy.utils import comp_geom as cg

test_suite = unittest.TestSuite()
test_suite.addTest(doctest.DocTestSuite(cg))

unittest.TextTestRunner().run(test_suite)
