import unittest

import porepy as pp


class TestDomain(unittest.TestCase):
    def check_key_value(self, domain, keys, values):
        tol = 1e-12
        for dim, key in enumerate(keys):
            if key in domain:
                self.assertTrue(abs(domain.pop(key) - values[dim]) < tol)
            else:
                self.assertTrue(False)  # Did not find correct key

    def test_unit_cube(self):
        domain = pp.UnitCubeDomain()
        keys = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        values = [0, 0, 0, 1, 1, 1]
        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_unit_square(self):
        domain = pp.UnitSquareDomain()
        keys = ["xmin", "ymin", "xmax", "ymax"]
        values = [0, 0, 1, 1]
        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_cube(self):
        physdims = [3.14, 1, 5]
        domain = pp.CubeDomain(physdims)
        keys = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        values = [0, 0, 0] + physdims

        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_squre(self):
        physdims = [2.71, 1e5]
        domain = pp.SquareDomain(physdims)
        keys = ["xmin", "ymin", "xmax", "ymax"]
        values = [0, 0] + physdims

        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_negative_domain(self):
        physdims = [1, -1]
        with self.assertRaises(ValueError):
            pp.SquareDomain(physdims)
