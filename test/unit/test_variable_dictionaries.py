""" Tests for variable storage in the data fields "state" and "iterate".
"""
import unittest

import porepy as pp


class TestState(unittest.TestCase):
    def setUp(self):
        self.d = {}

    def test_add_empty_state(self):
        """ Add an empty state dictionary
        """
        pp.set_state(self.d)
        self.assertIn(pp.STATE, self.d)

    def test_add_empty_iterate(self):
        """ Add an empty iterate dictionary
        """
        pp.set_iterate(self.d)
        self.assertIn(pp.STATE, self.d)
        self.assertIn(pp.ITERATE, self.d[pp.STATE])

    def test_add_state_twice(self):
        """ Add two state dictionaries.
        
        The existing foo value should be overwritten, while bar should be kept.
        """
        d1 = {"foo": 1, "bar": 2}
        d2 = {"foo": 3, "spam": 4}

        pp.set_state(self.d, d1)
        pp.set_state(self.d, d2)
        for key, val in zip(["foo", "bar", "spam"], [3, 2, 4]):
            self.assertIn(key, self.d[pp.STATE])
            self.assertEqual(self.d[pp.STATE][key], val)

    def test_add_iterate_twice_and_state(self):
        """ Add two state dictionaries.
        
        The existing foo value should be overwritten, while bar should be kept.
        Setting values in pp.STATE should not affect the iterate values.
        """
        d1 = {"foo": 1, "bar": 2}
        d2 = {"foo": 3, "spam": 4}

        pp.set_iterate(self.d, d1)
        pp.set_iterate(self.d, d2)
        pp.set_state(self.d, {"foo": 5})
        for key, val in zip(["foo", "bar", "spam"], [3, 2, 4]):
            self.assertIn(key, self.d[pp.STATE][pp.ITERATE])
            self.assertEqual(self.d[pp.STATE][pp.ITERATE][key], val)
        self.assertEqual(self.d[pp.STATE]["foo"], 5)


if __name__ == "__main__":
    unittest.main()
