""" Tests for the Parameters class.

Tests the class' methods for parameter setting and modification.
"""
import numpy as np
import unittest

from porepy.grids.structured import CartGrid
from porepy.params.data import Parameters


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.g = CartGrid([3, 2])
        self.p = Parameters()
        self.p.update_dictionaries("dummy", {"kw0": "param0", "list": [0, 1]})

    def test_add_keywords(self):
        keyword_list = ["flow", "transport"]
        self.p.update_dictionaries(keyword_list)
        keyword_list.append("dummy")
        assert sorted(self.p.keys()) == sorted(keyword_list)

    def test_update_empty_dictionary(self):
        keyword_list = ["flow"]
        d = {"porosity": np.ones(self.g.num_cells)}
        dictionary_list = [d]
        self.p.update_dictionaries(keyword_list, dictionary_list)
        assert "porosity" in self.p["flow"]

    def test_update_dictionary(self):
        keyword_list = ["flow"]
        d = {"porosity": np.ones(self.g.num_cells)}
        self.p.update_dictionaries(keyword_list, [d])

        d = {
            "porosity": 2 * np.ones(self.g.num_cells),
            "density": 3 * np.ones(self.g.num_cells),
        }
        self.p.update_dictionaries(keyword_list, [d])
        success = "density" in self.p["flow"]
        success *= np.all(np.isclose(self.p["flow"]["porosity"], 2))
        assert success

    def test_update_empty_dictionaries(self):
        keyword_list = ["flow", "transport"]
        d1 = {
            "porosity": 2 * np.ones(self.g.num_cells),
            "density": 3 * np.ones(self.g.num_cells),
        }
        d2 = {
            "porosity": 5 * np.ones(self.g.num_cells),
            "storage_weight": 4 * np.ones(self.g.num_cells),
        }
        self.p.update_dictionaries(keyword_list, [d1, d2])
        flow_p = self.p["flow"]
        assert np.all(np.isclose(flow_p["density"], 3))

    def test_set_from_other_subdictionary(self):
        """ Sets a property of "flow" keyword to the one stored for "dummy".
        """

        self.p.update_dictionaries("flow")
        self.p.set_from_other("flow", "dummy", ["kw0"])
        assert self.p["flow"]["kw0"] == self.p["dummy"]["kw0"]

    def test_modify_shared_property(self):
        """ Modifies a property shared by two keywords.
        """
        self.p.update_dictionaries(["transport", "flow"])
        self.p.set_from_other("flow", "dummy", ["kw0"])
        self.p.overwrite_shared_parameters(["kw0"], [13])
        success = not ("kw0" in self.p["transport"])
        success *= np.isclose(self.p["dummy"]["kw0"], 13)
        success *= np.isclose(self.p["flow"]["kw0"], 13)
        assert success

    def test_modify_shared_list(self):
        """ Modifies a property shared by two keywords.
        """
        self.p.update_dictionaries(["transport", "flow"])
        self.p.set_from_other("flow", "dummy", ["list"])
        new_list = [2, 5]
        self.p.modify_parameters("flow", ["list"], [new_list])
        assert self.p["dummy"]["list"] == new_list


if __name__ == "__main__":
    unittest.main()
