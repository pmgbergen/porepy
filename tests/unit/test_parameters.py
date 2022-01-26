""" Tests for the Parameters class and dictionary initialization.



"""
import unittest

import numpy as np

import porepy as pp
import porepy.params.parameter_dictionaries as dicts
from porepy.grids.structured import CartGrid
from porepy.params.data import Parameters


class TestParameters(unittest.TestCase):
    """Test the Parameter class' method for parameter setting and modification.

    _kw refers to outer dictionary and is the keyword that would be given to a
    discretization, whereas
    _key identifies individual parameters.

    """

    def setUp(self):
        self.g = CartGrid([3, 2])
        self.p = Parameters()
        self.p.update_dictionaries(
            "dummy_kw", {"string_key": "string_parameter", "list_key": [0, 1]}
        )

    def test_add_keywords(self):
        """New keywords are added.

        Calls update_dictionaries with a list of the new keywords and empty (default
        option) dictionaries.
        """
        keyword_kw_list = ["flow", "transport"]
        self.p.update_dictionaries(keyword_kw_list)
        keyword_kw_list.append("dummy_kw")
        self.assertListEqual(sorted(self.p.keys()), sorted(keyword_kw_list))

    def test_update_empty_dictionary(self):
        """New keyword added with a parameter.

        Calls update_dictionaries with a list of the new keyword and a list containing
        the corresponding data dictionary. This gives the parameters (see self.setUp)
        dummy_kw:   string_key, list_key
        flow:       porosity
        """
        keyword_kw_list = ["flow"]
        d = {"porosity": np.ones(self.g.num_cells)}
        dictionary_kw_list = [d]
        self.p.update_dictionaries(keyword_kw_list, dictionary_kw_list)
        self.assertIn("porosity", self.p["flow"])

    def test_update_dictionary(self):
        """Add parameters to a dictionary already containing parameters."""
        d = {"string_key": "foo", "density": 3 * np.ones(self.g.num_cells)}
        self.p.update_dictionaries(["dummy_kw"], [d])
        self.assertIn("density", self.p["dummy_kw"])
        self.assertEqual(self.p["dummy_kw"]["string_key"], "foo")

    def test_update_empty_dictionaries(self):
        keyword_kw_list = ["flow", "transport"]
        d1 = {
            "porosity": 2 * np.ones(self.g.num_cells),
            "density": 3 * np.ones(self.g.num_cells),
        }
        d2 = {
            "porosity": 5 * np.ones(self.g.num_cells),
            "storage_weight": 4 * np.ones(self.g.num_cells),
        }
        self.p.update_dictionaries(keyword_kw_list, [d1, d2])
        flow_p = self.p["flow"]
        self.assertTrue(np.all(np.isclose(flow_p["density"], 3)))

    def test_set_from_other_subdictionary(self):
        """Sets a property of "flow" keyword to the one stored for "dummy_kw"."""

        self.p.update_dictionaries("flow")
        self.p.set_from_other("flow", "dummy_kw", ["string_key"])
        self.assertEqual(self.p["flow"]["string_key"], self.p["dummy_kw"]["string_key"])

    def test_overwrite_shared_property(self):
        """Modifies a property shared by two keywords."""
        self.p.update_dictionaries(["transport", "flow"])
        self.p.set_from_other("flow", "dummy_kw", ["string_key"])
        self.p.overwrite_shared_parameters(["string_key"], [13])
        self.assertNotIn("string_key", self.p["transport"])
        self.assertAlmostEqual(self.p["dummy_kw"]["string_key"], 13)
        self.assertAlmostEqual(self.p["flow"]["string_key"], 13)

    def test_modify_shared_list(self):
        """Modifies a list parameter shared by two keywords.

        Note that the type of the shared parameter determines the behaviour of
        modify_parameters, so we also test for an array in next test.

        The parameter list_key is added from the dummy to the add_to kw. Then it is
        modified under the add_to kw. We check that it has changed also in dummy_kw,
        and that the process has not affected other_kw.
        """
        self.p.update_dictionaries(["add_to_kw", "other_kw"])
        self.p.set_from_other("add_to_kw", "dummy_kw", ["list_key"])
        new_list = [2, 5]
        self.p.modify_parameters("add_to_kw", ["list_key"], [new_list])
        self.assertListEqual(self.p["dummy_kw"]["list_key"], new_list)
        self.assertNotIn("list_key", self.p["other_kw"])

    def test_modify_shared_array(self):
        """Modifies an array parameter shared by two keywords.

        See previous test.
        Note that the dtypes of the arrays should ideally be the same.
        """
        self.p.update_dictionaries(["add_to_kw", "add_from_kw", "other_kw"])
        self.p["add_from_kw"]["array_key"] = np.array([0.0, 1.0])
        self.p.set_from_other("add_to_kw", "add_from_kw", ["array_key"])
        new_array = np.array([3.14, 42.0])
        self.p.modify_parameters("add_to_kw", ["array_key"], [new_array])
        self.assertTrue(
            np.all(np.isclose(self.p["add_from_kw"]["array_key"], new_array))
        )
        self.assertNotIn("array_key", self.p["other_kw"])

    def test_expand_scalars(self):
        """Expand scalars to arrays"""
        self.p.update_dictionaries(["dummy_kw"], [{"scalar": 1, "number": 2}])
        keys = ["scalar", "number", "not_present"]
        defaults = [3] * 3
        array_list = self.p.expand_scalars(2, "dummy_kw", keys, defaults)
        for i in range(3):
            self.assertEqual(array_list[i].size, 2)
            self.assertEqual(np.sum(array_list[i]), 2 * (i + 1))


class TestParameterDictionaries(unittest.TestCase):
    """Tests for parameter dictionary initialization.

    Tests the data module's functions for parameter setting and modification.
    TODO: Write tests for pp.initialize_data
    """

    def setUp(self):
        self.g = pp.CartGrid([3, 2])

    def test_default_flow_dictionary(self):
        """Test the default flow dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        dictionary = dicts.flow_dictionary(self.g)
        self.check_default_flow_dictionary(dictionary)

    def test_default_transport_dictionary(self):
        """Test the default transport dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        # The default darcy_flux needs face normals:
        self.g.compute_geometry()
        dictionary = dicts.transport_dictionary(self.g)
        # Check that all parameters have been added.
        p_list = [
            "source",
            "time_step",
            "second_order_tensor",
            "bc",
            "bc_values",
            "darcy_flux",
            "mass_weight",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = ["mass_weight"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        zeros = np.zeros(self.g.num_faces)
        self.assertTrue(np.all(np.isclose(dictionary["darcy_flux"], zeros)))

    def test_default_mechanics_dictionary(self):
        """Test the default mechanics dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        dictionary = dicts.mechanics_dictionary(self.g)
        # Check that all parameters have been added.
        p_list = [
            "source",
            "time_step",
            "fourth_order_tensor",
            "bc",
            "bc_values",
            "slip_distance",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        zeros = np.zeros(self.g.num_faces * self.g.dim)
        self.assertTrue(np.all(np.isclose(dictionary["slip_distance"], zeros)))
        self.assertEqual(dictionary["bc"].bc_type, "vectorial")

    def test_initialize_default_data(self):
        """Test default flow data initialization with default values.

        initialize_data returns a data dictionary d with default "keyword" Parameters
        stored in d["parameters"].
        """
        data = pp.initialize_default_data(self.g, {}, parameter_type="flow")
        self.check_default_flow_dictionary(data[pp.PARAMETERS]["flow"])

    def test_initialize_default_data_specified(self):
        """Test transport data initialization with default and specified values.

        We specify
            "foo": No default value.
            "bc": There is a default value, but of another type (pp.BoundaryCondition).
        All these are set (i.e. no checks on the specified parameters).
        """
        specified_parameters = {
            "porosity": np.zeros(self.g.num_cells),
            "foo": "bar",
            "bc": 15,
        }
        # The default darcy_flux needs face normals:
        self.g.compute_geometry()
        data = pp.initialize_default_data(self.g, {}, "transport", specified_parameters)
        dictionary = data[pp.PARAMETERS]["transport"]
        self.assertEqual(dictionary["foo"], "bar")
        zeros = np.zeros(self.g.num_cells)
        self.assertAlmostEqual(dictionary["bc"], 15)

    def test_initialize_default_data_other_keyword(self):
        """Test transport data initialization with keyword differing from
        parameter_type.

        We specify "foo", for which there is no default value.
        """
        specified_parameters = {"foo": "bar"}
        # The default darcy_flux needs face normals:
        self.g.compute_geometry()
        data = pp.initialize_default_data(
            self.g, {}, "transport", specified_parameters, keyword="not_transport"
        )
        dictionary = data[pp.PARAMETERS]["not_transport"]
        self.assertEqual(dictionary["foo"], "bar")

    def test_initialize_data_specified(self):
        """Test transport data initialization without default values.

        We specify
            "foo": No default value.
            "bc": There is a default value, but of another type (pp.BoundaryCondition).
        All these are set (i.e. no checks on the specified parameters).
        """
        specified_parameters = {
            "porosity": np.zeros(self.g.num_cells),
            "foo": "bar",
            "bc": 15,
        }
        data = pp.initialize_data(self.g, {}, "transport", specified_parameters)
        dictionary = data[pp.PARAMETERS]["transport"]
        self.assertEqual(dictionary["foo"], "bar")
        self.assertAlmostEqual(dictionary["bc"], 15)
        # second_order_tensor is added in the default dictionary, but should not be
        # present since we are testing initialize_data, not initialize_default_data.
        self.assertNotIn("second_order_tensor", dictionary)

    def check_default_flow_dictionary(self, dictionary):
        # Check that all parameters have been added.
        p_list = [
            "mass_weight",
            "source",
            "time_step",
            "second_order_tensor",
            "bc",
            "bc_values",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = ["mass_weight"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        self.assertTrue(
            np.all(np.isclose(dictionary["second_order_tensor"].values[2, 2], ones))
        )


if __name__ == "__main__":
    unittest.main()
