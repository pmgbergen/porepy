""" Tests for parameter dictionary initialization.

Tests the data module's functions for parameter setting and modification.
TODO: Write tests for pp.initialize_data
"""
import numpy as np
import unittest

import porepy as pp
import porepy.params.parameter_dictionaries as dicts


class TestParameterDictionaries(unittest.TestCase):
    def setUp(self):
        self.g = pp.CartGrid([3, 2])

    def test_default_flow_dictionary(self):
        """ Test the default flow dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        dictionary = dicts.flow_dictionary(self.g)
        self.check_default_flow_dictionary(dictionary)

    def test_default_transport_dictionary(self):
        """ Test the default transport dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        # The default darcy_flux needs face normals:
        self.g.compute_geometry()
        dictionary = dicts.transport_dictionary(self.g)
        # Check that all parameters have been added.
        p_list = [
            "aperture",
            "porosity",
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
        unitary_parameters = ["aperture", "porosity", "mass_weight"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        zeros = np.zeros(self.g.num_faces)
        self.assertTrue(np.all(np.isclose(dictionary["darcy_flux"], zeros)))

    def test_default_mechanics_dictionary(self):
        """ Test the default mechanics dictionary.

        Check that the correct parameters are present, and sample some of the values
        and check that they are correct.
        """
        dictionary = dicts.mechanics_dictionary(self.g)
        # Check that all parameters have been added.
        p_list = [
            "aperture",
            "porosity",
            "source",
            "time_step",
            "fourth_order_tensor",
            "bc",
            "bc_values",
            "slip_distance",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = ["aperture", "porosity"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        zeros = np.zeros(self.g.num_faces * self.g.dim)
        self.assertTrue(np.all(np.isclose(dictionary["slip_distance"], zeros)))
        self.assertEqual(dictionary["bc"].bc_type, "vectorial")

    def test_initialize_default_data(self):
        """ Test default flow data initialization with default values.

        initialize_data returns a data dictionary d with default "keyword" Parameters
        stored in d["parameters"].
        """
        data = pp.initialize_default_data(self.g, {}, parameter_type="flow")
        self.check_default_flow_dictionary(data[pp.PARAMETERS]["flow"])

    def test_initialize_default_data_specified(self):
        """ Test transport data initialization with default and specified values.

        We specify
            "porosity": There is a default value with the same type (and shape) as the
                value we pass.
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
        self.assertTrue(np.all(np.isclose(dictionary["porosity"], zeros)))
        self.assertAlmostEqual(dictionary["bc"], 15)

    def test_initialize_default_data_other_keyword(self):
        """ Test transport data initialization with keyword differing from
        parameter_type.

        We specify "foo", for which there is no default value.
        We check that the default value of porosity is indeed set.
        """
        specified_parameters = {"foo": "bar"}
        # The default darcy_flux needs face normals:
        self.g.compute_geometry()
        data = pp.initialize_default_data(
            self.g, {}, "transport", specified_parameters, keyword="not_transport"
        )
        dictionary = data[pp.PARAMETERS]["not_transport"]
        self.assertEqual(dictionary["foo"], "bar")
        ones = np.ones(self.g.num_cells)
        self.assertTrue(np.all(np.isclose(dictionary["porosity"], ones)))

    def test_initialize_data_specified(self):
        """ Test transport data initialization without default values.

        We specify
            "porosity": There is a default value with the same type (and shape) as the
                value we pass.
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
            "aperture",
            "porosity",
            "mass_weight",
            "source",
            "time_step",
            "second_order_tensor",
            "bc",
            "bc_values",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = ["aperture", "porosity", "mass_weight"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        self.assertTrue(
            np.all(np.isclose(dictionary["second_order_tensor"].values[2, 2], ones))
        )


if __name__ == "__main__":
    unittest.main()
