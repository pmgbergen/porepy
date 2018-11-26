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
        # The default discharge needs face normals:
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
            "discharge",
            "mass_weight",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = ["aperture", "porosity", "mass_weight"]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        zeros = np.zeros(self.g.num_faces)
        self.assertTrue(np.all(np.isclose(dictionary["discharge"], zeros)))

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

    def test_initialize_data_default(self):
        """ Test default flow data initialization.

        initialize_data returns a data dictionary d with default "keyword" Parameters
        stored in d["parameters"].
        """
        data = pp.initialize_data({}, self.g, keyword="flow")
        self.check_default_flow_dictionary(data[pp.keywords.PARAMETERS]["flow"])

    def test_initialize_data_specified(self):
        """ Test transport data initialization.

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
        # The default discharge needs face normals:
        self.g.compute_geometry()
        data = pp.initialize_data({}, self.g, "transport", specified_parameters)
        dictionary = data[pp.keywords.PARAMETERS]["transport"]
        success = dictionary["foo"] == "bar"
        zeros = np.zeros(self.g.num_cells)
        success *= np.all(np.isclose(dictionary["porosity"], zeros))
        success *= dictionary["bc"] == 15
        assert success

    def check_default_flow_dictionary(self, dictionary):
        # Check that all parameters have been added.
        p_list = [
            "aperture",
            "porosity",
            "fluid_compressibility",
            "mass_weight",
            "source",
            "time_step",
            "second_order_tensor",
            "bc",
            "bc_values",
        ]
        [self.assertIn(parameter, dictionary) for parameter in p_list]
        # Check some of the values:
        unitary_parameters = [
            "aperture",
            "porosity",
            "fluid_compressibility",
            "mass_weight",
        ]
        ones = np.ones(self.g.num_cells)
        for parameter in unitary_parameters:
            self.assertTrue(np.all(np.isclose(dictionary[parameter], ones)))
        self.assertTrue(
            np.all(np.isclose(dictionary["second_order_tensor"].values[2, 2], ones))
        )


if __name__ == "__main__":
    unittest.main()
