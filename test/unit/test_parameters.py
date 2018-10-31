import numpy as np
import unittest

from porepy.grids.structured import CartGrid
from porepy.params.data import Parameters


class TestGettersAndSetters(unittest.TestCase):
    def setUp(self):
        self.g = CartGrid([3, 2])
        self.nc = self.g.num_cells
        self.v = np.ones(self.nc)

    def test_biot_alpha_default(self):
        p = Parameters(self.g)
        self.assertEqual(p.biot_alpha, 1)
        self.assertEqual(p.get_biot_alpha(), 1)

    def test_biot_alpha_1(self):
        p = Parameters(self.g)
        p.biot_alpha = 0.5
        self.assertEqual(p.biot_alpha, 0.5)
        self.assertEqual(p.get_biot_alpha(), 0.5)

    def test_biot_alpha_2(self):
        p = Parameters(self.g)
        p.set_biot_alpha(0.2)
        self.assertEqual(p.biot_alpha, 0.2)
        self.assertEqual(p.get_biot_alpha(), 0.2)

    def test_biot_alpha_assertion(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_biot_alpha(1.2)
            p.set_biot_alpha(-1)

    #####

    def test_fluid_viscosity_default(self):
        p = Parameters(self.g)
        self.assertEqual(p.fluid_viscosity, 1)
        self.assertEqual(p.get_fluid_viscosity(), 1)

    def test_fluid_viscosity_attribute(self):
        p = Parameters(self.g)
        p.fluid_viscosity = 0.5
        self.assertEqual(p.fluid_viscosity, 0.5)
        self.assertEqual(p.get_fluid_viscosity(), 0.5)

    def test_fluid_viscosity_setter(self):
        p = Parameters(self.g)
        p.set_fluid_viscosity(0.2)
        self.assertEqual(p.fluid_viscosity, 0.2)
        self.assertEqual(p.get_fluid_viscosity(), 0.2)

    def test_fluid_viscosity_assertion(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_fluid_viscosity(-1)

    #####

    def test_fluid_compr_default(self):
        p = Parameters(self.g)
        self.assertEqual(p.fluid_compr, 0)
        self.assertEqual(p.get_fluid_compr(), 0)

    def test_fluid_compr_attribute(self):
        p = Parameters(self.g)
        p.fluid_compr = 0.5
        self.assertEqual(p.fluid_compr, 0.5)
        self.assertEqual(p.get_fluid_compr(), 0.5)

    def test_fluid_compr_setter(self):
        p = Parameters(self.g)
        p.set_fluid_compr(0.2)
        self.assertEqual(p.fluid_compr, 0.2)
        self.assertEqual(p.get_fluid_compr(), 0.2)

    def test_fluid_compr_assertion(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_fluid_compr(-1)

    #####
    def test_background_stress_default(self):
        p = Parameters(self.g)
        self.assertTrue(np.allclose(p.background_stress_mechanics, 0))
        for name in p.known_physics:
            self.assertTrue(np.allclose(p.get_background_stress(name), 0))

    def test_background_stress_setter(self):
        p = Parameters(self.g)
        for name in p.known_physics:
            p.set_background_stress(name, 0.5 * np.ones((3, 3)))
        self.assertTrue(np.allclose(p.background_stress_mechanics, 0.5))
        for name in p.known_physics:
            self.assertTrue(np.allclose(p.get_background_stress(name), 0.5))

    #####

    def _validate_ap(self, p, val=1):
        self.assertTrue(np.allclose(p.aperture, val * self.v))
        self.assertTrue(np.allclose(p.get_aperture(), val * self.v))

    def test_aperture_default(self, val=1):
        p = Parameters(self.g)
        self._validate_ap(p)
        # Set by scalar

    def test_aperture_set(self):
        p = Parameters(self.g)
        p.set_aperture(2)
        self._validate_ap(p, 2)
        # Set by variable name

    def test_aperture_attribute(self):
        p = Parameters(self.g)
        p.aperture = 3
        self._validate_ap(p, 3)
        # set by vector

    def test_aperture_set_vector(self):
        p = Parameters(self.g)
        p.set_aperture(4 * self.v)
        self._validate_ap(p, 4)
        # Set vector by variable name

    def test_aperture_set_vector_attribute(self):
        p = Parameters(self.g)
        p.aperture = 5 * self.v
        self._validate_ap(p, 5)

    def test_aperture_assertion(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_aperture(-1)

    def test_aperture_assertion_vector(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_aperture(self.v * -1)

    ##########

    def _validate_porosity(self, p, val=1):
        self.assertTrue(np.allclose(p.porosity, val * self.v))
        self.assertTrue(np.allclose(p.get_porosity(), val * self.v))

    def test_porosity_default(self, val=1):
        p = Parameters(self.g)
        self._validate_porosity(p)
        # Set by scalar

    def test_porosity_set(self):
        p = Parameters(self.g)
        p.set_porosity(.1)
        self._validate_porosity(p, .1)
        # Set by variable name

    def test_porosity_attribute(self):
        p = Parameters(self.g)
        p.porosity = .3
        self._validate_porosity(p, .3)
        # set by vector

    def test_porosity_set_vector(self):
        p = Parameters(self.g)
        p.set_porosity(.4 * self.v)
        self._validate_porosity(p, .4)
        # Set vector by variable name

    def test_porosity_set_vector_attribute(self):
        p = Parameters(self.g)
        p.porosity = .5 * self.v
        self._validate_porosity(p, .5)

    def test_porosity_assertion(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_porosity(-1)
            p.set_porosity(2)

    def test_porosity_assertion_vector(self):
        p = Parameters(self.g)
        with self.assertRaises(ValueError):
            p.set_porosity(self.v * -1)
            p.set_porosity(self.v * 2)

    #####

    if __name__ == "__main__":
        unittest.main()
