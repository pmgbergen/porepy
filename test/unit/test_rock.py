from __future__ import division

import numpy as np
import unittest

import porepy as pp


class TestRock(unittest.TestCase):
    def test_lame_from_young_poisson(self):
        e = 1
        nu = 0.1
        lam, mu = pp.params.rock.lame_from_young_poisson(e, nu)
        self.assertTrue(np.allclose(lam, 0.11363636363636363))
        self.assertTrue(np.allclose(mu, 0.45454545454545453))

    def test_poisson_from_lame(self):
        lam = 1
        mu = 0.5
        nu = pp.params.rock.poisson_from_lame(mu, lam)
        self.assertEqual(nu, 1 / 3)

    def test_unit_rock(self):
        R = pp.UnitRock()
        for prop, value in vars(R).items():
            if prop == "POISSON_RATIO":
                self.assertTrue(np.allclose(value, 0.25))
            else:
                self.assertTrue(np.allclose(value, 1))

    def test_sand_stone(self):
        R = pp.SandStone()
        self.assertEqual(R.PERMEABILITY, 1 * 9.869233e-13)
        self.assertEqual(R.POROSITY, 0.2)
        self.assertEqual(R.YOUNG_MODULUS, 5 * pp.KILOGRAM / pp.CENTI ** 2 * 1e5)
        self.assertEqual(R.POISSON_RATIO, 0.1)
        self.assertTrue(np.allclose(R.LAMBDA, 568181818.1818181))
        self.assertTrue(np.allclose(R.MU, 2272727272.7272725))

    def test_granite(self):
        R = pp.Granite()
        self.assertEqual(R.PERMEABILITY, 1e-8 * pp.DARCY)
        self.assertEqual(R.POROSITY, 0.01)
        self.assertEqual(R.YOUNG_MODULUS, 40 * pp.GIGA * pp.PASCAL)
        self.assertEqual(R.POISSON_RATIO, 0.2)
        self.assertEqual(R.DENSITY, 2700 * pp.KILOGRAM / pp.METER ** 3)
        self.assertTrue(np.allclose(R.LAMBDA, 11111111111.1111112))
        self.assertTrue(np.allclose(R.MU, 16666666666.6666667))

    def test_shale(self):
        R = pp.Shale()
        self.assertEqual(R.PERMEABILITY, 1e-5 * 9.869233e-13)
        self.assertEqual(R.POROSITY, 0.01)
        self.assertEqual(R.YOUNG_MODULUS, 1.5 * pp.KILOGRAM / pp.CENTI ** 2 * 1e5)
        self.assertEqual(R.POISSON_RATIO, 0.3)
        self.assertTrue(np.allclose(R.LAMBDA, 865384615.3846153))
        self.assertTrue(np.allclose(R.MU, 576923076.9230769))


if __name__ == "__main__":
    unittest.main()
