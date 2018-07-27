from __future__ import division

import numpy as np
import unittest

from porepy.grids.structured import CartGrid
from porepy.params import rock


class TestRock(unittest.TestCase):
    def test_lame_from_young_poisson(self):
        e = 1
        nu = 0.1
        lam, mu = rock.lame_from_young_poisson(e, nu)
        self.assertEqual(lam, 0.11363636363636363)
        self.assertEqual(mu, 0.45454545454545453)

    def test_poisson_from_lame(self):
        lam = 1
        mu = 0.5
        nu = rock.poisson_from_lame(mu, lam)
        self.assertEqual(nu, 1 / 3)

    def test_unit_rock(self):
        R = rock.UnitRock()
        for _, value in vars(R).items():
            self.assertTrue(np.allclose(value, 1))

    def test_sand_stone(self):
        R = rock.SandStone()
        self.assertTrue(R.PERMEABILITY, 1 * 9.869233e-13)
        self.assertTrue(R.POROSITY, 0.2)
        self.assertTrue(R.YOUNG_MODULUS, 50)
        self.assertTrue(R.POISSON_RATIO, 0.1)
        self.assertTrue(R.LAMBDA, 568181818.1818181)
        self.assertTrue(R.MU, 2272727272.7272725)

    def test_granite(self):
        R = rock.SandStone()
        self.assertTrue(R.PERMEABILITY, 1e-8 * 9.869233e-13)
        self.assertTrue(R.POROSITY, 0.01)
        self.assertTrue(R.YOUNG_MODULUS, 50)
        self.assertTrue(R.POISSON_RATIO, 0.2)
        self.assertTrue(R.LAMBDA, 1388888888.8888888)
        self.assertTrue(R.MU, 2083333333.3333335)

    def test_shale(self):
        R = rock.SandStone()
        self.assertTrue(R.PERMEABILITY, 1e-5 * 9.869233e-13)
        self.assertTrue(R.POROSITY, 0.01)
        self.assertTrue(R.YOUNG_MODULUS, 15)
        self.assertTrue(R.POISSON_RATIO, 0.3)
        self.assertTrue(R.LAMBDA, 865384615.3846153)
        self.assertTrue(R.MU, 576923076.9230769)
