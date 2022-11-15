"""Preliminary class for the Peng-Robinson equation of state."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["PengRobinsonMixture"]


class PengRobinsonMixture:
    
    @property
    def c2(self) -> float:
        return -7.5
    
    @property
    def c1(self) -> float:
        return 17.

    @property
    def c0(self) -> float:
        return -10.3

    @property
    def characteristic_discriminant(self) -> float:
        return (
            self.c2**2 * self.c1**2
            - 4 * self.c1**3
            - 4 * self.c2**3 * self.c0
            - 27 * self.c0**2
            + 18 * self.c2 * self.c1 * self.c0
        )
    
    @property
    def _Q(self) -> float:
        return (3 * self.c1 - self.c2**2) / 9.

    @property
    def _R(self) -> float:
        return (9 * self.c2 * self.c1 - 27 * self.c0 - 2 * self.c2**3) / 54.

    @property
    def _D(self) -> float:
        return self._Q**3 + self._R**2

    @property
    def _S(self) -> float:
        return (self._R + self._D**.5)**(1/3)
    
    @property
    def _T(self) -> float:
        return (self._R - self._D**.5)**(1/3)

    @property
    def Z1(self) -> float:
        return - self.c2 / 3 + self._S + self._T
    
    @property
    def Z2(self) -> float:
        return - self.c2 / 3 - (self._S + self._T) / 2 + 1j * math.sqrt(3) / 2 * (self._S - self._T)

    @property
    def Z3(self) -> float:
        return - self.c2 / 3 - (self._S + self._T) / 2 - 1j * math.sqrt(3) / 2 * (self._S - self._T)
