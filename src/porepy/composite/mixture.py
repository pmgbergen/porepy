import abc
from typing import Any, Generator, Literal, Optional
import porepy as pp
from porepy.composite.phase import Phase


class Mixture(abc.ABC):
    """
    https://github.com/pmgbergen/porepy/blob/539f876e3cda7f5b911db9784210fceba19980ea/src/porepy/composite/mixture.py#L866
    """

    def __init__(self) -> None:
        """ """
        self._phases: list[Phase] = []

    def __str__(self) -> str:
        """ """
        out = "Two phase flow, constant density. See https://github.com/pmgbergen/porepy/blob/539f876e3cda7f5b911db9784210fceba19980ea/src/porepy/composite/mixture.py#L866"
        return out

    @property
    def num_phases(self) -> int:
        """Number of *modelled* phases in the composition."""
        return len(self._phases)

    @property
    def phases(self) -> Generator[Phase, None, None]:
        """
        not really used...
        """
        for P in self._phases:
            yield P

    def get_phase(self, phase_id: int) -> Phase:
        """
        first phase = 0, second phase = 1
        """
        return self._phases[phase_id]

    @property
    def reference_phase(self) -> Phase:
        """ """
        assert self._phases, "No phases present in mixture."
        return self._phases[0]

    def add(self, phases: list[Phase]) -> None:
        """ """
        for phase in phases:
            self._phases.append(phase)

    def mixture_for_subdomain(self, subdomain: pp.Grid):
        """this is horrible"""
        for phase in self.phases:
            phase.subdomain = subdomain
        return self

    def apply_constraint(self, ell: int) -> None:
        """
        - hardcoded for two phase flow
        - TODO: this function is not located in the best place
        """

        if ell == 0:  # sorry...
            m = 1
        else:  # ell == 1
            m = 0

        self.get_phase(ell).apply_constraint = False
        self.get_phase(m).apply_constraint = True
