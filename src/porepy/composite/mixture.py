import abc
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
    def phases(self):  # -> Generator[Phase, None, None]:
        """
        Yields:
            Phases modelled by the composition class.
        """
        for P in self._phases:
            yield P

    def get_phase(self, phase_id):
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

    def mixture_for_subdomain(self, subdomain): 
        for phase in self.phases:
            phase.subdomain = subdomain
        return self

    def apply_constraint(self, ell):
        """
        - hardcoded for two phase flow
        - this is not the right place for this function
        - TODO: redo constraint
        """

        if ell == 0:  # sorry...
            m = 1
        else:  # ell == 1
            m = 0

        self.get_phase(ell).apply_constraint = False
        self.get_phase(m).apply_constraint = True


'''
class MixtureMixin:
    """
    in the model i expect:
    self._phases: list[Phase] = []
    """

    def num_phases(self) -> int:
        """ """
        return len(self._phases)

    def phases(self):
        """ """
        for P in self._phases:
            yield P

    def get_phase(self, phase_id):
        """ """
        return self._phases[phase_id]

    def apply_constraint(self):
        """
        - to be called in after_nonlinear_interation. and in prepare_simulation i guess... you have to define the saturation_m before newton

        - hardcoded for two phase flow
        """

        if self.ell == 0:  # sorry...
            m = 1
        else:  # ell == 1
            m = 0

        self.get_phase(m)._s = 1 - self.get_phase(m).saturation
'''
