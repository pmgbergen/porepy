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
        please, for the phases, the counter starts from 1
        please, please, it becomes a mess. So, first phase = 0, second phase = 1
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

    def mixture_for_subdomain(self, equation_system, subdomain):
        for phase in self.phases:
            phase.equation_system = equation_system  # i think is useless, you have to set equation system before bcs you need it for saturation_operator
            phase.subdomain = subdomain
        return self

    def apply_constraint(self, ell, equation_system, subdomains):
        """
        - hardcoded for two phase flow

        - this is not the right place for this function

        - TODO: redo constraint
        """

        if ell == 0:  # sorry...
            m = 1
        else:  # ell == 1
            m = 0

        self.get_phase(ell)._s = equation_system.md_variable(
            "saturation", subdomains
        )  ### I'm making a shallow copy of mixed dim var. TODO: check that it is ok and you are actually changing both copies
        self.get_phase(m)._s = pp.ad.Scalar(1, "one") - equation_system.md_variable(
            "saturation", subdomains
        )


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
