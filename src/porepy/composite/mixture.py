import abc
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
    def phases(self):  # -> Generator[Phase, None, None]: TODO: check the output
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
        """Returns the reference phase.
        As of now, the first, added phase is declared as reference phase.
        The fraction of the reference can be eliminated from the system using respective
        flags.
        Raises:
            AssertionError: If no phases were added to the mixture.
            TODO: read description
        """

        assert self._phases, "No phases present in mixture."
        return self._phases[0]

    def add(self, phases: list[Phase]) -> None:
        """Adds one or multiple components and phases to the mixture.
        Components and phases must be added before the system can be set up.
        Important:
            This method is meant to be called only once per mixture!
            This is due to the creation of respective AD variables.
            By calling it twice, their reference is overwritten and the previous
            variables remain as dangling parts in the AD system.
        Parameters:
            component: Component(s) to be added to this mixture.
            phases: Phase(s) to be added to this mixture.
        Raises:
            ValueError: If a component or phase was instantiated using a different
                AD system than the one used for this composition.
            TODO: read description
        """

        for phase in phases:
            self._phases.append(phase)


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
