"""Module containing mixins for providing initial values during a model setup."""

from __future__ import annotations

from .protocol import PorePyModel


class InitialConditionMixin(PorePyModel):
    """Convenience class to provide dedicated methods for setting initial values of
    model variables.

    Provides an override of the solution strategies ``initial_condition``, executing
    subsequently a routine to set initial values for primary variables.

    """

    def initial_condition(self) -> None:
        """Attaches to the implementation of ``initial_condition`` of the solution
        strategy.

        Calls the methods :meth:`set_initial_values_primary_variables` and copies values
        stored at iterate index 0 to all other time and iterate indices.

        """
        # First call super and let the solution strategy handle the initialization of
        # everything (likely with zeros).
        # Then run the functionality given by this class.
        super().initial_condition()
        self.set_initial_values_primary_variables()

        # updating variable values from current time step, to all previous and iterate
        val = self.equation_system.get_variable_values(iterate_index=0)
        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(
                val,
                iterate_index=iterate_index,
            )
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for primary variables.

        The base method does nothing except provide an interface and compatibility for
        super-calls to model-specific initialization procedures.

        """
        pass
