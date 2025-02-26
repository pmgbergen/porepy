"""Module containing mixins for providing initial values during a model setup."""

from __future__ import annotations

import porepy as pp


class InitialConditionMixin(pp.PorePyModel):
    """Convenience class to provide dedicated methods for setting initial values of
    model variables.

    Provides an override of the solution strategies ``initial_condition``, executing
    subsequently a routine to set initial values for primary variables.

    This clustering is used for initialization routines, where the order of setting
    IC is critical.

    Example:
        Let's consider a system with two variables, ``x, y``, where ``y`` is a secondary
        variable and we want to initialize the system using a relation ``y = y(x)``.

        .. code::python

            class ICPrimary(InitialConditionMixin):

                def set_initial_values_primary_variables(self) -> None:
                    super().set_initial_values_primary_variables()

                    for sd in self.mdg.subdomains():
                        self.equation_system.set_variable_values(
                            self.initial_x(sd),
                            [self.x([sd])],
                            iterate_index=0,
                        )

                def initial_x(self, sd: pp.Grid) -> np.ndarray:
                    # proceed to return some value
                    ...

            class ICSecondary(InitialConditionMixin):

                def initial_condition(self) -> None:
                    super().initial_condition()

                    for sd in self.mdg.subdomains():
                        self.equation_system.set_variable_values(
                            self.initial_y(sd),
                            [self.y([sd])],
                            iterate_index=0,
                        )

                def initial_y(self, sd: pp.Grid) -> np.ndarray:
                    x = self.x([sd]).value(self.equation_system)
                    # proceed to return some value depending on x
                    ...

            class MyIC(ICSecondary, ICPrimary):
                ...

        Notice that in all initialization methods, ``super()`` is called first. For the
        model using ``MyIC``, because of the order of inheritance, the code in
        ``ICSecondary`` is executed first, which in return executes the code of
        ``ICPrimary`` first. I.e., the update order is the reverse order in the
        inheritance tree. Thus, when using this approach, the IC update for ``y`` can
        reliably fetch the latest values for ``x`` on the boundary.

        Notice also, that ``set_initial_values_primary_variables`` has also a
        ``super()`` call on top. This makes it compatible in the case of a third,
        primary variable which should be initialized in the same sub-routine as ``x``.

    """

    def initial_condition(self) -> None:
        """Interface method for the solution strategy to be called to set initial values
        for all variables.

        Calls the methods :meth:`set_initial_values_primary_variables`.

        Can be overridden to set other initial conditions after a super-call.

        Important:
            The user must set initial values at ``iterate_index=0``. The solution
            strategy copies said values by default to all other indices in order to get
            a runable model.

        """
        self.set_initial_values_primary_variables()

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for primary variables.

        The base method does nothing except provide an interface and compatibility for
        super-calls to model-specific initialization procedures.

        """
