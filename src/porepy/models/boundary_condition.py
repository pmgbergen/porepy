import porepy as pp

from abc import ABC, abstractmethod
import numpy as np


from typing import Callable, Sequence


class BoundaryConditionMixin(ABC):
    """TODO"""

    mdg: pp.MixedDimensionalGrid

    @abstractmethod
    def update_boundary_conditions(self) -> None:
        """This method is called before a new time step to set the values of the
        boundary conditions. The specific boundary condition values should be updated in
        concrete overrides. By default, it does nothing.

        Note:
            One can use the convenience method `update_boundary_condition` for each
            boundary condition value.

        """

    def update_boundary_condition(
        self,
        name: str,
        function: Callable[[pp.BoundaryGrid], np.ndarray],
    ) -> None:
        """TODO

        TODO: Somewhere we must have an assertion that there is only 1 time step behind
        stored, and 1 iterate.
        """
        for bg, data in self.mdg.boundaries(return_data=True):
            # Set the known time step values.
            if name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = data[pp.ITERATE_SOLUTIONS][name][0]
            else:
                # No previous time step exists. The method was called during
                # the initialization.
                vals = function(bg)
            pp.set_solution_values(name=name, values=vals, data=data, time_step_index=0)

            # Set the unknown time step values.
            vals = function(bg)
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)

    def create_boundary_operator(
        self, name: str, domains: Sequence[pp.BoundaryGrid]
    ) -> pp.ad.Operator:
        """TODO
        Yury: this function wraps two line of code, but I decided to make it, because we
        later might decide to do something more / else here.

        """
        assert all(isinstance(x, pp.BoundaryGrid) for x in domains)
        return pp.ad.TimeDependentDenseArray(name=name, domains=domains)
