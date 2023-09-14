import porepy as pp

from abc import ABC, abstractmethod
import numpy as np


from typing import Callable, Sequence


class BoundaryConditionMixin(ABC):
    """TODO"""

    mdg: pp.MixedDimensionalGrid

    @abstractmethod
    def set_boundary_conditions(self, initial: bool) -> None:
        """This method is called before a new time step to set the values of the
        boundary conditions. The specific boundary condition values should be updated in
        concrete overrides. By default, it does nothing.

        Note:
            One can use the convenience method `_update_boundary_condition` for each
            boundary condition value.

        """

    def _update_boundary_condition(
        self,
        name: str,
        function: Callable[[Sequence[pp.BoundaryGrid]], np.ndarray],
        initial: bool,
    ) -> None:
        """TODO

        TODO: Somewhere we must have an assertion that there is only 1 time step behind
        stored, and 1 iterate.
        """

        for bg, data in self.mdg.boundaries(return_data=True):
            # Set the known time step value.
            if initial:
                vals = function([bg])
                pp.set_solution_values(
                    name=name, values=vals, data=data, time_step_index=0
                )
            else:
                vals = data[pp.ITERATE_SOLUTIONS][name][0]
                pp.set_solution_values(
                    name=name, values=vals, data=data, time_step_index=0
                )
            # Set the iterate value.
            vals = function([bg])
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)

    def make_boundary_operator(
        self, name: str, domains: Sequence[pp.BoundaryGrid]
    ) -> pp.ad.Operator:
        """TODO
        Yury: this function wraps two line of code, but I decided to make it, because we
        later might decide to do something more / else here.

        """
        assert all(isinstance(x, pp.BoundaryGrid) for x in domains)
        return pp.ad.TimeDependentDenseArray(name=name, domains=domains)
