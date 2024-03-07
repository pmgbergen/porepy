"""Module collecting useful convergence criteria for integration in solution strategy.

These can also serve as inspiration for how to define custom criteria to override
'variable_norm' in solution_strategy.

"""

from abc import ABC
from functools import partial

import numpy as np

import porepy as pp


class BaseMetric(ABC):
    """Base class for a metric on algebraic variant of mixed-dimensional variables."""

    def variable_norm(self, solution: np.ndarray) -> float:
        """Base method for measuring the size of physical states (increments, solution etc.)

        Parameters:
            solution: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of solution

        """
        raise NotImplementedError("This is an abstract class.")


class EuclideanMetric(BaseMetric):
    """Purely algebraic metric (blind to physics and dimension).

    Simple but fairly robust convergence criterion. More advanced options are
    e.g. considering errors for each variable and/or each grid separately,
    possibly using _l2_norm_cell

    We normalize by the size of the solution vector as proxy for domain size.

    """

    def variable_norm(self, solution: np.ndarray) -> float:
        """Implementation of Euclidean l2 norm of the full vector, scaled by vector size.

        Parameters:
            solution: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of solution

        """
        return np.linalg.norm(solution) / np.sqrt(solution.size)


class LebesgueMetric(BaseMetric):
    """Dimension-consistent Lebesgue metric (blind to physics), but separates dimensions."""

    def variable_norm(self, solution: np.ndarray) -> float:
        """Implementation of Lebesgue L2 norm of the physical solution.

        Parameters:
            solution: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of solution

        """
        # Initialize container for collecting separate L2 norms (squarred).
        integrals_squarred = []

        # Convert algebraic solution vector to mixed dimensional variable
        # TODO - The current implementation utilizes the datastructures in place, which is
        # not very elegant...the creation of a isolated MixedDimensionalVariable without
        # data transfer would be better.
        cached_variable_values = self.equation_system.get_variable_values(
            iterate_index=0
        )
        self.equation_system.set_variable_values(solution, iterate_index=0)

        # Treat each variable separately - with this separate also dimensions
        for variable in self.equation_system.variables:

            # Compute square of the L2 norm taking into account volume and specific volume
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, variable.dim), "l2_norm")
            sd = variable.domain
            integral_squarred = np.sum(
                self.volume_integral(l2_norm(variable) ** 2, [sd], dim=1).value(
                    self.equation_system
                )
            )

            # Collect the L2 norm squared.
            integrals_squarred.append(integral_squarred)

        # Reset the cached variable values
        self.equation_system.set_variable_values(
            cached_variable_values, iterate_index=0
        )

        # Squash all results by employing a consistent L2 approach.
        return np.sqrt(np.sum(integrals_squarred))
