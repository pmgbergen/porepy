"""Module collecting useful convergence criteria for integration in solution strategy.

These can also serve as inspiration for how to define custom criteria to override
methods for computing norms in solution_strategy. But moreover, they can be used
for convergence studies etc.

Example:
    # Given a model class `MyModel`, to equip it with a custom metric to be used in
    # the solution strategy, one can override the corresponding method as follows:

    class MyNewModel(MyModel):

        def compute_nonlinear_increment_norm(self, solution: np.ndarray) -> float:
            # Method for computing the norm of the nonlinear increment during
            # `check_convergence`.
            return pp.LebesgueMetric().variable_norm(self, solution)

"""

from abc import ABC
from functools import partial
from typing import Callable

import numpy as np

import porepy as pp


class BaseMetric(ABC):
    """Base class for a metric on algebraic variant of mixed-dimensional variables."""

    def variable_norm(self, values: np.ndarray) -> float:
        """Base method for measuring the size of physical states (increments, solution etc.)

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of values

        """
        raise NotImplementedError("This is an abstract class.")


class EuclideanMetric(BaseMetric):
    """Purely algebraic metric (blind to physics and dimension).

    Simple but fairly robust convergence criterion. More advanced options are
    e.g. considering errors for each variable and/or each grid separately,
    possibly using _l2_norm_cell

    We normalize by the size of the vector as proxy for domain size.

    """

    def variable_norm(self, values: np.ndarray) -> float:
        """Implementation of Euclidean l2 norm of the full vector, scaled by vector size.

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of values

        """
        return np.linalg.norm(values) / np.sqrt(values.size)


class LebesgueMetric(BaseMetric):
    """Dimension-consistent Lebesgue metric (blind to physics), but separates dimensions."""

    equation_system: pp.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    volume_integral: Callable[[pp.ad.Operator, list[pp.Grid], int], pp.ad.Operator]
    """General volume integral operator, defined in `pp.BalanceEquation`."""

    def variable_norm(self, values: np.ndarray) -> float:
        """Implementation of mixed-dimensional Lebesgue L2 norm of a physical state.

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of values

        """
        # Initialize container for collecting separate L2 norms (squarred).
        integrals_squarred = []

        # Use the equation system to get a view onto mixed-dimensional data structures.
        # Compute the L2 norm of each variable separately, automatically taking into
        # account volume and specific volume
        for variable in self.equation_system.variables:

            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, variable.dim), "l2_norm")
            sd = variable.domain
            indices = self.equation_system.dofs_of(variable)
            ad_values = pp.ad.DenseArray(values[indices])
            integral_squarred = np.sum(
                self.volume_integral(l2_norm(ad_values) ** 2, [sd], 1).value(
                    self.equation_system
                )
            )

            # Collect the L2 norm squared.
            integrals_squarred.append(integral_squarred)

        # Squash all results by employing a consistent L2 approach.
        return np.sqrt(np.sum(integrals_squarred))
