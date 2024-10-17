import porepy as pp
import numpy as np
from typing import Any

from porepy.numerics.nonlinear import line_search as ls
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class ConstraintLineSearchNonlinearSolver(
    ls.ConstraintLineSearch,
    ls.SplineInterpolationLineSearch,
    ls.LineSearchNewtonSolver,
):
    pass


class ConstraintFunctionsMomentumBalance(
    SquareDomainOrthogonalFractures, pp.momentum_balance.MomentumBalance
):
    """Enhance MomentumBalance for compatibility with the solver.

    The requirement for the constraint-based line search is that the model has the
    constraint functions implemented, see ConstraintLineSearch. This class adds the
    constraint functions to the MomentumBalance model.

    The values used are unitary, so we only test that the line search is performed
    without errors.

    """

    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the opening constraint.

        Negative for open fractures, positive for closed ones.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Indicator function as an operator.

        """
        nc = sum([g.num_cells for g in subdomains])
        ind = pp.ad.DenseArray(np.ones(nc))
        return ind

    def sliding_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the sliding constraint.

        Negative for sliding fractures, positive for non-sliding ones.

                Parameters:
            subdomains: List of subdomains.

        Returns:
            Indicator function as an operator.

        """
        nc = sum([g.num_cells for g in subdomains])
        ind = pp.ad.DenseArray(np.ones(nc))
        return ind


def test_line_search():
    """Test the line search on a model with constraint functions.

    The test is only to check that the line search is performed without errors. No
    checks are made on the results or that the line search does what it is supposed to.

    """
    model = ConstraintFunctionsMomentumBalance()
    solver_params = {
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "Global_line_search": True,
        "Local_line_search": True,
    }
    pp.run_time_dependent_model(model, solver_params)
