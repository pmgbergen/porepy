
import abc
from typing import List, Callable, Optional, Union, Tuple

from porepy import Grid
from porepy.numerics.mixed_dim.discretization_objects import NodeDiscretization, CouplingDiscretization


class DiscretizationFilter(abc.ABC):

    @abc.abstractmethod
    def node_filter(
            self,
            discr: NodeDiscretization,
            node_or_edge: Union[Grid, Tuple[Grid, Grid]],
    ) -> bool:
        pass

    @abc.abstractmethod
    def edge_coupling_filter(
            self,
            discr: CouplingDiscretization,
    ) -> bool:
        pass


class TrueFilter(DiscretizationFilter):

    def node_filter(self, *_) -> bool:
        return True

    def edge_coupling_filter(self, *_) -> bool:
        return True


class ListFilter(DiscretizationFilter):
    def __init__(self, variable_list, term_list):
        self.variable_list: Optional[List[str]] = variable_list
        self.term_list: Optional[List[str]] = term_list

        self.var_filter: Callable[[str], bool] = self.make_filter(self.variable_list)
        self.term_filter: Callable[[str], bool] = self.make_filter(self.term_list)

    def node_filter(
            self, discr: NodeDiscretization, _,
    ) -> bool:
        vf, tf = self.var_filter, self.term_filter
        return vf(discr.var) and tf(discr.term_key)

    def edge_coupling_filter(
            self, discr: CouplingDiscretization,
    ) -> bool:
        vf, tf = self.var_filter, self.term_filter
        if discr.g_h and discr.g_l:
            return (
                vf(discr.master_var)
                and vf(discr.slave_var)
                and vf(discr.edge_var)
            )

    def make_filter(self, var_term_list: Optional[List[str]]) -> Callable[[str], bool]:
        """ Construct a filter for variable and terms

        The result is a callable which takes one argument (a string).
        I
        filter is a list of strings.
        """
        def return_true(s):
            return True

        if not var_term_list:
            # If not variable or term list is passed, return a Callable
            # that always returns True.
            return return_true

        def _var_term_filter(x):
            include = set(key for key in var_term_list if not key.startswith("!"))
            exclude = set(key[1:] for key in var_term_list if key.startswith("!"))
            if include:
                # Keep elements only in include.
                include.difference_update(exclude)
                return x in include
            elif exclude:
                # Keep elements not in exclude
                return x not in exclude

        return _var_term_filter
