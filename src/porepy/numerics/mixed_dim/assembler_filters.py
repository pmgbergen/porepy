""" Discretization filters to allow partial discretization or assembly.

Content:
    AllPassFilter does no filtering.
    ListFilter: filters based on grid quantities, variables and lists.

New filter classes should be implemented by subclassing the abstract base
class AssemblerFilter.

Credits: Design idea and main implementation by Haakon Ervik.

"""
import abc
from typing import Callable, List, Optional, Tuple, Union

from porepy import Grid

# Discretizations can be defined either on a subdomain, on an
# edge (Tuple of two grids), or it is a coupling between
# two subdomains and an interface
grid_like_type = Union[
    Union[Grid, List[Grid]], Tuple[Grid, Grid], Tuple[Grid, Grid, Tuple[Grid, Grid]]
]


class AssemblerFilter(abc.ABC):
    """Abstract base class of filters for use with the Assembler."""

    @abc.abstractmethod
    def filter(
        self,
        grids: Optional[List[grid_like_type]] = None,
        variables: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
    ) -> bool:
        """Filter grids (in a general sense), variables and discretization terms.

        The filter should return true if the combination of grids, variables and
        terms are considered 'active' by the filter. Intended use is to allow the
        assembler to implement partial discretization or assembly.

        Parameters:
            grid: Grid-like quantity found in a pp.GridBucket.
                Can be either a Grid (GridBucket node), an interface (a GridBucket
                edge), or a combination of two neighboring grids and an interface.
            variables: List of variables.
            term: List of terms for discretization. See Assembler for further
                explanation.

        Returns:
            boolean: True if the grid-variable-term combination passes the filter.

        """


class AllPassFilter(AssemblerFilter):
    """All pass filter. The filter method always return True."""

    def filter(
        self,
        grids: Optional[List[grid_like_type]] = None,
        variables: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
    ) -> bool:
        """Filter grids (in a general sense), variables and discretization terms.

        The filter should return true if the combination of grids, variables and
        terms are considered 'active' by the filter. Intended use is to allow the
        assembler to implement partial discretization or assembly.

        Parameters:
            grid: Grid-like quantity found in a pp.GridBucket.
                Can be either a Grid (GridBucket node), an interface (a GridBucket
                edge), or a combination of two neighboring grids and an interface.
            variables: A variable, or a list of variables.
            term: List of terms for discretizations. See Assembler for further
                explanation.

        Returns:
            boolean: True if the grid-variable-term combination passes the filter.

        """
        return True


class ListFilter(AssemblerFilter):
    """Filter based on lists of (generalized) grids, variables and terms.

    The filter is initialized with lists of grids (specification below),
    variabels and terms that should pass the filter. The filter function will pass a
    combination of a grid, a set of variables and a term if they are all found
    in the lists of acceptables.

    If a list of grids, variables and/or  terms are not provided at the time of
    initialization, all objects of this the unspecified type will pass the filter.
    Thus, if neither grids, variables nor terms are specified, the filter effectively
    becomes an AllPassFilter.

    NOTE: If a list (say of grids) is given as an empty list, the filter will become
    no-pass fliterThis is to cover cases where dimension-filtering on grids in a GridBucket
    returns a empty list, which should result in no-pass, not all-pass behavior.
    The behavior for variable and term lists is similar.

    Acceptable variables and terms can be specified as a negation with the
    syntax !variable_name. It is not possible to use both negated and standard
    specification of, say, variables, but negated variables combined with standard
    terms (or reverse) is permissible.

    The generalized grids should be one of
        i) grids: nodes in the GridBucket
        ii) interfaces: (Grid, Grid) tuples, edges in the GridBucket.
        iii) couplings: (Grid, Grid, (Grid, Grid)) tuples, so an edge, together
            with its neighboring subdomains.

    """

    def __init__(
        self,
        grid_list: Optional[List[grid_like_type]] = None,
        variable_list: Optional[List[str]] = None,
        term_list: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters:
            grid_list: List of grid-like objects that should pass the filter.
                See class documentation for specification.
            variable_list: List of variables to pass the filter.
            term_list: List of terms to pass the filter.

        """
        # Helper functions, needed for no-pass and all-pass behavior
        def return_false(s):
            return False

        def return_true(s):
            return True

        if grid_list is None:
            # There should be no filtering based on grids

            self._grid_filter = return_true

        else:
            if len(grid_list) == 0:
                # This is considered a no-pass filter.
                # This will for instance be the case if a GridBucket is filtered
                # on a dimension that is not present (will return an empty list)

                self._grid_filter = return_false

            else:
                # Non-trivial filter
                self._grid_filter = self._make_grid_filter(grid_list)

        if variable_list is None:
            self._var_filter: Callable[[Optional[List[str]]], bool] = return_true
        else:
            if len(variable_list) == 0:
                self._var_filter = return_false
            else:
                self._variable_list: List[str] = variable_list
                self._var_filter = self._make_string_filter(self._variable_list)

        if term_list is None:
            self._term_filter: Callable[[Optional[List[str]]], bool] = return_true
        else:
            if len(term_list) == 0:
                self._term_filter = return_false
            else:
                self._term_list: List[str] = term_list
                self._term_filter = self._make_string_filter(self._term_list)

    def filter(
        self,
        grids: Optional[List[grid_like_type]] = None,
        variables: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
    ):
        """Filter grids (in a general sense), variables and discretization terms.

        See class documentation for how to use the filter.

        Parameters:
            grid: Grid-like quantity found in a pp.GridBucket.
                Can be either a Grid (GridBucket node), an interface (a GridBucket
                edge), or a combination of two neighboring grids and an interface.
            variables: A variable, or a list of variables. A list will be passed
                for off-diagonal terms (internal to nodes or edges), and for
                coupling terms.
            term: Term for a discretization. See Assembler for further explanation.

        Returns:
            boolean: True if the grid-variable-term combination passes the filter.

        """
        return (
            self._grid_filter(grids)
            and self._var_filter(variables)
            and self._term_filter(terms)
        )

    def _parse_grid_list(
        self, grid_list: List[grid_like_type]
    ) -> Tuple[List, List, List]:
        assert grid_list is not None

        nodes = []
        edges = []
        couplings = []
        self._grid_list = grid_list

        for g in grid_list:
            if isinstance(g, Grid):
                nodes.append(g)
            elif isinstance(g, tuple) and len(g) == 2:
                if not (isinstance(g[0], Grid) and isinstance(g[1], Grid)):
                    raise ValueError(f"Invalid grid-like object for filtering {g}")
                edges.append(g)
                # Also append the reverse ordering of the grids.
                edges.append((g[1], g[0]))
            else:
                if not len(g) == 3:
                    raise ValueError(f"Invalid grid-like object for filtering {g}")
                couplings.append(g)
        return nodes, edges, couplings

    def _make_grid_filter(self, grid_list):

        nodes, edges, couplings = self._parse_grid_list(grid_list)
        self._nodes: List[Grid] = nodes
        self._edges: List[Tuple[Grid, Grid]] = edges
        self._couplings: List[Tuple[Grid, Grid, Tuple[Grid, Grid]]] = couplings

        def _grid_filter(gl):
            if not isinstance(gl, list):
                gl = [gl]
            for g in gl:
                if (
                    g not in self._nodes
                    and g not in self._edges
                    and g not in self._couplings
                ):
                    return False
            return True

        return _grid_filter

    def _make_string_filter(
        self, var_term_list: Optional[List[str]] = None
    ) -> Callable[[Optional[List[str]]], bool]:
        """Construct a filter used to operate on strings

        The result is a callable which takes one argument (a string).

        filter is a list of strings.
        """

        def return_true(s):
            return True

        if not var_term_list:
            # If not variable or term list is passed, return a Callable
            # that always returns True.
            return return_true

        def _var_term_filter(x):
            if not x:
                # Filtering of a None type is always positive
                return True

            include = set(key for key in var_term_list if not key.startswith("!"))
            exclude = set(key[1:] for key in var_term_list if key.startswith("!"))

            if include and exclude:
                raise ValueError(
                    "A filter cannot combine negated and standard variables"
                )
            if include:
                return all([y in include for y in x])
            elif exclude:
                # Keep elements not in exclude
                return all([y not in exclude for y in x])

            # This should not be possible (either the strings start with !, or they don't)
            raise ValueError("Error in filter specification")

        return _var_term_filter

    def __repr__(self) -> str:
        s = "ListFilter based on"
        if self._nodes or self._edges or self._couplings:
            s += " (generalized) grids,"
        if self._variable_list:
            s += " variables "
        if self._term_list:
            s += " terms "

        s += "\n"
        s += "Filter has:\n"
        if self._nodes:
            s += f"In total {len(self._nodes)} standard grids\n"
        if self._edges:
            s += f"In total {len(self._edges)} interfaces\n"
        if self._couplings:
            s += f"In total {len(self._couplings)} geometric couplings\n"

        if self._variable_list:
            s += "Variables: "
            for v in self._variable_list:
                s += f"{v}, "
            s += "\n"

        if self._term_list:
            s += "Terms: "
            for t in self._term_list:
                s += f"{t}, "
            s += "\n"

        return s
