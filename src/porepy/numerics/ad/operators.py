""" Implementation of wrappers for Ad representations of several operators.
"""
from enum import Enum
from typing import Optional, List, Any, Tuple, Dict, Union, Callable
from itertools import count

import numpy as np
import porepy as pp
import networkx as nx
import scipy.sparse as sps
import matplotlib.pyplot as plt

__all__ = [
    "Operator",
    "Matrix",
    "Array",
    "Scalar",
    "Variable",
    "MergedVariable",
    "Function",
    "Discretization",
]


Operation = Enum("Operation", ["void", "add", "sub", "mul", "evaluate", "div"])


class Operator:
    """Superclass for all Ad operators.

    Objects of this class is not meant to be initiated directly; rather the various
    subclasses should be used. Instances of this class will still be created when
    subclasses are combined by operations.

    """

    def __init__(
        self,
        disc: Optional = None,
        name: Optional[str] = None,
        grid: Optional[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]] = None,
        tree: Optional["_Tree"] = None,
    ) -> None:
        if disc is not None:
            self._discr = disc
        if name is not None:
            self._name = name
            assert disc is not None
        if grid is not None:
            self.g = grid
        self._set_tree(tree)

    def _set_tree(self, tree=None):
        if tree is None:
            self.tree = _Tree(Operation.void)
        else:
            self.tree = tree

    def is_leaf(self) -> bool:
        """Check if this operator is a leaf in the tree-representation of an object.

        Returns:
            bool: True if the operator has no children. Note that this implies that the
                method parse() is expected to be implemented.
        """
        return len(self.tree.children) == 0

    def parse(self, gb) -> Any:
        """Translate the operator into a numerical expression.

        Subclasses that represent atomic operators (leaves in a tree-representation of
        an operator) should override this method to retutrn e.g. a number, an array or a
        matrix.

        This method should not be called on operators that are formed as combinations
        of atomic operators; such operators should be evaluated by an Equation object.

        """
        raise NotImplementedError("This type of operator cannot be parsed right away")

    def __repr__(self) -> str:
        return f"Operator formed by {self._tree._op} with {len(self._tree._children)} children"

    def viz(self):
        """Give a visualization of the operator tree that has this operator at the top."""
        G = nx.Graph()

        def parse_subgraph(node):
            G.add_node(node)
            if len(node.tree.children) == 0:
                return
            operation = node.tree.op
            G.add_node(operation)
            G.add_edge(node, operation)
            for child in node._tree._children:
                parse_subgraph(child)
                G.add_edge(child, operation)

        parse_subgraph(self)
        nx.draw(G, with_labels=True)
        plt.show()

    ### Below here are method for overloading aritmethic operators

    def __mul__(self, other):
        children = self._parse_other(other)
        tree = _Tree(Operation.mul, children)
        return Operator(tree=tree)

    def __truediv__(self, other):
        children = self._parse_other(other)
        return Operator(tree=_Tree(Operation.div, children))

    def __add__(self, other):
        children = self._parse_other(other)
        return Operator(tree=_Tree(Operation.add, children))

    def __sub__(self, other):
        children = [self, other]
        return Operator(tree=_Tree(Operation.sub, children))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def _parse_other(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return [self, pp.ad.Scalar(other)]
        elif isinstance(other, np.ndarray):
            return [self, pp.ad.Array(other)]
        elif isinstance(other, sps.spmatrix):
            return [self, pp.ad.Matrix(other)]
        elif isinstance(other, pp.ad.Operator) or isinstance(other, Operator):
            return [self, other]
        else:
            raise ValueError(f"Cannot parse {other} as an AD operator")


class Matrix(Operator):
    """Ad representation of a sparse matrix.

    For dense matrices, use an Array instead.

    This is a shallow wrapper around the real matrix; it is needed to combine the matrix
    with other types of Ad objects.

    """

    def __init__(self, mat: sps.spmatrix) -> None:
        """Construct an Ad representation of a matrix.

        Parameters:
            mat (sps.spmatrix): Sparse matrix to be represented.

        """
        self._mat = mat
        self._set_tree()

    def __repr__(self) -> str:
        return f"Matrix with shape {self._mat.shape} and {self._mat.data.size} elements"

    def parse(self, gb) -> sps.spmatrix:
        """Convert the Ad matrix into an actual matrix.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            sps.spmatrix: The wrapped matrix.

        """
        return self._mat


class Array(Operator):
    """Ad representation of a numpy array.

    For sparse matrices, use a Matrix instead.

    This is a shallow wrapper around the real array; it is needed to combine the array
    with other types of Ad objects.

    """

    def __init__(self, values):
        """Construct an Ad representation of a numpy array.

        Parameters:
            values (np.ndarray): Numpy array to be represented.

        """
        self._values = values
        self._set_tree()

    def __repr__(self) -> str:
        return f"Wrapped numpy array of size {self.values.size}"

    def parse(self, gb: pp.GridBucket) -> np.ndarray:
        """Convert the Ad Array into an actual array.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            np.ndarray: The wrapped array.

        """
        return self._values


class Scalar(Operator):
    """Ad representation of a scalar.

    This is a shallow wrapper around the real scalar; it may be useful to combine the
    scalar with other types of Ad objects.

    """

    def __init__(self, value):
        """Construct an Ad representation of a numpy array.

        Parameters:
            values (float): Number to be represented

        """
        self._value = value
        self._set_tree()

    def __repr__(self) -> str:
        return f"Wrapped scalar with value {self.value}"

    def parse(self):
        """Convert the Ad Scalar into an actual number.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            float: The wrapped number.

        """
        return self._value


class Variable(Operator):
    """Ad representation of a variable which on a single Grid or MortarGrid.

    For combinations of variables on different grids, see MergedVariable.

    Conversion of the variable into numerical value should be done with respect to the
    state of an array; see Equations. Therefore, the variable does not implement a
    parse() method.

    """

    # Identifiers for variables. The usage and reliability of this system is unclear.
    _ids = count(0)

    def __init__(
        self,
        name: str,
        ndof: Dict[str, int],
        grid_like: Union[pp.Grid, Tuple[pp.Grid, pp.Grid]],
        num_cells: int = 0,
    ):
        """Initiate an Ad representation of the variable.

        Parameters:
            name (str): Variable name.
            ndof (dict): Number of dofs per grid element.
            grid_like (pp.Grid, or Tuple of pp.Grid): Either a grid or an interface
                (combination of grids).
            num_cells (int): Number of cells in the grid. Only sued if the variable
                is on an interface.

        """
        self._name = name
        self._cells = ndof.get("cells", 0)
        self._faces = ndof.get("faces", 0)
        self._nodes = ndof.get("nodes", 0)
        self.g = grid_like

        # The number of cells in the grid. Will only be used if grid_like is a tuple
        # that is, if this is a mortar variable
        self._num_cells = num_cells

        self.id = next(self._ids)

        self._set_tree()

    def size(self) -> int:
        """Get the number of dofs for this grid.

        Returns:
            int: Number of dofs.

        """
        if isinstance(self.g, tuple):
            # This is a mortar grid. Assume that there are only cell unknowns
            return self._num_cells * self._cells
        else:
            return (
                self.g.num_cells * self._cells
                + self.g.num_faces * self._faces
                + self.g.num_nodes * self._nodes
            )

    def __repr__(self) -> str:
        s = (
            f"Variable {self._name}, id: {self.id}\n"
            f"Degrees of freedom in cells: {self._cells}, faces: {self._faces}, "
            f"nodes: {self._nodes}\n"
        )
        return s


class MergedVariable(Variable):
    """Ad representation of a collection of variables that invidiually live on separate
    grids of interfaces, but which it is useful to treat jointly.

    Conversion of the variables into numerical value should be done with respect to the
    state of an array; see Equations.  Therefore, the class does not implement a parse()
    method.

    Attributes:
        sub_vars (List of Variable): List of variable on different grids or interfaces.
        id (int): Counter of all variables. Used to identify variables. Usage of this
            term is not clear, it may change.
    """

    def __init__(self, variables: List[Variable]) -> None:
        """Create a merged representation of variables.

        Parameters:
            variables (list of Variable): Variables to be merged. Should all have the
                same name.

        """
        self.sub_vars = variables
        self.id = next(self._ids)

        # Take the name from the first variabe.
        self._name = variables[0]._name
        # Check that all variables have the same name.
        # We may release this in the future, but for now, we make it a requirement
        all_names = set(var._name for var in variables)
        assert len(all_names) == 1

        self._set_tree()

        self.is_interface = isinstance(self.sub_vars[0].g, tuple)

    def __repr__(self) -> str:
        sz = np.sum([var.size() for var in self.sub_vars])
        if self.is_interface:
            s = "Merged interface"
        else:
            s = "Merged"

        s += (
            f" variable with name {self._name}, id {self.id}\n"
            f"Composed of {len(self.sub_vars)} variables\n"
            f"Degrees of freedom in cells: {self.sub_vars[0]._cells}"
            f", faces: {self.sub_vars[0]._faces}, nodes: {self.sub_vars[0]._nodes}\n"
            f"Total size: {sz}\n"
        )

        return s


class Function(Operator):
    """Ad representation of a function."""

    def __init__(self, func: Callable, name: str):
        """Initialize a function.

        Parameters:
            func (Callable): Function which maps one or several Ad arrays to an
                Ad array.
            name (str): Name of the function.

        """
        self.func = func
        self._name = name
        self._set_tree()

    def __mul__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __add__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __sub__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __call__(self, *args):
        children = [self, *args]
        op = Operator(tree=_Tree(Operation.evaluate, children=children))
        return op

    def __repr__(self) -> str:
        s = f"AD function with name {self._name}"

        return s

    def parse(self, gb):
        """Parsing to an numerical value.

        The real work will be done by combining the function with arguments, during
        parsing of an operator tree.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            The object itself.

        """
        return self


class Discretization:
    """Wrapper to make a PorePy discretization compatible with the Ad framework.

    For any PorePy discretization class (e.g. Mpfa, Biot, etc.), the wrapper associates
    a discretization with all attributes of the class' attributes that ends with
    '_matrix_key'.

    Example:
        # Generate grid
        >>> g = pp.CartGrid([2, 2])
        # Associate an Ad representation of an Mpfa method, aimed this grid
        >>> discr = Discretization({g: pp.Mpfa('flow')})
        # The flux discretization of Mpfa can now be accesed by
        >>> discr.flux
        # While the discretization of boundary conditions is available by
        >>> discr.bound_flux.

    The representation of different discretization objects can be combined with other
    Ad objects into an operator tree, using lazy evaluation.

    It is assumed that the actual action of discretization (creation of the
    discretization matrices) is performed before the operator tree is parsed.

    """

    def __init__(
        self,
        grid_discr: Dict[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]], "pp.AbstractDiscretization"],
        name: Optional[str] = None,
        mat_dict_key: Optional[str] = None,
    ):
        """Construct a wrapper around a Discretization object for a set of grids.

        Different grids may be associated with different discetization classes, but they
        should have the same keywords to access discretization matrices (Tpfa and Mpfa
        are compatible in this sense).

        Parameters:
            grid_discr (dict): Mapping between grids, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            name (str): Name of the wrapper.
            mat_dict_key (str): Keyword used to access discretization matrices, if this
                is not the same as the keyword of the discretization. The only known
                case where this is necessary is for Mpfa applied to Biot's equations.

        """
        self.grid_discr = grid_discr
        key_set = []
        self.mat_dict_key = mat_dict_key

        # Get the name of this discretization.
        # If not provided, make a name by combining all discretization objects
        if name is None:
            names = []
            for discr in grid_discr.values():
                names.append(discr.__class__.__name__)

            # Uniquify and then merge into a string
            self.name = "_".join(list(set(names)))
        else:
            self.name = name

        # Loop over all discretizations, identify all attributes that ends with
        # "_matrix_key". These will be taken as discretizations (they are discretization
        # matrices for specific terms, to be).
        for i, discr in enumerate(grid_discr.values()):
            for s in dir(discr):
                if s.endswith("_matrix_key"):
                    if i == 0:
                        key = s[:-11]
                        key_set.append(key)
                    else:
                        if key not in key_set:
                            raise ValueError(
                                "Merged disrcetization should have uniform set of operations"
                            )

        # Make a merged discretization for each of the identified terms.
        # If some keys are not shared by all values in grid_discr, errors will result.
        for key in key_set:
            op = MergedOperator(grid_discr, key, self.mat_dict_key)
            setattr(self, key, op)

    def __repr__(self) -> str:

        discr_counter = {}

        for discr in self.grid_discr.values():
            if discr not in discr_counter:
                discr_counter[discr] = 0
            discr_counter[discr] += 1

        s = f"Merged discretization with name {self.name}. Sub discretizations:\n"
        for key, val in discr_counter.items():
            s += f"{val} occurences of discretization {key}"

        return s


class MergedOperator(Operator):
    """Representation of specific discretization fields for an Ad discretization.

    This is the bridge between the representation of discretization classes, implemented
    in Discretization, and the matrices resulting from a discretization.

    Objects of this class should not be access directly, but rather through the
    Discretization class.

    """

    def __init__(
        self,
        grid_discr: Dict[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]], "pp.AbstractDiscretization"],
        key: str,
        mat_dict_key: Optional[str] = None,
    ) -> None:
        """Initiate a merged discretization.

        Parameters:
            grid_discr (dict): Mapping between grids, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            key (str): Keyword that identifies this discretization matrix, e.g.
                for a class with an attribute foo_matrix_key, the key will be foo.
            mat_dict_key (str): Keyword used to access discretization matrices, if this
                is not the same as the keyword of the discretization. The only known
                case where this is necessary is for Mpfa applied to Biot's equations.

        """
        self.grid_discr = grid_discr
        self.key = key

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key

        self._set_tree(None)

    def __repr__(self) -> str:
        return f"Operator with key {self.key} defined on {len(self.grid_discr)} grids"

    def parse(self, gb):
        """Convert a merged operator into a sparse matrix by concatenating
        discretization matrices.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            sps.spmatrix: The merged discretization matrices for the associated matrix.

        """

        # Data structure for matrices
        mat = []

        # Loop over all grid-discretization combinations, get hold of the discretization
        # matrix for this grid quantity
        for g, discr in self.grid_discr.items():

            # Get data dictionary for either grid or interface
            if isinstance(g, pp.Grid):
                data = gb.node_props(g)
            else:
                data = gb.edge_props(g)

            # Use the specified mat_dict_key if available; if not, use the keyword of
            # the discretization object
            if self.mat_dict_key is not None:
                mat_dict_key = self.mat_dict_key
            else:
                mat_dict_key = discr.keyword

            mat_dict: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
                mat_dict_key
            ]

            # Get the submatrix for the right discretization
            key = self.key
            mat_key = getattr(discr, key + "_matrix_key")
            mat.append(mat_dict[mat_key])

        return sps.block_diag(mat)


class _Tree:
    """Simple implementation of a Tree class. Used to represent combinations of
    Ad operators.
    """

    # https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    def __init__(self, operation: Operation, children: Optional[List["_Tree"]] = None):

        self.op = operation

        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, node: Operator):
        assert isinstance(node, Operator) or isinstance(node, pp.ad.Operator)
        self.children.append(node)
