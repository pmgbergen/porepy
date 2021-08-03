""" Implementation of wrappers for Ad representations of several operators.
"""
import copy
from enum import Enum
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.params.tensor import SecondOrderTensor

__all__ = [
    "Operator",
    "Matrix",
    "Array",
    "Scalar",
    "Variable",
    "MergedVariable",
    "Function",
]


Operation = Enum(
    "Operation", ["void", "add", "sub", "mul", "evaluate", "div", "localeval", "apply"]
)


class Operator:
    """Superclass for all Ad operators.

    Objects of this class is not meant to be initiated directly; rather the various
    subclasses should be used. Instances of this class will still be created when
    subclasses are combined by operations.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        grid: Optional[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]] = None,
        tree: Optional["Tree"] = None,
    ) -> None:
        if name is not None:
            self._name = name
        if grid is not None:
            self.g = grid
        self._set_tree(tree)

    def _set_tree(self, tree=None):
        if tree is None:
            self.tree = Tree(Operation.void)
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
        return (
            f"Operator formed by {self.tree.op} with {len(self.tree.children)} children"
        )

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
        tree = Tree(Operation.mul, children)
        return Operator(tree=tree)

    def __truediv__(self, other):
        children = self._parse_other(other)
        return Operator(tree=Tree(Operation.div, children))

    def __add__(self, other):
        children = self._parse_other(other)
        return Operator(tree=Tree(Operation.add, children))

    def __sub__(self, other):
        children = [self, other]
        return Operator(tree=Tree(Operation.sub, children))

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
        self.shape = mat.shape

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

    def transpose(self) -> "Matrix":
        return Matrix(self._mat.transpose())


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
        return f"Wrapped numpy array of size {self._values.size}"

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
        return f"Wrapped scalar with value {self._value}"

    def parse(self, gb: pp.GridBucket) -> float:
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
        previous_timestep: bool = False,
        previous_iteration: bool = False,
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
        self._name: str = name
        self._cells: int = ndof.get("cells", 0)
        self._faces: int = ndof.get("faces", 0)
        self._nodes: int = ndof.get("nodes", 0)
        self.g = grid_like

        self.prev_time: bool = previous_timestep
        self.prev_iter: bool = previous_iteration

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

    def previous_timestep(self) -> "Variable":
        ndof = {"cells": self._cells, "faces": self._faces, "nodes": self._nodes}
        return Variable(self._name, ndof, self.g, previous_timestep=True)

    def previous_iteration(self) -> "Variable":
        ndof = {"cells": self._cells, "faces": self._faces, "nodes": self._nodes}
        return Variable(self._name, ndof, self.g, previous_iteration=True)

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

        # Use counter from superclass to ensure unique Variable ids
        self.id = next(Variable._ids)

        # Take the name from the first variabe.
        self._name = variables[0]._name
        # Check that all variables have the same name.
        # We may release this in the future, but for now, we make it a requirement
        all_names = set(var._name for var in variables)
        assert len(all_names) == 1

        self._set_tree()

        self.is_interface = isinstance(self.sub_vars[0].g, tuple)

        self.prev_time: bool = False
        self.prev_iter: bool = False

    def previous_timestep(self) -> "MergedVariable":
        new_subs = [var.previous_timestep() for var in self.sub_vars]
        new_var = MergedVariable(new_subs)
        new_var.prev_time = True
        return new_var

    def previous_iteration(self) -> "MergedVariable":
        new_subs = [var.previous_iteration() for var in self.sub_vars]
        new_var = MergedVariable(new_subs)
        new_var.prev_iter = True
        return new_var

    def copy(self) -> "MergedVariable":
        # A shallow copy should be sufficient here; the attributes are not expected to
        # change.
        return copy.deepcopy(self)

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
    """Ad representation of a function.

    The intended use is as a wrapper for operations on pp.ad.Ad_array objects,
    in forms which are not directly or easily expressed by the rest of the Ad
    framework.

    """

    def __init__(self, func: Callable, name: str, local=False):
        """Initialize a function.

        Parameters:
            func (Callable): Function which maps one or several Ad arrays to an
                Ad array.
            name (str): Name of the function.

        """
        self.func = func
        self._name = name
        self._operation = Operation.evaluate if not local else Operation.localeval
        self._set_tree()

    def __mul__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __add__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __sub__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __call__(self, *args):
        children = [self, *args]
        op = Operator(tree=Tree(self._operation, children=children))
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


class ApplicableOperator(Function):
    """Ad representation of operator providing metod 'apply'.
    This class is meant as base class.
    """

    def __init__(self) -> None:
        """Initialization empty."""
        pass

    def __repr__(self) -> str:
        s = "AD applicable operator."
        return s

    def __call__(self, *args):
        children = [self, *args]
        op = Operator(tree=Tree(Operation.apply, children=children))
        return op


class SecondOrderTensorAd(SecondOrderTensor, Operator):
    def __init__(self, kxx, kyy=None, kzz=None, kxy=None, kxz=None, kyz=None):
        super().__init__(kxx, kyy, kzz, kxy, kxz, kyz)
        self._set_tree()

    def __repr__(self) -> str:
        s = "AD second order tensor"

        return s

    def parse(self, gb: pp.GridBucket) -> np.ndarray:
        return self.values


class Tree:
    """Simple implementation of a Tree class. Used to represent combinations of
    Ad operators.
    """

    # https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    def __init__(self, operation: Operation, children: Optional[List[Operator]] = None):

        self.op = operation

        self.children: List[Operator] = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, node: Operator) -> None:
        #        assert isinstance(node, (Operator, "pp.ad.Operator"))
        self.children.append(node)
