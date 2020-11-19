import abc
from enum import Enum
from typing import Optional, List
from itertools import count

import numpy as np
import porepy as pp
import scipy.sparse as sps

__all__ = [
    "Operator",
    "MergedOperator",
    "Matrix",
    "Array",
    "Scalar",
    "Variable",
    "MergedVariable",
    "Function",
    "Discretization",
]


Operation = Enum("Operation", ["void", "add", "sub", "mul", "evaluate", "div"])


class Tree:
    # https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    def __init__(self, operation: Operation, children: Optional[List["Tree"]] = None):

        self._op = operation

        self._children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, node):
        assert isinstance(node, Operator) or isinstance(node, pp.ad.Operator)
        self._children.append(node)


class Operator:
    def __init__(self, disc=None, name=None, grid=None, tree=None):
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
            self._tree = Tree(Operation.void)
        else:
            self._tree = tree

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


class MergedOperator(Operator):
    # This will likely be converted to operator, that is, non-merged operators are removed
    def __init__(self, grid_discr, key, mat_dict_key: str = None):
        self.grid_discr = grid_discr
        self.key = key

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key

        self._set_tree(None)


class Matrix(Operator):
    def __init__(self, mat):
        self.mat = mat
        self._set_tree()


class Array(Operator):
    def __init__(self, values):
        self.values = values
        self._set_tree()


class Scalar(Operator):
    def __init__(self, value):
        self.value = value
        self._set_tree()


class Variable(Operator):

    _ids = count(0)

    def __init__(self, name, ndof, grid_like):
        self._name = name
        self._cells = ndof.get("cells", 0)
        self._faces = ndof.get("faces", 0)
        self._nodes = ndof.get("nodes", 0)
        self.g = grid_like
        self.id = next(self._ids)

        self._set_tree()

    def size(self) -> int:
        if isinstance(self.g, tuple):
            return 0
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
    # TODO: Is it okay to generate the same variable (grid, name) many times?
    # The whole concept needs a massive cleanup
    def __init__(self, variables):
        self.sub_vars = variables
        self.id = next(self._ids)
        self._name = variables[0]._name
        self._set_tree()

        self.is_interface = isinstance(self.sub_vars[0].g, tuple)

        all_names = set(var._name for var in variables)
        assert len(all_names) == 1

    def __repr__(self) -> str:
        s = (
            f"Merged variable with name {self._name}, id {self.id}\n"
            f"Composed of {len(self.sub_vars)} variables\n"
            f"Degrees of freedom in cells: {self.sub_vars[0]._cells}"
            f", faces: {self.sub_vars[0]._faces}, nodes: {self.sub_vars[0]._nodes}\n"
        )
        if not self.is_interface:
            sz = np.sum([var.size() for var in self.sub_vars])
            s += f"Total size: {sz}\n"

        return s


class Function(Operator):
    def __init__(self, func, name):
        self.func = func
        self.name = name
        self._set_tree()

    def __mul__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __add__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __sub__(self, other):
        raise RuntimeError("Functions should only be evaluated")

    def __call__(self, *args):
        children = [self, *args]
        op = Operator(tree=Tree(Operation.evaluate, children=children))
        breakpoint()
        return op

    def __repr__(self) -> str:
        s = f"AD function with name {self.name}"

        return s


class Discretization:
    def __init__(self, grid_discr, name=None, tree=None, mat_dict_key: str = None):

        self.grid_discr = grid_discr
        key_set = []
        self.mat_dict_key = mat_dict_key
        if name is None:
            names = []
            for discr in grid_discr.values():
                names.append(discr.__class__.__name__)

            self.name = "_".join(list(set(names)))
        else:
            self.name = name

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
