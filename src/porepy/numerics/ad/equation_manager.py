"""
* Resue assembly when relevant (if no operator that maps to a specific block has been changed)
* Concatenate equations with the same sequence of operators
  - Should use the same discretization object
  - divergence operators on different grids considered the same
* Concatenated variables will share ad derivatives. However, it should be possible to combine
  subsets of variables with other variables (outside the set) to assemble different terms
* 
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

from . import operators, grid_operators
from .forward_mode import initAdArrays

import porepy as pp

__all__ = ["Equation", "EquationManager"]


class Equation:
    def __init__(self):
        pass


class EquationManager:
    def __init__(self, gb):
        # need a way to print an equation. Separate class?
        self._equations = []
        self._gb = gb

        self._set_variables()

        # Separate a dof-manager from assembler?
        self._assembler = pp.Assembler(gb)

        # This goes to equation
        self._stored_matrices = {}

    def _set_variables(self):
        # Define variables
        variables = {}
        for g, d in self._gb:
            variables[g] = {}
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[g][var] = operators.Variable(var, info, g)

        for e, d in self._gb.edges():
            variables[e] = {}
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[e][var] = operators.Variable(var, info, e)

        self._variables = variables
        # Define discretizations

    def _operators_from_gb(self):
        # Define operators from information in the GridBucket.
        # Legacy style definition
        discr_dict = {}

        bc_dict = {}

        for g, d in self._gb:
            loc_bc_dict = {}
            for variable, term in d[pp.DISCRETIZATION].items():  # terms per variable
                for key, discr in term.items():
                    if (discr, key) not in discr_dict:
                        # This discretization has not been encountered before.
                        # Make operators of all its discretization matrices

                        # Temporarily make an inner dictionary to store the individual
                        # discretizations. It would be neat if we could rather write
                        # something like op.flux, instead of op['flux']
                        op_dict = {}

                        for s in dir(discr):
                            if s.endswith("_matrix_key"):
                                matrix_key = s[:-11]
                                op = operators.Operator(discr, matrix_key, g)
                                op_dict[matrix_key] = op
                        discr_dict[(variable, key)] = op_dict

                    loc_bc_dict[discr.keyword] = grid_operators.BoundaryCondition(
                        discr.keyword
                    )
            bc_dict[g] = loc_bc_dict

        self._discretizations = discr_dict
        self._bc = bc_dict

    def equate_to_zero(self, op: operators.Operator) -> None:
        # Define a full residual equation
        self._equations.append(op)

    def _get_matrix(self, data, op):
        # Move this into a class

        discr = op._discr
        key = op._name
        mat_dict = data[pp.DISCRETIZATION_MATRICES][discr.keyword]
        mat_key = getattr(discr, key + "_matrix_key")
        return mat_dict[mat_key]

    def set_state(self, state):
        self._state = state

    def _variables_of_equation(self, op: operators.Operator):

        if isinstance(op, operators.Variable):
            # We are at the bottom of the a branch of the tree
            return op
        else:
            # Look for variables among the children
            sub_variables = [
                self._variables_of_equation(child) for child in op._tree._children
            ]
            # Some work is needed to parse the information
            var_list = []
            for var in sub_variables:
                if isinstance(var, operators.Variable):
                    # Effectively, this node is one step from the leaf
                    var_list.append(var)
                elif isinstance(var, list):
                    # We are further up in the tree.
                    for sub_var in var:
                        if isinstance(sub_var, operators.Variable):
                            var_list.append(sub_var)
            return var_list

    def parse_equation(self, op: operators.Operator):
        # 1. Get all variables present in this equation
        # Uniquify by making this a set, and then sort on variable id
        variables = sorted(
            list(set(self._variables_of_equation(op))), key=lambda var: var.id
        )

        # 2. Get state of the variables, init ad
        # Make the AD variables active of sorts; so that when parsing the individual
        # operators, we can access the right variables

        # For each variable, get the global index
        inds = []
        for variable in variables:
            ind_var = []
            for sub_var in variable.sub_vars:
                ind_var.append(self._assembler.dof_ind(sub_var.g, sub_var._name))

            inds.append(np.hstack((i for i in ind_var)))

        # Initialize variables
        ad_vars = initAdArrays([self._state[ind] for ind in inds])
        self._ad = {var.id: ad for (var, ad) in zip(variables, ad_vars)}

        # 3. Parse operators. Matrices can be picked either from discretization matrices,
        # or from some central storage,
        eq = self._parse_operator(op)

        return eq

    def _parse_operator(self, op: operators.Operator):
        # Q: The parsing could also be moved to the operator classes
        tree = op._tree
        if isinstance(op, pp.ad.Variable):
            assert len(tree._children) == 0
            # Need access to state, grids, assembler, local_dof etc.

            # Really need a method to get state in all variables to which this
            # should have a coupling.
            # should use all variables in this equation. Need to pick out the right part of
            # it here (perhaps as by indexing a list of variables) for use to propagete through
            # the chain of operations
            return self._ad[op.id]
        if isinstance(op, grid_operators.BoundaryCondition):
            val = []
            for g in op.g:
                data = self._gb.node_props(g)
                val.append(data[pp.PARAMETERS][op.keyword]["bc_values"])

            return np.hstack((v for v in val))

        if isinstance(op, pp.ad.Matrix):
            return op.mat

        if len(tree._children) == 0:
            if isinstance(op, operators.MergedOperator):
                if op in self._stored_matrices:
                    return self._stored_matrices[op]
                else:
                    if isinstance(op, grid_operators.Divergence):
                        if op.scalar:
                            mat = [pp.fvutils.scalar_divergence(g) for g in op.g]
                        else:
                            mat = [pp.fvutils.vector_divergence(g) for g in op.g]
                    else:
                        mat = []
                        for g, discr in op.grid_discr.items():
                            if isinstance(g, pp.Grid):
                                data = self._gb.node_props(g)
                            else:
                                data = self._gb.edge_props(g)
                            key = op.key
                            mat_dict = data[pp.DISCRETIZATION_MATRICES][discr.keyword]
                            mat_key = getattr(discr, key + "_matrix_key")
                            mat.append(mat_dict[mat_key])

                    matrix = sps.block_diag(mat)
                    self._stored_matrices[op] = matrix
                    return matrix
            else:
                # Single grid
                assert False
                return self._get_matrix(op.g, op)

        results = [self._parse_operator(child) for child in tree._children]

        if tree._op == operators.Operation.add:
            assert len(results) == 2
            return results[0] + results[1]
        elif tree._op == operators.Operation.sub:
            assert len(results) == 2
            return results[0] - results[1]
        elif tree._op == operators.Operation.mul:
            return results[0] * results[1]

        elif tree._op == operators.Operation.eval:
            raise NotImplementedError("")
        else:
            raise ValueError("Should not happen")
