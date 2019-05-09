#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test of pp.Assembler(gb).

The tests together set up several combinations of variables. The grid always
consists of two nodes, joined by a single edge. There is one cell per grid
bucket item, and 1 or 2 variables on each item.

Cases covered are:
1. A single variable on all grids.
2. Two variables on all grids. No coupling internal to grids, simple coupling
    between grids and nodes.
3. As 2, but only one of the variables are active. Matrix equivalent to 1.
4. Two variables on all grids. Coupling between variables internal to nodes,
    no coupling between edge and nodes.
5. Same as 4, but with a single active variable.
6. Two variables on all grids. Coupling between node and edge, but not
    cross-variable coupling.
7. Same as 6, but also including cross-variable coupling on the edge.
8. Mixture of 6 and 7; one edge variable is depends on both node variables,
    the other only depends on 'itself' on the node.


NOTE: Throughout the tests, the indexing of the known disrcetization matrices
assumes that the order of the variables are the same as they are defined on the
grid bucket data dictionaries. If this under some system changes, everything
breaks.

"""
import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from test.test_utils import permute_matrix_vector


class TestAssembler(unittest.TestCase):
    def define_gb(self):
        g1 = pp.CartGrid([1, 1])
        g2 = pp.CartGrid([1, 1])
        g1.compute_geometry()
        g2.compute_geometry()

        g1.grid_num = 1
        g2.grid_num = 2

        gb = pp.GridBucket()
        gb.add_nodes([g1, g2])
        gb.add_edge([g1, g2], None)

        mg = pp.MortarGrid(2, {"left": g1, "right": g2}, sps.coo_matrix(1))
        mg.num_cells = 1
        gb.set_edge_prop([g1, g2], "mortar_grid", mg)
        return gb

    ### Test with no coupling between the subdomains

    def test_single_variable(self):
        """ A single variable, test that the basic mechanics of the assembler functions.
        """
        gb = self.define_gb()

        # Variable name assigned on nodes. Same for both grids
        variable_name = "variable_1"
        # Edge variable
        variable_name_e = "edge_variable"

        # Keyword for discretization operators.
        term_g1 = "operator_1"
        term_g2 = "operator_2"
        term_e = "operator_coupling"

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {variable_name: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name: {term_g1: MockNodeDiscretization(1)}
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name: {term_g2: MockNodeDiscretization(2)}
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_e: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term_e: {
                    g1: (variable_name, term_g1),
                    g2: (variable_name, term_g2),
                    e: (variable_name_e, MockEdgeDiscretization(1, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = general_assembler.block_dof[(g1, variable_name)]
        g2_ind = general_assembler.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = 2
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_different_operator_names(self):
        # Use different names for the two variables operators
        gb = self.define_gb()
        for g, d in gb:
            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {"variable_1": {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    "variable_1": {"operator_1": MockNodeDiscretization(1)}
                }
                g1 = g
            else:
                d[pp.PRIMARY_VARIABLES] = {"variable_2": {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    "variable_2": {"operator_2": MockNodeDiscretization(2)}
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {"variable_e": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_edge": {
                    g1: ("variable_1", "operator_1"),
                    g2: ("variable_2", "operator_2"),
                    e: ("variable_e", MockEdgeDiscretizationModifiesNode(1, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = general_assembler.block_dof[(g1, "variable_1")]
        g2_ind = general_assembler.block_dof[(g2, "variable_2")]
        A_known[g1_ind, g1_ind] = 2
        A_known[g2_ind, g2_ind] = 4

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_no_node_disretization(self):
        """ A single variable, where one of the nodes have no discretization
        object assigned.
        """
        gb = self.define_gb()
        variable_name = "variable_1"
        variable_name_edge = "variable_edge"
        discretization_operator = "operator_discr"

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {variable_name: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name: {discretization_operator: MockNodeDiscretization(1)}
                }
                g1 = g
            else:
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_edge: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_discretization": {
                    g1: (variable_name, discretization_operator),
                    g2: (variable_name, discretization_operator),
                    e: (variable_name_edge, MockEdgeDiscretization(1, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = general_assembler.block_dof[(g1, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_not_active(self):
        """ A single variable, but make this inactive. This should return an empty matrix
        """

        gb = self.define_gb()
        variable_name = "variable_1"
        variable_name_edge = "variable_edge"
        discretization_operator = "operator_discr"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {variable_name: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name: {discretization_operator: MockNodeDiscretization(1)}
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name: {discretization_operator: MockNodeDiscretization(2)}
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_edge: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_discretization": {
                    g1: (variable_name, discretization_operator),
                    g2: (variable_name, discretization_operator),
                    e: (variable_name_edge, MockEdgeDiscretization(1, 1)),
                }
            }

        general_assembler = pp.Assembler(gb, active_variables="var_11")
        # Give a false variable name
        A, b = general_assembler.assemble_matrix_rhs()
        self.assertTrue(A.shape == (0, 0))
        self.assertTrue(b.size == 0)

    def test_single_variable_multiple_node_discretizations(self):
        """ A single variable, with multiple discretizations for one of the nodes
        """
        gb = self.define_gb()
        variable_name = "variable_1"
        variable_name_edge = "variable_edge"
        operator_1 = "operator_1"
        operator_2 = "operator_2"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {variable_name: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name: {
                        operator_1: MockNodeDiscretization(1),
                        operator_2: MockNodeDiscretization(4),
                    }
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name: {operator_1: MockNodeDiscretization(2)}
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_edge: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_discretization": {
                    g1: (variable_name, operator_1),
                    g2: (variable_name, operator_1),
                    e: (variable_name_edge, MockEdgeDiscretization(1, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = general_assembler.block_dof[(g1, variable_name)]
        g2_ind = general_assembler.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 5
        A_known[g2_ind, g2_ind] = 2
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_multiple_edge_discretizations(self):
        """ A single variable, with multiple discretizations for one of the e
        """
        gb = self.define_gb()

        variable_name = "variable_1"
        variable_name_edge = "variable_edge"
        operator_1 = "operator_1"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {variable_name: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name: {operator_1: MockNodeDiscretization(1)}
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name: {operator_1: MockNodeDiscretization(2)}
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_edge: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_discretization_1": {
                    g1: (variable_name, operator_1),
                    g2: (variable_name, operator_1),
                    e: (variable_name_edge, MockEdgeDiscretization(1, 1)),
                },
                "coupling_discretization_2": {
                    g1: (variable_name, operator_1),
                    g2: (variable_name, operator_1),
                    e: (variable_name_edge, MockEdgeDiscretization(-3, 1)),
                },
            }

        general_assembler = pp.Assembler(gb)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 2], [0, 0, 2], [-2, -2, -2]])
        g1_ind = general_assembler.block_dof[(g1, variable_name)]
        g2_ind = general_assembler.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = 2
        assert np.allclose(A_known, A.todense())

    def test_two_variables_no_coupling(self):
        """ Two variables, no coupling between the variables. Test that the
        assembler can deal with more than one variable.
        """
        gb = self.define_gb()
        variable_name_1 = "variable_1"
        variable_name_2 = "variable_2"
        variable_name_edge_1 = "variable_edge_1"
        variable_name_edge_2 = "variable_edge_2"
        operator_1 = "operator_1"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator_1: MockNodeDiscretization(1)},
                    variable_name_2: {operator_1: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator_1: MockNodeDiscretization(3)},
                    variable_name_2: {operator_1: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_edge_1: {"cells": 1},
                variable_name_edge_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_edge_1: {operator_1: MockNodeDiscretization(5)},
                variable_name_edge_2: {operator_1: MockNodeDiscretization(6)},
            }

        general_assembler = pp.Assembler(gb)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((6, 6))

        g11_ind = general_assembler.block_dof[(g1, variable_name_1)]
        g12_ind = general_assembler.block_dof[(g1, variable_name_2)]
        g21_ind = general_assembler.block_dof[(g2, variable_name_1)]
        g22_ind = general_assembler.block_dof[(g2, variable_name_2)]
        e1_ind = general_assembler.block_dof[(e, variable_name_edge_1)]
        e2_ind = general_assembler.block_dof[(e, variable_name_edge_2)]
        A_known[g11_ind, g11_ind] = 1
        A_known[g12_ind, g12_ind] = 2
        A_known[g21_ind, g21_ind] = 3
        A_known[g22_ind, g22_ind] = 4
        A_known[e1_ind, e1_ind] = 5
        A_known[e2_ind, e2_ind] = 6
        assert np.allclose(A_known, A.todense())

    def test_two_variables_one_active(self):
        """ Define two variables, but then only assemble with respect to one
        of them. Should result in what is effectively a 1-variable system

        """
        gb = self.define_gb()
        variable_name_1 = "variable_1"
        variable_name_2 = "variable_2"
        operator = "operator"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(1)},
                    variable_name_2: {operator: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(3)},
                    variable_name_2: {operator: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_1: {operator: MockNodeDiscretization(5)},
                variable_name_2: {operator: MockNodeDiscretization(6)},
            }

        general_assembler = pp.Assembler(gb, active_variables=[variable_name_2])
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g12_ind = general_assembler.block_dof[(g1, variable_name_2)]
        g22_ind = general_assembler.block_dof[(g2, variable_name_2)]
        e2_ind = general_assembler.block_dof[(e, variable_name_2)]

        A_known[g12_ind, g12_ind] = 2
        A_known[g22_ind, g22_ind] = 4
        A_known[e2_ind, e2_ind] = 6

        assert np.allclose(A_known, A.todense())

    def test_two_variables_one_active_one_false_active_variable(self):
        """ Define two variables, the define as active one of the variables, and
        another active variable that is not used in the grid. This should be
        equivalent to defining a single active variable

        """
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        operator = "op"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(1)},
                    variable_name_2: {operator: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(3)},
                    variable_name_2: {operator: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_1: {operator: MockNodeDiscretization(5)},
                variable_name_2: {operator: MockNodeDiscretization(6)},
            }

        general_assembler = pp.Assembler(
            gb,
            active_variables=[
                variable_name_2,
                "variable_name_not_among_defined_variables",
            ],
        )
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g12_ind = general_assembler.block_dof[(g1, variable_name_2)]
        g22_ind = general_assembler.block_dof[(g2, variable_name_2)]
        e2_ind = general_assembler.block_dof[(e, variable_name_2)]

        A_known[g12_ind, g12_ind] = 2
        A_known[g22_ind, g22_ind] = 4
        A_known[e2_ind, e2_ind] = 6
        assert np.allclose(A_known, A.todense())

    ### Tests with coupling internal to each node

    def test_two_variables_coupling_within_node_and_edge(self):
        """ Two variables, coupling between the variables internal to each node.
        No coupling in the edge variable
        """
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        operator = "operator"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(1)},
                    variable_name_2: {operator: MockNodeDiscretization(2)},
                    variable_name_1
                    + "_"
                    + variable_name_2: {operator: MockNodeDiscretization(5)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(3)},
                    variable_name_2: {operator: MockNodeDiscretization(4)},
                    variable_name_2
                    + "_"
                    + variable_name_1: {operator: MockNodeDiscretization(6)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_1: {operator: MockNodeDiscretization(7)},
                variable_name_2: {operator: MockNodeDiscretization(8)},
                variable_name_1
                + "_"
                + variable_name_2: {operator: MockNodeDiscretization(6)},
                variable_name_2
                + "_"
                + variable_name_1: {operator: MockNodeDiscretization(1)},
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, g2, e, e]
        variables = [
            variable_name_1,
            variable_name_2,
            variable_name_1,
            variable_name_2,
            variable_name_1,
            variable_name_2,
        ]
        A_known = np.zeros((6, 6))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 3
        A_known[3, 3] = 4
        A_known[4, 4] = 7
        A_known[5, 5] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[0, 1] = 5
        A_known[3, 2] = 6
        A_known[4, 5] = 6
        A_known[5, 4] = 1

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(
            gb, active_variables=[variable_name_1, variable_name_2]
        )

        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            permuted_assembler.block_dof,
            permuted_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_two_variables_coupling_within_node_and_edge_one_active(self):
        """ Two variables, coupling between the variables internal to each node.
        One active variable.
        """
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        operator = "operator"
        operator_edge = "operator_edge"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(1)},
                    variable_name_2: {operator: MockNodeDiscretization(2)},
                    variable_name_1
                    + "_"
                    + variable_name_2: {operator: MockNodeDiscretization(5)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator: MockNodeDiscretization(3)},
                    variable_name_2: {operator: MockNodeDiscretization(4)},
                    variable_name_2
                    + "_"
                    + variable_name_1: {operator: MockNodeDiscretization(6)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_1: {operator_edge: MockNodeDiscretization(7)},
                variable_name_2: {operator_edge: MockNodeDiscretization(8)},
                variable_name_1
                + "_"
                + variable_name_2: {operator_edge: MockNodeDiscretization(6)},
                variable_name_2
                + "_"
                + variable_name_1: {operator_edge: MockNodeDiscretization(1)},
            }

        general_assembler = pp.Assembler(gb, active_variables=variable_name_1)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g11_ind = general_assembler.block_dof[(g1, variable_name_1)]
        g21_ind = general_assembler.block_dof[(g2, variable_name_1)]
        e1_ind = general_assembler.block_dof[(e, variable_name_1)]
        A_known[g11_ind, g11_ind] = 1

        A_known[g21_ind, g21_ind] = 3

        A_known[e1_ind, e1_ind] = 7

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_two_variables_coupling_between_node_and_edge(self):
        """ Two variables, coupling between the variables internal to each node.
        No coupling in the edge variable
        """
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        operator_1 = "operator_1"
        operator_2 = "operator_2"
        operator_edge = "operator_edge"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator_1: MockNodeDiscretization(1)},
                    variable_name_2: {operator_2: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    variable_name_1: {operator_1: MockNodeDiscretization(3)},
                    variable_name_2: {operator_2: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {
                variable_name_1: {"cells": 1},
                variable_name_2: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                variable_name_2: {operator_edge: MockNodeDiscretization(8)}
            }
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_discretization": {
                    g1: (variable_name_1, operator_1),
                    g2: (variable_name_1, operator_1),
                    e: (variable_name_1, MockEdgeDiscretization(1, 2)),
                }
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, g2, e, e]
        variables = [
            variable_name_1,
            variable_name_2,
            variable_name_1,
            variable_name_2,
            variable_name_1,
            variable_name_2,
        ]
        A_known = np.zeros((6, 6))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 3
        A_known[3, 3] = 4
        A_known[4, 4] = 1
        A_known[5, 5] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[4, 0] = -2
        A_known[4, 2] = -2
        A_known[0, 4] = 2
        A_known[2, 4] = 2

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(
            gb, active_variables=[variable_name_1, variable_name_2]
        )
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            permuted_assembler.block_dof,
            permuted_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_two_variables_coupling_between_node_and_edge_mixed_dependencies(self):
        """ Two variables, coupling between the variables internal to each node.
        No coupling in the edge variable
        """
        gb = self.define_gb()
        key_1 = "var_1"
        key_2 = "var_2"
        term = "op"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(1)},
                    key_2: {term: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(3)},
                    key_2: {term: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key_2: {term: MockNodeDiscretization(8)}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term: {
                    g1: (key_1, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(1, 2)),
                }
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, g2, e, e]
        variables = [key_1, key_2, key_1, key_2, key_1, key_2]
        A_known = np.zeros((6, 6))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 3
        A_known[3, 3] = 4
        A_known[4, 4] = 1
        A_known[5, 5] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[4, 0] = -2
        A_known[4, 3] = -2
        A_known[0, 4] = 2
        A_known[3, 4] = 2

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, active_variables=[key_1, key_2])
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            permuted_assembler.block_dof,
            permuted_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies(
        self
    ):
        """ One of the nodes has a single variable. A mortar variable depends on a combination
        mixture of the two variables
        """
        gb = self.define_gb()
        key_1 = "var_1"
        key_2 = "var_2"
        term = "op"
        for g, d in gb:

            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(1)},
                    key_2: {term: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                # Grid 2 has only one variable
                d[pp.PRIMARY_VARIABLES] = {key_2: {"cells": 1}}
                # We add a discretization to variable 1, but this should never be activated
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(3)},
                    key_2: {term: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key_2: {term: MockNodeDiscretization(8)}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term: {
                    g1: (key_1, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(1, 2)),
                }
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, e, e]
        variables = [key_1, key_2, key_2, key_1, key_2]
        A_known = np.zeros((5, 5))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 4
        A_known[3, 3] = 1
        A_known[4, 4] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[3, 0] = -2
        A_known[3, 2] = -2
        A_known[0, 3] = 2
        A_known[2, 3] = 2

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, active_variables=[key_1, key_2])
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            permuted_assembler.block_dof,
            permuted_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies_two_discretizations(
        self
    ):
        """ One of the nodes has a single variable. A mortar variable depends on a combination
        mixture of the two variables. The mortar variable has two discretizations.
        """
        gb = self.define_gb()
        key_1 = "var_1"
        key_2 = "var_2"
        term = "op"
        term2 = "term2"
        for g, d in gb:

            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(1)},
                    key_2: {term: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                # Grid 2 has only one variable
                d[pp.PRIMARY_VARIABLES] = {key_2: {"cells": 1}}
                # We add a discretization to variable 1, but this should never be activated
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(3)},
                    key_2: {term: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key_2: {term: MockNodeDiscretization(8)}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term: {
                    g1: (key_1, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(1, 2)),
                },
                term2: {
                    g1: (key_1, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(2, -4)),
                },
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, e, e]
        variables = [key_1, key_2, key_2, key_1, key_2]
        A_known = np.zeros((5, 5))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 4
        A_known[3, 3] = 3
        A_known[4, 4] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[3, 0] = 2
        A_known[3, 2] = 2
        A_known[0, 3] = -2
        A_known[2, 3] = -2

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, active_variables=[key_1, key_2])
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            permuted_assembler.block_dof,
            permuted_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies_two_discretizations_2(
        self
    ):
        """ One of the nodes has a single variable. A mortar variable depends on a combination
        mixture of the two variables. The mortar variable has two discretizations.
        """
        gb = self.define_gb()
        key_1 = "var_1"
        key_2 = "var_2"
        term = "op"
        term2 = "term2"
        for g, d in gb:

            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(1)},
                    key_2: {term: MockNodeDiscretization(2)},
                }
                g1 = g
            else:
                # Grid 2 has only one variable
                d[pp.PRIMARY_VARIABLES] = {key_2: {"cells": 1}}
                # We add a discretization to variable 1, but this should never be activated
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(3)},
                    key_2: {term: MockNodeDiscretization(4)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key_2: {term: MockNodeDiscretization(8)}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term: {
                    g1: (key_1, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(1, 2)),
                },
                term2: {
                    g1: (key_2, term),
                    g2: (key_2, term),
                    e: (key_1, MockEdgeDiscretization(0, 3)),
                },
            }

        # Assemble the global matrix
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        # Define "known" node/variable ordering and the corresponding solution matrix
        grids = [g1, g1, g2, e, e]
        variables = [key_1, key_2, key_2, key_1, key_2]
        A_known = np.zeros((5, 5))
        A_known[0, 0] = 1
        A_known[1, 1] = 2
        A_known[2, 2] = 4
        A_known[3, 3] = 1
        A_known[4, 4] = 8

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[3, 0] = -2
        A_known[3, 1] = -3
        A_known[3, 2] = -5
        A_known[0, 3] = 2
        A_known[1, 3] = 3
        A_known[2, 3] = 5

        # Permute the assembled matrix to the order defined above.
        A_permuted, _ = permute_matrix_vector(
            A,
            b,
            general_assembler.block_dof,
            general_assembler.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        new_assembler = pp.Assembler(gb, active_variables=[key_1, key_2])
        A_2, b_2 = general_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2, b_2, new_assembler.block_dof, new_assembler.full_dof, grids, variables
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_variable_one_sided_coupling_between_node_and_edge(self):
        """ Coupling between edge and one of the subdomains, but not the other
        """
        gb = self.define_gb()
        key = "var_1"
        term = "op"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {key: {term: MockNodeDiscretization(1)}}
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {key: {term: MockNodeDiscretization(3)}}
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                term: {
                    g1: (key, term),
                    g2: None,
                    e: (key, MockEdgeDiscretizationOneSided(2, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g11_ind = general_assembler.block_dof[(g1, key)]
        g22_ind = general_assembler.block_dof[(g2, key)]
        e1_ind = general_assembler.block_dof[(e, key)]

        A_known[g11_ind, g11_ind] = 1
        A_known[g22_ind, g22_ind] = 3
        A_known[e1_ind, e1_ind] = 2

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[e1_ind, g11_ind] = -1
        A_known[g11_ind, e1_ind] = 1

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_one_variable_one_sided_coupling_between_node_and_edge_different_operator_variable_names_modifies_node(
        self
    ):
        """ Coupling between edge and one of the subdomains, but not the other
        """
        gb = self.define_gb()
        term = "op"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {"var1": {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {"var1": {term: MockNodeDiscretization(1)}}
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {"var1": {term: MockNodeDiscretization(3)}}
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {"vare": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "tmp": {
                    g1: ("var1", term),
                    g2: None,
                    e: ("vare", MockEdgeDiscretizationOneSidedModifiesNode(2, 1)),
                }
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g11_ind = general_assembler.block_dof[(g1, "var1")]
        g22_ind = general_assembler.block_dof[(g2, "var1")]
        e1_ind = general_assembler.block_dof[(e, "vare")]

        A_known[g11_ind, g11_ind] = 2
        A_known[g22_ind, g22_ind] = 3
        A_known[e1_ind, e1_ind] = 2

        # Off-diagonal elements internal to the nodes: For the first node,
        # the first variable depends on the second, for the second, it is the
        # other way around.
        A_known[e1_ind, g11_ind] = -1
        A_known[g11_ind, e1_ind] = 1

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_partial_matrices_two_variables_single_discretization(self):
        """ A single variable, with multiple discretizations for one of the nodes.
        Do not add discretization matrices for individual terms
        """
        gb = self.define_gb()
        key_1 = "var_1"
        key_2 = "var_2"
        term = "op"
        term2 = "term2"
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            if g.grid_num == 1:
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(1)},
                    key_2: {term: MockNodeDiscretization(2)},
                    key_1 + "_" + key_2: {term: MockNodeDiscretization(5)},
                }
                g1 = g
            else:
                d[pp.DISCRETIZATION] = {
                    key_1: {term: MockNodeDiscretization(3)},
                    key_2: {term: MockNodeDiscretization(4)},
                    key_2 + "_" + key_1: {term: MockNodeDiscretization(6)},
                }
                g2 = g

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {key_1: {"cells": 1}, key_2: {"cells": 1}}
            d[pp.DISCRETIZATION] = {
                key_1: {term: MockNodeDiscretization(7)},
                key_2: {
                    term2: MockNodeDiscretization(8),
                    term: MockNodeDiscretization(7),
                },
                key_1 + "_" + key_2: {term: MockNodeDiscretization(6)},
                key_2 + "_" + key_1: {term: MockNodeDiscretization(1)},
            }

        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs(add_matrices=False)

        g11_ind = general_assembler.block_dof[(g1, key_1)]
        g12_ind = general_assembler.block_dof[(g1, key_2)]
        g21_ind = general_assembler.block_dof[(g2, key_1)]
        g22_ind = general_assembler.block_dof[(g2, key_2)]
        e1_ind = general_assembler.block_dof[(e, key_1)]
        e2_ind = general_assembler.block_dof[(e, key_2)]

        # First check first variable, which has only one term
        A_1_1 = np.zeros((6, 6))
        A_1_1[g11_ind, g11_ind] = 1
        A_1_1[g21_ind, g21_ind] = 3
        A_1_1[e1_ind, e1_ind] = 7
        self.assertTrue(np.allclose(A_1_1, A[term + "_" + key_1].todense()))

        # Second variable, first term
        A_2_2 = np.zeros((6, 6))
        A_2_2[g12_ind, g12_ind] = 2
        A_2_2[g22_ind, g22_ind] = 4
        A_2_2[e2_ind, e2_ind] = 7
        self.assertTrue(np.allclose(A_2_2, A[term + "_" + key_2].todense()))

        # Second variable, second term
        A_2_2 = np.zeros((6, 6))
        A_2_2[e2_ind, e2_ind] = 8
        self.assertTrue(np.allclose(A_2_2, A[term2 + "_" + key_2].todense()))

        # Mixed terms
        A_1_2 = np.zeros((6, 6))
        A_1_2[g11_ind, g12_ind] = 5
        A_1_2[e1_ind, e2_ind] = 6
        self.assertTrue(
            np.allclose(A_1_2, A[term + "_" + key_1 + "_" + key_2].todense())
        )


class MockNodeDiscretization(object):
    def __init__(self, value):
        self.value = value

    def assemble_matrix_rhs(self, g, data):
        return sps.coo_matrix(self.value), np.zeros(1)


class MockEdgeDiscretization(object):
    def __init__(self, diag_val, off_diag_val):
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, local_matrix
    ):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        cc[2, 2] = sps.coo_matrix(self.diag_val)
        cc[2, 0] = sps.coo_matrix(-self.off_diag_val)
        cc[2, 1] = sps.coo_matrix(-self.off_diag_val)
        cc[0, 2] = sps.coo_matrix(self.off_diag_val)
        cc[1, 2] = sps.coo_matrix(self.off_diag_val)

        return cc + local_matrix, np.empty(3)


class MockEdgeDiscretizationModifiesNode(object):
    def __init__(self, diag_val, off_diag_val):
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, local_matrix
    ):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        cc[2, 2] = sps.coo_matrix(self.diag_val)
        cc[2, 0] = sps.coo_matrix(-self.off_diag_val)
        cc[2, 1] = sps.coo_matrix(-self.off_diag_val)
        cc[0, 2] = sps.coo_matrix(self.off_diag_val)
        cc[1, 2] = sps.coo_matrix(self.off_diag_val)

        if g_master.grid_num == 1:
            local_matrix[0, 0] += sps.coo_matrix(self.off_diag_val)
            local_matrix[1, 1] += sps.coo_matrix(2 * self.off_diag_val)
        else:
            local_matrix[1, 1] += sps.coo_matrix(self.off_diag_val)
            local_matrix[0, 0] += sps.coo_matrix(2 * self.off_diag_val)

        return cc + local_matrix, np.empty(3)


class MockEdgeDiscretizationOneSided(object):
    # Discretization for the case where a mortar variable depends only on one side of the
    def __init__(self, diag_val, off_diag_val):
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(self, g_master, data_master, data_edge, local_matrix):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((2, 2))

        cc[1, 1] = sps.coo_matrix(self.diag_val)
        cc[1, 0] = sps.coo_matrix(-self.off_diag_val)

        cc[0, 1] = sps.coo_matrix(self.off_diag_val)

        return cc + local_matrix, np.empty(2)


class MockEdgeDiscretizationOneSidedModifiesNode(object):
    # Discretization for the case where a mortar variable depends only on one side of the
    def __init__(self, diag_val, off_diag_val):
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(self, g_master, data_master, data_edge, local_matrix):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((2, 2))

        cc[1, 1] = sps.coo_matrix(self.diag_val)
        cc[1, 0] = sps.coo_matrix(-self.off_diag_val)

        cc[0, 1] = sps.coo_matrix(self.off_diag_val)

        local_matrix[0, 0] += sps.coo_matrix(self.off_diag_val)

        return cc + local_matrix, np.empty(2)


if __name__ == "__main__":
    TestAssembler().test_single_variable_no_node_disretization()
    unittest.main()
