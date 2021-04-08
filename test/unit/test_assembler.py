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
9. Assemble opertors and parameters on grids.

NOTE: Throughout the tests, the indexing of the known disrcetization matrices
assumes that the order of the variables are the same as they are defined on the
grid bucket data dictionaries. If this under some system changes, everything
breaks.

"""
import unittest
from test.test_utils import permute_matrix_vector

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law
from porepy.numerics.mixed_dim.assembler_filters import ListFilter


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

        mg = pp.MortarGrid(2, {"left": g1, "right": g2})
        mg.num_cells = 1
        gb.set_edge_prop([g1, g2], "mortar_grid", mg)
        return gb

    def define_gb_three_grids(self):
        g1 = pp.CartGrid([1, 1])
        g2 = pp.CartGrid([1, 1])
        g3 = pp.CartGrid([1, 1])
        g1.compute_geometry()
        g2.compute_geometry()
        g3.compute_geometry()

        g1.grid_num = 1
        g2.grid_num = 2
        g3.grid_num = 3
        g1.dim = 2
        g2.dim = 1
        g3.dim = 1

        gb = pp.GridBucket()
        gb.add_nodes([g1, g2, g3])
        gb.add_edge([g1, g2], None)
        gb.add_edge([g1, g3], None)

        mg = pp.MortarGrid(1, {"left": g2, "right": g2})
        mg.num_cells = 1
        gb.set_edge_prop([g1, g2], "mortar_grid", mg)

        mg = pp.MortarGrid(1, {"left": g3, "right": g3})
        mg.num_cells = 1
        gb.set_edge_prop([g1, g3], "mortar_grid", mg)

        return gb

    ### Test with no coupling between the subdomains

    def test_single_variable(self):
        """A single variable, test that the basic mechanics of the assembler functions."""
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        g2_ind = dof_manager.block_dof[(g2, variable_name)]
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = dof_manager.block_dof[(g1, "variable_1")]
        g2_ind = dof_manager.block_dof[(g2, "variable_2")]
        A_known[g1_ind, g1_ind] = 2
        A_known[g2_ind, g2_ind] = 4

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_no_node_disretization(self):
        """A single variable, where one of the nodes have no discretization
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_not_active(self):
        """A single variable, but make this inactive. This should return an empty matrix"""

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

        filt = ListFilter(variable_list=["var_11"])

        # Give a false variable name
        general_assembler = pp.Assembler(gb)
        A, b = general_assembler.assemble_matrix_rhs(filt=filt)
        self.assertTrue(A.data.size == 0)
        self.assertTrue(np.sum(np.abs(b)) == 0)

    def test_explicitly_define_edge_variable_active(self):
        """Explicitly define edge and node variables as active. The result should
        be the same as if no active variable was defined.

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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[1, 0, 1], [0, 2, 1], [-1, -1, 1]])
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_define_edge_variable_inactive(self):
        """Define edge-variable as inactive. The resulting system should have
        no coupling term.

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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        filt = ListFilter(variable_list=[variable_name])
        A, b = general_assembler.assemble_matrix_rhs(filt=filt)

        # System matrix, the coupling terms should not have been assembled
        A_known = np.zeros((3, 3))
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        g2_ind = dof_manager.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = 2

        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_multiple_node_discretizations(self):
        """A single variable, with multiple discretizations for one of the nodes"""
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 1], [0, 0, 1], [-1, -1, 1]])
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        g2_ind = dof_manager.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 5
        A_known[g2_ind, g2_ind] = 2
        self.assertTrue(np.allclose(A_known, A.todense()))

    def test_single_variable_multiple_edge_discretizations(self):
        """A single variable, with multiple discretizations for one of the e"""
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.array([[0, 0, 2], [0, 0, 2], [-2, -2, -2]])
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        g2_ind = dof_manager.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = 2
        assert np.allclose(A_known, A.todense())

    def test_two_variables_no_coupling(self):
        """Two variables, no coupling between the variables. Test that the
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((6, 6))

        g11_ind = dof_manager.block_dof[(g1, variable_name_1)]
        g12_ind = dof_manager.block_dof[(g1, variable_name_2)]
        g21_ind = dof_manager.block_dof[(g2, variable_name_1)]
        g22_ind = dof_manager.block_dof[(g2, variable_name_2)]
        e1_ind = dof_manager.block_dof[(e, variable_name_edge_1)]
        e2_ind = dof_manager.block_dof[(e, variable_name_edge_2)]
        A_known[g11_ind, g11_ind] = 1
        A_known[g12_ind, g12_ind] = 2
        A_known[g21_ind, g21_ind] = 3
        A_known[g22_ind, g22_ind] = 4
        A_known[e1_ind, e1_ind] = 5
        A_known[e2_ind, e2_ind] = 6
        assert np.allclose(A_known, A.todense())

    def test_two_variables_one_active(self):
        """Define two variables, but then only assemble with respect to one
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

        general_assembler = pp.Assembler(gb)
        filt = ListFilter(variable_list=[variable_name_2])
        A, _ = general_assembler.assemble_matrix_rhs(filt=filt)

        A_known = np.zeros((6, 6))

        g12_ind = dof_manager.block_dof[(g1, variable_name_2)]
        g22_ind = dof_manager.block_dof[(g2, variable_name_2)]
        e2_ind = dof_manager.block_dof[(e, variable_name_2)]

        A_known[g12_ind, g12_ind] = 2
        A_known[g22_ind, g22_ind] = 4
        A_known[e2_ind, e2_ind] = 6

        assert np.allclose(A_known, A.todense())

    def test_filter_grid(self):
        # Use a list filter to only discretize on one node
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
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        g2_ind = dof_manager.block_dof[(g2, variable_name)]
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = 2
        assert np.allclose(A_known, A.todense())

    def test_two_variables_no_coupling(self):
        """Two variables, no coupling between the variables. Test that the
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((6, 6))

        g11_ind = dof_manager.block_dof[(g1, variable_name_1)]
        g12_ind = dof_manager.block_dof[(g1, variable_name_2)]
        g21_ind = dof_manager.block_dof[(g2, variable_name_1)]
        g22_ind = dof_manager.block_dof[(g2, variable_name_2)]
        e1_ind = dof_manager.block_dof[(e, variable_name_edge_1)]
        e2_ind = dof_manager.block_dof[(e, variable_name_edge_2)]
        A_known[g11_ind, g11_ind] = 1
        A_known[g12_ind, g12_ind] = 2
        A_known[g21_ind, g21_ind] = 3
        A_known[g22_ind, g22_ind] = 4
        A_known[e1_ind, e1_ind] = 5
        A_known[e2_ind, e2_ind] = 6
        assert np.allclose(A_known, A.todense())

    def test_two_variables_one_active(self):
        """Define two variables, but then only assemble with respect to one
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        filt = ListFilter(variable_list=[variable_name_2])
        A, _ = general_assembler.assemble_matrix_rhs(filt=filt)

        A_known = np.zeros((6, 6))

        g12_ind = dof_manager.block_dof[(g1, variable_name_2)]
        g22_ind = dof_manager.block_dof[(g2, variable_name_2)]
        e2_ind = dof_manager.block_dof[(e, variable_name_2)]

        A_known[g12_ind, g12_ind] = 2
        A_known[g22_ind, g22_ind] = 4
        A_known[e2_ind, e2_ind] = 6

        assert np.allclose(A_known, A.todense())

    def test_filter_grid(self):
        # Use a list filter to only discretize on one node
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
            d[pp.DISCRETIZATION] = {
                variable_name_e: {term_e: MockNodeDiscretization(7)}
            }
            d[pp.COUPLING_DISCRETIZATION] = {
                term_e: {
                    g1: (variable_name, term_g1),
                    g2: (variable_name, term_g2),
                    e: (variable_name_e, MockEdgeDiscretization(1, 1)),
                }
            }

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        g1_ind = dof_manager.block_dof[(g1, variable_name)]
        e_ind = dof_manager.block_dof[(e, variable_name_e)]

        # Only grid 1
        filt = ListFilter(grid_list=[g1])
        A, b = general_assembler.assemble_matrix_rhs(filt=filt)
        A_known = np.zeros((3, 3))
        A_known[g1_ind, g1_ind] = 1
        self.assertTrue(np.allclose(A_known, A.todense()))

        filt = ListFilter(grid_list=[e])
        A, b = general_assembler.assemble_matrix_rhs(filt=filt)
        A_known = np.zeros((3, 3))
        A_known[e_ind, e_ind] = 7
        self.assertTrue(np.allclose(A_known, A.todense()))

    ### Tests with coupling internal to each node
    def test_two_variables_coupling_within_node_and_edge(self):
        """Two variables, coupling between the variables internal to each node.
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
        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb)

        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    # Tests with node-edge couplings
    def test_two_variables_coupling_between_node_and_edge(self):
        """Two variables, coupling between the variables internal to each node.
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
        dof_manager = pp.DofManager(gb)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, dof_manager)
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_two_variables_coupling_between_node_and_edge_mixed_dependencies(self):
        """Two variables, coupling between the variables internal to each node.
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
        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb)
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies(
        self,
    ):
        """One of the nodes has a single variable. A mortar variable depends on a combination
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
        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, dof_manager)
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies_two_discretizations(
        self,
    ):
        """One of the nodes has a single variable. A mortar variable depends on a combination
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
        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        permuted_assembler = pp.Assembler(gb, dof_manager)
        A_2, b_2 = permuted_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2,
            b_2,
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_and_two_variables_coupling_between_node_and_edge_mixed_dependencies_two_discretizations_2(
        self,
    ):
        """One of the nodes has a single variable. A mortar variable depends on a combination
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
        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
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
            dof_manager.block_dof,
            dof_manager.full_dof,
            grids,
            variables,
        )
        self.assertTrue(np.allclose(A_known, A_permuted.todense()))

        # Next, define both variables to be active. Should be equivalent to
        # runing without the variables argument
        new_assembler = pp.Assembler(gb, dof_manager)
        A_2, b_2 = general_assembler.assemble_matrix_rhs()
        A_2_permuted, _ = permute_matrix_vector(
            A_2, b_2, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        self.assertTrue(np.allclose(A_known, A_2_permuted.todense()))

    def test_one_variable_one_sided_coupling_between_node_and_edge(self):
        """Coupling between edge and one of the subdomains, but not the other"""
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

        dof_manager = pp.DofManager(gb)            
        general_assembler = pp.Assembler(gb, dof_manager)
        A, b = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g11_ind = dof_manager.block_dof[(g1, key)]
        g22_ind = dof_manager.block_dof[(g2, key)]
        e1_ind = dof_manager.block_dof[(e, key)]

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
        self,
    ):
        """Coupling between edge and one of the subdomains, but not the other"""
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

        dof_manager = pp.DofManager(gb)            
        general_assembler = pp.Assembler(gb, dof_manager)
        A, _ = general_assembler.assemble_matrix_rhs()

        A_known = np.zeros((3, 3))

        g11_ind = dof_manager.block_dof[(g1, "var1")]
        g22_ind = dof_manager.block_dof[(g2, "var1")]
        e1_ind = dof_manager.block_dof[(e, "vare")]

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
        """A single variable, with multiple discretizations for one of the nodes.
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

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        A, _ = general_assembler.assemble_matrix_rhs(add_matrices=False)

        g11_ind = dof_manager.block_dof[(g1, key_1)]
        g12_ind = dof_manager.block_dof[(g1, key_2)]
        g21_ind = dof_manager.block_dof[(g2, key_1)]
        g22_ind = dof_manager.block_dof[(g2, key_2)]
        e1_ind = dof_manager.block_dof[(e, key_1)]
        e2_ind = dof_manager.block_dof[(e, key_2)]

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

    def test_assemble_operator_nodes(self):
        """Test assembly of operator on nodes"""
        gb = self.define_gb()
        key_1 = "var_1"
        term = "op"
        # Variable name assigned on nodes. Same for both grids
        i = 0
        for g, d in gb:
            if g.grid_num == 1:
                A, _ = MockNodeDiscretization(1).assemble_matrix_rhs(g, d)
                d[pp.DISCRETIZATION_MATRICES] = {key_1: {term: A}}
                g1_ind = i
            else:
                A, _ = MockNodeDiscretization(-1).assemble_matrix_rhs(g, d)
                d[pp.DISCRETIZATION_MATRICES] = {key_1: {term: A}}
                g2_ind = i
            i += 1
        for e, d in gb.edges():
            d[pp.DISCRETIZATION_MATRICES] = {}

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        Op = general_assembler.assemble_operator(key_1, term)

        A_known = np.array([[0, 0], [0, 0]])
        A_known[g1_ind, g1_ind] = 1
        A_known[g2_ind, g2_ind] = -1

        self.assertTrue(np.allclose(A_known, Op.todense()))

    def test_assemble_operator_edges(self):
        """Test assembly of operator on edges"""
        gb = self.define_gb()
        key_1 = "var_1"
        term = "op"
        # Variable name assigned on nodes. Same for both grids
        for _, d in gb.edges():
            A = sps.csc_matrix(np.array([[1, 2, 3]]))
            d[pp.DISCRETIZATION_MATRICES] = {key_1: {term: A}}
        for _, d in gb:
            d[pp.DISCRETIZATION_MATRICES] = {}

        general_assembler = pp.Assembler(gb)
        Op = general_assembler.assemble_operator(key_1, term)

        A_known = np.array([[1, 2, 3]])
        self.assertTrue(np.allclose(A_known, Op.todense()))

    def test_assemble_parameters_node(self):
        """Test assembly of operator on nodes"""
        gb = self.define_gb()
        key_1 = "var_1"
        term = "op"
        # Variable name assigned on nodes. Same for both grids
        i = 0
        for g, d in gb:
            if g.grid_num == 1:
                param = np.array([1, 3, 4])
                d[pp.PARAMETERS] = {key_1: {term: param}}
                g1_ind = i
            else:
                param = np.array([5, 6, 7])
                d[pp.PARAMETERS] = {key_1: {term: param}}

            i += 1
        for e, d in gb.edges():
            d[pp.PARAMETERS] = {}

        dof_manager = pp.DofManager(gb)
        general_assembler = pp.Assembler(gb, dof_manager)
        P = general_assembler.assemble_parameter(key_1, term)

        if g1_ind == 0:
            param_known = np.array([1, 3, 4, 5, 6, 7])
        else:
            param_known = np.array([5, 6, 7, 1, 3, 4])
        self.assertTrue(np.allclose(param_known, P))

    # Tests with explicit coupling between two nodes
    def test_direct_edge_coupling(self):
        gb = self.define_gb_three_grids()

        node_var = "node_var"
        edge_var = "edge_var"
        term = "op"

        edge_self_val = 1
        edge_other_val = -1

        node_discretization = MockNodeDiscretization(1)
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}
            d[pp.DISCRETIZATION] = {node_var: {term: node_discretization}}
        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {edge_var: {"cells": 1}}
            g1, g2 = gb.nodes_of_edge(e)
            if g1.grid_num == 2:
                e1 = e
                d[pp.COUPLING_DISCRETIZATION] = {
                    "coupling_discretization": {
                        g1: (node_var, term),
                        g2: (node_var, term),
                        e: (
                            edge_var,
                            MockEdgeDiscretizationEdgeCouplings(
                                node_discretization, edge_self_val, edge_other_val
                            ),
                        ),
                    }
                }
            elif g1.grid_num == 3:
                e2 = e
                d[pp.COUPLING_DISCRETIZATION] = {
                    "coupling_discretization": {
                        g1: (node_var, term),
                        g2: (node_var, term),
                        e: (
                            edge_var,
                            MockEdgeDiscretizationEdgeCouplings(
                                node_discretization, edge_self_val, -edge_other_val
                            ),
                        ),
                    }
                }

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        A, _ = assembler.assemble_matrix_rhs()

        e1_ind = dof_manager.block_dof[(e1, edge_var)]
        e2_ind = dof_manager.block_dof[(e2, edge_var)]

        self.assertTrue(A[e1_ind, e1_ind] == edge_self_val)
        self.assertTrue(A[e1_ind, e2_ind] == edge_other_val)
        self.assertTrue(A[e2_ind, e1_ind] == -edge_other_val)
        self.assertTrue(A[e2_ind, e2_ind] == edge_self_val)

    def test_variable_of_node(self):
        # Test function variable of node. Two nodes, different variables.
        # Also edge between nodes.
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        for g, d in gb:
            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {
                    variable_name_1: {"cells": 1},
                    variable_name_2: {"cells": 1},
                }
                g1 = g
            else:
                g2 = g
                d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}

        # Assemble the global matrix
        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        var = assembler.variables_of_grid(g1)
        self.assertTrue(variable_name_1 in var)
        self.assertTrue(variable_name_2 in var)

        var = assembler.variables_of_grid(g2)
        self.assertTrue(variable_name_1 in var)
        self.assertFalse(variable_name_2 in var)

        var = assembler.variables_of_grid(e)
        self.assertTrue(variable_name_1 in var)
        self.assertFalse(variable_name_2 in var)

    def test_str_repr_two_nodes_different_variables(self):
        # Assign two variables, check that the string returned by __str__ and
        # __repr__ contain the correct information
        # Test function variable of node. Two nodes, different variables.
        # Also edge between nodes.
        gb = self.define_gb()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        for g, d in gb:
            if g.grid_num == 1:
                d[pp.PRIMARY_VARIABLES] = {
                    variable_name_1: {"cells": 1},
                    variable_name_2: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}
                g.dim = 1

        for e, d in gb.edges():
            d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}

        # Assemble the global matrix
        dof_manager = pp.DofManager(gb)        
        assembler = pp.Assembler(gb, dof_manager)
        string = assembler.__str__()

        # Check that the target line is in the string, or else the below if is
        # meaningless
        self.assertTrue("Variable names" in string)
        for line in string.split("\n"):
            # Check that the variables information is included in the right line
            if "Variable names" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(variable_name_2 in line)

        rep = assembler.__repr__()
        self.assertTrue("in dimension 2" in rep)
        self.assertTrue("in dimension 1" in rep)

        for line in rep.split("\n"):
            if "in dimension 2" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(variable_name_2 in line)
            elif "in dimension 1" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(not variable_name_2 in line)

        self.assertTrue("dimensions 2 and 1" in rep)
        for line in rep.split("\n"):
            if "in dimensions 2 and 1" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(not variable_name_2 in line)

    def test_repr_three_nodes_three_edges_different_variables(self):
        # Assembler.__str__ will be the same as in the above two-node test. Focus on
        # __repr__, with heterogeneous physics
        gb = self.define_gb_three_grids()
        variable_name_1 = "var_1"
        variable_name_2 = "var_2"
        for g, d in gb:
            if g.grid_num < 3:
                d[pp.PRIMARY_VARIABLES] = {
                    variable_name_1: {"cells": 1},
                    variable_name_2: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}
                g3 = g

        for e, d in gb.edges():
            if g3 in e:
                d[pp.PRIMARY_VARIABLES] = {variable_name_1: {"cells": 1}}
            else:
                d[pp.PRIMARY_VARIABLES] = {
                    variable_name_1: {"cells": 1},
                    variable_name_2: {"cells": 1},
                }

        # Assemble the global matrix
        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        rep = assembler.__repr__()

        self.assertTrue("in dimension 2" in rep)
        self.assertTrue("in dimension 1" in rep)

        found_var_1_subdomain = False
        found_var_1_var_2_subdomain = False

        for line in rep.split("\n"):
            if "present in dimension 2" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(variable_name_2 in line)
            elif "present in dimension 1" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(variable_name_2 in line)
            if "subdomain in dimension 1" in line:
                if variable_name_1 in line and variable_name_2 in line:
                    found_var_1_var_2_subdomain = True
                elif variable_name_1 in line:
                    found_var_1_subdomain = True

        self.assertTrue(found_var_1_subdomain)
        self.assertTrue(found_var_1_var_2_subdomain)

        self.assertTrue("edges between dimensions" in rep)
        self.assertTrue("interface between dimension" in rep)

        found_var_1_interface = False
        found_var_1_var_2_interface = False

        for line in rep.split("\n"):
            if "edges between dimensions 2" in line:
                self.assertTrue(variable_name_1 in line)
                self.assertTrue(variable_name_2 in line)
            if "interface between dimension" in line:
                if variable_name_1 in line and variable_name_2 in line:
                    found_var_1_var_2_interface = True
                elif variable_name_1 in line:
                    found_var_1_interface = True

        self.assertTrue(found_var_1_interface)
        self.assertTrue(found_var_1_var_2_interface)


class MockNodeDiscretization(object):
    def __init__(self, value):
        self.value = value

    def assemble_matrix_rhs(self, g, data=None):
        return sps.coo_matrix(self.value), np.zeros(1)

    def ndof(self, g):
        return g.num_cells


class MockEdgeDiscretization(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    def __init__(self, diag_val, off_diag_val):
        super(MockEdgeDiscretization, self).__init__("")
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(
        self,
        g_primary,
        g_secondary,
        data_primary,
        data_secondary,
        data_edge,
        local_matrix,
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

    def ndof(self, mg):
        pass

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        pass


class MockEdgeDiscretizationModifiesNode(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    def __init__(self, diag_val, off_diag_val):
        super(MockEdgeDiscretizationModifiesNode, self).__init__("")
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(
        self,
        g_primary,
        g_secondary,
        data_primary,
        data_secondary,
        data_edge,
        local_matrix,
    ):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        cc[2, 2] = sps.coo_matrix(self.diag_val)
        cc[2, 0] = sps.coo_matrix(-self.off_diag_val)
        cc[2, 1] = sps.coo_matrix(-self.off_diag_val)
        cc[0, 2] = sps.coo_matrix(self.off_diag_val)
        cc[1, 2] = sps.coo_matrix(self.off_diag_val)

        if g_primary.grid_num == 1:
            local_matrix[0, 0] += sps.coo_matrix(self.off_diag_val)
            local_matrix[1, 1] += sps.coo_matrix(2 * self.off_diag_val)
        else:
            local_matrix[1, 1] += sps.coo_matrix(self.off_diag_val)
            local_matrix[0, 0] += sps.coo_matrix(2 * self.off_diag_val)

        return cc + local_matrix, np.empty(3)

    def ndof(self, mg):
        pass

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        pass


class MockEdgeDiscretizationOneSided(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    # Discretization for the case where a mortar variable depends only on one side of the
    def __init__(self, diag_val, off_diag_val):
        super(MockEdgeDiscretizationOneSided, self).__init__("")
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(self, g_primary, data_primary, data_edge, local_matrix):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((2, 2))
        cc[1, 1] = sps.coo_matrix(self.diag_val)
        cc[1, 0] = sps.coo_matrix(-self.off_diag_val)
        cc[0, 1] = sps.coo_matrix(self.off_diag_val)

        return cc + local_matrix, np.empty(2)

    def ndof(self, mg):
        pass

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        pass


class MockEdgeDiscretizationOneSidedModifiesNode(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    # Discretization for the case where a mortar variable depends only on one side of the
    def __init__(self, diag_val, off_diag_val):
        super(MockEdgeDiscretizationOneSidedModifiesNode, self).__init__("")
        self.diag_val = diag_val
        self.off_diag_val = off_diag_val

    def assemble_matrix_rhs(self, g_primary, data_primary, data_edge, local_matrix):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((2, 2))

        cc[1, 1] = sps.coo_matrix(self.diag_val)
        cc[1, 0] = sps.coo_matrix(-self.off_diag_val)

        cc[0, 1] = sps.coo_matrix(self.off_diag_val)

        local_matrix[0, 0] += sps.coo_matrix(self.off_diag_val)

        return cc + local_matrix, np.empty(2)

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        pass

    def ndof(self, mg):
        pass


class MockEdgeDiscretizationEdgeCouplings(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    def __init__(self, discr_node, self_val, other_val):
        super(MockEdgeDiscretizationEdgeCouplings, self).__init__("")
        self.self_val = self_val
        self.other_val = other_val
        self.edge_coupling_via_high_dim = True
        self.discr_grid = discr_node

    def ndof(self, mg):
        return mg.num_cells

    def assemble_matrix_rhs(
        self,
        g_primary,
        g_secondary,
        data_primary,
        data_secondary,
        data_edge,
        local_matrix,
    ):

        dof = [local_matrix[0, i].shape[1] for i in range(local_matrix.shape[1])]
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))
        cc[2, 2] = sps.coo_matrix(self.self_val)
        cc[2, 0] = sps.coo_matrix(1)
        cc[2, 1] = sps.coo_matrix(1)
        cc[0, 2] = sps.coo_matrix(1)
        cc[1, 2] = sps.coo_matrix(1)

        return cc + local_matrix, np.empty(3)

    def assemble_edge_coupling_via_high_dim(
        self,
        g_between,
        data_between,
        edge_primary,
        data_edge_primary,
        edge_secondary,
        data_edge_secondary,
        matrix,
        assemble_matrix=True,
        assemble_rhs=True,
    ):
        mg_primary = data_edge_primary["mortar_grid"]
        mg_secondary = data_edge_secondary["mortar_grid"]
        cc, rhs = self._define_local_block_matrix_edge_coupling(
            g_between, self.discr_grid, mg_primary, mg_secondary, matrix
        )
        cc[1, 2] = sps.coo_matrix(self.other_val)

        return cc + matrix, rhs

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        pass


class TestAssemblerFilters(unittest.TestCase):
    def test_all_pass(self):
        # The AllPassFilter should pass anything. Test this with by
        # sending in a variable
        filt = pp.assembler_filters.AllPassFilter()
        self.assertTrue(filt.filter(variables="var not in fliter"))

    def test_list_filter_grid_keyword(self):
        g1 = pp.CartGrid([1])
        g2 = pp.CartGrid([1])
        g3 = pp.CartGrid([1])

        # Single grid
        filt = pp.assembler_filters.ListFilter(grid_list=[g1])
        self.assertTrue(filt.filter(g1))
        self.assertFalse(filt.filter(g2))
        # Pass a list with one grid
        self.assertTrue(filt.filter([g1]))

        # two grids
        filt = pp.assembler_filters.ListFilter(grid_list=[g1, g2])
        self.assertTrue(filt.filter(g1))
        self.assertTrue(filt.filter(g2))
        self.assertFalse(filt.filter(g3))

        # interface
        filt = pp.assembler_filters.ListFilter(grid_list=[(g1, g2)])
        self.assertTrue(filt.filter((g1, g2)))
        self.assertFalse(filt.filter((g3, g2)))
        # Check that we can pass a list of interfaces
        self.assertTrue(filt.filter([(g1, g2)]))

        # couplings
        filter = pp.assembler_filters.ListFilter(grid_list=[(g1, g2, (g1, g2))])
        self.assertTrue(filter.filter((g1, g2, (g1, g2))))
        self.assertFalse(filter.filter((g1, g3, (g1, g3))))

    def test_list_filter_variable_keyword(self):
        # Note: Since variable and term filters share the implementation, we test
        # only the former
        v1 = "var1"
        v2 = "var2"
        v3 = "var3"

        filt = pp.assembler_filters.ListFilter(variable_list=["var1"])
        self.assertTrue(filt.filter(variables=[v1]))
        self.assertFalse(filt.filter(variables=[v2]))
        self.assertFalse(filt.filter(variables=[v1, v2]))

        filt = pp.assembler_filters.ListFilter(variable_list=["var1", "var2"])
        self.assertTrue(filt.filter(variables=[v1]))
        self.assertTrue(filt.filter(variables=[v2]))
        self.assertTrue(filt.filter(variables=[v2, v1]))
        self.assertFalse(filt.filter(variables=[v3]))
        self.assertFalse(filt.filter(variables=[v1, v3]))

        n1 = "!var1"
        filt = pp.assembler_filters.ListFilter(variable_list=[n1])
        self.assertTrue(filt.filter(variables=[v2]))
        self.assertFalse(filt.filter(variables=[v1]))

        # It should not be possible to create a filter with both a variable
        # and its negation
        self.assertRaises(ValueError, pp.assembler_filters.ListFilter, v1, n1)

    def test_grid_and_variable_keywords(self):
        var1 = "v1"
        var2 = "v2"

        g1 = pp.CartGrid(1)
        g2 = pp.CartGrid(2)

        filter = pp.assembler_filters.ListFilter(grid_list=[g1], variable_list=[var1])
        self.assertTrue(filter.filter(grids=[g1], variables=[var1]))
        self.assertFalse(filter.filter(grids=[g2], variables=[var1]))
        self.assertFalse(filter.filter(grids=[g1], variables=[var2]))


TestAssembler().test_filter_grid()
if __name__ == "__main__":
    unittest.main()
