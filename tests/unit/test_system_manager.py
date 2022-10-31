"""Tests of the EquationManager used in the Ad framework.

The tests focus on various assembly methods:
    * test_secondary_variable_assembly: Test of assembly when secondary variables are
        present.
    * test_assemble_subsystem: Assemble blocks of the full set of equations.
    * test_extract_subsystem: Extract a new EquationManager for a subset of equations.
    * test_schur_complement: Assemble a subsystem, using a Schur complement reduction.

"""
import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


class SystemManagerSetup:
    """Class to set up an EquationManager with a combination of variables
    and equations, designed to make it convenient to test critical functionality
    of the EM.
    """

    def __init__(self):
        sd_1 = pp.CartGrid(np.array([3, 1]))
        sd_2 = pp.CartGrid(np.array([4, 1]))

        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([sd_1, sd_2])

        for sd, data in mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES] = {
                "x": {"cells": 1},
                "y": {"cells": 1},
                "x_merged": {"cells": 1},
                "y_merged": {"cells": 1},
            }
            if sd == sd_2:
                # Add a third variable on the second grid
                data[pp.PRIMARY_VARIABLES].update({"z": {"cells": 1}})
            else:
                data[pp.PRIMARY_VARIABLES].update({"x2": {"cells": 1}})

            x = 2 * np.ones(sd.num_cells)
            y = 3 * np.ones(sd.num_cells)
            z = 4 * np.ones(sd.num_cells)

            data[pp.STATE] = {
                "x": x,
                "y": y,
                "z": z,
                "x_merged": x.copy(),
                "y_merged": y.copy(),
                "x2": x.copy(),
                pp.ITERATE: {"x": x.copy(), "y": y.copy(), "z": z.copy(),
                            "x_merged": x.copy(), "y_merged": y.copy(), 'x2': x.copy()},
            }

        # Ad representation of variables
        eq_manager = pp.ad.SystemManager(mdg)

        x_ad = eq_manager.create_variable("x", subdomains=[sd_2])
        y_ad = eq_manager.create_variable("y", subdomains=[sd_2])
        z_ad = eq_manager.create_variable("z", subdomains=[sd_2])

        x_merged = eq_manager.create_variable("x_merged", subdomains=[sd_1, sd_2])
        y_merged = eq_manager.create_variable("y_merged", subdomains=[sd_1, sd_2])

        projections = pp.ad.SubdomainProjections(subdomains=[sd_1, sd_2])
        proj = projections.cell_restriction([sd_2])

        # One equation with only simple variables (not merged)
        eq_simple = x_ad * y_ad + 2 * z_ad
        # One equation using only merged variables
        eq_merged = y_merged * (1 + x_merged)
        # One equation combining merged and simple variables
        eq_combined = x_ad + y_ad * (proj * y_merged)

        eq_manager.equations.update(
            {"simple": eq_simple, "merged": eq_merged, "combined": eq_combined}
        )

        self.x1 = x_ad
        self.x2 = eq_manager.create_variable("x2", subdomains=[sd_1])

        self.z1 = z_ad
        self.x_merged = x_merged
        self.y_merged = y_merged

        self.sd_1 = sd_1
        self.sd_2 = sd_2

        A, b = eq_manager.assemble()
        self.A = A
        self.b = b
        self.eq_manager = eq_manager

    ## Helper methods below.

    def var_size(self, var):
        # For a given variable, get its size (number of dofs) based on what
        # we know about the variables specified in self.__init__
        if var == self.x1:
            return self.sd_2.num_cells
        elif var == self.x_merged:
            return self.sd_1.num_cells + self.sd_2.num_cells
        elif var == self.y_merged:
            return self.sd_1.num_cells + self.sd_2.num_cells
        elif var == self.z1:
            return self.sd_2.num_cells
        else:
            raise ValueError

    def dof_ind(self, var):
        # For a given variable, get the global indices assigned by
        # the DofManager. Based on knowledge of how the variables were
        # defined in self.__init__

        def inds(g, n):
            return self.eq_manager.dofs_of(g, n)

        return self.eq_manager.dofs_of(var)
        if var in (self.x1, self.x2, self.z1):
            return self.eq_manager.dofs_of(var)# inds(self.x1.domain, self.x1.name)

#        elif var == self.x2:
#            return inds(self.x2.domain, self.x1.name)

        elif var == self.x_merged:
            dofs = np.hstack(
                [
                    inds(self.sd_1, self.x_merged.name),
                    inds(self.sd_2, self.x_merged.name),
                ]
            )
            return dofs

        elif var == self.y_merged:
            dofs = np.hstack(
                [
                    inds(self.sd_1, self.y_merged.name),
                    inds(self.sd_2, self.y_merged.name),
                ]
            )
            return dofs

#        elif var == self.z1:
#            return inds(self.z1.domain, self.z1.name)
        else:
            raise ValueError

    def eq_ind(self, name):
        # Get row indices of an equation, based on the (known) order in which
        # equations were added to the EquationManager
        if name == "simple":
            return np.arange(4)
        elif name == "merged":
            return 4 + np.arange(7)
        elif name == "combined":
            return 11 + np.arange(4)
        else:
            raise ValueError

    def block_size(self, name):
        # Get the size of an equation block
        if name == "simple":
            return 4
        elif name == "merged":
            return 7
        elif name == "combined":
            return 4
        else:
            raise ValueError


def _eliminate_columns_from_matrix(A, indices, reverse):
    # Helper method to extract submatrix by column indices
    if reverse:
        inds = np.setdiff1d(np.arange(A.shape[1]), indices)
    else:
        inds = indices
    return A[:, inds]


@pytest.fixture
def setup():
    # Method to deliver a setup to all tests
    return SystemManagerSetup()


def _eliminate_rows_from_matrix(A, indices, reverse):
    # Helper method to extract submatrix by row indices
    if reverse:
        inds = np.setdiff1d(np.arange(A.shape[0]), indices)
    else:
        inds = indices
    return A[inds]


def _compare_matrices(m1, m2):
    # Helper method to compare two matrices. Returns True only if the matrices are
    # identical up to a small tolerance.
    if m1.shape != m2.shape:
        # Matrices with different shape are unequal, unless the number of
        # rows and columns is zero in at least one dimension.
        if m1.shape[0] == 0 and m2.shape[0] == 0:
            return True
        elif m1.shape[1] == 0 and m2.shape[1] == 0:
            return True
        return False
    d = m1 - m2

    if d.data.size > 0:
        if np.max(np.abs(d.data)) > 1e-10:
            return False
    return True


@pytest.mark.parametrize(
    "var_names",
    [
        [],  # No secondary variables
        ["x"],  # A simple variable which is also part of a merged one
        ["x_merged"],  # Merged variable
        ["z"],  # Simple variable not available as merged
        ["z", "y_merged"],  # Combination of simple and merged.
    ],
)
def test_secondary_variable_assembly(setup, var_names):
    # Test of the standard assemble method. The only interesting test is the
    # secondary variable functionality (the other functionality is tested elsewhere).

    # The tests compare assembly by an EquationManager with explicitly defined
    # secondary variables with a 'truth' based on direct elimination of columns
    # in the Jacobian matrix.
    # The expected behavior is that the residual vector is fixed, while the
    # Jacobian matrix is altered only in columns that correspond to eliminated
    # variables.

#    secondary_variables = [getattr(setup, name) for name in var_names]
    variables = [var for var in setup.eq_manager.variables]
    for var in var_names:
        variables.remove(var)
    A, b = setup.eq_manager.assemble_subsystem(variables=variables)

    # Get dof indices of the variables that have been eliminated
    if len(var_names) > 0:
        dofs = np.sort(np.hstack([setup.eq_manager.dofs_of(var) for var in var_names]))
    else:
        dofs = []

    # The residual vectors should be the same
    assert np.allclose(b, setup.b)
    # Compare system matrices. Since the variables specify columns to eliminate,
    # we use the reverse argument
    assert _compare_matrices(
        A, _eliminate_columns_from_matrix(setup.A, dofs, reverse=True)
    )
    # Check that the size of equation blocks (rows) were correctly recorded
    for name in setup.eq_manager.equations:
        assert np.allclose(setup.eq_manager.assembled_equation_indices[name], setup.eq_ind(name))



@pytest.mark.parametrize(
    "var_names",
    [
        None,  # Should give all variables by default
        [],  # Empty list (as opposed to None) should give a system with zero columns
        ["x"],  # simple variable
        ["y_merged"],  # Merged variable
        ["x2", "y_merged"],  # Combination of simple and merged
        ["y_merged", "x"],  # Combination reversed - check just to be sure
        ["x_merged", "x"],  # Merged and simple version of same variable
    ],
)
@pytest.mark.parametrize(
    "eq_names",
    [
        None,  # should give all equations by default
        [],  # Empty list should give a system with zero rows
        ["combined"],  # Single equation
        ["simple", "merged"],  # Combination of two equations
        ["merged", "simple"],  # Combination reversed
        ["simple", "merged", "combined"],  # Three equations
    ],
)
def test_assemble_subsystem(setup, var_names, eq_names):
    # Test of functionality to assemble subsystems from an EquationManager.
    #
    # The test is based on assembly of a subsystem and comparing this to a truth
    # from assembly of the full system, and then explicitly dump rows and columns.

    eq_manager = setup.eq_manager

    # Convert variable names into variables
    if var_names is None:
        var_names = []
    variables = [var for var in setup.eq_manager.variables]
    for var in var_names:
        variables.remove(var)

    A_sub, b_sub = eq_manager.assemble_subsystem(equations=eq_names, variables=variables)

    # Get active rows and columns. If eq_names is None, all rows should be included.
    # If equation list is set to empty list, no indices are included.
    # Otherwise, get the numbering from setup.
    # Same logic for variables
    if eq_names is None:
        # If no equations are specified, all should be included.
        rows = np.arange(15)
        expected_blocks = np.array([0, 4, 11, 15])
    elif len(eq_names) > 0:
        # Do not sort row indices - these are allowed to change.
        rows = np.sort(np.hstack([setup.eq_ind(eq) for eq in eq_names]))
        expected_blocks = np.cumsum(
            np.hstack([0, [setup.block_size(eq) for eq in eq_names]])
        )
    else:
        # Equations are set to empty
        rows = []
        expected_blocks = np.array([0])

    if variables is None:
        # If no variables are specified, all should be included.
        cols = np.arange(18)
    elif len(variables) > 0:
        # Sort variable indices
        cols = np.sort(np.hstack([setup.dof_ind(var) for var in variables]))
    else:
        # Variables are set to empty
        cols = []

    # Uniquify columns, in case variables are represented many times.
    cols = np.unique(cols)

    # Check matrix and vector items
    assert np.allclose(b_sub, setup.b[rows])
    assert _compare_matrices(A_sub, setup.A[rows][:, cols])

    # Also check that the equation row sizes were correctly recorded.
    if eq_names is not None:
        for name in eq_names:
            assert np.allclose(setup.eq_manager.assembled_equation_indices[name].size, setup.block_size(name))
    else:
        for name in setup.eq_manager.equations:
            assert np.allclose(setup.eq_manager.assembled_equation_indices[name].size, setup.block_size(name))


@pytest.mark.parametrize(
    "var_names",
    [
        [],
        ["x1"],
        ["y_merged"],
        ["x1", "y_merged"],
        ["y_merged", "x1"],
        ["x_merged", "y_merged", "z1"],
    ],
)
@pytest.mark.parametrize(
    "eq_names",
    [
        [],
        ["simple"],
        ["simple", "merged"],
        ["merged", "combined"],
        ["simple", "merged", "combined"],
    ],
)
def test_extract_subsystem(setup, eq_names, var_names):
    # Check functionality to extract subsystems from the EquationManager.
    # The tests check that the expected variables and equations are present in
    # the subsystem.

    eq_manager = setup.eq_manager

    # Get Ad variables from strings
    variables = [getattr(setup, var) for var in var_names]

    # Secondary variables are constructed as atomic. The number of these is found by
    # the number of variables + the number of merged variables (known to be defined
    # on exactly two grids)
    num_atomic_vars = len(var_names) + sum(["merged" in var for var in var_names])
    # The number of secondary variables is also known
    num_secondary = 5 - num_atomic_vars

    new_manager = eq_manager.SubSystem(eq_names, variables)

    # Check that the number of variables and equations are as expected
    assert len(new_manager.secondary_variables) == num_secondary
    assert len(new_manager.equations) == len(eq_names)

    # Check that the active / primary equations are present in the new manager
    for eq in new_manager.equations:
        assert eq in eq_names

    # Check that the secondary variables are not among the active (primary) variables.
    for sec_var in new_manager.secondary_variables:

        # Check that the secondary variable has a representation in a form known to the
        # new EquationManager.
        assert sec_var in new_manager._variables_as_list()

        sec_name = sec_var._name
        for vi, active_name in enumerate(var_names):
            if sec_name[0] != active_name[0]:
                # Okay, this secondary variable has nothing in common with this active var
                continue
            else:
                # With the setup of the tests, this is acceptable only if the active
                # variable is x1, and the secondary variable has name 'x', is active on
                # grid sd_1.

                # Get the active variable
                active_var = variables[vi]
                assert active_var.domain == setup.sd_2
                assert active_var.name == "x"

                # Check the secondary variable
                assert sec_var._g == setup.sd_1
                assert sec_name == "x"


@pytest.mark.parametrize(
    "eq_var_to_exclude",
    # Combinations of variables and variables. These cannot be set independently, since
    # the block to be eliminated should be square and invertible.
    [
        [["simple"], ["z1"]],  # Single equation, atomic variable
        [["simple"], ["x1"]],  # Atomic version of variable which is also merged
        [["merged"], ["y_merged"]],  # Merged equation, variable is only merged
        [["merged"], ["x_merged"]],  # variable is both merged and atomic
        [["combined"], ["x1"]],  # Combined equation, atomic variable
        [["simple", "merged"], ["x1", "y_merged"]],  # Two equations, two variables
    ],
)
def test_schur_complement(setup, eq_var_to_exclude):
    # Test assembly by a Schur complement.
    # The test constructs the Schur complement 'manually', using known information on
    # the ordering of equations and unknowns, and compares with the method in EquationManager.

    # NOTE: Input parameters to this test are interpreted as equations and variables to be
    # eliminated (this makes interpretation of test results easier), while the method in
    # EquationManager expects equations and variables to be kept. Some work is needed to invert
    # the sets.

    eq_to_exclude, var_to_exclude = eq_var_to_exclude
    eq_manager = setup.eq_manager
    # Equations to keep
    # This construction preserves the order
    eq_name = [eq for eq in eq_manager.equations if eq not in eq_to_exclude]

    # Set of all variables. z and y are only given in one form as input to this test.
    all_variables = ["z1", "y_merged"]

    # x can be both merged and simple
    if "x_merged" in var_to_exclude:
        all_variables.append("x_merged")
    else:  # We can work with atomic representation of the x variable
        all_variables += ["x1", "x2"]

    # Names of variables to keep
    var_name = list(set(all_variables).difference(set(var_to_exclude)))

    # Convert from variable names to actual variables
    variables = [getattr(setup, name) for name in var_name]

    inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

    # Rows and columns to keep
    rows_keep = np.sort(np.hstack([setup.eq_ind(name) for name in eq_name]))
    cols_keep = np.sort(np.hstack([setup.dof_ind(var) for var in variables]))

    # Rows and columns to eliminate
    rows_elim = np.setdiff1d(np.arange(15), rows_keep)
    cols_elim = np.setdiff1d(np.arange(18), cols_keep)

    # Split the full matrix into [[A, B], [C, D]], where D is to be eliminated
    A = setup.A[rows_keep][:, cols_keep]
    B = setup.A[rows_keep][:, cols_elim]
    C = setup.A[rows_elim][:, cols_keep]
    D = setup.A[rows_elim][:, cols_elim]

    b_1 = setup.b[rows_elim]
    b_2 = setup.b[rows_keep]

    S_known = A - B * inverter(D) * C
    b_known = b_2 - B * inverter(D) * b_1

    # Compute Schur complement with method to be tested
    S, bS = eq_manager.assemble_schur_complement_system(eq_name, variables, inverter)

    assert np.allclose(bS, b_known)
    assert _compare_matrices(S, S_known)
    # Also check that the equation row sizes were correctly recorded.
    expected_blocks = np.cumsum(
        np.hstack([0, [setup.block_size(eq) for eq in eq_name]])
    )
    assert np.allclose(
        expected_blocks, setup.eq_manager.row_block_indices_last_assembled
    )
