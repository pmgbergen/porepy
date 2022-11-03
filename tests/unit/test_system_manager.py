"""Tests of the EquationManager used in the Ad framework.

The tests focus on various assembly methods:
    * test_secondary_variable_assembly: Test of assembly when secondary variables are
        present.
    * test_assemble_subsystem: Assemble blocks of the full set of equations.
    * test_extract_subsystem: Extract a new EquationManager for a subset of equations.
    * test_schur_complement: Assemble a subsystem, using a Schur complement reduction.

To be tested:
    Get and set methods for variables
    IdentifyDof


"""
import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


def test_variable_creation():
    """Test that variable creation from a SystemManager works as expected.

    The test generates a MixedDimensionalGrid, defines some variables on it, and checks
    that the variables have the correct sizes and names.

    Also tested is the methods num_dofs() and partly dofs_of()
    """
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal(simplex=False)

    sys_man = pp.ad.SystemManager(mdg)

    # Define the number of variables per grid item. Faces are included just to test that
    # this also works.
    dof_info_sd = {"cells": 1, "faces": 2}
    dof_info_intf = {"cells": 1}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    interfaces = mdg.interfaces()

    # Define one variable on all subdomains, one on a single subdomain, and one on
    # all interfaces (there is only one interface in this grid).
    subdomain_variable = sys_man.create_variable(
        "var_1", dof_info_sd, subdomains=subdomains
    )
    single_subdomain_variable = sys_man.create_variable(
        "var_2", dof_info_sd, single_subdomain
    )
    interface_variable = sys_man.create_variable(
        "var_3", dof_info_intf, interfaces=interfaces
    )

    # Also define a variable on subdomains without specifying the subdomains.
    # The resulting variable should have the same size as the one where all subdomains
    # are explicitly set.
    subdomain_variable_implicitly_defined = sys_man.create_variable(
        "var_4", dof_info_sd, subdomains=[]
    )

    # Check that the variables are created correctly
    assert subdomain_variable.name == "var_1"
    assert single_subdomain_variable.name == "var_2"
    assert interface_variable.name == "var_3"
    assert subdomain_variable_implicitly_defined.name == "var_4"

    # Check that the number of dofs is correct for each variable.
    num_subdomain_faces = sum([sd.num_faces for sd in subdomains])

    # Need a factor 2 here, since we have defined two variables on all subdomains.
    ndof_subdomains = 2 * (
        mdg.num_subdomain_cells() * dof_info_sd["cells"]
        + num_subdomain_faces * dof_info_sd["faces"]
    )
    ndof_single_subdomain = (
        single_subdomain[0].num_cells * dof_info_sd["cells"]
        + single_subdomain[0].num_faces * dof_info_sd["faces"]
    )
    ndof_interface = mdg.num_interface_cells() * dof_info_intf["cells"]

    assert (
        sys_man.dofs_of(subdomain_variable).size
        + sys_man.dofs_of(subdomain_variable_implicitly_defined).size
    ) == ndof_subdomains
    assert sys_man.dofs_of(single_subdomain_variable).size == ndof_single_subdomain
    assert sys_man.dofs_of(interface_variable).size == ndof_interface

    assert (
        sys_man.num_dofs() == ndof_subdomains + ndof_single_subdomain + ndof_interface
    )


def test_variable_tags():
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal(simplex=False)

    sys_man = pp.ad.SystemManager(mdg)

    # Define the number of variables per grid item. Faces are included just to test that
    # this also works.
    dof_info = {"cells": 1}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    # Define one variable on all subdomains, one on a single subdomain, and one on
    # all interfaces (there is only one interface in this grid).
    var_1 = sys_man.create_variable(
        "var_1", dof_info, subdomains=subdomains, tags={"tag1": 1}
    )
    var_2 = sys_man.create_variable("var_2", dof_info, single_subdomain)

    assert sys_man.variable_tags[var_1] == {"tag1": 1}
    # By default, variables should not have tags
    assert len(sys_man.variable_tags[var_2]) == 0

    # Add separate tags to separate variables
    sys_man.add_variable_tags([var_1, var_2], [{"tag_2": 2}, {"tag_3": 3}])
    assert sys_man.variable_tags[var_1]["tag_2"] == 2
    assert sys_man.variable_tags[var_2]["tag_3"] == 3

    # Add the same tag to both variables.
    # This will overwrite the previous tag for var_1, but not for var_2
    sys_man.add_variable_tags([var_1, var_2], [{"tag_2": 4}])
    assert sys_man.variable_tags[var_1]["tag_2"] == 4
    assert sys_man.variable_tags[var_2]["tag_2"] == 4

    # Add multiple tags to a single variable.
    # tag_3 will be overwritten, tag_4 is added as a boolean
    sys_man.add_variable_tags([var_2], [{"tag_3": 4}, {"tag_4": False}])
    assert sys_man.variable_tags[var_2]["tag_3"] == 4
    assert sys_man.variable_tags[var_2]["tag_4"] == False

    # Assign one tag to one variable
    sys_man.add_variable_tags([var_1], [{"tag_4": True}])
    assert sys_man.variable_tags[var_1]["tag_4"] == True


class SystemManagerSetup:
    """Class to set up a SystemManager with a combination of variables
    and equations, designed to make it convenient to test critical functionality
    of the SystemManager.

    The setup is intended for testing advanced functionality, like assembly of equations,
    construction of subsystems of equations etc. The below setup is in itself a test of
    basic functionality, like creation of variables and equations.
    TODO: We should have dedicated tests for variable creation, to make sure we cover
    all options.
    """

    def __init__(self):
        mdg, _ = pp.grids.standard_grids.md_grids_2d.two_intersecting()

        sys_man = pp.ad.SystemManager(mdg)

        # List of all subdomains
        subdomains = mdg.subdomains()
        # Also generate a variable on the top-dimensional domain
        sd_top = mdg.subdomains(dim=mdg.dim_max())[0]

        interfaces = mdg.interfaces()
        intf_top = mdg.interfaces(dim=mdg.dim_max() - 1)[0]

        self.name_sd_variable = "x"
        self.sd_variable = sys_man.create_variable(
            self.name_sd_variable, subdomains=subdomains
        )

        self.name_intf_variable = "y"
        self.intf_variable = sys_man.create_variable(
            self.name_intf_variable, interfaces=interfaces
        )

        self.name_sd_top_variable = "z"
        self.sd_top_variable = sys_man.create_variable(
            self.name_sd_top_variable, subdomains=[sd_top]
        )

        self.name_intf_top_variable = "w"
        self.intf_top_variable = sys_man.create_variable(
            self.name_intf_top_variable, interfaces=[intf_top]
        )

        # Set state and iterate for the variables.
        # The assigned numbers are not important, the comparisons below will be between
        # an assembled matrix for the full and for a reduced system, and similar for
        # the right hand side vector.
        global_vals = np.arange(sys_man.num_dofs())
        global_vals[sys_man.dofs_of([self.sd_variable])] = np.arange(
            mdg.num_subdomain_cells()
        )
        global_vals[sys_man.dofs_of([self.sd_top_variable])] = np.arange(
            sd_top.num_cells
        )
        global_vals[sys_man.dofs_of([self.intf_variable])] = np.arange(
            mdg.num_interface_cells()
        )
        global_vals[sys_man.dofs_of([self.intf_top_variable])] = np.arange(
            intf_top.num_cells
        )

        sys_man.set_variable_values(
            global_vals,
            variables=[
                self.name_sd_variable,
                self.name_sd_top_variable,
                self.name_intf_variable,
                self.name_intf_top_variable,
            ],
        )

        # Set equations on subdomains

        projections = pp.ad.SubdomainProjections(subdomains=subdomains)
        proj = projections.cell_restriction([sd_top])

        # One equation with only simple variables (not merged)
        self.eq_all_subdomains = self.sd_variable * self.sd_variable
        self.eq_all_subdomains.set_name("eq_all_subdomains")
        # One equation using only merged variables
        self.eq_single_subdomain = self.sd_top_variable * self.sd_top_variable
        self.eq_single_subdomain.set_name("eq_single_subdomain")
        # One equation combining merged and simple variables
        self.eq_combined = self.sd_top_variable * (proj * self.sd_variable)
        self.eq_combined.set_name("eq_combined")

        dof_all_subdomains = {sd: {"cells": 1} for sd in subdomains}
        dof_single_subdomain = {sd_top: {"cells": 1}}
        dof_combined = {sd_top: {"cells": 1}}

        sys_man.set_equation(self.eq_all_subdomains, dof_all_subdomains)
        sys_man.set_equation(self.eq_single_subdomain, dof_single_subdomain)
        sys_man.set_equation(self.eq_combined, dof_combined)

        # Define equations on the interfaces
        # Common for all interfaces
        self.eq_all_interfaces = self.intf_variable * self.intf_variable
        self.eq_all_interfaces.set_name("eq_all_interfaces")
        # The top interface only
        self.eq_single_interface = self.intf_top_variable * self.intf_top_variable
        self.eq_single_interface.set_name("eq_single_interface")
        # TODO: Should we do something on a combination as well?

        self.A, self.b = sys_man.assemble()
        self.sys_man = sys_man

        # Store subdomains and interfaces
        self.subdomains = subdomains
        self.sd_top = sd_top
        self.interfaces = interfaces
        self.intf_top = intf_top
        self.mdg = mdg

    ## Helper methods below.

    def var_size(self, var):
        # For a given variable, get its size (number of dofs) based on what
        # we know about the variables specified in self.__init__
        if var == self.sd_variable:
            return self.mdg.num_subdomain_cells()
        elif var == self.sd_top_variable:
            return self.sd_top.num_cells
        elif var == self.intf_variable:
            return self.mdg.num_interface_cells()
        elif var == self.intf_top_variable:
            return self.intf_top.num_cells
        else:
            raise ValueError

    def dof_ind(self, var):
        # For a given variable, get the global indices assigned by
        # the DofManager. Based on knowledge of how the variables were
        # defined in self.__init__

        def inds(g, n):
            return self.eq_manager.dofs_of(g, n)

        return self.eq_manager.dofs_of(var)
        if var in (self.x1, self.x2, self.z):
            return self.eq_manager.dofs_of(var)  # inds(self.x1.domain, self.x1.name)

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

        #        elif var == self.z:
        #            return inds(self.z.domain, self.z.name)
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


def _variable_from_setup(
    setup,
    as_str: bool,
    on_interface: bool,
    on_subdomain: bool,
    single_grid: bool,
    full_grid: bool,
):
    # Helper method to get a variable from the setup, either as a string or as
    # a variable object. The variable is either on a subdomain or an interface.
    vars = []
    if on_interface:
        if as_str:
            if single_grid:
                vars.append("intf_top_variable")
            if full_grid:
                vars.append("intf_variable")
        else:
            if single_grid:
                vars.append(setup.intf_top_variable)
            if full_grid:
                vars.append(setup.intf_variable)
    if on_subdomain:
        if as_str:
            if single_grid:
                vars.append("sd_top_variable")
            if full_grid:
                vars.append("sd_variable")
        else:
            if single_grid:
                vars.append(setup.sd_top_variable)
            if full_grid:
                vars.append(setup.sd_variable)
    return vars


@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("on_interface", [True, False])
@pytest.mark.parametrize("on_subdomain", [True, False])
@pytest.mark.parametrize("single_grid", [True, False])
@pytest.mark.parametrize("full_grid", [True, False])
@pytest.mark.parametrize("iterate", [True, False])
def test_set_get_methods(
    setup, as_str, on_interface, on_subdomain, single_grid, full_grid, iterate
):
    """Test the set and get methods of the SystemManager class.

    The test is performed for a number of different combinations of variables.

    Values are assigned to the STATE or ITERATE storage, and then retrieved.

    Both setting and adding values are tested.

    """
    eq_manager = setup.eq_manager

    np.random.seed(42)

    variables = _variable_from_setup(
        setup, as_str, on_interface, on_subdomain, single_grid, full_grid
    )

    sz = eq_manager.dofs_of(variables)

    # First generate random values, set them, and then retrieve them.
    vals = np.random.rand(sz.size)
    eq_manager.set_variable_values(vals, variables, to_iterate=iterate)
    retrieved_vals = eq_manager.get_variable_values(variables, from_iterate=iterate)
    assert np.allclose(vals, retrieved_vals)

    # Set new values without setting additive to True. This should overwrite the old
    # values.
    new_vals = np.random.rand(sz.size)
    eq_manager.set_variable_values(new_vals, variables, to_iterate=iterate)
    retrieved_vals2 = eq_manager.get_variable_values(variables, from_iterate=iterate)
    assert np.allclose(new_vals, retrieved_vals2)

    # Set the values again, this time with additive=True. This should double the
    # retrieved values.
    eq_manager.set_variable_values(
        new_vals, variables, to_iterate=iterate, additive=True
    )
    retrieved_vals3 = eq_manager.get_variable_values(variables, from_iterate=iterate)

    assert np.allclose(2 * new_vals, retrieved_vals3)


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
def test_projection_matrix(setup, var_names):
    # Test of the projection matrix method. The only interesting test is the
    # secondary variable functionality (the other functionality is tested elsewhere).

    # The tests compare assembly by an EquationManager with explicitly defined
    # secondary variables with a 'truth' based on direct elimination of columns
    # in the Jacobian matrix.
    # The expected behavior is that the residual vector is fixed, while the
    # Jacobian matrix is altered only in columns that correspond to eliminated
    # variables.

    variables = [var for var in setup.eq_manager.variables]
    for var in var_names:
        variables.remove(var)

    P = setup.eq_manager.projection_to(variables=variables)

    # Get dof indices of the variables that have been eliminated
    if len(var_names) > 0:
        removed_dofs = np.sort(
            np.hstack([setup.eq_manager.dofs_of(var) for var in var_names])
        )
    else:
        removed_dofs = []

    remaining_dofs = np.setdiff1d(np.arange(setup.eq_manager.num_dofs()), removed_dofs)

    assert np.allclose(P.indices, remaining_dofs)


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
        assert np.allclose(
            setup.eq_manager.assembled_equation_indices[name], setup.eq_ind(name)
        )


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

    A_sub, b_sub = eq_manager.assemble_subsystem(
        equations=eq_names, variables=variables
    )

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
            assert np.allclose(
                setup.eq_manager.assembled_equation_indices[name].size,
                setup.block_size(name),
            )
    else:
        for name in setup.eq_manager.equations:
            assert np.allclose(
                setup.eq_manager.assembled_equation_indices[name].size,
                setup.block_size(name),
            )


@pytest.mark.parametrize(
    "var_names",
    [
        [],
        ["x1"],
        ["y_merged"],
        ["x1", "y_merged"],
        ["y_merged", "x1"],
        ["x_merged", "y_merged", "z"],
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
        [["simple"], ["z"]],  # Single equation, atomic variable
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
    all_variables = ["z", "y_merged"]

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
