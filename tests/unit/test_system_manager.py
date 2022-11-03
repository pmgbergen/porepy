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
        sys_man.dofs_of([subdomain_variable]).size
        + sys_man.dofs_of([subdomain_variable_implicitly_defined]).size
    ) == ndof_subdomains
    assert sys_man.dofs_of([single_subdomain_variable]).size == ndof_single_subdomain
    assert sys_man.dofs_of([interface_variable]).size == ndof_interface

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

    assert sys_man.variable_tags["var_1"] == {"tag1": 1}
    # By default, variables should not have tags
    assert len(sys_man.variable_tags["var_2"]) == 0

    # Add separate tags to separate variables
    sys_man.add_variable_tags([var_1, var_2], [{"tag_2": 2}, {"tag_3": 3}])
    assert sys_man.variable_tags["var_1"]["tag_2"] == 2
    assert sys_man.variable_tags["var_2"]["tag_3"] == 3

    # Add the same tag to both variables.
    # This will overwrite the previous tag for var_1, but not for var_2
    sys_man.add_variable_tags(["var_1", "var_2"], [{"tag_2": 4}])
    assert sys_man.variable_tags["var_1"]["tag_2"] == 4
    assert sys_man.variable_tags["var_2"]["tag_2"] == 4

    # Add multiple tags to a single variable.
    # tag_3 will be overwritten, tag_4 is added as a boolean
    sys_man.add_variable_tags(["var_2"], [{"tag_3": 4}, {"tag_4": False}])
    assert sys_man.variable_tags["var_2"]["tag_3"] == 4
    assert sys_man.variable_tags["var_2"]["tag_4"] == False

    # Assign one tag to one variable
    sys_man.add_variable_tags(["var_1"], [{"tag_4": True}])
    assert sys_man.variable_tags["var_1"]["tag_4"] == True

    # Also let var_1 have tag_3, this is useful for testing of the get_variables_by_tag
    # method below.
    sys_man.add_variable_tags(["var_1"], [{"tag_3": 3}])

    # Next stage: Fetch variables based on tags
    # First, requets a tag which both variables have
    retrieved_vars_1 = sys_man.get_variables_by_tag("tag_4")
    assert var_1 in retrieved_vars_1 and var_2 in retrieved_vars_1

    # Request tag_4 again, but with a value of False. This should return only var_2
    retrieved_vars_2 = sys_man.get_variables_by_tag("tag_4", False)
    assert var_1 not in retrieved_vars_2 and var_2 in retrieved_vars_2

    # Request tag_1, which only var_1 has
    retrieved_vars_3 = sys_man.get_variables_by_tag("tag1")
    assert var_1 in retrieved_vars_3 and var_2 not in retrieved_vars_3

    # Request tag_1 again, but with a value different from that of var_1. This should
    # return an empty list.
    retrieved_vars_4 = sys_man.get_variables_by_tag("tag1", 2)
    assert len(retrieved_vars_4) == 0

    # Request tag_3, which both variables have, but with a value of 4. This should
    # return only var_2
    retrieved_vars_5 = sys_man.get_variables_by_tag("tag_3", 4)
    assert var_1 not in retrieved_vars_5 and var_2 in retrieved_vars_5


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

    def __init__(self, square_system=False):
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
        # Add one to avoid zero values, which yields singular matrices
        global_vals += 1
        sys_man.set_variable_values(
            global_vals,
            variables=[
                self.name_sd_variable,
                self.name_sd_top_variable,
                self.name_intf_variable,
                self.name_intf_top_variable,
            ],
        )
        sys_man.set_variable_values(
            global_vals,
            variables=[
                self.name_sd_variable,
                self.name_sd_top_variable,
                self.name_intf_variable,
                self.name_intf_top_variable,
            ],
            to_iterate=True,
        )

        # Set equations on subdomains

        projections = pp.ad.SubdomainProjections(subdomains=subdomains)
        proj = projections.cell_restriction([sd_top])

        # One equation for all subdomains
        self.eq_all_subdomains = self.sd_variable * self.sd_variable
        self.eq_all_subdomains.set_name("eq_all_subdomains")
        # One equation using only top subdomain variable
        self.eq_single_subdomain = self.sd_top_variable * self.sd_top_variable
        self.eq_single_subdomain.set_name("eq_single_subdomain")

        dof_all_subdomains = {sd: {"cells": 1} for sd in subdomains}
        dof_single_subdomain = {sd_top: {"cells": 1}}
        dof_combined = {sd_top: {"cells": 1}}

        sys_man.set_equation(self.eq_all_subdomains, dof_all_subdomains)
        sys_man.set_equation(self.eq_single_subdomain, dof_single_subdomain)

        # Define equations on the interfaces
        # Common for all interfaces
        self.eq_all_interfaces = self.intf_variable * self.intf_variable
        self.eq_all_interfaces.set_name("eq_all_interfaces")
        # The top interface only
        self.eq_single_interface = self.intf_top_variable * self.intf_top_variable
        self.eq_single_interface.set_name("eq_single_interface")

        # TODO: Should we do something on a combination as well?
        dof_all_interfaces = {intf: {"cells": 1} for intf in interfaces}
        dof_single_interface = {intf_top: {"cells": 1}}
        sys_man.set_equation(self.eq_all_interfaces, dof_all_interfaces)
        sys_man.set_equation(self.eq_single_interface, dof_single_interface)
        self.eq_inds = np.array(
            [
                mdg.num_subdomain_cells(),
                mdg.subdomains()[0].num_cells,
                mdg.num_interface_cells(),
                mdg.interfaces()[0].num_cells,
            ]
        )
        if not square_system:
            # One equation combining top and all subdomains.
            # Assigned last to avoid mess if omitted
            self.eq_combined = self.sd_top_variable * (proj * self.sd_variable)
            self.eq_combined.set_name("eq_combined")
            sys_man.set_equation(self.eq_combined, dof_combined)
            self.eq_inds = np.append(self.eq_inds, mdg.subdomains()[0].num_cells)

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

        return self.sys_man.dofs_of([var])

    def eq_ind(self, name):
        # Get row indices of an equation, based on the (known) order in which
        # equations were added to the SystemManager
        inds = self.eq_inds
        if name == "eq_all_subdomains":
            return np.arange(inds[0])
        elif name == "eq_single_subdomain":
            return inds[0] + np.arange(inds[1])
        elif name == "eq_all_interfaces":
            return sum(inds[:2]) + np.arange(inds[2])
        elif name == "eq_single_interface":
            return sum(inds[:3]) + np.arange(inds[3])
        elif name == "eq_combined":
            return sum(inds[:4]) + np.arange(inds[4])
        else:
            raise ValueError

    def block_size(self, name):
        # Get the size of an equation block
        return self.eq_ind(name).size


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
                vars.append("w")  # intf_top_variable
            if full_grid:
                vars.append("y")  # intf_variable
        else:
            if single_grid:
                vars.append(setup.intf_top_variable)
            if full_grid:
                vars.append(setup.intf_variable)
    if on_subdomain:
        if as_str:
            if single_grid:
                vars.append("y")  # sd_top_variable
            if full_grid:
                vars.append("x")
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
    sys_man = setup.sys_man

    np.random.seed(42)

    variables = _variable_from_setup(
        setup, as_str, on_interface, on_subdomain, single_grid, full_grid
    )

    inds = sys_man.dofs_of(variables)

    # First generate random values, set them, and then retrieve them.
    vals = np.zeros(sys_man.num_dofs())
    vals = np.random.rand(inds.size)

    sys_man.set_variable_values(vals, variables, to_iterate=iterate)
    retrieved_vals = sys_man.get_variable_values(variables, from_iterate=iterate)
    assert np.allclose(vals, retrieved_vals)

    # Set new values without setting additive to True. This should overwrite the old
    # values.
    new_vals = np.zeros(sys_man.num_dofs())
    new_vals = np.random.rand(inds.size)
    sys_man.set_variable_values(new_vals, variables, to_iterate=iterate)
    retrieved_vals2 = sys_man.get_variable_values(variables, from_iterate=iterate)
    assert np.allclose(new_vals, retrieved_vals2)

    # Set the values again, this time with additive=True. This should double the
    # retrieved values.
    sys_man.set_variable_values(new_vals, variables, to_iterate=iterate, additive=True)
    retrieved_vals3 = sys_man.get_variable_values(variables, from_iterate=iterate)

    assert np.allclose(2 * new_vals, retrieved_vals3)


@pytest.mark.parametrize(
    "var_names",
    [
        [],
        ["x"],
        ["y"],
        ["w"],
        ["z", "x"],
    ],
)
def test_projection_matrix(setup, var_names):
    # Test of the projection matrix method. The only interesting test is the
    # secondary variable functionality (the other functionality is tested elsewhere).

    # The tests compare assembly by a SystemManager with explicitly defined
    # secondary variables with a 'truth' based on direct elimination of columns
    # in the Jacobian matrix.
    # The expected behavior is that the residual vector is fixed, while the
    # Jacobian matrix is altered only in columns that correspond to eliminated
    # variables.

    variables = [var for var in setup.sys_man.variables]
    for var in var_names:
        variables.remove(var)

    proj = setup.sys_man.projection_to(variables=variables)

    # Get dof indices of the variables that have been eliminated
    if len(var_names) > 0:
        removed_dofs = np.sort(
            np.hstack([setup.sys_man.dofs_of([var]) for var in var_names])
        )
    else:
        removed_dofs = []

    remaining_dofs = np.setdiff1d(np.arange(setup.sys_man.num_dofs()), removed_dofs)

    assert np.allclose(proj.indices, remaining_dofs)


@pytest.mark.parametrize(
    "var_names",
    [
        [],  # No secondary variables
        ["x"],  # A simple variable which is also part of a merged one
        ["y"],  # Merged variable
        ["z"],  # Simple variable not available as merged
        ["z", "w"],  # Combination of simple and merged.
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
    variables = [var for var in setup.sys_man.variables]
    for var in var_names:
        variables.remove(var)
    A, b = setup.sys_man.assemble_subsystem(variables=variables)

    # Get dof indices of the variables that have been eliminated
    if len(var_names) > 0:
        dofs = np.sort(np.hstack([setup.sys_man.dofs_of([var]) for var in var_names]))
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
    for name in setup.sys_man.equations:
        assert np.allclose(
            setup.sys_man.assembled_equation_indices[name], setup.eq_ind(name)
        )


@pytest.mark.parametrize(
    "var_names",
    [
        None,  # Should give all variables by default
        [],  # Empty list (as opposed to None) should give a system with zero columns
        ["x"],  # md variable
        ["y"],  # single variable
        ["x", "y"],  # Combination of single and md
        ["y", "x"],  # Combination reversed - check just to be sure
        ["x", "w"],  # Subdomain and interface
    ],
)
@pytest.mark.parametrize(
    "eq_names",
    [
        None,  # should give all equations by default
        [],  # Empty list should give a system with zero rows
        ["eq_single_subdomain"],  # Single equation
        ["eq_single_subdomain", "eq_all_subdomains"],  # Combination of two equations
        ["eq_combined", "eq_single_subdomain"],  # Different combination
        ["eq_single_interface", "eq_all_interfaces"],  # Interface equations
        ["eq_single_interface", "eq_all_interfaces", "eq_single_subdomain"],  # Mixed
        [
            "eq_single_interface",
            "eq_all_interfaces",
            "eq_single_subdomain",
            "eq_all_subdomains",
            "eq_combined",
        ],  # All
    ],
)
def test_assemble_subsystem(setup, var_names, eq_names):
    # Test of functionality to assemble subsystems from an EquationManager.
    #
    # The test is based on assembly of a subsystem and comparing this to a truth
    # from assembly of the full system, and then explicitly dump rows and columns.

    sys_man = setup.sys_man

    # Convert variable names into variables
    if var_names is None:
        var_names = []
    variables = [var for var in setup.sys_man.variables]
    for var in var_names:
        variables.remove(var)

    A_sub, b_sub = sys_man.assemble_subsystem(equations=eq_names, variables=variables)

    # Get active rows and columns. If eq_names is None, all rows should be included.
    # If equation list is set to empty list, no indices are included.
    # Otherwise, get the numbering from setup.
    # Same logic for variables
    if eq_names is None:
        # If no equations are specified, all should be included.
        rows = np.arange(sum(setup.eq_inds))
    elif len(eq_names) > 0:
        # Do not sort row indices - these are allowed to change.
        rows = np.sort(np.hstack([setup.eq_ind(eq) for eq in eq_names]))
    else:
        # Equations are set to empty
        rows = []

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
                setup.sys_man.assembled_equation_indices[name].size,
                setup.block_size(name),
            )
    else:
        for name in setup.sys_man.equations:
            assert np.allclose(
                setup.sys_man.assembled_equation_indices[name].size,
                setup.block_size(name),
            )


@pytest.mark.parametrize(
    "var_names",
    [
        None,  # Should give all variables by default
        [],  # Empty list (as opposed to None) should give a system with zero columns
        ["x"],  # md variable
        ["y"],  # single variable
        ["x", "y"],  # Combination of single and md
        ["y", "x"],  # Combination reversed - check just to be sure
        ["x", "w"],  # Subdomain and interface
    ],
)
@pytest.mark.parametrize(
    "eq_names",
    [
        None,  # should give all equations by default
        [],  # Empty list should give a system with zero rows
        ["eq_single_subdomain"],  # Single equation
        ["eq_single_subdomain", "eq_all_subdomains"],  # Combination of two equations
        ["eq_combined", "eq_single_subdomain"],  # Different combination
        ["eq_single_interface", "eq_all_interfaces"],  # Interface equations
        ["eq_single_interface", "eq_all_interfaces", "eq_single_subdomain"],  # Mixed
        [
            "eq_single_interface",
            "eq_all_interfaces",
            "eq_single_subdomain",
            "eq_all_subdomains",
            "eq_combined",
        ],  # All
    ],
)
def extract_subsystem(setup, eq_names, var_names):
    # Check functionality to extract subsystems from the EquationManager.
    # The tests check that the expected variables and equations are present in
    # the subsystem.

    sys_man = setup.sys_man

    # Get Ad variables from strings
    variables = [getattr(setup, var) for var in var_names]

    # Secondary variables are constructed as atomic. The number of these is found by
    # the number of variables + the number of merged variables (known to be defined
    # on exactly two grids)
    num_atomic_vars = len(var_names) + sum(["merged" in var for var in var_names])
    # The number of secondary variables is also known
    num_secondary = 5 - num_atomic_vars

    new_manager = sys_man.SubSystem(eq_names, variables)

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
        [["eq_single_interface"], ["w"]],  # Single equation, atomic variable
        [["eq_all_subdomains"], ["x"]],  # MD variable
        [["eq_single_subdomain"], ["z"]],  # variable is both merged and atomic
        [
            ["eq_all_interfaces", "eq_single_subdomain"],
            ["z", "y"],
        ],  # Two equations, two variables
    ],
)
def test_schur_complement(eq_var_to_exclude):
    # Test assembly by a Schur complement.
    # The test constructs the Schur complement 'manually', using known information on
    # the ordering of equations and unknowns, and compares with the method in EquationManager.

    # NOTE: Input parameters to this test are interpreted as equations and variables to be
    # eliminated (this makes interpretation of test results easier), while the method in
    # EquationManager expects equations and variables to be kept. Some work is needed to invert
    # the sets.

    # Ensure the system is square by leaving out eq_combined
    setup = SystemManagerSetup(square_system=True)
    eq_to_exclude, var_to_exclude = eq_var_to_exclude
    sys_man = setup.sys_man
    # Equations to keep
    # This construction preserves the order
    eq_name = [eq for eq in sys_man.equations if eq not in eq_to_exclude]

    # Set of all variables. z and y are only given in one form as input to this test.
    all_variables = ["x", "w", "z", "y"]

    # Names of variables to keep
    var_name = list(set(all_variables).difference(set(var_to_exclude)))

    # Convert from variable names to actual variables
    # variables = [var for var in sys_man.]
    variables = var_name

    inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

    # Rows and columns to keep
    rows_keep = np.sort(np.hstack([setup.eq_ind(name) for name in eq_name]))
    cols_keep = np.sort(np.hstack([setup.dof_ind(var) for var in variables]))

    # Rows and columns to eliminate
    rows_elim = np.setdiff1d(np.arange(sum(setup.eq_inds)), rows_keep)
    cols_elim = np.setdiff1d(np.arange(setup.sys_man.num_dofs()), cols_keep)

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
    S, bS = sys_man.assemble_schur_complement_system(
        eq_name, variables, inverter=inverter
    )

    assert np.allclose(bS, b_known)
    assert _compare_matrices(S, S_known)
    # Also check that the equation row sizes were correctly recorded.
    expected_blocks = np.cumsum(
        np.hstack([0, [setup.block_size(eq) for eq in eq_name]])
    )
    # Fixme: I don't think we have this attribute anymore. Purge or update
    # assert np.allclose(
    #     expected_blocks, setup.sys_man.row_block_indices_last_assembled
    # )
