"""Tests of the EquationSystem used in the Ad machinery.

The tests focus on various assembly methods:
    * test_secondary_variable_assembly: Test of assembly when secondary variables are
        present.
    * test_assemble_subsystem: Assemble blocks of the full set of equations.
    * test_extract_subsystem: Extract a new EquationSystem for a subset of equations.
    * test_schur_complement: Assemble a subsystem, using a Schur complement reduction.

To be tested:
    Get and set methods for variables
    IdentifyDof


"""
import random
import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


def test_variable_creation():
    """Test that variable creation from a EquationSystem works as expected.

    The test generates a MixedDimensionalGrid, defines some variables on it, and checks
    that the variables have the correct sizes and names.

    Also tested is the methods num_dofs() and partly dofs_of()
    """
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal(simplex=False)

    sys_man = pp.ad.EquationSystem(mdg)

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
    subdomain_variable = sys_man.create_variables(
        "var_1", dof_info_sd, subdomains=subdomains
    )
    single_subdomain_variable = sys_man.create_variables(
        "var_2", dof_info_sd, subdomains=single_subdomain
    )
    interface_variable = sys_man.create_variables(
        "var_3", dof_info_intf, interfaces=interfaces
    )

    # Check that the variable storage in the EquationSystem is correct.
    # subdomain_variable is defined on two subdomains, the other on one.
    assert len(sys_man.variables) == 4

    # Tests of the md_variable method
    # Fetch a version of var_1. It should have the same sub-variables as the md-variable
    # returned from variable creation.
    subdomain_variable_fetched = sys_man.md_variable("var_1")
    assert len(subdomain_variable.sub_vars) == len(subdomain_variable_fetched.sub_vars)
    for sub_var in subdomain_variable_fetched.sub_vars:
        assert sub_var in subdomain_variable.sub_vars

    # Next, fetch a version of var_1, restricted to a single subdomain. This should
    # in practice be equivalent to var_2.
    single_subdomain_variable_fetched = sys_man.md_variable(
        "var_1", grids=single_subdomain
    )
    assert len(single_subdomain_variable_fetched.sub_vars) == 1
    assert (
        single_subdomain_variable_fetched.sub_vars[0].name == "var_1"
        and single_subdomain_variable_fetched.domain[0] == single_subdomain[0]
    )

    # Check that the variables are created correctly
    assert subdomain_variable.name == "var_1"
    assert single_subdomain_variable.name == "var_2"
    assert interface_variable.name == "var_3"

    # Check that the number of dofs is correct for each variable.
    num_subdomain_faces = sum([sd.num_faces for sd in subdomains])

    # Need a factor 2 here, since we have defined two variables on all subdomains.
    ndof_subdomains = (
        mdg.num_subdomain_cells() * dof_info_sd["cells"]
        + num_subdomain_faces * dof_info_sd["faces"]
    )
    ndof_single_subdomain = (
        single_subdomain[0].num_cells * dof_info_sd["cells"]
        + single_subdomain[0].num_faces * dof_info_sd["faces"]
    )
    ndof_interface = mdg.num_interface_cells() * dof_info_intf["cells"]

    assert (sys_man.dofs_of(subdomain_variable.sub_vars).size) == ndof_subdomains
    assert (
        sys_man.dofs_of(single_subdomain_variable.sub_vars).size
        == ndof_single_subdomain
    )
    assert sys_man.dofs_of(interface_variable.sub_vars).size == ndof_interface

    assert (
        sys_man.num_dofs() == ndof_subdomains + ndof_single_subdomain + ndof_interface
    )


def test_variable_tags():
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal(simplex=False)

    sys_man = pp.ad.EquationSystem(mdg)

    # Define the number of variables per grid item. Faces are included just to test that
    # this also works.
    dof_info = {"cells": 1}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    # Define one variable on all subdomains, one on a single subdomain, and one on
    # all interfaces (there is only one interface in this grid).
    var_1 = sys_man.create_variables(
        "var_1", dof_info, subdomains=subdomains, tags={"tag_1": 1}
    )
    var_2 = sys_man.create_variables("var_2", dof_info, subdomains=single_subdomain)

    assert var_1.tags == {"tag_1": 1}
    # By default, variables should not have tags
    assert len(var_2.tags) == 0

    # Add a tag to var_1. This will modify the underlying atomic variables, but not
    # var_1 itself.
    sys_man.update_variable_tags({"tag_2": 2}, [var_1])
    assert all(
        [var.tags["tag_2"] == 2 for var in sys_man.variables if var.name == "var_1"]
    )

    assert "tag_2" not in var_1.tags
    # However, if we fetch a new md-variable representing var_1, it should have the tag.
    var_1_new = sys_man.md_variable("var_1")
    assert "tag_2" in var_1_new.tags and var_1_new.tags["tag_2"] == 2

    # We can also add a tag to var_1, this will not be seen by the underlying atomic
    # variables.
    var_1.tags["tag_3"] = 3
    # Strictly speaking, this will also test the variables that are not part of var_1,
    # that should be fine, they should not have the tag.
    assert all(["tag_3" not in var.tags for var in sys_man.variables])

    # Add tags to var_2. This will be useful when we test filtering of variables below.
    sys_man.update_variable_tags({"tag_2": 4}, [var_2])
    sys_man.update_variable_tags({"tag_3": False}, [var_1])
    sys_man.update_variable_tags({"tag_3": True}, [var_2])

    ## Test of get_variables
    # First no filtering. This should give all variables.
    retrieved_var_1 = sys_man.get_variables()
    assert len(retrieved_var_1) == 3
    # Also uniquify, this should not change the length
    assert len(retrieved_var_1) == len(set(retrieved_var_1))

    # Filter on variable name var_1. Here we send in a list of atomic variables and
    # should recieve the same list.
    retrieved_var_2 = sys_man.get_variables(variables=var_1.sub_vars)
    assert len(retrieved_var_2) == 2
    assert all([var in retrieved_var_2 for var in var_1.sub_vars])

    # Filter on grids.
    retrieved_var_3 = sys_man.get_variables(grids=single_subdomain)
    assert len(retrieved_var_3) == 2
    assert all([var.domain == single_subdomain[0] for var in retrieved_var_3])

    # Filter on combination of grid and variable
    retrieved_var_4 = sys_man.get_variables(
        grids=single_subdomain, variables=var_1.sub_vars
    )
    assert len(retrieved_var_4) == 1

    # Filter on a non-existing tags
    retrieved_var_5 = sys_man.get_variables(tag_name="tag_4")
    assert len(retrieved_var_5) == 0
    # Filter on a tag that exists, but with a non-exiting value
    retrieved_var_6 = sys_man.get_variables(tag_name="tag_2", tag_value=5)
    assert len(retrieved_var_6) == 0

    # Filter on the name tag_1. This should give only var_1
    retrieved_var_7 = sys_man.get_variables(tag_name="tag_1")
    assert len(retrieved_var_7) == 2
    assert all([var in retrieved_var_7 for var in var_1.sub_vars])

    # Filter on the name tag_2. This should give var_1 and var_2
    retrieved_var_8 = sys_man.get_variables(tag_name="tag_2")
    assert len(retrieved_var_8) == 3

    # Filter on tag_2, with value 2. This should give only var_1
    retrieved_var_9 = sys_man.get_variables(tag_name="tag_2", tag_value=2)
    assert len(retrieved_var_9) == 2
    assert all([var in retrieved_var_9 for var in var_1.sub_vars])

    # Filter on tag_3, which takes boolean values. This should give only var_2
    retrieved_var_10 = sys_man.get_variables(tag_name="tag_3", tag_value=True)
    assert len(retrieved_var_10) == 1
    assert retrieved_var_10[0].domain == single_subdomain[0]


class EquationSystemSetup:
    """Class to set up a EquationSystem with a combination of variables
    and equations, designed to make it convenient to test critical functionality
    of the EquationSystem.

    The setup is intended for testing advanced functionality, like assembly of equations,
    construction of subsystems of equations etc. The below setup is in itself a test of
    basic functionality, like creation of variables and equations.
    TODO: We should have dedicated tests for variable creation, to make sure we cover
    all options.
    """

    def __init__(self, square_system=False):
        mdg, _ = pp.grids.standard_grids.md_grids_2d.two_intersecting()

        sys_man = pp.ad.EquationSystem(mdg)

        # List of all subdomains
        subdomains = mdg.subdomains()
        # Also generate a variable on the top-dimensional domain
        sd_top = mdg.subdomains(dim=mdg.dim_max())[0]

        interfaces = mdg.interfaces()
        intf_top = mdg.interfaces(dim=mdg.dim_max() - 1)[0]

        self.name_sd_variable = "x"
        self.sd_variable = sys_man.create_variables(
            self.name_sd_variable, subdomains=subdomains
        )

        # Let interface variables have size 2, this gives us a bit more to play with
        # in the testing of assembly.
        self.name_intf_variable = "y"
        self.intf_variable = sys_man.create_variables(
            self.name_intf_variable, dof_info={"cells": 2}, interfaces=interfaces
        )

        self.name_sd_top_variable = "z"
        self.sd_top_variable = sys_man.create_variables(
            self.name_sd_top_variable, subdomains=[sd_top]
        )

        self.name_intf_top_variable = "w"
        self.intf_top_variable = sys_man.create_variables(
            self.name_intf_top_variable, dof_info={"cells": 2}, interfaces=[intf_top]
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
            mdg.num_interface_cells() * 2
        )
        global_vals[sys_man.dofs_of([self.intf_top_variable])] = np.arange(
            intf_top.num_cells * 2
        )
        # Add one to avoid zero values, which yields singular matrices
        global_vals += 1
        all_variables = [
            self.name_sd_variable,
            self.name_sd_top_variable,
            self.name_intf_variable,
            self.name_intf_top_variable,
        ]
        sys_man.set_variable_values(
            global_vals, variables=all_variables, to_iterate=True, to_state=True
        )
        self.initial_values = global_vals

        # Set equations on subdomains

        projections = pp.ad.SubdomainProjections(subdomains=subdomains)
        proj = projections.cell_restriction([sd_top])

        # One equation for all subdomains
        self.eq_all_subdomains = self.sd_variable * self.sd_variable
        self.eq_all_subdomains.set_name("eq_all_subdomains")
        # One equation using only top subdomain variable
        self.eq_single_subdomain = self.sd_top_variable * self.sd_top_variable
        self.eq_single_subdomain.set_name("eq_single_subdomain")

        dof_all_subdomains = {"cells": 1}
        dof_single_subdomain = {"cells": 1}
        dof_combined = {"cells": 1}

        sys_man.set_equation(
            self.eq_all_subdomains,
            grids=subdomains,
            equations_per_grid_entity=dof_all_subdomains,
        )
        sys_man.set_equation(
            self.eq_single_subdomain,
            grids=[sd_top],
            equations_per_grid_entity=dof_single_subdomain,
        )

        # Define equations on the interfaces
        # Common for all interfaces
        self.eq_all_interfaces = self.intf_variable * self.intf_variable
        self.eq_all_interfaces.set_name("eq_all_interfaces")
        # The top interface only
        self.eq_single_interface = self.intf_top_variable * self.intf_top_variable
        self.eq_single_interface.set_name("eq_single_interface")

        # TODO: Should we do something on a combination as well?
        dof_all_interfaces = {"cells": 2}
        dof_single_interface = {"cells": 2}
        sys_man.set_equation(
            self.eq_all_interfaces,
            grids=interfaces,
            equations_per_grid_entity=dof_all_interfaces,
        )
        sys_man.set_equation(
            self.eq_single_interface,
            grids=[intf_top],
            equations_per_grid_entity=dof_single_interface,
        )
        self.eq_inds = np.array(
            [
                mdg.num_subdomain_cells(),
                mdg.subdomains()[0].num_cells,
                mdg.num_interface_cells() * 2,
                mdg.interfaces()[0].num_cells * 2,
            ]
        )
        if not square_system:
            # One equation combining top and all subdomains.
            # Assigned last to avoid mess if omitted
            self.eq_combined = self.sd_top_variable * (proj * self.sd_variable)
            self.eq_combined.set_name("eq_combined")
            sys_man.set_equation(
                self.eq_combined, grids=[sd_top], equations_per_grid_entity=dof_combined
            )
            self.eq_inds = np.append(self.eq_inds, mdg.subdomains()[0].num_cells)

        self.A, self.b = sys_man.assemble()
        self.sys_man = sys_man

        # Store subdomains and interfaces
        self.subdomains = mdg.subdomains()
        self.sd_top = sd_top
        self.interfaces = mdg.interfaces()
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
            return self.mdg.num_interface_cells() * 2
        elif var == self.intf_top_variable:
            return self.intf_top.num_cells * 2
        else:
            raise ValueError

    def dof_ind(self, var):
        # For a given variable, get the global indices assigned by
        # the DofManager. Based on knowledge of how the variables were
        # defined in self.__init__

        return self.sys_man.dofs_of([var])

    def eq_ind(self, name):
        # Get row indices of an equation, based on the (known) order in which
        # equations were added to the EquationSystem
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
    return EquationSystemSetup()


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
                vars.append("z")  # sd_top_variable
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
    """Test the set and get methods of the EquationSystem class.

    The test is performed for a number of different combinations of variables.

    Values are assigned to the STATE or ITERATE storage, and then retrieved.
    NOTE: Setting to STATE has not been parametrized, since this would double the number
    of tests, and since the asymmetry between the get and set methods (set can set to
    STATE and/or ITERATE, get can only get from one of them) requires some special
    handling in the test; see below.

    Both setting and adding values are tested.

    """
    sys_man = setup.sys_man

    np.random.seed(42)

    variables = _variable_from_setup(
        setup, as_str, on_interface, on_subdomain, single_grid, full_grid
    )
    # Indices of the active variables in this test configuration. Note that inds is
    # ordered according to variables, whereas the variable states to be returned later
    # are ordered according to the global ordering of unknowns.
    inds = sys_man.dofs_of(variables)

    # First generate random values, set them, and then retrieve them.
    vals = np.random.rand(inds.size)

    # Set values to ITERATE if specified, but not to state.
    sys_man.set_variable_values(vals, variables, to_iterate=iterate, to_state=False)
    retrieved_vals = sys_man.get_variable_values(variables, from_iterate=True)
    # Iterate may or may not have been updated; if not, it should have the default
    # value of 0.0
    if iterate:
        assert np.allclose(vals, retrieved_vals)
    else:
        # This was fetched from state, which still has the intial values.
        # To restrict to the variables of interest (those present in 'varables'),
        # we consider a restricted set of indices, as defined by 'inds', however, we
        # also need to do a sort, since get_variable_values returns the values in
        # the global ordering.
        assert np.allclose(setup.initial_values[np.sort(inds)], retrieved_vals)
    # State should not have been updated
    retrieved_vals_state = sys_man.get_variable_values(variables, from_iterate=False)
    assert np.allclose(setup.initial_values[np.sort(inds)], retrieved_vals_state)

    # Set values again, this time also to the state.
    sys_man.set_variable_values(vals, variables, to_iterate=iterate, to_state=True)
    # Retrieve only values from state; iterate should be the same as before (and the
    # additive mode is checked below).
    retrieved_vals_state = sys_man.get_variable_values(variables, from_iterate=False)
    assert np.allclose(vals, retrieved_vals_state)

    # Set new values without setting additive to True. This should overwrite the old
    # values.
    new_vals = np.random.rand(inds.size)
    sys_man.set_variable_values(new_vals, variables, to_iterate=iterate, to_state=False)
    retrieved_vals2 = sys_man.get_variable_values(variables, from_iterate=iterate)
    # Iterate has either been updated, or it still has the initial value
    if iterate:
        assert np.allclose(new_vals, retrieved_vals2)
    else:
        # This was fetched from state, which still has vals
        assert np.allclose(vals, retrieved_vals2)

    # Set values to state. This should overwrite the old values.
    sys_man.set_variable_values(new_vals, variables, to_iterate=iterate, to_state=True)
    retrieved_vals_state_2 = sys_man.get_variable_values(variables, from_iterate=False)
    assert np.allclose(new_vals, retrieved_vals_state_2)

    # Set the values again, this time with additive=True. This should double the
    # retrieved values.
    sys_man.set_variable_values(
        new_vals, variables, to_iterate=iterate, to_state=False, additive=True
    )
    retrieved_vals3 = sys_man.get_variable_values(variables, from_iterate=iterate)
    if iterate:
        assert np.allclose(2 * new_vals, retrieved_vals3)
    else:
        # This was fetched from state, which still has new_vals
        assert np.allclose(new_vals, retrieved_vals3)

    # And finally set to state, with additive=True. This should double the retrieved
    sys_man.set_variable_values(
        new_vals, variables, to_iterate=iterate, to_state=True, additive=True
    )
    retrieved_vals_state_3 = sys_man.get_variable_values(variables, from_iterate=False)
    assert np.allclose(2 * new_vals, retrieved_vals_state_3)


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

    # The tests compare assembly by a EquationSystem with explicitly defined
    # secondary variables with a 'truth' based on direct elimination of columns
    # in the Jacobian matrix.
    # The expected behavior is that the residual vector is fixed, while the
    # Jacobian matrix is altered only in columns that correspond to eliminated
    # variables.

    variables = [var for var in setup.sys_man.variables if var.name not in var_names]

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


def test_set_remove_equations(setup):

    sys_man = setup.sys_man

    dof_info_subdomain = {"cells": 1}
    dof_info_interface = {"cells": 2}

    # First try to set an equation that is already present. This should raise an error.
    with pytest.raises(ValueError):
        sys_man.set_equation(
            setup.eq_all_subdomains,
            grids=setup.subdomains,
            equations_per_grid_entity=dof_info_subdomain,
        )

    # Now remove all equations.
    eq_keys = list(sys_man.equations.keys())
    for name in eq_keys:
        sys_man.remove_equation(name)

    # Also remove an non-existing equation, check that an error is raised.
    with pytest.raises(ValueError):
        sys_man.remove_equation("nonexistent")

    # Now set the equation again.
    sys_man.set_equation(
        setup.eq_single_subdomain,
        grids=[setup.sd_top],
        equations_per_grid_entity=dof_info_subdomain,
    )

    # Check that the equation size has been set correctly
    blocks = sys_man._equation_image_space_composition
    assert np.allclose(
        blocks[setup.eq_single_subdomain.name][setup.sd_top],
        np.arange(setup.sd_top.num_cells * dof_info_subdomain["cells"]),
    )

    # Add a second equation, defined on both subdomains
    sys_man.set_equation(
        setup.eq_all_subdomains,
        grids=setup.subdomains,
        equations_per_grid_entity=dof_info_subdomain,
    )
    offset = 0
    for sd in setup.subdomains:
        assert np.allclose(
            blocks[setup.eq_all_subdomains.name][sd],
            offset + np.arange(sd.num_cells * dof_info_subdomain["cells"]),
        )
        offset += sd.num_cells * dof_info_subdomain["cells"]

    # Add a third equation, defined on the interface.
    # This time we switch the order of the interfaces. This should not matter for the
    # indices of the equation, since these are added in the order returned by
    # mdg.interfaces()
    sys_man.set_equation(
        setup.eq_all_interfaces,
        grids=setup.interfaces[::-1],
        equations_per_grid_entity=dof_info_interface,
    )
    offset = 0
    for intf in setup.interfaces:
        assert np.allclose(
            blocks[setup.eq_all_interfaces.name][intf],
            offset + np.arange(intf.num_cells * dof_info_interface["cells"]),
        )
        offset += intf.num_cells * dof_info_interface["cells"]


def test_parse_variable_like():
    """Test the private function _parse_variable_type().

    Thorough testing of this function allows us to assume variable parsing is properly
    handled for more advanced functions (e.g., assembly), thus there is no need to test
    those functions on varying input formats.

    We test only variables on subdomanis; the EquationSystem does not care about the
    type of domain for a variable.

    """
    setup = EquationSystemSetup()
    eq_system = setup.sys_man

    num_subdomains = setup.mdg.num_subdomains()
    num_interfaces = setup.mdg.num_interfaces()

    # First pass None, this should give all variables
    received_variables_1 = eq_system._parse_variable_type(None)
    assert len(received_variables_1) == len(set(received_variables_1))
    assert len(received_variables_1) == num_interfaces + num_subdomains + 2

    # Next pass an empty list. We expect to receive an empty list back
    received_variables_2 = eq_system._parse_variable_type([])
    assert len(received_variables_2) == 0

    # Pass a Variable, we should get it back
    var = setup.sd_top_variable.sub_vars[0]
    received_variables_3 = eq_system._parse_variable_type([var])
    assert len(received_variables_3) == 1
    assert received_variables_3[0] == var

    # Pass an md-variable with one sub-variable, check we get back the sub-variable
    received_variables_4 = eq_system._parse_variable_type([setup.sd_top_variable])
    assert len(received_variables_4) == 1
    assert received_variables_4[0] == setup.sd_top_variable.sub_vars[0]

    # Send an md-variable with two sub-variables
    received_variables_5 = eq_system._parse_variable_type([setup.sd_variable])
    assert len(received_variables_5) == len(setup.sd_variable.sub_vars)
    assert all([var in received_variables_5 for var in setup.sd_variable.sub_vars])

    # Send in the md-variable as a string, it should not make a difference
    received_variables_6 = eq_system._parse_variable_type([setup.name_sd_variable])
    assert len(received_variables_6) == len(setup.sd_variable.sub_vars)
    assert all([var in received_variables_6 for var in setup.sd_variable.sub_vars])

    # Send in two md-variables
    received_variables_7 = eq_system._parse_variable_type(
        [setup.sd_variable, setup.sd_top_variable]
    )
    assert len(received_variables_7) == len(setup.sd_variable.sub_vars) + len(
        setup.sd_top_variable.sub_vars
    )

    # Send in a combination of string and variables.
    # NOTE: While this combination goes against the specification in the type hints
    # (list of strings and list of variable is permitted, combined lists are not), the
    # current implementation actually allows for this. If the implementation is changed,
    # the checks involving received_variables_8 and _9 can be deleted.
    received_variables_8 = eq_system._parse_variable_type(
        setup.sd_variable.sub_vars + [setup.name_sd_top_variable]
    )
    assert len(received_variables_8) == len(setup.sd_variable.sub_vars) + len(
        setup.sd_top_variable.sub_vars
    )

    # Send in a combination of string and md-variables
    received_variables_9 = eq_system._parse_variable_type(
        [setup.sd_variable, setup.name_sd_top_variable]
    )
    assert len(received_variables_9) == len(setup.sd_variable.sub_vars) + len(
        setup.sd_top_variable.sub_vars
    )


def test_parse_single_equation():
    """Test the helper function for parsing single equations.

    We consider only the equation posed on all subdomains, parsing of other equations
    should be identical.

    The test considers restrictions of the equation to subsets of its domains,
    and verifies that the returned index sets are correctly ordered.

    """
    setup = EquationSystemSetup()
    eq_system = setup.sys_man

    # Represent the equation both by its string and its operator form.
    # This could have been parametrized to the price of higher cost.
    for eq in [setup.eq_all_subdomains, setup.eq_all_subdomains.name]:
        # The equation name.
        name = eq if isinstance(eq, str) else eq.name

        # First parse the equation as it is, without any restriction.
        # This should give back the full equation with no restriction.
        restriction_1 = eq_system._parse_single_equation(eq)
        assert len(restriction_1) == 1

        assert name in restriction_1
        assert restriction_1[name] is None

        # Next, restrict the equation to a single subdomain.
        restriction_2 = eq_system._parse_single_equation({eq: [setup.sd_top]})
        assert len(restriction_2) == 1
        # The numbering of the subdomanis in the EquationSystem is the same as that of
        # the MixedDimensionalGrid, thus the indices associated with this subdomain
        # will be 0-offset.
        assert np.allclose(restriction_2[name], np.arange(setup.sd_top.num_cells))

        # Next, permute the subdomains before sending them in. All subdomains are
        # present, thus the indices should cover all cells in the md-grid. Moreover,
        # the EquationSystem will sort the subdomains according in the same order as
        # the MixedDimensionalGrid.subdomains() method, thus the indices should again
        # be linear.
        eq_def = {eq: setup.subdomains[::-1]}
        restriction_3 = eq_system._parse_single_equation(eq_def)
        assert np.allclose(
            restriction_3[name], np.arange(setup.mdg.num_subdomain_cells())
        )


def test_parse_equations():
    """Test the helper function for parsing equations.

    The test focuses on the functionality of EquationSystem._parse_equation_like()
    beyond the parsing of individual equations, which is tested in the method
    test_parse_single_equation_like(). That is, we test the parsing of multiple
    equations and check that the order of the returned equations is correct.

    """
    setup = EquationSystemSetup()
    eq_system = setup.sys_man

    # All equations. The order is the same as that in the helper class
    # EquationSystemSetup.
    all_equation_names = [
        "eq_all_subdomains",
        "eq_single_subdomain",
        "eq_all_interfaces",
        "eq_single_interface",
        "eq_combined",
    ]

    # First pass None. This should give as all equations on all subdomains.
    received_equations_1 = eq_system._parse_equations(None)
    received_keys_1 = list(received_equations_1.keys())

    # We expect to receive all equations, thus the length of the dictionary should be
    # the same as the number of equations.
    assert len(received_equations_1) == len(all_equation_names)

    for eq, key in zip(all_equation_names, received_keys_1):
        # The keys of the received dictionary should be the same as the names of the
        # equations as they were set in the setup, and the order should be preserved.
        assert eq == key
        # There should be no restriction on indices.
        assert received_equations_1[eq] is None

    # Next, pass the single subdomain and all subdomains, in that order. We should
    # receive the same keys, but in reverse order.
    received_equations_2 = eq_system._parse_equations(
        [all_equation_names[1], all_equation_names[0]]
    )
    received_keys_2 = list(received_equations_2.keys())
    assert len(received_keys_2) == 2
    assert received_keys_2[0] == all_equation_names[0]
    assert received_keys_2[1] == all_equation_names[1]

    # Send in the all_subdomains equation in both unrestricted and restricted form.
    # The restriction should override the unrestricted form.
    received_equations_3 = eq_system._parse_equations(
        {all_equation_names[0]: None, all_equation_names[0]: [setup.sd_top]}
    )
    assert len(received_equations_3) == 1
    assert np.allclose(
        received_equations_3[all_equation_names[0]], np.arange(setup.sd_top.num_cells)
    )


@pytest.mark.parametrize(
    "var_names",
    [
        [],  # No secondary variables
        ["x"],  # A simple variable which is also part of a merged one
        ["y"],  # mixed-dimensional variable
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
    variables = [var for var in setup.sys_man.variables if var.name not in var_names]
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
        [
            "eq_all_subdomains",
            "eq_single_subdomain",
        ],  # The combination in reverse order
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
    """Test of functionality to assemble subsystems from an EquationSystem.

    The test is based on assembly of a subsystem and comparing this to a truth
    from assembly of the full system, and then explicitly dump rows and columns.

    We test combinations of one or more equations, together with one or more variables.
    Variables are only defined by strings; the alternative format of a variables is
    not considered, since the variables are only passed to EquationSystem.dofs_of()
    (via the method projection_to()).
    """
    sys_man = setup.sys_man

    # Convert variable names into variables
    if var_names is None:
        var_names = []
    variables = [var for var in setup.sys_man.variables if var.name not in var_names]

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
def test_extract_subsystem(setup, eq_names, var_names):
    # Check functionality to extract subsystems from the EquationManager.
    # The tests check that the expected variables and equations are present in
    # the subsystem.

    sys_man = setup.sys_man

    # Get Ad variables from strings
    variables = [getattr(setup, var) for var in var_names]

    # Secondary variables are constructed as atomic. The number of these is found by
    # the number of variables + the number of mixed-dimensional variables (known to be defined
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
    setup = EquationSystemSetup(square_system=True)
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
