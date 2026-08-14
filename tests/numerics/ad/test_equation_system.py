"""The module contains tests for evaluation of Ad operator trees through the
EquationSystem.

The tests focus on various assembly methods:
    * test_evaluate_variables: Test that variables are correctly evaluated at the
        current iterate, previous iterate and previous time step.
    * test_variable_creation: Test that variable creation from an EquationSystem works.
    * test_variable_tags: Tagging of variables, used for filtering in an EquationSystem.
    * test_set_get_methods: Get and set methods for variables, including methods for
        shifting between time steps and iterates.
    * test_set_remove_equations: Set and remove equations from an EquationSystem.
    * test_parse_variable_like, test_parse_single_equation, test_parse_equations:
        Parsing of equations, and the methods used to do so. Thorough testing here means
        other tests can get away with fewer parameter checks.
    * test_assemble: Assemble sub-blocks of the full set of equations.
    * test_extract_subsystem: Extract a new EquationSystem for a subset of equations.

"""

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.mdg_library import square_with_orthogonal_fractures
from porepy.numerics.ad.equation_system import GridEntity


def test_evaluate_variables():
    """Test that the combination of an Ad operator tree and the EquationSystem correctly
    evaluates variables at the current iterate, previous iterate and previous time step.

    The testing is based on parsing of individual variables and of the result of the
    time increment method. The time derivative method is not tested, since this is
    essentially the same as the time increment method (only division by dt is added).
    """
    mdg = pp.MixedDimensionalGrid()
    g_1 = pp.CartGrid([1, 1])
    g_2 = pp.CartGrid([1, 1])
    mdg.add_subdomains([g_1, g_2])

    equation_system = pp.ad.EquationSystem(mdg)

    # Define variables
    var_name = "foo"
    equation_system.create_variables(var_name, subdomains=[g_1, g_2])

    for sd, d in mdg.subdomains(return_data=True):
        vals_sol = np.ones([sd.num_cells])
        vals_it = 2 * np.ones([sd.num_cells])

        pp.set_solution_values(
            name=var_name, values=vals_sol, data=d, time_step_index=0
        )
        pp.set_solution_values(name=var_name, values=vals_it, data=d, iterate_index=0)
        # Provide values for previous iterate as well
        pp.set_solution_values(name=var_name, values=vals_it, data=d, iterate_index=1)

    # We only need to test a single variable, they should all be the same.
    single_variable = equation_system.variables[0]
    # Make a md wrapper around the variable.
    md_variable = equation_system.md_variable(single_variable.name)

    # Try testing both the individual variable and its mixed-dimensional wrapper. In the
    # below checks, the anticipated value of the variables is based on the values
    # assigned in the model method.
    known_jacs = [np.array([1, 0]), np.eye(2)]
    for var, known_jac in zip([single_variable, md_variable], known_jacs):
        # First evaluate the variable. This should give the iterate value.
        ad_array = var.value_and_jacobian(equation_system)
        assert isinstance(ad_array, pp.ad.AdArray)
        assert np.allclose(ad_array.val, 2)
        assert np.allclose(ad_array.jac.toarray(), known_jac)

        # Now create the variable at the previous iterate. This should also give the
        # most recent value in pp.ITERATE_SOLUTIONS, but it should not yield an AdArray.
        var_prev_iter = var.previous_iteration()
        ad_array_prev_iter = var_prev_iter.value_and_jacobian(equation_system)
        assert isinstance(ad_array_prev_iter, pp.ad.AdArray)
        assert np.allclose(ad_array_prev_iter.val, 2)
        assert np.allclose(ad_array_prev_iter.jac.toarray(), np.zeros(known_jac.shape))

        # Create the variable at the previous time step. This should give the most
        # recent value in pp.TIME_STEP_SOLUTIONS.
        var_prev_timestep = var.previous_timestep()
        ad_array_prev_timestep = var_prev_timestep.value_and_jacobian(equation_system)
        assert isinstance(ad_array_prev_timestep, pp.ad.AdArray)
        assert np.allclose(ad_array_prev_timestep.val, 1)
        assert np.allclose(
            ad_array_prev_timestep.jac.toarray(), np.zeros(known_jac.shape)
        )

        # Use the ad machinery to define the difference between the current and previous
        # time step. This should give an AdArray with the same value as that obtained by
        # subtracting the evaluated variables.
        var_increment = pp.ad.time_increment(var)
        ad_array_increment = var_increment.value_and_jacobian(equation_system)
        assert isinstance(ad_array_increment, pp.ad.AdArray)
        assert np.allclose(
            ad_array_increment.val, ad_array.val - ad_array_prev_timestep.val
        )
        assert np.allclose(ad_array_increment.jac.toarray(), known_jac)


def test_variable_creation():
    """Test that variable creation from a EquationSystem works as expected.

    The test generates a MixedDimensionalGrid, defines some variables on it, and checks
    that the variables have the correct sizes and names.

    Also tested is the methods num_dofs() and partly dofs_of()
    """

    mdg, _ = square_with_orthogonal_fractures("cartesian", {"cell_size": 0.5}, [1])

    equation_system = pp.ad.EquationSystem(mdg)

    # Define the number of variables per grid item. Faces are included just to test that
    # this also works.
    num_dof_per_cell, num_dof_per_face = 1, 2
    dof_info_sd = {
        GridEntity.cells: num_dof_per_cell,
        GridEntity.faces: num_dof_per_face,
    }
    dof_info_intf = {GridEntity.cells: num_dof_per_cell}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    interfaces = mdg.interfaces()

    # Define one variable on all subdomains, one on a single subdomain, and one on
    # all interfaces (there is only one interface in this grid).
    subdomain_variable = equation_system.create_variables(
        "var_1", dof_info_sd, subdomains=subdomains
    )
    single_subdomain_variable = equation_system.create_variables(
        "var_2", dof_info_sd, subdomains=single_subdomain
    )
    interface_variable = equation_system.create_variables(
        "var_3", dof_info_intf, interfaces=interfaces
    )

    # Check that the variable storage in the EquationSystem is correct.
    # subdomain_variable is defined on two subdomains, the other on one.
    assert len(equation_system.variables) == 4

    # Tests of the md_variable method
    # Fetch a version of var_1. It should have the same sub-variables as the md-variable
    # returned from variable creation.
    subdomain_variable_fetched = equation_system.md_variable("var_1")
    assert len(subdomain_variable.sub_vars) == len(subdomain_variable_fetched.sub_vars)
    for sub_var in subdomain_variable_fetched.sub_vars:
        assert sub_var in subdomain_variable.sub_vars

    # Next, fetch a version of var_1, restricted to a single subdomain. This should
    # in practice be equivalent to var_2.
    single_subdomain_variable_fetched = equation_system.md_variable(
        "var_1", domains=single_subdomain
    )
    assert len(single_subdomain_variable_fetched.sub_vars) == 1
    assert (
        single_subdomain_variable_fetched.sub_vars[0].name == "var_1"
        and single_subdomain_variable_fetched.domains[0] == single_subdomain[0]
    )

    # Check that the variables are created correctly
    assert subdomain_variable.name == "var_1"
    assert single_subdomain_variable.name == "var_2"
    assert interface_variable.name == "var_3"

    # Check that the number of dofs is correct for each variable.
    num_subdomain_faces = sum([sd.num_faces for sd in subdomains])

    # Compute the expected number of dofs for each variable, by multiplying the number
    # of cells and faces with the number of dofs per cell and face. Use the right
    # grids (single subdomain or the whole mdg) in the computations.
    ndof_subdomains = (
        mdg.num_subdomain_cells() * num_dof_per_cell
        + num_subdomain_faces * num_dof_per_face
    )
    ndof_single_subdomain = (
        single_subdomain[0].num_cells * num_dof_per_cell
        + single_subdomain[0].num_faces * num_dof_per_face
    )
    ndof_interface = mdg.num_interface_cells() * num_dof_per_cell

    assert (
        equation_system.dofs_of(subdomain_variable.sub_vars).size
    ) == ndof_subdomains
    assert (
        equation_system.dofs_of(single_subdomain_variable.sub_vars).size
        == ndof_single_subdomain
    )
    assert equation_system.dofs_of(interface_variable.sub_vars).size == ndof_interface

    assert (
        equation_system.num_dofs()
        == ndof_subdomains + ndof_single_subdomain + ndof_interface
    )


@pytest.mark.parametrize("variable_to_be_removed", ["var_1", "var_2", "var_3", None])
def test_remove_variables(variable_to_be_removed):
    """Test that removing one of three md-variables from an EquationSystem works as
    expected.

    The test generates a MixedDimensionalGrid and defines some variables on it. The test
    then removes one of the variables, and checks that the remaining variables are
    correctly stored in the EquationSystem.

    Parameters:
        variable_to_be_removed: The name of the variable to be removed. If None, no
            variable is removed.
    """
    mdg, _ = square_with_orthogonal_fractures("cartesian", {"cell_size": 0.5}, [1])

    equation_system = pp.ad.EquationSystem(mdg)
    # Define the number of variables per grid item. Faces are included just to test that
    # # this also works.
    num_dof_per_cell, num_dof_per_face = 1, 2
    dof_info_sd = {
        GridEntity.cells: num_dof_per_cell,
        GridEntity.faces: num_dof_per_face,
    }
    dof_info_intf = {GridEntity.cells: num_dof_per_cell}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    interfaces = mdg.interfaces()

    # Define one variable on all subdomains, one on a single subdomain, and one on all
    # interfaces (there is only one interface in this grid).
    var_1 = equation_system.create_variables(
        "var_1", dof_info_sd, subdomains=subdomains
    )
    var_2 = equation_system.create_variables(
        "var_2", dof_info_sd, subdomains=single_subdomain
    )
    var_3 = equation_system.create_variables(
        "var_3", dof_info_intf, interfaces=interfaces
    )

    if variable_to_be_removed:
        var_to_remove = equation_system.md_variable(variable_to_be_removed)
        equation_system.remove_variables([var_to_remove])
        # Check that the EquationSystem does not contain the removed variable anymore.
        assert all(
            variable.name != variable_to_be_removed
            for variable in equation_system.variables
        )
        assert all(
            variable.name != variable_to_be_removed
            for variable in equation_system.variable_indexer.indices
        )
        # Check that trying to remove the variable again raises an error.
        with pytest.raises(ValueError):
            equation_system.remove_variables([var_to_remove])
    else:
        equation_system.remove_variables([])

    # Identify remaining subvariables. This allows direct comparison with
    # equation_system.variables.
    remaining_vars = []
    for var in [var_1, var_2, var_3]:
        if var.name == variable_to_be_removed:
            continue
        remaining_vars.extend(var.sub_vars)

    remaining_dofs = np.hstack(
        [equation_system.dofs_of([var]) for var in remaining_vars]
    )

    # Check that the number of dofs is correct and that the dofs form a continuous
    # range.
    assert np.allclose(np.sort(remaining_dofs), np.arange(equation_system.num_dofs()))
    # Check that the number of variables is correct.
    assert len(equation_system.variables) == len(remaining_vars)


def test_variable_tags():
    """Test that variables can be tagged, and that the tags are correctly propagated
    to the underlying atomic variables.

    The test generates a MixedDimensionalGrid, defines some variables on it, assigns
    and modifies tags, and checks that the tags updated as expected.

    """
    mdg, _ = square_with_orthogonal_fractures("cartesian", {"cell_size": 0.5}, [1])

    equation_system = pp.ad.EquationSystem(mdg)

    # Define the number of variables per grid item. Faces are included just to test that
    # this also works.
    dof_info = {GridEntity.cells: 1}

    # Create variables on subdomain and interface.
    subdomains = mdg.subdomains()
    single_subdomain = mdg.subdomains(dim=mdg.dim_max())

    # Define one variable on all subdomains, one on a single subdomain, and one on
    # all interfaces (there is only one interface in this grid).
    var_1 = equation_system.create_variables(
        "var_1", dof_info, subdomains=subdomains, tags={"tag_1": 1}
    )
    var_2 = equation_system.create_variables(
        "var_2", dof_info, subdomains=single_subdomain
    )

    assert var_1.tags == {"tag_1": 1}
    # By default, variables should not have tags
    assert len(var_2.tags) == 0

    # Add a tag to var_1. This will modify the underlying atomic variables, but not
    # var_1 itself.
    equation_system.update_variable_tags({"tag_2": 2}, [var_1])
    assert all(
        [
            var.tags["tag_2"] == 2
            for var in equation_system.variables
            if var.name == "var_1"
        ]
    )

    assert "tag_2" not in var_1.tags
    # However, if we fetch a new md-variable representing var_1, it should have the tag.
    var_1_new = equation_system.md_variable("var_1")
    assert "tag_2" in var_1_new.tags and var_1_new.tags["tag_2"] == 2

    # We can also add a tag to var_1, this will not be seen by the underlying atomic
    # variables.
    var_1.tags["tag_3"] = 3
    # Strictly speaking, this will also test the variables that are not part of var_1,
    # that should be fine, they should not have the tag.
    assert all(["tag_3" not in var.tags for var in equation_system.variables])

    # Add tags to var_2. This will be useful when we test filtering of variables below.
    equation_system.update_variable_tags({"tag_2": 4}, [var_2])
    equation_system.update_variable_tags({"tag_3": False}, [var_1])
    equation_system.update_variable_tags({"tag_3": True}, [var_2])

    ## Test of get_variables
    # First no filtering. This should give all variables.
    retrieved_var_1 = equation_system.get_variables()
    assert len(retrieved_var_1) == 3
    # Also uniquify, this should not change the length
    assert len(retrieved_var_1) == len(set(retrieved_var_1))

    # Filter on variable name var_1. Here we send in a list of atomic variables and
    # should recieve the same list.
    retrieved_var_2 = equation_system.get_variables(variables=var_1.sub_vars)
    assert len(retrieved_var_2) == 2
    assert all([var in retrieved_var_2 for var in var_1.sub_vars])

    # Filter on grids.
    retrieved_var_3 = equation_system.get_variables(grids=single_subdomain)
    assert len(retrieved_var_3) == 2
    assert all([var.domains[0] == single_subdomain[0] for var in retrieved_var_3])

    # Filter on combination of grid and variable
    retrieved_var_4 = equation_system.get_variables(
        grids=single_subdomain, variables=var_1.sub_vars
    )
    assert len(retrieved_var_4) == 1

    # Filter on a non-existing tag name.
    retrieved_var_5 = equation_system.get_variables(tag_name="tag_4")
    assert len(retrieved_var_5) == 0
    # Filter on a tag that exists, but with a non-exiting value
    retrieved_var_6 = equation_system.get_variables(tag_name="tag_2", tag_value=5)
    assert len(retrieved_var_6) == 0

    # Filter on the name tag_1. This should give only var_1
    retrieved_var_7 = equation_system.get_variables(tag_name="tag_1")
    assert len(retrieved_var_7) == 2
    assert all([var in retrieved_var_7 for var in var_1.sub_vars])

    # Filter on the name tag_2. This should give var_1 and var_2
    retrieved_var_8 = equation_system.get_variables(tag_name="tag_2")
    assert len(retrieved_var_8) == 3

    # Filter on tag_2, with value 2. This should give only var_1
    retrieved_var_9 = equation_system.get_variables(tag_name="tag_2", tag_value=2)
    assert len(retrieved_var_9) == 2
    assert all([var in retrieved_var_9 for var in var_1.sub_vars])

    # Filter on tag_3, which takes boolean values. This should give only var_2
    retrieved_var_10 = equation_system.get_variables(tag_name="tag_3", tag_value=True)
    assert len(retrieved_var_10) == 1
    assert retrieved_var_10[0].domains[0] == single_subdomain[0]


class EquationSystemMockModel:
    """Class to set up a EquationSystem with a combination of variables
    and equations, designed to make it convenient to test critical functionality of the
    EquationSystem.

    The model is intended for testing advanced functionality, like assembly of
    equations, construction of subsystems of equations etc. The below model is in itself
    a test of basic functionality, like creation of variables and equations. TODO: We
    should have dedicated tests for variable creation, to make sure we cover all
    options.
    """

    def __init__(self, square_system=False):
        mdg, _ = square_with_orthogonal_fractures(
            "cartesian", {"cell_size": 0.5}, [0, 1]
        )

        equation_system = pp.ad.EquationSystem(mdg)

        # List of all subdomains
        subdomains = mdg.subdomains()
        # Also generate a variable on the top-dimensional domain
        sd_top = mdg.subdomains(dim=mdg.dim_max())[0]

        interfaces = mdg.interfaces()
        intf_top = mdg.interfaces(dim=mdg.dim_max() - 1)[0]

        self.name_sd_variable = "x"
        self.sd_variable = equation_system.create_variables(
            self.name_sd_variable, subdomains=subdomains
        )

        # Let interface variables have size 2, this gives us a bit more to play with
        # in the testing of assembly.
        self.name_intf_variable = "y"
        self.intf_variable = equation_system.create_variables(
            self.name_intf_variable,
            dof_info={GridEntity.cells: 2},
            interfaces=interfaces,
        )

        self.name_sd_top_variable = "z"
        self.sd_top_variable = equation_system.create_variables(
            self.name_sd_top_variable, subdomains=[sd_top]
        )

        self.name_intf_top_variable = "w"
        self.intf_top_variable = equation_system.create_variables(
            self.name_intf_top_variable,
            dof_info={GridEntity.cells: 2},
            interfaces=[intf_top],
        )

        # Set the time step and iterate solution values for the variables.
        # The assigned numbers are not important, the comparisons below will be between
        # an assembled matrix for the full and for a reduced system, and similar for
        # the right hand side vector.
        global_vals = np.arange(equation_system.num_dofs())
        global_vals[equation_system.dofs_of([self.sd_variable])] = np.arange(
            mdg.num_subdomain_cells()
        )
        global_vals[equation_system.dofs_of([self.sd_top_variable])] = np.arange(
            sd_top.num_cells
        )
        global_vals[equation_system.dofs_of([self.intf_variable])] = np.arange(
            mdg.num_interface_cells() * 2
        )
        global_vals[equation_system.dofs_of([self.intf_top_variable])] = np.arange(
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
        equation_system.set_variable_values(
            global_vals, variables=all_variables, iterate_index=0, time_step_index=0
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

        dof_all_subdomains = {GridEntity.cells: 1}
        dof_single_subdomain = {GridEntity.cells: 1}
        dof_combined = {GridEntity.cells: 1}

        equation_system.set_equation(
            self.eq_all_subdomains,
            equations_per_grid_entity=dof_all_subdomains,
        )
        equation_system.set_equation(
            self.eq_single_subdomain,
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
        dof_all_interfaces = {GridEntity.cells: 2}
        dof_single_interface = {GridEntity.cells: 2}
        equation_system.set_equation(
            self.eq_all_interfaces,
            equations_per_grid_entity=dof_all_interfaces,
        )
        equation_system.set_equation(
            self.eq_single_interface,
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
            self.eq_combined = self.sd_top_variable * (proj @ self.sd_variable)
            self.eq_combined.set_name("eq_combined")
            equation_system.set_equation(
                self.eq_combined, equations_per_grid_entity=dof_combined
            )
            self.eq_inds = np.append(self.eq_inds, mdg.subdomains()[0].num_cells)

        self.all_equation_names = [
            "eq_all_subdomains",
            "eq_single_subdomain",
            "eq_all_interfaces",
            "eq_single_interface",
            "eq_combined",
        ]
        self.all_variable_names = ["x", "y", "z", "w"]

        linear_system = equation_system.assemble()
        self.A = linear_system.matrix
        self.b = linear_system.rhs
        self.equation_system = equation_system

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

        return self.equation_system.dofs_of([var])

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

    def add_equation_on_empty_domain(self):
        # Add an equation on an empty domain to the equation system.
        empty_var = self.equation_system.create_variables(
            name="empty_var", subdomains=[]
        )
        empty_equation = empty_var * empty_var
        empty_equation.set_name("empty_equation")
        self.equation_system.set_equation(
            empty_equation, equations_per_grid_entity={GridEntity.cells: 1}
        )


@pytest.fixture(scope="function")
def model() -> EquationSystemMockModel:
    # Method to deliver a model to all tests
    return EquationSystemMockModel()


def _variable_from_model(
    model: EquationSystemMockModel,
    as_str: bool,
    on_interface: bool,
    on_subdomain: bool,
    single_grid: bool,
    full_grid: bool,
):
    # Helper method to get a variable from the model, either as a string or as
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
                vars.append(model.intf_top_variable)
            if full_grid:
                vars.append(model.intf_variable)
    if on_subdomain:
        if as_str:
            if single_grid:
                vars.append("z")  # sd_top_variable
            if full_grid:
                vars.append("x")
        else:
            if single_grid:
                vars.append(model.sd_top_variable)
            if full_grid:
                vars.append(model.sd_variable)
    return vars


@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("on_interface", [True, False])
@pytest.mark.parametrize("on_subdomain", [True, False])
@pytest.mark.parametrize("single_grid", [True, False])
@pytest.mark.parametrize("full_grid", [True, False])
@pytest.mark.parametrize("iterate", [True, False])
def test_set_get_methods(
    model: EquationSystemMockModel,
    as_str,
    on_interface,
    on_subdomain,
    single_grid,
    full_grid,
    iterate,
):
    """Test the set and get methods of the EquationSystem class.

    The test is performed for a number of different combinations of variables. The set
    and get methods are also tested together with the shift_time_step_values and
    shift_iterate_values methods which are used for handling multiple stored solutions.

    Values are assigned to ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS``, and
    then retrieved.

    NOTE: Setting to ``pp.TIME_STEP_SOLUTIONS`` has not been parametrized, since this
    would double the number of tests, and since the asymmetry between the get and set
    methods (set can set to ``pp.TIME_STEP_SOLUTIONS`` and/or ``pp.ITERATE_SOLUTIONS,``
    get can only get from one of them) requires some special handling in the test; see
    below.

    Both setting and adding values are tested.

    """

    equation_system: pp.ad.EquationSystem = model.equation_system

    np.random.seed(42)

    variables = _variable_from_model(
        model, as_str, on_interface, on_subdomain, single_grid, full_grid
    )
    # Indices of the active variables in this test configuration. Note that inds is
    # ordered according to variables, whereas the variable solutions to be returned
    # later are ordered according to the global ordering of unknowns.

    inds = equation_system.dofs_of(variables)

    # IMPLEMENATION NOTE: The first set of tests (down to checking of time_step_indices
    # other than 0) could possibly be streamlined by using a helper function and sending
    # in relevant arguments (additive, time_step_index, iterate_index etc.). However,
    # this would lead to a nested set of if-else statements that would require much more
    # comments to be understandable. The current implementation is more verbose, but
    # hopefully easier to understand.

    # First generate random values, set them, and then retrieve them.
    vals = np.random.rand(inds.size)

    # Set values to pp.ITERATE_SOLUTIONS if specified, but not to
    # pp.TIME_STEP_SOLUTIONS.
    if iterate:
        equation_system.set_variable_values(vals, variables, iterate_index=0)

    retrieved_vals = equation_system.get_variable_values(variables, iterate_index=0)
    # Iterate may or may not have been updated; if not, it should have the default
    # value of 0.0
    if iterate:
        assert np.allclose(vals, retrieved_vals)
    else:
        # This was fetched from the stored time steps, which still has the intial
        # values.
        # To restrict to the variables of interest (those present in 'variables'),
        # we consider a restricted set of indices, as defined by 'inds', however, we
        # also need to do a sort, since get_variable_values returns the values in
        # the global ordering.
        assert np.allclose(model.initial_values[np.sort(inds)], retrieved_vals)
    # The time step solution should not have been updated
    retrieved_vals_state = equation_system.get_variable_values(
        variables, time_step_index=0
    )
    assert np.allclose(model.initial_values[np.sort(inds)], retrieved_vals_state)

    # Set values again, this time also to the time step solutions.
    if iterate:
        equation_system.set_variable_values(
            vals, variables, iterate_index=0, time_step_index=0
        )
    else:
        equation_system.set_variable_values(vals, variables, time_step_index=0)
    # Retrieve only values from time step solutions; iterate should be the same as
    # before (and the additive mode is checked below).

    retrieved_vals_state = equation_system.get_variable_values(
        variables, time_step_index=0
    )

    assert np.allclose(vals, retrieved_vals_state)

    # Set new values without setting additive to True. This should overwrite the old
    # values.
    new_vals = np.random.rand(inds.size)
    if iterate:
        equation_system.set_variable_values(new_vals, variables, iterate_index=0)
        retrieved_vals2 = equation_system.get_variable_values(
            variables, iterate_index=0
        )
    if not iterate:
        retrieved_vals2 = equation_system.get_variable_values(
            variables, time_step_index=0
        )
    # Iterate has either been updated, or it still has the initial value
    if iterate:
        assert np.allclose(new_vals, retrieved_vals2)
    else:
        # This was fetched from the stored time step solutions, which still has vals
        assert np.allclose(vals, retrieved_vals2)

    # Set values to time step solutions. This should overwrite the old values.
    if iterate:
        equation_system.set_variable_values(
            new_vals, variables, iterate_index=0, time_step_index=0
        )
    else:
        equation_system.set_variable_values(new_vals, variables, time_step_index=0)
    retrieved_vals_state_2 = equation_system.get_variable_values(
        variables, time_step_index=0
    )
    assert np.allclose(new_vals, retrieved_vals_state_2)

    # Set the values again, this time with additive=True. This should double the
    # retrieved values.
    if iterate:
        equation_system.set_variable_values(
            new_vals, variables, iterate_index=0, additive=True
        )
        retrieved_vals3 = equation_system.get_variable_values(
            variables, iterate_index=0
        )
    elif not iterate:
        retrieved_vals3 = equation_system.get_variable_values(
            variables, time_step_index=0
        )

    if iterate:
        assert np.allclose(2 * new_vals, retrieved_vals3)
    else:
        # This was fetched from stored time step solutions, which still has new_vals
        assert np.allclose(new_vals, retrieved_vals3)

    # Set to time step solutions, with additive=True. This should double the retrieved
    if iterate:
        equation_system.set_variable_values(
            new_vals, variables, iterate_index=0, time_step_index=0, additive=True
        )
    else:
        equation_system.set_variable_values(
            new_vals, variables, time_step_index=0, additive=True
        )
    retrieved_vals_state_3 = equation_system.get_variable_values(
        variables, time_step_index=0
    )
    assert np.allclose(2 * new_vals, retrieved_vals_state_3)

    # Test storage of multiple values of time step and iterate solutions from here and
    # down. In practice this means checking that the functionality of shifting
    # dictionary values and then set the most recent time step/iterate value works as
    # expected.

    def _retrieve_and_check_time_step(known_values):
        # Helper method to retrieve values from time step solutions and check that they
        # are as expected.
        for ind, val in enumerate(known_values):
            assert np.allclose(
                equation_system.get_variable_values(variables, time_step_index=ind), val
            )

    # Building a few solution vectors and defining the desired solution indices
    vals0 = vals
    vals1 = vals0 * 2
    vals2 = vals0 * 3

    solution_indices = np.array([0, 1, 2])
    vals_mat = np.array([vals0, vals1, vals2])

    # Test setting values at several indices and then gathering them
    for i, val in zip(solution_indices, vals_mat):
        equation_system.set_variable_values(
            values=val, variables=variables, time_step_index=i
        )

    _retrieve_and_check_time_step([vals0, vals1, vals2])

    # Test functionality that shifts values to prepare setting of the most recent
    # solution values.
    equation_system.shift_time_step_values(max_index=len(solution_indices))
    # The expected result is that key 0 and 1 has the same values, and key 2 have the
    # values that were at key 1 before the values were shifted.
    _retrieve_and_check_time_step([vals0, vals0, vals1])

    # Test additive = True to make sure only the most recently stored values are added
    # to.
    equation_system.set_variable_values(
        values=vals0, variables=variables, time_step_index=0, additive=True
    )
    _retrieve_and_check_time_step([2 * vals0, vals0, vals1])

    # Finally test setting and getting values at a non-zero storage index
    equation_system.set_variable_values(
        values=vals2, variables=variables, time_step_index=2
    )

    retrieved_set_ind_vals2 = equation_system.get_variable_values(
        variables, time_step_index=2
    )

    assert np.allclose(retrieved_set_ind_vals2, vals2)


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
def test_projection_matrix(model: EquationSystemMockModel, var_names):
    # Test of the projection matrix method. The only interesting test is the
    # secondary variable functionality (the other functionality is tested elsewhere).

    # The tests compare assembly by a EquationSystem with explicitly defined
    # secondary variables with a 'truth' based on direct elimination of columns
    # in the Jacobian matrix.
    # The expected behavior is that the residual vector is fixed, while the
    # Jacobian matrix is altered only in columns that correspond to eliminated
    # variables.

    variables = [
        var for var in model.equation_system.variables if var.name not in var_names
    ]

    proj = model.equation_system.projection_to(variables=variables)

    # Get dof indices of the variables that have been eliminated
    if len(var_names) > 0:
        removed_dofs = np.sort(
            np.hstack([model.equation_system.dofs_of([var]) for var in var_names])
        )
    else:
        removed_dofs = []

    remaining_dofs = np.setdiff1d(
        np.arange(model.equation_system.num_dofs()), removed_dofs
    )

    assert np.allclose(proj.indices, remaining_dofs)


def test_set_remove_equations(model: EquationSystemMockModel):
    equation_system = model.equation_system

    dof_info_subdomain = {GridEntity.cells: 1}
    dof_info_interface = {GridEntity.cells: 2}

    # First try to set an equation that is already present. This should raise an error.
    with pytest.raises(ValueError):
        equation_system.set_equation(
            model.eq_all_subdomains,
            equations_per_grid_entity=dof_info_subdomain,
        )

    # Now remove all equations.
    eq_keys = list(equation_system.equations.keys())
    for name in eq_keys:
        equation_system.remove_equation(name)

    # Also remove an non-existing equation, check that an error is raised.
    with pytest.raises(ValueError):
        equation_system.remove_equation("nonexistent")

    # Now set the equation again.
    equation_system.set_equation(
        model.eq_single_subdomain,
        equations_per_grid_entity=dof_info_subdomain,
    )

    # Check that the mapping of equation to subdomain to global dof
    # indices is correctly set. Note: in this test, we access the indexer through
    # equation_system.equation_indexer (and don't assign it to a local variable),
    # because it recomputes every time the equations are changed. So it's a new indexer
    # each time it is accessed.
    equation_subdomain_blocks = (
        equation_system.equation_indexer.equation_image_space_composition
    )
    assert np.allclose(
        equation_subdomain_blocks[model.eq_single_subdomain.name][model.sd_top],
        np.arange(model.sd_top.num_cells * dof_info_subdomain[GridEntity.cells]),
    )
    # Check that the mapping of equation to grid entity block size
    # is set correctly.
    equation_grid_entity_blocks = equation_system.equation_image_size_info
    assert (
        equation_grid_entity_blocks[model.eq_single_subdomain.name]
        == dof_info_subdomain
    )

    # Add a second equation, defined on both subdomains
    equation_system.set_equation(
        model.eq_all_subdomains,
        equations_per_grid_entity=dof_info_subdomain,
    )
    equation_subdomain_blocks = (
        equation_system.equation_indexer.equation_image_space_composition
    )
    offset = 0
    for sd in model.subdomains:
        assert np.allclose(
            equation_subdomain_blocks[model.eq_all_subdomains.name][sd],
            offset + np.arange(sd.num_cells * dof_info_subdomain[GridEntity.cells]),
        )
        offset += sd.num_cells * dof_info_subdomain[GridEntity.cells]

    # Add a third equation, defined on the interface.
    # This time we switch the order of the interfaces. This should not matter for the
    # indices of the equation, since these are added in the order returned by
    # mdg.interfaces()
    equation_system.set_equation(
        model.eq_all_interfaces,
        equations_per_grid_entity=dof_info_interface,
    )
    equation_subdomain_blocks = (
        equation_system.equation_indexer.equation_image_space_composition
    )
    offset = 0
    for intf in model.interfaces:
        assert np.allclose(
            equation_subdomain_blocks[model.eq_all_interfaces.name][intf],
            offset + np.arange(intf.num_cells * dof_info_interface[GridEntity.cells]),
        )
        offset += intf.num_cells * dof_info_interface[GridEntity.cells]

    # Test updating an existing equation. Here we update the equation with a different
    # equation expression and a different number of degrees of freedom per cell. We
    # switch the order of the interfaces here as well, similarly to in the test above.
    mock_equation = model.intf_variable * model.intf_variable * model.intf_variable
    dof_all_interfaces = {GridEntity.cells: 3}
    equation_system.update_equation(
        new_equation=mock_equation,
        equation_name="eq_all_interfaces",
        equations_per_grid_entity=dof_all_interfaces,
    )
    equation_subdomain_blocks = (
        equation_system.equation_indexer.equation_image_space_composition
    )

    offset = 0
    for intf in model.interfaces:
        assert np.allclose(
            equation_subdomain_blocks[model.eq_all_interfaces.name][intf],
            offset + np.arange(intf.num_cells * dof_all_interfaces[GridEntity.cells]),
        )
        offset += intf.num_cells * dof_all_interfaces[GridEntity.cells]

    # Test updating an existing equation without changing grids and dof info.
    equation_system.update_equation(
        new_equation=mock_equation,
        equation_name="eq_all_interfaces",
    )
    equation_subdomain_blocks = (
        equation_system.equation_indexer.equation_image_space_composition
    )

    offset = 0
    for intf in model.interfaces:
        assert np.allclose(
            equation_subdomain_blocks[model.eq_all_interfaces.name][intf],
            offset + np.arange(intf.num_cells * dof_all_interfaces[GridEntity.cells]),
        )
        offset += intf.num_cells * dof_all_interfaces[GridEntity.cells]

    assert (
        list(equation_subdomain_blocks["eq_all_interfaces"].keys()) == model.interfaces
    )


def test_parse_variable_like(model: EquationSystemMockModel):
    """Test the private function _parse_variable_type().

    Thorough testing of this function allows us to assume variable parsing is properly
    handled for more advanced functions (e.g., assembly), thus there is no need to test
    those functions on varying input formats.

    We test only variables on subdomanis; the EquationSystem does not care about the
    type of domain for a variable.

    """
    equation_system = model.equation_system

    num_subdomains = model.mdg.num_subdomains()
    num_interfaces = model.mdg.num_interfaces()

    # First pass None, this should give all variables
    received_variables_1 = equation_system._parse_variable_type(None)
    assert len(received_variables_1) == len(set(received_variables_1))
    assert len(received_variables_1) == num_interfaces + num_subdomains + 2

    # Next pass an empty list. We expect to receive an empty list back
    received_variables_2 = equation_system._parse_variable_type([])
    assert len(received_variables_2) == 0

    # Pass a Variable, we should get it back
    var = model.sd_top_variable.sub_vars[0]
    received_variables_3 = equation_system._parse_variable_type([var])
    assert len(received_variables_3) == 1
    assert received_variables_3[0] == var

    # Pass an md-variable with one sub-variable, check we get back the sub-variable
    received_variables_4 = equation_system._parse_variable_type([model.sd_top_variable])
    assert len(received_variables_4) == 1
    assert received_variables_4[0] == model.sd_top_variable.sub_vars[0]

    # Send an md-variable with two sub-variables
    received_variables_5 = equation_system._parse_variable_type([model.sd_variable])
    assert len(received_variables_5) == len(model.sd_variable.sub_vars)
    assert all([var in received_variables_5 for var in model.sd_variable.sub_vars])

    # Send in the md-variable as a string, it should not make a difference
    received_variables_6 = equation_system._parse_variable_type(
        [model.name_sd_variable]
    )
    assert len(received_variables_6) == len(model.sd_variable.sub_vars)
    assert all([var in received_variables_6 for var in model.sd_variable.sub_vars])

    # Send in two md-variables
    received_variables_7 = equation_system._parse_variable_type(
        [model.sd_variable, model.sd_top_variable]
    )
    assert len(received_variables_7) == len(model.sd_variable.sub_vars) + len(
        model.sd_top_variable.sub_vars
    )

    # Send in a combination of string and variables.
    # NOTE: While this combination goes against the specification in the type hints
    # (list of strings and list of variable is permitted, combined lists are not), the
    # current implementation actually allows for this. If the implementation is changed,
    # the checks involving received_variables_8 and _9 can be deleted.
    received_variables_8 = equation_system._parse_variable_type(
        model.sd_variable.sub_vars + [model.name_sd_top_variable]
    )
    assert len(received_variables_8) == len(model.sd_variable.sub_vars) + len(
        model.sd_top_variable.sub_vars
    )

    # Send in a combination of string and md-variables
    received_variables_9 = equation_system._parse_variable_type(
        [model.sd_variable, model.name_sd_top_variable]
    )
    assert len(received_variables_9) == len(model.sd_variable.sub_vars) + len(
        model.sd_top_variable.sub_vars
    )


@pytest.mark.parametrize("ordered", [True, False])
def test_parse_variable_type_rejects_unknown_variable(
    model: EquationSystemMockModel, ordered: bool
) -> None:
    """An unregistered Variable must not be silently discarded in parsing variables.

    Parameters:
        model: The mock PorePy model. Needs to provide `mdg` and `equation_system`.
        ordered: The argument passed to `equation_system._parse_variable_type`. We test
            with both values, should not affect the test result.

    """
    unknown_variable = pp.ad.Variable(
        "unknown", {"cells": 1}, model.mdg.subdomains()[0]
    )

    with pytest.raises(ValueError, match="not registered"):
        model.equation_system._parse_variable_type([unknown_variable], ordered=ordered)


def test_parse_single_equation(model: EquationSystemMockModel):
    """Test the helper function for parsing a single equation.

    We consider only the equation posed on all subdomains, parsing of other equations
    should be identical.

    The test considers restrictions of the equation to subsets of its domains,
    and verifies that the returned index sets are correctly ordered.

    """
    equation_system = model.equation_system
    equation_indexer = equation_system.equation_indexer

    def get_restriction_dofs(equations: list[pp.ad.EquationOnDomain]):
        return np.concatenate(
            [
                equation_indexer.equation_image_space_composition[eq.name][eq.domain]
                for eq in equations
            ]
        )

    # Represent the equation both by its string and its operator form.
    # This could have been parametrized to the price of computational higher cost
    # (Pytest assembly overhead).
    for eq_or_name in [model.eq_all_subdomains, model.eq_all_subdomains.name]:
        # The equation name.
        name = eq_or_name if isinstance(eq_or_name, str) else eq_or_name.name
        eq = equation_system.equations[name]

        # First parse the equation as it is, without any restriction.
        # This should give back the full equation with no restriction.
        restriction_1 = equation_system._parse_equations([eq_or_name])
        # Four atomic equations (corresponding to this equation of 4 subdomains).
        assert len(restriction_1) == 4

        assert all(eq.name == eq_on_domain.name for eq_on_domain in restriction_1)

        # Next, restrict the equation to a single subdomain.
        restriction_2 = equation_system._parse_equations({eq_or_name: [model.sd_top]})
        assert len(restriction_2) == 1
        # The numbering of the subdomanis in the EquationSystem is the same as that of
        # the MixedDimensionalGrid, thus the indices associated with this subdomain
        # will be 0-offset.
        assert np.allclose(
            get_restriction_dofs(restriction_2), np.arange(model.sd_top.num_cells)
        )

        # Next, permute the subdomains before sending them in. All subdomains are
        # present, thus the indices should cover all cells in the md-grid. Moreover,
        # the EquationSystem will sort the subdomains according in the same order as
        # the MixedDimensionalGrid.subdomains() method, thus the indices should again
        # be linear.
        eq_def = {eq_or_name: model.subdomains[::-1]}
        restriction_3 = equation_system._parse_equations(eq_def)
        assert np.allclose(
            get_restriction_dofs(restriction_3),
            np.arange(model.mdg.num_subdomain_cells()),
        )


def test_parse_equations(model: EquationSystemMockModel):
    """Test the helper function for parsing equations.

    The test focuses on the functionality of EquationSystem._parse_equation()
    beyond the parsing of individual equations, which is tested in the method
    test_parse_single_equation(). That is, we test the parsing of multiple
    equations and check that the order of the returned equations is correct.
    In addition, we test that equations on empty domain are not parsed in the
    equation system.

    """
    equation_system = model.equation_system

    # All equations. The order is the same as that in the helper class
    # EquationSystemSetup.
    all_equation_names = model.all_equation_names
    all_equations = [
        equation_system.equations[eq_name] for eq_name in all_equation_names
    ]

    # First pass None. This should give as all equations on all subdomains.
    received_equations_1 = equation_system._parse_equations(None)

    # We expect all equations, thus names must be identical and ordered the same way.
    # The domains must be the same and also ordered the same way.
    expected_equations_1 = [
        pp.ad.EquationOnDomain(name=eq.name, domain=domain)
        for eq in all_equations
        for domain in eq.domains
    ]
    assert received_equations_1 == expected_equations_1

    # Next, pass the single subdomain and all subdomains, in that order.

    # Next, pass two equation names in the reversed order. We should receive the same
    # keys, but in canonical order.
    received_equations_2 = equation_system._parse_equations(
        [all_equation_names[1], all_equation_names[0]]
    )
    expected_equations_2 = [
        pp.ad.EquationOnDomain(name=name, domain=domain)
        for name in [all_equation_names[0], all_equation_names[1]]
        for domain in equation_system.equations[name].domains
    ]
    assert received_equations_2 == expected_equations_2

    # Send in the all_subdomains equation in both unrestricted and restricted form.
    # The restriction should override the unrestricted form.
    received_equations_3 = equation_system._parse_equations(
        {all_equation_names[0]: None, all_equation_names[0]: [model.sd_top]}
    )
    expected_equation_3 = [
        pp.ad.EquationOnDomain(name=all_equation_names[0], domain=model.sd_top)
    ]
    assert received_equations_3 == expected_equation_3

    # Add an equation on an empty domain to the equation system.
    model.add_equation_on_empty_domain()

    # Check that the empty equation is included in the equation system.
    assert "empty_equation" in equation_system.equations

    # Check that _parse_equations filters out equations on empty domain.
    assert "empty_equation" not in {
        eq.name for eq in equation_system._parse_equations()
    }


@pytest.mark.parametrize(
    "eq_names",
    [
        None,  # None gives the full system.
        [],  # An empty list will give a system with zero rows.
        ["eq_single_subdomain"],  # A single equation.
        ["eq_single_interface", "eq_all_subdomains"],  # Combination of two equations.
        # Combination of two equations, reversed order.
        ["eq_all_subdomains", "eq_single_interface"],
    ],
)
@pytest.mark.parametrize(
    "var_names",
    [
        None,  # None gives the full system.
        [],  # An empty list will give a system with zero columns.
        ["x"],  # A single variable.
        ["x", "w"],  # Combination of two variables.
        ["w", "x"],  # Combination of two variables, reversed order.
    ],
)
def test_assemble(
    model: EquationSystemMockModel,
    eq_names: list[str] | None,
    var_names: list[str] | None,
):
    """Test of functionality to assemble subsystems from an EquationSystem.

    The test is based on assembly of a subsystem and comparing this to a truth
    from assembly of the full system, and then explicitly dump rows and columns.

    We test combinations of 0 or more equations, together with 0 or more variables.
    Variables are only defined by strings; the alternative format of a variables is
    not considered, since the variables are only passed to EquationSystem.dofs_of()
    (via the method projection_to()), which is tested elsewhere.

    """
    equation_system = model.equation_system

    # Convert variable names into variables
    if var_names is None:
        variables = None
    else:
        variables = [
            var for var in model.equation_system.variables if var.name in var_names
        ]

    linear_system = equation_system.assemble(equations=eq_names, variables=var_names)
    A_sub = linear_system.matrix
    assert A_sub is not None
    b_sub = linear_system.rhs
    b_sub_only_rhs = equation_system.assemble(
        evaluate_jacobian=False, equations=eq_names, variables=var_names
    )

    # Check that the residual vector is the same regardless of whether the Jacobian
    # is evaluated or not.
    assert np.allclose(b_sub, b_sub_only_rhs)

    # Get active rows and columns. If eq_names is None, all rows should be included.
    # If equation list is set to empty list, no indices are included.
    # Otherwise, get the numbering from model.
    # Same logic for variables
    if eq_names is None:
        # If no equations are specified, all should be included.
        rows = np.arange(sum(model.eq_inds))
    elif len(eq_names) > 0:
        # Do not sort row indices - these are allowed to change.
        rows = np.sort(np.hstack([model.eq_ind(eq) for eq in eq_names]))
    else:
        # Equations are set to empty
        rows = []

    if var_names is None:
        # If no variables are specified, all should be included.
        cols = np.arange(model.A.shape[1])
    elif len(variables) > 0:
        # Sort variable indices
        cols = np.sort(np.hstack([model.dof_ind(var) for var in variables]))
    else:
        # Variables are set to empty
        cols = []

    # Uniquify columns, in case variables are represented many times.
    cols = np.unique(cols)

    # Check matrix and vector items
    assert np.allclose(b_sub, model.b[rows])
    assert pp.test_utils.arrays.compare_matrices(A_sub, model.A[rows][:, cols])

    # The restricted linear system comes with indexers local to it. Check that they
    # point to correct atomic variables and equations by comparing them to the globals.
    local_row_indexer = linear_system.equation_indexer
    local_col_indexer = linear_system.variable_indexer
    global_row_indexer = equation_system.equation_indexer
    global_col_indexer = equation_system.variable_indexer

    for eq, local_row_index in local_row_indexer.indices.items():
        global_row_index = global_row_indexer.indices[eq]

        for var, local_col_index in local_col_indexer.indices.items():
            global_col_index = global_col_indexer.indices[var]
            actual = A_sub[local_row_index][:, local_col_index]
            expected = model.A[global_row_index][:, global_col_index]
            assert pp.test_utils.arrays.compare_matrices(actual, expected)

            actual_rhs = b_sub[local_row_index]
            expected_rhs = model.b[global_row_index]
            assert pp.test_utils.arrays.compare_arrays(actual_rhs, expected_rhs)

    # The order of equations and variables should be the same, even if they were passed
    # to assembly in the reversed order.
    it = iter(global_row_indexer.indices)
    # This walks once over atomic equations in the global and local indexers (think
    # fast and slow pointers, respectively). There should be no permutation to succeed.
    assert all(x in it for x in local_row_indexer.indices)
    # Same for variables.
    it = iter(global_col_indexer.indices)
    assert all(x in it for x in local_col_indexer.indices)


@pytest.mark.parametrize(
    "eq_names",
    [
        None,  # None gives the full system.
        [],  # An empty list will give a system with zero rows.
        ["eq_single_subdomain"],  # A single equation.
        ["eq_single_interface", "eq_all_subdomains"],  # Combination of two equations.
        # Combination of two equations, reversed order.
        ["eq_all_subdomains", "eq_single_interface"],
    ],
)
@pytest.mark.parametrize(
    "var_names",
    [
        None,  # None gives the full system.
        [],  # An empty list will give a system with zero columns.
        ["x"],  # A single variable.
        ["x", "w"],  # Combination of two variables.
        ["w", "x"],  # Combination of two variables, reversed order.
    ],
)
def test_extract_subsystem(
    model: EquationSystemMockModel,
    eq_names: list[str] | None,
    var_names: list[str] | None,
):
    """Check functionality to extract subsystems from the EquationManager.

    The tests check that the expected variables and equations are present in
    the subsystem.

    We test combinations of 0 or more equations, together with 0 or more variables.
    Variables are only defined by strings; the alternative format of a variables is
    not considered, since the variables are only passed to EquationSystem.dofs_of()
    (via the method projection_to()) which is tested elsewhere.

    This test is run on a relatively limited set of equation-variable combinations
    (in particular compared to how the test was set up in the past). The reason is that
    parsing of variable and equation input is tested in separate tests, thus what
    remains is to test that given a set of equations and variables, the correct rows
    and columns are extracted from the full system.

    """
    equation_system = model.equation_system

    # Convert variable names into variables
    if var_names is None:
        var_names = []
    variables = [
        var for var in model.equation_system.variables if var.name in var_names
    ]

    new_manager = equation_system.SubSystem(eq_names, var_names)

    if eq_names is None:
        eq_names = model.all_equation_names

    # Check that the number of variables and equations are as expected
    assert len(new_manager.equations) == len(eq_names)

    # Check that the active / primary equations are present in the new manager
    for eq in new_manager.equations:
        assert eq in eq_names

    # Check that all variables were transferred to the new manager
    assert len(new_manager.variables) == len(variables)
    for var in new_manager.variables:
        assert var in variables


def test_assemble_ignores_empty_equations(model: EquationSystemMockModel):
    """Test that assemble() ignores equations defined on empty domains.

    An equation defined on an empty domain (no grids) should not contribute to the
    assembled system. After adding such an equation, assembling the system should
    produce the same matrix and residual vector as before the equation was added.

    """

    # Get the system of equations from the model.
    equation_system = model.equation_system

    # Store Baseline system matrix and vector.
    linear_system_ref = equation_system.assemble()
    A_ref = linear_system_ref.matrix
    assert A_ref is not None
    b_ref = linear_system_ref.rhs

    # Add equation on empty domain using the empty variable
    model.add_equation_on_empty_domain()

    # Check that the assembled system does not include the empty equation.
    linear_system = equation_system.assemble()
    A = linear_system.matrix
    assert A is not None
    b = linear_system.rhs
    assert np.allclose(b, b_ref)
    assert pp.test_utils.arrays.compare_matrices(A, A_ref)

    # Check bookkeeping does not suddenly include the empty equation.
    for eq_on_domain in linear_system.equation_indexer.indices:
        assert eq_on_domain.name != "empty_equation"
