"""
Tests for components of the Ad machinery. Specifically, the tests cover:
    * Ad representation of variables.
    * The forward Ad machinery as implemented in pp.ad.Ad_array.
See also test_grid_operators.
"""

import unittest

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array


def test_ad_variable_wrappers():
    # Tests that the wrapping of Ad variables, including previous iterates
    # and time steps, are carried out correctly.
    # See also test_variable_combinations, which specifically tests evaluation of
    # variables in a setting of multiple variables, including merged variables.

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    state_map = {}
    iterate_map = {}

    state_map_2, iterate_map_2 = {}, {}

    var = "foo"
    var2 = "bar"

    mortar_var = "mv"

    def _compare_ad_objects(a, b):
        va, ja = a.val, a.jac
        vb, jb = b.val, b.jac

        assert np.allclose(va, vb)
        assert ja.shape == jb.shape
        d = ja - jb
        if d.data.size > 0:
            assert np.max(np.abs(d.data)) < 1e-10

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        data[pp.PRIMARY_VARIABLES] = {var: {"cells": num_dofs}}

        val_state = np.random.rand(sd.num_cells * num_dofs)
        val_iterate = np.random.rand(sd.num_cells * num_dofs)

        data[pp.STATE] = {var: val_state, pp.ITERATE: {var: val_iterate}}
        state_map[sd] = val_state
        iterate_map[sd] = val_iterate

        # Add a second variable to the 2d grid, just for the fun of it
        if sd.dim == 2:
            data[pp.PRIMARY_VARIABLES][var2] = {"cells": 1}
            val_state = np.random.rand(sd.num_cells)
            val_iterate = np.random.rand(sd.num_cells)
            data[pp.STATE][var2] = val_state
            data[pp.STATE][pp.ITERATE][var2] = val_iterate
            state_map_2[sd] = val_state
            iterate_map_2[sd] = val_iterate

    for intf, data in mdg.interfaces(return_data=True):
        if intf.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        data[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": num_dofs}}

        val_state = np.random.rand(intf.num_cells * num_dofs)
        val_iterate = np.random.rand(intf.num_cells * num_dofs)

        data[pp.STATE] = {mortar_var: val_state, pp.ITERATE: {mortar_var: val_iterate}}
        state_map[intf] = val_state
        iterate_map[intf] = val_iterate

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    # Manually assemble state and iterate
    true_state = np.zeros(dof_manager.num_dofs())
    true_iterate = np.zeros(dof_manager.num_dofs())

    # Also a state array that differs from the stored iterates
    double_iterate = np.zeros(dof_manager.num_dofs())

    for (g, v) in dof_manager.block_dof:
        inds = dof_manager.grid_and_variable_to_dofs(g, v)
        if v == var2:
            true_state[inds] = state_map_2[g]
            true_iterate[inds] = iterate_map_2[g]
            double_iterate[inds] = 2 * iterate_map_2[g]
        else:
            true_state[inds] = state_map[g]
            true_iterate[inds] = iterate_map[g]
            double_iterate[inds] = 2 * iterate_map[g]

    subdomains = [
        mdg.subdomains(dim=2)[0],
        *mdg.subdomains(dim=1),
        mdg.subdomains(dim=0)[0],
    ]

    # Generate merged variables via the EquationManager.
    var_ad = eq_manager.merge_variables([(g, var) for g in subdomains])

    # Check equivalence between the two approaches to generation.

    # Check that the state is correctly evaluated.
    inds_var = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(g, var) for g in subdomains]
    )
    assert np.allclose(
        true_iterate[inds_var], var_ad.evaluate(dof_manager, true_iterate).val
    )

    # Check evaluation when no state is passed to the parser, and information must
    # instead be glued together from the MixedDimensionalGrid
    assert np.allclose(true_iterate[inds_var], var_ad.evaluate(dof_manager).val)

    # Evaluate the equation using the double iterate
    assert np.allclose(
        2 * true_iterate[inds_var], var_ad.evaluate(dof_manager, double_iterate).val
    )

    # Represent the variable on the previous time step. This should be a numpy array
    prev_var_ad = var_ad.previous_timestep()
    prev_evaluated = prev_var_ad.evaluate(dof_manager)
    assert isinstance(prev_evaluated, np.ndarray)
    assert np.allclose(true_state[inds_var], prev_evaluated)

    # Also check that state values given to the ad parser are ignored for previous
    # values
    assert np.allclose(
        prev_evaluated, prev_var_ad.evaluate(dof_manager, double_iterate)
    )

    ## Next, test edge variables. This should be much the same as the grid variables,
    # so the testing is less thorough.
    # Form an edge variable, evaluate this
    edge_list = [intf for intf in mdg.interfaces()]
    var_edge = eq_manager.merge_variables([(e, mortar_var) for e in edge_list])

    edge_inds = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(e, mortar_var) for e in edge_list]
    )
    assert np.allclose(
        true_iterate[edge_inds], var_edge.evaluate(dof_manager, true_iterate).val
    )

    # Finally, test a single variable; everything should work then as well
    g = mdg.subdomains(dim=2)[0]
    v1 = eq_manager.variable(g, var)
    v2 = eq_manager.variable(g, var2)

    ind1 = dof_manager.grid_and_variable_to_dofs(g, var)
    ind2 = dof_manager.grid_and_variable_to_dofs(g, var2)

    assert np.allclose(true_iterate[ind1], v1.evaluate(dof_manager, true_iterate).val)
    assert np.allclose(true_iterate[ind2], v2.evaluate(dof_manager, true_iterate).val)

    v1_prev = v1.previous_timestep()
    assert np.allclose(true_state[ind1], v1_prev.evaluate(dof_manager, true_iterate))


@pytest.mark.parametrize(
    "grids",
    [
        [pp.CartGrid(np.array([4, 1]))],
        [pp.CartGrid(np.array([4, 1])), pp.CartGrid(np.array([2, 2]))],
    ],
)
@pytest.mark.parametrize(
    "variables",
    [["foo"], ["foo", "bar"]],
)
def test_variable_combinations(grids, variables):
    # Test combinations of variables, and merged variables, on different grids.
    # Main check is if Jacobian matrices are of the right size.

    # Make MixedDimensionalGrid, populate with necessary information
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(grids)
    for sd, data in mdg.subdomains(return_data=True):
        data[pp.STATE] = {}
        data[pp.PRIMARY_VARIABLES] = {}
        for var in variables:
            data[pp.PRIMARY_VARIABLES].update({var: {"cells": 1}})
            data[pp.STATE][var] = np.random.rand(sd.num_cells)

    # Ad boilerplate
    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    # Standard Ad variables
    ad_vars = [eq_manager.variable(g, var) for g in grids for var in variables]
    # Merge variables over all grids
    merged_vars = []
    for var in variables:
        gv = [(g, var) for g in grids]
        merged_vars.append(eq_manager.merge_variables(gv))

    # First check of standard variables. If this fails, something is really wrong
    for sd in grids:
        data = mdg.subdomain_data(sd)
        for var in ad_vars:
            if sd == var._g:
                expr = var.evaluate(dof_manager)
                # Check that the size of the variable is correct
                assert np.allclose(expr.val, data[pp.STATE][var._name])
                # Check that the Jacobian matrix has the right number of columns
                assert expr.jac.shape[1] == dof_manager.num_dofs()

    # Next, check that merged variables are handled correctly.
    for var in merged_vars:
        expr = var.evaluate(dof_manager)
        vals = []
        for sub_var in var.sub_vars:
            vals.append(mdg.subdomain_data(sub_var._g)[pp.STATE][sub_var._name])

        assert np.allclose(expr.val, np.hstack([v for v in vals]))
        assert expr.jac.shape[1] == dof_manager.num_dofs()

    # Finally, check that the size of the Jacobian matrix is correct when combining
    # variables (this will cover both variables and merged variable with the same name,
    # and with different name).
    for sd in grids:
        for var in ad_vars:
            nc = var.size()
            cols = np.arange(nc)
            data = np.ones(nc)
            for mv in merged_vars:
                nr = mv.size()

                # The variable must be projected to the full set of grid for addition
                # to be meaningful. This requires a bit of work.
                sv_size = np.array([sv.size() for sv in mv.sub_vars])
                mv_grids = [sv._g for sv in mv.sub_vars]
                ind = mv_grids.index(var._g)
                offset = np.hstack((0, np.cumsum(sv_size)))[ind]
                rows = offset + np.arange(nc)
                P = pp.ad.Matrix(sps.coo_matrix((data, (rows, cols)), shape=(nr, nc)))

                eq = eq = mv + P * var
                expr = eq.evaluate(dof_manager)
                # Jacobian matrix size is set according to the dof manager,
                assert expr.jac.shape[1] == dof_manager.num_dofs()


def test_ad_discretization_class():
    # Test of the mother class of all discretizations (pp.ad.Discretization)

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    subdomains = [g for g in mdg.subdomains()]
    sub_list = subdomains[:2]

    # Make two Mock discretizations, with different keywords
    key = "foo"
    sub_key = "bar"
    discr = _MockDiscretization(key)
    sub_discr = _MockDiscretization(sub_key)

    # Ad wrappers
    # This is mimics the old init of Discretization, before it was decided to
    # make that class semi-ABC. Still checks the wrap method
    discr_ad = pp.ad.Discretization()
    discr_ad.subdomains = subdomains
    discr_ad._discretization = discr
    pp.ad._ad_utils.wrap_discretization(discr_ad, discr, subdomains)
    sub_discr_ad = pp.ad.Discretization()
    sub_discr_ad.subdomains = sub_list
    sub_discr_ad._discretization = sub_discr
    pp.ad._ad_utils.wrap_discretization(sub_discr_ad, sub_discr, sub_list)

    # values
    known_val = np.random.rand(len(subdomains))
    known_sub_val = np.random.rand(len(sub_list))

    # Assign a value to the discretization matrix, with the right key
    for vi, sd in enumerate(subdomains):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES] = {key: {"foobar": known_val[vi]}}

    # Same with submatrix
    for vi, sd in enumerate(sub_list):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES].update(
            {sub_key: {"foobar": known_sub_val[vi]}}
        )

    # Compare values under parsing. Note we need to pick out the diagonal, due to the
    # way parsing make block matrices.
    assert np.allclose(known_val, discr_ad.foobar.parse(mdg).diagonal())
    assert np.allclose(known_sub_val, sub_discr_ad.foobar.parse(mdg).diagonal())


## Below are helpers for tests of the Ad wrappers.


def _compare_matrices(m1, m2):
    if isinstance(m1, pp.ad.Matrix):
        m1 = m1._mat
    if isinstance(m2, pp.ad.Matrix):
        m2 = m2._mat
    if m1.shape != m2.shape:
        return False
    d = m1 - m2
    if d.data.size > 0:
        if np.max(np.abs(d.data)) > 1e-10:
            return False
    return True


def _list_ind_of_grid(subdomains, g):
    for i, gl in enumerate(subdomains):
        if g == gl:
            return i

    raise ValueError("grid is not in list")


class _MockDiscretization:
    def __init__(self, key):
        self.foobar_matrix_key = "foobar"
        self.not_matrix_keys = "failed"

        self.keyword = key


class AdArrays(unittest.TestCase):
    """Tests for the implementation of the main Ad array class,
    that is, the functionality needed for the forward Ad operations.
    """

    def test_add_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(-10, 0)
        c = a + b
        self.assertTrue(c.val == -9 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == -10 and b.jac == 0)

    def test_add_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a + b
        self.assertTrue(np.allclose(c.val, 13) and np.allclose(c.jac, 4.0))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_add_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a + b
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and np.allclose(a.jac, 2))
        self.assertTrue(b == 3)

    def test_add_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b + a
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(3, 0)
        c = a - b
        self.assertTrue(c.val == -2 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == 3 and a.jac == 0)

    def test_sub_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a - b
        self.assertTrue(np.allclose(c.val, -5) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_sub_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a - b
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b - a
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_mul_scal_ad_scal(self):
        a = Ad_array(3, 0)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 0)
        self.assertTrue(a.val == 3 and a.jac == 0)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_scal(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_var(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, -4)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == -6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == -4)

    def test_mul_ad_var_scal(self):
        a = Ad_array(3, 3)
        b = 3
        c = a * b
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_scar_ad_var(self):
        a = Ad_array(3, 3)
        b = 3
        c = b * a
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_ad_var_mat(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = x * A
        sol = np.array([30, 36, 42])
        jac = np.diag([3, 2, 1]) * A

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )
        self.assertTrue(np.all(A == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    def test_advar_mul_vec(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = np.array([1, 3, 10])
        f = x * A
        sol = np.array([1, 6, 30])
        jac = np.diag([3, 6, 10])

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )

    def test_advar_m_mul_vec_n(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        vec = np.array([1, 2])
        R = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 0]]))
        y = R * x
        z = y * vec
        Jy = np.array([[1, 0, 3], [0, 2, 0]])
        Jz = np.array([[1, 0, 3], [0, 4, 0]])
        self.assertTrue(np.all(y.val == [4, 2]))
        self.assertTrue(np.sum(y.full_jac().A - Jy) == 0)
        self.assertTrue(np.all(z.val == [4, 4]))
        self.assertTrue(np.sum(z.full_jac().A - Jz) == 0)

    def test_mul_sps_advar(self):
        J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        x = Ad_array(np.array([1, 2, 3]), J)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = A * x

        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all(f.jac == A * J.A))

    def test_mul_advar_vectors(self):
        Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), Ja)
        b = Ad_array(np.array([1, 1, 1]), Jb)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A * a + b

        self.assertTrue(np.all(f.val == [15, 33, 51]))
        self.assertTrue(np.sum(f.full_jac() != A * Ja + Jb) == 0)
        self.assertTrue(
            np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]])))
            == 0
        )
        self.assertTrue(
            np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
            == 0
        )

    def test_power_advar_scalar(self):
        a = Ad_array(2, 3)
        b = a**2
        self.assertTrue(b.val == 4 and b.jac == 12)

    def test_power_advar_advar(self):
        a = Ad_array(4, 4)
        b = Ad_array(-8, -12)
        c = a**b
        jac = -(2 + 3 * np.log(4)) / 16384
        self.assertTrue(np.allclose(c.val, 4**-8) and np.allclose(c.jac, jac))

    def test_rpower_advar_scalar(self):
        # Make an Ad_array with value 2 and derivative 3.
        a = Ad_array(2, 3)
        b = 2**a
        self.assertTrue(b.val == 4 and b.jac == 12 * np.log(2))

        c = 2 ** (-a)
        self.assertTrue(c.val == 1 / 4 and c.jac == 2 ** (-2) * np.log(2) * (-3))

    def test_rpower_advar_vector_scalar(self):
        J = sps.csc_matrix(np.array([[1, 2], [2, 3], [0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), J)
        b = 3**a
        bJac = np.array(
            [
                [3 * np.log(3) * 1, 3 * np.log(3) * 2],
                [9 * np.log(3) * 2, 9 * np.log(3) * 3],
                [27 * np.log(3) * 0, 27 * np.log(3) * 1],
            ]
        )

        self.assertTrue(np.all(b.val == [3, 9, 27]))
        self.assertTrue(np.all(b.jac.A == bJac))

    def test_div_advar_scalar(self):
        a = Ad_array(10, 6)
        b = 2
        c = a / b
        self.assertTrue(c.val == 5, c.jac == 2)

    def test_div_advar_advar(self):
        # a = x ^ 3: b = x^2: x = 2
        a = Ad_array(8, 12)
        b = Ad_array(4, 4)
        c = a / b
        self.assertTrue(c.val == 2 and np.allclose(c.jac, 1))

    def test_full_jac(self):
        J = np.array(
            [
                [1, 3, 5, 1, 2],
                [1, 5, 1, 2, 5],
                [6, 2, 4, 6, 0],
                [2, 4, 1, 9, 9],
                [6, 2, 1, 45, 2],
            ]
        )

        a = Ad_array(np.array([1, 2, 3, 4, 5]), J.copy())  # np.array([J1, J2]))

        self.assertTrue(np.sum(a.full_jac() != J) == 0)

    def test_copy_scalar(self):
        a = Ad_array(1, 0)
        b = a.copy()
        self.assertTrue(a.val == b.val)
        self.assertTrue(a.jac == b.jac)
        a.val = 2
        a.jac = 3
        self.assertTrue(b.val == 1)
        self.assertTrue(b.jac == 0)

    def test_copy_vector(self):
        a = Ad_array(np.ones((3, 1)), np.ones((3, 1)))
        b = a.copy()
        self.assertTrue(np.allclose(a.val, b.val))
        self.assertTrue(np.allclose(a.jac, b.jac))
        a.val[0] = 3
        a.jac[2] = 4
        self.assertTrue(np.allclose(b.val, np.ones((3, 1))))
        self.assertTrue(np.allclose(b.jac, np.ones((3, 1))))


if __name__ == "__main__":
    unittest.main()
