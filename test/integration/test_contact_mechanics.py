#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various integration tests for contact mechanics.
"""
import numpy as np
import unittest
import scipy.sparse.linalg as spla

import porepy as pp


class TestContactMechanics(unittest.TestCase):
    def _solve(self, model):
        sol = solve_contact_mechanics(model)
        gb = model.gb

        nd = gb.dim_max()

        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][model.surface_variable]
        contact_force = d_1[pp.STATE][model.contact_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((2, -1), order="F")

        contact_force = contact_force.reshape((2, -1), order="F")

        return u_mortar_local_decomposed, contact_force

    def test_pull_top_positive_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=0.001)

        u_mortar, contact_force = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_pull_bottom_positive_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=-0.001, ux_top=0, uy_top=0)

        u_mortar, contact_force = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_push_top_zero_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=-0.001)

        u_mortar, contact_force = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

    def test_push_bottom_zero_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0.001, ux_top=0, uy_top=0)

        u_mortar, contact_force = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))


def solve_contact_mechanics(setup):
    """
    Function for solving linear elasticity with a non-linear Coulomb contact.

    There are some assumtions on the variable and discretization names given to the
    grid bucket:
        'u': The displacement variable
        'lam': The mortar variable
        'mpsa': The mpsa discretization

    In addition to the standard parameters for mpsa we also require the following
    under the contact mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.

    Arguments:
        setup: A setup class with methods:
                set_parameters(g, data_node, mg, data_edge): assigns data to grid bucket.
                    Returns the keyword for the linear elastic parameters and a keyword
                    for the contact mechanics parameters.
                create_grid(): Create and return the grid bucket
                initial_condition(): Returns initial guess for 'u' and 'lam'.
            and attributes:
                out_name(): returns a string. The data from the simulation will be
                written to the file 'res_data/' + setup.out_name and the vtk files to
                'res_plot/' + setup.out_name
    """
    # Define mixed-dimensional grid
    gb = setup.create_grid()

    # Extract the grids we use
    ambient_dim = gb.dim_max()
    # Pick up grid of highest dimension - there should be a single one of these
    g_max = gb.grids_of_dimension(ambient_dim)[0]
    # Obtain simulation data for the grid, and the edge (in the GridBucket
    # sense) between the grid and itself, that is, the link over the fracture.
    # set simulation parameters
    setup.set_parameters(gb)

    # Define rotations
    pp.contact_conditions.set_projections(gb)

    # Set up assembler and discretize
    # setup.discretize(gb)

    assembler = pp.Assembler(gb)
    setup.assembler = assembler

    # prepare for iteration
    setup.initial_condition(assembler)

    u0 = gb.node_props(g_max)[pp.STATE][setup.displacement_variable]
    errors = []

    def l2_error_cell(g, u, uref=None):
        if uref is None:
            norm = np.reshape(u ** 2, (g.dim, g.num_cells), order="F") * g.cell_volumes
        else:
            norm = (
                np.reshape((u - uref) ** 2, (g.dim, g.num_cells), order="F")
                * g.cell_volumes
            )
        return np.sum(norm)

    counter_newton = 0
    converged_newton = False
    max_newton = 15

    assembler.discretize()

    while counter_newton <= max_newton and not converged_newton:
        counter_newton += 1
        # Calculate numerical friction bound used in the contact condition
        # Clip the bound to be non-negative

        assembler.discretize(term_filter=setup.friction_coupling_term)

        # assembler.discretize(term_filter=)

        # Re-discretize and solve
        A, b = assembler.assemble_matrix_rhs()

        sol = spla.spsolve(A, b)

        # Split solution into displacement variable and mortar variable
        assembler.distribute_variable(sol)

        for g, d in gb:
            if g.dim == ambient_dim:
                d[pp.STATE][setup.displacement_variable] = d[
                    setup.displacement_variable
                ]
            elif g.dim == ambient_dim - 1:
                d[pp.STATE][setup.contact_variable] = d[setup.contact_variable]

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            if mg.dim == ambient_dim - 1:
                d[pp.STATE][setup.surface_variable] = d[setup.surface_variable]

        u = gb.node_props(g_max)[setup.displacement_variable]

        solution_norm = l2_error_cell(g_max, u)
        iterate_difference = l2_error_cell(g_max, u, u0)

        if iterate_difference / solution_norm < 1e-10:
            converged_newton = True

        errors.append(np.sum((u - u0) ** 2) / np.sum(u ** 2))

        # Prepare for next iteration
        u0 = u

    if counter_newton > max_newton and not converged_newton:
        raise ValueError("Newton iterations did not converge")

    return sol


class SetupContactMechanics(unittest.TestCase):
    def __init__(self, ux_bottom, uy_bottom, ux_top, uy_top):
        self.ux_bottom = ux_bottom
        self.uy_bottom = uy_bottom
        self.ux_top = ux_top
        self.uy_top = uy_top

        self.displacement_variable = "u"
        self.surface_variable = "mortar_u"
        self.contact_variable = "contact_force"

        self.friction_parameter_key = "friction"
        self.surface_parameter_key = "surface"
        self.mechanics_parameter_key = "mechanics"

        self.friction_coupling_term = "contact_conditions"

        self.mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }

    def create_grid(self, rotate_fracture=False):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        if rotate_fracture:
            self.frac_pts = np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            self.frac_pts = np.array([[0.3, 0.7], [0.5, 0.5]])
        frac_edges = np.array([[0], [1]])

        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        self.gb = gb
        return gb

    def set_parameters(self, gb):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """

        ambient_dim = gb.dim_max()

        for g, d in gb:
            if g.dim == ambient_dim:
                # Rock parameters
                rock = pp.Granite()
                lam = rock.LAMBDA * np.ones(g.num_cells) / pp.GIGA
                mu = rock.MU * np.ones(g.num_cells) / pp.GIGA

                k = pp.FourthOrderTensor(g.dim, mu, lam)

                # Define boundary regions
                top = g.face_centers[g.dim - 1] > np.max(g.nodes[1]) - 1e-9
                bot = g.face_centers[g.dim - 1] < np.min(g.nodes[1]) + 1e-9

                # Define boundary condition on sub_faces
                bc = pp.BoundaryConditionVectorial(g, top + bot, "dir")
                frac_face = g.tags["fracture_faces"]
                bc.is_neu[:, frac_face] = False
                bc.is_dir[:, frac_face] = True

                # Set the boundary values
                u_bc = np.zeros((g.dim, g.num_faces))

                u_bc[0, bot] = self.ux_bottom
                u_bc[1, bot] = self.uy_bottom
                u_bc[0, top] = self.ux_top
                u_bc[1, top] = self.uy_top

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": u_bc.ravel("F"),
                        "source": 0,
                        "fourth_order_tensor": k,
                    },
                )

            elif g.dim == 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.friction_parameter_key,
                    {"friction_coefficient": friction},
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            pp.initialize_data(mg, d, self.friction_parameter_key, {})

        # Define discretization
        # For the 2D domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(self.mechanics_parameter_key)

        empty_discr = pp.VoidDiscretization(
            self.friction_parameter_key, ndof_cell=ambient_dim
        )

        coloumb = pp.ColoumbContact(self.friction_parameter_key, ambient_dim)

        # Define discretization parameters

        for g, d in gb:
            if g.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": ambient_dim}
                }
                d[pp.DISCRETIZATION] = {self.displacement_variable: {"mpsa": mpsa}}
            elif g.dim == ambient_dim - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_variable: {"cells": ambient_dim}
                }
                d[pp.DISCRETIZATION] = {self.contact_variable: {"empty": empty_discr}}
            else:
                d[pp.PRIMARY_VARIABLES] = {}

        # And define a Robin condition on the mortar grid
        contact = pp.PrimalContactCoupling(self.friction_parameter_key, mpsa, coloumb)

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            if g_h.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    self.surface_variable: {"cells": ambient_dim}
                }

                #                d[pp.DISCRETIZATION] = {self.surface_variable: {'surface_mpsa': mpsa_surface}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    self.friction_coupling_term: {
                        g_h: (self.displacement_variable, "mpsa"),
                        g_l: (self.contact_variable, "empty"),
                        (g_h, g_l): (self.surface_variable, contact),
                    }
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

    def discretize(self, gb):
        g_max = gb.grids_of_dimension(gb.dim_max())[0]
        d = gb.node_props(g_max)

        mpsa = d[pp.DISCRETIZATION][self.displacement_variable]["mpsa"]
        mpsa.discretize(g_max, d)

    def initial_condition(self, assembler):
        """
        Initial guess for Newton iteration.
        """
        gb = assembler.gb

        ambient_dimension = gb.dim_max()

        for g, d in gb:
            d[pp.STATE] = {}
            if g.dim == ambient_dimension:
                # Initialize displacement variable
                ind = assembler.dof_ind(g, self.displacement_variable)
                d[pp.STATE][self.displacement_variable] = np.zeros_like(ind)

            elif g.dim == ambient_dimension - 1:
                # Initialize contact variable
                ind = assembler.dof_ind(g, self.contact_variable)

                traction = np.vstack(
                    (np.zeros(g.num_cells), -100 * np.ones(g.num_cells))
                ).ravel(order="F")

                d[pp.STATE][self.contact_variable] = traction

        for e, d in gb.edges():
            d[pp.STATE] = {}

            mg = d["mortar_grid"]

            if mg.dim == 1:
                ind = assembler.dof_ind(e, self.surface_variable)
                d[pp.STATE][self.surface_variable] = np.zeros_like(ind)

    def _set_friction_coefficient(self, g):
        friction_coefficient = 0.5 * np.ones(g.num_cells)
        return friction_coefficient


if __name__ == "__main__":
    unittest.main()
