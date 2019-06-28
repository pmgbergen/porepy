#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for contact mechanics with pressure coupling.

We have the full Biot equations in the matrix, and mass conservation and contact
conditions in the non-intersecting fracture(s). For the contact mechanical part of this
test, please refer to test_contact_mechanics.
"""
import numpy as np
import scipy.sparse as sps
import unittest
import scipy.sparse.linalg as spla

import porepy as pp
from test_contact_mechanics import SetupContactMechanics
from porepy.utils.derived_discretizations import implicit_euler as discretizations


class TestContactMechanicsBiot(unittest.TestCase):
    def _solve(self, model):
        _ = solve_biot(model)
        gb = model.gb

        nd = gb.dim_max()

        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][model.surface_variable]
        contact_force = d_1[pp.STATE][model.contact_variable]
        fracture_pressure = d_1[pp.STATE][model.scalar_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((2, -1), order="F")

        contact_force = contact_force.reshape((2, -1), order="F")

        return u_mortar_local_decomposed, contact_force, fracture_pressure

    def test_pull_top_positive_opening(self):

        model = SetupContactMechanicsBiot(
            ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_pull_bottom_positive_opening(self):

        model = SetupContactMechanicsBiot(
            ux_bottom=0, uy_bottom=-0.001, ux_top=0, uy_top=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_push_top_zero_opening(self):

        model = SetupContactMechanicsBiot(
            ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=-0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_push_bottom_zero_opening(self):

        model = SetupContactMechanicsBiot(
            ux_bottom=0, uy_bottom=0.001, ux_top=0, uy_top=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_positive_fracture_pressure_positive_opening(self):

        model = SetupContactMechanicsBiot(
            ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=0, source_value=0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Fracture pressure is positive
        self.assertTrue(np.all(fracture_pressure > 1e-7))


def solve_biot(setup):
    """
    Function for solving the time dependent Biot equations with a non-linear Coulomb
    contact condition on the fractures.

    See solve_contact_mechanics in test_contact_mechanics for assumptions on the
    mechanical parameters set. In addition, we require parameters and discretizations
    for the pressure and coupling terms, see SetupContactMechanicsBiot.

    Arguments:
        setup: A setup class with methods:
                set_parameters(): assigns data for the contact mechanics problem. See
                    test_contact_mechanics.
                set_scalar_parameters(): assign data for the scalar parameter, here
                    pressure
                create_grid(): Create and return the grid bucket
                initial_condition(): Returns initial conditions.
            and attributes:
                end_time: End time time of simulation.
    """
    gb = setup.create_grid()
    # Extract the grids we use
    dim = gb.dim_max()
    g_max = gb.grids_of_dimension(dim)[0]
    d_max = gb.node_props(g_max)

    # set parameters
    setup.set_parameters(gb)
    setup.set_scalar_parameters()
    setup.initial_condition()

    # Shorthand for some parameters
    dt = d_max[pp.PARAMETERS][setup.scalar_parameter_key]["time_step"]
    setup.assign_discretisations()
    # Define rotations
    pp.contact_conditions.set_projections(gb)
    # Set up assembler and get initial condition
    assembler = pp.Assembler(gb)

    u0 = d_max[pp.STATE][setup.displacement_variable].reshape((dim, -1), order="F")

    # Discretize with the Biot class
    setup.discretize_biot(gb)

    def l2_error_cell(g, u, uref=None):
        if uref is None:
            norm = np.reshape(u ** 2, (g.dim, g.num_cells), order="F") * g.cell_volumes
        else:
            norm = (
                np.reshape((u - uref) ** 2, (g.dim, g.num_cells), order="F")
                * g.cell_volumes
            )
        return np.sum(norm)

    t = 0.0
    T = setup.end_time
    k = 0
    times = []
    newton_it = 0

    while t < T:
        t += dt
        k += 1
        times.append(t)
        # Prepare for Newton iteration
        counter_newton = 0
        converged_newton = False
        max_newton = 12
        while counter_newton <= max_newton and not converged_newton:
            counter_newton += 1
            # Rediscretize the contact conditions (remaining discretizations assumed
            # constant in time).
            assembler.discretize(term_filter=setup.friction_coupling_term)

            # Reassemble and solve
            A, b = assembler.assemble_matrix_rhs()
            sol = sps.linalg.spsolve(A, b)

            # Split solution in the different variables
            assembler.distribute_variable(sol)
            u = d_max[pp.STATE][setup.displacement_variable].reshape(
                (dim, -1), order="F"
            )
            # Calculate the errorsolution_norm = l2_error_cell(g_max, u)
            solution_norm = l2_error_cell(g_max, u)
            iterate_difference = l2_error_cell(g_max, u, u0)
            if iterate_difference / solution_norm < 1e-10:
                converged_newton = True

            # Prepare for next Newton iteration
            u0 = u
            newton_it += 1

    return sol


class SetupContactMechanicsBiot(SetupContactMechanics):
    def __init__(self, ux_bottom, uy_bottom, ux_top, uy_top, source_value=0):
        super().__init__(ux_bottom, uy_bottom, ux_top, uy_top)
        self.scalar_variable = "p"
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"
        self.pressure_source_value = source_value

        self.scalar_scale = 1
        self.length_scale = 1
        self.time_step = 1
        self.end_time = 1

    def biot_alpha(self):
        return 1

    def sources(self, g, key, t=0):
        if key == self.mechanics_parameter_key:
            values = np.zeros((g.dim, g.num_cells))
            values = values.ravel("F")
        elif key == self.scalar_parameter_key:
            if g.dim == 2:
                values = np.zeros(g.num_cells)
            else:
                values = (
                    self.pressure_source_value * self.time_step * np.ones(g.num_cells)
                )
        else:
            raise ValueError("No BC values implemented for keyword " + str(key))
        return values

    def set_scalar_parameters(self):
        gb = self.gb
        ambient_dim = gb.dim_max()
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        k_frac = 100
        a = 1e-3
        for g, d in gb:
            # Define boundary regions
            top = g.face_centers[ambient_dim - 1] > self.box["ymax"] - 1e-9
            bot = g.face_centers[ambient_dim - 1] < self.box["ymin"] + 1e-9

            # Define boundary condition
            bc = pp.BoundaryCondition(g, top + bot, "dir")
            bc_values = np.zeros(g.num_faces)

            if g.dim == ambient_dim:
                kxx = 1 * tensor_scale * np.ones(g.num_cells)
                K = pp.SecondOrderTensor(ambient_dim, kxx)
                alpha = self.biot_alpha()
                mass_weight = 1e-1

                pp.initialize_data(
                    g,
                    d,
                    self.scalar_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "mass_weight": mass_weight,
                        "aperture": np.ones(g.num_cells),
                        "biot_alpha": alpha,
                        "time_step": self.time_step,
                        "source": self.sources(g, self.scalar_parameter_key, 0),
                        "second_order_tensor": K,
                    },
                )
                # Add Biot alpha and time step to the mechanical parameters
                d[pp.PARAMETERS].update_dictionaries(
                    self.mechanics_parameter_key,
                    {"biot_alpha": self.biot_alpha(), "time_step": self.time_step},
                )

            elif g.dim == ambient_dim - 1:
                kxx = k_frac * tensor_scale * np.ones(g.num_cells)
                K = pp.SecondOrderTensor(ambient_dim, kxx)
                mass_weight = 1e-1  # compressibility
                cross_sectional_area = a * np.ones(g.num_cells)
                pp.initialize_data(
                    g,
                    d,
                    self.scalar_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "mass_weight": mass_weight,
                        "source": self.sources(g, self.scalar_parameter_key, 0),
                        "second_order_tensor": K,
                        "aperture": cross_sectional_area,
                        "time_step": self.time_step,
                    },
                )
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"biot_alpha": 1},  # Enters in the div d term for the fracture
                )
            else:
                # No intersections yet
                raise NotImplementedError

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            k = k_frac * tensor_scale * np.ones(mg.num_cells)
            k_n = 2 / a * k
            data_edge = pp.initialize_data(
                e, data_edge, self.scalar_parameter_key, {"normal_diffusivity": k_n}
            )

    def assign_discretisations(self):
        gb = self.gb
        ambient_dim = gb.dim_max()
        key_s, key_m = self.scalar_parameter_key, self.mechanics_parameter_key
        var_s, var_d = self.scalar_variable, self.displacement_variable
        # Define discretization
        # For the 2D domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(key_m)

        empty_discr = pp.VoidDiscretization(
            self.friction_parameter_key, ndof_cell=ambient_dim
        )
        diff_disc_s = discretizations.ImplicitMpfa(key_s)
        mass_disc_s = discretizations.ImplicitMassMatrix(key_s, variable=var_s)
        source_disc_s = pp.ScalarSource(key_s)
        div_u_disc = pp.DivU(
            key_m, variable=var_d, mortar_variable=self.surface_variable
        )
        grad_p_disc = pp.GradP(key_m)
        div_u_disc_frac = pp.DivU(
            key_m, variable=var_d, mortar_variable=self.surface_variable
        )
        stabilisiation_disc_s = pp.BiotStabilization(key_s, variable=var_s)
        coloumb = pp.ColoumbContact(self.friction_parameter_key, ambient_dim)

        # Define discretization parameters
        for g, d in gb:
            if g.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    var_d: {"cells": ambient_dim},
                    var_s: {"cells": 1},
                }
                d[pp.DISCRETIZATION] = {
                    var_d: {"mpsa": mpsa},
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "stabilisation": stabilisiation_disc_s,
                        "source": source_disc_s,
                    },
                    var_d + "_" + var_s: {"grad_p": grad_p_disc},
                    var_s + "_" + var_d: {"div_u": div_u_disc},
                }
            elif g.dim == ambient_dim - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_variable: {"cells": ambient_dim},
                    var_s: {"cells": 1},
                }
                d[pp.DISCRETIZATION] = {
                    self.contact_variable: {"empty": empty_discr},
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "source": source_disc_s,
                    },
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

        # And define a Robin condition on the mortar grid
        contact = pp.PrimalContactCoupling(self.friction_parameter_key, mpsa, coloumb)
        div_u_coupling = pp.DivUCoupling(
            self.displacement_variable, div_u_disc, div_u_disc_frac
        )
        # This discretization needs the keyword used to store the grad p discretization:
        fracture_pressure_to_contact = pp.PressureContributionToForceBalance(
            key_m, mass_disc_s, mass_disc_s
        )
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            if g_h.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    self.surface_variable: {"cells": ambient_dim},
                    self.scalar_variable: {"cells": 1},
                }

                d[pp.COUPLING_DISCRETIZATION] = {
                    self.friction_coupling_term: {
                        g_h: (var_d, "mpsa"),
                        g_l: (self.contact_variable, "empty"),
                        (g_h, g_l): (self.surface_variable, contact),
                    },
                    self.scalar_coupling_term: {
                        g_h: (var_s, "diffusion"),
                        g_l: (var_s, "diffusion"),
                        e: (self.scalar_variable, pp.RobinCoupling(key_s, diff_disc_s)),
                    },
                    "div_u_coupling": {
                        g_h: (
                            var_s,
                            "mass",
                        ),  # This is really the div_u, but this is not implemented
                        g_l: (var_s, "mass"),
                        e: (self.surface_variable, div_u_coupling),
                    },
                    "pressure_to_force_balance": {
                        g_h: (var_s, "mass"),
                        g_l: (var_s, "mass"),
                        e: (self.surface_variable, fracture_pressure_to_contact),
                    },
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

    def discretize_biot(self, gb):
        """
        Discretization of Biot equations is done once, instead of separate
        discretization methods for each of the classes DivU, GradP and BiotStabilization.
        """
        g = gb.grids_of_dimension(gb.dim_max())[0]
        d = gb.node_props(g)
        biot = pp.Biot(
            mechanics_keyword=self.mechanics_parameter_key,
            flow_keyword=self.scalar_parameter_key,
            vector_variable=self.displacement_variable,
            scalar_variable=self.scalar_variable,
        )
        biot.discretize(g, d)

    def initial_condition(self):
        """
        Initial guess for Newton iteration.
        """
        gb = self.gb

        ambient_dimension = gb.dim_max()

        for g, d in gb:
            nc_nd = g.num_cells * ambient_dimension
            # Initial value for the scalar variable
            initial_scalar_value = 0 * np.ones(g.num_cells)
            if g.dim == ambient_dimension:
                # Initialize displacement variable
                key_m = self.mechanics_parameter_key
                bc_dict = {"bc_values": d[pp.PARAMETERS][key_m]["bc_values"]}
                pp.set_state(
                    d,
                    {
                        self.displacement_variable: np.zeros(nc_nd),
                        self.scalar_variable: initial_scalar_value,
                        key_m: bc_dict,
                    },
                )
            elif g.dim == ambient_dimension - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (np.zeros(g.num_cells), -100 * np.ones(g.num_cells))
                ).ravel(order="F")
                pp.set_state(
                    d,
                    {
                        self.contact_variable: traction,
                        self.scalar_variable: initial_scalar_value,
                    },
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            if mg.dim == 1:
                nc_nd = mg.num_cells * ambient_dimension
                pp.set_state(
                    d,
                    {
                        self.surface_variable: np.zeros(nc_nd),
                        self.scalar_variable: np.zeros(mg.num_cells),
                    },
                )

    def _set_friction_coefficient(self, g):
        friction_coefficient = 0.5 * np.ones(g.num_cells)
        return friction_coefficient


if __name__ == "__main__":
    unittest.main()
