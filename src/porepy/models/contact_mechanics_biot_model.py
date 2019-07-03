"""
This is a setup class for solving the biot equations with contact between the fractures.

The domain $[0, 2]\times[0, 1]$ with six fractures. We do not consider any fluid, and
solve only for the linear elasticity coupled to the contact

NOTE: This module should be considered an experimental feature, which will likely
undergo major changes (or be deleted).

"""
import numpy as np
import porepy as pp

import porepy.models.contact_mechanics_model as contact_model
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations


class ContactMechanicsBiot(contact_model.ContactMechanics):
    def __init__(self, mesh_args, folder_name):
        super().__init__(mesh_args, folder_name)

        # Temperature
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        # Scaling coefficients
        self.scalar_scale = 1
        self.length_scale = 1

        # Time
        self.time_step = 1 * self.length_scale ** 2
        self.end_time = self.time_step * 1

        # Initial scalar value
        self.initial_scalar = 0

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

    def bc_type(self, g, key, t=0):
        if key == self.mechanics_parameter_key:
            # Use parent class method for mechanics
            bc = super().bc_type(g)
        elif key == self.scalar_parameter_key:
            # Define boundary regions
            all_bf, *_ = self.domain_boundary_sides(g)
            # Define boundary condition on faces
            bc = pp.BoundaryCondition(g, all_bf, "dir")
        else:
            raise ValueError("No BCs implemented for keyword " + str(key))
        return bc

    def bc_values(self, g, key, t=0):
        # Set the boundary values
        if key == self.mechanics_parameter_key:
            bc_values = super().bc_values(g)
        elif key == self.scalar_parameter_key:
            bc_values = np.zeros(g.num_faces)
        else:
            raise ValueError("No BC values implemented for keyword " + str(key))
        return bc_values

    def source(self, g, key, t=0):
        if key == self.mechanics_parameter_key:
            values = super().source(g)
        elif key == self.scalar_parameter_key:
            values = np.zeros(g.num_cells)
        else:
            raise ValueError("No source values implemented for keyword " + str(key))
        return values

    def biot_alpha(self):
        return 1

    def compute_aperture(self, g):
        apertures = np.ones(g.num_cells)
        if g.dim < self.Nd:
            apertures *= 0.1
        return apertures

    def set_parameters(self):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """

        self.set_mechanics_parameters()
        self.set_scalar_parameters()

    def set_mechanics_parameters(self):
        """
        Set the parameters for the simulation.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = np.ones(g.num_cells)
                mu = np.ones(g.num_cells)
                C = pp.FourthOrderTensor(g.dim, mu, lam)

                # Define boundary condition
                bc = self.bc_type(g, self.mechanics_parameter_key)
                # Default internal BC is Neumann. We change to Dirichlet for the contact
                # problem. I.e., the mortar variable represents the displacement on the
                # fracture faces.
                frac_face = g.tags["fracture_faces"]
                bc.is_neu[:, frac_face] = False
                bc.is_dir[:, frac_face] = True
                # BC and source values
                bc_val = self.bc_values(g, self.mechanics_parameter_key)
                source_val = self.source(g, self.mechanics_parameter_key)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        "time_step": self.time_step,
                        "biot_alpha": self.biot_alpha(),
                    },
                )

            elif g.dim == self.Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction, "time_step": self.time_step},
                )
        # Should we keep this, @EK?
        for _, d in gb.edges():
            mg = d["mortar_grid"]

            # Parameters for the surface diffusion.
            mu = 1
            lmbda = 1

            pp.initialize_data(
                mg, d, self.mechanics_parameter_key, {"mu": mu, "lambda": lmbda}
            )

    def set_scalar_parameters(self):
        gb = self.gb
        self.Nd = gb.dim_max()

        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1
        alpha = self.biot_alpha()
        for g, d in gb:
            bc = self.bc_type(g, self.scalar_parameter_key)
            bc_values = self.bc_values(g, self.scalar_parameter_key)
            source_values = self.source(g, self.scalar_parameter_key, 0)

            a = self.compute_aperture(g)
            cross_sectional_area = np.power(a, self.gb.dim_max() - g.dim) * np.ones(
                g.num_cells
            )
            diffusivity = pp.SecondOrderTensor(self.Nd, kappa * np.ones(g.num_cells))

            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": alpha,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "aperture": cross_sectional_area,
                    "time_step": self.time_step,
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g1, _ = self.gb.nodes_of_edge(e)
            
            a = self.compute_aperture(g1)
            mg = data_edge["mortar_grid"]
            
            normal_diffusivity = 2 / kappa * mg.slave_to_mortar_int() * a
            
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"normal_diffusivity": normal_diffusivity},
            )

    def assign_discretisations(self):
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        # Shorthand
        key_s, key_m = self.scalar_parameter_key, self.mechanics_parameter_key
        var_s, var_d = self.scalar_variable, self.displacement_variable

        # Define discretization
        # For the Nd domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(key_m)
        empty_discr = pp.VoidDiscretization(key_m, ndof_cell=self.Nd)
        # Scalar discretizations (all dimensions)
        diff_disc_s = IE_discretizations.ImplicitMpfa(key_s)
        mass_disc_s = IE_discretizations.ImplicitMassMatrix(key_s, var_s)
        source_disc_s = pp.ScalarSource(key_s)
        # Coupling discretizations
        # All dimensions
        div_u_disc = pp.DivU(
            key_m, key_s, variable=var_d, mortar_variable=self.mortar_displacement_variable
        )
        # Nd
        grad_p_disc = pp.GradP(key_m)
        stabilization_disc_s = pp.BiotStabilization(key_s, var_s)

        # Assign node discretizations
        for g, d in self.gb:
            if g.dim == self.Nd:
                d[pp.DISCRETIZATION] = {
                    var_d: {"mpsa": mpsa},
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "stabilization": stabilization_disc_s,
                        "source": source_disc_s,
                    },
                    var_d + "_" + var_s: {"grad_p": grad_p_disc},
                    var_s + "_" + var_d: {"div_u": div_u_disc},
                }
            elif g.dim == self.Nd - 1:
                d[pp.DISCRETIZATION] = {
                    self.contact_traction_variable: {"empty": empty_discr},
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "source": source_disc_s,
                    },
                }

        # Define edge discretizations for the mortar grid
        contact_law = pp.ColoumbContact(self.mechanics_parameter_key, self.Nd)
        contact_discr = pp.PrimalContactCoupling(
            self.mechanics_parameter_key, mpsa, contact_law
        )
        # Account for the mortar displacements effect on scalar balance in the matrix,
        # as an internal boundary contribution, fracture, aperture changes appear as a
        # source contribution.
        div_u_coupling = pp.DivUCoupling(
            self.displacement_variable, div_u_disc, div_u_disc
        )
        # Account for the pressure contributions to the force balance on the fracture
        # (see contact_discr).
        # This discretization needs the keyword used to store the grad p discretization:
        grad_p_key = key_m
        matrix_scalar_to_force_balance = pp.MatrixScalarToForceBalance(
            grad_p_key, mass_disc_s, mass_disc_s
        )
        if self.subtract_fracture_pressure:
            fracture_scalar_to_force_balance = pp.FractureScalarToForceBalance(
                mass_disc_s, mass_disc_s
            )

        for e, d in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                d[pp.COUPLING_DISCRETIZATION] = {
                    self.friction_coupling_term: {
                        g_h: (var_d, "mpsa"),
                        g_l: (self.contact_traction_variable, "empty"),
                        (g_h, g_l): (self.mortar_displacement_variable, contact_discr),
                    },
                    self.scalar_coupling_term: {
                        g_h: (var_s, "diffusion"),
                        g_l: (var_s, "diffusion"),
                        e: (
                            self.mortar_scalar_variable,
                            pp.RobinCoupling(key_s, diff_disc_s),
                        ),
                    },
                    "div_u_coupling": {
                        g_h: (
                            var_s,
                            "mass",
                        ),  # This is really the div_u, but this is not implemented
                        g_l: (var_s, "mass"),
                        e: (self.mortar_displacement_variable, div_u_coupling),
                    },
                    "matrix_scalar_to_force_balance": {
                        g_h: (var_s, "mass"),
                        g_l: (var_s, "mass"),
                        e: (
                            self.mortar_displacement_variable,
                            matrix_scalar_to_force_balance,
                        ),
                    },
                }
                if self.subtract_fracture_pressure:
                    d[pp.COUPLING_DISCRETIZATION].update(
                        {
                            "matrix_scalar_to_force_balance": {
                                g_h: (var_s, "mass"),
                                g_l: (var_s, "mass"),
                                e: (
                                    self.mortar_displacement_variable,
                                    fracture_scalar_to_force_balance,
                                ),
                            }
                        }
                    )
            else:
                raise ValueError(
                    "assign_discretizations assumes no fracture intersections."
                )

    def assign_variables(self):
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        # First for the nodes
        for g, d in self.gb:
            if g.dim == self.Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": self.Nd},
                    self.scalar_variable: {"cells": 1},
                }
            elif g.dim == self.Nd - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_traction_variable: {"cells": self.Nd},
                    self.scalar_variable: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}
        # Then for the edges
        for e, d in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self.Nd},
                    self.mortar_scalar_variable: {"cells": 1},
                }

    def discretize_biot(self, gb):
        """
        To save computational time, the full Biot equation (without contact mechanics)
        is discretized once. This computing the same terms multiple times.
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
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super().initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = self.initial_scalar * np.ones(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
            if g.dim == self.Nd:
                bc_values = d[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})


def run_biot(setup, atol=1e-10):
    """
    Function for solving the time dependent Biot equations with a non-linear Coulomb
    contact condition on the fractures.

    The parameter keyword from the elasticity is assumed the same as the
    parameter keyword from the contact condition.

    In addition to the standard parameters for Biot we also require the following
    under the mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.

    Arguments:
        setup: A setup class with methods:
                set_parameters(): assigns data to grid bucket.
                assign_discretizations_and_variables(): assign the appropriate
                    discretizations and variables to each node and edge of the grid
                    bucket.
                create_grid(): Create grid bucket and set rotations for all fractures.
                initial_condition(): Set initial guesses for the iterates (contact
                     traction and mortar displacement) and the scalar variable.
            and attributes:
                end_time: End time time of simulation.
                time_step: Time step size
    """
    if "gb" not in setup.__dict__:
        setup.create_grid()
    gb = setup.gb

    # Extract the grids we use
    g_max = gb.grids_of_dimension(setup.Nd)[0]
    d_max = gb.node_props(g_max)

    # Assign parameters, variables and discretizations
    setup.set_parameters()
    setup.initial_condition()
    setup.assign_variables()
    setup.assign_discretisations()
    # Set up assembler and get initial condition for the displacements
    assembler = pp.Assembler(gb)
    u = d_max[pp.STATE][setup.displacement_variable]

    # Discretize with the biot class
    setup.discretize_biot(gb)
    # Prepare for the time loop
    errors = []
    t = getattr(setup, "time", 0)
    dt = setup.time_step
    t_end = setup.end_time
    k = 0
    times = [t]
    while t < t_end:
        t += dt
        k += 1
        print("Time step: ", k, "/", int(np.ceil(t_end / dt)))

        times.append(t)
        # Prepare for Newton
        counter_newton = 0
        converged_newton = False
        max_newton = 10
        newton_errors = []
        while counter_newton <= max_newton and not converged_newton:
            print("Newton iteration number: ", counter_newton, "/", max_newton)
            # One Newton iteration:
            sol, u, error, converged_newton = pp.models.contact_mechanics_model.newton_iteration(
                assembler, setup, u
            )
            counter_newton += 1
            newton_errors.append(error)
        # Prepare for next time step
        assembler.distribute_variable(sol)
        errors.append(newton_errors)
