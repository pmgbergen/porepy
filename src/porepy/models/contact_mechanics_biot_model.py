"""
This is a setup class for solving the Biot equations with contact mechanics at the fractures.

The class ContactMechanicsBiot inherits from ContactMechanics, which is a model for
the purely mechanical problem with contact conditions on the fractures. Here, we
expand to a model where the displacement solution is coupled to a scalar variable, e.g.
pressure (Biot equations) or temperature. Parameters, variables and discretizations are
set in the model class, and the problem may be solved using run_biot.

NOTE: This module should be considered an experimental feature, which will likely
undergo major changes (or be deleted).
"""
import logging
import time
from typing import Dict, List

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp
import porepy.models.contact_mechanics_model as contact_model
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanicsBiot(contact_model.ContactMechanics):
    def __init__(self, params: Dict = None) -> None:
        super(ContactMechanicsBiot, self).__init__(params)

        # Time
        self.time = 0
        self.time_step = 1
        self.end_time = 1

        # Temperature
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        # Scaling coefficients
        self.scalar_scale = 1
        self.length_scale = 1

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

    def bc_type_mechanics(self, g):
        # Use parent class method for mechanics
        return super().bc_type(g)

    def bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """
        Note that Dirichlet values should be divided by length_scale, and Neumann values
        by scalar_scale.
        """
        # Set the boundary values
        return super().bc_values(g)

    def bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """
        Note that Dirichlet values should be divided by scalar_scale.
        """
        return np.zeros(g.num_faces)

    def source_mechanics(self, g: pp.Grid) -> np.ndarray:
        return super().source(g)

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    def biot_alpha(self, g: pp.Grid) -> float:
        return 1

    def aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.Nd:
            aperture *= 0.1
        return aperture

    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """
        Compute the stress in the highest-dimensional grid based on the displacement
        and pressure states in that grid, adjacent interfaces and global boundary
        conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]['stress'].

        Parameters:
            previous_iterate (boolean, optional): If True, use values from previous
                iteration to compute the stress. Defaults to False.

        """
        # First the mechanical part of the stress
        super().reconstruct_stress(previous_iterate)

        g = self._nd_grid()
        d = self.gb.node_props(g)

        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]

        if previous_iterate:
            p = d[pp.STATE][pp.ITERATE][self.scalar_variable]
        else:
            p = d[pp.STATE][self.scalar_variable]

        # Stress contribution from the scalar variable
        d[pp.STATE]["stress"] += matrix_dictionary["grad_p"] * p

        # Is it correct there is no contribution from the global boundary conditions?

    def specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self.aperture(g)
        return np.power(a, self.Nd - g.dim)

    def set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        self.set_scalar_parameters()
        self.set_mechanics_parameters()

    def set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = np.ones(g.num_cells) / self.scalar_scale
                mu = np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self.bc_type_mechanics(g)
                # BC and source values
                bc_val = self.bc_values_mechanics(g)
                source_val = self.source_mechanics(g)

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
                        "biot_alpha": self.biot_alpha(g),
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

        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    def set_scalar_parameters(self) -> None:
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1 * self.scalar_scale
        for g, d in self.gb:
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)
            source_values = self.source_scalar(g)

            specific_volume = self.specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self.biot_alpha(g)
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": alpha,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "time_step": self.time_step,
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a_l = self.aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self.specific_volume(g_h)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            normal_diffusivity = kappa * 2 / (mg.secondary_to_mortar_avg() * a_l)
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"normal_diffusivity": normal_diffusivity},
            )

    def assign_discretizations(self) -> None:
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
            key_m,
            key_s,
            variable=var_d,
            mortar_variable=self.mortar_displacement_variable,
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
            else:
                d[pp.DISCRETIZATION] = {
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "source": source_disc_s,
                    }
                }

        # Define edge discretizations for the mortar grid
        contact_law = pp.ColoumbContact(self.mechanics_parameter_key, self.Nd, mpsa)
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
                            "fracture_scalar_to_force_balance": {
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
                d[pp.COUPLING_DISCRETIZATION] = {
                    self.scalar_coupling_term: {
                        g_h: (var_s, "diffusion"),
                        g_l: (var_s, "diffusion"),
                        e: (
                            self.mortar_scalar_variable,
                            pp.RobinCoupling(key_s, diff_disc_s),
                        ),
                    }
                }

    def assign_variables(self) -> None:
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
                d[pp.PRIMARY_VARIABLES] = {self.scalar_variable: {"cells": 1}}
        # Then for the edges
        for e, d in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self.Nd},
                    self.mortar_scalar_variable: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {self.mortar_scalar_variable: {"cells": 1}}

    def discretize_biot(self) -> None:
        """
        To save computational time, the full Biot equation (without contact mechanics)
        is discretized once. This is to avoid computing the same terms multiple times.
        """
        g = self._nd_grid()
        d = self.gb.node_props(g)
        biot = pp.Biot(
            mechanics_keyword=self.mechanics_parameter_key,
            flow_keyword=self.scalar_parameter_key,
            vector_variable=self.displacement_variable,
            scalar_variable=self.scalar_variable,
        )
        biot.discretize(g, d)

    def initial_condition(self) -> None:
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super().initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
            if g.dim == self.Nd:
                bc_values = self.bc_values_mechanics(g)
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            initial_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_value

    def export_step(self) -> None:
        pass

    def export_pvd(self) -> None:
        pass

    def discretize(self) -> None:
        """Discretize all terms"""
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb)

        g_max = self.gb.grids_of_dimension(self.Nd)[0]

        tic = time.time()
        logger.info("Discretize")

        # Discretization is a bit cumbersome, as the Biot discetization removes the
        # one-to-one correspondence between discretization objects and blocks in the matrix.
        # First, Discretize with the biot class
        self.discretize_biot()

        # Next, discretize term on the matrix grid not covered by the Biot discretization,
        # i.e. the source term
        filt = pp.assembler_filters.ListFilter(grid_list=[g_max], term_list=["source"])
        self.assembler.discretize(filt=filt)

        # Build a list of all edges, and all couplings
        edge_list = []
        for e, _ in self.gb.edges():
            edge_list.append(e)
            edge_list.append((e[0], e[1], e))
        if len(edge_list) > 0:
            filt = pp.assembler_filters.ListFilter(grid_list=edge_list)
            self.assembler.discretize(filt=filt)

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for dim in range(0, self.Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) > 0:
                filt = pp.assembler_filters.ListFilter(grid_list=grid_list)
                self.assembler.discretize(filt=filt)

        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    def before_newton_loop(self) -> None:
        """Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_parameters()

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        filt = pp.assembler_filters.ListFilter(term_list=[self.friction_coupling_term])
        self.assembler.discretize(filt=filt)

    def after_newton_iteration(self, solution: np.ndarray) -> None:
        self.update_state(solution)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: List, iteration_counter: int
    ) -> None:
        self.assembler.distribute_variable(solution)
        self.save_mechanical_bc_values()

    def save_mechanical_bc_values(self) -> None:
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key = self.mechanics_parameter_key
        g = self.gb.grids_of_dimension(self.Nd)[0]
        d = self.gb.node_props(g)
        d[pp.STATE][key]["bc_values"] = d[pp.PARAMETERS][key]["bc_values"].copy()

    def after_newton_divergence(self) -> None:
        raise ValueError("Newton iterations did not converge")

    def initialize_linear_solver(self) -> None:

        solver = self.params.get("linear_solver", "direct")

        if solver == "direct":
            """In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.
            """
            self.linear_solver = "direct"

        else:
            raise ValueError("unknown linear solver " + solver)

    def assemble_and_solve_linear_system(self, tol):

        A, b = self.assembler.assemble_matrix_rhs()
        logger.debug("Max element in A {0:.2e}".format(np.max(np.abs(A))))
        logger.debug(
            "Max {0:.2e} and min {1:.2e} A sum.".format(
                np.max(np.sum(np.abs(A), axis=1)), np.min(np.sum(np.abs(A), axis=1))
            )
        )
        if self.linear_solver == "direct":
            return spla.spsolve(A, b)
        elif self.linear_solver == "pyamg":
            print("pyamg")
            g = self.gb.grids_of_dimension(self.Nd)[0]
            assembler = self.assembler
            mechanics_dof = assembler.full_dof(g, self.displacement_variable)
            mortar_dof = np.setdiff1d(np.arange(b.size), mechanics_dof)

            A_el_m = A[mechanics_dof, :][:, mortar_dof]
            A_m_el = A[mortar_dof, :][:, mechanics_dof]
            A_m_m = A[mortar_dof, :][:, mortar_dof]

            # Also create a factorization of the mortar variable.
            # This is relatively cheap, so why not
            A_m_m_solve = sps.linalg.factorized(A_m_m)

            def precond_schur(r):
                # Mortar residual
                rm = r[mortar_dof]
                # Residual for the elasticity is the local one, plus the mapping of the
                # mortar residual
                r_el = r[mechanics_dof] - A_el_m * A_m_m_solve(rm)
                # Solve, using specified solver
                du = self.mechanics_precond(r_el)
                # Map back to mortar residual
                dm = A_m_m_solve(rm - A_m_el * du)
                x = np.zeros_like(r)
                x[mechanics_dof] = du
                x[mortar_dof] = dm
                return x

            M = sps.linalg.LinearOperator(A.shape, precond_schur)
            x, info = sps.linalg.gmres(A, b, M=M, restart=100, maxiter=10, tol=tol)

            return x
