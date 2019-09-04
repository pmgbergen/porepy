#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module contains completed setups for simple contact mechanics problems,
with and without poroelastic effects in the deformation of the Nd domain.
"""
import numpy as np
from scipy.spatial.distance import cdist
import logging

import porepy as pp
import porepy.models.contact_mechanics_model
import porepy.models.contact_mechanics_biot_model
import porepy.utils.derived_discretizations.implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanicsExample(pp.models.contact_mechanics_model.ContactMechanics):
    def __init__(self, mesh_args, folder_name):
        self.mesh_args = mesh_args
        self.folder_name = folder_name
        
        super(ContactMechanicsExample, self).__init__()

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            frac_pts (np.array): Nd x (number of fracture points), the coordinates of
                the fracture endpoints.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            gb (pp.GridBucket): The produced grid bucket.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

        """
        # List the fracture points
        self.frac_pts = np.array([[0.2, 0.8], [0.5, 0.5]])
        # Each column defines one fracture
        frac_edges = np.array([[0], [1]])
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

    def domain_boundary_sides(self, g):
        """
        Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        north = g.face_centers[1] > box["ymax"] - tol
        south = g.face_centers[1] < box["ymin"] + tol
        if self.Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    def _set_friction_coefficient(self, g):

        nodes = g.nodes

        tips = nodes[:, [0, -1]]

        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = 0.5 * (1 + beta * np.exp(-R * D ** 2))
        return friction_coefficient


class ContactMechanicsBiotExample(porepy.models.contact_mechanics_biot_model.ContactMechanicsBiot):
    def __init__(self, mesh_args, folder_name):
        super().__init__()
        self.mesh_args = mesh_args
        self.folder_name = folder_name

        # Time
        # The time attribute may be used e.g. to update BCs.
        self.time = 0
        self.time_step = 1 * self.length_scale ** 2
        self.end_time = self.time_step * 1

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

    def bc_type_mechanics(self, g):
        # Use parent class method for mechanics
        return super().bc_type(g)

    def bc_type_scalar(self, g):
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def bc_values_mechanics(self, g):
        # Set the boundary values
        return super().bc_values(g)

    def bc_values_scalar(self, g):
        return np.zeros(g.num_faces)

    def source_mechanics(self, g):
        return super().source(g)

    def source_scalar(self, g):
        return np.zeros(g.num_cells)

    def biot_alpha(self):
        return 1

    def compute_aperture(self, g):
        apertures = np.ones(g.num_cells)
        if g.dim < self.Nd:
            apertures *= 0.1
        return apertures

    def set_parameters(self):
        """
        Set the parameters for the simulation.
        """
        self.set_scalar_parameters()
        self.set_mechanics_parameters()

    def set_mechanics_parameters(self):
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
            
    def _set_friction_coefficient(self, g):

        nodes = g.nodes

        tips = nodes[:, [0, -1]]

        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = 0.5 * (1 + beta * np.exp(-R * D ** 2))
        return friction_coefficient

    def set_scalar_parameters(self):
        gb = self.gb
        self.Nd = gb.dim_max()

        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1
        alpha = self.biot_alpha()
        for g, d in gb:
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)
            source_values = self.source_scalar(g)

            a = self.compute_aperture(g)
            specific_volume = np.power(a, self.gb.dim_max() - g.dim) * np.ones(
                g.num_cells
            )
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

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
        is discretized once. This is to avoid computing the same terms multiple times.
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
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
            if g.dim == self.Nd:
                bc_values = d[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

    def export_step(self):
        pass

    def export_pvd(self):
        pass


