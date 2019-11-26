"""
This is a setup class for solving the THM equations with contact mechanics at the fractures, if present.

The class ContactMechanicsBiot inherits from ContactMechanics, which is a model for
the purely mechanical problem with contact conditions on the fractures, and ContactMechanicsBiot, 
where the displacement solution is coupled to a scalar variable, e.g.
pressure (Biot equations) or temperature.  
Here, we expand to two scalar variables. The "scalar_variable" used in ContactMechanicsBiot is 
assumed to be the pressure, and the Biot discretization is applied to this. Then the discretizations 
are copied for the TM coupling, and TH coupling discretizations are provided. 
Note that we solve for the temperature increment T-T_0, and that the  
energy balance equation is divided by T_0 (in Kelvin!) to make the HM and TM coupling terms as similar as possible.
Parameters, variables and discretizations are set in the model class, and the problem may be solved using run_biot.

In addition, the discretization yields a stabilization term for each of the scalar equations.

Equation scaling: For monolithic solution of coupled systems, the condition number of the global
matrix may become a severe restriction. To alleviate this, the model is set up with three scaling
parameters. length_scaling allows to solve on a unit size domain, and result interpretation on e.g.
a kilometer scale. pressure_scale and temperature_scale may be used to solve for scaled pressure 
(p_scaled = p_physical/pressure_scale) and temperature. For typical reservoir conditions, choosing
a large (e.g. 1e6) pressure_scale is a good place to start. To obtain an idea about the effect
on the matrix, set the logging level to DEBUG.

NOTE: This module should be considered an experimental feature, which will likely
undergo major changes (or be deleted).
"""
import numpy as np
import porepy as pp
import logging
import time
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy.models.contact_mechanics_biot_model as parent_model
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)


class THM(parent_model.ContactMechanicsBiot):
    def __init__(self, params=None):
        super().__init__(params)

        # temperature
        self.temperature_variable = "T"
        self.mortar_temperature_advection_variable = (
            "mortar_advection_" + self.temperature_variable
        )
        self.advection_term = "advection_" + self.temperature_variable
        self.advection_coupling_term = "advection_coupling_" + self.temperature_variable
        self.mortar_temperature_variable = "mortar_" + self.temperature_variable
        self.temperature_coupling_term = "robin_" + self.temperature_variable
        self.temperature_parameter_key = "temperature"

        # Scaling coefficients for temperature
        self.temperature_scale = 1
        self.T_0_Kelvin = pp.CELSIUS_to_KELVIN(0)

        # Temperature pressure coupling
        self.t2s_weight_key = "t2s_coupling_weight"
        self.s2t_weight_key = "s2t_coupling_weight"
        self.t2s_coupling_term = "s_effect_on_t"
        self.s2t_coupling_term = "t_effect_on_s"
        # Temperature mechanics coupling
        self.mechanics_temperature_parameter_key = "mech_temperature"

    def set_parameters(self):
        """
        Set the parameters for the simulation.
        """
        self.set_scalar_parameters()
        self.set_temperature_parameters()
        self.set_mechanics_parameters()

    def bc_type_temperature(self, g):
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def bc_values_temperature(self, g):
        return np.zeros(g.num_faces)

    def source_temperature(self, g):
        return np.zeros(g.num_cells)

    def biot_beta(self, g):
        """
        TM coupling coefficient
        """
        if g.dim < self.Nd:
            return 1 / self.T_0_Kelvin
        return 1

    def scalar_temperature_coupling_coefficient(self, g):
        """
        TH coupling coefficient
        """
        return -1

    def compute_aperture(self, g):
        apertures = np.ones(g.num_cells)
        if g.dim < self.Nd:
            apertures *= 0.1
        return apertures

    def set_mechanics_parameters(self):
        """
        Set the parameters for the simulation.
        """
        super().set_mechanics_parameters()
        for g, d in self.gb:
            if g.dim == self.Nd:
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    {
                        "biot_alpha": self.biot_beta(g),
                        "bc_values": self.bc_values_mechanics(g),
                    },
                )

    def set_scalar_parameters(self):
        super().set_scalar_parameters()
        for g, d in self.gb:
            a = self.compute_aperture(g)
            specific_volume = np.power(a, self.gb.dim_max() - g.dim) * np.ones(
                g.num_cells
            )
            t2s_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volume
                * self.temperature_scale
            )
            pp.initialize_data(
                g, d, self.scalar_parameter_key, {self.t2s_weight_key: t2s_coupling}
            )

    def set_temperature_parameters(self):

        tensor_scale = self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
        kappa = 1 * tensor_scale
        heat_capacity = 1
        mass_weight = heat_capacity * self.temperature_scale / self.T_0_Kelvin

        for g, d in self.gb:
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)
            source_values = self.source_scalar(g)

            a = self.compute_aperture(g)
            specific_volume = np.power(a, self.gb.dim_max() - g.dim) * np.ones(
                g.num_cells
            )
            thermal_conductivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )
            advection_weight = heat_capacity * self.temperature_scale / self.T_0_Kelvin
            s2t_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volume
                * self.scalar_scale
            )
            pp.initialize_data(
                g,
                d,
                self.temperature_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": self.biot_beta(g),
                    "source": source_values,
                    "second_order_tensor": thermal_conductivity,
                    "advection_weight": advection_weight,
                    "time_step": self.time_step,
                    self.s2t_weight_key: s2t_coupling,
                    "darcy_flux": np.zeros(g.num_faces),
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g1, _ = self.gb.nodes_of_edge(e)
            a = self.compute_aperture(g1)
            mg = data_edge["mortar_grid"]

            normal_diffusivity = kappa * 2 / (mg.slave_to_mortar_int() * a)
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.temperature_parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                    "darcy_flux": np.zeros(mg.num_cells),
                },
            )

    def assign_variables(self):
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super().assign_variables()
        # First for the nodes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.temperature_variable: {"cells": 1}})

        # Then for the edges
        for e, d in self.gb.edges():
            d[pp.PRIMARY_VARIABLES].update(
                {
                    self.mortar_temperature_variable: {"cells": 1},
                    self.mortar_temperature_advection_variable: {"cells": 1},
                }
            )

    def assign_discretizations(self):
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        super().assign_discretizations()
        # Shorthand
        key_s = self.scalar_parameter_key
        key_t = self.temperature_parameter_key
        key_m = self.mechanics_parameter_key
        # distinguish the coupling terms from temperature terms (grad_p, as opposed to grad_T)
        key_mt = self.mechanics_temperature_parameter_key
        var_s = self.scalar_variable
        var_t = self.temperature_variable
        var_d = self.displacement_variable
        # Define discretization
        # Scalar discretizations (all dimensions)
        diff_disc_t = IE_discretizations.ImplicitMpfa(key_t)
        adv_disc_t = IE_discretizations.ImplicitUpwind(key_t)
        mass_disc_t = IE_discretizations.ImplicitMassMatrix(key_t, var_t)

        # Coupling discretizations
        # All dimensions
        div_u_disc_t = DivUEnergy(
            key_m,
            key_t,
            variable=var_d,
            mortar_variable=self.mortar_displacement_variable,
        )
        # Nd
        grad_t_disc = pp.GradP(
            key_mt
        )  # pp.GradP(key_m) # Kanskje denne (og andre) b;r erstattes av spesiallagde
        # varianter som henter ut s-varianten og ganger med alpha/beta?
        stabilization_disc_t = pp.BiotStabilization(key_t, var_t)
        s2t_disc = IE_discretizations.ImplicitMassMatrix(
            keyword=self.temperature_parameter_key,
            variable=var_s,
            mass_weight_key=self.s2t_weight_key,
        )
        t2s_disc = IE_discretizations.ImplicitMassMatrix(
            keyword=self.scalar_parameter_key,
            variable=var_t,
            mass_weight_key=self.t2s_weight_key,
        )

        # Assign node discretizations
        for g, d in self.gb:
            if g.dim == self.Nd:
                d[pp.DISCRETIZATION].update(
                    {
                        var_t: {
                            "diffusion": diff_disc_t,
                            self.advection_term: adv_disc_t,
                            "mass": mass_disc_t,
                            "stabilization": stabilization_disc_t,
                        },
                        var_d + "_" + var_t: {"grad_p": grad_t_disc},
                        var_t + "_" + var_d: {"div_u": div_u_disc_t},
                        var_t + "_" + var_s: {self.s2t_coupling_term: s2t_disc},
                        var_s + "_" + var_t: {self.t2s_coupling_term: t2s_disc},
                    }
                )
            else:
                d[pp.DISCRETIZATION].update(
                    {
                        var_t: {
                            "diffusion": diff_disc_t,
                            self.advection_term: adv_disc_t,
                            "mass": mass_disc_t,
                        },
                        var_t + "_" + var_s: {self.s2t_coupling_term: s2t_disc},
                        var_s + "_" + var_t: {self.t2s_coupling_term: t2s_disc},
                    }
                )
        # Account for the mortar displacements effect on energy balance in the matrix,
        # as an internal boundary contribution
        div_u_coupling_t = pp.DivUCoupling(
            self.displacement_variable, div_u_disc_t, div_u_disc_t
        )
        # Account for the temperature contributions to the force balance on the fracture
        # (see contact_discr).
        # This discretization needs the keyword used to store the grad p discretization:
        grad_t_key = key_mt
        matrix_temperature_to_force_balance = pp.MatrixScalarToForceBalance(
            grad_t_key, mass_disc_t, mass_disc_t
        )
        adv_coupling_t = IE_discretizations.ImplicitUpwindCoupling(key_t)
        diff_coupling_t = pp.RobinCoupling(key_t, diff_disc_t)

        for e, d in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)

            d[pp.COUPLING_DISCRETIZATION].update(
                {
                    self.advection_coupling_term: {
                        g_h: (self.temperature_variable, self.advection_term),
                        g_l: (self.temperature_variable, self.advection_term),
                        e: (self.mortar_temperature_advection_variable, adv_coupling_t),
                    },
                    self.temperature_coupling_term: {
                        g_h: (var_t, "diffusion"),
                        g_l: (var_t, "diffusion"),
                        e: (self.mortar_temperature_variable, diff_coupling_t),
                    },
                }
            )
            if g_h.dim == self.Nd:
                d[pp.COUPLING_DISCRETIZATION].update(
                    {
                        "div_u_coupling_t": {
                            g_h: (
                                var_t,
                                "mass",
                            ),  # This is really the div_u, but this is not implemented
                            g_l: (var_t, "mass"),
                            e: (self.mortar_displacement_variable, div_u_coupling_t),
                        },
                        "matrix_temperature_to_force_balance": {
                            g_h: (var_t, "mass"),
                            g_l: (var_t, "mass"),
                            e: (
                                self.mortar_displacement_variable,
                                matrix_temperature_to_force_balance,
                            ),
                        },
                    }
                )

    def initial_condition(self):
        """
        In addition to the values set by the parent class, we set initial value for the temperature
        variable, and a previous iterate value for the scalar value. The latter is used for 
        computation of Darcy fluxes, needed for the advective term of the energy equation.
        """
        super().initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            cell_zeros = np.zeros(g.num_cells)
            state = {self.temperature_variable: cell_zeros}
            iterate = {self.scalar_variable: cell_zeros}  # For initial flux
            d[pp.STATE].update(state)
            if "previous_iterate" in d[pp.STATE]:
                d[pp.STATE]["previous_iterate"].update(iterate)
            else:
                d[pp.STATE]["previous_iterate"] = iterate
        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            cell_zeros = np.zeros(mg.num_cells)
            state = {
                self.mortar_temperature_variable: cell_zeros,
                self.mortar_temperature_advection_variable: cell_zeros,
            }
            iterate = {self.mortar_scalar_variable: cell_zeros}
            d[pp.STATE].update(state)
            d[pp.STATE]["previous_iterate"].update(iterate)

    def compute_fluxes(self):
        pp.fvutils.compute_darcy_flux(
            self.gb,
            keyword=self.scalar_parameter_key,
            keyword_store=self.temperature_parameter_key,
            d_name="darcy_flux",
            p_name=self.scalar_variable,
            lam_name=self.mortar_scalar_variable,
            from_iterate=True,
        )

    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self.create_grid()
        self.set_parameters()
        self.initial_condition()
        self.assign_variables()
        self.assign_discretizations()
        self.discretize()
        self.initialize_linear_solver()

    def discretize(self):
        """ Discretize all terms
        """
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb)

        g_max = self._nd_grid()

        tic = time.time()
        logger.info("Discretize")

        # Discretization is a bit cumbersome, as the Biot discetization removes the
        # one-to-one correspondence between discretization objects and blocks in the matrix.
        # First, Discretize with the biot class
        self.discretize_biot()
        self.copy_biot_discretizations()
        # Next, discretize term on the matrix grid not covered by the Biot discretization,
        # i.e. the source term
        pressure_terms = ["source", self.t2s_coupling_term]
        self.assembler.discretize(grid=g_max, term_filter=pressure_terms)
        # Then the temperature discretizations
        temperature_terms = [
            "source",
            self.s2t_coupling_term,
            "diffusion",
            "mass",
            self.advection_term,
        ]
        self.assembler.discretize(
            grid=self._nd_grid(),
            term_filter=temperature_terms,
            variable_filter=self.temperature_variable,
        )

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for g, _ in self.gb:
            if g.dim < self.Nd:
                self.assembler.discretize(grid=g)

        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    def before_newton_iteration(self):
        """ Re-discretize the nonlinear terms
        """
        self.compute_fluxes()
        terms = [
            self.friction_coupling_term,
            self.advection_term,
            self.advection_coupling_term,
        ]
        self.assembler.discretize(term_filter=terms)

    def copy_biot_discretizations(self):
        g = self.gb.grids_of_dimension(self.Nd)[0]
        d = self.gb.node_props(g)
        # For grad p term of u equation
        weight_grad_t = self.biot_beta(g) / self.biot_alpha(g)
        # Account for scaling
        weight_grad_t *= self.temperature_scale / self.scalar_scale
        # Stabilization is derived from the grad p discretization
        weight_stabilization = self.biot_beta(g) / self.biot_alpha(g)
        weight_stabilization *= self.temperature_scale / self.scalar_scale
        # The stabilization terms appear in the T/p equations, whereof only the first
        # is divided by T_0_Kelvin.
        weight_stabilization *= 1 / self.T_0_Kelvin
        matrices_s = d[pp.DISCRETIZATION_MATRICES][self.scalar_parameter_key]
        matrices_ms = d[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        matrices_t = d[pp.DISCRETIZATION_MATRICES][self.temperature_parameter_key]
        matrices_mt = {}
        matrices_t["div_u"] = matrices_s["div_u"].copy()
        matrices_t["bound_div_u"] = matrices_s["bound_div_u"].copy()
        matrices_mt["grad_p"] = weight_grad_t * matrices_ms["grad_p"]
        matrices_t["biot_stabilization"] = (
            weight_stabilization * matrices_s["biot_stabilization"]
        )
        matrices_mt["bound_displacement_pressure"] = (
            weight_grad_t * matrices_ms["bound_displacement_pressure"]
        )
        # For div u term of t equation
        weight_div_u = self.biot_beta(g)
        key_m_from_t = self.mechanics_temperature_parameter_key
        d[pp.DISCRETIZATION_MATRICES][key_m_from_t] = matrices_mt
        d[pp.PARAMETERS][key_m_from_t] = {"biot_alpha": weight_div_u}
        bc_dict = {"bc_values": self.bc_values_mechanics(g)}
        state = {key_m_from_t: bc_dict}
        pp.set_state(d, state)

    def save_mechanical_bc_values(self):
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key, key_t = (
            self.mechanics_parameter_key,
            self.mechanics_temperature_parameter_key,
        )
        g = self.gb.grids_of_dimension(self.Nd)[0]
        d = self.gb.node_props(g)
        d[pp.STATE][key]["bc_values"] = d[pp.PARAMETERS][key]["bc_values"].copy()
        d[pp.STATE][key_t]["bc_values"] = d[pp.PARAMETERS][key_t]["bc_values"].copy()

    def update_state(self, solution_vector):
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE]["previous_iterate"] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            assembler (pp.Assembler): assembler for self.gb.
            solution_vector (np.array): solution vector for the current iterate.

        """
        super().update_state(solution_vector)
        assembler = self.assembler
        variable_names = []
        for pair in assembler.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in assembler.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    if name == self.mortar_scalar_variable:
                        mortar_p = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.edge_props(g)
                        data[pp.STATE]["previous_iterate"][
                            self.mortar_scalar_variable
                        ] = mortar_p.copy()
                else:
                    data = self.gb.node_props(g)

                    # g is a node (not edge)

                    # For the fractures, update the contact force
                    if name == self.scalar_variable:
                        p = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.node_props(g)
                        data[pp.STATE]["previous_iterate"][
                            self.scalar_variable
                        ] = p.copy()


class DivUEnergy(pp.DivU):
    def assemble_int_bound_displacement_source(
        self, g, data, data_edge, cc, matrix, rhs, self_ind
    ):
        pass


if __name__ == "__main__":
    print("hei")
    M = THM()
    pp.run_time_dependent_model(M)
    g = M._nd_grid()
    print(M.gb.node_props(g)[pp.STATE]["pressure"])
