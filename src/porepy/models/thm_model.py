"""
This is a setup class for solving the THM equations with contact mechanics at the
fractures, if present.

We build on two other model classes: The class ContactMechanicsBiot inherits from
ContactMechanics, which is a model for the purely mechanical problem with contact
conditions on the fractures, and ContactMechanicsBiot, where the displacement solution is
coupled to a scalar variable, e.g. pressure (Biot equations) or temperature.

Here, we expand to two scalar variables. The "scalar_variable" used in ContactMechanicsBiot
is assumed to be the pressure, and the Biot discretization is applied to this. Then the
discretizations are copied for the TM coupling, and TH coupling discretizations are
provided. Note that the energy balance equation is divided by T_0 (in Kelvin!) to make
the HM and TM coupling terms as similar as possible. Parameters, variables and discretizations
are set in the model class, and the problem may be solved using run_biot.

In addition, the discretization yields a stabilization term for each of the scalar
equations.

Equation scaling: For monolithic solution of coupled systems, the condition number of
the global matrix may become a severe restriction. To alleviate this, the model is set
up with three scaling parameters. length_scaling allows to solve on a unit size domain,
and result interpretation on e.g. a kilometer scale. pressure_scale and temperature_scale
may be used to solve for scaled pressure (p_scaled = p_physical/pressure_scale) and
temperature. For typical reservoir conditions, choosing a large (e.g. 1e6) pressure_scale
is a good place to start. To obtain an idea about the effect on the matrix, set the
logging level to DEBUG.

NOTE: This module should be considered an experimental feature, which will likely
undergo major changes (or be deleted).
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import porepy as pp
import porepy.models.contact_mechanics_biot_model as parent_model
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class THM(parent_model.ContactMechanicsBiot):
    """This is a shell class for poro-elastic contact mechanics problems.

    Setting up such problems requires a lot of boilerplate definitions of variables,
    parameters and discretizations. This class is intended to provide a standardized
    setup, with all discretizations in place and reasonable parameter and boundary
    values. The intended use is to inherit from this class, and do the necessary
    modifications and specifications for the problem to be fully defined. The minimal
    adjustment needed is to specify the method create_grid().

    Attributes:
        time (float): Current time.
        time_step (float): Size of an individual time step
        end_time (float): Time at which the simulation should stop.

        displacement_variable (str): Name assigned to the displacement variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in Paraview export.
        mortar_displacement_variable (str): Name assigned to the displacement variable
            on the fracture walls. Will be used throughout the simulations, including in
            Paraview export.
        contact_traction_variable (str): Name assigned to the variable for contact
            forces in the fracture. Will be used throughout the simulations, including
            in Paraview export.
        scalar_variable (str): Name assigned to the pressure variable. Will be used
            throughout the simulations, including in Paraview export.
        temperature_variable (str): Name assigned to the temperature variable. Will be used
            throughout the simulations, including in Paraview export.
        mortar scalar_variable (str): Name assigned to the interface scalar variable
            representing flux between grids. Will be used throughout the simulations,
            including in Paraview export.

        mechanics_parameter_key (str): Keyword used to define parameters and
            discretizations for the mechanics problem.
        scalar_parameter_key (str): Keyword used to define parameters and
            discretizations for the flow problem.
        temperature_parameter_key (str): Keyword used to define parameters and
            discretizations for the temperature problem.
        mechanics_temperature_parameter_key (str): Keyword used to define parameters and
            discretizations for the coupling between temperature and mechanics.

        params (dict): Dictionary of parameters used to control the solution procedure.
        viz_folder_name (str): Folder for visualization export.
        gb (pp.GridBucket): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'
        scalar_scale (float): Scaling coefficient for the pressure variable. Can be used
            to get comparable size of the mechanical and flow problem.
        scalar_scale (float): Scaling coefficient for the vector variable. Can be used
            to get comparable size of the mechanical and flow problem.
        temperature_scale (float): Scaling coefficient for the temperature variable.
            Can be used to get comparable size of the mechanical and flow problem.
            NOTE: This has not been properly tested, assign unit value to stay safe.

    Except from the grid, all attributes are given natural values at initialization of
    the class.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)

        # temperature
        self.temperature_variable: str = "T"
        self.mortar_temperature_advection_variable: str = (
            "mortar_advection_" + self.temperature_variable
        )
        self.advection_term: str = "advection_" + self.temperature_variable
        self.advection_coupling_term: str = (
            "advection_coupling_" + self.temperature_variable
        )
        self.mortar_temperature_variable: str = "mortar_" + self.temperature_variable
        self.temperature_coupling_term: str = "robin_" + self.temperature_variable

        self.temperature_parameter_key: str = "temperature"

        # Scaling coefficients for temperature
        # NOTE: temperature_scale different from 1 has not been tested, and will likely
        # introduce errors.
        self.temperature_scale: float = 1
        self.T_0_Kelvin: float = pp.CELSIUS_to_KELVIN(0)

        # Temperature pressure coupling
        self.t2s_parameter_key: str = "t2s_parameters"
        self.s2t_parameter_key: str = "s2t_parameters"
        self.s2t_coupling_term: str = "s_effect_on_t"
        self.t2s_coupling_term: str = "t_effect_on_s"

        # Keyword needed to specify parameters and discretizations for the
        # temperature mechanics coupling
        self.mechanics_temperature_parameter_key: str = "mech_temperature"

    @pp.time_logger(sections=module_sections)
    def before_newton_iteration(self) -> None:
        """Re-discretize the nonlinear terms"""
        self.compute_fluxes()
        terms = [
            self.friction_coupling_term,
            self.advection_term,
            self.advection_coupling_term,
        ]
        filt = pp.assembler_filters.ListFilter(term_list=terms)
        self.assembler.discretize(filt=filt)

    @pp.time_logger(sections=module_sections)
    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """
        Compute the stress in the highest-dimensional grid based on the displacement
        pressure and temperature states in that grid, adjacent interfaces and global
        boundary conditions.

        The stress is stored in the data dictionary of the highest dimensional grid,
        in [pp.STATE]['stress'].

        Parameters:
            previous_iterate (boolean, optional): If True, use values from previous
                iteration to compute the stress. Defaults to False.

        """
        # First the hydro-mechanical part of the stress
        super().reconstruct_stress(previous_iterate)
        g = self._nd_grid()
        d = self.gb.node_props(g)

        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][
            self.mechanics_temperature_parameter_key
        ]
        if previous_iterate:
            T = d[pp.STATE][pp.ITERATE][self.temperature_variable]
        else:
            T = d[pp.STATE][self.temperature_variable]

        # Stress contribution from the scalar variable
        d[pp.STATE]["stress"] += matrix_dictionary["grad_p"] * T

    @pp.time_logger(sections=module_sections)
    def compute_fluxes(self) -> None:
        """Compute the fluxes in the mixed-dimensional grid from the current state of
        the pressure variables.

        """
        pp.fvutils.compute_darcy_flux(
            self.gb,
            keyword=self.scalar_parameter_key,
            keyword_store=self.temperature_parameter_key,
            d_name="darcy_flux",
            p_name=self.scalar_variable,
            lam_name=self.mortar_scalar_variable,
            from_iterate=True,
        )

    @pp.time_logger(sections=module_sections)
    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        self._set_scalar_parameters()
        self._set_temperature_parameters()
        self._set_mechanics_parameters()

    @pp.time_logger(sections=module_sections)
    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        super()._set_mechanics_parameters()
        for g, d in self.gb:
            if g.dim == self._Nd:
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    {
                        "biot_alpha": self._biot_beta(g),
                        "bc_values": self._bc_values_mechanics(g),
                        "p_reference": np.zeros(g.num_cells),
                    },
                )

    @pp.time_logger(sections=module_sections)
    def _set_scalar_parameters(self) -> None:
        """Set parameters for the pressure / mass conservation equation."""
        # Most values are handled as if this was a poro-elastic problem
        super()._set_scalar_parameters()
        for g, d in self.gb:
            t2s_coupling = (
                self._scalar_temperature_coupling_coefficient(g)
                * self._specific_volume(g)
                * self.temperature_scale
            )
            pp.initialize_data(
                g,
                d,
                self.t2s_parameter_key,
                {"mass_weight": t2s_coupling, "time_step": self.time_step},
            )

    @pp.time_logger(sections=module_sections)
    def _set_temperature_parameters(self) -> None:
        """Parameters for the temperature equation."""
        tensor_scale: float = (
            self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
        )
        kappa: float = 1 * tensor_scale
        heat_capacity = 1
        mass_weight: float = heat_capacity * self.temperature_scale / self.T_0_Kelvin
        for g, d in self.gb:
            # By default, we set the same type of boundary conditions as for the
            # pressure problem, that is, zero Dirichlet everywhere
            bc = self._bc_type_temperature(g)
            bc_values = self._bc_values_temperature(g)
            source_values = self._source_temperature(g)

            specific_volume = self._specific_volume(g)
            thermal_conductivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )
            advection_weight = heat_capacity * self.temperature_scale / self.T_0_Kelvin
            s2t_coupling = (
                self._scalar_temperature_coupling_coefficient(g)
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
                    "biot_alpha": self._biot_beta(g),
                    "source": source_values,
                    "second_order_tensor": thermal_conductivity,
                    "advection_weight": advection_weight,
                    "time_step": self.time_step,
                },
            )
            pp.initialize_data(
                g,
                d,
                self.s2t_parameter_key,
                {"mass_weight": s2t_coupling, "time_step": self.time_step},
            )

        # Assign diffusivity in the normal direction of the fractures.
        # Also initialize fluxes.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]

            a_l = self._aperture(g_l)
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self._specific_volume(g_h)
            )  #
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
                self.temperature_parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                },
            )

    @pp.time_logger(sections=module_sections)
    def _bc_type_temperature(self, g: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    @pp.time_logger(sections=module_sections)
    def _bc_values_temperature(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_faces)

    @pp.time_logger(sections=module_sections)
    def _source_temperature(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    @pp.time_logger(sections=module_sections)
    def _biot_beta(self, g: pp.Grid) -> float:
        """
        TM coupling coefficient
        """
        return 1.0

    @pp.time_logger(sections=module_sections)
    def _scalar_temperature_coupling_coefficient(self, g: pp.Grid) -> float:
        """
        TH coupling coefficient
        """
        return -1.0

    @pp.time_logger(sections=module_sections)
    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()

        # The remaining variables to define is the temperature on the nodes
        for _, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.temperature_variable: {"cells": 1}})

        # And advective and diffusive fluxes on the edges
        for _, d in self.gb.edges():
            d[pp.PRIMARY_VARIABLES].update(
                {
                    self.mortar_temperature_variable: {"cells": 1},
                    self.mortar_temperature_advection_variable: {"cells": 1},
                }
            )

    @pp.time_logger(sections=module_sections)
    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        # Call parent class for disrcetizations for the poro-elastic system.
        super()._assign_discretizations()

        # What remains is terms related to temperature

        # Shorthand for parameter keywords
        key_t = self.temperature_parameter_key
        key_m = self.mechanics_parameter_key

        # distinguish the coupling terms from temperature terms (grad_p, as opposed to grad_T)
        key_mt = self.mechanics_temperature_parameter_key
        var_s = self.scalar_variable
        var_t = self.temperature_variable
        var_d = self.displacement_variable

        # Define discretization
        # Scalar discretizations (will be assigned to all dimensions)
        diff_disc_t = IE_discretizations.ImplicitMpfa(key_t)
        adv_disc_t = IE_discretizations.ImplicitUpwind(key_t)
        mass_disc_t = IE_discretizations.ImplicitMassMatrix(key_t, var_t)
        source_disc_t = pp.ScalarSource(key_t)

        # Coupling discretizations
        # All dimensions
        div_u_disc_t = pp.DivU(
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
            keyword=self.s2t_parameter_key, variable=var_s
        )
        t2s_disc = IE_discretizations.ImplicitMassMatrix(
            keyword=self.t2s_parameter_key, variable=var_t
        )

        # Assign node discretizations
        for g, d in self.gb:
            if g.dim == self._Nd:
                d[pp.DISCRETIZATION].update(
                    {  # advection-diffusion equation for temperature
                        var_t: {
                            "diffusion": diff_disc_t,
                            self.advection_term: adv_disc_t,
                            "mass": mass_disc_t,
                            # Also the stabilization term from Biot
                            "stabilization": stabilization_disc_t,
                            "source": source_disc_t,
                        },
                        # grad T term in the momentuum equation
                        var_d + "_" + var_t: {"grad_p": grad_t_disc},
                        # div u term in the energy equation
                        var_t + "_" + var_d: {"div_u": div_u_disc_t},
                        # Accumulation of pressure in energy equation
                        var_t + "_" + var_s: {self.s2t_coupling_term: s2t_disc},
                        # Accumulation of temperature / energy in pressure equation
                        var_s + "_" + var_t: {self.t2s_coupling_term: t2s_disc},
                    }
                )
            else:
                # Inside fracture network
                d[pp.DISCRETIZATION].update(
                    {  # Advection-diffusion equation, no biot stabilization
                        var_t: {
                            "diffusion": diff_disc_t,
                            self.advection_term: adv_disc_t,
                            "mass": mass_disc_t,
                            "source": source_disc_t,
                        },
                        # Standard coupling terms
                        var_t + "_" + var_s: {self.s2t_coupling_term: s2t_disc},
                        var_s + "_" + var_t: {self.t2s_coupling_term: t2s_disc},
                    }
                )
        # Next, the edges in the gb.
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

        # Coupling of advection and diffusion terms in the energy equation
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
            if g_h.dim == self._Nd:
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

    @pp.time_logger(sections=module_sections)
    def _initial_condition(self) -> None:
        """
        In addition to the values set by the parent class, we set initial value for the
        temperature variable, and a previous iterate value for the scalar value. The
        latter is used for computation of Darcy fluxes, needed for the advective term of
        the energy equation. The Darcy flux parameter is also initialized.
        """
        super()._initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            cell_zeros = np.zeros(g.num_cells)
            state = {self.temperature_variable: cell_zeros}
            iterate = {self.scalar_variable: cell_zeros}  # For initial flux
            pp.set_state(d, state)
            pp.set_iterate(d, iterate)
            # Initial Darcy fluxes for advective flux.
            pp.initialize_data(
                g,
                d,
                self.temperature_parameter_key,
                {
                    "darcy_flux": np.zeros(g.num_faces),
                },
            )

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            cell_zeros = np.zeros(mg.num_cells)
            state = {
                self.mortar_temperature_variable: cell_zeros,
                self.mortar_temperature_advection_variable: cell_zeros,
            }
            iterate = {self.mortar_scalar_variable: cell_zeros}

            pp.set_state(d, state)
            pp.set_iterate(d, iterate)
            # Initial Darcy fluxes for advective flux.
            d = pp.initialize_data(
                e,
                d,
                self.temperature_parameter_key,
                {
                    "darcy_flux": np.zeros(mg.num_cells),
                },
            )

    @pp.time_logger(sections=module_sections)
    def _save_mechanical_bc_values(self) -> None:
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key, key_t = (
            self.mechanics_parameter_key,
            self.mechanics_temperature_parameter_key,
        )
        g = self._nd_grid()
        d = self.gb.node_props(g)
        d[pp.STATE][key]["bc_values"] = d[pp.PARAMETERS][key]["bc_values"].copy()
        d[pp.STATE][key_t]["bc_values"] = d[pp.PARAMETERS][key_t]["bc_values"].copy()

    @pp.time_logger(sections=module_sections)
    def _discretize(self) -> None:
        """Discretize all terms"""
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.gb)
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb, self.dof_manager)

        tic = time.time()
        logger.info("Discretize")

        # Discretization is a bit cumbersome, as the Biot discetization removes the
        # one-to-one correspondence between discretization objects and blocks in the matrix.
        # First, Discretize with the biot class
        self._discretize_biot()
        self._copy_biot_discretizations()

        # Next, discretize term on the matrix grid not covered by the Biot discretization,
        # i.e. the source, diffusion and mass terms
        filt = pp.assembler_filters.ListFilter(
            grid_list=[self._nd_grid()],
            variable_list=[self.scalar_variable],
            term_list=["source", "diffusion", "mass"],
        )
        self.assembler.discretize(filt=filt)

        # Then the temperature discretizations
        temperature_terms = ["source", "diffusion", "mass", self.advection_term]
        filt = pp.assembler_filters.ListFilter(
            grid_list=[self._nd_grid()],
            variable_list=[self.temperature_variable],
            term_list=temperature_terms,
        )
        self.assembler.discretize(filt=filt)

        # Coupling terms
        coupling_terms = [self.s2t_coupling_term, self.t2s_coupling_term]
        filt = pp.assembler_filters.ListFilter(
            grid_list=[self._nd_grid()],
            variable_list=[self.temperature_variable, self.scalar_variable],
            term_list=coupling_terms,
        )
        self.assembler.discretize(filt=filt)

        # Build a list of all edges, and all couplings
        edge_list: List[
            Union[
                Tuple[pp.Grid, pp.Grid],
                Tuple[pp.Grid, pp.Grid, Tuple[pp.Grid, pp.Grid]],
            ]
        ] = []
        for e, _ in self.gb.edges():
            edge_list.append(e)
            edge_list.append((e[0], e[1], e))
        if len(edge_list) > 0:
            filt = pp.assembler_filters.ListFilter(grid_list=edge_list)  # type: ignore
            self.assembler.discretize(filt=filt)

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for dim in range(0, self._Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) > 0:
                filt = pp.assembler_filters.ListFilter(grid_list=grid_list)
                self.assembler.discretize(filt=filt)

        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    @pp.time_logger(sections=module_sections)
    def _copy_biot_discretizations(self) -> None:
        """The Biot discretization is designed to discretize a single term of the
        grad_p type. It should not be difficult to generalize this, but pending such
        an update, the below code copies the discretization matrices from the flow
        related keywords to those of the temperature.

        """
        g: pp.Grid = self.gb.grids_of_dimension(self._Nd)[0]
        d: Dict = self.gb.node_props(g)
        beta = self._biot_beta(g)
        alpha = self._biot_alpha(g)

        # For grad p term of u equation
        weight_grad_t = beta / alpha

        # Account for scaling
        weight_grad_t *= self.temperature_scale / self.scalar_scale

        # Stabilization is derived from the grad p discretization
        weight_stabilization = beta / alpha
        weight_stabilization *= self.temperature_scale / self.scalar_scale

        # The stabilization terms appear in the T/p equations, whereof only the first
        # is divided by T_0_Kelvin.
        weight_stabilization *= 1 / self.T_0_Kelvin

        # Matrix dictionaries for the different subproblems
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
        weight_div_u = beta
        key_m_from_t = self.mechanics_temperature_parameter_key
        d[pp.DISCRETIZATION_MATRICES][key_m_from_t] = matrices_mt
        pp.initialize_data(
            g,
            d,
            key_m_from_t,
            {"biot_alpha": weight_div_u},
        )
        bc_dict = {"bc_values": self._bc_values_mechanics(g)}
        state = {key_m_from_t: bc_dict}
        pp.set_state(d, state)

    @pp.time_logger(sections=module_sections)
    def _update_iterate(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        Calls ContactMechanicsBiot version, and additionally updates the iterate solutions
        in d[pp.STATE][pp.ITERATE] are updated for the scalar variable, to be used
        for flux computations by compute_darcy_fluxes.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        # super().update_state(solution_vector)
        super()._update_iterate(solution_vector)

        dof_manager = self.dof_manager

        variable_names = []
        for pair in dof_manager.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in dof_manager.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    if name == self.mortar_scalar_variable:
                        mortar_p = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.edge_props(g)
                        data[pp.STATE][pp.ITERATE][
                            self.mortar_scalar_variable
                        ] = mortar_p.copy()
                else:
                    data = self.gb.node_props(g)

                    # g is a node (not edge)

                    # For the fractures, update the contact force
                    if name == self.scalar_variable:
                        p = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.node_props(g)
                        data[pp.STATE][pp.ITERATE][self.scalar_variable] = p.copy()
                    elif name == self.temperature_variable:
                        T = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.node_props(g)
                        data[pp.STATE][pp.ITERATE][self.temperature_variable] = T.copy()
