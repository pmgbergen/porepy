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
up with three scaling parameters. length_scaling allows solving on a unit size domain,
and result interpretation on e.g. a kilometer scale. pressure_scale and temperature_scale
may be used to solve for scaled pressure (p_scaled = p_physical/pressure_scale) and
temperature. For typical reservoir conditions, choosing a large (e.g. 1e6) pressure_scale
is a good place to start.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import porepy as pp
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations


class THMAdObjects(
    pp.models.contact_mechanics_biot_model.ContactMechanicsBiotAdObjects
):
    """Storage class for ad related objects.

    Stored objects include variables, compound ad operators and projections.
    """

    temperature: pp.ad.Variable
    advective_interface_flux: pp.ad.Variable
    conductive_interface_flux: pp.ad.Variable
    heat_conduction_discretization: Union[pp.ad.MpfaAd, pp.ad.TpfaAd]


class THM(pp.ContactMechanicsBiot):
    """This is a shell class for poroelastic contact mechanics problems.

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
            including in ParaView export.
        mortar_displacement_variable (str): Name assigned to the displacement variable
            on the fracture walls. Will be used throughout the simulations, including in
            ParaView export.
        contact_traction_variable (str): Name assigned to the variable for contact
            forces in the fracture. Will be used throughout the simulations, including
            in ParaView export.
        scalar_variable (str): Name assigned to the pressure variable. Will be used
            throughout the simulations, including in ParaView export.
        temperature_variable (str): Name assigned to the temperature variable. Will be used
            throughout the simulations, including in ParaView export.
        mortar scalar_variable (str): Name assigned to the interface scalar variable
            representing flux between grids. Will be used throughout the simulations,
            including in ParaView export.

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
        gb (pp.GridTree): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations have converged.
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
        self._use_ad = True

    def before_newton_iteration(self) -> None:
        """Re-discretize the nonlinear terms"""
        self.compute_fluxes()
        if self._use_ad:
            self._eq_manager.equations["interface_heat_advection"].discretize(self.gb)
            self._eq_manager.equations["subdomain_energy_balance"].discretize(self.gb)
        else:
            terms = [
                self.friction_coupling_term,
                self.advection_term,
                self.advection_coupling_term,
            ]
            filt = pp.assembler_filters.ListFilter(term_list=terms)
            self.assembler.discretize(filt=filt)

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

    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        self._set_scalar_parameters()
        self._set_temperature_parameters()
        self._set_mechanics_parameters()

    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        super()._set_mechanics_parameters()
        for g, d in self.gb:
            if g.dim == self._Nd:
                mech_params = d[pp.PARAMETERS][self.mechanics_parameter_key].copy()
                mech_params.update({"biot_alpha": self._biot_beta(g)})
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    mech_params,
                )

    def _set_scalar_parameters(self) -> None:
        """Set parameters for the pressure / mass conservation equation."""
        # Most values are handled as if this was a poroelastic problem
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

    def _set_temperature_parameters(self) -> None:
        """Parameters for the temperature equation."""
        tensor_scale: float = (
            self.temperature_scale / self.length_scale**2 / self.T_0_Kelvin
        )
        kappa: float = 1 * tensor_scale
        c_heat = 1
        mass_weight: float = c_heat * self.temperature_scale / self.T_0_Kelvin
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
            heat_capacity = c_heat * np.ones(g.num_cells) * self.temperature_scale
            advection_weight = heat_capacity / self.T_0_Kelvin
            advection_weight_faces = (
                c_heat * np.ones(g.num_faces) * self.temperature_scale / self.T_0_Kelvin
            )
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
                    "advection_weight": advection_weight,  # TODO: remove on ad purge
                    "advection_weight_boundary": advection_weight_faces,
                    "heat_capacity": heat_capacity,
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
            pp.initialize_data(
                e,
                data_edge,
                self.temperature_parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                },
            )

    def _bc_type_temperature(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type specification.

        Args:
            g: Grid representing a subdomain

        Returns:
            bc: BoundaryCondition object.

        Note that currently, Neumann values are collected by both advection
        and convection discretization. Consider dividing by two when assigning
        values using _bc_values_temperature.
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        bc = pp.BoundaryCondition(g, all_bf, "dir")
        return bc

    def _bc_values_temperature(self, g: pp.Grid) -> np.ndarray:
        """Boundary condition values.


        Args:
            g: Grid representing a subdomain

        Returns:
            values: cell-wise array.

        Note that currently, Neumann values are collected by both advection
        and convection discretization. Consider dividing by two.
        """
        values = np.zeros(g.num_faces)
        return values

    def _source_temperature(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    def _biot_beta(self, g: pp.Grid) -> Union[float, np.ndarray]:
        """
        TM coupling coefficient
        """
        if self._use_ad:
            return 1.0 * np.ones(g.num_cells)
        else:
            return 1.0

    def _scalar_temperature_coupling_coefficient(self, g: pp.Grid) -> float:
        """
        TH coupling coefficient
        """
        return -1.0

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

    def _assign_ad_variables(self) -> None:
        """Assign variables to self._ad


        Assigns the following attributes to self._ad in addition to those set by
        the parent class:
            pressure: primary variable in all subdomains.
            interface_flux: Primary variable on interfaces of codimension 1 (usually
                all interfaces).

        Returns
        -------
        None

        """
        super()._assign_ad_variables()
        subdomains: List[pp.Grid] = [g for g, _ in self.gb]

        interfaces = [e for e, d in self.gb.edges() if d["mortar_grid"].codim == 1]
        # Primary variables on Ad form
        self._ad.temperature = self._eq_manager.merge_variables(
            [(g, self.temperature_variable) for g in subdomains]
        )
        self._ad.advective_interface_flux = self._eq_manager.merge_variables(
            [(e, self.mortar_temperature_advection_variable) for e in interfaces]
        )
        self._ad.conductive_interface_flux = self._eq_manager.merge_variables(
            [(e, self.mortar_temperature_variable) for e in interfaces]
        )

    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        # Call parent class for discretizations for the poroelastic system.
        super()._assign_discretizations()
        if self._use_ad:
            return
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
        grad_t_disc = pp.GradP(key_mt)

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
                            "stabilization": stabilization_disc_t,
                            "source": source_disc_t,
                        },
                        # grad T term in the momentum equation
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

    def _assign_equations(self):
        """Assign equations for mixed-dimensional flow and deformation.

        The following equations are assigned to the equation manager by the
        call to super. Some of the methods producing these equations are
        modified in this class to account for coupling terms from temperature,
        which appear in _stress and _subdomain_flow_equation
            "momentum" in the nd subdomains
            "contact_mechanics_normal" in all fracture subdomains
            "contact_mechanics_tangential" in all fracture subdomains
            "force_balance" at the matrix-fracture interfaces

            "subdomain_flow" in all subdomains
            "interface_flow" on all interfaces of codimension 1

        The following equations are added by this class:
             "subdomain_energy_balance" in all subdomains
             "interface_heat_conduction" on all interfaces of codimension 1
             "interface_heat_advection" on all interfaces of codimension 1

        Returns
        -------
        None.

        """
        super()._assign_equations()

        # Now, assign the two flow equations not present in the parent model.

        subdomains: List[pp.Grid] = [g for g, _ in self.gb]
        interfaces = [e for e, d in self.gb.edges() if d["mortar_grid"].codim == 1]

        # Construct equations
        subdomain_energy_balance_eq: pp.ad.Operator = (
            self._subdomain_energy_balance_equation(subdomains)
        )
        interface_heat_conduction_eq: pp.ad.Operator = (
            self._interface_heat_conduction_equation(interfaces)
        )
        interface_heat_advection_eq: pp.ad.Operator = (
            self._interface_heat_advection_equation(interfaces)
        )
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_energy_balance": subdomain_energy_balance_eq,
                "interface_heat_conduction": interface_heat_conduction_eq,
                "interface_heat_advection": interface_heat_advection_eq,
            },
        )

    def _subdomain_flow_equation(self, subdomains: List[pp.Grid]):
        """Mass balance equation for slightly compressible flow in a deformable medium.

        Method calls super and adds the TH coupling term.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """
        # The equation is the same as for the isothermal model, except that the temperature
        # dependency in the density will give one more term.
        eq = super()._subdomain_flow_equation(subdomains)
        discr = pp.ad.MassMatrixAd(self.t2s_parameter_key, subdomains)
        eq += discr.mass * (
            self._ad.temperature - self._ad.temperature.previous_timestep()
        )
        return eq

    def _heat_flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Heat flux.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains for which heat fluxes are defined, normally all.

        Returns
        -------
        flux : pp.ad.Operator
            Flux on ad form.

        Note:
            The ad flux discretization used here is stored for consistency with
            self._interface_heat_conduction_equation, where self._ad.flux_discretization
            is applied.
        """
        ad = self._ad
        # Discretization of advective flux
        upwind_ad = pp.ad.UpwindAd(self.temperature_parameter_key, subdomains)
        # Discretization of the conductive flux
        flux_discr = pp.ad.MpfaAd(self.temperature_parameter_key, subdomains)

        # Store to ensure consistency in interface flux
        ad.heat_conduction_discretization = flux_discr

        # (Fluid) enthalpy
        enthalpy = self._enthalpy(subdomains)

        # Heat capacity of the fluid on the boundary. This is needed to transfer
        # Dirichlet conditions (given in temperature) to an expression for enthalpy.
        heat_capacity_boundary = pp.ad.ParameterMatrix(
            self.temperature_parameter_key,
            "advection_weight_boundary",
            subdomains,
        )
        # Boundary values, set either as enhtalpy fluxes, or temperature values.
        bc_values = pp.ad.ParameterArray(
            self.temperature_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        fluid_flux = self._fluid_flux(subdomains)

        flux = (
            # The conductive flux acts on temperature differences, not enthalpy.
            flux_discr.flux * ad.temperature
            + flux_discr.bound_flux * bc_values  # Conductive boundary fluxes
            + fluid_flux * (upwind_ad.upwind * enthalpy)  # Advective flux
            # Dirichlet boundary data is assumed to be given in temperature
            - upwind_ad.bound_transport_dir
            * fluid_flux
            * (heat_capacity_boundary * bc_values)
            # Advective flux coming from lower-dimensional subdomains
            - upwind_ad.bound_transport_neu
            * (
                ad.mortar_projections_scalar.mortar_to_primary_int
                * ad.advective_interface_flux
                + bc_values
            )
            # Conductive flux from lower-dimensional subdomains
            + flux_discr.bound_flux
            * ad.mortar_projections_scalar.mortar_to_primary_int
            * ad.conductive_interface_flux
        )
        return flux

    def _subdomain_energy_balance_equation(self, subdomains: List[pp.Grid]):
        """Energy balance equation.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """

        ad = self._ad
        g_frac: List[pp.Grid] = self.gb.grids_of_dimension(self._Nd - 1).tolist()
        mass_discr = pp.ad.MassMatrixAd(self.temperature_parameter_key, subdomains)

        heat_source = pp.ad.ParameterArray(
            param_keyword=self.temperature_parameter_key,
            array_keyword="source",
            grids=subdomains,
        )
        scalar_discr = pp.ad.MassMatrixAd(self.s2t_parameter_key, subdomains)

        div_scalar = pp.ad.Divergence(grids=subdomains)
        biot_accumulation_primary = self._biot_terms_heat([self._nd_grid()])
        heat_flux = self._heat_flux(subdomains)

        # Accumulation term on all subdomains. Contributions from both pressure and
        # temperature dependency in the density.
        accumulation_all = mass_discr.mass * (
            ad.temperature - ad.temperature.previous_timestep()
        ) + scalar_discr.mass * (
            self._ad.pressure - self._ad.pressure.previous_timestep()
        )

        # Time scaling of flux terms, including interdimensional
        eq = (
            ad.time_step
            * (
                # Heat fluxes internal to the subdomain, and from lower-dimensional
                # neighboring subdomains (see self._heat_flux())
                div_scalar * heat_flux
                # Conductive heat flux from higher-dimensional neighbors
                - ad.mortar_projections_scalar.mortar_to_secondary_int
                * ad.conductive_interface_flux
                # Advective heat flux from higher-dimensional neighbors
                - ad.mortar_projections_scalar.mortar_to_secondary_int
                * ad.advective_interface_flux
            )
            + accumulation_all
            + ad.subdomain_projections_scalar.cell_prolongation([self._nd_grid()])
            * biot_accumulation_primary
            - heat_source
        )
        if len(subdomains) > 1:
            # Volume change term in the fracture.
            heat_capacity = pp.ad.ParameterMatrix(
                self.temperature_parameter_key,
                "heat_capacity",
                g_frac,
            )
            volume = self._volume_change(g_frac)
            volume.set_name("volume")
            accumulation_fracs = heat_capacity * volume
            eq -= (
                ad.subdomain_projections_scalar.cell_prolongation(g_frac)
                * accumulation_fracs
            )
        eq.set_name("Subdomain energy balance")

        return eq

    def _interface_heat_conduction_equation(
        self, interfaces: List[Tuple[pp.Grid, pp.Grid]]
    ):
        """Equation for conductive interface heat fluxes.

        Parameters
        ----------
        interfaces : List[Tuple[pp.Grid, pp.Grid]]
            DESCRIPTION.

        Returns
        -------
        interface_heat_conduction_eq : pp.ad.Operator
            The interface equation on ad form.

        """
        # Interface equation: \lambda = -\kappa (t_l - t_h)
        # Robin_ad.mortar_discr represents -\kappa. The involved term is
        # reconstruction of T_h on internal boundary, which has contributions
        # from cell center temperature, external boundary and interface flux
        # on internal boundaries (including those corresponding to "other"
        # fractures).

        # Create list of subdomains. Ensure matrix grid is present so that bc
        # and vector_source_subdomains are consistent with ad.heat_conduction_discretization
        subdomains = [self._nd_grid()]
        # All all subdomains, then uniquify the list
        for interface in interfaces:
            for sd in interface:
                if sd not in subdomains:
                    subdomains.append(sd)

        ad = self._ad
        flux_discr = ad.heat_conduction_discretization
        interface_discr = pp.ad.RobinCouplingAd(
            self.temperature_parameter_key, interfaces
        )

        bc = pp.ad.ParameterArray(
            self.temperature_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )

        # Construct primary (higher-dimensional) temperature
        # IMPLEMENTATION NOTE: this could possibly do with a sub-method
        t_primary = (
            flux_discr.bound_pressure_cell * ad.temperature
            + flux_discr.bound_pressure_face
            * ad.mortar_projections_scalar.mortar_to_primary_int
            * ad.conductive_interface_flux
            + flux_discr.bound_pressure_face * bc
        )
        # Project the two temperatures to the interface and equate with interface flux
        interface_heat_conduction_eq: pp.ad.Operator = (
            interface_discr.mortar_discr
            * (
                ad.mortar_projections_scalar.primary_to_mortar_avg * t_primary
                - ad.mortar_projections_scalar.secondary_to_mortar_avg * ad.temperature
            )
            + ad.conductive_interface_flux
        )
        return interface_heat_conduction_eq

    def _interface_heat_advection_equation(
        self, interfaces: List[Tuple[pp.Grid, pp.Grid]]
    ):
        """Equation for advective interface heat fluxes.

        Parameters
        ----------
        interfaces : List[Tuple[pp.Grid, pp.Grid]]
            DESCRIPTION.

        Returns
        -------
        interface_heat_conduction_eq : pp.ad.Operator
            The interface equation on ad form.

        Equation:

            advective_flux = fluid_flux * upwind_weight * t

        """
        # Create list of subdomains. Ensure matrix grid is present so that bc
        # and vector_source_subdomains are consistent with ad.flux_discretization
        subdomains = [self._nd_grid()]
        for interface in interfaces:
            for sd in interface:
                if sd not in subdomains:
                    subdomains.append(sd)

        ad = self._ad
        fluid_flux = self._ad.interface_flux
        discr = pp.ad.UpwindCouplingAd(self.temperature_parameter_key, interfaces)
        proj = ad.mortar_projections_scalar
        trace = pp.ad.Trace(subdomains)
        enthalpy = self._enthalpy(subdomains)
        # Project the two temperatures to the interface and equate with interface flux
        interface_heat_conduction_eq: pp.ad.Operator = (
            fluid_flux
            * (
                discr.upwind_primary
                * proj.primary_to_mortar_avg
                * trace.trace
                * enthalpy
                + discr.upwind_secondary * proj.secondary_to_mortar_avg * enthalpy
            )
            - ad.advective_interface_flux
        )
        return interface_heat_conduction_eq

    def _enthalpy(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Ad representation of enthalpy.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            List of subdomains.

        Returns
        -------
        enthalpy : pp.ad.Operator
            enthalpy ad operator.

        """
        # The enthalpy in this implementation is modeled as the product of tempearature
        # and the heat capacity of the fluid.
        heat_capacity = pp.ad.ParameterMatrix(
            self.temperature_parameter_key,
            "advection_weight",
            subdomains,
        )
        enthalpy = heat_capacity * self._ad.temperature
        enthalpy.set_name("enthalpy")
        return enthalpy

    def _stress(
        self,
        matrix_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """Ad representation of thermoporomechanical stress.


        Parameters
        ----------
        matrix_subdomains : List[pp.Grid]
            List of N-dimensional grids, usually with a single entry.

        Returns
        -------
        stress : pp.ad.Operator
            Stress operator.

        """

        poromechanical_stress = super()._stress(matrix_subdomains)
        discr = pp.ad.BiotAd(
            self.mechanics_temperature_parameter_key,
            matrix_subdomains,
            self.temperature_parameter_key,
        )
        thermal_stress: pp.ad.Operator = (
            discr.grad_p
            * self._ad.subdomain_projections_scalar.cell_restriction(matrix_subdomains)
            * self._ad.temperature
        )
        # Hitherto, we have assumed we operate on self.temperature_variable = T-T_ref.
        # If this assumption is overturned, add the following to stress (see biot model):
        # - discr.grad_p * t_reference

        thermal_stress.set_name("thermal_stress")
        stress: pp.ad.Operator = poromechanical_stress + thermal_stress
        stress.set_name("thermoporomechanical_stress")
        return stress

    def _biot_terms_heat(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Biot terms, div(u) and stabilization.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Matrix subdomains, expected to have length=1.

        Returns
        -------
        biot_terms : pp.ad.Operator
            Ad operator representing d/dt div(u) and stabilization terms of the
            Biot flow equation in the matrix.


        """
        ad = self._ad
        parameter_key = self.temperature_parameter_key
        ad_variable = ad.temperature

        div_u_discr = pp.ad.DivUAd(
            self.mechanics_temperature_parameter_key,
            grids=subdomains,
            mat_dict_keyword=parameter_key,
        )
        stabilization_discr = pp.ad.BiotStabilizationAd(parameter_key, subdomains)
        biot_alpha = pp.ad.ParameterMatrix(
            parameter_key,
            array_keyword="biot_alpha",
            grids=subdomains,
        )
        bc_mech = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        bc_mech_previous = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="bc_values_previous_timestep",
            grids=subdomains,
        )
        # The "div_u" really represents the time increment d/dt div(u), thus
        # all contributions are defined on differences between current and previous
        # state. There are three components: matrix, external boundary and
        # internal boundary (fractures). The last term requires projection of
        # displacements from interfaces
        matrix_div_u: pp.ad.Operator = div_u_discr.div_u * (
            ad.displacement - ad.displacement.previous_timestep()
        )
        external_boundary_div_u: pp.ad.Operator = div_u_discr.bound_div_u * (
            bc_mech - bc_mech_previous
        )
        internal_boundary_div_u: pp.ad.Operator = (
            div_u_discr.bound_div_u
            * ad.subdomain_projections_vector.face_restriction(subdomains)
            * ad.mortar_projections_vector.mortar_to_primary_int
            * (
                ad.interface_displacement
                - ad.interface_displacement.previous_timestep()
            )
        )
        div_u_terms: pp.ad.Operator = biot_alpha * (
            matrix_div_u + external_boundary_div_u + internal_boundary_div_u
        )
        div_u_terms.set_name("div_u")

        # The stabilization term is also defined on a time increment, but only
        # considers the matrix subdomain and no boundary contributions.
        stabilization_term: pp.ad.Operator = (
            stabilization_discr.stabilization
            * ad.subdomain_projections_scalar.cell_restriction(subdomains)
            * (ad_variable - ad_variable.previous_timestep())
        )
        stabilization_term.set_name("Biot stabilization")

        biot_terms: pp.ad.Operator = div_u_terms + stabilization_term
        return biot_terms

    def _set_ad_objects(self) -> None:
        """Sets the storage class self._ad


        Returns
        -------
        None

        """
        self._ad = THMAdObjects()

    def _initial_condition(self) -> None:
        """
        In addition to the values set by the parent class, we set initial value for the
        Darcy flux parameter.
        """
        super()._initial_condition()
        key = self.temperature_parameter_key
        for g, d in self.gb:
            pp.initialize_data(g, d, key, {"darcy_flux": np.zeros(g.num_faces)})

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(e, d, key, {"darcy_flux": np.zeros(mg.num_cells)})

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
        d[pp.PARAMETERS][key]["bc_values_previous_timestep"] = d[pp.PARAMETERS][key][
            "bc_values"
        ].copy()
        d[pp.PARAMETERS][key_t]["bc_values_previous_timestep"] = d[pp.PARAMETERS][
            key_t
        ]["bc_values"].copy()

    def _discretize(self) -> None:
        """Discretize all terms"""
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.gb)
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb, self.dof_manager)

        if self._use_ad:
            self._eq_manager.discretize(self.gb)
            # self._copy_biot_discretizations()
            return
        # else:
        #     self.assembler.discretize()
        #     self._copy_biot_discretizations()
        #     return
        # Discretization is a bit cumbersome, as the Biot discretization removes the
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

    def _copy_biot_discretizations(self) -> None:
        """The Biot discretization is designed to discretize a single term of the
        grad_p type. It should not be difficult to generalize this, but pending such
        an update, the below code copies the discretization matrices from the flow
        related keywords to those of the temperature.

        The Biot class incorporates the "alpha" coefficient in the discretization
        matrices stored for the grad p term and the stabilization, but not for
        the div_u term (multiplied upon assembly). Consequently, after copying the
        matrices from flow, we multiply the former by weight = beta / alpha.

        This method assigns the following discretization matrices for the keyword
        self.temperature_parameter_key:
            "div_u", "bound_div_u", "biot_stabilization"
        and these for self.mechanics_temperature_parameter_key:
            "grad_p", "bound_displacement_pressure"

        Implementation note:
        For now, the ad version of the class discretizes the temperature terms
        separately. For now, we keep this copy method and assert that the two are
        equivalent. TODO: Purge after removing non-ad?
        """
        key_m_from_t = self.mechanics_temperature_parameter_key

        g: pp.Grid = self.gb.grids_of_dimension(self._Nd)[0]
        d: Dict = self.gb.node_props(g)
        beta = self._biot_beta(g)
        alpha = self._biot_alpha(g)

        weight = beta / alpha

        # Account for scaling
        weight *= self.temperature_scale / self.scalar_scale

        if self._use_ad:
            # The ignored typing has to do with legacy scalar coupling coefficient
            # definition and will be purged along with non-ad version of self.
            assert np.all(np.isclose(weight, weight[0]))  # type:ignore
            weight = weight[0]  # type:ignore

        # Matrix dictionaries for the different sub-problems
        matrices_s = d[pp.DISCRETIZATION_MATRICES][self.scalar_parameter_key]
        matrices_ms = d[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        matrices_t = d[pp.DISCRETIZATION_MATRICES][self.temperature_parameter_key]
        if key_m_from_t not in d[pp.DISCRETIZATION_MATRICES]:
            d[pp.DISCRETIZATION_MATRICES][key_m_from_t] = dict()
        matrices_mt = d[pp.DISCRETIZATION_MATRICES][key_m_from_t]
        for key, w in zip(
            ["div_u", "bound_div_u", "biot_stabilization"], [1, 1, weight]
        ):
            new = matrices_s[key] * w
            if self._use_ad:
                old = matrices_t[key]
                assert np.all(np.isclose(new.data, old.data))
            matrices_t.update({key: new})
        for key in ["grad_p", "bound_displacement_pressure"]:
            new = matrices_ms[key].copy() * weight
            if self._use_ad:
                old = matrices_mt[key]
                assert np.all(np.isclose(new.data, old.data))

            matrices_mt.update({key: new})
