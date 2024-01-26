"""Contains a general composit flow class without reactions

The grid, expected Phases and components can be modified in respective methods.

"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pypardiso
import scipy.sparse as sps

import porepy as pp
from porepy.composite.base import Component, Compound
from porepy.composite.flash import logger


class CompositionalFlowModel:
    """Non-isothermal, compositional flow with gas and liquid phase.

    The fluid mixture is created during :meth:`prepare_simulation`
    using the sub-routine :meth:`set_composition`,
    which can be inherited and overwritten.

    Parameters:
        params: general model parameters including

            - 'file_name' (str): name of file for exporting simulation results
            - 'folder_name' (str): abs. path to directory for saving simulation results

    """

    def __init__(self, params: dict, verbosity: int = 0) -> None:

        # setting logging verbosity
        self._verbosity: int = verbosity
        if verbosity:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # projections/ prolongation from global dofs to primary and secondary variables
        # will be set during `prepare simulations`
        self._prolong_prim: sps.spmatrix
        self._prolong_sec: sps.spmatrix
        # all unknowns of the system
        self._system_vars: list[str]

        # data for Schur complement expansion
        self._schur_expansion = (None, None, None)

        self._source_arrays: dict[Component, pp.ad.DenseArray] = dict()
        """Dictionary of source arrays per component."""

        self._enthalpy_source: pp.ad.DenseArray
        """Constant array representing the injected extensive enthalpy."""

        self._porosity_ad: pp.ad.DenseArray
        """Constant array representing the density."""

        self._pressure_equation_name: str = "pressure-equation"
        """Name of the global pressure equation."""

        self._energy_equation_name: str = "energy-equation"
        """Name of the global energy balance equation."""

        self._mass_balance_name: str = "mass-balance"
        """(Partial) Name of the mass balance equation per component."""

        ### PUBLIC

        ## flow and transport variables
        self.p: pp.ad.MixedDimensionalVariable
        """Pressure variable in MPa."""
        self.h: pp.ad.MixedDimensionalVariable
        """Spec. Enthalpy (of the fluid) variable in kJ / mol."""
        self.T: pp.ad.MixedDimensionalVariable
        """Temperature variable in Kelvin."""

        ### COMPOSITION SETUP
        self.fluid: pp.composite.NonReactiveMixture
        self.flash: pp.composite.FlashNR
        self.species: list[str] = ["H2O", "CO2"]
        """List of names of present species (or chemical formulas) such that they
        can be loaded by the composite subpackage."""

        self.num_cells: list[int] = [10, 5]  # 50, 15
        self.phys_dim: list[int] = [10, 5]
        """Number of cells in each direction, physical dimensions will be set equal.
        I.e. unit cells."""

        self.params: dict = params
        """Parameter dictionary passed at instantiation."""
        self.converged: bool = False
        """Indicator if current Newton-step converged."""
        self.dt: pp.composite.AdProperty = pp.composite.AdProperty("timestep")
        """Timestep size."""
        self.dt.value = 1.

        self.porosity = 1.
        """Base porosity of model domain."""
        self.permeability = 1.
        """Base permeability of model domain."""

        self.pressure_scale: float = 1e6
        """Multiplicative factor to convert the pressure from the flow scale to
        the thermodynamic scale (Pa)."""
        self.energy_scale: float = 1e3
        """Multiplicative factor to convert the enthalpy from the flow scale to the
        thermodynamic scale (J / mol)."""

        ## Initial Conditions
        self.initial_pressure: float = 10.
        """Initial pressure in the domain in MPa."""
        
        self.initial_component_fractions: list[float] = [0.99, 0.01]
        """Contains per component in composition the initial feed fraction."""
        self.initial_solute_fractions: list[float] = [0.01]
        """Contains initial solute fractions per compound and its solutes"""

        ## Injection parameters
        self.injection_pressure = 0
        """Pressure at injection in MPa."""
        self.injection_temperature = 0  # pp.composite.T_REF + 30  # 70 deg C
        """Temperature of injected mass for source terms in Kelvin."""

        ## Boundary conditions
        self.outflow_boundary_pressure: float = self.initial_pressure
        """Dirichlet boundary pressure for the outflow in MPA for the advective flux."""
        self.inflow_boundary_pressure: float = 12.
        """Dirichlet boundary pressure for the inflow in MPa for the advective flux."""
        self.inflow_boundary_temperature: float = 450
        """Temperature at the inflow boundary for the advective flux."""
        self.inflow_boundary_feed: list[float] = [0.99, 0.01]
        """Contains per component in composition the feed fraction at the inflow
        boundary.

        Always provide at least trace amounts of each component to avoid washing-out
        effects.

        """
        self.inflow_boundary_solute_feed: list[float] = [0.08]
        """Contains inflow solute fractions per compound and its solutes"""
        self.heated_boundary_temperature = 600
        """Dirichlet boundary temperature in Kelvin for the conductive flux,
        bottom boundary."""
        self.upper_boundary_temperature: float = 500
        """Initial temperature in the domain in K."""
        self.injection_feed: list[float] = [0.0, 0.0]
        """Contains per component in composition the amount of injected moles.

        Always included trace amounts of each components to avoid the
        washing-out effect."""

        ## TO BE COMPUTED after obtaining the equilibrium at the boundary.
        self.inflow_boundary_advective_weights: dict[Component, float] = dict()
        """Contains per component (key) the scalar part of the advective flux on
        the inflow boundary.
        To be set in :meth:`set_composition` after computing the equilibrium at the
        inflow.
        """
        self.inflow_boundary_advective_weights_solutes: dict[pp.composite.ChemicalSpecies, float] = dict()
        """Contains per solute (key) the scalar part of the advective transport for the solute transport."""
        self.inflow_boundary_advective_weight_pressure: float
        """Value for scalar part of the advective flux on the
        inflow boundary for the global pressure equation.
        To be set in :meth:`set_composition` after computing the equilibrium at the
        inflow.
        """
        self.inflow_boundary_advective_weight_energy: float
        """Value for scalar part of advective flux in energy equation.
        To be set in :meth:`set_composition`.
        """
        self.injected_ext_enthalpy: float
        """The amount of injected extensive enthalpy.
        To be set in :meth:`set_composition` after an equilibrium computation at the
        injection point."""

        ## grid and AD system
        self.domain: pp.MixedDimensionalGrid
        """Computational domain, to be set in :meth:`create_grid`."""
        self.create_grid()
        self.ad_system: pp.ad.EquationSystem = pp.ad.EquationSystem(self.domain)
        """AD System for this model."""
        # exporter
        self._exporter: pp.Exporter = pp.Exporter(
            self.domain,
            params["file_name"],
            folder_name=params["folder_name"],
            export_constants_separately=False,
        )

        self.system_names: dict[str, list] = {
            "primary-equations": list(),
            "secondary-equations": list(),
            "primary-variables": list(),
            "secondary-variables": list(),
        }
        """Dictionary containing names of primary and secondary equations and variables.
        """

        self.transport_equations: list[str] = list()
        """List of names of transport equations. This includes

        - ``num_comp - 1`` mass balances for fluid components except reference component
        - 1 pressure equation
        - 1 energy balance formulated for enthalpy.

        """

        # Parameter keywords
        self.flow_keyword: str = "flow"
        """Flow keyword for storing BC information in data dictionaries."""
        self.mass_keyword: str = "mass"
        """Mass keyword for storing BC information in data dictionaries."""
        self.energy_keyword: str = "energy"
        """Energy keyword for storing BC information in data dictionaries."""
        self.upwind_keyword: str = "upwind"
        """Upwinding keyword for storing BC information in data dictionaries."""

        ## References to discretization operators
        # they will be set during `prepare_simulation`
        self.mass_matrix: pp.ad.MassMatrixAd
        self.div: pp.ad.Divergence
        self.advective_flux: pp.ad.MpfaAd
        self.advective_upwind_pressure: pp.ad.UpwindAd
        self.advective_upwind_energy: pp.ad.UpwindAd
        self.advective_upwind_component: dict[Component, pp.ad.UpwindAd] = dict()
        self.advective_upwind_solute: dict[pp.composite.ChemicalSpecies, pp.ad.UpwindAd] = dict()
        self.advective_bc_pressure: pp.ad.DenseArray
        self.advective_weight_bc_pressure: pp.ad.DenseArray
        self.advective_weight_bc_energy: pp.ad.DenseArray
        self.advective_weight_bc_component: dict[Component, pp.ad.DenseArray] = dict()
        self.advective_weight_bc_solute: dict[pp.composite.ChemicalSpecies, pp.ad.DenseArray] = dict()
        self.conductive_flux: pp.ad.MpfaAd
        self.conductive_upwind_energy: pp.ad.UpwindAd
        self.conductive_bc_temperature: pp.ad.DenseArray
        self.conductive_weight_bc_energy: pp.ad.DenseArray

    def create_grid(self) -> None:
        """Assigns a cartesian grid as computational domain.
        Overwrites/sets the instance variables 'mdg'.
        """
        phys_dims = self.phys_dim
        n_cells = self.num_cells
        bounding_box_points = np.array([[0, phys_dims[0]], [0, phys_dims[1]]])
        self.box = pp.geometry.domain.bounding_box_of_point_cloud(bounding_box_points)
        sg = pp.CartGrid(n_cells, phys_dims)
        self.domain = pp.MixedDimensionalGrid()
        self.domain.add_subdomains(sg)
        self.domain.compute_geometry()

    def _domain_boundary_sides(
        self, sd: pp.Grid, tol: Optional[float] = 1e-10
    ) -> pp.domain.DomainSides:
        # Get domain boundary sides
        box = self.box
        east = np.abs(box["xmax"] - sd.face_centers[0]) <= tol
        west = np.abs(box["xmin"] - sd.face_centers[0]) <= tol
        if self.domain.dim_max() == 1:
            north = np.zeros(sd.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = np.abs(box["ymax"] - sd.face_centers[1]) <= tol
            south = np.abs(box["ymin"] - sd.face_centers[1]) <= tol
        if self.domain.dim_max() < 3:
            top = np.zeros(sd.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = np.abs(box["zmax"] - sd.face_centers[2]) <= tol
            bottom = np.abs(box["zmin"] - sd.face_centers[2]) <= tol
        all_bf = sd.get_boundary_faces()

        return all_bf, east, west, north, south, top, bottom

    def create_mixture(self) -> pp.composite.ThermodynamicState:
        """Defines the composition for which the simulation should be run and performs
        the initial isothermal equilibrium calculations.

        Set initial values for p, T and feed here.

        Use this method to inherit and override the composition,
        while keeping the (generic) rest of the model.

        Assumes the model domain ``mdg`` is already set.

        """
        ## creating composition
        species = pp.composite.load_species(self.species)

        comps = [
            pp.composite.peng_robinson.NaClBrine.from_species(species[0]),
            pp.composite.peng_robinson.CO2.from_species(species[1]),
        ]

        phases = [
            pp.composite.Phase(
                pp.composite.peng_robinson.PengRobinson(gaslike=False), name="L"
            ),
            pp.composite.Phase(
                pp.composite.peng_robinson.PengRobinson(gaslike=True), name="G"
            ),
        ]

        self.fluid = pp.composite.NonReactiveMixture(comps, phases)
        self.fluid.set_up(ad_system=self.ad_system)
        ## setting thermodynamic state in terms of p-T-z
        nc = self.domain.num_subdomain_cells()
        vec = np.ones(nc)
        T_vals = self.ad_system.get_variable_values([self.T.name], time_step_index=0)

        # initial fractions
        idx = 0
        for i, comp in enumerate(self.fluid.components):
            if comp != self.fluid.reference_component:
                self.ad_system.set_variable_values(
                    self.initial_component_fractions[i] * vec,
                    variables=[comp.fraction.name],
                    iterate_index=0,
                    time_step_index=0,
                )
            if isinstance(comp, Compound):
                for solute in comp.solutes:
                    self.ad_system.set_variable_values(
                        self.initial_solute_fractions[idx] * vec,
                        variables=[comp.solute_fraction_of[solute].name],
                        iterate_index=0,
                        time_step_index=0,
                    )
                    idx += 1
        # Setting initial molalities
        comps[0].compute_molalities(vec * self.initial_solute_fractions[0], store=True)

        ## initialize and construct flasher
        logger.info("Computing initial equilibrium ...\n")
        self.flash = pp.composite.FlashNR(self.fluid)
        self.flash.use_armijo = True
        self.flash.armijo_parameters["rho"] = 0.99
        self.flash.armijo_parameters["j_max"] = 50
        self.flash.armijo_parameters["return_max"] = True
        self.flash.newton_update_chop = 1.0
        self.flash.tolerance = 1e-6
        self.flash.max_iter = 120
        _, initial_state = self.flash.flash(
            state={
                "p": vec * self.initial_pressure * self.pressure_scale,
                "T": T_vals,
            },
            feed=[comps[1].fraction],
            verbosity=self._verbosity,
        )

        ## configuration at boundary
        p = np.ones(1) * self.inflow_boundary_pressure * self.pressure_scale
        T = np.ones(1) * self.inflow_boundary_temperature
        feed = [np.ones(1) * self.inflow_boundary_feed[i] for i in range(len(comps))]
        # boundary molalities
        comps[0].compute_molalities(np.ones(1) * self.inflow_boundary_solute_feed[0], store=True)
        logger.info(f"Computing inflow boundary equilibrium ...\n")
        _, boundary_state = self.flash.flash(
            state={"p": p, "T": T},
            feed=feed,
            verbosity=self._verbosity,
        )
        rho = boundary_state.rho[0]
        self.inflow_boundary_advective_weight_energy = rho * boundary_state.h[0] / self.energy_scale
        self.inflow_boundary_advective_weight_pressure = rho

        phase_densities = []
        for j, phase in enumerate(phases):
            if boundary_state.s[j][0] > 0:
                prop = phase.compute_properties(p, T, boundary_state.X[j], store=False)
                phase_densities.append(prop.rho[0])
            else:
                phase_densities.append(0.0)

        self.inflow_boundary_advective_weights = dict()
        for i, comp in enumerate(comps):
            advective_weight = sum(
                [
                    boundary_state.s[j] * phase_densities[j] * boundary_state.X[j][i]
                    for j in range(len(phases))
                ]
            )
            self.inflow_boundary_advective_weights.update({comp: advective_weight[0]})
        # Advective weight for solute
        self.inflow_boundary_advective_weights_solutes = dict()
        self.inflow_boundary_advective_weights_solutes.update(
            {comps[0].solutes[0]: self.inflow_boundary_advective_weights[comps[0]] * self.inflow_boundary_solute_feed[0]}
        )

        ## Configuration at injection
        if self.injection_pressure and self.injection_temperature:
            p = np.ones(1) * self.injection_pressure * self.pressure_scale
            T = np.ones(1) * self.injection_temperature
            feed = [np.ones(1) * self.injection_feed[i] for i in range(len(comps))]
            logger.info(f"Computing injection equilibrium ...\n")
            _, injection_state = self.flash.flash(
                state={"p": p, "T": T},
                feed=feed,
                verbosity=self._verbosity,
            )
            self.injected_ext_enthalpy = (injection_state.rho * injection_state.h)[0] / self.energy_scale
        else:
            self.injected_ext_enthalpy = 0.0


        # Re-store molalities to initial state
        comps[0].compute_molalities(vec * self.initial_solute_fractions[0], store=True)

        return initial_state

    def prepare_simulation(self) -> None:
        """Preparing essential simulation configurations.

        Method needs to be called prior to applying any solver.

        """
        # creating primary transport variables

        subdomains = self.domain.subdomains()

        self.p = self.ad_system.create_variables("pressure", subdomains=subdomains)
        self.h = self.ad_system.create_variables("enthalpy", subdomains=subdomains)
        self.T = self.ad_system.create_variables("temperature", subdomains=subdomains)

        # setting initial values for pressure and temperature
        T_vals = []
        for sd in subdomains:
            T_vals = self._T_gradient(sd)
        T_vals = np.hstack(T_vals)
        vec = np.ones(self.domain.num_subdomain_cells())
        self.ad_system.set_variable_values(
            vec * self.initial_pressure,
            variables=[self.p.name],
            iterate_index=0,
            time_step_index=0,
        )
        self.ad_system.set_variable_values(
            T_vals,
            variables=[self.T.name],
            iterate_index=0,
            time_step_index=0,
        )
        # must set initial values for AD to work
        self.ad_system.set_variable_values(
            vec * 0.0,
            variables=[self.h.name],
            iterate_index=0,
            time_step_index=0,
        )

        initial_state = self.create_mixture()
        # setting initial enthalpy values after equilibrium as been computed
        self.ad_system.set_variable_values(
            initial_state.h / self.energy_scale,
            variables=[self.h.name],
            iterate_index=0,
            time_step_index=0,
        )
        # setting initial molar fractions
        for j, phase in enumerate(self.fluid.phases):
            if phase != self.fluid.reference_phase:
                self.ad_system.set_variable_values(
                    initial_state.y[j],
                    variables=[phase.fraction.name],
                    iterate_index=0,
                    time_step_index=0,
                )
                self.ad_system.set_variable_values(
                    initial_state.s[j],
                    variables=[phase.saturation.name],
                    iterate_index=0,
                    time_step_index=0,
                )
            for i, comp in enumerate(self.fluid.components):
                self.ad_system.set_variable_values(
                    initial_state.X[j][i],
                    variables=[phase.fraction_of[comp].name],
                    iterate_index=0,
                    time_step_index=0,
                )

        # getting the names of primary and secondary variables
        primary_vars = [self.p.name, self.h.name] + self.fluid.feed_fraction_variables
        for _, solute_fractions in self.fluid.solute_fraction_variables.items():
            primary_vars += solute_fractions
        secondary_vars = [self.T.name] + self.fluid.saturation_variables
        secondary_vars += self.fluid.molar_fraction_variables

        # defining system variables
        self.system_names["primary-variables"] = primary_vars
        self.system_names["secondary-variables"] = secondary_vars
        self._system_vars = primary_vars + secondary_vars

        # prepare prolongations for the solver
        self._prolong_prim = self.ad_system.projection_to(primary_vars).transpose()
        self._prolong_sec = self.ad_system.projection_to(secondary_vars).transpose()

        # setting up data dictionaries per grid and discretizations
        self._set_up()

        # primary equations
        self._set_pressure_equation()
        self._set_mass_balance_equations()
        self._set_solute_mass_balance_equations()
        self._set_energy_balance_equation()
        # secondary equations: Phase fraction relations and enthalpy constraint
        sec_eq = [eq for eq in self.fluid.equations]  # deep copy
        for phase, equ in self.fluid.phase_fraction_relation.items():
            if phase != self.fluid.reference_phase:
                sec_eq.append(equ.name)
                self.ad_system.set_equation(equ, subdomains, {"cells": 1})
        name = "enthalpy-constraint"
        equ = (self.h - self.fluid.enthalpy / self.energy_scale) / self.h # / self.T**2
        equ.set_name(name)
        self.ad_system.set_equation(equ, subdomains, {"cells": 1})
        self.system_names["secondary-equations"] = [name] + sec_eq

        self.ad_system.discretize()
        self._export()
        self._exporter.write_pvd()

        self.transport_equations = [
            equ for equ in self.system_names["primary-equations"]
        ]

    def _export(self) -> None:
        self._exporter.write_vtu(self._system_vars, time_dependent=True)

    ### SET-UP -------------------------------------------------------------------------

    def _T_gradient(self, sd: pp.Grid) -> np.ndarray:

        y_coords = sd.cell_centers[1]
        b = np.abs(self.box['ymin'] - self.box['ymax'])
        dT = (self.heated_boundary_temperature - self.upper_boundary_temperature)

        return self.heated_boundary_temperature - dT * (y_coords - self.box['ymin'])/ b

    def _set_up(self) -> None:
        """Set model components including

            - source terms,
            - boundary values,
            - permeability tensor

        A modularization of the solid skeleton properties is still missing.
        """

        for sd, data in self.domain.subdomains(return_data=True):

            # general quantities
            injection_point = self.unitary_source(sd)
            zero_vector_source = np.zeros((self.domain.dim_max(), sd.num_cells)).ravel(
                "F"
            )
            transmissibility = pp.SecondOrderTensor(
                self.permeability * np.ones(sd.num_cells)
            )
            porosity = self.porosity * np.ones(sd.num_cells)
            # needed for weight in conductive flux
            self._porosity_ad = pp.ad.DenseArray(porosity.copy(), "porosity")

            # storage for mass matrix
            pp.initialize_data(
                sd,
                data,
                self.mass_keyword,
                {"mass_weight": porosity.copy()},
            )

            # source terms per component
            for i, component in enumerate(self.fluid.components):
                source_array = pp.ad.DenseArray(
                    injection_point * self.injection_feed[i],
                    name=f"source-{component.name}",
                )
                self._source_arrays.update({component: source_array})

            # BC pressure and mass balance
            val, bc_p = self.bc_pressure(sd)
            self.advective_bc_pressure = pp.ad.DenseArray(val, name=f"BC-pressure")
            # storage for advective flux
            pp.initialize_data(
                sd,
                data,
                self.flow_keyword,
                {
                    "bc": bc_p,
                    "bc_values": val,
                    "second_order_tensor": transmissibility,
                    "vector_source": zero_vector_source.copy(),
                    "ambient_dimension": self.domain.dim_max(),
                    # "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            # storage for advective flux upwinding
            val = self.bc_advective_weight_pressure(sd)
            self.advective_weight_bc_pressure = pp.ad.DenseArray(
                val.copy(),
                name=f"BC-advective-weight-{self._pressure_equation_name}",
            )
            pp.initialize_data(
                sd,
                data,
                f"{self.upwind_keyword}-{self.flow_keyword}",
                {
                    "bc": bc_p,
                    "bc_values": val,
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            # storage for advective flux upwinding per component mass balance
            for component in self.fluid.components:
                # skip eliminated reference component in case of global pressure equation
                if component != self.fluid.reference_component:
                    val_c = self.bc_advective_weight_component(sd, component)
                    bc = pp.ad.DenseArray(
                        val_c.copy(),
                        name=f"BC-advective-weight-{component.name}",
                    )
                    self.advective_weight_bc_component.update({component: bc})
                    pp.initialize_data(
                        sd,
                        data,
                        f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}",
                        {
                            "bc": bc_p,
                            "bc_values": val_c.copy(),
                            "darcy_flux": np.zeros(sd.num_faces),
                        },
                    )
                if isinstance(component, Compound):
                    for solute in component.solutes:
                        val_c = self.bc_advective_weight_solute(sd, solute)
                        bc = pp.ad.DenseArray(
                            val_c.copy(),
                            name=f"BC-advective-weight-{component.name}-{solute.name}",
                        )
                        self.advective_weight_bc_solute.update({solute: bc})
                        pp.initialize_data(
                            sd,
                            data,
                            f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}-{solute.name}",
                            {
                                "bc": bc_p,
                                "bc_values": val_c.copy(),
                                "darcy_flux": np.zeros(sd.num_faces),
                            },
                        )

            # enthalpy sources due to injection
            vec = injection_point * self.injected_ext_enthalpy
            self._enthalpy_source = pp.ad.DenseArray(vec, name="source-enthalpy")

            # BC energy balance
            val, bc_T = self.bc_conductive_flux(sd)
            self.conductive_bc_temperature = pp.ad.DenseArray(
                val, name="BC-temperature"
            )
            # storage for conductive flux
            pp.initialize_data(
                sd,
                data,
                self.energy_keyword,
                {
                    "bc": bc_T,
                    "bc_values": val.copy(),
                    "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells)),
                    "vector_source": zero_vector_source.copy(),
                    "ambient_dimension": self.domain.dim_max(),
                    # "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            # storage for advective flux upwinding in energy equation
            val = self.bc_advective_weight_energy(sd)
            self.advective_weight_bc_energy = pp.ad.DenseArray(
                val.copy(),
                name=f"BC-advective-weight-{self._energy_equation_name}",
            )
            pp.initialize_data(
                sd,
                data,
                f"{self.upwind_keyword}-{self.energy_keyword}-advective",
                {
                    "bc": bc_p,  # NOTE advective upwinding based on pressure BC
                    "bc_values": val.copy(),
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            val = self.bc_conductive_weight_energy(sd)
            self.conductive_weight_bc_energy = pp.ad.DenseArray(
                val.copy(),
                name=f"BC-conductive-weight-{self._energy_equation_name}",
            )
            # storage for conductive flux upwinding in energy equation
            pp.initialize_data(
                sd,
                data,
                f"{self.upwind_keyword}-{self.energy_keyword}-conductive",
                {
                    "bc": bc_T,
                    "bc_values": val.copy(),
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )

        # For now we consider only a single domain
        for intf, data in self.domain.interfaces(return_data=True):
            raise NotImplementedError("Mixed dimensional case not yet available.")

        ### Instantiating discretization operators
        subdomains = self.domain.subdomains()
        # mass matrix
        self.mass_matrix = pp.ad.MassMatrixAd(self.mass_keyword, subdomains)
        # divergence
        self.div = pp.ad.Divergence(subdomains=subdomains, name="divergence")

        # advective flux
        mpfa = pp.ad.MpfaAd(self.flow_keyword, subdomains)
        self.advective_flux = (
            mpfa.flux @ self.p + mpfa.bound_flux @ self.advective_bc_pressure
        )
        self.advective_upwind_pressure = pp.ad.UpwindAd(
            f"{self.upwind_keyword}-{self.flow_keyword}", subdomains
        )

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_keyword, subdomains)
        self.conductive_flux = (
            mpfa.flux @ self.T + mpfa.bound_flux @ self.conductive_bc_temperature
        )
        # upwind energy
        self.advective_upwind_energy = pp.ad.UpwindAd(
            f"{self.upwind_keyword}-{self.energy_keyword}-advective", subdomains
        )
        self.conductive_upwind_energy = pp.ad.UpwindAd(
            f"{self.upwind_keyword}-{self.energy_keyword}-conductive", subdomains
        )

        for component in self.fluid.components:
            if component != self.fluid.reference_component:
                self.advective_upwind_component.update(
                    {
                        component: pp.ad.UpwindAd(
                            f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}",
                            subdomains,
                        )
                    }
                )
            if isinstance(component, Compound):
                for solute in component.solutes:
                    self.advective_upwind_solute.update(
                        {
                            solute: pp.ad.UpwindAd(
                                f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}-{solute.name}",
                                subdomains,
                            )
                        }
                    )

    ## Boundary Conditions

    def _inlet_faces(self, sd: pp.Grid, tol: Optional[float] = 1e-10) -> np.ndarray:
        box = self.box
        d = np.abs(box["ymax"] - box["ymin"])
        west = np.abs(box["xmin"] - sd.face_centers[0]) <= tol
        # including only half of faces around center
        mid_half = np.abs(d - sd.face_centers[1]) <= (d / 4)
        lim_up = box["ymax"] - d / self.num_cells[1]
        lim_low = box["ymin"] + d / self.num_cells[1]
        # excluding corner cells
        mid = (sd.face_centers[1] < lim_up) & (sd.face_centers[1] > lim_low)

        return west & mid
    
    def _unit_advective_flux(self, sd: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)
        idx_west = self._inlet_faces(sd)
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = 1.
        return vals, idx_west

    def bc_pressure(self, sd: pp.Grid) -> tuple[np.ndarray, pp.BoundaryCondition]:
        """BC for advective flux (Darcy). Override for modifications.

        Phys. Dimensions of ADVECTIVE FLUX:

            - Dirichlet conditions: [MPa]
            - Neumann conditions: [m^3 / m^2 s]

        """
        _, idx_east, *_ = self._domain_boundary_sides(sd)
        vals, idx_west = self._unit_advective_flux(sd)
        idx = np.zeros(sd.num_faces, dtype=bool)

        if self.inflow_boundary_pressure:
            vals[idx_east] = self.outflow_boundary_pressure
            idx = idx | idx_east
        if self.inflow_boundary_pressure:
            vals[idx_west] = self.inflow_boundary_pressure
            idx = idx | idx_west

        bc = pp.BoundaryCondition(sd, faces=idx, cond="dir")

        return vals, bc

    def bc_advective_weight_pressure(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the advective flux in pressure equation."""
        vals, idx_west = self._unit_advective_flux(sd)
        vals[idx_west] = self.inflow_boundary_advective_weight_pressure
        return vals

    def bc_advective_weight_component(
        self, sd: pp.Grid, component: Component
    ) -> np.ndarray:
        """BC values for the weight in the advective flux in component mass balance."""
        vals, idx_west = self._unit_advective_flux(sd)
        vals[idx_west] = self.inflow_boundary_advective_weights[component]

        return vals
    
    def bc_advective_weight_solute(
        self, sd: pp.Grid, solute: pp.composite.ChemicalSpecies
    ) -> np.ndarray:
        """BC values for the weight in the advective flux in solute transport."""
        vals, idx_west = self._unit_advective_flux(sd)
        vals[idx_west] = self.inflow_boundary_advective_weights_solutes[solute]
        return vals

    def bc_advective_weight_energy(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the advective flux in component mass balance."""
        vals, idx_west = self._unit_advective_flux(sd)
        vals[idx_west] = self.inflow_boundary_advective_weight_energy

        return vals

    def bc_conductive_flux(
        self, sd: pp.Grid
    ) -> tuple[np.ndarray, pp.BoundaryCondition]:
        """Conductive BC for Fourier flux in energy equation.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:

            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [kJ / m^2 s] (density * specific enthalpy * heat flux)
              (same as convective enthalpy flux)

        """
        _, _, idx_west, idx_north, idx_south, *_ = self._domain_boundary_sides(sd)
        idx_west = self._inlet_faces(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = self.heated_boundary_temperature
        vals[idx_north] = self.upper_boundary_temperature
        # vals[idx_west] = self.inflow_boundary_temperature

        idx = idx_south | idx_north  # | idx_west
        bc = pp.BoundaryCondition(sd, idx, cond="dir")

        return vals, bc

    def bc_conductive_weight_energy(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the conductive flux."""
        _, _, idx_west, idx_north, idx_south, *_ = self._domain_boundary_sides(sd)
        idx_west = self._inlet_faces(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = 1.
        vals[idx_north] = 1.
        # vals[idx_west] = 1.
        vals *= 1.  # TODO this needs proper computation

        return vals

    ## Source terms

    def unitary_source(self, g: pp.Grid) -> np.ndarray:
        """Unitary, single-cell source term in center of first grid part

        |-----|-----|-----|
        |  .  |     |     |
        |-----|-----|-----|

        Phys. Dimensions:
            - mass source:          [mol / m^3 / s]
            - enthalpy source:      [J / m^3 / s] = [kg m^2 / m^3 / s^3]

        Parameters:
            g: grid (single-dime domain)

        Returns:
            an array with a non-zero entry for the source cell defined here.
        """
        # find and set single-cell source
        vals = np.zeros(g.num_cells)
        point_source = np.array([[0.5], [0.5]])
        source_cell = g.closest_cell(point_source)
        vals[source_cell] = 1.0

        return vals

    ### solution strategy --------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.converged = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(
        self,
        flash_tol: float = 1e-4,
        flash_max_iter: int = 120,
        flash_armijo_iter: int = 150,
        use_iterate: bool = False,
    ) -> None:
        """Re-discretizes the Upwind operators and the fluxes."""

        logger.info("\n")
        p = self.ad_system.get_variable_values([self.p.name], iterate_index=0) * self.pressure_scale
        h = self.ad_system.get_variable_values([self.h.name], iterate_index=0) * self.energy_scale
        T = self.ad_system.get_variable_values([self.T.name], iterate_index=0)
        for compound, solute_fractions in self.fluid.solute_fraction_variables.items():
            X = [self.ad_system.get_variable_values([s], iterate_index=0) for s in solute_fractions]
            compound.compute_molalities(*tuple(X), store=True)
        iterate_state = self.fluid.get_fractional_state_from_vector()
        iterate_state.T = T
        self.flash.tolerance = flash_tol
        self.flash.max_iter = flash_max_iter
        self.flash.armijo_parameters["j_max"] = flash_armijo_iter

        if use_iterate:
            try:
                success, state = self.flash.flash(
                    state={"p": p, "h": h},
                    eos_kwargs={"apply_smoother": True},
                    guess_from_state=iterate_state,  # iterate state
                    verbosity=self._verbosity,
                )
            except Exception as err:
                logger.warn(f"\nFlash crashed:\n{str(err)}")
                success = 2
        else:
            success = 3

        if success != 0:  # try with custom initial guess.
            self.flash.max_iter = 200
            self.flash.armijo_parameters["j_max"] = 120
            logger.warn(f"\n.. Attempting flash with computed initial guess ..\n")
            try:
                success, state = self.flash.flash(
                    state={"p": p, "h": h},
                    eos_kwargs={"apply_smoother": True},
                    guess_from_state=None,  # iterate state
                    feed=iterate_state.z,
                    verbosity=self._verbosity,
                )
            except Exception as err:
                logger.warn(f"\nFlash crashed:\n{str(err)}\n")
                success = 2

        if success in [0]:
            logger.info(
                f".. Newton iteration {self._nonlinear_iteration}:"
                + " Flash succeeded. Updating values .."
            )
            T = state.T
            self.ad_system.set_variable_values(T, [self.T.name], iterate_index=0)

            for j, phase in enumerate(self.fluid.phases):
                # storing phase fractions of independent phases
                if phase != self.fluid.reference_phase:
                    self.ad_system.set_variable_values(
                        state.y[j], [phase.fraction.name], iterate_index=0
                    )
                    self.ad_system.set_variable_values(
                        state.s[j], [phase.saturation.name], iterate_index=0
                    )
                # storing phase compositions
                for i, comp in enumerate(self.fluid.components):
                    self.ad_system.set_variable_values(
                        state.X[j][i],
                        [phase.fraction_of[comp].name],
                        iterate_index=0,
                    )

            logger.info(
                f".. Newton iteration {self._nonlinear_iteration}:"
                + " Flash succeeded. Computing properties .."
            )

            # This is done this way to include derivatives for all system variables
            state_vector = self.ad_system.get_variable_values(
                self._system_vars, iterate_index=0
            )
            self.fluid.compute_properties_from_vector(
                self.p * self.pressure_scale,
                self.T,
                state=state_vector,
                as_ad=True,
                derivatives=self._system_vars,
                Z_as_AD = True,
            )

        else:
            x = self.ad_system.get_variable_values(
                variables=self._system_vars, iterate_index=0
            )
            logger.warn(
                f".. Newton iteration {self._nonlinear_iteration}:"
                + " Flash failed."
            )
            return False, x

        logger.info(
            f".. Newton iteration {self._nonlinear_iteration}:"
            + " Re-discretizing upwind .."
        )
        # Advective flux upwinding based on latest pressure values
        pp.fvutils.compute_darcy_flux(
            self.domain,
            keyword=self.flow_keyword,
            keyword_store=f"{self.upwind_keyword}-{self.flow_keyword}",
            p_name=self.p.name,
            from_iterate=True,
        )

        # conductive flux upwinding based on latest temperature values
        pp.fvutils.compute_darcy_flux(
            self.domain,
            keyword=self.energy_keyword,
            keyword_store=f"{self.upwind_keyword}-{self.energy_keyword}-conductive",
            p_name=self.T.name,
            from_iterate=True,
        )
        # copy advective flux to dictionaries for mass balance upwinding and advective
        # energy upwinding
        for _, data in self.domain.subdomains(return_data=True):
            # get the flux
            kw = f"{self.upwind_keyword}-{self.flow_keyword}"
            flux = data["parameters"][kw]["darcy_flux"]

            # copy the flux to the pressure equation dictionary
            kw = f"{self.upwind_keyword}-{self.energy_keyword}-advective"
            data["parameters"][kw]["darcy_flux"] = np.copy(flux)

            # copy the flux to the dictionaries belonging to mass balance per component
            for component in self.fluid.components:
                # skip the eliminated component mass balance
                if component != self.fluid.reference_component:
                    kw = f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}"
                    data["parameters"][kw]["darcy_flux"] = np.copy(flux)
                if isinstance(component, Compound):
                    for solute in component.solutes:
                        kw = f"{self.upwind_keyword}-{self.flow_keyword}-{component.name}-{solute.name}"
                        data["parameters"][kw]["darcy_flux"] = np.copy(flux)

        # TODO recompute the effective conductive BC based on new saturation values.
        self.advective_upwind_pressure.upwind.discretize(self.domain)
        self.conductive_upwind_energy.upwind.discretize(self.domain)
        for _, upwind in self.advective_upwind_component.items():
            upwind.upwind.discretize(self.domain)
        for _, upwind in self.advective_upwind_solute.items():
            upwind.upwind.discretize(self.domain)

        return True, 0

    def after_newton_iteration(self, update_vector: np.ndarray) -> None:
        """Distributes solution of iteration additively to the iterate state of the
        variables. Increases the iteration counter.
        """
        self._nonlinear_iteration += 1

        inv_A_ss, b_s, A_sp = self._schur_expansion
        x_s = inv_A_ss * (b_s - A_sp * update_vector)
        DX = self._prolong_prim * update_vector + self._prolong_sec * x_s

        # post-processing eliminated component fraction additively to iterate
        self.ad_system.set_variable_values(
            DX,
            variables=self._system_vars,
            iterate_index=0,
            additive=True,
        )

    def after_newton_convergence(self, solution: np.ndarray) -> None:
        """Copies the values from the iterate to the current timestep.
        Exports the results."""
        # write global solution
        self.ad_system.set_variable_values(
            solution,
            variables=self._system_vars,
            time_step_index=0,
        )
        logger.info(
            f".. Newton iteration {self._nonlinear_iteration}:"
            + " exporting state"
        )
        self._export()

    def after_newton_failure(self, update_vector: np.ndarray) -> None:
        """Reset iterate state to previous time step."""
        logger.info(
            f".. Newton iteration {self._nonlinear_iteration}:"
            + "FAILED. Re-setting iterate to previous timestep."
        )
        X = self.ad_system.get_variable_values(
            variables=self._system_vars, time_step_index=0
        )
        self.ad_system.set_variable_values(
            X, variables=self._system_vars, iterate_index=0
        )
        # safe progress
        self._exporter.write_pvd()

    def after_simulation(self) -> None:
        """Writes PVD file."""
        logger.info(f"Simulation finished. Writing PVD\n")
        self._exporter.write_pvd()

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Performs a Newton step for the whole system by constructing a Schur
        complement using the equilibrium equations and non-primary variables."""

        # non-linear Schur complement elimination of secondary variables
        A_pp, b_p = self.ad_system.assemble_subsystem(
            self.system_names["primary-equations"],
            self.system_names["primary-variables"],
        )
        A_sp, _ = self.ad_system.assemble_subsystem(
            self.system_names["secondary-equations"],
            self.system_names["primary-variables"],
        )
        A_ps, _ = self.ad_system.assemble_subsystem(
            self.system_names["primary-equations"],
            self.system_names["secondary-variables"],
        )
        A_ss, b_s = self.ad_system.assemble_subsystem(
            self.system_names["secondary-equations"],
            self.system_names["secondary-variables"],
        )

        res = np.linalg.norm(np.hstack([b_p, b_s]))
        logger.info(
            f".. Newton iteration {self._nonlinear_iteration}: res = {res}"
        )
        if res < tol:
            logger.info(
                f".. Newton iteration {self._nonlinear_iteration}: converged"
            )
            self.converged = True
            x = self.ad_system.get_variable_values(
                variables=self._system_vars, iterate_index=0
            )
            return x

        inv_A_ss = np.linalg.inv(A_ss.A)
        inv_A_ss = sps.csr_matrix(inv_A_ss)

        A = A_pp - A_ps * inv_A_ss * A_sp
        A = sps.csr_matrix(A)
        b = b_p - A_ps * inv_A_ss * b_s
        self._schur_expansion = (inv_A_ss, b_s, A_sp)

        # dx = sps.linalg.spsolve(A, b)
        dx = pypardiso.spsolve(A, b)

        return dx

    ### MODEL EQUATIONS ----------------------------------------------------------------

    def _set_pressure_equation(self) -> None:
        """Sets the global pressure equation."""

        ### ACCUMULATION
        accumulation = self.mass_matrix.mass @ (
            self.fluid.density - self.fluid.density.previous_timestep()
        )

        ### ADVECTION
        advective_weight = [self.phase_mobility(phase) for phase in self.fluid.phases]
        advective_weight = pp.ad.sum_operator_list(
            advective_weight, name=f"advective-weight-{self._pressure_equation_name}"
        )

        advection = (
            self.advective_flux
            * (self.advective_upwind_pressure.upwind @ advective_weight)
            - self.advective_upwind_pressure.bound_transport_dir
            @ (self.advective_flux * self.advective_weight_bc_pressure)
            - self.advective_upwind_pressure.bound_transport_neu
            @ self.advective_weight_bc_pressure
        )

        ### SOURCE
        source_arrays = [self._source_arrays[comp] for comp in self.fluid.components]
        source = self.mass_matrix.mass @ pp.ad.sum_operator_list(
            source_arrays, name=f"source-{self._pressure_equation_name}"
        )

        ### PRESSURE EQUATION
        # minus in advection already included
        equation: pp.ad.Operator = accumulation + self.dt * (
            self.div @ advection - source
        )  # type: ignore

        self.system_names["primary-equations"].append(self._pressure_equation_name)
        equation.set_name(self._pressure_equation_name)
        self.ad_system.set_equation(equation, self.domain.subdomains(), {"cells": 1})

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per component.

        The reference component is excluded.

        """

        for component in self.fluid.components:
            # skipping reference component, since feed fraction eliminated by
            # unity
            if component == self.fluid.reference_component:
                continue

            name = f"{self._mass_balance_name}-{component.name}"

            advective_weight_bc = self.advective_weight_bc_component[component]
            advective_upwind = self.advective_upwind_component[component]

            ### ACCUMULATION
            accumulation = self.mass_matrix.mass @ (
                component.fraction * self.fluid.density
                - component.fraction.previous_timestep()
                * self.fluid.density.previous_timestep()
            )

            ### ADVECTION
            advective_weight = [
                phase.normalized_fraction_of[component] * self.phase_mobility(phase)
                for phase in self.fluid.phases
            ]
            # sum over all phases
            advective_weight = pp.ad.sum_operator_list(
                advective_weight, name=f"advective-weight-{name}"
            )

            advection = (
                self.advective_flux * (advective_upwind.upwind @ advective_weight)
                - advective_upwind.bound_transport_dir
                @ (self.advective_flux * advective_weight_bc)
                - advective_upwind.bound_transport_neu @ advective_weight_bc
            )

            ### SOURCE
            source = self.mass_matrix.mass @ self._source_arrays[component]

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equation: pp.ad.Operator = accumulation + self.dt * (
                self.div @ advection - source
            )  # type: ignore

            self.system_names["primary-equations"].append(name)
            equation.set_name(name)
            self.ad_system.set_equation(
                equation, self.domain.subdomains(), {"cells": 1}
            )

    def _set_solute_mass_balance_equations(self) -> None:
        """Set mass balance equations per solute in compounds."""

        for component in self.fluid.components:
            # skipping reference component, since feed fraction eliminated by
            # unity
            if not isinstance(component, Compound):
                continue

            for solute in component.solutes:
                name = f"{self._mass_balance_name}-{component.name}-{solute.name}"

                advective_weight_bc = self.advective_weight_bc_solute[solute]
                advective_upwind = self.advective_upwind_solute[solute]

                ### ACCUMULATION
                accumulation = self.mass_matrix.mass @ (
                    component.solute_fraction_of[solute]
                    * component.fraction
                    * self.fluid.density
                    - component.solute_fraction_of[solute].previous_timestep()
                    * component.fraction.previous_timestep()
                    * self.fluid.density.previous_timestep()
                )

                ### ADVECTION
                advective_weight = [
                    phase.normalized_fraction_of[component] * self.phase_mobility(phase)
                    for phase in self.fluid.phases
                ]
                # sum over all phases
                advective_weight = pp.ad.sum_operator_list(
                    advective_weight, name=f"advective-weight-{name}"
                )
                # scale moles by solute fraction
                advective_weight = component.solute_fraction_of[solute] * advective_weight

                advection = (
                    self.advective_flux * (advective_upwind.upwind @ advective_weight)
                    - advective_upwind.bound_transport_dir
                    @ (self.advective_flux * advective_weight_bc)
                    - advective_upwind.bound_transport_neu @ advective_weight_bc
                )

                ### SOURCE
                # source = self.mass_matrix.mass @ self._source_arrays_solutes[solute]

                ### MASS BALANCE PER COMPONENT
                # minus in advection already included
                equation: pp.ad.Operator = accumulation + self.dt * (
                    self.div @ advection  # - source
                )  # type: ignore

                self.system_names["primary-equations"].append(name)
                equation.set_name(name)
                self.ad_system.set_equation(
                    equation, self.domain.subdomains(), {"cells": 1}
                )

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        ### ACCUMULATION
        accumulation = self.mass_matrix.mass @ (
            self.h * self.fluid.density
            - self.h.previous_timestep() * self.fluid.density.previous_timestep()
        )

        ### ADVECTION
        advective_weight = [
            phase.enthalpy / self.energy_scale * self.phase_mobility(phase) for phase in self.fluid.phases
        ]
        # sum over all phases
        advective_weight = pp.ad.sum_operator_list(
            advective_weight, name=f"advective-weight-{self._energy_equation_name}"
        )

        advection = (
            self.advective_flux
            * (self.advective_upwind_energy.upwind @ advective_weight)
            - self.advective_upwind_energy.bound_transport_dir
            @ (self.advective_flux * self.advective_weight_bc_energy)
            - self.advective_upwind_energy.bound_transport_neu
            @ self.advective_weight_bc_energy
        )

        ### CONDUCTION
        conductive_weight = [
            phase.saturation * phase.conductivity for phase in self.fluid.phases
        ]
        conductive_weight = self._porosity_ad * pp.ad.sum_operator_list(
            conductive_weight, name=f"conductive-weight-{self._energy_equation_name}"
        )

        conduction = (
            self.conductive_flux
            * (self.conductive_upwind_energy.upwind @ conductive_weight)
            - self.conductive_upwind_energy.bound_transport_dir
            @ (self.conductive_flux * self.conductive_weight_bc_energy)
            - self.conductive_upwind_energy.bound_transport_neu
            @ self.conductive_weight_bc_energy
        )

        ### SOURCE
        # enthalpy source due to injection
        source = self.mass_matrix.mass @ self._enthalpy_source

        ### GLOBAL ENERGY BALANCE
        equation: pp.ad.Operator = accumulation + self.dt * (
            self.div @ (advection + conduction) - source
        )  # type: ignore

        self.system_names["primary-equations"].append(self._energy_equation_name)
        equation.set_name(self._energy_equation_name)
        self.ad_system.set_equation(equation, self.domain.subdomains(), {"cells": 1})

    ### CONSTITUTIVE LAWS --------------------------------------------------------------

    def phase_mobility(self, phase: pp.composite.Phase) -> pp.ad.Operator:
        """Auxiliary function to compute the phase mobility."""
        return phase.density * self.rel_perm(phase.saturation) / phase.viscosity

    def rel_perm(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation
