"""Contains a general composit flow class without reactions

The grid, expected Phases and components can be modified in respective methods.

"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plot
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp
from porepy.composite.phase import Phase


class CompositionalFlowModel(pp.models.abstract_model.AbstractModel):
    """Non-isothermal flow consisting of water in liquid and vapor phase
    and salt in liquid phase.

    The phase equilibria equations for water are given using k-values. The composition class is
    instantiated in the constructor using a sub-routine which can be inherited and overwritten.

    Parameters:
        params: general model parameters including

            - 'use_ad' (bool): indicates whether :module:`porepy.ad` is used or not
            - 'file_name' (str): name of file for exporting simulation results
            - 'folder_name' (str): absolute path to directory for saving simulation results
            - 'eliminate_ref_phase' (bool): flag to eliminate reference phase fractions.
               Defaults to TRUE.
            - 'use_pressure_equation' (bool): flag to use the global pressure equation, instead
               of the feed fraction unity. Defaults to TRUE.
            - 'monolithic' (bool): flag to solve monolithically, instead of using a Schur
               complement elimination for secondary variables. Defaults to TRUE.

    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)

        ### PUBLIC

        self.params: dict = params
        """Parameter dictionary passed at instantiation."""

        self.converged: bool = False
        """Indicator if current Newton-step converged."""

        self.dt: float = 0.1
        """Timestep size."""

        ## Initial Conditions
        self.initial_pressure: float = 100.
        """Initial pressure in the domain in kPa."""

        self.initial_temperature: float = 343.15  # 50 dec Celsius
        """Initial temperature in the domain in kPa."""

        self.initial_component_fractions: dict[pp.composite.Component, float] = dict()
        """Contains per component (key) the initial feed fraction.

        To be set in :meth:`set_composition`.

        """

        ## Injection parameters
        self.mass_sources: dict[pp.composite.Component, float] = dict()
        """Contains per component (key) the amount of injected moles.

        To be set in :meth:`set_composition`.

        """

        self.enthalpy_sources: dict[pp.composite.Component, float] = dict()
        """Contains per component (key) the amount of injected extensive enthalpy.

        To be set in :meth:`set_composition`.

        """

        self.injection_temperature = 303.16
        """Temperature of injected mass for source terms in Kelvin."""

        self.injection_pressure = 150.
        """Pressure at injection in kPA."""
        
        ## Boundary conditions
        self.boundary_temperature = 423.15  # 150 Celsius
        """Dirichlet boundary temperature in Kelvin for the conductive flux."""
        self.outflow_boundary_pressure: float = 100.
        """Dirichlet boundary pressure for the outflow in kPA for the advective flux."""
        self.inflow_boundary_pressure: float = 130.
        """Dirichlet boundary pressure for the inflow in kPa for the advective flux."""
        self.inflow_boundary_temperature: float = self.injection_temperature
        """Temperature at the inflow boundary for computing influxing densities."""
        self.inflow_boundary_saturations: dict[Phase, float] = dict()
        """Saturations per phase (key) at inflow boundary to computing influxing moles.
        To be set in :meth:`set_composition`.
        """
        self.inflow_boundary_composition: dict[Phase, dict[pp.composite.Component, float]] = {}
        """Composition per phase (key) to compute influxing moles.
        To be set in :meth:`set_composition`.
        """
        self.inflow_boundary_advective_component: dict[pp.composite.Component, float] = dict()
        """Contains per component (key) the scalar part of the advective flux on the
        inflow boundary.
        To be set in :meth:`set_composition`.
        """
        self.inflow_boundary_advective_pressure: float
        """Value for scalar part of the advective flux on the
        inflow boundary for the global pressure equation.
        To be set in :meth:`set_composition`.
        """
        self.inflow_boundary_advective_energy: float
        """Value for scalar part of advective flux in energy equation.
        To be set in :meth:`set_composition`.
        """
        self.boundary_conductive: float
        """Value for scalar part of conductive flux in energy equation.
        To be set in :meth:`set_composition`.
        """

        ## Porous parameters
        self.porosity = 1.0
        """Base porosity of model domain."""

        self.permeability = 0.03
        """Base permeability of model domain."""

        self.mdg: pp.MixedDimensionalGrid
        """Computational domain, to be set in :meth:`create_grid`."""
        self.create_grid()

        self.ad_system: pp.ad.ADSystem = pp.ad.ADSystem(self.mdg)
        """AD System for this model."""

        self.dof_man: pp.DofManager = self.ad_system.dof_manager
        """DOF manager of AD System."""

        # contains information about the primary system
        self.flow_subsystem: dict[str, list] = dict()
        """Dictionary containing names of primary and secondary equations and variables."""

        # Parameter keywords
        self.flow_keyword: str = "flow"
        """Flow keyword for storing values in data dictionaries."""
        self.flow_upwind_keyword: str = f"upwind_{self.flow_keyword}"
        """Flow-upwinding keyword for storing values in data dictionaries."""
        self.mass_keyword: str = "mass"
        """Mass keyword for storing values in data dictionaries."""
        self.energy_keyword: str = "energy"
        """Energy keyword for storing values in data dictionaries."""
        self.energy_upwind_keyword: str = f"upwind_{self.energy_keyword}"
        """Energy-upwinding keyword for storing values in data dictionaries."""

        ## References to discretization operators
        # they will be set during `prepare_simulation`
        self.mass_matrix: pp.ad.MassMatrixAd
        self.div: pp.ad.Divergence
        self.advective_flux: pp.ad.MpfaAd
        self.advective_upwind: pp.ad.UpwindAd
        self.advective_upwind_bc: pp.ad.BoundaryCondition
        self.advective_upwind_component: dict[pp.composite.Component, pp.ad.UpwindAd] = dict()
        self.advective_upwind_component_bc: dict[pp.composite.Component, pp.ad.BoundaryCondition] = dict()
        self.conductive_flux: pp.ad.MpfaAd
        self.conductive_upwind: pp.ad.UpwindAd
        self.conductive_upwind_bc: pp.ad.BoundaryCondition
        self.advective_upwind_energy: pp.ad.UpwindAd
        self.advective_upwind_energy_bc: pp.ad.BoundaryCondition

        ### COMPOSITION SETUP
        self.composition: pp.composite.Composition
        self.reference_component: pp.composite.Component
        self.set_composition()

        ### PRIVATE

        # projections/ prolongation from global dofs to primary and secondary variables
        # will be set during `prepare simulations`
        self._prolong_prim: sps.spmatrix
        self._prolong_sec: sps.spmatrix
        self._prolong_system: sps.spmatrix
        # system variables, depends whether solver monolithic or not
        self._system_vars: list[str]
        self._export_vars: list[str]
        # exporter
        self._exporter: pp.Exporter = pp.Exporter(
            self.mdg,
            params["file_name"],
            folder_name=params["folder_name"],
            export_constants_separately=False,
        )
        # list of grids as ordered in GridBucket
        self._grids = [g for g in self.mdg.subdomains()]
        # list of edges as ordered in GridBucket
        self._edges = [e for e in self.mdg.interfaces()]
        # data for Schur complement expansion
        self._for_expansion = (None, None, None)
        # solver strategy. If monolithic, the model will take care of flash calculations
        # if not, it will solve only the primary system
        self._monolithic = params.get("monolithic", True)
        # a reference to the operator representing the reference phase saturation eliminated
        # by unity
        self._s_R: Optional[pp.ad.Operator] = None
        self._elim_ref_phase: bool = params.get("eliminate_ref_phase", True)
        # use the pressure equation instead of the feed fraction unity
        self._use_pressure_equation: bool = params.get("use_pressure_equation", True)
        # a representation of the component feed eliminated by unity
        self._z_R: Optional[pp.ad.Operator] = None
        self._system_equations: list[str] = list()

    def create_grid(self) -> None:
        """Assigns a cartesian grid as computational domain.
        Overwrites/sets the instance variables 'mdg'.
        """
        refinement = 7
        phys_dims = [3, 1]
        # n_cells = [4, 2]
        n_cells = [i * refinement for i in phys_dims]
        bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]]])
        self.box = pp.geometry.bounding_box.from_points(bounding_box_points)
        sg = pp.CartGrid(n_cells, phys_dims)
        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(sg)
        self.mdg.compute_geometry()

    def set_composition(self) -> None:
        """Define the composition for which the simulation should be run and performs
        the initial (p-T) equilibrium calculations for the initial state given in p,T and feed.

        Set initial values for p, T and feed here.

        Use this method to inherit and override the composition, while keeping the (generic)
        rest of the model.

        Assumes the model domain ``mdg`` is already set.

        """
        ## creating composition
        self.composition = pp.composite.Composition(self.ad_system)
        water = pp.composite.H2O(self.ad_system)
        salt = pp.composite.NaCl(self.ad_system)
        L = self.composition._phases[0]
        G = self.composition._phases[1]
        self.composition.add_component(water)
        self.composition.add_component(salt)
        self.reference_component = water
        
        ## configuration
        initial_salt_concentration = 0.1
        k_water = 2.
        k_salt = 0.1
        injected_moles_water = 7000.0 / pp.composite.H2O.molar_mass()
        # initial fractions
        self.initial_component_fractions.update({
            water: 1. - initial_salt_concentration,
            salt: initial_salt_concentration,
        })
        # saturate only liquid phase at boundary
        self.inflow_boundary_saturations.update({
            L: 1.,
            G: 0.,
        })
        # inflow of water AND salt to avoid washing out
        self.inflow_boundary_composition.update({
            L: {
                water: 0.99,
                salt: 0.01,
            },
            G: {  # not so relevant since we feed only liquid
                water: 1.,
                salt: 0.,
            },
        })
        # TODO this needs to be ADified properly
        rho_L = L.rho0
        rho_G = G.density(
                self.inflow_boundary_pressure, self.inflow_boundary_temperature
        )
        s_L = self.inflow_boundary_saturations[L]
        s_G = self.inflow_boundary_saturations[G]
        for component in self.composition.components:
            chi_cL = self.inflow_boundary_composition[L][component]
            chi_cG = self.inflow_boundary_composition[G][component]

            advective_scalar = (
                rho_L * chi_cL * s_L
                + rho_G * chi_cG * s_G
            ) # TODO this should be scaled somehow with porosity
            self.inflow_boundary_advective_component.update({
                component: advective_scalar
            })
        h_L = L.specific_enthalpy(
                self.inflow_boundary_pressure, self.inflow_boundary_temperature
        )
        h_G = G.specific_enthalpy(
                self.inflow_boundary_pressure, self.inflow_boundary_temperature
        )
        self.inflow_boundary_advective_energy = (rho_L * s_L * h_L + rho_G * s_G * h_G)  # TODO same as above
        self.inflow_boundary_advective_pressure = (rho_L * s_L + rho_G * s_G)

        self.boundary_conductive = 1.
        
        # constant k-values
        self.composition.k_values = {
            water: k_water,
            salt: k_salt
        }

        # mass sources
        self.mass_sources.update({
            water : 0. * injected_moles_water,  
            salt : 0. * injected_moles_water,  # trace amounts to avoid washing out
        })

        # source is extensive, multiply with injected moles (kJ)
        h_water = self.composition.reference_phase.specific_enthalpy(
            self.injection_pressure,
            self.injection_temperature
        ) * injected_moles_water
        self.enthalpy_sources = {
            water: 0. * h_water,
            salt: 0. * h_water,
        }

        ## setting of initial values
        nc = self.mdg.num_subdomain_cells()

        # setting water feed fraction
        water_frac = self.initial_component_fractions[water] * np.ones(nc)
        self.ad_system.set_var_values(water.fraction_name, water_frac, True)

        # setting salt feed fraction
        salt_frac = self.initial_component_fractions[salt] * np.ones(nc)
        self.ad_system.set_var_values(salt.fraction_name, salt_frac, True)

        # setting initial pressure
        p_vals = self.initial_pressure * np.ones(nc)
        self.ad_system.set_var_values(self.composition.p_name, p_vals, True)

        # setting initial temperature
        T_vals = self.initial_temperature * np.ones(nc)
        self.ad_system.set_var_values(self.composition.T_name, T_vals, True)

        # set zero enthalpy values at the beginning to get the AD framework properly started
        h_vals = np.zeros(nc)
        self.ad_system.set_var_values(self.composition.h_name, h_vals, True)

        self.composition.initialize()
        self.composition.isothermal_flash(copy_to_state=True, initial_guess="feed")
        self.composition.evaluate_saturations()
        # This corrects the initial values
        self.composition.evaluate_specific_enthalpy()

    def _P_vap(self, T: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Implements vapor pressure using the Buck equation
        
        P_vap = 0.061121 * exp( (18.678 - T / 234.5) * ( T / (257.14 + T)) )
        
        where P_vap is returned in [kPa] and T given in Celsius
        """
        T_celsius = T - 272.15

        arg = (18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))

        ad_exp = pp.ad.Function(pp.ad.exp, name="exp")

        P_vap = 0.61121 * ad_exp(arg)
        P_vap.set_name("p_vap_Buck")
        return P_vap

    def prepare_simulation(self) -> None:
        """Preparing essential simulation configurations.

        Method needs to be called after the composition has been set and prior to applying any
        solver.

        """
        # Define primary and secondary variables/system which are secondary and primary in
        # the composition subsystem
        self.flow_subsystem.update(
            {
                "primary_equations": list(),
                "secondary_equations": list(),
                "primary_vars": list(),
                "secondary_vars": list(),
            }
        )
        # the primary variables of the flash, are secondary for the flow, and vice versa.
        # deep copies to not mess with the flash subsystems
        primary_vars = [var for var in self.composition.ph_subsystem["secondary_vars"]]
        secondary_vars = [var for var in self.composition.ph_subsystem["primary_vars"]]

        # eliminate the reference component fraction if pressure equation is used
        if self._use_pressure_equation:
            primary_vars.remove(self.reference_component.fraction_name)

        # saturations are also secondary in the flow
        # we eliminate the saturation of the reference phase by unity, if requested        
        for phase in self.composition.phases:
            primary_vars.remove(phase.saturation_name)
            # the reference phase molar fraction is also secondary in the composition class
            if phase == self.composition.reference_phase:
                primary_vars.remove(phase.fraction_name)
            # add the eliminated phase saturation to secondaries only if not eliminated
            if phase == self.composition.reference_phase and self._elim_ref_phase:
                continue
            else:
                secondary_vars.append(phase.saturation_name)

        # defining system variables
        self.flow_subsystem.update({"primary_vars": primary_vars})
        self.flow_subsystem.update({"secondary_vars": secondary_vars})
        self._system_vars = primary_vars + secondary_vars

        # complete set of variables for export of results
        export_vars = set(self._system_vars)
        if self._use_pressure_equation:
            export_vars.add(self.reference_component.fraction_name)
        if self._elim_ref_phase:
            export_vars.add(self.composition.reference_phase.fraction_name)
            export_vars.add(self.composition.reference_phase.saturation_name)
        self._export_vars = list(export_vars)

        # prepare prolongations for the solver
        self._prolong_prim = self.dof_man.projection_to(primary_vars).transpose()
        self._prolong_sec = self.dof_man.projection_to(secondary_vars).transpose()
        self._prolong_system = self.dof_man.projection_to(self._system_vars).transpose()

        # if eliminated, express the reference fractions by unity
        if self._elim_ref_phase:
            self._s_R = self.composition.get_reference_phase_saturation_by_unity()
        if self._use_pressure_equation:
            self._z_R = self._get_reference_feed_by_unity()

        # deep copies, just in case
        sec_eq = [eq for eq in self.composition.ph_subsystem["equations"]]
        self.flow_subsystem.update({"secondary_equations": sec_eq})

        self._set_up()

        # setting equations
        # primary equations
        if self._use_pressure_equation:
            self._set_pressure_equation()
        else:
            self._set_feed_fraction_unity_equation()
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()
        # secondary equations to obtain derivative of saturations
        self._set_phase_fraction_relation_equations()

        self.ad_system.discretize()
        self._export()

        self._system_equations = [equ for equ in self.flow_subsystem["primary_equations"]]
        self._system_equations += [equ for equ in self.flow_subsystem["secondary_equations"]]

    def _export(self) -> None:
        self._exporter.write_vtu(self._export_vars, time_dependent=True)

    ### SET-UP --------------------------------------------------------------------------------
    
    def _set_up(self) -> None:
        """Set model components including

            - source terms,
            - boundary values,
            - permeability tensor

        A modularization of the solid skeleton properties is still missing.
        """

        for sd, data in self.mdg.subdomains(return_data=True):

            source = self._unitary_source(sd)
            unit_tensor = pp.SecondOrderTensor(np.ones(sd.num_cells))
            zero_vector_source = np.zeros((self.mdg.dim_max(), sd.num_cells))

            ### MASS PARAMETERS AND MASS SOURCES
            param_dict = dict()
            # weight for accumulation term
            param_dict.update({"mass_weight": self.porosity * np.ones(sd.num_cells)})
            # source terms per component
            for component in self.composition.components:
                param_dict.update({
                    f"source_{component.name}": source * self.mass_sources[component]
                })

            pp.initialize_data(
                sd,
                data,
                self.mass_keyword,
                param_dict,
            )

            ### MASS BALANCE EQUATIONS
            # advective flux in mass balance
            bc_advective, bc_vals = self._bc_advective_flux(sd)
            transmissibility = pp.SecondOrderTensor(
                self.permeability * np.ones(sd.num_cells)
            )
            pp.initialize_data(
                sd,
                data,
                self.flow_keyword,
                {
                    "bc": bc_advective,
                    "bc_values": bc_vals,
                    "second_order_tensor": transmissibility,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            # upwind bc for advective flux of global pressure equation
            bc_vals = self._bc_advective_weight_pressure(sd)
            if self._use_pressure_equation:
                # upwind bc for pressure equation
                pp.initialize_data(
                    sd,
                    data,
                    self.flow_upwind_keyword,
                    {
                        "bc": bc_advective,
                        "bc_values": bc_vals,
                        "darcy_flux": np.zeros(sd.num_faces),
                    },
                )
            # upwind bc per component mass balance
            for component in self.composition.components:
                # skip eliminated reference component in case of global pressure equation
                if component == self.reference_component and self._use_pressure_equation:
                    continue
                bc_vals = self._bc_advective_weight_component(sd, component)
                pp.initialize_data(
                    sd,
                    data,
                    f"{self.flow_upwind_keyword}_{component.name}",
                    {
                        "bc": bc_advective,
                        "bc_values": bc_vals,
                        "darcy_flux": np.zeros(sd.num_faces),
                    },
                )

            ### ENERGY EQUATION
            # conductive flux in energy equation
            # ernergy equation sources
            param_dict = dict()
            # general enthalpy sources e.g., hot skeleton
            param_dict.update({"source": source * 0.0})
            # enthalpy sources due to substance mass source
            for component in self.composition.components:
                param_dict.update(
                    {
                        f"source_{component.name}": source * self.enthalpy_sources[component]
                    }
                )
            bc_conductive, bc_vals = self._bc_conductive_flux(sd)
            param_dict.update(
                {
                    "bc": bc_conductive,
                    "bc_values": bc_vals,
                    "second_order_tensor": unit_tensor,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                }
            )
            pp.initialize_data(
                sd,
                data,
                self.energy_keyword,
                param_dict,
            )

            # advective flux upwinding parameters in energy equation
            bc_vals = self._bc_advective_weight_energy(sd)
            pp.initialize_data(
                sd,
                data,
                f"{self.energy_upwind_keyword}_advective",
                {
                    "bc": bc_advective,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )
            # conductive flux upwinding parameters 
            bc_vals = self._bc_conductive_weight(sd)
            pp.initialize_data(
                sd,
                data,
                f"{self.energy_upwind_keyword}_conductive",
                {
                    "bc": bc_conductive,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )

        # For now we consider only a single domain
        for intf, data in self.mdg.interfaces(return_data=True):
            raise NotImplementedError("Mixed dimensional case not yet available.")

        ### Instantiating discretization operators
        # mass matrix
        self.mass_matrix = pp.ad.MassMatrixAd(self.mass_keyword, self._grids)
        # divergence
        self.div = pp.ad.Divergence(subdomains=self._grids, name="Divergence")

        # advective flux
        mpfa = pp.ad.MpfaAd(self.flow_keyword, self._grids)
        bc = pp.ad.BoundaryCondition(self.flow_keyword, self._grids)
        self.advective_flux = mpfa.flux * self.composition.p + mpfa.bound_flux * bc
        # advective upwind for global pressure
        if self._use_pressure_equation:
            self.advective_upwind = pp.ad.UpwindAd(self.flow_upwind_keyword, self._grids)
            self.advective_upwind_bc = pp.ad.BoundaryCondition(
                self.flow_upwind_keyword, self._grids
            )
        # advective upwind per component mass balance
        for component in self.composition.components:
            # skip eliminated reference component in case of global pressure equation
            if component == self.reference_component and self._use_pressure_equation:
                continue
            kw = f"{self.flow_upwind_keyword}_{component.name}"
            upwind = pp.ad.UpwindAd(kw, self._grids)
            upwind_bc = pp.ad.BoundaryCondition(kw, self._grids)
            self.advective_upwind_component.update({
                component: upwind
            })
            self.advective_upwind_component_bc.update({
                component: upwind_bc
            })

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_keyword, self._grids)
        bc = pp.ad.BoundaryCondition(self.energy_keyword, self._grids)
        self.conductive_flux = (
            mpfa.flux * self.composition.T + mpfa.bound_flux * bc
        )
        # advective upwind in energy equation
        kw = f"{self.energy_upwind_keyword}_advective"
        self.advective_upwind_energy = pp.ad.UpwindAd(kw, self._grids)
        self.advective_upwind_energy_bc = pp.ad.BoundaryCondition(kw, self._grids)
        # conductive upwind energy
        kw = f"{self.energy_upwind_keyword}_conductive"
        self.conductive_upwind = pp.ad.UpwindAd(kw, self._grids)
        self.conductive_upwind_bc = pp.ad.BoundaryCondition(kw, self._grids)

    ## Boundary Conditions

    def _bc_advective_flux(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for advective flux (Darcy). Override for modifications.

        Phys. Dimensions of ADVECTIVE FLUX:

            - Dirichlet conditions: [kPa]
            - Neumann conditions: [m^3 / m^2 s]

        """
        _, idx_east, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_east] = self.outflow_boundary_pressure

        if self.inflow_boundary_pressure:
            bc = pp.BoundaryCondition(sd, np.logical_or(idx_east, idx_west), "dir")
            vals[idx_west] = self.inflow_boundary_pressure
        else:
            bc = pp.BoundaryCondition(sd, idx_east, "dir")

        return bc, vals

    def _bc_advective_weight_pressure(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the advective flux in pressure equation."""
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_boundary_advective_pressure

        return vals

    def _bc_advective_weight_component(
        self, sd: pp.Grid, component: pp.composite.Component
    ) -> np.ndarray:
        """BC values for the scalar part in the advective flux in component mass balance."""
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_boundary_advective_component[component]

        return vals

    def _bc_advective_weight_energy(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the advective flux in component mass balance."""
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_boundary_advective_energy

        return vals

    def _bc_conductive_flux(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """Conductive BC for Fourier flux in energy equation. Override for modifications.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:

            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [kJ / m^2 s] (density * specific enthalpy * Darcy flux)
              (same as convective enthalpy flux)

        """
        _, _, _, _, idx_south, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = self.boundary_temperature
        bc = pp.BoundaryCondition(sd, idx_south, "dir")

        return bc, vals

    def _bc_conductive_weight(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the conductive flux."""
        _, _, _, _, idx_south, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = self.boundary_conductive

        return vals

    def _bc_diff_disp_flux(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for diffusive-dispersive flux (Darcy). Override for modifications.

        Phys. Dimensions of FICK's LAW OF DIFFUSION:

            - Dirichlet conditions: [-] (molar, fractional: constant concentration at boundary)
            - Neumann conditions: [mol / m^2 s]
              (same as advective flux)

        """
        raise NotImplementedError("Diffusive-Dispersive Boundary Flux not available.")

    ## Source terms

    def _unitary_source(self, g: pp.Grid) -> np.ndarray:
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

    ### NEWTON --------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.converged = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Re-discretizes the Upwind operators and the fluxes."""
        # MPFA flux upwinding # TODO this is a bit hacky, what if data dict structure changes?
        # compute the advective flux (grad P) ONLY ONCE and store it in under the flow keyword
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            self.flow_keyword,
            self.flow_keyword,
            p_name=self.composition.p_name,
            from_iterate=True,
        )

        # we now proceed and see where the flux is needed
        for sd, data in self.mdg.subdomains(return_data=True):
            # get the flux
            flux  = data["parameters"][self.flow_keyword]["darcy_flux"]

            # copy the flux to the pressure equation dictionary
            if self._use_pressure_equation:
                data["parameters"][self.flow_upwind_keyword]["darcy_flux"] = np.copy(flux)

            # copy the flux to the dictionaries belonging to mass balance per component
            for component in self.composition.components:
                # skip the eliminated component mass balance
                if component == self.reference_component and self._use_pressure_equation:
                    continue
                kw = f"{self.flow_upwind_keyword}_{component.name}"
                data["parameters"][kw]["darcy_flux"] = np.copy(flux)
            
            # copy the flux to the dictionary for the advective part in the energy equation
            kw = f"{self.energy_upwind_keyword}_advective"
            data["parameters"][kw]["darcy_flux"] = np.copy(flux)

        # compute the conductive flux (grad T) for upwinding in conduction in energy equation
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            self.energy_keyword,
            f"{self.energy_upwind_keyword}_conductive",
            p_name=self.composition.T_name,
            from_iterate=True,
        )

        ## re-discretize the upwinding
        ## it is enough to call only discretize on the upwind itself, since it computes the
        ## boundary matrices along
        # for pressure equation
        if self._use_pressure_equation:
            self.advective_upwind.upwind.discretize(self.mdg)
        # for component mass balance
        for upwind in self.advective_upwind_component.values():
            upwind.upwind.discretize(self.mdg)
        # two upwinding classes for energy balance
        self.advective_upwind_energy.upwind.discretize(self.mdg)
        self.conductive_upwind.upwind.discretize(self.mdg)
        
        # for eq in self.ad_system._equations.values():
        #     eq.discretize(self.mdg)

        if not self._monolithic:
            print(f".. .. isenthalpic flash at iteration {self._nonlinear_iteration}.")
            success = self.composition.isenthalpic_flash(False, initial_guess="feed")
            if not success:
                raise RuntimeError("FAILURE: Isenthalpic flash.")
            else:
                flash_i = self.composition.flash_history[-1].get("iterations", "")
                print(f".. .. Success: Isenthalpic flash after iteration {flash_i}.")
            self.composition.evaluate_saturations(False)

    def after_newton_iteration(
        self, solution_vector: np.ndarray, iteration: int
    ) -> None:
        """Distributes solution of iteration additively to the iterate state of the variables.
        Increases the iteration counter.
        """
        self._nonlinear_iteration += 1

        if self._monolithic:
            # expand
            DX = self._prolong_system * solution_vector
        else:
            inv_A_ss, b_s, A_sp = self._for_expansion
            x_s = inv_A_ss * (b_s - A_sp * solution_vector)
            DX = self._prolong_prim * solution_vector + self._prolong_sec * x_s

        # post-processing eliminated component fraction additively to iterate
        self.dof_man.distribute_variable(
                values=DX,
                variables=self._system_vars,
                additive=True,
                to_iterate=True,
            )

        # post-process eliminated saturation variable
        if self._elim_ref_phase:
            self._post_process_saturation(False)
        # post-processing eliminated component fraction
        if self._use_pressure_equation:
            self._post_process_feed(False)
        # post process composition, including eliminated phase fraction
        if self._monolithic:
            self.composition.post_process_fractions(False)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Distributes the values from the iterate state to the the state (for next time step).
        Exports the results.

        """
        # write global solution
        self.dof_man.distribute_variable(solution, variables=self._system_vars)
        # post-process eliminated saturation variable
        if self._elim_ref_phase:
            self._post_process_saturation(True)
        # post-processing eliminated component fraction
        if self._use_pressure_equation:
            self._post_process_feed(True)
        # post process composition, including eliminated phase fraction
        if self._monolithic:
            self.composition.post_process_fractions(True)
        # export converged results
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Reset iterate state to previous state."""
        X = self.dof_man.assemble_variable()
        self.dof_man.distribute_variable(X, to_iterate=True)

    def after_simulation(self) -> None:
        """Writes PVD file."""
        self._exporter.write_pvd()

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """APerforms a Newton step for the whole system in a monolithic way, by constructing
        a Schur complement using the equilibrium equations and non-primary variables.

        :return: If converged, returns the solution. Until then, returns the update.
        :rtype: numpy.ndarray
        """

        if self._monolithic:
            A, b = self.ad_system.assemble_subsystem(
                equations=self._system_equations, variables=self._system_vars
                )
        else:
            # non-linear Schur complement elimination of secondary variables
            A_pp, b_p = self.ad_system.assemble_subsystem(
                self.flow_subsystem["primary_equations"],
                self.flow_subsystem["primary_vars"]
            )
            A_sp, _ = self.ad_system.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["primary_vars"]
            )
            A_ps, _ = self.ad_system.assemble_subsystem(
                self.flow_subsystem["primary_equations"],
                self.flow_subsystem["secondary_vars"]
            )
            A_ss, b_s = self.ad_system.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["secondary_vars"]
            )

            inv_A_ss = np.linalg.inv(A_ss.A)
            inv_A_ss = sps.csr_matrix(inv_A_ss)

            A = A_pp - A_ps * inv_A_ss * A_sp
            A = sps.csr_matrix(A)
            b = b_p - A_ps * inv_A_ss * b_s
            self._for_expansion = (inv_A_ss, b_s, A_sp)

        if np.linalg.norm(b) < tol:
            self.converged = True
            x = self.dof_man.assemble_variable(variables=self._system_vars, from_iterate=True)
            return x

        dx = spla.spsolve(A, b)

        return dx

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    def _post_process_saturation(self, copy_to_state: bool = False) -> None:
        """calculates the saturation of the eliminated phase.

        Parameters:
            copy_to_state (bool): Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

        """
        assert self._s_R is not None
        s_R = self._s_R.evaluate(self.ad_system.dof_manager).val
        s_R[s_R < 0.] = 0.
        s_R[s_R > 1.] = 1.
        self.ad_system.set_var_values(
            self.composition.reference_phase.saturation_name, s_R, copy_to_state
        )

    def _post_process_feed(self, copy_to_state: bool = False) -> None:
        """calculates the fraction of the eliminated component.

        Parameters:
            copy_to_state (bool): Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

        """
        assert self._z_R is not None
        z_R = self._z_R.evaluate(self.ad_system.dof_manager).val
        z_R[z_R < 0.] = 0.
        z_R[z_R > 1.] = 1.
        self.ad_system.set_var_values(
            self.reference_component.fraction_name, z_R, copy_to_state
        )

    def print_matrix(self, print_dense: bool = False):
        print("PRIMARY BLOCK:")
        print("Primary variables:")
        self.composition.print_ordered_vars(self.flow_subsystem["primary_vars"])
        for equ in self.flow_subsystem["primary_equations"]:
            A,b = self.ad_system.assemble_subsystem(equ, self.flow_subsystem["primary_vars"])
            print("---")
            print(equ)
            self.composition.print_system(A, b, print_dense)
        print("---")
        print("SECONDARY BLOCK:")
        print("Secondary variables:")
        self.composition.print_ordered_vars(self.flow_subsystem["secondary_vars"])
        for equ in self.flow_subsystem["secondary_equations"]:
            A,b = self.ad_system.assemble_subsystem(equ, self.flow_subsystem["secondary_vars"])
            print("---")
            print(equ)
            self.composition.print_system(A, b, print_dense)
        print("---")

    def print_matrix2(self, print_dense: bool = False):
        print("Variables:")
        self.composition.print_ordered_vars(self._system_vars)
        print("PRIMARY BLOCK:")
        for equ in self.flow_subsystem["primary_equations"]:
            A,b = self.ad_system.assemble_subsystem(equ, self._system_vars)
            print("---")
            print(equ)
            self.composition.print_system(A, b, print_dense)
            self.matrix_plot(A)
        print("---")
        print("SECONDARY BLOCK:")
        for equ in self.flow_subsystem["secondary_equations"]:
            A,b = self.ad_system.assemble_subsystem(equ, self._system_vars)
            print("---")
            print(equ)
            self.composition.print_system(A, b, print_dense)
        print("---")

    def matrix_plot(self, A):
        # plot.figure()
        # plot.matshow(A.todense())
        # plot.colorbar(orientation="vertical")
        # plot.set_cmap("terrain")
        # plot.show()
        A = A.todense()
        ax = plot.axes()
        ax.matshow(A, cmap=plot.cm.Blues)

        # for j in range(A.shape[1]):
        #     for i in range(A.shape[0]):
        #         c = A[i,j]
        #         if c == 0.:
        #             continue
        #         ax.text(i + 0.5, j + 0.5, str(c), va='center', ha='center')
        # ax = plot.axes()

        # ax.set_xlim(0, A.shape[1])
        # ax.set_ylim(0, A.shape[0])
        # ax.set_xticks(np.arange(A.shape[1]))
        # ax.set_yticks(np.arange(A.shape[0]))
        # ax.grid()

        plot.show()

    ### MODEL EQUATIONS -----------------------------------------------------------------------

    def _get_reference_feed_by_unity(self) -> pp.ad.Operator:
        """Returns a representation of the reference component fraction by unity"""
        equation = pp.ad.Scalar(1.)
        for component in self.composition.components:
            if component != self.reference_component:
                equation -= component.fraction
        return equation

    def _set_feed_fraction_unity_equation(self) -> None:
        """Sets the equation representing the feed fraction unity.

        Performs additionally an index reduction on this algebraic equation.

        """

        name = "feed_fraction_unity"
        self.flow_subsystem["primary_equations"].append(name)

        unity = pp.ad.Scalar(1.)

        # index reduction of algebraic unitarity constraint
        # demand exponential decay of rate of change
        time_derivative = list()
        for component in self.composition.components:
            unity -= component.fraction
            time_derivative.append(
                component.fraction
                - component.fraction.previous_timestep()
            )

        # parameter for a linear combination of the original algebraic constraint and its
        # time derivative
        decay = self.dt / (2 * np.pi)
        # final equation
        equation = sum(time_derivative) + self.dt * decay * unity

        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(name, equation, num_equ_per_dof=image_info)

    def _set_pressure_equation(self) -> None:
        """Sets the global pressure equation."""
        cp = self.composition
        upwind_adv = self.advective_upwind
        upwind_adv_bc = self.advective_upwind_bc

        ### ACCUMULATION
        accumulation = self.mass_matrix.mass * (
                cp.density(False, self._elim_ref_phase)
                - cp.density(True, self._elim_ref_phase)
            )
        
        ### ADVECTION
        advection_scalar = list()
        for phase in cp.phases:
            # eliminate reference phase saturation if requested
            if phase == cp.reference_phase and self._elim_ref_phase:
                assert self._s_R is not None
                k_re = self.rel_perm(self._s_R)
            else:
                k_re = self.rel_perm(phase.saturation)

            scalar_part = (
                phase.density(cp.p, cp.T)
                * k_re
                / phase.dynamic_viscosity(cp.p, cp.T)
            )
            advection_scalar.append(scalar_part)
        # sum over all phases
        advection_scalar = sum(advection_scalar)

        advection = (
            self.advective_flux * (upwind_adv.upwind * advection_scalar)
            - upwind_adv.bound_transport_dir * self.advective_flux * upwind_adv_bc
            - upwind_adv.bound_transport_neu * upwind_adv_bc
        )

        ### SOURCE
        source_arrays = list()
        for component in cp.components:
            keyword = f"source_{component.name}"
            source_arrays.append(
                pp.ad.ParameterArray(self.mass_keyword, keyword, subdomains=self._grids)
            )
        source = self.mass_matrix.mass * sum(source_arrays)

        ### PRESSURE EQUATION
        # minus in advection already included
        equation = accumulation + self.dt * (self.div * advection - source)
        equ_name = "pressure_equation"
        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per component.
        
        If the pressure equation is used, excludes the equation for the reference component.
        
        """
        cp = self.composition

        for component in cp.components:
            # exclude one mass balance equation if requested
            if self._use_pressure_equation and component == self.reference_component:
                continue

            upwind_adv = self.advective_upwind_component[component]
            upwind_adv_bc = self.advective_upwind_component_bc[component]

            ### ACCUMULATION
            accumulation = self.mass_matrix.mass * (
                component.fraction * cp.density(False, self._elim_ref_phase)
                - component.fraction.previous_timestep()
                * cp.density(True, self._elim_ref_phase)
            )

            ### ADVECTION
            advection_scalar = list()
            for phase in cp.phases:
                # eliminate reference phase saturation if requested
                if phase == cp.reference_phase and self._elim_ref_phase:
                    assert self._s_R is not None
                    k_re = self.rel_perm(self._s_R)
                else:
                    k_re = self.rel_perm(phase.saturation)

                scalar_part = (
                    phase.density(cp.p, cp.T)
                    * phase.ext_fraction_of_component(component)
                    * k_re
                    / phase.dynamic_viscosity(cp.p, cp.T)
                )

                advection_scalar.append(scalar_part)
            # sum over all phases
            advection_scalar = sum(advection_scalar)

            advection = (
                self.advective_flux * (upwind_adv.upwind * advection_scalar)
                - upwind_adv.bound_transport_dir * self.advective_flux * upwind_adv_bc
                - upwind_adv.bound_transport_neu * upwind_adv_bc
            )

            ### SOURCE
            keyword = f"source_{component.name}"
            source = pp.ad.ParameterArray(self.mass_keyword, keyword, subdomains=self._grids)
            source = self.mass_matrix.mass * source

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equation = accumulation + self.dt * (self.div * advection - source)
            equ_name = f"mass_balance_{component.name}"
            image_info = dict()
            for sd in self.mdg.subdomains():
                image_info.update({sd: {"cells": 1}})
            self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
            self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        # creating operators, parameters and shorter namespaces
        cp = self.composition
        upwind_adv = self.advective_upwind_energy
        upwind_adv_bc = self.advective_upwind_energy_bc
        upwind_cond = self.conductive_upwind
        upwind_cond_bc = self.conductive_upwind_bc

        ### ACCUMULATION
        accumulation = self.mass_matrix.mass * (
            cp.h * cp.density(False, self._elim_ref_phase)
            - cp.h.previous_timestep() * cp.density(True, self._elim_ref_phase)
        )

        ### ADVECTION
        advective_scalar = list()
        for phase in cp.phases:
            # eliminate reference phase saturation if requested
            if phase == cp.reference_phase and self._elim_ref_phase:
                assert self._s_R is not None
                k_re = self.rel_perm(self._s_R)
            else:
                k_re = self.rel_perm(phase.saturation)

            scalar_part = (
                phase.density(cp.p, cp.T)
                * phase.specific_enthalpy(cp.p, cp.T)
                * k_re
                / phase.dynamic_viscosity(cp.p, cp.T)
            )
            advective_scalar.append(scalar_part)
        # sum over all phases
        advective_scalar = sum(advective_scalar)

        advection = (
            self.advective_flux * (upwind_adv.upwind * advective_scalar)
            - upwind_adv.bound_transport_dir * self.advective_flux * upwind_adv_bc
            - upwind_adv.bound_transport_neu * upwind_adv_bc
        )

        ### CONDUCTION
        porosity = pp.ad.ParameterArray(
            self.mass_keyword, "mass_weight", subdomains=self._grids
        )
        conductive_scalar = list()
        # TODO scalar part depends on saturations, effective boundary conductivity needs recomputation
        for phase in cp.phases:
            # eliminate reference phase saturation
            if phase == cp.reference_phase and self._elim_ref_phase:
                assert self._s_R is not None
                s_e = self._s_R
            else:
                s_e = phase.saturation
            
            scalar_part = (
                s_e
                * phase.thermal_conductivity(cp.p, cp.T)
            )
            conductive_scalar.append(scalar_part)
        # sum over all phases
        conductive_scalar = porosity * sum(conductive_scalar)

        conduction = (
            self.conductive_flux * (upwind_cond.upwind * conductive_scalar)
            - upwind_cond.bound_transport_dir * self.conductive_flux * upwind_cond_bc
            - upwind_cond.bound_transport_neu * upwind_cond_bc
        )

        ### SOURCE
        # rock enthalpy source
        source = pp.ad.ParameterArray(
            self.energy_keyword, "source", subdomains=self._grids
        )
        # enthalpy source due to mass source
        for component in cp.components:
            kw = f"source_{component.name}"
            source += pp.ad.ParameterArray(
                self.energy_keyword, kw, subdomains=self._grids
            )
        source = self.mass_matrix.mass * source

        ### GLOBAL ENERGY BALANCE
        equation = accumulation + self.dt * (self.div * (advection + conduction) - source)
        equ_name = "energy_balance"
        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_phase_fraction_relation_equations(self) -> None:
        cp = self.composition

        # exclude reference phase is requested
        if self._elim_ref_phase:
            phases = [phase for phase in cp.phases if phase != cp.reference_phase]
        else:
            phases = [phase for phase in cp.phases]

        # get equations and equation names
        equations = [
            cp.get_phase_fraction_relation(phase, self._elim_ref_phase) for phase in phases
        ]
        equ_names = [f"phase_fraction_relation_{phase.name}" for phase in phases]

        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        # setting equations
        for name, equ in zip(equ_names, equations):
            self.ad_system.set_equation(name, equ, num_equ_per_dof=image_info)
        self.flow_subsystem["secondary_equations"] += equ_names

    ### CONSTITUTIVE LAWS ---------------------------------------------------------------------

    def rel_perm(self, saturation: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation
