"""Contains a general composit flow class without reactions

The grid, expected Phases and components can be modified in respective methods.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import pypardiso

import porepy as pp
from porepy.composite.component import Component


class CompositionalFlowModel:
    """Non-isothermal flow with gas and liquid phase.

    The composition class is instantiated during :meth:`prepare_simulation`
    using the sub-routine :meth:`set_composition`,
    which can be inherited and overwritten.

    Parameters:
        params: general model parameters including

            - 'use_ad' (bool): indicates whether :module:`porepy.ad` is used or not
            - 'file_name' (str): name of file for exporting simulation results
            - 'folder_name' (str): abs. path to directory for saving simulation results
            - 'eliminate_ref_phase' (bool): flag to eliminate reference phase fractions.
               Defaults to TRUE.
            - 'use_pressure_equation' (bool): flag to use the global pressure equation,
               instead of the feed fraction unity. Defaults to TRUE.
            - 'monolithic' (bool): flag to solve monolithically,
               instead of using a Schur complement elimination for secondary variables.
               Defaults to TRUE.

    """

    def __init__(self, params: dict) -> None:
        ### PUBLIC

        self.params: dict = params
        """Parameter dictionary passed at instantiation."""
        self.converged: bool = False
        """Indicator if current Newton-step converged."""
        self.dt: float = 0.1
        """Timestep size."""

        self.porosity = 1.0
        """Base porosity of model domain."""
        self.permeability = 0.03
        """Base permeability of model domain."""

        ## Initial Conditions
        self.initial_pressure: float = 1.0
        """Initial pressure in the domain in MPa."""
        self.initial_temperature: float = pp.composite.T_REF + 70  # 70 deg C
        """Initial temperature in the domain in K."""
        self.initial_component_fractions: list[float] = [0.99, 0.01]
        """Contains per component in composition the initial feed fraction."""

        ## Injection parameters
        self.injection_pressure = 2.0
        """Pressure at injection in MPA."""
        self.injection_temperature = pp.composite.T_REF + 30  # 70 deg C
        """Temperature of injected mass for source terms in Kelvin."""

        ## TO BE COMPUTED after obtaining the injection equilibrium
        self.injected_moles: list[float] = [0., 0.]
        """Contains per component in composition the amount of injected moles.

        Always included trace amounts of each components to avoid the
        washing-out effect."""
        self.injected_ext_enthalpy: float = 0.0
        """The amount of injected extensive enthalpy.
        To be set in :meth:`set_composition` after an equilibrium computation at the
        injection point."""

        ## Boundary conditions
        self.outflow_boundary_pressure: float = self.initial_pressure
        """Dirichlet boundary pressure for the outflow in MPA for the advective flux."""
        self.inflow_boundary_pressure: float = 3.0
        """Dirichlet boundary pressure for the inflow in MPa for the advective flux."""
        self.inflow_boundary_temperature: float = pp.composite.T_REF + 30  # 30 deg C
        """Temperature at the inflow boundary for the advective flux."""
        self.inflow_boundary_composition: list[float] = [0.99, 0.01]
        """Contains per component in composition the feed fraction at the inflow
        boundary.

        Always provide at least trace amounts of each component to avoid washing-out
        effects.

        """
        self.heated_boundary_temperature = 550
        """Dirichlet boundary temperature in Kelvin for the conductive flux,
        bottom boundary."""

        ## TO BE COMPUTED after obtaining the equilibrium at the boundary.
        self.inflow_boundary_advective_component: dict[str, float]
        """Contains per component name (key) the scalar part of the advective flux on
        the inflow boundary.
        To be set in :meth:`set_composition` after computing the equilibrium at the
        inflow.
        """
        self.inflow_boundary_advective_pressure: float
        """Value for scalar part of the advective flux on the
        inflow boundary for the global pressure equation.
        To be set in :meth:`set_composition` after computing the equilibrium at the
        inflow.
        """
        self.inflow_boundary_advective_energy: float
        """Value for scalar part of advective flux in energy equation.
        To be set in :meth:`set_composition`.
        """

        ## grid and AD system
        self.mdg: pp.MixedDimensionalGrid
        """Computational domain, to be set in :meth:`create_grid`."""
        self.create_grid()
        self.ad_system: pp.ad.EquationSystem = pp.ad.EquationSystem(self.mdg)
        """AD System for this model."""

        # contains information about the primary system
        self.flow_subsystem: dict[str, list] = dict()
        """Dictionary containing names of primary and secondary equations and variables.
        """

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
        self.injection_keyword: str = "injection"
        """Injection keyword for point sources"""

        ## References to discretization operators
        # they will be set during `prepare_simulation`
        self.mass_matrix: pp.ad.MassMatrixAd
        self.div: pp.ad.Divergence
        self.advective_flux: pp.ad.MpfaAd
        self.advective_upwind: pp.ad.UpwindAd
        self.advective_upwind_bc: pp.ad.BoundaryCondition
        self.advective_upwind_component: dict[Component, pp.ad.UpwindAd] = dict()
        self.advective_upwind_component_bc: dict[
            Component, pp.ad.BoundaryCondition
        ] = dict()
        self.conductive_flux: pp.ad.MpfaAd
        self.conductive_upwind: pp.ad.UpwindAd
        self.conductive_upwind_bc: pp.ad.BoundaryCondition
        self.advective_upwind_energy: pp.ad.UpwindAd
        self.advective_upwind_energy_bc: pp.ad.BoundaryCondition

        ### COMPOSITION SETUP
        self.composition: pp.composite.Mixture
        self.flash: pp.composite.Flash

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

        # data for Schur complement expansion
        self._for_expansion = (None, None, None)
        # solver strategy. If monolithic, the model will take care of flash calculations
        # if not, it will solve only the primary system
        self._monolithic = params.get("monolithic", True)
        # a reference to the operator representing the reference phase saturation
        # eliminated by unity
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
        bounding_box_points = np.array([[0, phys_dims[0]], [0, phys_dims[1]]])
        self.box = pp.geometry.domain.bounding_box_of_point_cloud(bounding_box_points)
        sg = pp.CartGrid(n_cells, phys_dims)
        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(sg)
        self.mdg.compute_geometry()

    def _domain_boundary_sides(
        self, sd: pp.Grid, tol: Optional[float] = 1e-10
    ) -> pp.domain.DomainSides:
        # Get domain boundary sides
        box = self.box
        east = np.abs(box["xmax"] - sd.face_centers[0]) <= tol
        west = np.abs(box["xmin"] - sd.face_centers[0]) <= tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(sd.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = np.abs(box["ymax"] - sd.face_centers[1]) <= tol
            south = np.abs(box["ymin"] - sd.face_centers[1]) <= tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(sd.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = np.abs(box["zmax"] - sd.face_centers[2]) <= tol
            bottom = np.abs(box["zmin"] - sd.face_centers[2]) <= tol
        all_bf = sd.get_boundary_faces()

        return all_bf, east, west, north, south, top, bottom

    def set_composition(self) -> None:
        """Defines the composition for which the simulation should be run and performs
        the initial isothermal equilibrium calculations.

        Set initial values for p, T and feed here.

        Use this method to inherit and override the composition,
        while keeping the (generic) rest of the model.

        Assumes the model domain ``mdg`` is already set.

        """
        ## creating composition
        self.composition = pp.composite.PengRobinsonMixture(self.ad_system)
        h2o = pp.composite.H2O(self.ad_system)
        co2 = pp.composite.CO2(self.ad_system)
        LIQ = pp.composite.PR_Phase(self.ad_system, gas_like=False, name='L')
        GAS = pp.composite.PR_Phase(self.ad_system, gas_like=True, name='G')
        self.composition.add([h2o, co2], [LIQ, GAS])

        ## setting thermodynamic state in terms of p-T-z
        nc = self.mdg.num_subdomain_cells()
        vec = np.ones(nc)

        # initial fractions
        for i, comp in enumerate(self.composition.components):
            frac_val = self.initial_component_fractions[i] * vec
            self.ad_system.set_variable_values(
                frac_val, variables=[comp.fraction.name], to_iterate=True, to_state=True
            )

        # setting initial pressure
        p_vals = self.initial_pressure * vec
        self.ad_system.set_variable_values(
            p_vals, variables=[self.composition.p.name], to_iterate=True, to_state=True
        )

        # setting initial temperature
        T_vals = self.initial_temperature * vec
        self.ad_system.set_variable_values(
            T_vals, variables=[self.composition.T.name], to_iterate=True, to_state=True
        )

        # set zero enthalpy values at the beginning
        # to get the AD framework properly started
        h_vals = np.zeros(nc)
        self.ad_system.set_variable_values(
            h_vals, variables=[self.composition.h.name], to_iterate=True, to_state=True
        )

        ## initialize and construct flasher
        print("Computing initial, domain-wide equilibrium ...")
        self.composition.initialize()
        self.flash = pp.composite.Flash(self.composition)
        self.flash.use_armijo = True
        self.flash.armijo_parameters["rho"] = 0.99
        self.flash.armijo_parameters["j_max"] = 50
        self.flash.armijo_parameters["return_max"] = True
        self.flash.newton_update_chop = 1.0
        self.flash.flash_tolerance = 5e-6
        self.flash.max_iter_flash = 140
        self.flash.flash("pT", "npipm", "rachford_rice", True, False)
        self.flash.post_process_fractions(True)
        self.flash.evaluate_specific_enthalpy(True)
        self.composition.compute_roots()

        ## configuration at boundary and injection points
        self._set_boundary_and_source_properties()

    def _set_boundary_and_source_properties(self):
        """Auxiliary function to compute properties at the boundary and in sources
        in a separate Composition instance."""

        ## INFLOW BOUNDARY
        nc = 1
        C = pp.composite.PengRobinsonMixture(nc=nc)
        adsys = C.ad_system
        h2o = pp.composite.H2O(adsys)
        co2 = pp.composite.CO2(adsys)
        LIQ = pp.composite.PR_Phase(adsys, gas_like=False, name='L')
        GAS = pp.composite.PR_Phase(adsys, gas_like=True, name='G')
        C.add([h2o, co2], [LIQ, GAS])

        # setting thermodynamic state at boundary
        vec = np.ones(nc)

        for i, comp in enumerate(C.components):
            frac_vals = self.inflow_boundary_composition[i] * vec
            adsys.set_variable_values(
                frac_vals,
                variables=[comp.fraction.name],
                to_iterate=True,
                to_state=True,
            )

        adsys.set_variable_values(
            self.inflow_boundary_temperature * vec,
            variables=[C.T.name],
            to_iterate=True,
            to_state=True,
        )
        adsys.set_variable_values(
            self.inflow_boundary_pressure * vec,
            variables=[C.p.name],
            to_iterate=True,
            to_state=True,
        )
        adsys.set_variable_values(
            0 * vec, variables=[C.h.name], to_iterate=True, to_state=True
        )

        print("Computing inflow boundary equilibrium ...")
        C.initialize()
        F = pp.composite.Flash(C)
        F.use_armijo = True
        F.armijo_parameters["rho"] = 0.99
        F.armijo_parameters["j_max"] = 50
        F.armijo_parameters["return_max"] = True
        F.newton_update_chop = 1.0
        F.flash_tolerance = 1e-8
        F.max_iter_flash = 140
        F.flash("pT", "npipm", "rachford_rice", True, False)
        F.post_process_fractions(True)
        F.evaluate_specific_enthalpy(True)
        C.precompute(apply_smoother=False)

        # computing mass entering the system through boundary
        self.inflow_boundary_advective_component = dict()
        for component in C.components:

            advective_scalar = sum(
                [
                    phase.density(C.p, C.T)
                    * phase.saturation
                    * phase.normalized_fraction_of_component(component)
                    for phase in C.phases
                ]
            ).evaluate(adsys).val[0]

            self.inflow_boundary_advective_component.update(
                {component.name: advective_scalar}
            )

        # computing energy introduced through the boundary by above mass
        self.inflow_boundary_advective_energy = (
            (C.density() * C.h).evaluate(adsys).val[0]
        )

        # computing mass entering the system in the pressure equation
        self.inflow_boundary_advective_pressure = C.density().evaluate(adsys).val[0]

    def prepare_simulation(self) -> None:
        """Preparing essential simulation configurations.

        Method needs to be called prior to applying any solver.

        """
        self.set_composition()

        # Storage of primary and secondary vars and equs for splitting solver
        self.flow_subsystem.update(
            {
                "primary_equations": list(),
                "secondary_equations": list(),
                "primary_vars": list(),
                "secondary_vars": list(),
            }
        )
        # the primary vars of the flash, are secondary for the flow, and vice versa.
        # deep copies to not mess with the flash subsystems
        primary_vars = [var for var in self.composition.AD.ph_subsystem["secondary-variables"]]
        secondary_vars = [var for var in self.composition.AD.ph_subsystem["primary-variables"]]

        # eliminate the reference component fraction if pressure equation is used
        if self._use_pressure_equation:
            primary_vars.remove(self.composition.reference_component.fraction_name)

        # saturations are also secondary in the flow
        # we eliminate the saturation of the reference phase by unity, if requested
        for phase in self.composition.phases:
            primary_vars.remove(phase.saturation_name)
            # the ref phase molar fraction is also secondary in the composition class
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
            export_vars.add(self.composition.reference_component.fraction_name)
        if self._elim_ref_phase:
            export_vars.add(self.composition.reference_phase.fraction_name)
            export_vars.add(self.composition.reference_phase.saturation_name)
        self._export_vars = list(export_vars)

        # prepare prolongations for the solver
        self._prolong_prim = self.ad_system.projection_to(primary_vars).transpose()
        self._prolong_sec = self.ad_system.projection_to(secondary_vars).transpose()
        self._prolong_system = self.ad_system.projection_to(
            self._system_vars
        ).transpose()

        # if eliminated, express the reference fractions by unity
        if self._elim_ref_phase:
            self._s_R = self.composition.get_reference_phase_saturation_by_unity()
        if self._use_pressure_equation:
            self._z_R = self._get_reference_feed_by_unity()

        # deep copies, just in case
        sec_eq = [eq for eq in self.composition.AD.ph_subsystem["equations"]]
        sec_eq += [eq for eq in self.flash.complementary_equations]
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

        self._system_equations = [
            equ for equ in self.flow_subsystem["primary_equations"]
        ]
        self._system_equations += [
            equ for equ in self.flow_subsystem["secondary_equations"]
        ]

    def _export(self) -> None:
        self._exporter.write_vtu(self._export_vars, time_dependent=True)

    ### SET-UP -------------------------------------------------------------------------

    def _set_up(self) -> None:
        """Set model components including

            - source terms,
            - boundary values,
            - permeability tensor

        A modularization of the solid skeleton properties is still missing.
        """

        for sd, data in self.mdg.subdomains(return_data=True):

            injection_point = self._unitary_source(sd)
            unit_tensor = pp.SecondOrderTensor(np.ones(sd.num_cells))
            zero_vector_source = np.zeros((self.mdg.dim_max(), sd.num_cells))

            ### MASS PARAMETERS AND MASS SOURCES
            param_dict = dict()
            # weight for accumulation term
            param_dict.update({"mass_weight": self.porosity * np.ones(sd.num_cells)})
            # source terms per component
            for i, component in enumerate(self.composition.components):
                kw = f"{self.injection_keyword}_{component.name}"
                param_dict.update({kw: injection_point * self.injected_moles[i]})

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
                if (
                    component == self.composition.reference_component
                    and self._use_pressure_equation
                ):
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
            # enthalpy sources due to injection
            param_dict.update(
                {self.injection_keyword: injection_point * self.injected_ext_enthalpy}
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
        subdomains = self.mdg.subdomains()
        # mass matrix
        self.mass_matrix = pp.ad.MassMatrixAd(self.mass_keyword, subdomains)
        # divergence
        self.div = pp.ad.Divergence(subdomains=subdomains, name="Divergence")

        # advective flux
        mpfa = pp.ad.MpfaAd(self.flow_keyword, subdomains)
        bc = pp.ad.BoundaryCondition(self.flow_keyword, subdomains)
        self.advective_flux = mpfa.flux * self.composition.p + mpfa.bound_flux * bc
        # advective upwind for global pressure
        if self._use_pressure_equation:
            self.advective_upwind = pp.ad.UpwindAd(self.flow_upwind_keyword, subdomains)
            self.advective_upwind_bc = pp.ad.BoundaryCondition(
                self.flow_upwind_keyword, subdomains
            )
        # advective upwind per component mass balance
        for component in self.composition.components:
            # skip eliminated reference component in case of global pressure equation
            if component == self.composition.reference_component and self._use_pressure_equation:
                continue
            kw = f"{self.flow_upwind_keyword}_{component.name}"
            upwind = pp.ad.UpwindAd(kw, subdomains)
            upwind_bc = pp.ad.BoundaryCondition(kw, subdomains)
            self.advective_upwind_component.update({component: upwind})
            self.advective_upwind_component_bc.update({component: upwind_bc})

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_keyword, subdomains)
        bc = pp.ad.BoundaryCondition(self.energy_keyword, subdomains)
        self.conductive_flux = mpfa.flux * self.composition.T + mpfa.bound_flux * bc
        # advective upwind in energy equation
        kw = f"{self.energy_upwind_keyword}_advective"
        self.advective_upwind_energy = pp.ad.UpwindAd(kw, subdomains)
        self.advective_upwind_energy_bc = pp.ad.BoundaryCondition(kw, subdomains)
        # conductive upwind energy
        kw = f"{self.energy_upwind_keyword}_conductive"
        self.conductive_upwind = pp.ad.UpwindAd(kw, subdomains)
        self.conductive_upwind_bc = pp.ad.BoundaryCondition(kw, subdomains)

    ## Boundary Conditions

    def _bc_advective_flux(
        self, sd: pp.Grid
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for advective flux (Darcy). Override for modifications.

        Phys. Dimensions of ADVECTIVE FLUX:

            - Dirichlet conditions: [MPa]
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
        self, sd: pp.Grid, component: Component
    ) -> np.ndarray:
        """BC values for the scalar part in the advective flux in component mass balance."""
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_boundary_advective_component[component.name]

        return vals

    def _bc_advective_weight_energy(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the advective flux in component mass balance."""
        _, _, idx_west, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_boundary_advective_energy

        return vals

    def _bc_conductive_flux(
        self, sd: pp.Grid
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """Conductive BC for Fourier flux in energy equation. Override for modifications.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:

            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [kJ / m^2 s] (density * specific enthalpy * Darcy flux)
              (same as convective enthalpy flux)

        """
        _, _, idx_west, idx_north, idx_south, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = self.heated_boundary_temperature
        vals[idx_north] = self.initial_temperature
        vals[idx_west] = self.inflow_boundary_temperature

        idx = np.logical_or(idx_west, np.logical_or(idx_south, idx_north))
        bc = pp.BoundaryCondition(sd, idx, "dir")

        return bc, vals

    def _bc_conductive_weight(self, sd: pp.Grid) -> np.ndarray:
        """BC values for the scalar part in the conductive flux."""
        _, _, idx_west, idx_north, idx_south, *_ = self._domain_boundary_sides(sd)

        vals = np.zeros(sd.num_faces)
        vals[idx_south] = 10.0
        vals[idx_north] = 10.0
        vals[idx_west] = 10.0

        return vals

    def _bc_diff_disp_flux(
        self, sd: pp.Grid
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for diffusive-dispersive flux (Darcy). Override for modifications.

        Phys. Dimensions of FICK's LAW OF DIFFUSION:

            - Dirichlet conditions: [-] (molar, fractional)
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
        # MPFA flux upwinding # TODO this is a bit hacky,
        # what if data dict structure changes?
        # compute the advective flux (grad P) ONLY ONCE
        # and store it in under the flow keyword
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            self.flow_keyword,
            self.flow_keyword,
            p_name=self.composition.p.name,
            from_iterate=True,
        )

        # we now proceed and see where the flux is needed
        for sd, data in self.mdg.subdomains(return_data=True):
            # get the flux
            flux = data["parameters"][self.flow_keyword]["darcy_flux"]

            # copy the flux to the pressure equation dictionary
            if self._use_pressure_equation:
                data["parameters"][self.flow_upwind_keyword]["darcy_flux"] = np.copy(
                    flux
                )

            # copy the flux to the dictionaries belonging to mass balance per component
            for component in self.composition.components:
                # skip the eliminated component mass balance
                if (
                    component == self.composition.reference_component
                    and self._use_pressure_equation
                ):
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
            p_name=self.composition.T.name,
            from_iterate=True,
        )

        ## re-discretize the upwinding
        ## it is enough to call only discretize on the upwind itself,
        ## since it computes the boundary matrices along
        # for pressure equation
        if self._use_pressure_equation:
            self.advective_upwind.upwind.discretize(self.mdg)
        # for component mass balance
        for upwind in self.advective_upwind_component.values():
            upwind.upwind.discretize(self.mdg)
        # two upwinding classes for energy balance
        self.advective_upwind_energy.upwind.discretize(self.mdg)
        self.conductive_upwind.upwind.discretize(self.mdg)

        ## TODO recompute the effective boundary conduction.

        if not self._monolithic:
            print(f".. .. isenthalpic flash at iteration {self._nonlinear_iteration}.")
            success = self.flash.flash("ph", "npipm", "iterate", False, False)
            if not success:
                raise RuntimeError("FAILURE: Isenthalpic flash.")

            self.flash.post_process_fractions(False)

    def after_newton_iteration(
        self, solution_vector: np.ndarray, iteration: int
    ) -> None:
        """Distributes solution of iteration additively to the iterate state of the
        variables. Increases the iteration counter.
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
        self.ad_system.set_variable_values(
            values=DX,
            variables=self._system_vars,
            to_iterate=True,
            additive=True,
        )

        # post-process eliminated saturation variable
        if self._elim_ref_phase:
            self._post_process_saturation(False)
        # post-processing eliminated component fraction
        if self._use_pressure_equation:
            self._post_process_feed(False)
        # post process composition, including eliminated phase fraction
        if self._monolithic:
            self.flash.post_process_fractions(False)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Distributes the values from the iterate state to the the state (for next time step).
        Exports the results.

        """
        # write global solution
        self.ad_system.set_variable_values(
            solution, variables=self._system_vars, to_state=True
        )
        # post-process eliminated saturation variable
        if self._elim_ref_phase:
            self._post_process_saturation(True)
        # post-processing eliminated component fraction
        if self._use_pressure_equation:
            self._post_process_feed(True)
        # post process composition, including eliminated phase fraction
        if self._monolithic:
            self.flash.post_process_fractions(True)
        # export converged results
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Reset iterate state to previous state."""
        X = self.ad_system.get_variable_values()
        self.ad_system.set_variable_values(X, to_iterate=True)

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
                self.flow_subsystem["primary_vars"],
            )
            A_sp, _ = self.ad_system.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["primary_vars"],
            )
            A_ps, _ = self.ad_system.assemble_subsystem(
                self.flow_subsystem["primary_equations"],
                self.flow_subsystem["secondary_vars"],
            )
            A_ss, b_s = self.ad_system.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["secondary_vars"],
            )

            inv_A_ss = np.linalg.inv(A_ss.A)
            inv_A_ss = sps.csr_matrix(inv_A_ss)

            A = A_pp - A_ps * inv_A_ss * A_sp
            A = sps.csr_matrix(A)
            b = b_p - A_ps * inv_A_ss * b_s
            self._for_expansion = (inv_A_ss, b_s, A_sp)

        res = np.linalg.norm(b)
        print(f".. .. flow residual norm: {res}")
        if res < tol:
            self.converged = True
            x = self.ad_system.get_variable_values(
                variables=self._system_vars, from_iterate=True
            )
            return x

        # dx = spla.spsolve(A, b)
        dx = pypardiso.spsolve(A, b)

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
        s_R = self._s_R.evaluate(self.ad_system).val
        s_R[s_R < 0.0] = 0.0
        s_R[s_R > 1.0] = 1.0
        self.ad_system.set_variable_values(
            s_R,
            variables=[self.composition.reference_phase.saturation_name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    def _post_process_feed(self, copy_to_state: bool = False) -> None:
        """calculates the fraction of the eliminated component.

        Parameters:
            copy_to_state (bool): Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

        """
        assert self._z_R is not None
        z_R = self._z_R.evaluate(self.ad_system).val
        z_R[z_R < 0.0] = 0.0
        z_R[z_R > 1.0] = 1.0
        self.ad_system.set_variable_values(
            z_R,
            variables=[self.composition.reference_component.fraction_name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    ### MODEL EQUATIONS ----------------------------------------------------------------

    def _get_reference_feed_by_unity(self) -> pp.ad.Operator:
        """Returns a representation of the reference component fraction by unity"""
        equation = pp.ad.Scalar(1.0)
        for component in self.composition.components:
            if component != self.composition.reference_component:
                equation -= component.fraction
        return equation

    def _set_feed_fraction_unity_equation(self) -> None:
        """Sets the equation representing the feed fraction unity.

        Performs additionally an index reduction on this algebraic equation.

        """

        unity = pp.ad.Scalar(1.0)

        # index reduction of algebraic unitarity constraint
        # demand exponential decay of rate of change
        time_derivative = list()
        for component in self.composition.components:
            unity -= component.fraction
            time_derivative.append(
                component.fraction - component.fraction.previous_timestep()
            )

        # parameter for a linear combination of the original algebraic constraint and
        # its time derivative
        decay = self.dt / (2 * np.pi)
        # final equation
        equation = sum(time_derivative) + self.dt * decay * unity

        name = "feed_fraction_unity"
        self.flow_subsystem["primary_equations"].append(name)
        equation.set_name(name)
        self.ad_system.set_equation(equation, self.mdg.subdomains(), {"cells": 1})

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
                phase.density(cp.p, cp.T) * k_re / phase.dynamic_viscosity(cp.p, cp.T)
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
            source_arrays.append(
                pp.ad.ParameterArray(
                    self.mass_keyword,
                    f"{self.injection_keyword}_{component.name}",
                    subdomains=self.mdg.subdomains(),
                )
            )
        source = self.mass_matrix.mass * sum(source_arrays)

        ### PRESSURE EQUATION
        # minus in advection already included
        equation = accumulation + self.dt * (self.div * advection - source)

        name = "pressure_equation"
        self.flow_subsystem["primary_equations"].append(name)
        equation.set_name(name)
        self.ad_system.set_equation(equation, self.mdg.subdomains(), {"cells": 1})

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per component.

        If the pressure equation is used,
        excludes the equation for the reference component.

        """
        cp = self.composition

        for component in cp.components:
            # exclude one mass balance equation if requested
            if self._use_pressure_equation and component == self.composition.reference_component:
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
                    * phase.normalized_fraction_of_component(component)
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
            source = pp.ad.ParameterArray(
                self.mass_keyword,
                f"{self.injection_keyword}_{component.name}",
                subdomains=self.mdg.subdomains(),
            )
            source = self.mass_matrix.mass * source

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equation = accumulation + self.dt * (self.div * advection - source)

            name = f"mass_balance_{component.name}"
            self.flow_subsystem["primary_equations"].append(name)
            equation.set_name(name)
            self.ad_system.set_equation(equation, self.mdg.subdomains(), {"cells": 1})

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
                * phase.specific_enthalpy(
                    cp.p,
                    cp.T,
                    *[phase.normalized_fraction_of_component(comp) for comp in phase]
                )
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
            self.mass_keyword, "mass_weight", subdomains=self.mdg.subdomains()
        )
        conductive_scalar = list()
        for phase in cp.phases:
            # eliminate reference phase saturation
            if phase == cp.reference_phase and self._elim_ref_phase:
                assert self._s_R is not None
                s_e = self._s_R
            else:
                s_e = phase.saturation

            scalar_part = s_e * phase.thermal_conductivity(cp.p, cp.T)
            conductive_scalar.append(scalar_part)
        # sum over all phases
        conductive_scalar = porosity * sum(conductive_scalar)

        # TODO scalar part depends on saturations,
        # effective boundary conductivity needs recomputation
        # currently set to 1 using redundant self.boundary_conductive
        conduction = (
            self.conductive_flux * (upwind_cond.upwind * conductive_scalar)
            - upwind_cond.bound_transport_dir * self.conductive_flux * upwind_cond_bc
            - upwind_cond.bound_transport_neu * upwind_cond_bc
        )

        ### SOURCE
        # enthalpy source due to injection
        source = pp.ad.ParameterArray(
            self.energy_keyword,
            self.injection_keyword,
            subdomains=self.mdg.subdomains(),
        )
        source = self.mass_matrix.mass * source

        ### GLOBAL ENERGY BALANCE
        equation = accumulation + self.dt * (
            self.div * (advection + conduction) - source
        )

        name = "energy_balance"
        self.flow_subsystem["primary_equations"].append(name)
        equation.set_name(name)
        self.ad_system.set_equation(equation, self.mdg.subdomains(), {"cells": 1})

    def _set_phase_fraction_relation_equations(self) -> None:
        cp = self.composition

        # exclude reference phase is requested
        if self._elim_ref_phase:
            phases = [phase for phase in cp.phases if phase != cp.reference_phase]
        else:
            phases = [phase for phase in cp.phases]

        # get equations and equation names
        equations = [
            cp.get_phase_fraction_relation(phase, self._elim_ref_phase)
            for phase in phases
        ]
        equ_names = [f"phase_fraction_relation_{phase.name}" for phase in phases]

        # setting equations
        for name, equ in zip(equ_names, equations):
            equ.set_name(name)
            self.ad_system.set_equation(equ, self.mdg.subdomains(), {"cells": 1})
        self.flow_subsystem["secondary_equations"] += equ_names

    ### CONSTITUTIVE LAWS --------------------------------------------------------------

    def rel_perm(self, saturation: pp.ad.MixedDimensionalVariable) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation
